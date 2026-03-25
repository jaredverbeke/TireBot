#!/usr/bin/env python3
import csv
import html
import math
import re
import subprocess
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import altair as alt

from recommend_tires import (
    Segment,
    interpolate_missing_surface_values,
    load_segments,
    load_tires_with_optional_brr,
    score_tires,
)
from risk_model import build_features_for_route_and_tire, load_labeled_examples, predict_knn


ROOT = Path(__file__).parent
ROUTES_DIR = ROOT / "Routes"
TIRES_CSV = ROOT / "Gravel and MTB Tire Testing by John Karrasch  - Overall CRR.csv"
BRR_CRR_CSV = ROOT / "data" / "brr_crr.csv"
TIRE_MASS_CSV = ROOT / "data" / "tire_mass_overrides.csv"
BUILD_NUMBER_PATH = ROOT / "data" / "build_number.txt"
RISK_LABELS_PATH = ROOT / "data" / "risk_labels.csv"
# Whitepaper: GitHub Pages site (serves docs/index.html + WHITEPAPER.md).
WHITEPAPER_ONLINE_PAGES_URL = "https://jaredverbeke.github.io/TireBot/"

FEEDBACK_EMAIL = "jaredverbeke@gmail.com"
STRIPE_DONATE_URL = "https://buy.stripe.com/eVq5kF933eDd7NN6bQ7Re00"

# Segment CSVs list distances in miles; summarize_route also reports miles.
KM_TO_MI = 0.621371
M_TO_FT = 3.28084
EARTH_RADIUS_KM = 6371.0

# GPX grade gates for aero: exclude steep descents (coasting) and steep climbs (low speed).
AERO_EXCLUDE_DOWNHILL_GRADE = -0.012  # below ~-1.2% net → treat as downhill for aero
AERO_EXCLUDE_STEEP_CLIMB_GRADE = 0.07  # above ~7% → "large climb", aero neglected

MTB_MIN_WIDTH_IN = 2.2
MTB_MIN_WIDTH_MM = MTB_MIN_WIDTH_IN * 25.4
ABOVE_CATEGORY_MTB_ONLY_THRESHOLD_MI = 2.0

# Rough-surface impedance penalty (tunable). Units: adds CRR-equivalent to route_score.
IMPEDANCE_GAMMA = 2.2
IMPEDANCE_K_BY_SURFACE = {"road": 0.0, "cat1": 0.00018, "cat2": 0.00055, "cat3": 0.00110, "above": 0.00160}
# Target stiffness proxy by surface (psi*mm/kg). Rougher surfaces want lower stiffness (more compliance).
IMPEDANCE_TARGET_STIFFNESS = {"road": 26.0, "cat1": 15.0, "cat2": 13.5, "cat3": 12.0, "above": 11.0}
# Support floor: narrow tires under heavier riders require minimum pressure.
SUPPORT_PSI_FACTOR = 16.0  # psi ≈ factor * (weight_kg / width_mm)

# Flat/issue risk (proxy) driven by rough-surface exposure and tire width.
RISK_NO_ROUGH_MILES = 0.25
RISK_MTB_NO_RISK_MIN_IN = 2.2
RISK_MTB_NO_RISK_MIN_MM = RISK_MTB_NO_RISK_MIN_IN * 25.4
RISK_MTB_CAP_MED_IN = 2.1
RISK_MTB_CAP_MED_MM = RISK_MTB_CAP_MED_IN * 25.4
RISK_SHORT_CAT3_MILES = 3.0
RISK_SHORT_CAT3_LOW_MM = 40.0
RISK_SHORT_CAT3_MED_MAX_MM = 34.5


def km_to_mi(km: float) -> float:
    return km * KM_TO_MI


def above_distance_mi(segments: List[Segment]) -> float:
    return km_to_mi(sum(s.distance_km for s in segments if s.surface == "above"))


def cat3_distance_mi(segments: List[Segment]) -> float:
    return km_to_mi(sum(s.distance_km for s in segments if s.surface in {"cat3", "above"}))


def cat2_distance_mi(segments: List[Segment]) -> float:
    return km_to_mi(sum(s.distance_km for s in segments if s.surface == "cat2"))


def tire_issue_risk_for_tire(
    width_mm: float,
    cat2_mi: float,
    cat3_mi: float,
    *,
    tire_name: str = "",
    tire_class: str = "",
    above_mi: float = 0.0,
) -> str:
    """Low/Medium/High risk proxy based on rough miles and tire width.

    Rule: MTB tires ≥ 2.4\" are treated as no Cat 3 risk.
    """
    w = float(width_mm or 0.0)
    c2 = max(0.0, float(cat2_mi))
    c3 = max(0.0, float(cat3_mi))
    name = (tire_name or "").lower()
    is_road = (str(tire_class or "").strip().lower() == "road") or ("corsa" in name)

    # Above Category override: any 'above' miles implies extra durability risk.
    # - Road tires: High
    # - MTB >= 2.2": Low (still)
    # - < 48mm: Medium
    if float(above_mi or 0.0) > 0:
        if is_road:
            return "High"
        if w >= RISK_MTB_NO_RISK_MIN_MM:
            return "Low"
        if w < 48.0:
            return "Medium"

    # Cat 3 drives issue risk more than Cat 2; Cat 2 contributes but is de-emphasized.
    rough_mi = (0.2 * c2) + (1.0 * c3)

    if rough_mi < RISK_NO_ROUGH_MILES:
        return "Low"
    if w >= RISK_MTB_NO_RISK_MIN_MM:
        return "Low"

    # Road tires on rough exposure should never show as Low.

    # If Cat 3 exposure is short, cap risk: 40mm+ is Low; very narrow stays Medium at worst.
    if c3 < RISK_SHORT_CAT3_MILES:
        if w >= RISK_SHORT_CAT3_LOW_MM and not is_road:
            return "Low"
        if w <= RISK_SHORT_CAT3_MED_MAX_MM:
            return "Medium"
        return "Medium" if is_road else "Low"

    # Narrower tires amplify risk on rough (cat3) exposure.
    # reference width: 50 mm (gravel+). below that increases risk.
    ref = 50.0
    vuln = max(0.0, (ref - w) / ref)
    score = rough_mi * (1.0 + 3.0 * vuln)
    if is_road:
        score *= 1.35

    if score < 2.0:
        return "Low" if not is_road else "Medium"
    if score < 6.0:
        return "Medium"
    # Cap: 2.1\" MTB-ish tires shouldn't be "High" even on very rough courses.
    if w >= RISK_MTB_CAP_MED_MM:
        return "Medium"
    return "High"


def tire_issue_risk_label(
    *,
    examples: List,
    segments: List[Segment],
    tire_width_mm: float,
    tire_class: str,
    fallback_label: str,
) -> str:
    if not examples:
        return fallback_label
    try:
        feats = build_features_for_route_and_tire(segments=segments, width_mm=tire_width_mm, tire_class=tire_class)
        pred = predict_knn(query=feats, examples=examples, k=5)
        return pred or fallback_label
    except Exception:
        return fallback_label


def risk_badge_html(label: str) -> str:
    lab = (label or "").strip()
    colors = {
        "Low": ("#15803d", "rgba(34, 197, 94, 0.18)"),
        "Medium": ("#a16207", "rgba(234, 179, 8, 0.22)"),
        "High": ("#b91c1c", "rgba(239, 68, 68, 0.18)"),
    }
    fg, bg = colors.get(lab, ("#0b1220", "rgba(37, 99, 235, 0.10)"))
    return (
        f'<span style="display:inline-block;padding:0.18rem 0.55rem;border-radius:999px;'
        f'border:1px solid {bg};background:{bg};color:{fg};font-weight:700;font-size:0.82rem;">{html.escape(lab)}</span>'
    )


def style_risk(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    def risk_style(v: object) -> str:
        s = str(v)
        if s == "Low":
            return "background-color: rgba(34, 197, 94, 0.18); color: #14532d; font-weight: 700;"
        if s == "Medium":
            return "background-color: rgba(234, 179, 8, 0.22); color: #713f12; font-weight: 700;"
        if s == "High":
            return "background-color: rgba(239, 68, 68, 0.18); color: #7f1d1d; font-weight: 700;"
        return ""

    if "Risk" in df.columns:
        return df.style.applymap(risk_style, subset=["Risk"])
    return df.style


def current_git_sha_short() -> Optional[str]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True).strip()
        return sha or None
    except Exception:
        return None


def current_build_number() -> Optional[int]:
    try:
        raw = BUILD_NUMBER_PATH.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        return int(raw)
    except Exception:
        return None


def load_risk_examples() -> List:
    try:
        return load_labeled_examples(RISK_LABELS_PATH)
    except Exception:
        return []


def inject_styles() -> None:
    st.markdown(
        """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
<style>
    :root {
        /* MidSouth-inspired: black base + full accent palette (not a direct copy) */
        --tb-bg: #050607;
        --tb-surface: #0b0d10;
        --tb-surface-2: #0f1217;
        --tb-ink: rgba(255, 255, 255, 0.92);
        --tb-text-dim: rgba(255, 255, 255, 0.70);
        --tb-border: rgba(255, 255, 255, 0.14);

        /* MidSouth palette */
        --tb-yellow: #FFD400;
        --tb-blue: #0B84F3;
        --tb-green: #1F6F4B;
        --tb-orange: #F26A2E;

        --tb-focus: rgba(255, 212, 0, 0.28);
    }

    .stApp {
        font-family: "Outfit", ui-sans-serif, system-ui, sans-serif;
        letter-spacing: 0.01em;
        background: var(--tb-bg);
        color: var(--tb-ink);
    }
    /* Default readable text on dark background (avoid overriding light inputs). */
    .stMarkdown, .stMarkdown p, .stMarkdown li { color: var(--tb-ink); }
    h1, h2, h3 {
        font-family: "Space Grotesk", "Outfit", ui-sans-serif, system-ui, sans-serif !important;
        letter-spacing: -0.02em;
        color: var(--tb-ink);
    }
    .block-container { padding-top: 1.25rem; padding-bottom: 3rem; max-width: 1120px; }

    [data-testid="stAppViewContainer"] { background: var(--tb-bg); }
    [data-testid="stHeader"] { background: transparent; }
    /* Many Streamlit widgets use theme tokens; ensure they are readable. */
    [data-testid="stWidgetLabel"], .stTextInput label, .stNumberInput label, .stSelectbox label, .stRadio label {
        color: var(--tb-ink) !important;
    }

    /* Metrics (st.metric) must be readable on dark bg */
    [data-testid="stMetric"] * {
        color: var(--tb-ink) !important;
    }
    [data-testid="stMetricLabel"] {
        color: var(--tb-text-dim) !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--tb-ink) !important;
    }

    /* Selectbox / dropdown (BaseWeb) on dark background */
    [data-baseweb="select"] > div {
        background: #ffffff !important;
        border-color: rgba(0, 0, 0, 0.14) !important;
        color: #050607 !important;
    }
    [data-baseweb="select"] input {
        color: #050607 !important;
        -webkit-text-fill-color: #050607 !important;
        caret-color: var(--tb-orange) !important;
    }
    [data-baseweb="select"] [data-testid="stMarkdownContainer"] {
        color: #050607 !important;
    }
    /* Dropdown menu */
    ul[role="listbox"] {
        background: #ffffff !important;
        border: 1px solid rgba(0, 0, 0, 0.14) !important;
    }
    li[role="option"] {
        color: #050607 !important;
    }

    /* Primary button: white box with black text */
    .stButton > button[kind="primary"] {
        background: #ffffff !important;
        color: #050607 !important;
        border: 1px solid rgba(242, 106, 46, 0.55) !important;
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.45) !important;
    }
    /* Some Streamlit buttons (e.g. form submit) render via different wrappers */
    .stFormSubmitButton > button,
    .stFormSubmitButton > button[kind="primary"],
    button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background: #ffffff !important;
        color: #050607 !important;
        -webkit-text-fill-color: #050607 !important;
        border: 1px solid rgba(242, 106, 46, 0.55) !important;
    }
    .stFormSubmitButton > button *,
    button[kind="primary"] * {
        color: #050607 !important;
        -webkit-text-fill-color: #050607 !important;
        fill: #050607 !important;
    }

    /* Donate button (Stripe link) */
    .tb-donate-btn {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.56rem 0.92rem;
        border-radius: 999px;
        background: var(--tb-yellow);
        color: #050607 !important;
        border: 1px solid rgba(255, 255, 255, 0.18);
        font-weight: 900;
        text-decoration: none !important;
        box-shadow: 0 18px 44px rgba(255, 212, 0, 0.14);
    }
    .tb-donate-btn:hover {
        filter: saturate(1.05);
        transform: translateY(-1px);
        box-shadow: 0 24px 56px rgba(255, 212, 0, 0.18);
    }
    .tb-donate-sub {
        color: var(--tb-text-dim);
        font-size: 0.92rem;
        margin: 0.55rem 0 0;
        max-width: 44rem;
        line-height: 1.45;
    }

    /* Text inputs / text areas should stay light with dark text */
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea,
    textarea,
    input[type="text"],
    input[type="number"] {
        background: #ffffff !important;
        color: #050607 !important;
        -webkit-text-fill-color: #050607 !important;
        caret-color: var(--tb-orange) !important;
    }
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div {
        background: #ffffff !important;
        border-color: rgba(0, 0, 0, 0.14) !important;
    }
    *:focus-visible {
        outline: 3px solid var(--tb-focus);
        outline-offset: 2px;
        border-radius: 10px;
    }

    /* Fun but clean: solid blocks + accent glow (no gradients) */
    .tb-hero {
        border: 1px solid var(--tb-border);
        border-radius: 18px;
        background: var(--tb-surface);
        padding: 1.25rem 1.35rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 18px 60px rgba(0, 0, 0, 0.55);
    }
    .tb-eyebrow {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: var(--tb-text-dim);
        margin: 0 0 0.35rem;
    }

    .tb-card {
        border: 1px solid var(--tb-border);
        border-radius: 18px;
        background: var(--tb-surface);
        padding: 1.05rem 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.45);
    }
    .tb-section-label { color: var(--tb-orange); font-weight: 800; }
    .tb-muted { color: var(--tb-text-dim); }

    .tb-chip {
        display: inline-block;
        padding: 0.28rem 0.62rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.16);
        font-size: 0.78rem;
        font-weight: 650;
        margin-right: 0.45rem;
        margin-top: 0.35rem;
        color: rgba(255, 255, 255, 0.86);
    }
    .tb-chip:nth-child(4n+1) { border-color: rgba(255, 212, 0, 0.55); }
    .tb-chip:nth-child(4n+2) { border-color: rgba(11, 132, 243, 0.55); }
    .tb-chip:nth-child(4n+3) { border-color: rgba(31, 111, 75, 0.55); }
    .tb-chip:nth-child(4n+4) { border-color: rgba(242, 106, 46, 0.55); }

    /* MidSouth-ish: big clean links instead of gradients */
    a, a:visited {
        color: rgba(255, 255, 255, 0.92);
        text-decoration: none;
        border-bottom: 2px solid rgba(255, 212, 0, 0.75);
    }
    a:hover {
        border-bottom-color: rgba(242, 106, 46, 0.92);
    }

    /* Streamlit sliders: move active color to orange */
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        box-shadow: 0 0 0 4px rgba(242, 106, 46, 0.22) !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] div[aria-valuenow] {
        background-color: rgba(242, 106, 46, 1) !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] div[aria-hidden="true"] {
        background-color: rgba(242, 106, 46, 0.55) !important;
    }

    /* Primary button styling is defined once above (white box, black text). */

    [data-testid="stExpander"] details {
        border: 1px solid var(--tb-border) !important;
        border-radius: 18px !important;
        background: var(--tb-surface-2) !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--tb-border);
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.45);
        background: var(--tb-surface);
    }

    /* Section separators: straight orange line */
    .tb-divider-label {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin: 1.65rem 0 1.05rem;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: rgba(255, 255, 255, 0.78);
    }
    .tb-divider-label::after {
        content: "";
        flex: 1;
        height: 2px;
        background: rgba(242, 106, 46, 0.92);
        box-shadow: 0 0 0 1px rgba(242, 106, 46, 0.10);
    }

    /* Streamlit divider (<hr>): force straight orange line */
    hr {
        border: none !important;
        height: 2px !important;
        margin: 1.35rem 0 !important;
        background: rgba(242, 106, 46, 0.92) !important;
        box-shadow: 0 0 0 1px rgba(242, 106, 46, 0.10);
        opacity: 1;
    }

    /* Streamlit divider component (some builds render dotted/oval shapes) */
    [data-testid="stDivider"] {
        height: 2px !important;
        background: rgba(242, 106, 46, 0.92) !important;
        border-radius: 999px;
        box-shadow: 0 0 0 1px rgba(242, 106, 46, 0.10);
    }
    [data-testid="stDivider"] * {
        display: none !important;
    }

    /* Our own section rule */
    .tb-rule {
        height: 2px;
        width: 100%;
        background: rgba(242, 106, 46, 0.92);
        border-radius: 999px;
        box-shadow: 0 0 0 1px rgba(242, 106, 46, 0.10);
        margin: 1.35rem 0;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def discover_events(routes_dir: Path) -> Dict[str, Path]:
    events: Dict[str, Path] = {}
    for child in sorted(routes_dir.iterdir(), key=lambda p: p.name.lower()):
        if child.is_dir():
            segment_files = sorted(
                [p for p in child.glob("*.csv") if "segment" in p.name.lower()],
                key=lambda p: p.name.lower(),
            )
            if segment_files:
                # Use folder name for display, first matching segments CSV for data.
                events[child.name] = segment_files[0]
            else:
                # Nested routes, e.g. Routes/User uploads/<route_name>/route_segments.csv
                for sub in sorted(child.iterdir(), key=lambda p: p.name.lower()):
                    if not sub.is_dir():
                        continue
                    nested = sorted(
                        [p for p in sub.glob("*.csv") if "segment" in p.name.lower()],
                        key=lambda p: p.name.lower(),
                    )
                    if nested:
                        events[f"{child.name} / {sub.name}"] = nested[0]

    # Fallback: if no folder-based events are found, support top-level segment CSVs.
    if not events:
        for path in routes_dir.glob("*.csv"):
            if "segment" in path.name.lower():
                events[path.stem] = path
    return dict(sorted(events.items(), key=lambda x: x[0].lower()))


def find_event_gpx(route_csv: Path) -> Optional[Path]:
    parent = route_csv.parent
    preferred = parent / "route.gpx"
    if preferred.exists():
        return preferred
    gpx_files = sorted(parent.glob("*.gpx"), key=lambda p: p.name.lower())
    if not gpx_files:
        return None
    return gpx_files[0]


def route_submission_mailto() -> str:
    subject = "TireBot route — segment CSV + GPX"
    body = (
        "Hi,\n\n"
        "Please add this race to TireBot. Attached:\n"
        "- Segment CSV (filename must contain \"segment\", distances in miles)\n"
        "- GPX (required — elevation and track length)\n\n"
        "Event name:\n\n"
        "Thanks!\n"
    )
    q_sub = urllib.parse.quote(subject)
    q_body = urllib.parse.quote(body)
    return f"mailto:{FEEDBACK_EMAIL}?subject={q_sub}&body={q_body}"


def route_roughness_score(segments: List[Segment]) -> float:
    if not segments:
        return 0.0
    surface_score = {"road": 0.0, "cat1": 0.9, "cat2": 1.8, "cat3": 3.0, "above": 3.8}
    total_distance = sum(s.distance_km for s in segments)
    if total_distance <= 0:
        return 0.0
    weighted = sum(surface_score[s.surface] * s.distance_km for s in segments)
    return weighted / total_distance


def estimate_pressure(width_mm: float, weight_kg: float, speed_tier: str, roughness: float) -> Tuple[float, float]:
    speed_adj = {"pro": 2.0, "amateur": 0.0, "ride": -2.0}[speed_tier]
    weight_adj = (weight_kg - 75.0) * 0.12
    width_adj = -0.35 * (width_mm - 40.0)
    roughness_drop = roughness * 1.2

    base_center = 36.0 + speed_adj + weight_adj + width_adj - roughness_drop
    support_floor = SUPPORT_PSI_FACTOR * (weight_kg / max(1.0, width_mm))
    base_center = max(base_center, support_floor)
    front = base_center - 1.2
    rear = base_center + 1.0

    # Practical clamp range for tubeless gravel/xc setups.
    front = max(17.0, min(42.0, front))
    rear = max(19.0, min(45.0, rear))
    return round(front, 1), round(rear, 1)


def load_tire_mass_overrides(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    values: Dict[str, float] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("tire_name") or "").strip()
            if not name:
                continue
            try:
                values[name] = float((row.get("weight_g") or "").strip())
            except ValueError:
                continue
    return values


def speed_tier_from_avg_mph(mph: float) -> str:
    """Legacy coarse tiers used in the heuristic pressure model."""
    if mph >= 23.0:
        return "pro"
    if mph < 18.0:
        return "ride"
    return "amateur"


def is_leadville_route(route_csv: Path, event_label: str) -> bool:
    label = (event_label or "").lower()
    if "leadville" in label:
        return True
    parts = [p.lower() for p in route_csv.parts]
    return any("leadville" in p for p in parts)


def phase_weight(race_position: float, early_boost: float) -> float:
    p = max(0.0, min(1.0, race_position))
    if p <= 0.25:
        return early_boost
    if p <= 0.70:
        return 1.15
    return 1.0


def effective_weighted_distance(segments: List[Segment], early_boost: float) -> float:
    return sum(
        seg.distance_km * phase_weight(seg.race_position, early_boost) * seg.technicality * seg.selection_risk
        for seg in segments
    )


def segments_first_n_minutes(segments: List[Segment], avg_speed_mph: float, minutes: float) -> List[Segment]:
    """Approximate the first N minutes of the race by distance at avg speed.

    Uses segment ordering by race_position and truncates the last segment proportionally.
    """
    if not segments or avg_speed_mph <= 0 or minutes <= 0:
        return []
    target_km = (avg_speed_mph * 1.60934) * (minutes / 60.0)
    if target_km <= 0:
        return []

    out: List[Segment] = []
    remaining = target_km
    ordered = sorted(segments, key=lambda s: s.race_position)
    for seg in ordered:
        if remaining <= 1e-9:
            break
        if seg.distance_km <= remaining + 1e-9:
            out.append(seg)
            remaining -= seg.distance_km
            continue
        # Partial segment
        frac = max(0.0, min(1.0, remaining / max(seg.distance_km, 1e-9)))
        out.append(
            Segment(
                name=f"{seg.name} (partial)",
                distance_km=seg.distance_km * frac,
                surface=seg.surface,
                technicality=seg.technicality,
                selection_risk=seg.selection_risk,
                race_position=seg.race_position,
            )
        )
        remaining = 0.0
        break
    return out


def estimate_rr_watts(total_score: float, weighted_distance: float, speed_mph: float, weight_kg: float) -> float:
    if weighted_distance <= 0:
        return 0.0
    effective_crr = total_score / weighted_distance
    system_mass_kg = weight_kg + 9.0  # rider + bike/kit estimate
    speed_mps = speed_mph * 0.44704
    watts = effective_crr * system_mass_kg * 9.80665 * speed_mps
    return round(watts, 1)


def estimate_rr_watts_raw(total_score: float, weighted_distance: float, speed_mph: float, weight_kg: float) -> float:
    if weighted_distance <= 0:
        return 0.0
    effective_crr = total_score / weighted_distance
    system_mass_kg = weight_kg + 9.0
    speed_mps = speed_mph * 0.44704
    return effective_crr * system_mass_kg * 9.80665 * speed_mps


def impedance_penalty_route_score(
    segments: List[Segment],
    early_boost: float,
    width_mm: float,
    weight_kg: float,
    speed_tier: str,
) -> float:
    """Extra route_score to model rough-surface impedance losses.

    Uses a simple stiffness proxy:
      stiffness ~ (avg_psi * width_mm) / rider_weight_kg

    On rougher surfaces, being too stiff adds CRR-equivalent penalty.
    """
    if not segments or width_mm <= 0 or weight_kg <= 0:
        return 0.0

    penalty = 0.0
    for seg in segments:
        k = IMPEDANCE_K_BY_SURFACE.get(seg.surface, 0.0)
        target = IMPEDANCE_TARGET_STIFFNESS.get(seg.surface, 0.0)
        if k <= 0 or target <= 0:
            continue

        # Segment-specific heuristic pressure: rougher segments allow/encourage lower PSI.
        roughness_surface = {"road": 0.0, "cat1": 0.9, "cat2": 1.8, "cat3": 3.0}.get(seg.surface, 1.2)
        f_psi, r_psi = estimate_pressure(width_mm, weight_kg, speed_tier, roughness_surface)
        avg_psi = max(1.0, (f_psi + r_psi) / 2.0)

        stiffness = (avg_psi * max(1.0, width_mm)) / max(1e-9, weight_kg)
        if stiffness <= target:
            continue

        stiff_ratio = (stiffness / target) - 1.0
        delta_crr = k * (stiff_ratio**IMPEDANCE_GAMMA)
        w = phase_weight(seg.race_position, early_boost) * seg.technicality * seg.selection_risk
        penalty += delta_crr * seg.distance_km * w

    return penalty


def estimate_aero_penalty_watts(width_mm: float, speed_mph: float) -> float:
    # Reference 40 mm as aero baseline; wider tires add drag.
    baseline_width_mm = 40.0
    speed_ref_mps = 8.94  # ~20 mph
    speed_mps = speed_mph * 0.44704
    width_delta = width_mm - baseline_width_mm
    penalty = 0.35 * width_delta * ((speed_mps / speed_ref_mps) ** 3)
    return round(penalty, 1)


def estimate_tire_mass_grams(width_mm: float, tire_name: str, overrides: Dict[str, float]) -> float:
    if tire_name in overrides:
        return overrides[tire_name]
    # Lightweight proxy if exact tire mass is unavailable.
    # ~400 g at 40 mm, scaled by width.
    est = 400.0 + (width_mm - 40.0) * 13.0
    return max(300.0, min(950.0, est))


def haversine_km(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))


def gpx_parse_track_points(gpx_path: Path) -> List[Tuple[float, float, Optional[float]]]:
    raw = gpx_path.read_text(encoding="utf-8", errors="ignore")
    pts: List[Tuple[float, float, Optional[float]]] = []
    for m in re.finditer(
        r'<trkpt\s+lat="([0-9\.-]+)"\s+lon="([0-9\.-]+)"[^>]*>(.*?)</trkpt>',
        raw,
        re.DOTALL | re.IGNORECASE,
    ):
        inner = m.group(3)
        em = re.search(r"<ele>([-0-9.]+)</ele>", inner, re.IGNORECASE)
        ele = float(em.group(1)) if em else None
        pts.append((float(m.group(1)), float(m.group(2)), ele))
    return pts


def _smooth_1d(values: List[float], window: int) -> List[float]:
    if len(values) < window or window < 3:
        return list(values)
    half = window // 2
    out: List[float] = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def gpx_aero_eligible_breakdown(gpx_path: Optional[Path]) -> Tuple[float, float, float]:
    """Horizontal km where aero applies vs total GPX horizontal km.

    Aero-eligible segments are not steep downhills (coasting) nor steep climbs.
    Returns ``(d_aero_km, d_total_km, fraction)``. If GPX/elevation missing, ``(0, 0, 1.0)``.
    """
    if gpx_path is None or not gpx_path.exists():
        return 0.0, 0.0, 1.0
    pts = gpx_parse_track_points(gpx_path)
    if len(pts) < 2:
        return 0.0, 0.0, 1.0
    if any(p[2] is None for p in pts):
        return 0.0, 0.0, 1.0
    lats = [p[0] for p in pts]
    lons = [p[1] for p in pts]
    eles = [float(p[2]) for p in pts]
    win = min(5, len(eles))
    if win >= 3:
        eles = _smooth_1d(eles, win)
    d_aero = 0.0
    d_total = 0.0
    for i in range(1, len(pts)):
        h_km = haversine_km((lats[i - 1], lons[i - 1]), (lats[i], lons[i]))
        if h_km < 1e-9:
            continue
        dz_m = eles[i] - eles[i - 1]
        grade = dz_m / (h_km * 1000.0)
        d_total += h_km
        if AERO_EXCLUDE_DOWNHILL_GRADE <= grade <= AERO_EXCLUDE_STEEP_CLIMB_GRADE:
            d_aero += h_km
    if d_total <= 0:
        return 0.0, 0.0, 1.0
    return d_aero, d_total, d_aero / d_total


def gpx_total_elevation_gain_m(gpx_path: Path) -> float:
    # GPX 1.1: <ele> is meters above reference ellipsoid (WGS84).
    raw = gpx_path.read_text(encoding="utf-8", errors="ignore")
    elevations = [float(x) for x in re.findall(r"<ele>([-0-9.]+)</ele>", raw)]
    if len(elevations) < 2:
        return 0.0
    gain = 0.0
    for i in range(1, len(elevations)):
        delta = elevations[i] - elevations[i - 1]
        if delta > 0:
            gain += delta
    return gain


def gpx_track_length_mi(gpx_path: Path) -> float:
    """Great-circle distance along `<trkpt>` sequence; reported in miles (not read from GPX tags)."""
    raw = gpx_path.read_text(encoding="utf-8", errors="ignore")
    pts = [(float(a), float(b)) for a, b in re.findall(r'<trkpt lat="([0-9\.-]+)" lon="([0-9\.-]+)">', raw)]
    if len(pts) < 2:
        return 0.0
    total_km = sum(haversine_km(pts[i - 1], pts[i]) for i in range(1, len(pts)))
    return round(total_km * KM_TO_MI, 2)


def estimate_tire_mass_penalty_watts(
    tire_mass_g: float,
    route_distance_km: float,
    total_elev_gain_m: float,
    speed_mph: float,
) -> float:
    # Compare against a 450 g per-tire baseline (x2 tires).
    baseline_pair_kg = 0.9
    pair_mass_kg = (tire_mass_g * 2.0) / 1000.0
    delta_mass_kg = pair_mass_kg - baseline_pair_kg
    if abs(delta_mass_kg) < 1e-9:
        return 0.0
    speed_kmh = speed_mph * 1.60934
    if speed_kmh <= 0.1:
        return 0.0
    total_time_s = (route_distance_km / speed_kmh) * 3600.0
    if total_time_s <= 1.0:
        return 0.0
    watts = (delta_mass_kg * 9.80665 * total_elev_gain_m) / total_time_s
    return round(watts, 2)


def rank_by_fastest_total_watts(
    ranked_by_rr: List,
    weighted_distance: float,
    route_distance_km: float,
    total_elev_gain_m: float,
    speed_mph: float,
    weight_kg: float,
    mass_overrides: Dict[str, float],
    segments: Optional[List[Segment]] = None,
    early_boost: float = 1.8,
    speed_tier: str = "amateur",
    aero_distance_fraction: float = 1.0,
) -> List[Dict[str, float]]:
    frac = max(0.0, min(1.0, aero_distance_fraction))
    rows: List[Dict[str, float]] = []
    for r in ranked_by_rr:
        width_mm = r.width_mm if r.width_mm is not None else 45.0
        impedance_score = (
            impedance_penalty_route_score(segments, early_boost, width_mm, weight_kg, speed_tier)
            if segments is not None
            else 0.0
        )
        rr_watts = estimate_rr_watts(r.total_score + impedance_score, weighted_distance, speed_mph, weight_kg)
        impedance_watts = round(estimate_rr_watts_raw(impedance_score, weighted_distance, speed_mph, weight_kg), 2)
        aero_penalty_watts = round(estimate_aero_penalty_watts(width_mm, speed_mph) * frac, 1)
        tire_mass_g = estimate_tire_mass_grams(width_mm, r.tire_name, mass_overrides)
        mass_penalty_watts = estimate_tire_mass_penalty_watts(
            tire_mass_g, route_distance_km, total_elev_gain_m, speed_mph
        )
        total_watts = round(rr_watts + aero_penalty_watts + mass_penalty_watts, 2)
        rows.append(
            {
                "tire_name": r.tire_name,
                "width_mm": width_mm,
                "score": round(r.total_score, 4),
                "rr_watts": rr_watts,
                "impedance_watts": impedance_watts,
                "aero_penalty_watts": aero_penalty_watts,
                "tire_mass_g": round(tire_mass_g, 0),
                "mass_penalty_watts": mass_penalty_watts,
                "total_watts": total_watts,
            }
        )
    return sorted(rows, key=lambda x: x["total_watts"])


def dataframe_height_for_rows(
    n_rows: int,
    *,
    header_px: int = 52,
    row_px: int = 34,
    extra_px: int = 12,
    max_px: int = 900,
) -> int:
    """Compute a table height that shows all rows without scrolling.

    Streamlit's rendered row height varies a bit by OS/browser; we include a small buffer.
    """
    n = max(0, int(n_rows))
    h = header_px + (n * row_px) + extra_px
    return min(max_px, max(header_px + extra_px, h))


def render_feedback_footer(route_label: Optional[str] = None) -> None:
    """Opens the visitor's email client (mailto:). No server-side email on Streamlit Cloud."""
    st.markdown('<div class="tb-rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="tb-card">', unsafe_allow_html=True)
    st.markdown('<p class="tb-section-label">Contact</p>', unsafe_allow_html=True)
    st.subheader("Feedback")
    st.caption(
        "Opens your email app to send to the TireBot maintainer. Nothing is transmitted from this server."
    )
    ctx = route_label if route_label and route_label != "Pick a Route/Event" else "Not specified"

    with st.form("tirebot_feedback_form", clear_on_submit=False):
        message = st.text_area(
            "Your message",
            placeholder="Bugs, feature ideas, or data corrections…",
            height=120,
            max_chars=4000,
        )
        send = st.form_submit_button("Email feedback", use_container_width=True, type="primary")

    if send:
        if not message.strip():
            st.warning("Please enter a message before sending.")
            st.session_state.pop("tirebot_feedback_mailto", None)
        else:
            body = f"Route / context: {ctx}\n\n{message.strip()}"
            max_body = 1800
            if len(body) > max_body:
                body = body[: max_body - 3] + "..."
            q_sub = urllib.parse.quote("TireBot feedback")
            q_body = urllib.parse.quote(body)
            st.session_state["tirebot_feedback_mailto"] = (
                f"mailto:{FEEDBACK_EMAIL}?subject={q_sub}&body={q_body}"
            )

    mailto_href = st.session_state.get("tirebot_feedback_mailto")
    if mailto_href:
        st.success(f"Click below to send from your email app to **{FEEDBACK_EMAIL}**.")
        st.markdown(f"[**Open email to {FEEDBACK_EMAIL}**]({mailto_href})")
        if st.button("Dismiss email link", key="tirebot_feedback_dismiss"):
            st.session_state.pop("tirebot_feedback_mailto", None)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


def summarize_route(path: Path) -> Dict[str, float]:
    # Small helper to show rough course composition to the rider.
    segments = load_segments(path)
    totals = {"road": 0.0, "cat1": 0.0, "cat2": 0.0, "cat3": 0.0, "above": 0.0}
    for seg in segments:
        totals[seg.surface] += seg.distance_km
    total_dist = sum(totals.values()) or 1.0
    return {
        "distance_km": total_dist,
        "distance_mi": km_to_mi(total_dist),
        "road_pct": (totals["road"] / total_dist) * 100.0,
        "cat1_pct": (totals["cat1"] / total_dist) * 100.0,
        "cat2_pct": (totals["cat2"] / total_dist) * 100.0,
        "cat3_pct": (totals["cat3"] / total_dist) * 100.0,
        "above_pct": (totals["above"] / total_dist) * 100.0,
    }


def main() -> None:
    st.set_page_config(page_title="TreadLab: Pick the perfect tire.", page_icon="🚴", layout="wide")
    inject_styles()
    sha = current_git_sha_short()
    build_no = current_build_number()
    st.markdown(
        f"""
<div class="tb-hero">
  <p class="tb-eyebrow">Gravel · MTB · race day setup</p>
  <h1>TreadLab: Pick the perfect tire.</h1>
  <p>Route-aware tire, pressure, and risk recommendations for race day.</p>
  <div style="margin-top: 0.85rem;">
    <span class="tb-chip">Route-aware</span>
    <span class="tb-chip">CRR-based</span>
    <span class="tb-chip">Race-day pressure</span>
    <span class="tb-chip">Risk labels</span>
  </div>
  <div style="margin-top: 0.95rem; display:flex; gap:0.65rem; flex-wrap:wrap; align-items:center;">
    <a class="tb-donate-btn" href="{STRIPE_DONATE_URL}" target="_blank" rel="noopener noreferrer">Donate $5</a>
    <span class="tb-donate-sub">Optional tip to support TreadLab’s data + development.</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    if build_no is not None and sha:
        st.caption(f"Build: **#{build_no}** · `{sha}`")
    elif build_no is not None:
        st.caption(f"Build: **#{build_no}**")
    elif sha:
        st.caption(f"Build: `{sha}`")
    st.markdown('<div class="tb-card">', unsafe_allow_html=True)
    st.markdown('<p class="tb-section-label">Science &amp; assumptions</p>', unsafe_allow_html=True)
    st.markdown("### Methodology")
    st.markdown(
        f"**[Read the whitepaper]({WHITEPAPER_ONLINE_PAGES_URL})** — methodology, data sources, assumptions, and calculations."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    events = discover_events(ROUTES_DIR)

    st.markdown('<div class="tb-card">', unsafe_allow_html=True)
    st.markdown('<p class="tb-section-label">Your ride</p>', unsafe_allow_html=True)
    st.subheader("Ride inputs")

    if not events:
        st.warning(
            "No routes found in `Routes/`. Add event folders with a `*segment*.csv` and a **GPX** for that course, "
            f"or email **both** to **[{FEEDBACK_EMAIL}]({route_submission_mailto()})** to have a course added."
        )
        render_feedback_footer(None)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    tpl = ROUTES_DIR / "example_route_segments.csv"
    with st.expander("Don't see your event? Send a route", expanded=False):
        st.markdown(
            f"Email **{FEEDBACK_EMAIL}** with:\n\n"
            "- **Segment CSV** — required; filename should contain `segment`, with distances in **miles** "
            "(`segment_start` / `segment_end` or `distance_mi`). See the whitepaper / README for surface columns.\n"
            "- **GPX** — **required**; used for elevation gain and track length when we add the route to the library.\n\n"
            f"[Open email draft]({route_submission_mailto()}) (attach your files in your mail app after it opens)."
        )
        if tpl.exists():
            st.download_button(
                "Download segment CSV template",
                data=tpl.read_bytes(),
                file_name="tirebot_segment_template.csv",
                mime="text/csv",
                key="tirebot_segment_template_dl",
            )

    with st.form("tirebot_inputs"):
        i1, i2 = st.columns(2)
        route_options = ["Pick a Route/Event"] + list(events.keys())
        event_label = i1.selectbox(
            "Route",
            route_options,
            index=0,
            help="Routes are discovered from the Routes folder.",
        )
        weight_kg = i2.number_input("Rider weight (kg)", min_value=45.0, max_value=130.0, value=75.0, step=0.5)

        avg_speed_mph = st.slider(
            "Average speed (mph)",
            min_value=10.0,
            max_value=35.0,
            value=20.0,
            step=0.5,
            help="Rolling resistance uses this speed everywhere. Aero penalty uses this speed scaled by GPX: only "
            "distance that is not a steep downhill or steep climb counts (see results footnote).",
        )

        with st.expander("Advanced options", expanded=False):
            early_boost = st.slider(
                "Early-race weighting",
                min_value=1.0,
                max_value=3.0,
                value=1.8,
                step=0.1,
                help="Higher values prioritize early-race performance. Applied by segment position: first 25% × this value, "
                "25–70% × 1.15, last 30% × 1.0.",
            )
            st.caption(
                "How it works: TireBot weights each segment by its position in the course. "
                "If you’re racing for the front group, a higher early weighting can better reflect the cost of losing contact early."
            )
            top_n = st.slider("Top tire options", min_value=3, max_value=20, value=5, step=1)

        st.markdown("**Race start tire** (optional)")
        show_first_90 = st.checkbox(
            "Find best tire for the race start",
            value=False,
            key="tirebot_show_race_start",
            help="Pick an early-race time window (minutes) and re-rank tires on just that initial portion of the route.",
        )
        race_start_minutes = st.slider(
            "Race start window (minutes)",
            min_value=30,
            max_value=90,
            value=60,
            step=5,
            key="tirebot_race_start_minutes",
            help="How much of the route to consider for the start-of-race recommendation.",
        )
        early_speed_mph = st.slider(
            "Race start speed (mph)",
            min_value=10.0,
            max_value=40.0,
            value=23.0,
            step=0.5,
            key="tirebot_race_start_speed_mph",
            help="Used only for the race-start view. This speed determines the distance cutoff and the watts in that early ranking.",
        )

        submitted = st.form_submit_button("Generate recommendation", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    route_context_str = event_label

    if not submitted:
        st.info("Choose a route, rider weight, and average speed, then run **Generate recommendation**.")
        render_feedback_footer(route_context_str)
        return

    if event_label == "Pick a Route/Event":
        st.warning("Please select a route/event before generating recommendations.")
        render_feedback_footer(route_context_str)
        return

    speed_tier = speed_tier_from_avg_mph(avg_speed_mph)

    route_csv = events[event_label]
    tires = load_tires_with_optional_brr(TIRES_CSV, BRR_CRR_CSV)
    if is_leadville_route(route_csv, event_label):
        mtb_only = [
            t
            for t in tires
            if (t.get("width_mm") or 0.0) >= MTB_MIN_WIDTH_MM
            and "corsa" not in str(t.get("tire_name", "")).lower()
        ]
        if mtb_only:
            tires = mtb_only
            st.info(f"Leadville filter: showing **MTB tires only** (≥ {MTB_MIN_WIDTH_IN:.1f}\" width).")
    segments = load_segments(route_csv)

    above_mi = above_distance_mi(segments)
    if above_mi > ABOVE_CATEGORY_MTB_ONLY_THRESHOLD_MI:
        mtb_only = [t for t in tires if (t.get("width_mm") or 0.0) >= MTB_MIN_WIDTH_MM]
        if mtb_only:
            tires = mtb_only
            st.info(
                f"Above Category filter: route has **{above_mi:.1f} mi** tagged `above` → recommending **MTB tires only** (≥ {MTB_MIN_WIDTH_IN:.1f}\")."
            )

    mass_overrides = load_tire_mass_overrides(TIRE_MASS_CSV)
    route_stats = summarize_route(route_csv)
    route_gpx = find_event_gpx(route_csv)
    total_elev_gain_m = gpx_total_elevation_gain_m(route_gpx) if route_gpx else 0.0
    gpx_track_mi = gpx_track_length_mi(route_gpx) if route_gpx else 0.0
    _d_aero_km, _d_gpx_km, aero_distance_fraction = gpx_aero_eligible_breakdown(route_gpx)
    aero_course_mi = route_stats["distance_mi"] * aero_distance_fraction
    if not route_gpx:
        aero_footnote_html = (
            "<strong>Aero penalty</strong> uses the average speed slider over the full route (no GPX for this event). "
        )
    elif _d_gpx_km <= 0:
        aero_footnote_html = (
            "<strong>Aero penalty</strong> uses the average speed slider over the full route "
            "(GPX track has no elevation on points, or track is too short to analyze). "
        )
    else:
        aero_footnote_html = (
            f"<strong>Aero penalty</strong> uses {avg_speed_mph:.1f} mph, scaled by GPX distance where grade is between "
            f"about {abs(AERO_EXCLUDE_DOWNHILL_GRADE) * 100:.1f}% descent and {AERO_EXCLUDE_STEEP_CLIMB_GRADE * 100:.0f}% climb "
            f"(steeper downhills and steeper climbs excluded). "
            f"That is ~{aero_distance_fraction * 100:.0f}% of segment course miles (~{aero_course_mi:.1f} mi of {route_stats['distance_mi']:.1f} mi). "
        )
    elev_gain_ft = total_elev_gain_m * M_TO_FT
    ranked_by_rr = score_tires(tires, segments, early_boost)
    if not ranked_by_rr:
        st.error("No tires could be scored. Check CSV data completeness.")
        render_feedback_footer(route_context_str)
        return

    weighted_distance = effective_weighted_distance(segments, early_boost)
    ranked = rank_by_fastest_total_watts(
        ranked_by_rr,
        weighted_distance,
        route_stats["distance_km"],
        total_elev_gain_m,
        avg_speed_mph,
        weight_kg,
        mass_overrides,
        segments=segments,
        early_boost=early_boost,
        speed_tier=speed_tier,
        aero_distance_fraction=aero_distance_fraction,
    )
    winner = ranked[0]
    winner_width = winner["width_mm"]
    roughness = route_roughness_score(segments)
    front_psi, rear_psi = estimate_pressure(winner_width, weight_kg, speed_tier, roughness)
    pressure_source = "Heuristic model"
    winner_rr_watts = winner["rr_watts"]
    winner_aero_penalty = winner["aero_penalty_watts"]
    winner_total_watts = winner["total_watts"]
    cat2_mi = cat2_distance_mi(segments)
    cat3_mi = cat3_distance_mi(segments)
    risk_examples = load_risk_examples()
    risk_label_fallback = tire_issue_risk_for_tire(
        winner_width,
        cat2_mi,
        cat3_mi,
        tire_name=winner["tire_name"],
        tire_class=winner.get("tire_class", ""),
        above_mi=above_mi,
    )
    risk_label = tire_issue_risk_label(
        examples=risk_examples,
        segments=segments,
        tire_width_mm=winner_width,
        tire_class=str(winner.get("tire_class", "")),
        fallback_label=risk_label_fallback,
    )
    risk_badge = risk_badge_html(risk_label)

    st.markdown('<div class="tb-divider-label">Snapshot</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Route Distance", f"{route_stats['distance_mi']:.1f} mi")
    m2.metric("Avg speed", f"{avg_speed_mph:.1f} mph")
    m3.metric("Fastest Tire", winner["tire_name"])
    m4.metric("Pressure (F / R)", f"{front_psi:.1f} / {rear_psi:.1f} psi")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="tb-card">', unsafe_allow_html=True)
        st.markdown('<p class="tb-section-label">Winner</p>', unsafe_allow_html=True)
        st.subheader("Recommendation")
        wn = html.escape(str(winner["tire_name"]))
        st.markdown(
            f"""
<div class="tb-rec-grid">
  <div class="tb-rec-item"><span class="tb-rec-k">Tire choice</span><span class="tb-rec-v">{wn}</span></div>
  <div class="tb-rec-item"><span class="tb-rec-k">Pressure (F / R)</span><span class="tb-rec-v">{front_psi:.1f} / {rear_psi:.1f} psi</span></div>
  <div class="tb-rec-item"><span class="tb-rec-k">Tire issue risk</span><span class="tb-rec-v">{risk_badge}</span></div>
  <div class="tb-rec-item"><span class="tb-rec-k">Rolling resistance</span><span class="tb-rec-v">{winner_rr_watts:.1f} W</span></div>
  <div class="tb-rec-item"><span class="tb-rec-k">Aero width penalty</span><span class="tb-rec-v">{winner_aero_penalty:+.1f} W</span></div>
  <div class="tb-rec-item"><span class="tb-rec-k">Tire mass penalty</span><span class="tb-rec-v">{winner['mass_penalty_watts']:+.2f} W ({winner['tire_mass_g']:.0f} g / tire)</span></div>
  <div class="tb-rec-item"><span class="tb-rec-k">Total resistance</span><span class="tb-rec-v">{winner_total_watts:.1f} W</span></div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="tb-muted">Source: {pressure_source}. Route length from segments: {route_stats["distance_mi"]:.1f} mi. '
            + (
                f'GPX track (lat/lon): {gpx_track_mi:.1f} mi; elevation gain: {elev_gain_ft:,.0f} ft ({total_elev_gain_m:.0f} m, standard GPX &lt;ele&gt;). '
                if route_gpx
                else ""
            )
            + aero_footnote_html
            + f"Surface mix, early-race weighting {early_boost:.1f}, average speed {avg_speed_mph:.1f} mph, rider weight {weight_kg:.1f} kg. Segment CSV uses miles along course.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="tb-card">', unsafe_allow_html=True)
        st.markdown('<p class="tb-section-label">Course mix</p>', unsafe_allow_html=True)
        st.subheader("Route composition")
        st.markdown(f"**Tire issue risk (proxy):** `{risk_label}` (based on Cat 2/3 miles + tire width)")
        st.caption(
            f"Cat 2 distance: **{cat2_mi:.1f} mi** · Cat 3 distance: **{cat3_mi:.1f} mi**. "
            f"MTB tires ≥ {RISK_MTB_NO_RISK_MIN_IN:.1f}\" are treated as Low risk on Cat 3."
        )
        st.progress(min(max(route_stats["road_pct"] / 100.0, 0.0), 1.0), text=f"Road: {route_stats['road_pct']:.1f}%")
        st.progress(min(max(route_stats["cat1_pct"] / 100.0, 0.0), 1.0), text=f"Cat 1: {route_stats['cat1_pct']:.1f}%")
        st.progress(min(max(route_stats["cat2_pct"] / 100.0, 0.0), 1.0), text=f"Cat 2: {route_stats['cat2_pct']:.1f}%")
        st.progress(min(max(route_stats["cat3_pct"] / 100.0, 0.0), 1.0), text=f"Cat 3: {route_stats['cat3_pct']:.1f}%")
        st.progress(min(max(route_stats.get("above_pct", 0.0) / 100.0, 0.0), 1.0), text=f"Above: {route_stats.get('above_pct', 0.0):.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="tb-divider-label">Full comparison</div>', unsafe_allow_html=True)
    st.subheader("Top tire rankings")
    rows = []
    for idx, result in enumerate(ranked[:top_n], start=1):
        width_text = f"{result['width_mm']:.1f}"
        tire_width = result["width_mm"]
        f_psi, r_psi = estimate_pressure(tire_width, weight_kg, speed_tier, roughness)
        risk_fallback = tire_issue_risk_for_tire(
            tire_width,
            cat2_mi,
            cat3_mi,
            tire_name=result["tire_name"],
            tire_class=result.get("tire_class", ""),
            above_mi=above_mi,
        )
        risk = tire_issue_risk_label(
            examples=risk_examples,
            segments=segments,
            tire_width_mm=tire_width,
            tire_class=str(result.get("tire_class", "")),
            fallback_label=risk_fallback,
        )
        rows.append(
            {
                "Rank": idx,
                "Tire": result["tire_name"],
                "Width (mm)": width_text,
                "Risk": risk,
                "Route Score": result["score"],
                "RR (W)": result["rr_watts"],
                "Imp (W)": result.get("impedance_watts", 0.0),
                "Aero (W)": result["aero_penalty_watts"],
                "Tire Mass (g)": result["tire_mass_g"],
                "Mass (W)": result["mass_penalty_watts"],
                "Total (W)": result["total_watts"],
                "F PSI": f_psi,
                "R PSI": r_psi,
            }
        )

    st.dataframe(
        style_risk(pd.DataFrame(rows)),
        use_container_width=True,
        hide_index=True,
        height=dataframe_height_for_rows(len(rows)),
        column_config={
            "Route Score": st.column_config.NumberColumn(format="%.4f"),
            "RR (W)": st.column_config.NumberColumn(format="%.1f W"),
            "Imp (W)": st.column_config.NumberColumn(format="%+.1f W"),
            "Aero (W)": st.column_config.NumberColumn(format="%+.1f W"),
            "Tire Mass (g)": st.column_config.NumberColumn(format="%.0f g"),
            "Mass (W)": st.column_config.NumberColumn(format="%+.2f W"),
            "Total (W)": st.column_config.NumberColumn(format="%.1f W"),
            "F PSI": st.column_config.NumberColumn(format="%.1f"),
            "R PSI": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    if show_first_90:
        early_speed = float(early_speed_mph)
        early_speed_tier = speed_tier_from_avg_mph(early_speed)
        minutes = float(race_start_minutes)
        early_segments = segments_first_n_minutes(segments, early_speed, minutes)
        if not early_segments:
            st.warning("Could not compute a race-start subset for this route.")
        else:
            early_route_km = sum(s.distance_km for s in early_segments)
            early_weighted_distance = effective_weighted_distance(early_segments, early_boost)
            early_ranked_by_rr = score_tires(tires, early_segments, early_boost)
            early_elev_gain_m = total_elev_gain_m * (early_route_km / max(route_stats["distance_km"], 1e-9))
            early_ranked = rank_by_fastest_total_watts(
                early_ranked_by_rr,
                early_weighted_distance,
                early_route_km,
                early_elev_gain_m,
                early_speed,
                weight_kg,
                mass_overrides,
                segments=early_segments,
                early_boost=early_boost,
                speed_tier=early_speed_tier,
                aero_distance_fraction=aero_distance_fraction,
            )
            early_winner = early_ranked[0]
            st.markdown('<div class="tb-divider-label">Early race</div>', unsafe_allow_html=True)
            st.subheader("Best tire for race start")
            st.caption(
                f"Approximates the first {minutes:.0f} minutes as the first {km_to_mi(early_route_km):.1f} mi of the route at {early_speed:.1f} mph."
            )
            e1, e2, e3 = st.columns(3)
            e1.metric("Early best tire", early_winner["tire_name"])
            e2.metric("Early total resistance", f"{early_winner['total_watts']:.1f} W")
            e3.metric("Early aero penalty", f"{early_winner['aero_penalty_watts']:+.1f} W")

            early_rows = []
            early_cat2_mi = cat2_distance_mi(early_segments)
            early_cat3_mi = cat3_distance_mi(early_segments)
            early_above_mi = above_distance_mi(early_segments)
            for idx, result in enumerate(early_ranked[: min(top_n, 8)], start=1):
                f_psi, r_psi = estimate_pressure(result["width_mm"], weight_kg, early_speed_tier, roughness)
                risk_fallback = tire_issue_risk_for_tire(
                    result["width_mm"],
                    early_cat2_mi,
                    early_cat3_mi,
                    tire_name=result["tire_name"],
                    tire_class=result.get("tire_class", ""),
                    above_mi=early_above_mi,
                )
                risk = tire_issue_risk_label(
                    examples=risk_examples,
                    segments=early_segments,
                    tire_width_mm=float(result["width_mm"] or 0.0),
                    tire_class=str(result.get("tire_class", "")),
                    fallback_label=risk_fallback,
                )
                early_rows.append(
                    {
                        "Rank": idx,
                        "Tire": result["tire_name"],
                        "Width (mm)": f"{result['width_mm']:.1f}",
                        "Risk": risk,
                        "RR (W)": result["rr_watts"],
                        "Imp (W)": result.get("impedance_watts", 0.0),
                        "Aero (W)": result["aero_penalty_watts"],
                        "Mass (W)": result["mass_penalty_watts"],
                        "Total (W)": result["total_watts"],
                        "F PSI": f_psi,
                        "R PSI": r_psi,
                    }
                )
            st.dataframe(
                style_risk(pd.DataFrame(early_rows)),
                use_container_width=True,
                hide_index=True,
                height=dataframe_height_for_rows(len(early_rows), max_px=420),
            )

    st.markdown('<div class="tb-divider-label">Performance over course</div>', unsafe_allow_html=True)
    st.subheader("Top 5 performance vs course distance")
    st.caption(
        "Chart shows **cumulative time difference** vs the current winner (0 line) as the route progresses. "
        "This is a simplified model using segment surfaces + impedance and converts watts→time using an estimated rider aero/drivetrain baseline."
    )

    try:
        tire_by_name = {str(t.get("tire_name", "")): t for t in tires}
        top5 = ranked[: min(5, len(ranked))]
        if len(top5) >= 2 and segments:
            ordered = sorted(segments, key=lambda s: s.race_position)
            speed_mps = avg_speed_mph * 0.44704
            system_mass_kg = weight_kg + 9.0

            def delta_crr_for_segment(seg: Segment, *, width_mm: float) -> float:
                k = IMPEDANCE_K_BY_SURFACE.get(seg.surface, 0.0)
                target = IMPEDANCE_TARGET_STIFFNESS.get(seg.surface, 0.0)
                if k <= 0 or target <= 0:
                    return 0.0
                roughness_surface = {
                    "road": 0.0,
                    "cat1": 0.9,
                    "cat2": 1.8,
                    "cat3": 3.0,
                    "above": 3.6,
                }.get(seg.surface, 1.2)
                f_psi, r_psi = estimate_pressure(width_mm, weight_kg, speed_tier, roughness_surface)
                avg_psi = max(1.0, (f_psi + r_psi) / 2.0)
                stiffness = (avg_psi * max(1.0, width_mm)) / max(1e-9, weight_kg)
                if stiffness <= target:
                    return 0.0
                stiff_ratio = (stiffness / target) - 1.0
                return k * (stiff_ratio**IMPEDANCE_GAMMA)

            # Precompute per-tire constants.
            per_tire = {}
            for r in top5:
                t = tire_by_name.get(str(r["tire_name"]))
                if not t:
                    continue
                sv = interpolate_missing_surface_values(t)
                if sv is None:
                    continue
                width_mm = float(r.get("width_mm") or t.get("width_mm") or 45.0)
                aero_w = float(estimate_aero_penalty_watts(width_mm, avg_speed_mph)) * float(aero_distance_fraction)
                mass_g = estimate_tire_mass_grams(width_mm, str(r["tire_name"]), mass_overrides)
                mass_w = float(
                    estimate_tire_mass_penalty_watts(mass_g, route_stats["distance_km"], total_elev_gain_m, avg_speed_mph)
                )
                per_tire[str(r["tire_name"])] = {
                    "surface_vals": sv,
                    "width_mm": width_mm,
                    "aero_w": aero_w,
                    "mass_w": mass_w,
                }

            winner_name = str(top5[0]["tire_name"])
            if winner_name in per_tire and len(per_tire) >= 2:
                cumulative_mi = 0.0
                cumulative_delta_s = {name: 0.0 for name in per_tire.keys()}
                points = []
                for seg in ordered:
                    seg_mi = km_to_mi(seg.distance_km)
                    if seg_mi <= 0:
                        continue
                    base_time_s = (seg_mi / max(1e-9, avg_speed_mph)) * 3600.0
                    cumulative_mi += seg_mi

                    def seg_total_watts(tire_name: str) -> float:
                        meta = per_tire[tire_name]
                        sv = meta["surface_vals"]
                        surface_key = "cat3" if seg.surface == "above" else seg.surface
                        base_crr = float(sv.get(surface_key, 0.0))
                        crr = base_crr + delta_crr_for_segment(seg, width_mm=float(meta["width_mm"]))
                        rr_w = crr * system_mass_kg * 9.80665 * speed_mps
                        return rr_w + float(meta["aero_w"]) + float(meta["mass_w"])

                    winner_w = max(1e-6, seg_total_watts(winner_name))
                    # Convert Δwatts -> Δtime using a more realistic denominator than "tire watts only".
                    # At these speeds, most rider power is spent on rider/position aero + drivetrain losses.
                    rho = 1.225  # kg/m^3 (sea level)
                    cda = 0.30  # typical gravel rider aero area
                    drivetrain_w = 10.0
                    rider_aero_w = 0.5 * rho * cda * (speed_mps**3)
                    other_w = drivetrain_w + rider_aero_w
                    denom_w = max(1.0, winner_w + other_w)
                    for name in list(per_tire.keys()):
                        tw = max(1e-6, seg_total_watts(name))
                        delta_w = tw - winner_w
                        cumulative_delta_s[name] += base_time_s * (delta_w / denom_w)
                        points.append(
                            {
                                "Distance (mi)": cumulative_mi,
                                "Tire": name,
                                "Cumulative Δ (min)": cumulative_delta_s[name] / 60.0,
                            }
                        )

                dfp = pd.DataFrame(points)
                # Segment overlay: surface category bands along the x-axis.
                seg_rows = []
                cum = 0.0
                for seg in ordered:
                    seg_mi = km_to_mi(seg.distance_km)
                    if seg_mi <= 0:
                        continue
                    start = cum
                    end = cum + seg_mi
                    cum = end
                    surf = (seg.surface or "").strip().lower()
                    surf_label = {
                        "road": "Road",
                        "cat1": "Cat 1",
                        "cat2": "Cat 2",
                        "cat3": "Cat 3",
                        "above": "Above",
                    }.get(surf, surf or "Unknown")
                    seg_rows.append({"Start (mi)": start, "End (mi)": end, "Surface": surf_label})
                dfs = pd.DataFrame(seg_rows)

                seg_layer = (
                    alt.Chart(dfs)
                    .mark_rect(opacity=0.14)
                    .encode(
                        x=alt.X("Start (mi):Q", title="Course distance (mi)"),
                        x2="End (mi):Q",
                        y=alt.value(0),
                        y2=alt.value(280),
                        color=alt.Color("Surface:N", title="Segment category"),
                        tooltip=[
                            alt.Tooltip("Surface:N"),
                            alt.Tooltip("Start (mi):Q", format=".1f"),
                            alt.Tooltip("End (mi):Q", format=".1f"),
                        ],
                    )
                )

                line_layer = (
                    alt.Chart(dfp)
                    .mark_line()
                    .encode(
                        x=alt.X("Distance (mi):Q", title="Course distance (mi)"),
                        y=alt.Y("Cumulative Δ (min):Q", title="Cumulative time difference vs winner (min)"),
                        color=alt.Color("Tire:N", title="Tire"),
                        tooltip=[
                            alt.Tooltip("Distance (mi):Q", format=".1f"),
                            alt.Tooltip("Tire:N"),
                            alt.Tooltip("Cumulative Δ (min):Q", format=".2f"),
                        ],
                    )
                )

                chart = (
                    alt.layer(seg_layer, line_layer)
                    .resolve_scale(color="independent")
                    .properties(height=280)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("Not enough tire data to render the performance chart.")
        else:
            st.info("Select a route and generate recommendations to see the performance chart.")
    except Exception:
        st.info("Performance chart unavailable for this route/data.")

    render_feedback_footer(route_context_str)


if __name__ == "__main__":
    main()
