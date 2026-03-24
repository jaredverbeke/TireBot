#!/usr/bin/env python3
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

from recommend_tires import Segment, load_segments, load_tires_with_optional_brr, score_tires


ROOT = Path(__file__).parent
ROUTES_DIR = ROOT / "Routes"
TIRES_CSV = ROOT / "Gravel and MTB Tire Testing by John Karrasch  - Overall CRR.csv"
BRR_CRR_CSV = ROOT / "data" / "brr_crr.csv"
PRESSURE_BASELINE_CSV = ROOT / "data" / "wolf_tooth_baseline.csv"
TIRE_MASS_CSV = ROOT / "data" / "tire_mass_overrides.csv"
WHITEPAPER_PATH = ROOT / "docs" / "WHITEPAPER.md"
WHITEPAPER_PDF_PATH = ROOT / "docs" / "WHITEPAPER.pdf"
WHITEPAPER_URL = "https://github.com/jaredverbeke/TireBot/blob/main/docs/WHITEPAPER.pdf"

# Segment CSVs list distances in miles; summarize_route also reports miles.
KM_TO_MI = 0.621371
M_TO_FT = 3.28084
EARTH_RADIUS_KM = 6371.0


def km_to_mi(km: float) -> float:
    return km * KM_TO_MI


SPEED_OPTIONS = {
    "Pro (23+ mph avg)": "pro",
    "Amateur race (18-23 mph avg)": "amateur",
    "Ride (under 18 mph avg)": "ride",
}


def inject_styles() -> None:
    st.markdown(
        """
<style>
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.5rem;
        max-width: 1100px;
    }
    .tb-hero {
        background: linear-gradient(120deg, #1f2937 0%, #3f2f1d 52%, #8b5e34 100%);
        border-radius: 16px;
        padding: 22px 24px;
        color: #f3f4f6;
        margin-bottom: 1rem;
        border: 1px solid rgba(217, 119, 6, 0.35);
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.25);
    }
    .tb-hero h1 {
        font-size: 1.65rem;
        line-height: 1.2;
        margin: 0;
    }
    .tb-hero p {
        margin: 0.45rem 0 0;
        color: #e2e8f0;
        font-size: 0.98rem;
    }
    .tb-chip {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        background: rgba(217, 119, 6, 0.16);
        border: 1px solid rgba(217, 119, 6, 0.5);
        font-size: 0.8rem;
        margin-right: 0.4rem;
    }
    .tb-card {
        border: 1px solid rgba(217, 119, 6, 0.24);
        border-radius: 14px;
        background: rgba(17, 24, 39, 0.5);
        padding: 0.85rem 1rem;
        margin-bottom: 0.9rem;
    }
    .tb-muted {
        color: #94a3b8;
        font-size: 0.9rem;
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

    # Fallback: if no folder-based events are found, support top-level segment CSVs.
    if not events:
        for path in routes_dir.glob("*.csv"):
            if "segment" in path.name.lower():
                events[path.stem] = path
    return dict(sorted(events.items(), key=lambda x: x[0].lower()))


def find_event_gpx(route_csv: Path) -> Optional[Path]:
    parent = route_csv.parent
    gpx_files = sorted(parent.glob("*.gpx"), key=lambda p: p.name.lower())
    if not gpx_files:
        return None
    return gpx_files[0]


def route_roughness_score(segments: List[Segment]) -> float:
    if not segments:
        return 0.0
    surface_score = {"road": 0.0, "cat1": 0.9, "cat2": 1.8, "cat3": 3.0}
    total_distance = sum(s.distance_km for s in segments)
    if total_distance <= 0:
        return 0.0
    weighted = sum(surface_score[s.surface] * s.distance_km for s in segments)
    return weighted / total_distance


def mph_to_kph(speed_mph: float) -> float:
    return speed_mph * 1.60934


def estimate_pressure(width_mm: float, weight_kg: float, speed_tier: str, roughness: float) -> Tuple[float, float]:
    speed_adj = {"pro": 2.0, "amateur": 0.0, "ride": -2.0}[speed_tier]
    weight_adj = (weight_kg - 75.0) * 0.12
    width_adj = -0.35 * (width_mm - 40.0)
    roughness_drop = roughness * 1.2

    base_center = 36.0 + speed_adj + weight_adj + width_adj - roughness_drop
    front = base_center - 1.5
    rear = base_center + 1.0

    # Practical clamp range for tubeless gravel/xc setups.
    front = max(17.0, min(42.0, front))
    rear = max(19.0, min(45.0, rear))
    return round(front, 1), round(rear, 1)


def load_pressure_baseline(path: Path) -> List[Dict[str, float]]:
    if not path.exists():
        return []
    rows: List[Dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            try:
                rows.append(
                    {
                        "terrain_class": float((row.get("terrain_class") or "").strip()),
                        "weight_kg": float((row.get("weight_kg") or "").strip()),
                        "width_mm": float((row.get("width_mm") or "").strip()),
                        "rear_psi": float((row.get("rear_psi") or "").strip()),
                        "front_psi": float((row.get("front_psi") or "").strip()),
                    }
                )
            except ValueError:
                continue
    return rows


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


def map_roughness_to_terrain_class(roughness: float) -> int:
    # Coarse mapping into Wolf Tooth-style terrain classes.
    if roughness <= 0.45:
        return 4  # pavement-biased
    if roughness <= 1.6:
        return 2  # hardpack / mixed
    return 3  # rough / rocks


def estimate_pressure_from_baseline(
    width_mm: float,
    weight_kg: float,
    terrain_class: int,
    baseline_rows: List[Dict[str, float]],
) -> Tuple[float, float]:
    # Weighted nearest-neighbor interpolation over (terrain, weight, width).
    candidates = [r for r in baseline_rows if int(round(r["terrain_class"])) == terrain_class]
    if not candidates:
        candidates = baseline_rows
    if not candidates:
        raise ValueError("No baseline rows available.")

    scored = []
    for r in candidates:
        d_weight = abs(r["weight_kg"] - weight_kg) / 10.0
        d_width = abs(r["width_mm"] - width_mm) / 5.0
        d_terrain = abs(r["terrain_class"] - terrain_class) * 1.5
        dist = max(0.05, d_weight + d_width + d_terrain)
        scored.append((dist, r))
    scored.sort(key=lambda x: x[0])
    neighbors = scored[: min(4, len(scored))]

    total_w = 0.0
    front = 0.0
    rear = 0.0
    for dist, row in neighbors:
        w = 1.0 / dist
        total_w += w
        front += w * row["front_psi"]
        rear += w * row["rear_psi"]
    front /= total_w
    rear /= total_w
    return round(front, 1), round(rear, 1)


def avg_speed_mph_from_tier(speed_tier: str) -> float:
    return {"pro": 24.0, "amateur": 20.0, "ride": 16.0}[speed_tier]


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


def estimate_rr_watts(total_score: float, weighted_distance: float, speed_mph: float, weight_kg: float) -> float:
    if weighted_distance <= 0:
        return 0.0
    effective_crr = total_score / weighted_distance
    system_mass_kg = weight_kg + 9.0  # rider + bike/kit estimate
    speed_mps = speed_mph * 0.44704
    watts = effective_crr * system_mass_kg * 9.80665 * speed_mps
    return round(watts, 1)


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

    def segment_km(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        lat1, lon1 = map(math.radians, p1)
        lat2, lon2 = map(math.radians, p2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, h)))

    total_km = sum(segment_km(pts[i - 1], pts[i]) for i in range(1, len(pts)))
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
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for r in ranked_by_rr:
        width_mm = r.width_mm if r.width_mm is not None else 45.0
        rr_watts = estimate_rr_watts(r.total_score, weighted_distance, speed_mph, weight_kg)
        aero_penalty_watts = estimate_aero_penalty_watts(width_mm, speed_mph)
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
                "aero_penalty_watts": aero_penalty_watts,
                "tire_mass_g": round(tire_mass_g, 0),
                "mass_penalty_watts": mass_penalty_watts,
                "total_watts": total_watts,
            }
        )
    return sorted(rows, key=lambda x: x["total_watts"])


def summarize_route(path: Path) -> Dict[str, float]:
    # Small helper to show rough course composition to the rider.
    segments = load_segments(path)
    totals = {"road": 0.0, "cat1": 0.0, "cat2": 0.0, "cat3": 0.0}
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
    }


def main() -> None:
    st.set_page_config(page_title="TireBot", page_icon="🚴", layout="wide")
    inject_styles()
    st.markdown(
        """
<div class="tb-hero">
  <h1>TireBot Race Setup Advisor</h1>
  <p>Dial your setup for gravel race speed with route-aware tire and pressure recommendations.</p>
  <div style="margin-top: 0.7rem;">
    <span class="tb-chip">Route-Aware</span>
    <span class="tb-chip">CRR-Based</span>
    <span class="tb-chip">Race-Day Pressure</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="tb-card">', unsafe_allow_html=True)
    st.markdown("### Methodology")
    st.markdown(
        f"[Read the TireBot whitepaper]({WHITEPAPER_URL}) to understand data sources, assumptions, and calculations."
    )
    if WHITEPAPER_PDF_PATH.exists():
        whitepaper_pdf = WHITEPAPER_PDF_PATH.read_bytes()
        st.download_button(
            "Download Whitepaper (.pdf)",
            data=whitepaper_pdf,
            file_name="TireBot_Whitepaper.pdf",
            mime="application/pdf",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    events = discover_events(ROUTES_DIR)
    if not events:
        st.error("No segment CSV files found in Routes/. Add a route segments CSV and reload.")
        return

    st.markdown('<div class="tb-card">', unsafe_allow_html=True)
    st.subheader("Ride Inputs")
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

        with st.expander("Advanced options", expanded=False):
            speed_label = st.radio("Speed tier", list(SPEED_OPTIONS.keys()), index=1)
            default_speed_mph = avg_speed_mph_from_tier(SPEED_OPTIONS[speed_label])
            avg_speed_mph = st.number_input(
                "Average speed (mph)",
                min_value=10.0,
                max_value=35.0,
                value=float(default_speed_mph),
                step=0.1,
                help="Used directly for aero penalty and rolling resistance power calculations.",
            )
            early_boost = st.slider("Early-race weighting", min_value=1.0, max_value=3.0, value=1.8, step=0.1)
            top_n = st.slider("Top tire options", min_value=3, max_value=20, value=8, step=1)

        submitted = st.form_submit_button("Generate Recommendation", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not submitted:
        st.info("Select route and weight, then click Generate Recommendation.")
        return

    if event_label == "Pick a Route/Event":
        st.warning("Please select a route/event before generating recommendations.")
        return

    speed_tier = SPEED_OPTIONS[speed_label]

    route_csv = events[event_label]
    tires = load_tires_with_optional_brr(TIRES_CSV, BRR_CRR_CSV)
    segments = load_segments(route_csv)
    baseline_rows = load_pressure_baseline(PRESSURE_BASELINE_CSV)
    mass_overrides = load_tire_mass_overrides(TIRE_MASS_CSV)
    route_stats = summarize_route(route_csv)
    route_gpx = find_event_gpx(route_csv)
    total_elev_gain_m = gpx_total_elevation_gain_m(route_gpx) if route_gpx else 0.0
    gpx_track_mi = gpx_track_length_mi(route_gpx) if route_gpx else 0.0
    elev_gain_ft = total_elev_gain_m * M_TO_FT
    ranked_by_rr = score_tires(tires, segments, early_boost)
    if not ranked_by_rr:
        st.error("No tires could be scored. Check CSV data completeness.")
        return

    avg_speed_kph = mph_to_kph(avg_speed_mph)
    weighted_distance = effective_weighted_distance(segments, early_boost)
    ranked = rank_by_fastest_total_watts(
        ranked_by_rr,
        weighted_distance,
        route_stats["distance_km"],
        total_elev_gain_m,
        avg_speed_mph,
        weight_kg,
        mass_overrides,
    )
    winner = ranked[0]
    winner_width = winner["width_mm"]
    roughness = route_roughness_score(segments)
    terrain_class = map_roughness_to_terrain_class(roughness)
    if baseline_rows:
        front_psi, rear_psi = estimate_pressure_from_baseline(
            winner_width, weight_kg, terrain_class, baseline_rows
        )
        pressure_source = "Wolf Tooth baseline lookup/interpolation"
    else:
        front_psi, rear_psi = estimate_pressure(winner_width, weight_kg, speed_tier, roughness)
        pressure_source = "Heuristic fallback (add baseline CSV rows to switch)"
    winner_rr_watts = winner["rr_watts"]
    winner_aero_penalty = winner["aero_penalty_watts"]
    winner_total_watts = winner["total_watts"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Route Distance", f"{route_stats['distance_mi']:.1f} mi")
    m2.metric("Speed Assumption", f"{avg_speed_mph:.1f} mph")
    m3.metric("Fastest Tire", winner["tire_name"])
    m4.metric("Pressure (F / R)", f"{front_psi:.1f} / {rear_psi:.1f} psi")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="tb-card">', unsafe_allow_html=True)
        st.subheader("Recommendation")
        st.markdown(
            f"**Tire choice:** `{winner['tire_name']}`\n\n"
            f"**Pressure:** `{front_psi:.1f} psi front / {rear_psi:.1f} psi rear`\n\n"
            f"**Rolling resistance:** `{winner_rr_watts:.1f} W`\n\n"
            f"**Aero width penalty:** `{winner_aero_penalty:+.1f} W`\n\n"
            f"**Tire mass penalty:** `{winner['mass_penalty_watts']:+.2f} W` ({winner['tire_mass_g']:.0f} g each)\n\n"
            f"**Total resistance power:** `{winner_total_watts:.1f} W`"
        )
        st.markdown(
            f'<p class="tb-muted">Source: {pressure_source}. Route length from segments: {route_stats["distance_mi"]:.1f} mi. '
            + (
                f'GPX track (lat/lon): {gpx_track_mi:.1f} mi; elevation gain: {elev_gain_ft:,.0f} ft ({total_elev_gain_m:.0f} m, standard GPX &lt;ele&gt;). '
                if route_gpx
                else ""
            )
            + f"Surface mix, early-race weighting {early_boost:.1f}, speed tier {speed_label}, average speed {avg_speed_mph:.1f} mph, rider weight {weight_kg:.1f} kg. Segment CSV uses miles along course.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="tb-card">', unsafe_allow_html=True)
        st.subheader("Route Composition")
        st.progress(min(max(route_stats["road_pct"] / 100.0, 0.0), 1.0), text=f"Road: {route_stats['road_pct']:.1f}%")
        st.progress(min(max(route_stats["cat1_pct"] / 100.0, 0.0), 1.0), text=f"Cat 1: {route_stats['cat1_pct']:.1f}%")
        st.progress(min(max(route_stats["cat2_pct"] / 100.0, 0.0), 1.0), text=f"Cat 2: {route_stats['cat2_pct']:.1f}%")
        st.progress(min(max(route_stats["cat3_pct"] / 100.0, 0.0), 1.0), text=f"Cat 3: {route_stats['cat3_pct']:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)

    st.subheader("Top tire rankings")
    rows = []
    for idx, result in enumerate(ranked[:top_n], start=1):
        width_text = f"{result['width_mm']:.1f}"
        tire_width = result["width_mm"]
        if baseline_rows:
            f_psi, r_psi = estimate_pressure_from_baseline(
                tire_width, weight_kg, terrain_class, baseline_rows
            )
        else:
            f_psi, r_psi = estimate_pressure(tire_width, weight_kg, speed_tier, roughness)
        rows.append(
            {
                "Rank": idx,
                "Tire": result["tire_name"],
                "Width (mm)": width_text,
                "Route Score": result["score"],
                "Rolling Resistance (W)": result["rr_watts"],
                "Aero Penalty (W)": result["aero_penalty_watts"],
                "Tire Mass (g)": result["tire_mass_g"],
                "Mass Penalty (W)": result["mass_penalty_watts"],
                "Total Resistance (W)": result["total_watts"],
                "Front PSI": f_psi,
                "Rear PSI": r_psi,
            }
        )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Route Score": st.column_config.NumberColumn(format="%.4f"),
            "Rolling Resistance (W)": st.column_config.NumberColumn(format="%.1f W"),
            "Aero Penalty (W)": st.column_config.NumberColumn(format="%+.1f W"),
            "Tire Mass (g)": st.column_config.NumberColumn(format="%.0f g"),
            "Mass Penalty (W)": st.column_config.NumberColumn(format="%+.2f W"),
            "Total Resistance (W)": st.column_config.NumberColumn(format="%.1f W"),
            "Front PSI": st.column_config.NumberColumn(format="%.1f"),
            "Rear PSI": st.column_config.NumberColumn(format="%.1f"),
        },
    )


if __name__ == "__main__":
    main()
