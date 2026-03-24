#!/usr/bin/env python3
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st

from recommend_tires import Segment, load_segments, load_tire_data, score_tires


ROOT = Path(__file__).parent
ROUTES_DIR = ROOT / "Routes"
TIRES_CSV = ROOT / "Gravel and MTB Tire Testing by John Karrasch  - Overall CRR.csv"
PRESSURE_BASELINE_CSV = ROOT / "data" / "wolf_tooth_baseline.csv"

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


def summarize_route(path: Path) -> Dict[str, float]:
    # Small helper to show rough course composition to the rider.
    segments = load_segments(path)
    totals = {"road": 0.0, "cat1": 0.0, "cat2": 0.0, "cat3": 0.0}
    for seg in segments:
        totals[seg.surface] += seg.distance_km
    total_dist = sum(totals.values()) or 1.0
    return {
        "distance_km": total_dist,
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
    tires = load_tire_data(TIRES_CSV)
    segments = load_segments(route_csv)
    baseline_rows = load_pressure_baseline(PRESSURE_BASELINE_CSV)
    ranked = score_tires(tires, segments, early_boost)
    if not ranked:
        st.error("No tires could be scored. Check CSV data completeness.")
        return

    winner = ranked[0]
    winner_width = winner.width_mm if winner.width_mm is not None else 45.0
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
    avg_speed_mph = avg_speed_mph_from_tier(speed_tier)
    avg_speed_kph = mph_to_kph(avg_speed_mph)
    weighted_distance = effective_weighted_distance(segments, early_boost)
    winner_rr_watts = estimate_rr_watts(winner.total_score, weighted_distance, avg_speed_mph, weight_kg)

    route_stats = summarize_route(route_csv)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Route Distance", f"{route_stats['distance_km']:.1f} km")
    m2.metric("Speed Assumption", f"{avg_speed_mph:.1f} mph")
    m3.metric("Best Tire", winner.tire_name)
    m4.metric("Pressure (F / R)", f"{front_psi:.1f} / {rear_psi:.1f} psi")

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="tb-card">', unsafe_allow_html=True)
        st.subheader("Recommendation")
        st.markdown(
            f"**Tire choice:** `{winner.tire_name}`\n\n"
            f"**Pressure:** `{front_psi:.1f} psi front / {rear_psi:.1f} psi rear`\n\n"
            f"**Rolling resistance:** `{winner_rr_watts:.1f} W`"
        )
        st.markdown(
            f'<p class="tb-muted">Source: {pressure_source}. Model inputs include route surface mix, early-race weighting {early_boost:.1f}, speed tier {speed_label}, and rider weight {weight_kg:.1f} kg.</p>',
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
        width_text = f"{result.width_mm:.1f}" if result.width_mm is not None else "n/a"
        tire_width = result.width_mm if result.width_mm is not None else 45.0
        if baseline_rows:
            f_psi, r_psi = estimate_pressure_from_baseline(
                tire_width, weight_kg, terrain_class, baseline_rows
            )
        else:
            f_psi, r_psi = estimate_pressure(tire_width, weight_kg, speed_tier, roughness)
        rows.append(
            {
                "Rank": idx,
                "Tire": result.tire_name,
                "Width (mm)": width_text,
                "Score": round(result.total_score, 4),
                "Rolling Resistance (W)": estimate_rr_watts(result.total_score, weighted_distance, avg_speed_mph, weight_kg),
                "Front PSI": f_psi,
                "Rear PSI": r_psi,
            }
        )

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.NumberColumn(format="%.4f"),
            "Rolling Resistance (W)": st.column_config.NumberColumn(format="%.1f W"),
            "Front PSI": st.column_config.NumberColumn(format="%.1f"),
            "Rear PSI": st.column_config.NumberColumn(format="%.1f"),
        },
    )


if __name__ == "__main__":
    main()
