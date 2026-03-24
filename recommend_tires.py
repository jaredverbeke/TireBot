#!/usr/bin/env python3
import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


SURFACE_TO_COL = {
    "road": "Smooth Pavement",
    "cat1": "Cat 1 Gravel ",
    "cat2": "Cat 2 Gravel ",
    "cat3": "Cat 3 Gravel",
}


@dataclass
class Segment:
    name: str
    distance_km: float
    surface: str
    technicality: float
    selection_risk: float
    race_position: float


@dataclass
class TireResult:
    tire_name: str
    width_mm: Optional[float]
    total_score: float
    weighted_rr: float


def parse_float(value: str) -> Optional[float]:
    raw = value.strip()
    if raw in {"", "-", "N/A"}:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def parse_width_mm(tire_name: str) -> Optional[float]:
    match = re.search(r"x\s*([0-9]+(?:\.[0-9]+)?)\s*$", tire_name.strip())
    if not match:
        return None
    size = float(match.group(1))
    # Heuristic: values above 10 are likely mm, below 10 are inches.
    return size if size > 10 else size * 25.4


def load_tire_data(csv_path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    header_idx = -1
    for i, row in enumerate(all_rows):
        if "Tire " in row and "Smooth Pavement" in row:
            header_idx = i
            break
    if header_idx < 0:
        raise ValueError("Could not find tire CSV header row.")

    header = all_rows[header_idx]
    col_to_idx = {name: header.index(name) for name in header if name}
    if "Tire " not in col_to_idx:
        raise ValueError("Missing Tire column in CSV.")

    for row in all_rows[header_idx + 1 :]:
        if len(row) <= col_to_idx["Tire "]:
            continue
        tire_name = row[col_to_idx["Tire "]].strip()
        if not tire_name:
            continue

        parsed = {
            "tire_name": tire_name,
            "width_mm": parse_width_mm(tire_name),
        }
        valid_points = 0
        for surface, col in SURFACE_TO_COL.items():
            idx = col_to_idx.get(col)
            val = parse_float(row[idx] if idx is not None and idx < len(row) else "")
            parsed[surface] = val
            if val is not None:
                valid_points += 1

        if valid_points == 0:
            continue
        rows.append(parsed)
    return rows


def interpolate_missing_surface_values(tire: Dict[str, object]) -> Optional[Dict[str, float]]:
    values: Dict[str, Optional[float]] = {k: tire[k] for k in SURFACE_TO_COL.keys()}  # type: ignore[index]

    if values["road"] is None and values["cat1"] is not None:
        values["road"] = values["cat1"] * 0.5

    if values["cat1"] is None and values["road"] is not None and values["cat2"] is not None:
        values["cat1"] = (values["road"] + values["cat2"]) / 2.0

    if values["cat2"] is None and values["cat1"] is not None and values["cat3"] is not None:
        values["cat2"] = (values["cat1"] + values["cat3"]) / 2.0

    if values["cat3"] is None and values["cat2"] is not None:
        values["cat3"] = values["cat2"] * 1.35

    if any(values[s] is None for s in SURFACE_TO_COL.keys()):
        return None

    return {k: float(v) for k, v in values.items() if v is not None}


def load_segments(csv_path: Path) -> List[Segment]:
    def parse_or_default(value: Optional[str], default: float) -> float:
        if value is None:
            return default
        text = value.strip()
        if text == "":
            return default
        return float(text)

    segments: List[Segment] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("No segments were loaded from route CSV.")

    has_distance = "distance_km" in (rows[0].keys() if rows else {})
    has_start_end = "segment_start" in (rows[0].keys() if rows else {}) and "segment_end" in (rows[0].keys() if rows else {})

    total_distance_km = 0.0
    if has_start_end:
        # Use the farthest endpoint as route length for race_position normalization.
        total_distance_km = max(float((r.get("segment_end") or "0").strip() or "0") for r in rows)
        if total_distance_km <= 0:
            raise ValueError("Invalid segment_end values; cannot determine route length.")

    for row in rows:
        surface = (row.get("surface_type") or "").strip().lower()
        if surface not in SURFACE_TO_COL:
            raise ValueError(f"Invalid surface_type '{surface}'. Expected one of: {list(SURFACE_TO_COL.keys())}")

        if has_distance:
            distance_km = float((row.get("distance_km") or "0").strip() or "0")
            race_pos = parse_or_default(row.get("race_position"), 0.5)
        elif has_start_end:
            start_km = float((row.get("segment_start") or "0").strip() or "0")
            end_km = float((row.get("segment_end") or "0").strip() or "0")
            if end_km < start_km:
                raise ValueError(f"segment_end < segment_start for segment '{row.get('segment_name', 'unnamed')}'.")
            distance_km = end_km - start_km
            # Midpoint of segment relative to total route length.
            race_pos = ((start_km + end_km) / 2.0) / total_distance_km
        else:
            raise ValueError(
                "Route CSV must include either distance_km or segment_start and segment_end columns."
            )

        selection_risk = parse_or_default(row.get("selection_risk"), 1.0)
        technicality = parse_or_default(row.get("technicality"), 1.0)
        # Keep 0 values from effectively deleting segment influence.
        if selection_risk <= 0:
            selection_risk = 1.0
        if technicality <= 0:
            technicality = 1.0

        segments.append(
            Segment(
                name=(row.get("segment_name") or "").strip() or "unnamed",
                distance_km=distance_km,
                surface=surface,
                technicality=technicality,
                selection_risk=selection_risk,
                race_position=max(0.0, min(1.0, race_pos)),
            )
        )

    return segments


def phase_weight(race_position: float, early_boost: float) -> float:
    p = max(0.0, min(1.0, race_position))
    if p <= 0.25:
        return early_boost
    if p <= 0.70:
        return 1.15
    return 1.0


def score_tires(
    tires: List[Dict[str, object]],
    segments: List[Segment],
    early_boost: float,
) -> List[TireResult]:
    results: List[TireResult] = []
    for tire in tires:
        surface_vals = interpolate_missing_surface_values(tire)
        if surface_vals is None:
            continue

        rr_total = 0.0
        for seg in segments:
            weight = phase_weight(seg.race_position, early_boost) * seg.technicality * seg.selection_risk
            rr_total += surface_vals[seg.surface] * seg.distance_km * weight

        total = rr_total
        results.append(
            TireResult(
                tire_name=str(tire["tire_name"]),
                width_mm=tire.get("width_mm"),  # type: ignore[arg-type]
                total_score=total,
                weighted_rr=rr_total,
            )
        )
    return sorted(results, key=lambda x: x.total_score)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend gravel race tires using CRR data and route segment weighting."
    )
    parser.add_argument(
        "--tires-csv",
        default="Gravel and MTB Tire Testing by John Karrasch  - Overall CRR.csv",
        help="Path to tire CRR CSV",
    )
    parser.add_argument(
        "--route-csv",
        default="Routes/example_route_segments.csv",
        help="Path to route segments CSV",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="Number of top recommendations to print",
    )
    parser.add_argument(
        "--early-boost",
        type=float,
        default=1.8,
        help="Weight multiplier for first quarter of race",
    )
    args = parser.parse_args()

    tires_path = Path(args.tires_csv)
    route_path = Path(args.route_csv)
    tires = load_tire_data(tires_path)
    segments = load_segments(route_path)
    scored = score_tires(tires, segments, args.early_boost)

    if not scored:
        raise RuntimeError("No tires could be scored from the provided data.")

    print("Top tire recommendations")
    print("=" * 80)
    print(f"{'Rank':<5} {'Tire':<45} {'Width(mm)':>10} {'Total':>10} {'RR Cost':>10}")
    print("-" * 80)
    for i, r in enumerate(scored[: args.top_n], start=1):
        width_text = f"{r.width_mm:.1f}" if r.width_mm is not None else "n/a"
        print(
            f"{i:<5} {r.tire_name[:45]:<45} {width_text:>10} "
            f"{r.total_score:>10.4f} {r.weighted_rr:>10.4f}"
        )

    winner = scored[0]
    print("\nRecommended now:")
    print(
        f"- {winner.tire_name} (score={winner.total_score:.4f}), with early-race weighting "
        f"set to {args.early_boost:.2f}"
    )


if __name__ == "__main__":
    main()
