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
    "above": "Above Category",
}

# Route segment CSVs use miles for segment_start / segment_end / distance columns.
MI_TO_KM = 1.60934


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


def normalize_tire_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def _strip_key_row(row: Dict[str, object]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in row.items():
        if k is None:
            continue
        key = str(k).strip()
        out[key] = "" if v is None else str(v).strip()
    return out


def _first_float(row: Dict[str, str], *header_names: str) -> Optional[float]:
    for h in header_names:
        if h in row:
            v = parse_float(row[h])
            if v is not None:
                return v
    return None


def load_brr_crr_csv(csv_path: Path) -> List[Dict[str, object]]:
    """Optional overrides from Bicycle Rolling Resistance (manual Pro View export / copy).

    Supported headers (strip-safe):

    - Tire name: ``tire_name`` or ``Tire``
    - Road / smooth: ``road_crr``, ``smooth_pavement_crr``, ``Smooth Pavement``; if those are
      empty, ``BRR Drum`` is used as a fallback for the road surface.
    - Gravel: ``cat1_crr`` / ``Cat 1 Gravel``, same pattern for cat2 and cat3.

    Values must be CRR coefficients (same units as the Karrasch CSV), not watts.
    """
    if not csv_path.exists():
        return []
    rows_out: List[Dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = _strip_key_row(row)
            name = (r.get("tire_name") or r.get("Tire") or "").strip()
            if not name or name.startswith("#"):
                continue
            if name.lower() == "tire":
                continue
            rec: Dict[str, object] = {
                "tire_name": name,
                "width_mm": parse_width_mm(name),
            }
            rec["road"] = _first_float(
                r,
                "road_crr",
                "smooth_pavement_crr",
                "Smooth Pavement",
                "BRR Drum",
            )
            rec["cat1"] = _first_float(r, "cat1_crr", "Cat 1 Gravel")
            rec["cat2"] = _first_float(r, "cat2_crr", "Cat 2 Gravel")
            rec["cat3"] = _first_float(r, "cat3_crr", "Cat 3 Gravel")
            if sum(1 for s in SURFACE_TO_COL if rec[s] is not None) == 0:
                continue
            rows_out.append(rec)
    return rows_out


def merge_brr_crr_into_tires(
    base_tires: List[Dict[str, object]], brr_rows: List[Dict[str, object]]
) -> List[Dict[str, object]]:
    """Apply BRR CRR values on top of base dataset; add tires that exist only in BRR."""
    out: List[Dict[str, object]] = []
    for t in base_tires:
        out.append(dict(t))
    by_norm: Dict[str, int] = {normalize_tire_name(str(t["tire_name"])): i for i, t in enumerate(out)}

    for br in brr_rows:
        key = normalize_tire_name(str(br["tire_name"]))
        if key in by_norm:
            idx = by_norm[key]
            for surf in SURFACE_TO_COL.keys():
                if br.get(surf) is not None:
                    out[idx][surf] = br[surf]
        else:
            new_tire: Dict[str, object] = {
                "tire_name": br["tire_name"],
                "width_mm": br.get("width_mm"),
            }
            if new_tire["width_mm"] is None:
                new_tire["width_mm"] = parse_width_mm(str(br["tire_name"]))
            for surf in SURFACE_TO_COL.keys():
                new_tire[surf] = br.get(surf)
            out.append(new_tire)
            by_norm[key] = len(out) - 1
    return out


def load_tires_with_optional_brr(base_csv: Path, brr_csv: Optional[Path]) -> List[Dict[str, object]]:
    tires = load_tire_data(base_csv)
    if brr_csv is not None:
        brr_rows = load_brr_crr_csv(brr_csv)
        if brr_rows:
            tires = merge_brr_crr_into_tires(tires, brr_rows)
    return tires


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

    row_keys = set(rows[0].keys() if rows else [])
    has_distance_mi = "distance_mi" in row_keys
    has_distance_km_col = "distance_km" in row_keys
    has_distance = has_distance_mi or has_distance_km_col
    has_start_end = "segment_start" in row_keys and "segment_end" in row_keys

    total_route_mi = 0.0
    if has_start_end:
        # segment_end is in miles; used only for race_position ratios.
        total_route_mi = max(float((r.get("segment_end") or "0").strip() or "0") for r in rows)
        if total_route_mi <= 0:
            raise ValueError("Invalid segment_end values; cannot determine route length.")

    for row in rows:
        surface = (row.get("surface_type") or "").strip().lower()
        if surface not in SURFACE_TO_COL:
            raise ValueError(f"Invalid surface_type '{surface}'. Expected one of: {list(SURFACE_TO_COL.keys())}")

        if has_distance:
            # Per-row distance in miles (prefer distance_mi; distance_km column name is legacy miles).
            if has_distance_mi:
                dist_mi = float((row.get("distance_mi") or "0").strip() or "0")
            else:
                dist_mi = float((row.get("distance_km") or "0").strip() or "0")
            distance_km = dist_mi * MI_TO_KM
            race_pos = parse_or_default(row.get("race_position"), 0.5)
        elif has_start_end:
            start_mi = float((row.get("segment_start") or "0").strip() or "0")
            end_mi = float((row.get("segment_end") or "0").strip() or "0")
            if end_mi < start_mi:
                raise ValueError(f"segment_end < segment_start for segment '{row.get('segment_name', 'unnamed')}'.")
            distance_km = (end_mi - start_mi) * MI_TO_KM
            race_pos = ((start_mi + end_mi) / 2.0) / total_route_mi
        else:
            raise ValueError(
                "Route CSV must include either distance_mi (or legacy distance_km in miles) or segment_start and segment_end in miles."
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
        "--brr-csv",
        default="data/brr_crr.csv",
        help="Optional BRR Pro View CRR overrides (set to non-existent path to skip)",
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
    brr_path = Path(args.brr_csv)
    tires = load_tires_with_optional_brr(tires_path, brr_path)
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
