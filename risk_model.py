from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from recommend_tires import Segment


@dataclass(frozen=True)
class RiskFeatures:
    width_mm: float
    tire_class: str
    total_mi: float
    road_mi: float
    cat1_mi: float
    cat2_mi: float
    cat3_mi: float
    above_mi: float


@dataclass(frozen=True)
class LabeledExample:
    features: RiskFeatures
    label: str


def segments_surface_miles(segments: List[Segment]) -> Dict[str, float]:
    out = {"road": 0.0, "cat1": 0.0, "cat2": 0.0, "cat3": 0.0, "above": 0.0}
    for s in segments:
        if s.surface in out:
            out[s.surface] += s.distance_km * 0.621371
    return out


def build_features_for_route_and_tire(
    *,
    segments: List[Segment],
    width_mm: float,
    tire_class: str,
) -> RiskFeatures:
    miles = segments_surface_miles(segments)
    total = sum(miles.values()) or 1e-9
    return RiskFeatures(
        width_mm=float(width_mm or 0.0),
        tire_class=(tire_class or "unknown").strip().lower(),
        total_mi=float(total),
        road_mi=float(miles["road"]),
        cat1_mi=float(miles["cat1"]),
        cat2_mi=float(miles["cat2"]),
        cat3_mi=float(miles["cat3"]),
        above_mi=float(miles["above"]),
    )


def load_labeled_examples(labels_csv_path: Path) -> List[LabeledExample]:
    if not labels_csv_path.exists():
        return []
    out: List[LabeledExample] = []
    with labels_csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                width_mm = float((row.get("width_mm") or "0").strip() or "0")
                tire_class = (row.get("tire_class") or "unknown").strip().lower()
                label = (row.get("label") or "").strip()
                total_mi = float((row.get("total_mi") or "0").strip() or "0")
                if not label or total_mi <= 0:
                    continue
                feats = RiskFeatures(
                    width_mm=width_mm,
                    tire_class=tire_class,
                    total_mi=total_mi,
                    road_mi=float((row.get("road_mi") or "0").strip() or "0"),
                    cat1_mi=float((row.get("cat1_mi") or "0").strip() or "0"),
                    cat2_mi=float((row.get("cat2_mi") or "0").strip() or "0"),
                    cat3_mi=float((row.get("cat3_mi") or "0").strip() or "0"),
                    above_mi=float((row.get("above_mi") or "0").strip() or "0"),
                )
                out.append(LabeledExample(features=feats, label=label))
            except Exception:
                continue
    return out


def _pct(x: float, total: float) -> float:
    return float(x) / max(1e-9, float(total))


def distance(a: RiskFeatures, b: RiskFeatures) -> float:
    # Normalize key dimensions to comparable scales.
    # Width: ~10mm ~= 1.0 distance unit.
    dw = abs(a.width_mm - b.width_mm) / 10.0

    # Surface mix (percent of course).
    ap = (_pct(a.cat2_mi, a.total_mi), _pct(a.cat3_mi, a.total_mi), _pct(a.above_mi, a.total_mi))
    bp = (_pct(b.cat2_mi, b.total_mi), _pct(b.cat3_mi, b.total_mi), _pct(b.above_mi, b.total_mi))
    dmix = abs(ap[0] - bp[0]) + abs(ap[1] - bp[1]) + abs(ap[2] - bp[2])

    # Absolute rough miles (so 8mi Above matters more than 0.5mi Above).
    drough = (abs(a.cat3_mi - b.cat3_mi) + 1.2 * abs(a.above_mi - b.above_mi)) / 10.0

    # Tire type mismatch penalty.
    dclass = 0.0 if (a.tire_class == b.tire_class) else 0.75

    return (0.9 * dw) + (1.8 * dmix) + (1.2 * drough) + dclass


def predict_knn(
    *,
    query: RiskFeatures,
    examples: List[LabeledExample],
    k: int = 5,
) -> Optional[str]:
    if not examples:
        return None
    scored = [(distance(query, ex.features), ex.label) for ex in examples]
    scored.sort(key=lambda t: t[0])
    top = scored[: max(1, min(int(k), len(scored)))]

    # Inverse-distance vote.
    votes: Dict[str, float] = {}
    for d, lab in top:
        w = 1.0 / max(1e-6, d)
        votes[lab] = votes.get(lab, 0.0) + w
    return max(votes.items(), key=lambda kv: kv[1])[0] if votes else None

