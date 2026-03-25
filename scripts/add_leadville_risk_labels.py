#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommend_tires import load_segments, load_tires_with_optional_brr  # noqa: E402


TIRES_CSV = ROOT / "Gravel and MTB Tire Testing by John Karrasch  - Overall CRR.csv"
BRR_CRR_CSV = ROOT / "data" / "brr_crr.csv"
ROUTE_CSV = ROOT / "Routes" / "Leadville 100" / "Leadville 100 segments.csv"
OUT = ROOT / "data" / "risk_labels.csv"

MTB_MIN_WIDTH_MM = 2.2 * 25.4

LABELS = {
    "Conti Dubnital Rapid Race 29 x 2.4": "Low",
    "Maxxis Aspen 29 x 2.4": "Low",
    "Pirelli Scorpion XC RC Team 29 x 2.4": "Low",
    "Schwalbe Rick XC Pro Speed 29 x 2.4": "Low",
    "VIttoria Barzo XC Race 29 x 2.4": "Low",
    "Vittoria Mezcal XC Race 29 x 2.4": "Low",
    "Vittoria Peyote XC Race 29 x 2.4": "Low",
    "Vittoria Terreno XC Race 29 x 2.4": "Low",
    "Schwalbe G-One Speed PRO 29 x 2.35": "Medium",
    "Specialized Air Trak Flex Lite 29 x 2.35": "Medium",
    "Maxxis Aspen ST 170 29 x 2.25": "Medium",
    "Schwalbe Rick XC Pro Speed 29 x 2.25": "Medium",
    "Schwalbe Thunder Burt SG 29 x 2.25": "Medium",
    "Scwhalbe Thunder Burt SG 29 x 2.25": "Medium",
    "Vittoria Peyote XC Race 29 x 2.25": "Medium",
    "Conti Dubnital Rapid Race 29 x 2.2": "High",
    "Conti Race King Black Chili 29 x 2.2": "High",
    "Kenda Rush Pro SCT 29 x 2.2": "High",
}


def _sum_mi(segments, surface: str) -> float:
    km = sum(s.distance_km for s in segments if s.surface == surface)
    return km * 0.621371


def main() -> None:
    tires = load_tires_with_optional_brr(TIRES_CSV, BRR_CRR_CSV)
    segs = load_segments(ROUTE_CSV)

    road_mi = _sum_mi(segs, "road")
    cat1_mi = _sum_mi(segs, "cat1")
    cat2_mi = _sum_mi(segs, "cat2")
    cat3_mi = _sum_mi(segs, "cat3")
    above_mi = _sum_mi(segs, "above")
    total_mi = (sum(s.distance_km for s in segs) * 0.621371) if segs else 0.0

    tire_by_name = {str(t.get("tire_name", "")).strip(): t for t in tires}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    exists = OUT.exists()
    with OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "route",
                "event_label",
                "tire_name",
                "width_mm",
                "tire_class",
                "road_mi",
                "cat1_mi",
                "cat2_mi",
                "cat3_mi",
                "above_mi",
                "total_mi",
                "label",
            ],
        )
        if not exists:
            w.writeheader()

        wrote = 0
        missing = []
        for tire_name, label in LABELS.items():
            t = tire_by_name.get(tire_name)
            if not t:
                missing.append(tire_name)
                continue
            width_mm = float(t.get("width_mm") or 0.0)
            tire_class = str(t.get("tire_class") or "").strip().lower() or "unknown"

            # Keep training data consistent with app filters for this route.
            if width_mm < MTB_MIN_WIDTH_MM:
                continue
            if "corsa" in tire_name.lower():
                continue

            w.writerow(
                {
                    "route": str(ROUTE_CSV.relative_to(ROOT)),
                    "event_label": "Leadville 100",
                    "tire_name": tire_name,
                    "width_mm": f"{width_mm:.1f}",
                    "tire_class": tire_class,
                    "road_mi": f"{road_mi:.3f}",
                    "cat1_mi": f"{cat1_mi:.3f}",
                    "cat2_mi": f"{cat2_mi:.3f}",
                    "cat3_mi": f"{cat3_mi:.3f}",
                    "above_mi": f"{above_mi:.3f}",
                    "total_mi": f"{total_mi:.3f}",
                    "label": label,
                }
            )
            wrote += 1

    if missing:
        raise SystemExit(f"Missing tires in CSV: {missing}")
    print(f"Appended {wrote} labeled rows to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

