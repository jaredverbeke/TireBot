#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_PATH = ROOT / "data" / "build_number.txt"


def main() -> None:
    current = ""
    if BUILD_PATH.exists():
        current = BUILD_PATH.read_text(encoding="utf-8").strip()

    prompt = f"Enter build number (current: {current or 'none'}): "
    raw = input(prompt).strip()
    if not raw:
        raise SystemExit("Aborted: no build number entered.")
    n = int(raw)  # validates numeric
    if n <= 0:
        raise SystemExit("Build number must be a positive integer.")

    BUILD_PATH.parent.mkdir(parents=True, exist_ok=True)
    BUILD_PATH.write_text(f"{n}\n", encoding="utf-8")
    print(f"Wrote {BUILD_PATH.relative_to(ROOT)} = {n}")


if __name__ == "__main__":
    main()

