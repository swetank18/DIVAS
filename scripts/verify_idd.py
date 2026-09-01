#!/usr/bin/env python3
"""Phase 0: confirm the IDD assumptions before anything depends on them.

    python3 scripts/verify_idd.py --root ~/datasets/IDD_Segmentation
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from divas.perception.datasets.idd import DRIVABLE_LEVEL3_NAMES, verify


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--level", type=int, default=3, choices=(1, 3))
    ap.add_argument("--sample", type=int, default=50)
    args = ap.parse_args()

    report = verify(Path(args.root), level=args.level, sample=args.sample)
    print(json.dumps(report, indent=2))
    if not report["exists"]:
        print("\nFAIL:", report["error"], file=sys.stderr)
        return 1

    total = sum(s.get("frames", 0) for s in report["splits"].values())
    print(f"\nframes found: {total}")
    print("drivable classes assumed:", ", ".join(DRIVABLE_LEVEL3_NAMES))
    ok = [s.get("plausible") for s in report["splits"].values() if "plausible" in s]
    if ok and not all(ok):
        print("\nWARNING: drivable pixel fraction is outside 10-70%. The label-id "
              "mapping in divas/perception/datasets/idd.py is probably wrong for "
              "this IDD release -- check its idd_labels definition.", file=sys.stderr)
        return 2
    return 0 if total else 1


if __name__ == "__main__":
    raise SystemExit(main())
