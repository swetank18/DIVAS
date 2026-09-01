#!/usr/bin/env python3
"""Run the ablation across the scenario suite and print the table.

This script produces the single most persuasive artifact in the submission:
each row removes exactly one mechanism from the row below it, so a difference
in the numbers can be attributed to a single change rather than to the stack
as a whole.

    python3 scripts/run_ablation.py --seeds 3 --jobs 8
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path as FsPath

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

from divas.eval import runner, scenarios
from divas.eval.metrics import aggregate


def _one(job):
    scenario_name, stack_index, seed = job
    sc = scenarios.get(scenario_name)
    st = runner.ABLATION[stack_index]
    m = runner.run(sc, st, seed=seed)
    return scenario_name, stack_index, m.as_row(), m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--scenarios", nargs="*", default=list(scenarios.SCENARIOS))
    ap.add_argument("--stacks", nargs="*", default=None,
                    help="stack names to include (default: all)")
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    idx = [
        i for i, s in enumerate(runner.ABLATION)
        if args.stacks is None or s.name in args.stacks
    ]
    jobs = [
        (sc, i, seed)
        for sc in args.scenarios
        for i in idx
        for seed in range(args.seeds)
    ]
    print(f"{len(jobs)} runs: {len(args.scenarios)} scenarios x {len(idx)} stacks "
          f"x {args.seeds} seeds, {args.jobs} workers\n")

    results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for n, (sc, si, row, m) in enumerate(ex.map(_one, jobs), 1):
            results.setdefault(si, []).append(m)
            flag = "OK " if row["success"] else ("HIT" if row["collision"] else "T/O")
            print(f"  [{n:3d}/{len(jobs)}] {flag} {sc:22s} {runner.ABLATION[si].name:32s} "
                  f"prog={row['progress_m']:6.1f}m v={row['mean_speed']:.1f}")

    print("\n" + "=" * 132)
    print("ABLATION  (each row adds one mechanism to the row above)")
    print("=" * 132)
    hdr = (f"{'stack':34s} {'succ':>6s} {'coll':>6s} {'t/o':>6s} {'prog_m':>7s} "
           f"{'v_mean':>7s} {'TTCok':>7s} {'minClr':>7s} {'d_safe':>7s} "
           f"{'jerk':>6s} {'latacc':>7s} {'e2e_ms':>7s}")
    print(hdr)
    print("-" * 132)
    table = []
    for si in idx:
        a = aggregate(results[si])
        table.append(a)
        print(f"{a['stack']:34s} {a['success_rate']:>6.2f} {a['collision_rate']:>6.2f} "
              f"{a['timeout_rate']:>6.2f} {a['mean_progress_m']:>7.1f} {a['mean_speed']:>7.2f} "
              f"{str(a['min_ttc_no_collision']):>7s} {str(a['min_clearance_m']):>7s} "
              f"{a['mean_d_safe_m']:>7.2f} {a['max_jerk']:>6.1f} {a['max_lat_accel']:>7.2f} "
              f"{a['e2e_ms_p95']:>7.1f}")
    print("=" * 132)
    for si in idx:
        print(f"  {runner.ABLATION[si].name:34s} {runner.ABLATION[si].description}")

    with open(args.out, "w") as f:
        json.dump(
            {"table": table,
             "runs": {runner.ABLATION[si].name: [r.as_row() for r in results[si]]
                      for si in idx}},
            f, indent=2,
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
