#!/usr/bin/env python3
"""Export closed-loop runs as JSON, for the interactive 3-D replay.

The web replay must show the *same* episodes the ablation table reports, not a
re-enactment. So this runs the real runner over the real scenarios and dumps
the trace it records -- ego pose, tracked actors, the planned path and the
safety margin, per control step -- rather than anything drawn by hand.

    python3 scripts/export_replay.py --scenario pedestrian_crossing

Two arms per scenario, on the same seed and therefore against identical
traffic, because the whole argument is a comparison. The output carries the
outcome of each arm including where and what it hit, so the page cannot
accidentally show a collision-free baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

from divas.eval import runner
from divas.eval.scenarios import SCENARIOS, get


def r3(v) -> float:
    """Round hard. These files are downloaded by a browser, and full float64
    repr triples the size for precision no viewer can resolve."""
    return round(float(v), 3)


def export_arm(scenario, stack, seed: int, stride: int):
    world_box = {}

    original_build = scenario.build

    def build(s):
        w = original_build(s)
        world_box["w"] = w
        return w

    scenario = scenario.__class__(
        name=scenario.name, description=scenario.description, build=build,
        goal_progress=scenario.goal_progress, time_limit=scenario.time_limit,
        tests=getattr(scenario, "tests", ""),
    )
    cfg = runner.RunnerConfig(record=True, record_every=stride)
    m = runner.run(scenario, stack, seed=seed, cfg=cfg)
    tr = m.trace
    world = world_box["w"]

    frames = []
    for i in range(len(tr["t"])):
        ex, ey, eth = tr["ego"][i]
        path = tr["path"][i]
        frames.append({
            "t": r3(tr["t"][i]),
            "ego": [r3(ex), r3(ey), r3(eth)],
            "v": r3(tr["v"][i]),
            "d_safe": r3(tr["d_safe"][i]),
            "progress": r3(tr["progress"][i]),
            "actors": [[r3(a[0]), r3(a[1]), r3(a[2]), a[3]] for a in tr["actors"][i]],
            # Every third path point: the planner emits them ~0.5 m apart and a
            # ribbon in a browser does not need that.
            "path": ([[r3(p[0]), r3(p[1])] for p in path[::3]]
                     if path is not None else []),
        })

    return {
        "stack": stack.name,
        "success": bool(m.success),
        "collision": m.collision_with or None,
        "progress_m": r3(m.progress),
        "mean_speed": r3(m.mean_speed),
        "min_clearance_m": (r3(m.min_clearance)
                            if m.min_clearance is not None else None),
        "frames": frames,
    }, world


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="pedestrian_crossing",
                    choices=sorted(SCENARIOS))
    ap.add_argument("--left", default="baseline_conventional")
    ap.add_argument("--right", default="cv_pred_fixed_margin")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--stride", type=int, default=2, help="record every Nth step")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scenario = get(args.scenario)
    arms, world = [], None
    for name in (args.left, args.right):
        stack = next(s for s in runner.ABLATION if s.name == name)
        arm, world = export_arm(scenario, stack, args.seed, args.stride)
        arms.append(arm)
        outcome = ("reached the goal" if arm["success"]
                   else (f"HIT {arm['collision']}" if arm["collision"] else "timed out"))
        print(f"  {name:32s} {outcome:28s} {arm['progress_m']:6.1f} m, "
              f"{len(arm['frames'])} frames")

    poly = world.road.polygon()
    statics = []
    for o in world.statics:
        if hasattr(o, "radius"):
            statics.append({"kind": "pothole", "x": r3(o.x), "y": r3(o.y),
                            "r": r3(o.radius)})
        else:
            statics.append({"kind": "block", "x": r3(o.x), "y": r3(o.y),
                            "theta": r3(o.theta), "l": r3(o.length),
                            "w": r3(o.width)})

    payload = {
        "scenario": args.scenario,
        "description": scenario.description,
        "tests": getattr(scenario, "tests", ""),
        "seed": args.seed,
        "road": [[r3(p[0]), r3(p[1])] for p in poly],
        "centerline": [[r3(p[0]), r3(p[1])] for p in world.road.centerline],
        "statics": statics,
        "vehicle": {"length": r3(world.params.length),
                    "width": r3(world.params.width)},
        "arms": arms,
    }

    out = FsPath(args.out or f"docs/replay-{args.scenario}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, separators=(",", ":")))
    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
