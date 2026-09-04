#!/usr/bin/env python3
"""What does perception error cost the planner, in metres?

Segmentation is normally scored with IoU, which answers "how many pixels did
the network get right". That is not the question the vehicle asks. The vehicle
asks: *given what the network saw, does the planner still find the same way
through?*

So every frame is planned **twice** -- once on the network's free space, once
on the annotator's -- through the same Hybrid A* that produced the closed-loop
ablation. The comparison gives three numbers that IoU cannot:

* **plan availability**: the fraction of frames a path is found at all;
* **reach agreement**: how far the plan gets, predicted against ground truth;
* **path divergence**: the mean lateral distance between the two paths, which
  is the quantity that actually decides whether the vehicle ends up somewhere
  else.

A frame where IoU is 0.85 but both planners produce the same path is a frame
where perception was good enough. A frame where IoU is 0.95 and the paths
diverge by two metres is not, and the pixel metric cannot tell them apart.

    ~/carla-venv/bin/python3 scripts/eval_perception_planning.py --n 200
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

import cv2

from divas.perception.bev import (
    reachable_goal,
    BevSpec, Camera, free_space_to_grid, max_reliable_range,
)
from divas.perception.datasets.idd_polygons import (
    ROAD, SHOULDER, find_pairs, load_record, rasterize,
)
from divas.perception.models.drivable import (
    DrivableSegmenter, confusion, iou_from_confusion,
)
from divas.planning import FallbackPlanner, PlannerConfig, RRTConfig
from divas.types import EgoState, VehicleParams


def free_of(mask, shoulder: bool) -> np.ndarray:
    out = mask == ROAD
    if shoulder:
        out |= mask == SHOULDER
    return out


def lateral_divergence(a, b) -> float:
    """Mean distance from each point of ``a`` to the nearest point of ``b``.

    Symmetric would be tidier; this direction is the one that matters, because
    it asks how far the *predicted* plan strays from where the true free space
    would have taken the vehicle.
    """
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return float("nan")
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="~/IDD_Segmentation")
    ap.add_argument("--checkpoint", default="divas/perception/models/drivable_idd.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--height", type=float, default=1.35)
    ap.add_argument("--pitch", type=float, default=4.0)
    ap.add_argument("--shoulder", action="store_true")
    ap.add_argument("--out", default="docs/perception-planning.json")
    args = ap.parse_args()

    seg = DrivableSegmenter.load(FsPath(args.checkpoint))
    pairs = find_pairs(FsPath(args.root).expanduser(), args.split)[: args.n]
    params = VehicleParams()
    planner = FallbackPlanner(params, PlannerConfig(time_budget_ms=None),
                              RRTConfig(time_budget_ms=None))

    cm = np.zeros((3, 3), dtype=np.int64)
    rows = []
    for i, pair in enumerate(pairs):
        photo = cv2.imread(str(pair.image), cv2.IMREAD_COLOR)[:, :, ::-1]
        h, w = photo.shape[:2]
        truth = rasterize(load_record(pair.label), size=(w, h))
        pred = seg(photo)
        cm += confusion(pred, truth)

        cam = Camera.from_fov(w, h, fov_deg=args.fov, height=args.height,
                              pitch=math.radians(args.pitch))
        reach = min(max_reliable_range(cam, h), 30.0)
        spec = BevSpec(forward=reach, behind=3.0, lateral=12.0)

        g_true = free_space_to_grid(free_of(truth, args.shoulder), cam, spec)
        g_pred = free_space_to_grid(free_of(pred, args.shoulder), cam, spec)
        goal = reachable_goal(g_true, reach)

        start = EgoState(x=0.0, y=0.0, theta=0.0, v=6.0)
        r_true = planner.plan(g_true, start, goal)
        r_pred = planner.plan(g_pred, start, goal)

        xy_t = r_true.path.xy if len(r_true.path) >= 2 else None
        xy_p = r_pred.path.xy if len(r_pred.path) >= 2 else None
        rows.append({
            "key": pair.key,
            "goal_m": round(float(goal[0]), 2),
            "truth_ok": bool(r_true.success),
            "pred_ok": bool(r_pred.success),
            "truth_reach": round(float(xy_t[-1, 0]) if xy_t is not None else 0.0, 2),
            "pred_reach": round(float(xy_p[-1, 0]) if xy_p is not None else 0.0, 2),
            "divergence_m": round(lateral_divergence(xy_p, xy_t), 3),
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(pairs)}")

    iou = iou_from_confusion(cm)
    ok_t = np.mean([r["truth_ok"] for r in rows])
    ok_p = np.mean([r["pred_ok"] for r in rows])
    div = np.array([r["divergence_m"] for r in rows], dtype=float)
    div = div[np.isfinite(div)]
    reach_t = np.array([r["truth_reach"] for r in rows])
    reach_p = np.array([r["pred_reach"] for r in rows])

    print(f"\n{len(rows)} frames of IDD {args.split}, "
          f"shoulder {'counted' if args.shoulder else 'excluded'} as drivable\n")
    print("segmentation")
    for n, v in zip(("road", "drivable fallback", "not drivable"), iou):
        print(f"  IoU {n:20s} {v:.3f}")
    print("\nplanning, same goal on both grids")
    print(f"  path found, ground-truth free space   {ok_t:.3f}")
    print(f"  path found, predicted free space      {ok_p:.3f}")
    print(f"  reach, ground truth                   {reach_t.mean():5.1f} m")
    print(f"  reach, predicted                      {reach_p.mean():5.1f} m")
    print(f"\n  mean path divergence                  {div.mean():5.2f} m")
    print(f"  median                                {np.median(div):5.2f} m")
    print(f"  90th percentile                       {np.percentile(div, 90):5.2f} m")
    print(f"  frames within half a lane (1.5 m)     {100*np.mean(div < 1.5):5.1f}%")

    payload = {
        "split": args.split, "frames": len(rows), "shoulder": args.shoulder,
        "iou": {n: (None if np.isnan(v) else round(float(v), 4))
                for n, v in zip(("road", "drivable_fallback", "not_drivable"), iou)},
        "plan_found_truth": round(float(ok_t), 4),
        "plan_found_pred": round(float(ok_p), 4),
        "reach_truth_m": round(float(reach_t.mean()), 3),
        "reach_pred_m": round(float(reach_p.mean()), 3),
        "divergence_m": {"mean": round(float(div.mean()), 3),
                         "median": round(float(np.median(div)), 3),
                         "p90": round(float(np.percentile(div, 90)), 3),
                         "within_1p5m": round(float(np.mean(div < 1.5)), 4)},
        "rows": rows,
    }
    out = FsPath(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
