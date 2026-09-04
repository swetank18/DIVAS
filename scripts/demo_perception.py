#!/usr/bin/env python3
"""Photograph in, planned path out -- the whole stack on a real Indian road.

This is the demo the project exists to make. A frame from the India Driving
Dataset goes in; stage 1 segments the drivable area, stage 2 projects it onto
the ground plane, and the *same* Hybrid A* that produced the ablation numbers
searches the resulting occupancy grid. Nothing here is drawn by hand and
nothing is supplied by a simulator.

    ~/carla-venv/bin/python3 scripts/demo_perception.py --n 6

Three panels per frame:

* the photograph, with the network's free space painted on it -- road in one
  colour, ``drivable fallback`` (the unpaved shoulder Cityscapes has no label
  for) in another;
* the bird's-eye occupancy grid that free space projects to;
* the path Hybrid A* found through it.

**What this does and does not show.** It is stage 1 -> 2 -> 5 on real imagery,
which is the claim. It is not closed-loop: a single photograph has no next
frame, so there is no control step and no prediction -- those need the
simulator, and they are what the CARLA and 2-D results cover. Also, IDD ships
no camera calibration, so the metre scale of the bird's-eye view comes from a
nominal focal length and mounting height. The geometry is self-consistent; the
absolute distances are an assumption, and the figure says so.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

import cv2

from divas.perception.bev import (
    BevSpec, Camera, free_space_to_grid, max_reliable_range, reachable_goal,
)
from divas.perception.datasets.idd_polygons import (
    OTHER, ROAD, SHOULDER, find_pairs, load_record, rasterize,
)
from divas.perception.models.drivable import DrivableSegmenter
from divas.planning import FallbackPlanner, PlannerConfig, RRTConfig
from divas.types import EgoState, VehicleParams

ROAD_RGB = (42, 157, 143)          # teal, matching the stack's own plots
SHOULDER_RGB = (233, 196, 106)     # sand
PATH_RGB = (231, 111, 81)


def overlay(image_rgb, pred, alpha=0.45):
    out = image_rgb.astype(np.float32).copy()
    for cls, colour in ((ROAD, ROAD_RGB), (SHOULDER, SHOULDER_RGB)):
        m = pred == cls
        if m.any():
            out[m] = (1 - alpha) * out[m] + alpha * np.asarray(colour, np.float32)
    return out.astype(np.uint8)


def render(fig_path, photo, pred, grid, result, cam, reach, title, truth=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), dpi=120)

    axes[0].imshow(overlay(photo, pred))
    axes[0].set_title("1 · what the camera sees, free space from the network",
                      fontsize=10, loc="left")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[0].legend(handles=[
        Patch(color=np.array(ROAD_RGB) / 255, label="road"),
        Patch(color=np.array(SHOULDER_RGB) / 255, label="drivable fallback (shoulder)"),
    ], loc="lower left", fontsize=7, framealpha=0.85)

    occ = grid.occupied_mask()
    extent = (grid.origin[0], grid.origin[0] + occ.shape[1] * grid.resolution,
              grid.origin[1], grid.origin[1] + occ.shape[0] * grid.resolution)
    for ax, with_path in ((axes[1], False), (axes[2], True)):
        ax.imshow(~occ, origin="lower", extent=extent, cmap="Greys_r",
                  vmin=0, vmax=1.6, interpolation="nearest")
        ax.set_aspect("equal")
        ax.set_xlabel("forward, m"); ax.set_ylabel("left, m")
        ax.axhline(0, color="#bbb", lw=0.6, ls=":")
        ax.plot([0], [0], marker="s", ms=7, color=np.array(ROAD_RGB) / 255, zorder=5)
        ax.axvline(reach, color="#c1121f", lw=0.9, ls="--")
        ax.text(reach, extent[3] * 0.86, " flat-ground limit", fontsize=7,
                color="#c1121f", rotation=90, va="top")
    axes[1].set_title("2 · projected to the ground plane", fontsize=10, loc="left")
    axes[2].set_title("3 · Hybrid A* over the free space", fontsize=10, loc="left")

    if result is not None and len(result.path) >= 2:
        xy = result.path.xy
        axes[2].plot(xy[:, 0], xy[:, 1], color=np.array(PATH_RGB) / 255, lw=2.6,
                     zorder=6)
        axes[2].plot(xy[-1, 0], xy[-1, 1], "o", ms=6,
                     color=np.array(PATH_RGB) / 255, zorder=7)

    fig.suptitle(title, fontsize=12, x=0.006, ha="left", fontweight="bold")
    fig.text(0.006, 0.015,
             "IDD ships no camera calibration: the bird's-eye scale uses a "
             "nominal 90° field of view and a 1.35 m mounting height. "
             "Geometry is self-consistent; absolute metres are an assumption.",
             fontsize=7, color="#666")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(fig_path, facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="~/IDD_Segmentation")
    ap.add_argument("--checkpoint", default="divas/perception/models/drivable_idd.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--stride", type=int, default=97, help="spread the picks out")
    ap.add_argument("--sequence", default=None,
                    help="restrict to one IDD drive sequence, e.g. 63. Frames "
                         "within a sequence come from the same vehicle on the "
                         "same drive, so the reel reads as one journey -- though "
                         "IDD samples them sparsely (median gap of ~2400 frame "
                         "ids), so it is a montage and not continuous video")
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--height", type=float, default=1.35)
    ap.add_argument("--pitch", type=float, default=4.0, help="degrees, nose down")
    ap.add_argument("--shoulder", action="store_true",
                    help="count drivable fallback as free space")
    ap.add_argument("--out", default="demo/perception")
    ap.add_argument("--video", default=None,
                    help="also encode the figures into an MP4 reel at this path")
    ap.add_argument("--fps", type=int, default=2)
    args = ap.parse_args()

    ckpt = FsPath(args.checkpoint)
    if not ckpt.exists():
        print(f"no checkpoint at {ckpt} -- train first:\n"
              f"  ~/carla-venv/bin/python3 scripts/train_drivable.py", file=sys.stderr)
        return 2

    seg = DrivableSegmenter.load(ckpt)
    pairs = find_pairs(FsPath(args.root).expanduser(), args.split)
    if args.sequence:
        pairs = [p for p in pairs if p.key.split("/")[0] == args.sequence]
        if not pairs:
            print(f"no frames in sequence {args.sequence!r}", file=sys.stderr)
            return 2
    picks = pairs[:: max(args.stride, 1)][: args.n]

    out = FsPath(args.out); out.mkdir(parents=True, exist_ok=True)
    params = VehicleParams()
    planner = FallbackPlanner(params, PlannerConfig(time_budget_ms=None),
                              RRTConfig(time_budget_ms=None))

    for i, pair in enumerate(picks):
        bgr = cv2.imread(str(pair.image), cv2.IMREAD_COLOR)
        photo = bgr[:, :, ::-1]
        h, w = photo.shape[:2]
        pred = seg(photo)

        cam = Camera.from_fov(w, h, fov_deg=args.fov, height=args.height,
                              pitch=math.radians(args.pitch))
        reach = min(max_reliable_range(cam, h), 30.0)
        free = (pred == ROAD) | ((pred == SHOULDER) if args.shoulder else False)
        grid = free_space_to_grid(free, cam, BevSpec(forward=reach, behind=3.0,
                                                     lateral=12.0))

        goal = reachable_goal(grid, reach)
        res = planner.plan(grid, EgoState(x=0.0, y=0.0, theta=0.0, v=6.0), goal)

        drivable_pct = 100.0 * float(free.mean())
        title = (f"DIVAS · IDD {args.split}/{pair.key} — "
                 f"{drivable_pct:.0f}% of frame drivable, "
                 f"plan {'found' if res.success else 'partial'} "
                 f"to {goal[0]:.0f} m")
        fig = out / f"perception_{i:02d}_{pair.key.replace('/', '_')}.png"
        render(fig, photo, pred, grid, res, cam, reach, title)
        print(f"  {fig}  drivable {drivable_pct:5.1f}%  "
              f"goal {goal[0]:5.1f} m  {'OK' if res.success else 'partial'}")

    print(f"\nwrote {len(picks)} figures to {out}")

    if args.video:
        import shutil
        import subprocess
        if shutil.which("ffmpeg") is None:
            print("ffmpeg not found; figures are in", out, file=sys.stderr)
            return 0
        vid = FsPath(args.video)
        vid.parent.mkdir(parents=True, exist_ok=True)
        # A reel rather than a continuous drive, and the difference matters:
        # IDD samples frames out of its sequences, so consecutive images are
        # from the same drive but seconds apart. Encoding them at 2 fps makes
        # that obvious instead of implying a video the dataset does not carry.
        subprocess.run(
            ["ffmpeg", "-y", "-framerate", str(args.fps),
             "-pattern_type", "glob", "-i", str(out / "perception_*.png"),
             "-vf", "scale=1600:-2", "-c:v", "libx264", "-preset", "slow",
             "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
             str(vid)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"wrote {vid}  ({len(picks)} frames at {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
