#!/usr/bin/env python3
"""Real detections -> world-frame tracks -> predicted future paths.

    python3 scripts/run_prediction_demo.py demo/carla_town10_ours.png
    python3 scripts/run_prediction_demo.py frame1.jpg frame2.jpg --dt 0.5

One image: YOLOv8n detects actors, boxes are projected to ground points
(flat-ground, nominal FOV -- see divas/perception/tracking.py), and every
track starts with zero velocity, because one photo has no motion in it.

Two images (frame1 then frame2, ``dt`` seconds apart): actors are matched
between frames and get a real, measured velocity from actual displacement.
This is the version that produces a meaningful predicted path -- a
stationary-velocity track just gets pushed around by the social-force
repulsion term with no forward motion of its own.

Either way, the predictor is divas.prediction.predictors.SocialForcePredictor
-- the same physics rollout used everywhere else in this repo, not a
trained model. No model in this project predicts future motion; see
divas/perception/tracking.py and divas/prediction/predictors.py for why.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from divas.perception.bev import Camera, project
from divas.perception.detection import ObjectDetector
from divas.perception.tracking import detections_to_ground, ground_to_tracks
from divas.prediction.predictors import SocialForcePredictor

_COLORS = {
    "car": (255, 200, 0), "truck": (255, 120, 0), "bus": (255, 60, 60),
    "motorcycle": (0, 200, 255), "bicycle": (0, 150, 255),
    "pedestrian": (255, 0, 200), "animal": (180, 0, 255),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image1")
    ap.add_argument("image2", nargs="?", default=None,
                     help="optional second frame, dt seconds later, for a real velocity")
    ap.add_argument("--dt", type=float, default=0.5, help="seconds between image1 and image2")
    ap.add_argument("--horizon", type=float, default=3.0, help="prediction horizon, seconds")
    ap.add_argument("--fov", type=float, default=90.0, help="assumed horizontal FOV, degrees")
    ap.add_argument("--cam-height", type=float, default=1.35, help="assumed camera height, m")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    img1 = np.array(Image.open(args.image1).convert("RGB"))
    h, w = img1.shape[:2]
    cam = Camera.from_fov(w, h, fov_deg=args.fov, height=args.cam_height)
    print(f"camera: nominal FOV={args.fov} deg, height={args.cam_height} m "
          f"-- not calibrated, metres are indicative only")

    det = ObjectDetector(confidence_threshold=args.conf)
    d1 = det.predict(img1)
    g1 = detections_to_ground(d1, cam)
    print(f"frame 1: {len(d1)} detected, {len(g1)} projected to ground "
          f"(dropped = above horizon or bad box)")

    g2 = None
    if args.image2:
        img2 = np.array(Image.open(args.image2).convert("RGB"))
        d2 = det.predict(img2)
        g2 = detections_to_ground(d2, cam)
        print(f"frame 2: {len(d2)} detected, {len(g2)} projected to ground")
        tracks = ground_to_tracks(g2, g1, dt=args.dt)
    else:
        print("single image -- every track starts at zero velocity, see module docstring")
        tracks = ground_to_tracks(g1)

    for t in tracks:
        speed = float(np.hypot(t.vx, t.vy))
        print(f"  track {t.id:2d}  {t.cls:12s}  x={t.x:6.1f}m y={t.y:6.1f}m  "
              f"speed={speed:.1f} m/s")

    if not tracks:
        print("no trackable actors -- nothing to predict")
        return 0

    predictor = SocialForcePredictor(horizon=args.horizon)
    traj_set = predictor.predict(tracks, grid=None, ego=None)

    print(f"\npredicted {args.horizon}s ahead:")
    for tr in traj_set:
        path = tr.mean_path()
        end = path[-1]
        conf = tr.confidence
        print(f"  track {tr.track_id:2d}  {tr.cls:12s}  now=({tracks[tr.track_id].x:.1f},"
              f"{tracks[tr.track_id].y:.1f})  -> +{args.horizon}s=({end[0]:.1f},{end[1]:.1f})  "
              f"confidence={conf:.2f}")

    # -- overlay: current box (solid) + predicted path (dots fading ahead) --
    base = Image.fromarray(img1 if not args.image2 else np.array(Image.open(args.image2).convert("RGB")))
    draw = ImageDraw.Draw(base)
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:
        font = ImageFont.load_default()

    dets_for_overlay = d2 if args.image2 else d1
    for d in dets_for_overlay:
        x1, y1, x2, y2 = d.box_xyxy.tolist()
        color = _COLORS.get(d.divas_class, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    for tr in traj_set:
        color = _COLORS.get(tr.cls, (255, 255, 255))
        path = tr.mean_path()
        n = len(path)
        for i, (px, py) in enumerate(path):
            u, v, valid = project(cam, np.array([px]), np.array([py]))
            if not valid[0]:
                continue
            r = 3
            alpha = 0.3 + 0.7 * (i / max(n - 1, 1))
            faded = tuple(int(c * alpha + 255 * (1 - alpha) * 0.3) for c in color)
            ux, vy = float(u[0]), float(v[0])
            draw.ellipse([ux - r, vy - r, ux + r, vy + r], fill=faded)

    out = Path(args.out) if args.out else Path(args.image1).with_name(
        Path(args.image1).stem + "_prediction.png")
    base.save(out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
