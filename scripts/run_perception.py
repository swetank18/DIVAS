#!/usr/bin/env python3
"""Run full stage-1 perception on an image: drivable mask + object detection.

    python3 scripts/run_perception.py demo/carla_town10_ours.png

Proves stage 1 is real pixel-in, structured-output-out: a pretrained
SegFormer checkpoint for the drivable mask, a pretrained YOLOv8n for boxed
actors (car, truck, bus, motorcycle, bicycle, pedestrian, animal). No
training done or claimed for either -- see the module docstrings in
divas/perception/segmentation.py and divas/perception/detection.py for
what that does and doesn't buy you.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from divas.perception.detection import ObjectDetector
from divas.perception.segmentation import DrivableSegmenter

_BOX_COLORS = {
    "car": (255, 200, 0),
    "truck": (255, 120, 0),
    "bus": (255, 60, 60),
    "motorcycle": (0, 200, 255),
    "bicycle": (0, 150, 255),
    "pedestrian": (255, 0, 200),
    "animal": (180, 0, 255),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="path to an RGB image")
    ap.add_argument("--out", default=None, help="overlay PNG path (default: <image>_perception.png)")
    ap.add_argument("--conf", type=float, default=0.35, help="detection confidence threshold")
    args = ap.parse_args()

    src = Path(args.image)
    out = Path(args.out) if args.out else src.with_name(src.stem + "_perception.png")

    img = np.array(Image.open(src).convert("RGB"))
    print(f"loaded {src}  {img.shape[1]}x{img.shape[0]}")

    print("loading DrivableSegmenter (SegFormer-B0/Cityscapes, CPU)...")
    seg = DrivableSegmenter()
    mask = seg.predict(img)
    frac = float(mask.mean())
    print(f"drivable fraction: {frac:.1%}")

    print("loading ObjectDetector (YOLOv8n/COCO, CPU)...")
    det = ObjectDetector(confidence_threshold=args.conf)
    detections = det.predict(img)
    print(f"detected {len(detections)} road-relevant actor(s):")
    for d in detections:
        x1, y1, x2, y2 = d.box_xyxy.round(0).astype(int)
        print(f"  {d.divas_class:12s} conf={d.confidence:.2f}  box=({x1},{y1})-({x2},{y2})  coco={d.coco_class}")

    # -- overlay: drivable mask in green, boxes on top -----------------
    overlay_arr = img.copy()
    green = np.zeros_like(img)
    green[..., 1] = 255
    overlay_arr = np.where(
        mask[..., None], (0.55 * overlay_arr + 0.45 * green).astype(np.uint8), overlay_arr
    )
    overlay = Image.fromarray(overlay_arr)
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:
        font = ImageFont.load_default()

    for d in detections:
        x1, y1, x2, y2 = d.box_xyxy.tolist()
        color = _BOX_COLORS.get(d.divas_class, (255, 255, 255))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{d.divas_class} {d.confidence:.2f}"
        tb = draw.textbbox((x1, y1), label, font=font)
        draw.rectangle([tb[0], tb[1] - 2, tb[2] + 4, tb[3] + 2], fill=color)
        draw.text((x1 + 2, y1 - 2), label, fill=(0, 0, 0), font=font)

    overlay.save(out)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
