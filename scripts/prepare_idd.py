#!/usr/bin/env python3
"""Cache IDD at training resolution, once.

Two costs are paid here rather than every epoch:

* **Rasterising polygons.** Each annotation is 100-200 polygons of JSON. Doing
  that inside the data loader would dominate training time and, worse, would
  make an epoch's duration depend on how cluttered the scenes in it were.
* **Decoding 1920x1080 PNGs.** At 512x288 the model never sees those pixels.
  Decoding full-resolution PNGs and immediately throwing away 93% of them is
  the single largest waste available in this pipeline.

Writes ``<cache>/<split>/<key>.jpg`` and ``<key>.png`` -- image as JPEG because
it is a photograph, mask as PNG because a lossy label map is not a label map.

    python3 scripts/prepare_idd.py --root ~/IDD_Segmentation
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

from divas.perception.datasets.idd_polygons import (
    IGNORE, OTHER, ROAD, SHOULDER, find_pairs, load_record, rasterize,
)


def _one(job):
    pair_image, pair_label, key, cache, w, h = job
    import cv2
    out_img = FsPath(cache) / f"{key.replace('/', '_')}.jpg"
    out_lab = FsPath(cache) / f"{key.replace('/', '_')}.png"
    if out_img.exists() and out_lab.exists():
        return key, None
    img = cv2.imread(str(pair_image), cv2.IMREAD_COLOR)
    if img is None:
        return key, "unreadable image"
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    try:
        mask = rasterize(load_record(FsPath(pair_label)), size=(w, h))
    except Exception as exc:                       # truncated json, etc.
        return key, f"bad label: {exc}"
    cv2.imwrite(str(out_img), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(str(out_lab), mask)
    return key, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="~/IDD_Segmentation")
    ap.add_argument("--cache", default="~/IDD_cache")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    root = FsPath(args.root).expanduser()
    cache_root = FsPath(args.cache).expanduser()

    for split in args.splits:
        pairs = find_pairs(root, split)
        out = cache_root / split
        out.mkdir(parents=True, exist_ok=True)
        jobs = [(str(p.image), str(p.label), p.key, str(out), args.width, args.height)
                for p in pairs]
        print(f"{split}: {len(jobs)} pairs -> {out}")
        bad = []
        done = 0
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for key, err in ex.map(_one, jobs, chunksize=16):
                done += 1
                if err:
                    bad.append((key, err))
                if done % 500 == 0:
                    print(f"  {done}/{len(jobs)}")
        print(f"  done: {done - len(bad)} cached, {len(bad)} skipped")
        for key, err in bad[:10]:
            print(f"    {key}: {err}")

    # Class balance over the cached masks -- the weights the loss needs, and a
    # number worth quoting: on Indian roads the shoulder is not a rounding
    # error.
    import cv2
    counts = np.zeros(4, dtype=np.int64)
    files = sorted((cache_root / args.splits[0]).glob("*.png"))[:800]
    for f in files:
        m = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        for i, cls in enumerate((ROAD, SHOULDER, OTHER, IGNORE)):
            counts[i] += int((m == cls).sum())
    total = counts.sum()
    print(f"\nclass balance over {len(files)} cached masks:")
    for name, n in zip(("road", "drivable fallback", "not drivable", "ignore"), counts):
        print(f"  {name:20s} {100 * n / total:5.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
