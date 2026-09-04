#!/usr/bin/env python3
"""Train stage 1: drivable-area segmentation on IDD.

    python3 scripts/prepare_idd.py            # once
    ~/carla-venv/bin/python3 scripts/train_drivable.py --epochs 30

Reports **per-class IoU**, not pixel accuracy. On a typical IDD frame 55% of
pixels are not drivable, so a model that predicts "not drivable" everywhere
scores 55% accuracy and is worth nothing. IoU on the two drivable classes is
the number that means something, and ``drivable fallback`` -- the unpaved
shoulder -- is the one to watch, because it is the class Western datasets do
not have and the reason this project trains on IDD at all.

Checkpoints on best mean IoU over the *drivable* classes rather than over all
three: the model is a free-space detector, and its skill at "not drivable"
is a byproduct.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path as FsPath

import numpy as np

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from divas.perception.datasets.idd_polygons import (
    CLASS_NAMES, IGNORE, OTHER, ROAD, SHOULDER,
)
from divas.perception.models.drivable import (
    N_CLASSES, build_model, confusion, free_space_iou, iou_from_confusion,
    normalise,
)


class CachedIDD(Dataset):
    """Reads the cache written by ``scripts/prepare_idd.py``."""

    def __init__(self, cache: FsPath, split: str, augment: bool = False) -> None:
        self.dir = FsPath(cache).expanduser() / split
        self.keys = sorted(p.stem for p in self.dir.glob("*.png"))
        self.augment = augment
        if not self.keys:
            raise SystemExit(f"no cached data in {self.dir}; run prepare_idd.py first")

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int):
        k = self.keys[i]
        img = cv2.imread(str(self.dir / f"{k}.jpg"), cv2.IMREAD_COLOR)[:, :, ::-1]
        mask = cv2.imread(str(self.dir / f"{k}.png"), cv2.IMREAD_GRAYSCALE)
        if self.augment:
            img, mask = self._augment(img, mask)
        return torch.from_numpy(normalise(img)), torch.from_numpy(mask.astype(np.int64))

    def _augment(self, img, mask):
        # Horizontal flip only. No vertical flip and no rotation: the camera is
        # rigidly mounted and the horizon does not move, so those would train
        # the model on views the vehicle can never produce.
        if np.random.rand() < 0.5:
            img, mask = img[:, ::-1], mask[:, ::-1]
        # Photometric jitter, which is the variation that *is* real -- Indian
        # daylight ranges from overcast to hard glare within one drive.
        if np.random.rand() < 0.8:
            img = img.astype(np.float32)
            img *= np.random.uniform(0.6, 1.4)                  # exposure
            img += np.random.uniform(-20, 20)                   # black level
            img = np.clip(img, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(img), np.ascontiguousarray(mask)


def class_frequencies(ds: "CachedIDD", limit: int = 800) -> np.ndarray:
    """Pixel frequency per class, read off the cached masks.

    Measured rather than assumed. The distribution is what sets the loss
    weights, and on IDD it is strongly imbalanced -- ``drivable fallback`` is
    about 4% of pixels -- so a wrong split here quietly teaches the model to
    skip the one class the dataset was chosen for.
    """
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for k in ds.keys[:limit]:
        m = cv2.imread(str(ds.dir / f"{k}.png"), cv2.IMREAD_GRAYSCALE)
        for c in range(N_CLASSES):
            counts[c] += int((m == c).sum())
    return counts / max(counts.sum(), 1)


def evaluate(model, loader, device):
    model.eval()
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                pred = model(x)["out"].argmax(1)
            cm += confusion(pred.cpu().numpy(), y.numpy())
    return iou_from_confusion(cm), cm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", default="~/IDD_cache")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="divas/perception/models/drivable_idd.pt")
    ap.add_argument("--log", default="docs/drivable-training.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train = CachedIDD(args.cache, "train", augment=True)
    val = CachedIDD(args.cache, "val", augment=False)
    print(f"train {len(train)}  val {len(val)}  device {device}")

    tl = DataLoader(train, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)
    vl = DataLoader(val, batch_size=args.batch, shuffle=False,
                    num_workers=args.workers, pin_memory=True)

    model = build_model().to(device)
    # Class weights are *measured from the cache*, not written down. An earlier
    # version hard-coded a 20/26/55 split taken from a single frame that
    # happened to be mostly shoulder; the true distribution over 800 masks is
    # 25.4 / 4.3 / 70.3, so `drivable fallback` is a rare class rather than a
    # co-equal one, and weights built from the wrong split would have trained
    # the model to ignore exactly the class the dataset was chosen for.
    freq = class_frequencies(train)
    print("measured class balance: " + "  ".join(
        f"{n} {100*f:.2f}%" for n, f in zip(CLASS_NAMES, freq)))
    # Inverse frequency, square-rooted. Raw inverse frequency puts ~16x the
    # weight on a 4% class and the model chases it into false positives across
    # the whole frame; the square root is the usual compromise.
    inv = 1.0 / np.sqrt(np.maximum(freq, 1e-6))
    w = torch.tensor(inv / inv.sum() * N_CLASSES, dtype=torch.float32, device=device)
    print("loss weights: " + np.array2string(w.cpu().numpy(), precision=3))
    lossf = nn.CrossEntropyLoss(weight=w, ignore_index=IGNORE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * len(tl), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best, history = -1.0, []
    out = FsPath(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in tl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = lossf(model(x)["out"], y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += float(loss.item()) * x.size(0)
            n += x.size(0)

        iou, cm = evaluate(model, vl, device)
        drivable_miou = float(np.nanmean(iou[[ROAD, SHOULDER]]))
        fs_iou = free_space_iou(cm)
        history.append({"epoch": epoch, "loss": total / max(n, 1),
                        "iou": [None if np.isnan(v) else round(float(v), 4) for v in iou],
                        "drivable_miou": round(drivable_miou, 4),
                        "free_space_iou": round(fs_iou, 4)})
        print(f"epoch {epoch:3d}  loss {total/max(n,1):.4f}  "
              f"IoU road {iou[ROAD]:.3f}  fallback {iou[SHOULDER]:.3f}  "
              f"other {iou[OTHER]:.3f}  drivable mIoU {drivable_miou:.3f}  "
              f"free-space {fs_iou:.3f}  ({time.time()-t0:.0f}s)")

        if drivable_miou > best:
            best = drivable_miou
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "drivable_miou": best, "classes": list(CLASS_NAMES)}, out)
            print(f"          saved {out} (best)")

        FsPath(args.log).parent.mkdir(parents=True, exist_ok=True)
        FsPath(args.log).write_text(json.dumps(
            {"history": history, "best_drivable_miou": best,
             "classes": list(CLASS_NAMES)}, indent=2))

    print(f"\nbest drivable mIoU {best:.3f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
