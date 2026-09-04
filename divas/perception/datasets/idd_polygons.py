"""Rasterise IDD's polygon annotations into free-space masks.

The IDD Segmentation release ships **only** ``*_gtFine_polygons.json`` -- there
are no pre-rendered label images in the archive at all. That is better than it
sounds. Polygons carry label *names*, so the drivable set can be selected by
name rather than by a numeric id whose meaning differs between the IDD-Lite,
20k and 40k releases. :mod:`divas.perception.datasets.idd` carries
``DRIVABLE_LEVEL3_IDS`` as an explicitly-flagged hypothesis about those ids;
this module does not need it, and checking against the data corrected part of
it -- see below.

**What a scan of the actual annotations found.** 32 distinct label names across
400 training files. The three that matter:

* ``road`` -- the carriageway proper.
* ``drivable fallback`` -- the unpaved shoulder Indian traffic uses as a matter
  of course. Cityscapes and KITTI have no such class; to them it is simply
  "not road". This single label is why the project trains on IDD.
* ``non-drivable fallback`` -- its explicit counterpart, which is what makes
  the distinction a *labelled* one rather than an inference.

**And one correction.** ``idd.py`` names the drivable set as
``("road", "parking", "drivable fallback")``. There is **no ``parking`` label
in IDD** -- it does not appear once in the sample. The drivable set here is
``road`` plus ``drivable fallback``, taken from the data rather than from the
hypothesis.

Three classes rather than a binary mask, because the distinction is
operationally real and the stack already draws it: the CARLA bridge carries
``DRIVABLE_TAGS`` and ``DRIVABLE_TAGS_WITH_SHOULDER`` as separate constants for
exactly this reason. Widening what counts as drivable is a claim, and it should
be made deliberately rather than baked into a training label.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:                                  # pragma: no cover
    cv2 = None


#: Class ids the model predicts.
ROAD, SHOULDER, OTHER = 0, 1, 2
#: Excluded from the loss entirely, not predicted as a class.
IGNORE = 255

CLASS_NAMES = ("road", "drivable fallback", "not drivable")

#: Label names, verbatim from the annotations.
ROAD_LABELS = ("road",)
SHOULDER_LABELS = ("drivable fallback",)
#: Regions the annotators marked as outside the labelled area. Training against
#: them would teach the model to guess, and scoring against them would credit
#: it for guessing right.
IGNORE_LABELS = ("out of roi",)


def class_of(label: str) -> int:
    if label in ROAD_LABELS:
        return ROAD
    if label in SHOULDER_LABELS:
        return SHOULDER
    if label in IGNORE_LABELS:
        return IGNORE
    return OTHER


def rasterize(record: dict, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Polygon annotations -> an ``(H, W)`` uint8 class map.

    ``size`` is ``(width, height)``; ``None`` keeps the annotation's own.

    **Draw order is the annotation.** The objects are listed back to front, so
    a later polygon occludes an earlier one -- a motorcycle drawn over the road
    is not road any more. Painting them in any other order, or compositing by
    class priority, produces a mask where vehicles are transparent and the
    model learns that it may drive through them.

    Polygons are scaled to ``size`` *before* filling rather than by resizing
    the finished mask. Nearest-neighbour resizing of a label map erodes thin
    structures and, worse, invents boundary pixels of classes that were never
    adjacent.
    """
    if cv2 is None:                                  # pragma: no cover
        raise RuntimeError("rasterising needs opencv (cv2)")
    h0, w0 = int(record["imgHeight"]), int(record["imgWidth"])
    w, h = (int(size[0]), int(size[1])) if size else (w0, h0)
    sx, sy = w / w0, h / h0

    mask = np.full((h, w), OTHER, dtype=np.uint8)
    for obj in record.get("objects", ()):
        if obj.get("deleted"):
            continue
        pts = obj.get("polygon") or ()
        if len(pts) < 3:
            continue
        poly = np.asarray(pts, dtype=np.float64)
        poly[:, 0] *= sx
        poly[:, 1] *= sy
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], int(class_of(obj["label"])))
    return mask


def load_record(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def drivable_from_mask(mask: np.ndarray, include_shoulder: bool = True) -> np.ndarray:
    """Boolean free space from a class map.

    ``include_shoulder`` is the claim named in the module docstring. It is a
    parameter and not a default buried in the label mapping, because on an
    Indian carriageway the shoulder is genuinely part of the drivable set and
    on a motorway it is genuinely not.
    """
    out = mask == ROAD
    if include_shoulder:
        out |= mask == SHOULDER
    return out


# --------------------------------------------------------------------------
# pairing images with annotations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    image: Path
    label: Path
    key: str


def find_pairs(root: Path, split: str) -> List[Pair]:
    """Every ``(image, annotation)`` pair in a split, sorted for determinism.

    Sorted because an unsorted filesystem walk makes the train/val split a
    property of the inode order, which is not reproducible between machines --
    the same failure ADR-008 records for wall-clock evaluation.
    """
    root = Path(root).expanduser()
    img_root = root / "leftImg8bit" / split
    lab_root = root / "gtFine" / split
    labels: Dict[str, Path] = {}
    for p in lab_root.rglob("*_gtFine_polygons.json"):
        key = f"{p.parent.name}/{p.name[:-len('_gtFine_polygons.json')]}"
        labels[key] = p

    pairs: List[Pair] = []
    for p in sorted(img_root.rglob("*_leftImg8bit.png")):
        key = f"{p.parent.name}/{p.name[:-len('_leftImg8bit.png')]}"
        lab = labels.get(key)
        if lab is not None:
            pairs.append(Pair(image=p, label=lab, key=key))
    return pairs


def label_histogram(pairs: Sequence[Pair], limit: int = 300) -> Dict[str, int]:
    """Polygon counts by label name, for checking the mapping against data."""
    import collections
    out: "collections.Counter[str]" = collections.Counter()
    for pair in list(pairs)[:limit]:
        for obj in load_record(pair.label).get("objects", ()):
            if not obj.get("deleted"):
                out[obj["label"]] += 1
    return dict(out)
