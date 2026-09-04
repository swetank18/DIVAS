"""India Driving Dataset (IDD) -- loading and drivable-area remapping.

IDD is the reason this project trains on Indian roads rather than on KITTI or
Cityscapes, and one label in particular is why it is the right choice:
**drivable fallback**.  Western datasets have no class for the unpaved shoulder
that Indian traffic uses as a matter of course -- it is simply "not road" to
them.  IDD labels it, and collapsing it into the drivable class is precisely
the free-space definition this stack plans over.

Phase 0 exists to check assumptions rather than inherit them, so nothing here
is hard-coded silently:

* ``DRIVABLE_LEVEL3_IDS`` is a *hypothesis* about IDD's level-3 label ids.
  It must be checked against the ``idd_labels`` definition shipped with the
  copy of the dataset actually downloaded -- id assignments differ between
  IDD-Lite, IDD-20k and IDD-40k releases.  :func:`verify` reports what it
  finds so the hypothesis can be confirmed or corrected before any training
  run depends on it.
* The public IDD segmentation release is single-image: no radar, no ego-motion,
  no tracked sequences.  It can train stage 1 and it cannot train stage 4.
  That is the open item flagged in ``PROJECT_OVERVIEW.md`` section 7.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is present in this environment
    cv2 = None


# Level-3 label ids believed to be traversable.  VERIFY IN PHASE 0.
DRIVABLE_LEVEL3_IDS: Tuple[int, ...] = (0, 1, 2)     # road, parking, drivable fallback
DRIVABLE_LEVEL3_NAMES: Tuple[str, ...] = ("road", "parking", "drivable fallback")

# Level-1 (IDD-Lite) uses a much coarser grouping; 0 is the drivable group.
DRIVABLE_LEVEL1_IDS: Tuple[int, ...] = (0,)


@dataclass
class IDDConfig:
    root: Path
    split: str = "train"
    level: int = 3                    # 3 for IDD-20k/40k, 1 for IDD-Lite
    image_dir: str = "leftImg8bit"
    label_dir: str = "gtFine"
    image_suffix: str = "_leftImg8bit.png"
    label_suffix: str = "_gtFine_labellevel3Ids.png"

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser()

    @property
    def drivable_ids(self) -> Tuple[int, ...]:
        return DRIVABLE_LEVEL3_IDS if self.level == 3 else DRIVABLE_LEVEL1_IDS


@dataclass
class IDDDrivableDataset:
    """Pairs of (image, binary drivable mask).

    Returns numpy arrays, not tensors.  Stage 1 will wrap this in a torch
    ``Dataset``; keeping the loader framework-free means the data pipeline can
    be verified in Phase 0 before torch is even installed.
    """

    cfg: IDDConfig
    pairs: List[Tuple[Path, Path]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.pairs:
            self.pairs = self._discover()

    def _discover(self) -> List[Tuple[Path, Path]]:
        c = self.cfg
        img_root = c.root / c.image_dir / c.split
        lbl_root = c.root / c.label_dir / c.split
        if not img_root.is_dir():
            return []
        out: List[Tuple[Path, Path]] = []
        for img in sorted(img_root.rglob(f"*{c.image_suffix}")):
            stem = img.name[: -len(c.image_suffix)]
            lbl = lbl_root / img.parent.name / f"{stem}{c.label_suffix}"
            if lbl.is_file():
                out.append((img, lbl))
        return out

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        if cv2 is None:
            raise RuntimeError("opencv is required to read IDD images")
        img_path, lbl_path = self.pairs[i]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError(f"could not read {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ids = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)
        if ids is None:
            raise IOError(f"could not read {lbl_path}")
        return img, to_drivable_mask(ids, self.cfg.drivable_ids)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        for i in range(len(self)):
            yield self[i]


def to_drivable_mask(label_ids: np.ndarray, drivable: Sequence[int]) -> np.ndarray:
    """Collapse a level-N id map to a binary free-space mask.

    This one line is the whole architectural bet: 26 semantic classes reduce to
    the only question the planner asks, which is whether the vehicle may put a
    wheel there.
    """
    ids = np.asarray(label_ids)
    if ids.ndim == 3:
        ids = ids[..., 0]
    return np.isin(ids, np.asarray(drivable)).astype(np.uint8)


def class_histogram(ds: IDDDrivableDataset, limit: int = 200) -> Dict[int, int]:
    """Pixel count per raw label id over the first ``limit`` frames."""
    if cv2 is None:
        raise RuntimeError("opencv is required")
    hist: Dict[int, int] = {}
    for _, lbl_path in ds.pairs[:limit]:
        ids = cv2.imread(str(lbl_path), cv2.IMREAD_UNCHANGED)
        if ids is None:
            continue
        if ids.ndim == 3:
            ids = ids[..., 0]
        vals, counts = np.unique(ids, return_counts=True)
        for v, n in zip(vals.tolist(), counts.tolist()):
            hist[int(v)] = hist.get(int(v), 0) + int(n)
    return dict(sorted(hist.items()))


def verify(root: Path, level: int = 3, sample: int = 50) -> dict:
    """Phase 0 check: is IDD present, readable, and labelled as assumed?

    Reports rather than asserts.  The label-id hypothesis in
    :data:`DRIVABLE_LEVEL3_IDS` is confirmed by looking at the histogram it
    produces: if the "drivable" fraction of a road-scene dataset comes out at
    2% or 95%, the mapping is wrong, not the dataset.
    """
    root = Path(root).expanduser()
    report: dict = {"root": str(root), "exists": root.is_dir(), "splits": {}}
    if not report["exists"]:
        report["error"] = (
            "IDD not found. Download from idd.insaan.iiit.ac.in and point "
            "--root at the directory containing leftImg8bit/ and gtFine/."
        )
        return report

    for split in ("train", "val", "test"):
        cfg = IDDConfig(root=root, split=split, level=level)
        if level == 1:
            cfg.label_suffix = "_gtFine_labellevel1Ids.png"
        ds = IDDDrivableDataset(cfg)
        entry: dict = {"frames": len(ds)}
        if len(ds) and cv2 is not None:
            fracs = []
            for i in range(min(sample, len(ds))):
                try:
                    _, mask = ds[i]
                except (IOError, RuntimeError):
                    continue
                fracs.append(float(mask.mean()))
            if fracs:
                entry["drivable_fraction_mean"] = round(float(np.mean(fracs)), 4)
                entry["drivable_fraction_range"] = [
                    round(float(np.min(fracs)), 4),
                    round(float(np.max(fracs)), 4),
                ]
                entry["plausible"] = 0.10 <= float(np.mean(fracs)) <= 0.70
            entry["label_ids_present"] = sorted(class_histogram(ds, limit=20))
        report["splits"][split] = entry
    return report
