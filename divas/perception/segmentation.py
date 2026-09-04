"""Stage 1 -- real perception, not a ground-truth stub.

Every other stage in this repo has run against :mod:`divas.sim.world`'s
perfect segmentation and perfect tracks.  This module is the first thing
that takes an actual image and produces a drivable-area mask from pixels,
not from simulator state.

No model is trained here -- that is Phase 2 proper, and needs the IDD
dataset and a GPU box, neither of which this machine has.  What this gives
instead: a real pretrained semantic-segmentation network (SegFormer-B0,
Cityscapes weights, off the shelf) doing real CPU inference on a real
frame, with its output remapped to the same binary drivable mask contract
the rest of the stack expects.  It will not be well-tuned to Indian roads
-- Cityscapes is a Western driving dataset, which is exactly the mismatch
this whole project argues against -- but it proves the perception stage
is a real pixel-in, mask-out module rather than simulator state relabelled.

Swap-in path once IDD training lands: same ``predict(image) -> mask``
signature, different weights.  Nothing downstream changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: Cityscapes trainId -> class, for the checkpoint this module loads.
#: Index is the trainId the model outputs per pixel.
CITYSCAPES_TRAIN_IDS = (
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle",
    "bicycle",
)

#: Only the paved carriageway.  Mirrors the DRIVABLE_TAGS /
#: DRIVABLE_TAGS_WITH_SHOULDER split in divas/sim/carla_bridge.py --
#: "sidewalk" and "terrain" are not fair game even though they are flat.
DRIVABLE_CLASSES = ("road",)

_MODEL_ID = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"


@dataclass
class DrivableSegmenter:
    """Wraps a pretrained SegFormer checkpoint as a drivable-mask predictor.

    CPU-only by construction -- this machine has no GPU, and a hackathon
    demo should not depend on one being available at pitch time either.
    """

    drivable_classes: Sequence[str] = DRIVABLE_CLASSES
    device: str = "cpu"

    def __post_init__(self) -> None:
        # Imported lazily so importing this module never requires torch
        # unless a segmenter is actually constructed -- the rest of divas
        # has no torch dependency and should stay that way.
        import torch
        from transformers import (
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )

        self._torch = torch
        self.processor = SegformerImageProcessor.from_pretrained(_MODEL_ID)
        self.model = SegformerForSemanticSegmentation.from_pretrained(_MODEL_ID)
        self.model.to(self.device).eval()
        self._drivable_ids = [
            CITYSCAPES_TRAIN_IDS.index(c) for c in self.drivable_classes
        ]

    def predict(self, image: np.ndarray) -> np.ndarray:
        """RGB ``(H, W, 3)`` uint8 -> boolean drivable mask, ``(H, W)``.

        Output is resized back to the input resolution with nearest-
        neighbour interpolation, so mask edges stay class boundaries rather
        than being smoothed into some intermediate label.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) RGB, got {image.shape}")
        h, w = image.shape[:2]

        inputs = self.processor(images=image, return_tensors="pt")
        with self._torch.no_grad():
            logits = self.model(**inputs).logits  # (1, C, h', w')

        upsampled = self._torch.nn.functional.interpolate(
            logits, size=(h, w), mode="bilinear", align_corners=False
        )
        class_map = upsampled.argmax(dim=1)[0].cpu().numpy()  # (H, W)
        return np.isin(class_map, self._drivable_ids)

    def drivable_fraction(self, image: np.ndarray) -> float:
        mask = self.predict(image)
        return float(mask.mean())
