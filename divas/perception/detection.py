"""Stage 1 -- real object detection: boxes, classes, confidence.

:mod:`divas.perception.segmentation` answers "which pixels are road."
This answers "what is on the road, and where" -- car, truck, bus,
motorcycle, bicycle, pedestrian, animal -- as image-space boxes, which is
what ``PROJECT_OVERVIEW.md`` calls the ``Detections[]`` half of stage 1's
output.

No training here either, same reasoning as the segmenter: a pretrained
YOLOv8n (COCO classes) doing real CPU inference on a real frame, remapped
to this project's class vocabulary (:data:`divas.types.CLASS_EXTENT`).

**What this is not**: these boxes are in image pixels, not world metres.
Turning a box into a :class:`divas.types.Track` needs depth or a
camera-to-ground homography -- that is stage 2, BEV projection, and it is
still a ground-truth stub (see ``CONTEXT.md``). This module stops at
"here is a labelled box in the image," which is the honest boundary of
what a single frame without depth can give you.

**What COCO cannot give you**: there is no "autorickshaw" class in COCO.
That gap is real and worth saying out loud in a demo -- it is exactly the
kind of Western-dataset mismatch this project's own thesis is about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

#: COCO class name -> this project's actor class vocabulary
#: (divas.types.CLASS_EXTENT). Anything not listed here is either not
#: road-relevant (COCO has 80 classes, most are indoor objects) or has no
#: honest equivalent and is dropped rather than mapped to something wrong.
COCO_TO_DIVAS_CLASS = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "person": "pedestrian",
    # COCO's animal classes -- all fold to "animal", the closest this
    # project's vocabulary has. A cow on an Indian road and a COCO
    # "cow" detection are at least the same species; a dog is not, but
    # "animal in the road" is the operative fact for the risk field.
    "dog": "animal",
    "cat": "animal",
    "horse": "animal",
    "sheep": "animal",
    "cow": "animal",
    "elephant": "animal",
    "bear": "animal",
}

#: Only keep detections whose class is road-relevant. COCO's other 65
#: classes (chair, laptop, pizza, ...) are real detections but noise here.
_RELEVANT = frozenset(COCO_TO_DIVAS_CLASS)

_WEIGHTS = "yolov8n.pt"


@dataclass
class Detection:
    """One detected actor, in image pixels."""

    box_xyxy: np.ndarray   # (4,) float: x1, y1, x2, y2 in pixel coords
    coco_class: str
    divas_class: str       # remapped via COCO_TO_DIVAS_CLASS
    confidence: float

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.box_xyxy
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])


@dataclass
class ObjectDetector:
    """Pretrained YOLOv8n wrapped to this project's class vocabulary.

    CPU-only, same reasoning as :class:`~divas.perception.segmentation.DrivableSegmenter`:
    this machine has no GPU, and a hackathon demo should not depend on one.
    """

    confidence_threshold: float = 0.35
    classes: Optional[Sequence[str]] = None  # divas classes to keep; None = all mapped

    def __post_init__(self) -> None:
        # Lazy import: divas has no ultralytics/torch dependency unless a
        # detector is actually constructed.
        from ultralytics import YOLO

        self.model = YOLO(_WEIGHTS)

    def predict(self, image: np.ndarray) -> List[Detection]:
        """RGB ``(H, W, 3)`` uint8 -> list of :class:`Detection`.

        Runs at COCO's native 80-class output, then drops and remaps down
        to this project's vocabulary -- an indoor-object false positive
        (e.g. "chair") is discarded here rather than reported as noise
        downstream.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected (H, W, 3) RGB, got {image.shape}")

        results = self.model.predict(
            image, conf=self.confidence_threshold, verbose=False
        )[0]

        out: List[Detection] = []
        names = results.names
        for box in results.boxes:
            coco_cls = names[int(box.cls[0])]
            if coco_cls not in _RELEVANT:
                continue
            divas_cls = COCO_TO_DIVAS_CLASS[coco_cls]
            if self.classes is not None and divas_cls not in self.classes:
                continue
            out.append(
                Detection(
                    box_xyxy=box.xyxy[0].cpu().numpy(),
                    coco_class=coco_cls,
                    divas_class=divas_cls,
                    confidence=float(box.conf[0]),
                )
            )
        return out

    def counts(self, image: np.ndarray) -> dict:
        """``{divas_class: count}`` -- the quick summary for a demo."""
        dets = self.predict(image)
        out: dict = {}
        for d in dets:
            out[d.divas_class] = out.get(d.divas_class, 0) + 1
        return out
