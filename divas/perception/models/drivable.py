"""Stage 1: drivable-area segmentation.

The stack plans over free space, and until now the free space came from the
simulator. This is the module that replaces that stub with a network that
looks at a photograph.

**Why this architecture.** LR-ASPP over a MobileNetV3-Large backbone (Howard
et al., ICCV 2019), fine-tuned from ImageNet weights. Three reasons, in order
of how much they mattered:

* It is designed for exactly this job -- dense semantic segmentation at
  real-time rates on a device with no headroom. The stack budgets stage 1 at
  10 Hz alongside a 20 Hz control loop, on a laptop 3050 today and a Jetson
  later, and a ResNet-101 DeepLab does not fit that budget at any resolution
  worth using.
* 3.2 M parameters trains to convergence on 7,000 images without the
  regularisation gymnastics a larger head would need on a dataset this size.
* Fine-tuning from ImageNet is what makes 7,000 images enough at all. Training
  from scratch on IDD alone would spend most of the data learning edges.

**Three classes, not two.** ``road``, ``drivable fallback`` and everything
else. The shoulder is a separate output because whether it counts as drivable
is a *decision*, and the stack already draws that distinction on the simulator
side -- ``DRIVABLE_TAGS`` against ``DRIVABLE_TAGS_WITH_SHOULDER``. Collapsing
it in the label would hard-code the answer where nobody could see it, and on a
sampled IDD frame the shoulder is the *larger* of the two: 26.0% against 19.5%
for the carriageway.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    from torch import nn
    HAVE_TORCH = True
except ImportError:                                   # pragma: no cover
    torch = None
    nn = object
    HAVE_TORCH = False

from divas.perception.datasets.idd_polygons import OTHER, ROAD, SHOULDER

N_CLASSES = 3
#: Normalisation the ImageNet backbone was trained with. Getting this wrong
#: costs several points of IoU and looks exactly like a bad learning rate.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(n_classes: int = N_CLASSES, pretrained: bool = True):
    """LR-ASPP MobileNetV3-Large with a ``n_classes`` head."""
    if not HAVE_TORCH:                                # pragma: no cover
        raise RuntimeError("stage 1 needs torch; see sim/README or STATUS.md")
    from torchvision.models.segmentation import (
        LRASPP_MobileNet_V3_Large_Weights,
        lraspp_mobilenet_v3_large,
    )
    weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
    model = lraspp_mobilenet_v3_large(weights=weights)
    # The pretrained head predicts 21 COCO/VOC classes. Replacing both branches
    # rather than slicing the old head: the classes are unrelated, so the
    # inherited weights are not a warm start, they are noise with a prior.
    low = model.classifier.low_classifier.in_channels
    high = model.classifier.high_classifier.in_channels
    model.classifier.low_classifier = nn.Conv2d(low, n_classes, 1)
    model.classifier.high_classifier = nn.Conv2d(high, n_classes, 1)
    return model


def normalise(img: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> CHW float32, ImageNet-normalised."""
    x = img.astype(np.float32) / 255.0
    x = (x - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / np.asarray(
        IMAGENET_STD, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


@dataclass
class DrivableSegmenter:
    """Inference wrapper -- the object stage 1 hands to stage 2.

    Deliberately returns the *class map* and lets the caller decide what counts
    as drivable, rather than returning a boolean. See the module docstring.
    """

    model: object
    device: str = "cuda"
    size: Tuple[int, int] = (512, 288)          # (w, h)

    @staticmethod
    def load(checkpoint: Path, device: str = "cuda",
             size: Tuple[int, int] = (512, 288)) -> "DrivableSegmenter":
        model = build_model(pretrained=False)
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"] if "model" in state else state)
        model.eval().to(device)
        return DrivableSegmenter(model=model, device=device, size=size)

    @torch.no_grad() if HAVE_TORCH else staticmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """RGB uint8 ``(H, W, 3)`` -> class map ``(H, W)`` uint8, input size."""
        import cv2
        h0, w0 = image.shape[:2]
        small = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(normalise(small))[None].to(self.device)
        logits = self.model(x)["out"]
        pred = logits.argmax(1)[0].to(torch.uint8).cpu().numpy()
        # Upsampled nearest, because this is a label map: interpolating between
        # class ids 0 and 2 would invent class 1 along every boundary.
        return cv2.resize(pred, (w0, h0), interpolation=cv2.INTER_NEAREST)

    def free_space(self, image: np.ndarray, include_shoulder: bool = True) -> np.ndarray:
        m = self(image)
        out = m == ROAD
        if include_shoulder:
            out |= m == SHOULDER
        return out


def confusion(pred: np.ndarray, target: np.ndarray, n: int = N_CLASSES,
              ignore: int = 255) -> np.ndarray:
    """``(n, n)`` counts, rows = truth. Ignore-labelled pixels are excluded."""
    keep = target != ignore
    p = pred[keep].astype(np.int64)
    t = target[keep].astype(np.int64)
    return np.bincount(t * n + p, minlength=n * n).reshape(n, n)


def iou_from_confusion(cm: np.ndarray) -> np.ndarray:
    """Per-class intersection over union.

    IoU rather than pixel accuracy, because the classes are wildly imbalanced:
    a model that predicts "not drivable" everywhere scores 55% accuracy on a
    typical IDD frame and is useless.
    """
    inter = np.diag(cm).astype(np.float64)
    union = cm.sum(1) + cm.sum(0) - np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / np.maximum(union, 1), np.nan)


def free_space_iou(cm: np.ndarray) -> float:
    """Binary drivable-vs-not IoU, collapsed from the three-class confusion.

    This is the number the planner actually consumes, and it is much higher
    than the three-class mean suggests -- 0.948 against 0.737 at epoch 18.
    The gap is not flattery, it is where the errors go: of shoulder pixels,
    88.2% are called shoulder and a further 5.3% are called road, so 93.5% are
    correctly seen as *some kind of drivable*. Only 6.4% are lost to
    non-drivable. Confusing the shoulder with the road costs the free-space
    mask nothing; confusing it with a wall costs everything, and the
    three-class metric charges the same for both.

    Reported alongside the per-class numbers rather than instead of them: the
    shoulder distinction still matters for whether the stack is *allowed* to
    use it, which is a policy decision downstream.
    """
    drivable = (ROAD, SHOULDER)
    inter = sum(cm[i, j] for i in drivable for j in drivable)
    total = cm.sum()
    truth_d = sum(cm[i].sum() for i in drivable)
    pred_d = sum(cm[:, j].sum() for j in drivable)
    union = truth_d + pred_d - inter
    return float(inter / union) if union else float("nan")
