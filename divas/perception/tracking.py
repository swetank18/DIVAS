"""Turn real detections into world-frame tracks the predictor can consume.

Closes the gap flagged in ``divas/perception/detection.py``: an
:class:`~divas.perception.detection.ObjectDetector` box is in image
pixels, and :class:`~divas.prediction.predictors.SocialForcePredictor`
needs a :class:`divas.types.Track` in vehicle-frame metres with a
velocity. This module is the bridge -- image box -> ground point (via the
flat-ground inverse of :mod:`divas.perception.bev`) -> a track, with
velocity estimated from the same actor's position in a previous frame
when one is given.

**Ground-point assumption.** The point where a box touches the road --
its bottom-centre pixel -- is projected onto ``z = 0``. Wrong for a
motorcycle leant over or a box that doesn't reach the object's true
contact point, same flat-ground caveat as ``bev.py`` generally: exact for
a level road, and error grows with range.

**No calibration.** Absent real camera intrinsics, ``Camera.from_fov`` is
used the same way the rest of the perception stack does for IDD frames --
nominal geometry, self-consistent, only as accurate as the assumed FOV
and mounting height. Say so wherever a resulting metre figure is quoted.

**Single photo has no velocity.** With one frame, every track gets
``vx = vy = 0`` -- there is nothing to compute a velocity from, and
inventing one would be worse than admitting there isn't one. Pass a
previous frame's detections (``prev`` below) to get a real, measured
velocity from actual displacement over a known ``dt``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from divas.perception.bev import Camera
from divas.perception.detection import Detection
from divas.types import Track


def pixel_to_ground(cam: Camera, u: float, v: float) -> Optional[Tuple[float, float]]:
    """Inverse of ``bev.project``: image pixel -> vehicle-frame ground point.

    ``None`` where the pixel is at or above the horizon -- there the
    flat-ground mapping diverges (see ``bev.horizon_row``) and any
    "distance" returned would be fiction, not a far-away point.
    """
    dv = (v - cam.cy) / cam.fy
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    denom = dv * c + s
    if denom <= 1e-6:
        return None
    zc = cam.height * (c - dv * s) / denom
    x = zc + cam.x_offset
    if x <= 0.0:
        return None
    zr = cam.height * s + zc * c
    du = (u - cam.cx) / cam.fx
    xc = du * zr
    y = -xc
    return float(x), float(y)


@dataclass
class GroundDetection:
    """A detection after projection: class + world-frame ground point."""

    divas_class: str
    x: float
    y: float
    confidence: float


def detections_to_ground(detections: List[Detection], cam: Camera) -> List[GroundDetection]:
    """Project each box's bottom-centre pixel to a ground point.

    Boxes whose bottom edge is above the horizon (a bird mid-frame, a
    detection artefact) are dropped -- ``pixel_to_ground`` returning
    ``None`` is the signal, not an error to work around.
    """
    out = []
    for d in detections:
        x1, y1, x2, y2 = d.box_xyxy
        u = 0.5 * (x1 + x2)
        v = y2  # bottom edge -- where the object meets the road
        g = pixel_to_ground(cam, u, v)
        if g is not None:
            out.append(GroundDetection(d.divas_class, g[0], g[1], d.confidence))
    return out


def _match(prev: List[GroundDetection], cur: List[GroundDetection],
           max_dist: float = 4.0) -> List[Tuple[int, int]]:
    """Greedy nearest-neighbour match, same class only, within ``max_dist``.

    Not a real tracker -- no Hungarian assignment, no occlusion handling,
    no ID persistence across more than two frames. Good enough to turn
    "two photos a moment apart" into a velocity estimate; a real deployment
    needs the EKF tracker that's still Phase 3 stub work, not this.
    """
    pairs = []
    used_cur = set()
    for i, p in enumerate(prev):
        best_j, best_d = None, max_dist
        for j, c in enumerate(cur):
            if j in used_cur or c.divas_class != p.divas_class:
                continue
            d = math.hypot(c.x - p.x, c.y - p.y)
            if d < best_d:
                best_j, best_d = j, d
        if best_j is not None:
            pairs.append((i, best_j))
            used_cur.add(best_j)
    return pairs


def ground_to_tracks(
    cur: List[GroundDetection],
    prev: Optional[List[GroundDetection]] = None,
    dt: float = 0.5,
) -> List[Track]:
    """Ground-projected detections -> :class:`Track` list, velocity if possible.

    With no ``prev``, every track is stationary (``vx = vy = 0``) -- see
    the module docstring on why that's the honest choice for a single
    photo rather than a guessed velocity.
    """
    tracks: List[Track] = []
    matched_cur = set()
    if prev is not None:
        for i, (pi, ci) in enumerate(_match(prev, cur)):
            p, c = prev[pi], cur[ci]
            vx = (c.x - p.x) / dt
            vy = (c.y - p.y) / dt
            tracks.append(Track(id=i, x=c.x, y=c.y, vx=vx, vy=vy, cls=c.divas_class))
            matched_cur.add(ci)
    next_id = len(tracks)
    for j, c in enumerate(cur):
        if j in matched_cur:
            continue
        tracks.append(Track(id=next_id, x=c.x, y=c.y, vx=0.0, vy=0.0, cls=c.divas_class))
        next_id += 1
    return tracks
