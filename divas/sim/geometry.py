"""Geometry shared by every simulator backend.

The built-in world and the CARLA bridge both report collisions, clearance and
time-to-collision. If each implemented that arithmetic itself the two would
drift, and then a number measured in CARLA could not be compared with the same
number measured in the built-in simulator -- which is the whole reason for
having both.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

#: An actor as the metrics see it: ``(x, y, yaw, half_len, half_wid, v, label)``
Box = Tuple[float, float, float, float, float, float, str]


def oriented_box_sdf(px, py, x: float, y: float, yaw: float,
                     hl: float, hw: float) -> np.ndarray:
    """Signed distance to an oriented rectangle. Negative inside."""
    c, s = math.cos(-yaw), math.sin(-yaw)
    dx = np.asarray(px) - x
    dy = np.asarray(py) - y
    lx = c * dx - s * dy
    ly = s * dx + c * dy
    qx = np.abs(lx) - hl
    qy = np.abs(ly) - hw
    outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
    return outside + np.minimum(np.maximum(qx, qy), 0.0)


def footprint_discs(ego, params) -> Tuple[np.ndarray, float]:
    """Ego footprint disc centres in world coordinates, and their radius."""
    offs, r = params.footprint_discs()
    c, s = math.cos(ego.theta), math.sin(ego.theta)
    return np.stack([ego.x + offs * c, ego.y + offs * s], axis=1), r


def min_clearance(ego, params, boxes: Sequence[Box]) -> float:
    """Smallest surface-to-surface gap between ego and any actor, metres."""
    centres, r = footprint_discs(ego, params)
    best = np.inf
    for x, y, yaw, hl, hw, _v, _lab in boxes:
        d = float(oriented_box_sdf(centres[:, 0], centres[:, 1], x, y, yaw, hl, hw).min())
        best = min(best, d - r)
    return best


def first_hit(ego, params, boxes: Sequence[Box]) -> Optional[str]:
    """Label of the first actor the ego footprint overlaps, or ``None``."""
    centres, r = footprint_discs(ego, params)
    for x, y, yaw, hl, hw, _v, label in boxes:
        if float(oriented_box_sdf(centres[:, 0], centres[:, 1], x, y, yaw, hl, hw).min()) <= r:
            return label
    return None


def time_to_collision(ego, params, boxes: Sequence[Box]) -> float:
    """Constant-velocity TTC against the nearest closing actor.

    Against an oriented **ellipse** in the actor's heading frame, not a
    circumscribed disc. With discs, an ego of 1.07 m radius plus a truck of
    3.7 m demand 4.8 m of separation, so passing an oncoming truck three
    metres to the side scores as an imminent collision and the headline safety
    metric reads near zero on runs that were never in danger.
    """
    best = np.inf
    evx, evy = ego.v * math.cos(ego.theta), ego.v * math.sin(ego.theta)
    ehl, ehw = params.half_extent
    for x, y, yaw, hl, hw, v, _lab in boxes:
        c, s = math.cos(-yaw), math.sin(-yaw)
        dx, dy = ego.x - x, ego.y - y
        rx = c * dx - s * dy
        ry = s * dx + c * dy
        dvx, dvy = evx - v * math.cos(yaw), evy - v * math.sin(yaw)
        rvx = c * dvx - s * dvy
        rvy = s * dvx + c * dvy
        A, B = hl + ehl, hw + ehw
        qa = (rvx / A) ** 2 + (rvy / B) ** 2
        qb = 2 * (rx * rvx / A**2 + ry * rvy / B**2)
        qc = (rx / A) ** 2 + (ry / B) ** 2 - 1.0
        if qa < 1e-12 or qc <= 0.0:
            continue
        disc = qb * qb - 4 * qa * qc
        if disc < 0.0:
            continue
        t = (-qb - math.sqrt(disc)) / (2 * qa)
        if t > 0.0:
            best = min(best, float(t))
    return best
