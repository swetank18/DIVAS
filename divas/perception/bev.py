"""Stage 2: image-plane free space -> a bird's-eye occupancy grid.

Stage 1 says which *pixels* are drivable. The planner needs to know which
*metres* are. This module is the projection between them, and it is the point
at which a photograph becomes something Hybrid A* can search.

**The flat-ground assumption, stated up front.** Every pixel is projected onto
the plane ``z = 0`` in the vehicle frame. That is exact for a level road and
wrong everywhere else, and the error is not symmetric: an upslope makes free
space appear *further away* than it is, a downslope makes it appear nearer.
The magnitude grows with range -- a 2 degree grade misplaces a point at 30 m by
roughly a metre -- which is why the grid is truncated well before the horizon
rather than filled to the vanishing point. Nothing here estimates the ground
plane; that is stereo or LiDAR work and it is honest future work, not a
detail.

**Why the inverse direction.** The mapping runs BEV cell -> image pixel, not
image pixel -> BEV cell. Projecting pixels forward scatters them: near the
horizon one pixel covers many metres and leaves the far field full of holes,
while the foreground writes the same cell hundreds of times. Sampling each
grid cell exactly once gives complete coverage and makes the cost independent
of image resolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from divas.types import OccupancyGrid


@dataclass(frozen=True)
class Camera:
    """Pinhole intrinsics plus mounting, in the vehicle frame.

    Vehicle frame is the stack's own: ``x`` forward, ``y`` left, ``z`` up, with
    the origin on the ground under the rear axle.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    height: float = 1.35          # m above the road
    pitch: float = 0.0            # rad, positive = nose down
    x_offset: float = 1.5         # m forward of the origin

    @staticmethod
    def from_fov(width: int, height_px: int, fov_deg: float = 90.0,
                 **kw) -> "Camera":
        """Intrinsics from a horizontal field of view.

        The usual fallback when a dataset ships no calibration -- which IDD
        does not. Anything derived this way carries a *nominal* scale: the
        geometry is self-consistent and the absolute metres are only as good
        as the assumed FOV and mounting height. Say so wherever the number is
        quoted.
        """
        f = 0.5 * width / math.tan(0.5 * math.radians(fov_deg))
        return Camera(fx=f, fy=f, cx=0.5 * width, cy=0.5 * height_px, **kw)


@dataclass(frozen=True)
class BevSpec:
    """The patch of ground to rasterise, in metres."""

    forward: float = 32.0         # m ahead of the camera
    behind: float = 4.0           # m behind, so the ego's own footprint exists
    lateral: float = 16.0         # m either side
    resolution: float = 0.25      # m per cell, matching ADR-001
    #: Half-width of the near-field corridor assumed drivable, metres. See
    #: :func:`min_visible_range` for why it is needed at all.
    near_field_halfwidth: float = 1.6

    @property
    def shape(self) -> Tuple[int, int]:
        ny = int(round(2 * self.lateral / self.resolution))
        nx = int(round((self.forward + self.behind) / self.resolution))
        return ny, nx


def project(cam: Camera, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ground points ``(x, y)`` in the vehicle frame -> pixel ``(u, v, valid)``.

    ``valid`` is False where the point falls behind the camera, which must be
    masked rather than divided by: a point behind the lens has a negative
    depth and projects to a perfectly plausible-looking pixel in front of it.
    """
    xc = -(y)                                   # camera x is to the right
    yc = cam.height                             # ground is below the camera
    zc = x - cam.x_offset                       # camera z is forward

    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    yr = yc * c - zc * s
    zr = yc * s + zc * c

    valid = zr > 1e-3
    zsafe = np.where(valid, zr, 1.0)
    u = cam.cx + cam.fx * (xc / zsafe)
    v = cam.cy + cam.fy * (yr / zsafe)
    return u, v, valid


def free_space_to_grid(
    drivable: np.ndarray,
    cam: Camera,
    spec: Optional[BevSpec] = None,
    unknown_is_occupied: bool = True,
) -> OccupancyGrid:
    """A boolean image-plane free-space mask -> an ego-centred occupancy grid.

    ``unknown_is_occupied`` decides what happens to ground that the camera
    cannot see -- behind the vehicle, outside the frustum, or beyond the
    truncation range. It defaults to occupied, which is the only safe reading:
    "I have no evidence this is drivable" and "this is drivable" are different
    statements, and a planner that confuses them will happily route through
    the blind spot beside the vehicle.
    """
    spec = spec or BevSpec()
    ny, nx = spec.shape
    h, w = drivable.shape[:2]

    xs = -spec.behind + (np.arange(nx) + 0.5) * spec.resolution
    ys = -spec.lateral + (np.arange(ny) + 0.5) * spec.resolution
    gx, gy = np.meshgrid(xs, ys)

    u, v, valid = project(cam, gx, gy)
    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    inside = valid & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

    np.clip(ui, 0, w - 1, out=ui)
    np.clip(vi, 0, h - 1, out=vi)
    sampled = drivable[vi, ui] & inside

    grid = OccupancyGrid.empty(
        (float(xs[0] - 0.5 * spec.resolution), float(xs[-1] + 0.5 * spec.resolution)),
        (float(ys[0] - 0.5 * spec.resolution), float(ys[-1] + 0.5 * spec.resolution)),
        spec.resolution,
    )
    occupied = ~sampled if unknown_is_occupied else ~(sampled | ~inside)

    # Bridge the near-field blind spot: the strip from the ego to the closest
    # ground the camera can see. Kept to a corridor about a vehicle wide, so
    # it cannot invent free space out to the sides where the planner might
    # actually want to go.
    near = min_visible_range(cam, h)
    if np.isfinite(near) and near > 0.0:
        bridge = (gx <= near) & (np.abs(gy) <= spec.near_field_halfwidth)
        occupied &= ~bridge

    grid.data = occupied.astype(np.float32)
    grid.invalidate_cache()
    return grid


def min_visible_range(cam: Camera, image_height: int) -> float:
    """Nearest ground distance the camera can see, metres.

    A forward-facing camera cannot see the road under its own bumper: the
    ground between the vehicle and this range falls below the bottom edge of
    the frame. Projecting a single photograph therefore leaves a wedge of
    "unseen" directly ahead, and since unseen is treated as occupied, **the
    planner starts inside an obstacle and every plan fails**. That is exactly
    what the first run of the demo did -- every plan came back partial, with
    nothing drawn.

    A vehicle in motion resolves this by carrying previous frames forward; a
    still image has no previous frame. :func:`free_space_to_grid` therefore
    takes the corridor from the ego to this range as drivable, on the
    reasoning that the vehicle is standing on it. That is an assumption and it
    is narrow -- a corridor barely wider than the car -- rather than a blanket
    clearing of the near field.
    """
    dv = ((image_height - 1) - cam.cy) / cam.fy
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    denom = dv * c + s
    if denom <= 1e-6:
        return float("inf")
    return float(cam.height * (c - dv * s) / denom + cam.x_offset)


def horizon_row(cam: Camera) -> float:
    """Image row of the horizon, where the flat-ground mapping diverges.

    Any pixel at or above this row projects to infinity or behind the camera.
    Sampling there produces free space stretching to the edge of the grid from
    a few pixels of sky, which looks like an enormous open road and is the
    most dangerous artefact this projection can produce.
    """
    return cam.cy - cam.fy * math.tan(cam.pitch) if abs(cam.pitch) < 1.5 else cam.cy


def max_reliable_range(cam: Camera, image_height: int,
                       margin_rows: int = 8) -> float:
    """Furthest ground distance worth trusting, metres.

    Derived from the row ``margin_rows`` below the horizon: closer to it than
    that, one row of pixels spans tens of metres and the projection is noise.
    Use it to set :attr:`BevSpec.forward` rather than guessing.
    """
    v = horizon_row(cam) + margin_rows
    if v >= image_height:
        v = image_height - 1.0
    dv = (v - cam.cy) / cam.fy
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    # Inverting the forward map. With yc = height and zc = range:
    #     dv = (yc*c - zc*s) / (yc*s + zc*c)
    # solving for zc gives the expression below. Getting the two signs the
    # other way round -- which is easy, and which this function did at first
    # -- makes the denominator negative for every row above the horizon and
    # the range comes back as infinity, i.e. "trust everything".
    denom = dv * c + s
    if denom <= 1e-6:
        return float("inf")
    zc = cam.height * (c - dv * s) / denom
    return float(zc + cam.x_offset)
