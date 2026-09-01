"""CARLA bridge -- implements :class:`divas.sim.interface.SimWorld`.

Open decision #1 resolved to CARLA, so this replaces ``divas.sim.world`` as the
simulator without anything above stage 3 changing. That was the whole point of
ADR-005 and of the stage contracts.

**Hardware.** CARLA is Unreal Engine and wants a dedicated GPU with 6 GB or
more of VRAM. It will not run usefully on integrated graphics -- expect it to
refuse to start or to render at a couple of frames per second, which makes a
20 Hz control loop meaningless. Everything in this module is written to run on
such a machine; nothing in it needs one to be *reviewed* or unit tested, which
is why the CARLA calls are kept apart from the logic below.

**Version.** Written against the CARLA 0.9.15 Python API. The client package is
version-locked to the server: install the ``carla`` wheel that ships with the
simulator build you run, not whatever pip resolves to.

Layout of this file, deliberately:

* pure functions at the top -- rasterisation, control conversion, class
  mapping. No CARLA import reaches them, and the tests exercise them directly.
* :class:`CarlaWorld` at the bottom -- the part that talks to the simulator.

Untested bridge code is where projects like this die: the perception work
lands, the bridge has a sign error in the steering conversion, and two days go
into debugging the planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from divas.types import (
    CLASS_EXTENT,
    EgoState,
    OccupancyGrid,
    Track,
    VehicleParams,
)

try:                                            # pragma: no cover
    import carla                                # type: ignore
    HAVE_CARLA = True
except ImportError:                             # pragma: no cover
    carla = None
    HAVE_CARLA = False


# --------------------------------------------------------------------------
# pure logic -- no CARLA needed, and unit tested
# --------------------------------------------------------------------------

#: CARLA blueprint id fragments mapped to the classes this stack reasons about.
#: Order matters: the first fragment found in the blueprint id wins, so the
#: specific two-wheelers must be tested before the generic vehicle fallback.
BLUEPRINT_CLASS: Sequence[Tuple[str, str]] = (
    ("walker", "pedestrian"),
    ("harley", "motorcycle"),
    ("yamaha", "motorcycle"),
    ("kawasaki", "motorcycle"),
    ("vespa", "motorcycle"),
    ("gazelle", "bicycle"),
    ("diamondback", "bicycle"),
    ("crossbike", "bicycle"),
    ("carlamotors", "truck"),
    ("firetruck", "truck"),
    ("ambulance", "truck"),
    ("sprinter", "truck"),
    ("volkswagen.t2", "truck"),
    ("fusorosa", "bus"),
    ("bus", "bus"),
    ("truck", "truck"),
    ("vehicle", "car"),
)


def classify_blueprint(blueprint_id: str) -> str:
    """CARLA blueprint id -> one of our classes.

    CARLA has no auto-rickshaw, which is the single most consequential gap
    between its actor set and an Indian road. Until a custom asset exists,
    two-wheelers and small cars stand in for it, and the deck should not claim
    otherwise.
    """
    bid = blueprint_id.lower()
    for fragment, cls in BLUEPRINT_CLASS:
        if fragment in bid:
            return cls
    return "unknown"


def carla_to_odom(x: float, y: float, yaw_deg: float) -> Tuple[float, float, float]:
    """CARLA world coordinates -> this stack's odom frame.

    CARLA is **left-handed**: x forward, **y to the right**, z up, with yaw in
    degrees increasing clockwise seen from above. This stack is right-handed
    with y to the left and theta counter-clockwise in radians.

    So the conversion is a y-flip and a yaw negation. It is three lines and it
    is the single most likely place for this bridge to be silently wrong: get
    it backwards and everything still runs, the vehicle just steers the wrong
    way into obstacles that appear mirrored. Hence a dedicated function with a
    dedicated test rather than an inline expression at each call site.
    """
    return x, -y, -math.radians(yaw_deg)


def odom_to_carla(x: float, y: float, theta: float) -> Tuple[float, float, float]:
    """Inverse of :func:`carla_to_odom`. Returns ``(x, y, yaw_deg)``."""
    return x, -y, -math.degrees(theta)


def carla_vector_to_odom(vx: float, vy: float) -> Tuple[float, float]:
    """Velocity or any free vector, same handedness flip."""
    return vx, -vy


def control_from_command(
    accel: float,
    steer_rad: float,
    params: VehicleParams,
    max_steer_deg: float,
) -> Tuple[float, float, float]:
    """``(accel, steer)`` from stage 6 -> CARLA ``(throttle, steer, brake)``.

    CARLA takes a normalised steering command, not a road-wheel angle, and the
    normalisation constant is a property of the spawned vehicle -- read it from
    ``get_physics_control().wheels[0].max_steer_angle`` rather than assuming,
    because it differs by several degrees between blueprints and a wrong
    constant looks exactly like a badly tuned controller.

    The longitudinal mapping is deliberately crude: throttle and brake are
    proportional to the requested acceleration against the vehicle's limits.
    A faithful mapping needs the engine and brake force curves, and until that
    exists the tracking error belongs to this function, not to the MPC. This is
    the first thing to check if the ego undershoots its speed reference in
    CARLA but not in the built-in simulator.
    """
    max_steer_rad = math.radians(max(max_steer_deg, 1e-3))
    steer = float(np.clip(steer_rad / max_steer_rad, -1.0, 1.0))
    if accel >= 0.0:
        throttle = float(np.clip(accel / max(params.max_accel, 1e-3), 0.0, 1.0))
        brake = 0.0
    else:
        throttle = 0.0
        brake = float(np.clip(-accel / max(abs(params.min_accel), 1e-3), 0.0, 1.0))
    return throttle, steer, brake


@dataclass
class DrivableRaster:
    """A whole town's drivable area, rasterised once.

    Querying ``Map.get_waypoint`` per cell per tick is hopeless -- a 64 x 64 m
    window at 0.25 m is 65k queries every 100 ms. The road network does not
    move, so it is rasterised once at connect time and the rolling window is
    then a crop.
    """

    origin: Tuple[float, float]
    resolution: float
    mask: np.ndarray                    # (ny, nx) bool, True = drivable

    @staticmethod
    def from_points(
        points: np.ndarray,
        widths: np.ndarray,
        resolution: float = 0.25,
        pad: float = 12.0,
    ) -> "DrivableRaster":
        """Rasterise from waypoint centres and their lane widths.

        ``points`` is (N, 2) in world coordinates; ``widths`` is (N,) full lane
        widths. Each waypoint paints a disc of radius ``width / 2``, which
        scallops the edges slightly at coarse sampling and is why the caller
        should generate waypoints at half the resolution or finer.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        widths = np.asarray(widths, dtype=np.float64).reshape(-1)
        if points.size == 0:
            raise ValueError("no waypoints: is the map loaded?")
        x0 = float(points[:, 0].min() - pad)
        y0 = float(points[:, 1].min() - pad)
        nx = int(np.ceil((points[:, 0].max() + pad - x0) / resolution))
        ny = int(np.ceil((points[:, 1].max() + pad - y0) / resolution))
        mask = np.zeros((ny, nx), dtype=bool)

        rad_cells = np.maximum((widths / 2.0) / resolution, 1.0)
        ix = ((points[:, 0] - x0) / resolution).astype(np.int32)
        iy = ((points[:, 1] - y0) / resolution).astype(np.int32)
        # Painted with a stamp per distinct radius rather than per waypoint:
        # a town is ~20k waypoints and a Python loop over them is seconds.
        for r in np.unique(np.round(rad_cells).astype(np.int32)):
            sel = np.round(rad_cells).astype(np.int32) == r
            if not sel.any():
                continue
            k = int(r)
            dy, dx = np.mgrid[-k:k + 1, -k:k + 1]
            stamp = (dx * dx + dy * dy) <= k * k
            oy, ox = np.nonzero(stamp)
            oy -= k
            ox -= k
            for cx, cy in zip(ix[sel], iy[sel]):
                yy = cy + oy
                xx = cx + ox
                ok = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
                mask[yy[ok], xx[ok]] = True
        return DrivableRaster((x0, y0), resolution, mask)

    def window(
        self, cx: float, cy: float, half_extent: float, resolution: float
    ) -> OccupancyGrid:
        """Crop an ego-centred, world-aligned rolling window (ADR-001).

        Anything outside the rasterised area is occupied, which is correct:
        off the map is not drivable.
        """
        grid = OccupancyGrid.empty(
            (cx - half_extent, cx + half_extent),
            (cy - half_extent, cy + half_extent),
            resolution,
        )
        gx, gy = np.meshgrid(np.arange(grid.nx), np.arange(grid.ny))
        wx, wy = grid.cell_to_world(gx, gy)
        sx = ((wx - self.origin[0]) / self.resolution).astype(np.int32)
        sy = ((wy - self.origin[1]) / self.resolution).astype(np.int32)
        inb = ((sx >= 0) & (sx < self.mask.shape[1])
               & (sy >= 0) & (sy < self.mask.shape[0]))
        np.clip(sx, 0, self.mask.shape[1] - 1, out=sx)
        np.clip(sy, 0, self.mask.shape[0] - 1, out=sy)
        drivable = np.where(inb, self.mask[sy, sx], False)
        grid.data = (~drivable).astype(np.float32)
        grid.invalidate_cache()
        return grid


def stamp_actors(
    grid: OccupancyGrid,
    boxes: Sequence[Tuple[float, float, float, float, float]],
) -> OccupancyGrid:
    """Paint oriented actor footprints ``(x, y, yaw, half_len, half_wid)``."""
    out = grid.copy()
    if not boxes:
        return out
    gx, gy = np.meshgrid(np.arange(out.nx), np.arange(out.ny))
    wx, wy = out.cell_to_world(gx, gy)
    data = out.data
    for x, y, yaw, hl, hw in boxes:
        c, s = math.cos(-yaw), math.sin(-yaw)
        dx, dy = wx - x, wy - y
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        data[(np.abs(lx) <= hl) & (np.abs(ly) <= hw)] = 1.0
    out.data = data
    out.invalidate_cache()
    return out


@dataclass
class Route:
    """A polyline route with arc-length progress. Satisfies ``RouteSource``."""

    points: np.ndarray                       # (N, 2)

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(seg)])

    def _nearest(self, x: float, y: float) -> int:
        return int(np.argmin(np.linalg.norm(self.points - np.array([x, y]), axis=1)))

    def progress(self, x: float, y: float) -> float:
        return float(self._cum[self._nearest(x, y)])

    def point_ahead(self, x: float, y: float, distance: float) -> np.ndarray:
        i = self._nearest(x, y)
        j = int(np.searchsorted(self._cum, self._cum[i] + distance))
        return self.points[min(j, len(self.points) - 1)].copy()

    @property
    def length(self) -> float:
        return float(self._cum[-1])


# Weather presets.  The rain and fog entries are not decoration: the deck's own
# challenge table names "fusion accuracy in rain/dust/glare" as a risk, and
# these are how that claim gets tested rather than asserted.
WEATHER_PRESETS: Dict[str, dict] = {
    "clear_noon": dict(cloudiness=5, precipitation=0, precipitation_deposits=0,
                       sun_altitude_angle=70, fog_density=0, wetness=0),
    "clear_sunset": dict(cloudiness=10, precipitation=0, precipitation_deposits=0,
                         sun_altitude_angle=8, fog_density=0, wetness=0),
    "hard_rain": dict(cloudiness=90, precipitation=80, precipitation_deposits=70,
                      sun_altitude_angle=30, fog_density=10, wetness=80),
    "wet_dusk": dict(cloudiness=70, precipitation=0, precipitation_deposits=60,
                     sun_altitude_angle=3, fog_density=5, wetness=70),
    "fog": dict(cloudiness=60, precipitation=0, precipitation_deposits=0,
                sun_altitude_angle=40, fog_density=65, fog_distance=8, wetness=10),
    "dust_haze": dict(cloudiness=35, precipitation=0, precipitation_deposits=0,
                      sun_altitude_angle=20, fog_density=25, fog_distance=25,
                      wetness=0),
    "night": dict(cloudiness=30, precipitation=0, precipitation_deposits=0,
                  sun_altitude_angle=-12, fog_density=8, wetness=15),
}
