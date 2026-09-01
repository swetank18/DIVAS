"""Stage contracts for the DIVAS pipeline.

Every module in this repository communicates through the types defined here and
nothing else.  They are the Python mirror of the ROS 2 messages in ``msgs/`` --
see ``EXECUTION_PLAN.md`` section 2.  Keeping the algorithms dependent on these
dataclasses rather than on ``rclpy`` is what lets the whole stack be unit tested
without a ROS environment, and what lets a ground-truth stub be swapped for a
real module with no change downstream.

Frame convention throughout: ego body frame, x forward, y left, theta CCW from
the x axis.  SI units everywhere (m, m/s, rad, s).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

# --------------------------------------------------------------------------
# Vehicle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class VehicleParams:
    """Kinematic bicycle parameters and actuation limits.

    Defaults describe a compact Indian hatchback; ``rover()`` gives the
    small-scale prototype of Phase 6 so the same stack runs on both.
    """

    wheelbase: float = 2.5
    length: float = 3.9
    width: float = 1.7
    rear_overhang: float = 0.7
    max_steer: float = 0.5           # rad
    max_steer_rate: float = 0.3      # rad/s
    max_accel: float = 2.0           # m/s^2
    min_accel: float = -4.0          # m/s^2 (braking)
    max_speed: float = 13.9          # m/s ~ 50 km/h, the hard cap
    cruise_speed: float = 9.0        # m/s ~ 32 km/h, what we actually aim for
    max_lat_accel: float = 3.0       # m/s^2, comfort bound
    # Longitudinal jerk limit.  A real driveline cannot step from full braking
    # to full acceleration instantly, so this belongs in the *dynamics*.
    # Expressing it as a cost penalty instead makes any change of mind
    # expensive in proportion to how large the change is, which is a ratchet:
    # once the controller commits to braking, un-braking is the most expensive
    # thing it can propose, and the vehicle crawls to a halt.
    max_jerk: float = 8.0            # m/s^3

    @property
    def collision_radius(self) -> float:
        """Radius of a single disc that covers the footprint.

        Used only for coarse checks; :meth:`footprint_discs` is the one the
        planner uses because a single disc over a 3.9 m car is far too
        conservative to fit through a gap between an auto and a divider.
        """
        return 0.5 * float(np.hypot(self.length, self.width))

    @property
    def half_extent(self) -> Tuple[float, float]:
        """``(half_length, half_width)``, for Minkowski-sum keep-outs."""
        return (self.length / 2.0, self.width / 2.0)

    def footprint_discs(self, n: int = 3) -> Tuple[np.ndarray, float]:
        """Approximate the rectangular footprint by ``n`` discs along the axis.

        Returns ``(offsets, radius)`` where ``offsets`` are longitudinal
        distances from the rear axle to each disc centre.
        """
        radius = 0.5 * float(np.hypot(self.length / n, self.width))
        # Rear axle sits ``rear_overhang`` behind the body's rear edge.
        back = -self.rear_overhang
        front = self.length - self.rear_overhang
        centres = np.linspace(
            back + self.length / (2 * n), front - self.length / (2 * n), n
        )
        return centres.astype(np.float64), radius

    @staticmethod
    def rover() -> "VehicleParams":
        """Small-scale prototype for Phase 6."""
        return VehicleParams(
            wheelbase=0.32,
            length=0.50,
            width=0.28,
            rear_overhang=0.09,
            max_steer=0.6,
            max_steer_rate=1.5,
            max_accel=1.0,
            min_accel=-1.5,
            max_jerk=6.0,
            max_speed=1.5,
            cruise_speed=1.0,
            max_lat_accel=2.0,
        )


@dataclass
class EgoState:
    """Ego pose and motion.  ``delta`` is the current road-wheel angle."""

    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    v: float = 0.0
    delta: float = 0.0
    a: float = 0.0        # applied longitudinal acceleration, for jerk limiting
    t: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v], dtype=np.float64)


@dataclass
class ControlCmd:
    """Output of stage 6.  Mirrors ``ackermann_msgs/AckermannDrive``."""

    accel: float = 0.0
    steer: float = 0.0


# --------------------------------------------------------------------------
# Stage 3 output: occupancy + tracks
# --------------------------------------------------------------------------


@dataclass
class OccupancyGrid:
    """Probabilistic occupancy over a bird's-eye grid.

    ``data[iy, ix]`` is P(occupied) in [0, 1].  ``origin`` is the world
    coordinate of the *corner* of cell ``[0, 0]``.  In this pipeline it is a
    64 x 64 m rolling window at 0.25 m/cell, ego-centred but world-aligned --
    it translates with the vehicle and does not rotate with it, so the plan
    made at 4 Hz and the control step consuming it at 20 Hz share one frame.
    See ``docs/decisions/ADR-001-local-costmap-frame.md``.
    """

    origin: Tuple[float, float]
    resolution: float
    data: np.ndarray  # (ny, nx) float32

    OCCUPIED_THRESHOLD: float = 0.5

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)
        if self.data.ndim != 2:
            raise ValueError(f"occupancy data must be 2-D, got {self.data.shape}")

    # -- geometry ---------------------------------------------------------
    @property
    def nx(self) -> int:
        return self.data.shape[1]

    @property
    def ny(self) -> int:
        return self.data.shape[0]

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` in world coordinates."""
        x0, y0 = self.origin
        return (x0, y0, x0 + self.nx * self.resolution, y0 + self.ny * self.resolution)

    @classmethod
    def empty(
        cls,
        x_range: Tuple[float, float] = (-5.0, 35.0),
        y_range: Tuple[float, float] = (-10.0, 10.0),
        resolution: float = 0.2,
    ) -> "OccupancyGrid":
        nx = int(round((x_range[1] - x_range[0]) / resolution))
        ny = int(round((y_range[1] - y_range[0]) / resolution))
        return cls(
            origin=(x_range[0], y_range[0]),
            resolution=resolution,
            data=np.zeros((ny, nx), dtype=np.float32),
        )

    def world_to_cell(self, x, y):
        """Vectorised world -> integer cell index.  No bounds checking."""
        x0, y0 = self.origin
        ix = np.floor((np.asarray(x) - x0) / self.resolution).astype(np.int32)
        iy = np.floor((np.asarray(y) - y0) / self.resolution).astype(np.int32)
        return ix, iy

    def cell_to_world(self, ix, iy):
        """Cell index -> world coordinate of the cell *centre*."""
        x0, y0 = self.origin
        x = x0 + (np.asarray(ix) + 0.5) * self.resolution
        y = y0 + (np.asarray(iy) + 0.5) * self.resolution
        return x, y

    def in_bounds(self, x, y) -> np.ndarray:
        xmin, ymin, xmax, ymax = self.bounds
        x = np.asarray(x)
        y = np.asarray(y)
        return (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)

    def occupancy_at(self, x, y, outside: float = 1.0) -> np.ndarray:
        """Sample occupancy.  Outside the map defaults to *occupied*.

        Treating unmapped space as blocked keeps the planner from escaping
        through the edge of its own perception, which is the single most
        common way a free-space planner produces a beautiful, useless path.
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        ok = self.in_bounds(x, y)
        ix, iy = self.world_to_cell(x, y)
        out = np.full(x.shape, outside, dtype=np.float32)
        if np.any(ok):
            out[ok] = self.data[iy[ok], ix[ok]]
        return out

    # -- derived fields ---------------------------------------------------
    def occupied_mask(self) -> np.ndarray:
        return self.data >= self.OCCUPIED_THRESHOLD

    def distance_field(self) -> np.ndarray:
        """Metres from each cell to the nearest occupied cell.

        Cached because the planner queries it once per node expansion and
        recomputing an EDT over a 200x100 grid inside the search loop is the
        difference between a 30 ms and a 3 s replan.
        """
        cached = getattr(self, "_distance_field", None)
        if cached is not None:
            return cached
        occ = self.occupied_mask()
        if not occ.any():
            df = np.full(self.data.shape, 1e3, dtype=np.float32)
        else:
            df = ndimage.distance_transform_edt(~occ).astype(np.float32)
            df *= self.resolution
        self._distance_field = df
        return df

    def signed_distance_field(self) -> np.ndarray:
        """Signed distance: positive in free space, negative inside obstacles.

        The unsigned EDT is flat (zero) everywhere inside an obstacle, so its
        gradient there is zero -- anything that has already penetrated an
        obstacle feels no force pushing it back out.  That silently let
        predicted agents drift off the drivable area.  The signed field has a
        usable gradient everywhere, which both the predictor's obstacle force
        and the controller's barrier term depend on.
        """
        cached = getattr(self, "_sdf", None)
        if cached is not None:
            return cached
        occ = self.occupied_mask()
        if not occ.any():
            sdf = np.full(self.data.shape, 1e3, dtype=np.float32)
        elif occ.all():
            sdf = np.full(self.data.shape, -1e3, dtype=np.float32)
        else:
            outside_d = ndimage.distance_transform_edt(~occ)
            inside_d = ndimage.distance_transform_edt(occ)
            sdf = ((outside_d - inside_d) * self.resolution).astype(np.float32)
        self._sdf = sdf
        return sdf

    def signed_clearance_at(self, x, y, outside: float = -1.0) -> np.ndarray:
        """Sample the signed distance field.  Off-map counts as inside."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        ok = self.in_bounds(x, y)
        ix, iy = self.world_to_cell(x, y)
        sdf = self.signed_distance_field()
        out = np.full(x.shape, outside, dtype=np.float32)
        if np.any(ok):
            out[ok] = sdf[iy[ok], ix[ok]]
        return out

    def clearance_at(self, x, y, outside: float = 0.0) -> np.ndarray:
        """Distance to the nearest obstacle at a world point."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        ok = self.in_bounds(x, y)
        ix, iy = self.world_to_cell(x, y)
        df = self.distance_field()
        out = np.full(x.shape, outside, dtype=np.float32)
        if np.any(ok):
            out[ok] = df[iy[ok], ix[ok]]
        return out

    def invalidate_cache(self) -> None:
        self._distance_field = None
        self._sdf = None

    def copy(self) -> "OccupancyGrid":
        return OccupancyGrid(self.origin, self.resolution, self.data.copy())


@dataclass
class Track:
    """One tracked dynamic actor.  Mirrors ``divas_msgs/Track``."""

    id: int
    x: float
    y: float
    vx: float
    vy: float
    cls: str = "unknown"
    radius: Optional[float] = None     # circumscribed; derived if unset
    cov: Optional[np.ndarray] = None  # 4x4 over (x, y, vx, vy)
    half_length: Optional[float] = None
    half_width: Optional[float] = None

    def __post_init__(self) -> None:
        hl, hw = CLASS_EXTENT.get(self.cls, CLASS_EXTENT["unknown"])
        if self.half_length is None:
            self.half_length = float(hl)
        if self.half_width is None:
            self.half_width = float(hw)
        if self.radius is None:
            self.radius = float(np.hypot(self.half_length, self.half_width))

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    @property
    def velocity(self) -> np.ndarray:
        return np.array([self.vx, self.vy], dtype=np.float64)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    @property
    def position_uncertainty(self) -> float:
        """Scalar 1-sigma position uncertainty, metres."""
        if self.cov is None:
            return 0.0
        return float(np.sqrt(max(self.cov[0, 0] + self.cov[1, 1], 0.0) / 2.0))


# Half-extents (half_length, half_width) in metres per class.  Modelling an
# actor as a single disc is tempting and wrong: a bus circumscribed by a disc
# blocks ten metres of lateral space and makes every scenario unsolvable.  The
# risk field uses these as an oriented ellipse instead.
CLASS_EXTENT = {
    "car": (2.0, 0.85),
    "truck": (3.5, 1.2),
    "bus": (5.5, 1.3),
    "autorickshaw": (1.3, 0.7),
    "motorcycle": (1.0, 0.35),
    "bicycle": (0.9, 0.3),
    "pedestrian": (0.3, 0.3),
    "animal": (0.9, 0.4),
    "unknown": (1.0, 0.6),
}

# Circumscribed radius, for coarse checks only.
CLASS_RADIUS = {k: float(np.hypot(*v)) for k, v in CLASS_EXTENT.items()}

# How fast an actor can change its mind.  A two-wheeler weaving through a gap
# and a bus holding its line deserve very different prediction spreads.
CLASS_AGILITY = {
    "car": 0.5,
    "truck": 0.2,
    "bus": 0.2,
    "autorickshaw": 0.8,
    "motorcycle": 1.0,
    "bicycle": 0.8,
    "pedestrian": 1.0,
    "animal": 1.0,
    "unknown": 0.8,
}


# --------------------------------------------------------------------------
# Stage 4 output: prediction
# --------------------------------------------------------------------------


@dataclass
class TrajectoryMode:
    """One hypothesis for how an actor will move."""

    probability: float
    points: np.ndarray  # (T, 2) positions at t = dt, 2*dt, ... T*dt

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)


@dataclass
class PredictedTrajectory:
    """The full multi-modal prediction for one track."""

    track_id: int
    modes: List[TrajectoryMode]
    radius: float = 0.5
    cls: str = "unknown"
    half_length: Optional[float] = None
    half_width: Optional[float] = None

    def __post_init__(self) -> None:
        hl, hw = CLASS_EXTENT.get(self.cls, CLASS_EXTENT["unknown"])
        if self.half_length is None:
            self.half_length = float(hl)
        if self.half_width is None:
            self.half_width = float(hw)

    # Length scale over which mode disagreement is considered total
    # uncertainty.  3 m is roughly one vehicle width plus a lane's worth of
    # lateral freedom -- if the modes disagree by that much, we know nothing
    # useful about where this actor will be.
    SPREAD_SCALE: float = 3.0

    def weights(self) -> np.ndarray:
        w = np.array([m.probability for m in self.modes], dtype=np.float64)
        s = w.sum()
        return w / s if s > 1e-9 else np.full(w.size, 1.0 / max(w.size, 1))

    def mean_path(self) -> np.ndarray:
        """Probability-weighted mean of the modes, ``(T, 2)``."""
        w = self.weights()
        pts = np.stack([m.points for m in self.modes])  # (K, T, 2)
        return np.einsum("k,ktd->td", w, pts)

    def spread_profile(self) -> np.ndarray:
        """Weighted std of mode positions at each step, ``(T,)`` in metres."""
        pts = np.stack([m.points for m in self.modes])
        if pts.shape[0] == 1:
            return np.zeros(pts.shape[1])
        w = self.weights()
        mean = np.einsum("k,ktd->td", w, pts)
        var = np.einsum("k,ktd->t", w, (pts - mean[None]) ** 2)
        return np.sqrt(np.maximum(var, 0.0))

    def confidence_profile(self) -> np.ndarray:
        """Per-step confidence in [0, 1] -- the input to the dynamic margin.

        Defined from how far apart the modes actually are in space, not from
        the entropy of their probabilities.  Entropy is the wrong measure
        here: three equally likely modes that all predict nearly the same
        path describe a *certain* future, and would be scored as maximally
        uncertain.  What the controller needs to know is how much room the
        prediction could be wrong by, in metres.
        """
        return np.exp(-self.spread_profile() / self.SPREAD_SCALE)

    @property
    def confidence(self) -> float:
        """Scalar summary: confidence at the end of the horizon, the worst
        point.  Reported for logging; the controller uses the full profile."""
        prof = self.confidence_profile()
        return float(prof[-1]) if prof.size else 1.0

@dataclass
class TrajectorySet:
    """Stage 4's output for the whole scene."""

    trajectories: List[PredictedTrajectory] = field(default_factory=list)
    dt: float = 0.1
    horizon: float = 3.0

    @property
    def n_steps(self) -> int:
        return int(round(self.horizon / self.dt))

    def __len__(self) -> int:
        return len(self.trajectories)

    def __iter__(self):
        return iter(self.trajectories)


# --------------------------------------------------------------------------
# Stage 5 output: path
# --------------------------------------------------------------------------


@dataclass
class Path:
    """Kinematically feasible path.  Columns are ``(x, y, theta, kappa)``."""

    points: np.ndarray
    # True when the path ends because the vehicle must stop (blocked, or the
    # search was cut short); False when it ends merely because the route goal
    # was that far ahead.  Without the distinction the controller brakes to a
    # halt at the edge of every plan, and the vehicle crawls down an empty
    # road at a quarter of its speed.
    terminal_stop: bool = False

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 4)

    def __len__(self) -> int:
        return int(self.points.shape[0])

    @property
    def xy(self) -> np.ndarray:
        return self.points[:, :2]

    @property
    def theta(self) -> np.ndarray:
        return self.points[:, 2]

    @property
    def kappa(self) -> np.ndarray:
        return self.points[:, 3]

    @property
    def length(self) -> float:
        if len(self) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.xy, axis=0), axis=1).sum())

    def curvature_profile(self, window: float = 2.5, tail: float = 3.0) -> np.ndarray:
        """Geometric curvature |dtheta/ds|, smoothed over ``window`` metres.

        The ``kappa`` column comes straight from the lattice primitive that
        produced each node, so it only ever takes one of the planner's handful
        of steering values.  A path that curves gently still reports the
        curvature of the tightest primitive it used, which collapses any
        curvature-based speed limit to a crawl.  Measuring curvature from the
        path's own geometry instead gives the value the vehicle will actually
        experience.
        """
        n = len(self)
        if n < 3:
            return np.abs(self.kappa)
        s = self.arc_lengths()
        ds = np.maximum(np.diff(s), 1e-6)
        dth = np.diff(np.unwrap(self.theta))
        k = np.abs(np.concatenate([[dth[0] / ds[0]], dth / ds]))
        step = max(float(np.mean(ds)), 1e-6)
        w = int(np.clip(round(window / step), 1, max(n - 1, 1)))
        k = ndimage.uniform_filter1d(k, size=w, mode="nearest")

        # A lattice path that reached its goal region ends wherever the search
        # happened to get within tolerance, usually with a hard turn onto the
        # goal.  That final curvature is an artifact of truncation, not a
        # property of the route -- and left alone it sets the speed limit for
        # the entire horizon, so the vehicle crawls the whole way because of
        # three metres of path it will re-plan long before reaching.
        if not self.terminal_stop and s[-1] > tail + 2.0:
            keep = s <= (s[-1] - tail)
            if keep.any():
                k[~keep] = k[keep][-1]
        return k

    def closest_index(self, x: float, y: float) -> int:
        if len(self) == 0:
            return 0
        d = np.linalg.norm(self.xy - np.array([x, y]), axis=1)
        return int(np.argmin(d))

    def arc_lengths(self) -> np.ndarray:
        if len(self) == 0:
            return np.zeros(0)
        seg = np.linalg.norm(np.diff(self.xy, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(seg)])

    def resample(self, spacing: float) -> "Path":
        """Uniform arc-length resampling -- the controller wants even spacing."""
        if len(self) < 2:
            return Path(self.points.copy(), self.terminal_stop)
        s = self.arc_lengths()
        n = max(int(np.floor(s[-1] / spacing)) + 1, 2)
        s_new = np.linspace(0.0, s[-1], n)
        cols = [np.interp(s_new, s, self.points[:, i]) for i in (0, 1)]
        # Angles must be unwrapped before interpolation or a pi -> -pi
        # crossing produces a full spurious rotation.
        cols.append(np.interp(s_new, s, np.unwrap(self.points[:, 2])))
        cols.append(np.interp(s_new, s, self.points[:, 3]))
        return Path(np.stack(cols, axis=1), self.terminal_stop)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def wrap_angle(a):
    """Wrap to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


__all__ = [
    "VehicleParams",
    "EgoState",
    "ControlCmd",
    "OccupancyGrid",
    "Track",
    "TrajectoryMode",
    "PredictedTrajectory",
    "TrajectorySet",
    "Path",
    "CLASS_RADIUS",
    "CLASS_EXTENT",
    "CLASS_AGILITY",
    "wrap_angle",
]
