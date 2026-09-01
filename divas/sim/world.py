"""A minimal 2-D world that speaks the stage contracts.

This is deliberately *not* CARLA and *not* Simulink.  Open decision #1 in
``EXECUTION_PLAN.md`` is unresolved, so the vertical slice runs on a simulator
that depends on neither.  Its job is narrow: produce a ground-truth
:class:`OccupancyGrid` and ground-truth :class:`Track` list in the ego frame,
so stages 5 and 6 can be built, closed-loop and measured before stages 1-4
exist.  When the simulator decision lands, CARLA/Simulink replaces this file
and nothing downstream changes.

An "unstructured road" here is a drivable polygon of varying width with no
lane concept anywhere in the representation -- which is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.path import Path as MplPath

from divas.types import (
    CLASS_EXTENT,
    CLASS_RADIUS,
    EgoState,
    OccupancyGrid,
    Track,
    VehicleParams,
    wrap_angle,
)


# --------------------------------------------------------------------------
# static scene
# --------------------------------------------------------------------------


@dataclass
class Circle:
    """Pothole, pole, cattle, debris -- anything roughly round."""

    x: float
    y: float
    radius: float
    label: str = "obstacle"

    def sdf(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        return np.hypot(px - self.x, py - self.y) - self.radius


@dataclass
class Rect:
    """Parked vehicle, median, construction barrier.  Oriented."""

    x: float
    y: float
    length: float
    width: float
    theta: float = 0.0
    label: str = "obstacle"

    def sdf(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        c, s = np.cos(-self.theta), np.sin(-self.theta)
        dx = px - self.x
        dy = py - self.y
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        qx = np.abs(lx) - self.length / 2
        qy = np.abs(ly) - self.width / 2
        outside = np.hypot(np.maximum(qx, 0.0), np.maximum(qy, 0.0))
        inside = np.minimum(np.maximum(qx, qy), 0.0)
        return outside + inside


@dataclass
class Road:
    """Drivable corridor: a centreline with per-station half-widths.

    No lane markings, no lane graph -- just the region the vehicle may
    physically occupy.  Width varies along the road because Indian
    carriageways narrow without warning.
    """

    centerline: np.ndarray            # (N, 2)
    left_width: np.ndarray            # (N,)
    right_width: np.ndarray           # (N,)

    def __post_init__(self) -> None:
        self.centerline = np.asarray(self.centerline, dtype=np.float64).reshape(-1, 2)
        n = len(self.centerline)
        self.left_width = np.broadcast_to(
            np.asarray(self.left_width, dtype=np.float64), (n,)
        ).copy()
        self.right_width = np.broadcast_to(
            np.asarray(self.right_width, dtype=np.float64), (n,)
        ).copy()

    @staticmethod
    def straight(
        length: float = 200.0,
        half_width: float = 4.0,
        n: int = 226,
        start: float = -25.0,
    ) -> "Road":
        # The corridor starts *behind* the ego.  Without this the vehicle
        # begins flush against the road's own end cap and every plan fails
        # its first collision check -- an artifact of the scenario, not of
        # the planner, and an easy one to misdiagnose.
        s = np.linspace(start, length, n)
        cl = np.stack([s, np.zeros_like(s)], axis=1)
        return Road(cl, np.full(n, half_width), np.full(n, half_width))

    @staticmethod
    def winding(
        length: float = 200.0,
        half_width: float = 4.5,
        amplitude: float = 6.0,
        wavelength: float = 90.0,
        pinch: float = 1.5,
        n: int = 226,
        start: float = -25.0,
    ) -> "Road":
        """A curving corridor that also narrows -- the realistic case."""
        s = np.linspace(start, length, n)
        y = amplitude * np.sin(2 * np.pi * s / wavelength)
        cl = np.stack([s, y], axis=1)
        squeeze = pinch * np.sin(2 * np.pi * s / (wavelength * 0.6)) ** 2
        return Road(cl, half_width - squeeze, half_width - squeeze * 0.5)

    def tangents(self) -> np.ndarray:
        d = np.gradient(self.centerline, axis=0)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return d / np.maximum(n, 1e-9)

    def normals(self) -> np.ndarray:
        t = self.tangents()
        return np.stack([-t[:, 1], t[:, 0]], axis=1)  # left-hand normal

    def polygon(self) -> np.ndarray:
        nrm = self.normals()
        left = self.centerline + nrm * self.left_width[:, None]
        right = self.centerline - nrm * self.right_width[:, None]
        return np.concatenate([left, right[::-1]], axis=0)

    def station_of(self, x: float, y: float) -> int:
        d = np.linalg.norm(self.centerline - np.array([x, y]), axis=1)
        return int(np.argmin(d))

    def progress(self, x: float, y: float) -> float:
        """Arc length travelled along the centreline, metres."""
        i = self.station_of(x, y)
        seg = np.linalg.norm(np.diff(self.centerline[: i + 1], axis=0), axis=1)
        return float(seg.sum())

    def arc_table(self) -> np.ndarray:
        seg = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(seg)])

    def offset_point(self, s: float, lateral: float) -> np.ndarray:
        """Point at arc length ``s`` along the corridor, offset laterally.

        Scenario obstacles must be placed in the road's own frame.  Placing a
        pothole at a world y-coordinate puts it on the tarmac on a straight
        road and in a field on a winding one, which silently turns a
        navigation test into an unsolvable one.
        """
        cum = self.arc_table()
        i = int(np.clip(np.searchsorted(cum, s), 0, len(self.centerline) - 1))
        return self.centerline[i] + self.normals()[i] * lateral

    def lateral_offset(self, x: float, y: float) -> float:
        """Signed distance left of the centreline, metres."""
        i = self.station_of(x, y)
        n = self.normals()[i]
        d = np.array([x, y]) - self.centerline[i]
        return float(d[0] * n[0] + d[1] * n[1])

    def half_width_at(self, x: float, y: float) -> Tuple[float, float]:
        i = self.station_of(x, y)
        return float(self.left_width[i]), float(self.right_width[i])

    def point_ahead(self, x: float, y: float, distance: float) -> np.ndarray:
        """Local goal: a point ``distance`` further along the centreline.

        This stands in for the global route.  It is intentionally crude --
        the contribution of this project is what happens *between* here and
        the vehicle, not the routing.
        """
        i = self.station_of(x, y)
        seg = np.linalg.norm(np.diff(self.centerline, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        target = cum[i] + distance
        j = int(np.searchsorted(cum, target))
        j = min(j, len(self.centerline) - 1)
        return self.centerline[j].copy()


# --------------------------------------------------------------------------
# dynamic actors
# --------------------------------------------------------------------------


# Intelligent Driver Model parameters per class.  Deliberately a *different*
# model family from the social-force predictor that will be scored against
# these actors: if the simulated traffic obeyed the same equations the
# predictor assumes, "interaction-aware prediction helps" would be a statement
# about the benchmark rather than about the world.
IDM_DEFAULTS = {
    #                v0     T     a_max   b     s0
    "car":          (9.0,  1.1,   1.8,   2.2,   2.5),
    "truck":        (7.0,  1.6,   0.9,   1.6,   4.0),
    "bus":          (7.0,  1.6,   0.9,   1.6,   4.0),
    "autorickshaw": (7.0,  0.8,   1.6,   2.5,   1.5),
    "motorcycle":   (10.0, 0.6,   2.5,   3.0,   1.0),
    "bicycle":      (5.0,  0.8,   1.0,   1.5,   0.8),
    "unknown":      (8.0,  1.0,   1.5,   2.0,   2.0),
}


@dataclass
class Actor:
    """A mixed-traffic agent with a scripted intent.

    Scripted rather than reactive on purpose: the scenario must be
    *repeatable* so that an ablation compares planners, not luck.
    """

    id: int
    cls: str
    x: float
    y: float
    theta: float
    v: float
    policy: str = "constant"
    params: dict = field(default_factory=dict)
    alive: bool = True

    @property
    def extent(self) -> Tuple[float, float]:
        """``(half_length, half_width)`` -- oriented, not a disc."""
        return tuple(self.params.get("extent", CLASS_EXTENT.get(self.cls, (1.0, 0.6))))

    @property
    def radius(self) -> float:
        """Circumscribed radius, for coarse rasterisation and repulsion."""
        return float(np.hypot(*self.extent))

    def step(self, dt: float, t: float, world: "World") -> None:
        p = self.params
        if self.policy == "constant":
            pass
        elif self.policy == "cutin":
            # Swerves across the ego's path once the ego closes to within
            # ``trigger_gap`` metres -- the sub-second scenario the fixed
            # safety buffer cannot survive.
            gap = np.hypot(self.x - world.ego.x, self.y - world.ego.y)
            if not p.get("_fired") and gap < p.get("trigger_gap", 14.0):
                p["_fired"] = True
                p["_t0"] = t
            if p.get("_fired"):
                dur = p.get("duration", 1.6)
                frac = np.clip((t - p["_t0"]) / dur, 0.0, 1.0)
                target = p.get("heading_change", -0.55)
                self.theta = p.get("_theta0", self.theta) if frac >= 1.0 else self.theta
                if frac < 1.0:
                    self.theta += target * dt / dur
        elif self.policy == "crossing":
            if t >= p.get("start_time", 0.0):
                pass
            else:
                self.v = 0.0
            if t >= p.get("start_time", 0.0):
                self.v = p.get("cruise", 1.3)
        elif self.policy == "erratic":
            # Wobbling auto-rickshaw / two-wheeler: heading random walk.
            rng = world.rng
            self.theta += rng.normal(0.0, p.get("sigma", 0.06))
            self.v = float(
                np.clip(self.v + rng.normal(0.0, 0.25), p.get("vmin", 2.0), p.get("vmax", 9.0))
            )
        elif self.policy == "stop_and_go":
            period = p.get("period", 6.0)
            self.v = p.get("cruise", 6.0) if (t % period) < period * 0.6 else 0.0
        elif self.policy == "reactive":
            self._react(dt, world)
        elif self.policy == "wait_then_go":
            self.v = p.get("cruise", 5.0) if t >= p.get("start_time", 3.0) else 0.0

        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt

    # -- reactive behaviour ----------------------------------------------
    def _neighbours(self, world: "World"):
        """Everything this actor can see, as ``(x, y, theta, v, half_len, half_wid)``.

        The ego is included.  An actor that ignores the vehicle under test
        would make the whole exercise circular.
        """
        out = []
        for o in world.actors:
            if o is self or not o.alive:
                continue
            hl, hw = o.extent
            out.append((o.x, o.y, o.theta, o.v, hl, hw))
        e = world.ego
        ehl, ehw = world.params.half_extent
        out.append((e.x, e.y, e.theta, e.v, ehl, ehw))
        return out

    def _react(self, dt: float, world: "World") -> None:
        """IDM longitudinal control plus lateral gap seeking.

        Longitudinal: standard Intelligent Driver Model against the nearest
        leader in this actor's own corridor.  Lateral: drift away from close
        neighbours and back toward the drivable corridor, with an overtaking
        bias when the leader is much slower and there is room to pass.

        The overtaking bias is the point of the whole exercise.  It is the
        behaviour a constant-velocity predictor cannot anticipate -- the actor
        is travelling straight right up to the moment it is not -- and it is
        commonplace on an Indian road.
        """
        p = self.params
        cls = self.cls if self.cls in IDM_DEFAULTS else "unknown"
        v0, T, a_max, b, s0 = IDM_DEFAULTS[cls]
        v0 = float(p.get("v0", v0))
        hl, hw = self.extent

        c, s = np.cos(-self.theta), np.sin(-self.theta)
        corridor = hw + float(p.get("corridor", 1.1))

        # -- find the leader: nearest thing ahead within our corridor
        gap, lead_v = np.inf, 0.0
        lat_force = 0.0
        free_left = free_right = float(p.get("scan", 20.0))
        for ox, oy, oth, ov, ohl, ohw in self._neighbours(world):
            dx, dy = ox - self.x, oy - self.y
            ax = c * dx - s * dy           # along our heading
            ay = s * dx + c * dy           # left of our heading
            if abs(ay) < corridor + ohw and 0.0 < ax < 60.0:
                g = ax - hl - ohl
                if g < gap:
                    gap, lead_v = g, ov * np.cos(oth - self.theta)
            # lateral pressure from anything roughly abreast
            if -hl - 2.0 < ax < 25.0:
                sep = abs(ay) - hw - ohw
                if sep < 4.0:
                    push = np.exp(-max(sep, 0.05) / 1.5)
                    lat_force -= np.sign(ay) * push
                    if ay > 0:
                        free_left = min(free_left, max(sep, 0.0))
                    else:
                        free_right = min(free_right, max(sep, 0.0))

        # -- IDM
        if np.isfinite(gap):
            gap = max(gap, 0.3)
            dv = self.v - lead_v
            s_star = s0 + max(self.v * T + self.v * dv / (2 * np.sqrt(a_max * b)), 0.0)
            accel = a_max * (1.0 - (self.v / v0) ** 4 - (s_star / gap) ** 2)
        else:
            accel = a_max * (1.0 - (self.v / v0) ** 4)
        accel = float(np.clip(accel, -6.0, a_max))
        self.v = float(np.clip(self.v + accel * dt, 0.0, v0 * 1.3))

        # -- overtaking bias: a slow leader close ahead, and room to one side
        if np.isfinite(gap) and gap < 22.0 and lead_v < 0.75 * v0:
            urgency = float(np.clip((22.0 - gap) / 22.0, 0.0, 1.0))
            side = 1.0 if free_left >= free_right else -1.0
            lat_force += side * urgency * float(p.get("overtake", 1.6))

        # -- stay on the drivable corridor
        off = world.road.lateral_offset(self.x, self.y)
        lw, rw = world.road.half_width_at(self.x, self.y)
        limit = (lw if off > 0 else rw) - hw - 0.4
        if abs(off) > limit:
            lat_force -= np.sign(off) * 2.5 * (abs(off) - limit)
        lat_force -= 0.12 * off                      # weak pull to the centre

        # -- lateral force -> heading, rate limited
        max_rate = float(p.get("yaw_rate", 0.45))
        dtheta = float(np.clip(lat_force * 0.28, -max_rate, max_rate)) * dt
        road_dir = np.arctan2(*world.road.tangents()[world.road.station_of(self.x, self.y)][::-1])
        self.theta = float(
            wrap_angle(self.theta + dtheta
                       + 0.6 * dt * wrap_angle(road_dir - self.theta) * 0.5)
        )

    def to_track(self, ego: EgoState, rng: np.random.Generator, noise: float) -> Track:
        """Ground-truth actor -> :class:`Track`, in the odom frame, with noise.

        Reported world-aligned rather than rotated into the ego body frame:
        the local costmap is a world-aligned rolling window (see
        :meth:`World.ground_truth_grid`), so keeping tracks in the same frame
        means nothing downstream ever transforms anything.

        Noise grows with range, which is what makes prediction confidence --
        and therefore the dynamic safety margin -- vary meaningfully instead
        of being a constant dressed up as a variable.
        """
        ex, ey = self.x, self.y
        evx = self.v * np.cos(self.theta)
        evy = self.v * np.sin(self.theta)

        rng_m = float(np.hypot(self.x - ego.x, self.y - ego.y))
        sigma_p = noise * (0.15 + 0.012 * rng_m)
        sigma_v = noise * (0.30 + 0.020 * rng_m)
        if noise > 0.0:
            ex += rng.normal(0.0, sigma_p)
            ey += rng.normal(0.0, sigma_p)
            evx += rng.normal(0.0, sigma_v)
            evy += rng.normal(0.0, sigma_v)
        cov = np.diag([sigma_p**2, sigma_p**2, sigma_v**2, sigma_v**2])
        return Track(
            id=self.id,
            x=float(ex),
            y=float(ey),
            vx=float(evx),
            vy=float(evy),
            cls=self.cls,
            radius=self.radius,
            cov=cov,
            half_length=self.extent[0],
            half_width=self.extent[1],
        )


# --------------------------------------------------------------------------
# world
# --------------------------------------------------------------------------


@dataclass
class World:
    road: Road
    statics: List[object] = field(default_factory=list)
    actors: List[Actor] = field(default_factory=list)
    ego: EgoState = field(default_factory=EgoState)
    params: VehicleParams = field(default_factory=VehicleParams)
    t: float = 0.0
    track_noise: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._road_poly = MplPath(self.road.polygon())

    # -- dynamics ---------------------------------------------------------
    def step(self, dt: float, accel: float, steer: float) -> None:
        """Advance ego (kinematic bicycle, rate-limited steering) and actors."""
        p = self.params
        steer = float(np.clip(steer, -p.max_steer, p.max_steer))
        max_d = p.max_steer_rate * dt
        self.ego.delta += float(np.clip(steer - self.ego.delta, -max_d, max_d))
        accel = float(np.clip(accel, p.min_accel, p.max_accel))
        # Jerk limit, same as the steering rate limit above and for the same
        # reason: the actuator cannot follow a step command.
        max_da = p.max_jerk * dt
        accel = float(np.clip(accel, self.ego.a - max_da, self.ego.a + max_da))
        self.ego.a = accel

        e = self.ego
        e.x += e.v * np.cos(e.theta) * dt
        e.y += e.v * np.sin(e.theta) * dt
        e.theta = float(wrap_angle(e.theta + e.v / p.wheelbase * np.tan(e.delta) * dt))
        e.v = float(np.clip(e.v + accel * dt, 0.0, p.max_speed))
        e.t = self.t = self.t + dt

        for a in self.actors:
            if a.alive:
                a.step(dt, self.t, self)

    # -- perception ground truth -----------------------------------------
    def ground_truth_grid(
        self,
        half_extent: float = 32.0,
        resolution: float = 0.25,
        include_actors: bool = True,
    ) -> OccupancyGrid:
        """Ego-centred, world-aligned local costmap -- a rolling window.

        Everything outside the drivable polygon is occupied, plus static
        obstacles, plus (optionally) current actor footprints.  Actors are
        included because the planner needs them as obstacles *now*; where they
        are going is stage 4's problem, not stage 3's.

        The window translates with the ego but does not rotate with it.  This
        is what a real local costmap does (ROS ``costmap_2d`` in the odom
        frame), and it means the plan produced at 4 Hz and the control step
        that consumes it at 20 Hz share one frame, with no transform to get
        stale in between.
        """
        grid = OccupancyGrid.empty(
            (self.ego.x - half_extent, self.ego.x + half_extent),
            (self.ego.y - half_extent, self.ego.y + half_extent),
            resolution,
        )
        gx, gy = np.meshgrid(np.arange(grid.nx), np.arange(grid.ny))
        wx, wy = grid.cell_to_world(gx, gy)

        pts = np.stack([wx.ravel(), wy.ravel()], axis=1)
        inside = self._road_poly.contains_points(pts).reshape(wx.shape)
        occ = (~inside).astype(np.float32)

        for obj in self.statics:
            occ[obj.sdf(wx, wy) <= 0.0] = 1.0

        if include_actors:
            for a in self.actors:
                if not a.alive:
                    continue
                hl, hw = a.extent
                occ[Rect(a.x, a.y, 2 * hl, 2 * hw, a.theta).sdf(wx, wy) <= 0.0] = 1.0

        grid.data = occ
        grid.invalidate_cache()
        return grid

    def ground_truth_grids(
        self, half_extent: float = 32.0, resolution: float = 0.25
    ) -> Tuple[OccupancyGrid, OccupancyGrid]:
        """``(static, full)`` -- the same window with and without actors.

        A real fusion stage keeps a static layer and a dynamic object list
        separately, and the two consumers want different things: the planner
        needs actors as obstacles *now*, while the predictor must not see them
        in the grid at all -- it already repels agents from each other through
        its social term, and a grid containing those agents applies the
        repulsion twice.

        Both come from one point-in-polygon pass, which is the expensive part.
        """
        static = self.ground_truth_grid(half_extent, resolution, include_actors=False)
        full = static.copy()
        if self.actors:
            gx, gy = np.meshgrid(np.arange(full.nx), np.arange(full.ny))
            wx, wy = full.cell_to_world(gx, gy)
            occ = full.data
            for a in self.actors:
                if not a.alive:
                    continue
                hl, hw = a.extent
                occ[Rect(a.x, a.y, 2 * hl, 2 * hw, a.theta).sdf(wx, wy) <= 0.0] = 1.0
            full.data = occ
            full.invalidate_cache()
        return static, full

    def ground_truth_tracks(self) -> List[Track]:
        return [
            a.to_track(self.ego, self.rng, self.track_noise)
            for a in self.actors
            if a.alive
        ]

    def local_goal(self, lookahead: float = 25.0) -> np.ndarray:
        """Route target, odom frame.  Stands in for the global route."""
        return self.road.point_ahead(self.ego.x, self.ego.y, lookahead)

    # -- evaluation -------------------------------------------------------
    def ego_footprint_discs(self) -> Tuple[np.ndarray, float]:
        offs, r = self.params.footprint_discs()
        c, s = np.cos(self.ego.theta), np.sin(self.ego.theta)
        cx = self.ego.x + offs * c
        cy = self.ego.y + offs * s
        return np.stack([cx, cy], axis=1), r

    def collision(self) -> Optional[str]:
        """Return a description of what we hit, or ``None``."""
        centres, r = self.ego_footprint_discs()
        for a in self.actors:
            if not a.alive:
                continue
            hl, hw = a.extent
            box = Rect(a.x, a.y, 2 * hl, 2 * hw, a.theta, a.cls)
            if float(box.sdf(centres[:, 0], centres[:, 1]).min()) <= r:
                return f"actor:{a.cls}#{a.id}"
        for obj in self.statics:
            if float(obj.sdf(centres[:, 0], centres[:, 1]).min()) <= r:
                return f"static:{obj.label}"
        if not self._road_poly.contains_points(centres).all():
            return "off_road"
        return None

    def clearance_to_actors(self) -> float:
        """Smallest surface-to-surface gap to any actor, metres."""
        centres, r = self.ego_footprint_discs()
        best = np.inf
        for a in self.actors:
            if not a.alive:
                continue
            hl, hw = a.extent
            box = Rect(a.x, a.y, 2 * hl, 2 * hw, a.theta, a.cls)
            best = min(best, float(box.sdf(centres[:, 0], centres[:, 1]).min()) - r)
        return best

    def time_to_collision(self) -> float:
        """Constant-velocity TTC against the nearest closing actor.

        The standard surrogate safety metric; reported as a distribution over
        each run because the minimum alone hides how often we were close.

        Computed against an **oriented elliptical** boundary in the actor's
        heading frame, not a circumscribed disc.  With discs, the ego (1.07 m
        radius) plus a truck (3.7 m radius) demand 4.8 m of separation, so
        passing an oncoming truck three metres to the side scores as an
        imminent collision and the headline safety metric reads near zero on
        runs that were never in danger.
        """
        best = np.inf
        e = self.ego
        evx, evy = e.v * np.cos(e.theta), e.v * np.sin(e.theta)
        ehl, ehw = self.params.half_extent
        for a in self.actors:
            if not a.alive:
                continue
            ahl, ahw = a.extent
            # Work in the actor's heading frame so the ellipse is axis-aligned.
            c, s = np.cos(-a.theta), np.sin(-a.theta)
            dx, dy = e.x - a.x, e.y - a.y
            rx = c * dx - s * dy
            ry = s * dx + c * dy
            dvx, dvy = evx - a.v * np.cos(a.theta), evy - a.v * np.sin(a.theta)
            rvx = c * dvx - s * dvy
            rvy = s * dvx + c * dvy
            A = ahl + ehl
            B = ahw + ehw
            # ((rx+rvx t)/A)^2 + ((ry+rvy t)/B)^2 = 1  ->  quadratic in t
            qa = (rvx / A) ** 2 + (rvy / B) ** 2
            qb = 2 * (rx * rvx / A**2 + ry * rvy / B**2)
            qc = (rx / A) ** 2 + (ry / B) ** 2 - 1.0
            if qa < 1e-12 or qc <= 0.0:
                continue
            disc = qb**2 - 4 * qa * qc
            if disc < 0.0:
                continue
            t = (-qb - np.sqrt(disc)) / (2 * qa)
            if t > 0.0:
                best = min(best, float(t))
        return best
