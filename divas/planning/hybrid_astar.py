"""Stage 5 -- Hybrid A* over the free-space occupancy grid.

No lane graph is consulted anywhere in this file, because there isn't one.  The
search runs directly over the drivable area produced by stage 3, with
kinematic feasibility built into the motion primitives rather than bolted on as
a smoothing pass afterwards -- a smoothed infeasible path is still infeasible,
it just hides it from the plot.

Three properties matter more than optimality here:

* **A hard time budget.**  The planner runs at 2-5 Hz against a 20-50 Hz
  controller.  Exceeding the budget is worse than returning a mediocre path,
  so the search always returns *something* -- see ``partial``.
* **Time-aware risk.**  Cost is sampled from the :class:`RiskField` at the
  time the vehicle would actually arrive at each point, so the planner avoids
  where traffic *will be*, not where it is now.
* **Signed collision checking.**  A pose already inside an obstacle must read
  as a collision, which the unsigned distance transform cannot express.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from divas.prediction.risk import RiskField
from divas.types import EgoState, OccupancyGrid, Path, VehicleParams, wrap_angle


@dataclass
class PlannerConfig:
    xy_resolution: float = 0.6        # m, state-lattice cell
    n_theta: int = 24                 # heading bins
    n_steer: int = 7                  # motion primitives per expansion
    step_length: float = 1.6          # m of arc per primitive
    substeps: int = 3                 # collision samples per primitive
    goal_tolerance: float = 3.0       # m
    # Plan with more margin than the vehicle strictly needs.  The controller
    # tracks the path with error, so a path that clears an obstacle by the
    # footprint radius exactly is a path that hits it.
    inflation: float = 0.35           # m added to the footprint discs
    preferred_clearance: float = 1.0  # m; below this, cost is added

    w_length: float = 1.0
    w_steer: float = 0.6
    w_steer_change: float = 1.2
    w_risk: float = 26.0
    w_clearance: float = 8.0
    # Hysteresis toward the previous plan.  Hybrid A* has no memory, so when
    # two ways round an obstacle score almost equally the choice can flip
    # between consecutive replans.  The vehicle then commits to neither, and
    # arrives at the obstacle having gone half way round each -- which is
    # exactly where the obstacle is.  Small enough that a genuinely better
    # route still wins; large enough to break a tie the same way twice.
    w_hysteresis: float = 0.25        # cost per metre of deviation, per metre driven
    hysteresis_cap: float = 3.0       # m; beyond this, deviation is not penalised further

    max_iterations: int = 4000
    # Wall-clock deadline.  ``None`` disables it, leaving ``max_iterations``
    # as the only termination condition.
    #
    # A deadline is right on the vehicle -- overrunning is worse than a
    # mediocre path.  It is *wrong* in evaluation: the search returns whatever
    # it has reached when the clock runs out, so results depend on how loaded
    # the machine is.  Two ablation runs of identical code and seeds differed
    # by 0.62 vs 0.79 success and 0.27 vs 0.12 collisions purely because one
    # ran 240 simulations across 12 workers and the other 144.  Any comparison
    # drawn across those rows would have been measuring CPU contention.
    time_budget_ms: Optional[float] = 120.0
    plan_speed_floor: float = 3.0     # m/s, for time-parameterising the path
    heuristic_downsample: int = 6


@dataclass
class PlanResult:
    path: Path
    success: bool
    reason: str = ""
    partial: bool = False
    iterations: int = 0
    elapsed_ms: float = 0.0
    cost: float = float("inf")

    def __bool__(self) -> bool:
        return self.success


class HybridAStar:
    def __init__(
        self,
        params: Optional[VehicleParams] = None,
        config: Optional[PlannerConfig] = None,
    ) -> None:
        self.params = params or VehicleParams()
        self.cfg = config or PlannerConfig()
        self._steers = np.linspace(
            -self.params.max_steer, self.params.max_steer, self.cfg.n_steer
        )
        self._disc_offsets, self._disc_radius = self.params.footprint_discs()

    # -- fast grid samplers ----------------------------------------------
    @staticmethod
    def _make_sampler(grid: OccupancyGrid):
        """Direct array indexing.  ``OccupancyGrid.signed_clearance_at`` is
        ~30 us of numpy overhead per call; the search makes tens of thousands
        of them, which is the difference between a 60 ms and a 3 s replan."""
        sdf = grid.signed_distance_field()
        x0, y0 = grid.origin
        res = grid.resolution
        nx, ny = grid.nx, grid.ny

        def sample(px: np.ndarray, py: np.ndarray) -> np.ndarray:
            ix = ((px - x0) / res).astype(np.int32)
            iy = ((py - y0) / res).astype(np.int32)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            np.clip(ix, 0, nx - 1, out=ix)
            np.clip(iy, 0, ny - 1, out=iy)
            out = sdf[iy, ix]
            return np.where(ok, out, -1.0)  # off-map counts as occupied

        return sample

    def _previous_path_field(self, grid: OccupancyGrid, previous: Optional[Path]):
        """Distance-to-previous-path sampler, or ``None``.

        Rasterised into one EDT rather than evaluated per node: the search
        queries it tens of thousands of times, and a per-node scan over the
        previous path's points would cost more than the search itself.
        """
        if previous is None or len(previous) < 2:
            return None
        mask = np.ones((grid.ny, grid.nx), dtype=bool)
        pts = previous.resample(max(grid.resolution, 0.2)).xy
        x0, y0 = grid.origin
        ix = ((pts[:, 0] - x0) / grid.resolution).astype(np.int32)
        iy = ((pts[:, 1] - y0) / grid.resolution).astype(np.int32)
        ok = (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
        if not ok.any():
            return None
        mask[iy[ok], ix[ok]] = False
        field = (ndimage.distance_transform_edt(mask) * grid.resolution).astype(np.float32)
        np.clip(field, 0.0, self.cfg.hysteresis_cap, out=field)
        nx, ny, res = grid.nx, grid.ny, grid.resolution

        def sample(px, py):
            jx = ((px - x0) / res).astype(np.int32)
            jy = ((py - y0) / res).astype(np.int32)
            inb = (jx >= 0) & (jx < nx) & (jy >= 0) & (jy < ny)
            np.clip(jx, 0, nx - 1, out=jx)
            np.clip(jy, 0, ny - 1, out=jy)
            return np.where(inb, field[jy, jx], self.cfg.hysteresis_cap)

        return sample

    # -- heuristic --------------------------------------------------------
    def _geodesic_heuristic(
        self, grid: OccupancyGrid, goal: np.ndarray
    ) -> Optional[Tuple[np.ndarray, float, Tuple[float, float]]]:
        """Obstacle-aware 2-D distance-to-goal on a coarsened grid.

        Straight-line distance is a terrible heuristic in front of a wall: the
        search expands the entire dead end first.  This is the standard
        "holonomic with obstacles" term, computed on a downsampled grid so it
        costs a few milliseconds rather than a few hundred.

        Deliberately *optimistic* -- obstacles are not inflated by the vehicle
        width -- so the heuristic stays admissible and never blocks a gap the
        full-resolution search could actually thread.
        """
        f = self.cfg.heuristic_downsample
        blocked = grid.distance_field() < 0.15
        ny, nx = blocked.shape
        ny2, nx2 = ny // f, nx // f
        if ny2 < 2 or nx2 < 2:
            return None
        coarse = blocked[: ny2 * f, : nx2 * f].reshape(ny2, f, nx2, f).max(axis=(1, 3))
        cres = grid.resolution * f
        x0, y0 = grid.origin

        gx = int((goal[0] - x0) / cres)
        gy = int((goal[1] - y0) / cres)
        gx = int(np.clip(gx, 0, nx2 - 1))
        gy = int(np.clip(gy, 0, ny2 - 1))
        if coarse[gy, gx]:  # goal inside an obstacle: snap to nearest free
            free = np.argwhere(~coarse)
            if free.size == 0:
                return None
            d = np.linalg.norm(free - np.array([gy, gx]), axis=1)
            gy, gx = free[int(np.argmin(d))]

        INF = np.inf
        dist = np.full((ny2, nx2), INF, dtype=np.float64)
        dist[gy, gx] = 0.0
        heap = [(0.0, gy, gx)]
        nbrs = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.41421356), (-1, 1, 1.41421356),
            (1, -1, 1.41421356), (1, 1, 1.41421356),
        ]
        while heap:
            d, y, x = heapq.heappop(heap)
            if d > dist[y, x]:
                continue
            for dy, dx, w in nbrs:
                ny_, nx_ = y + dy, x + dx
                if not (0 <= ny_ < ny2 and 0 <= nx_ < nx2) or coarse[ny_, nx_]:
                    continue
                nd = d + w * cres
                if nd < dist[ny_, nx_]:
                    dist[ny_, nx_] = nd
                    heapq.heappush(heap, (nd, ny_, nx_))
        return dist, cres, (x0, y0)

    # -- expansion --------------------------------------------------------
    def _expand(self, x, y, theta):
        """Arc motion primitives for every steering angle at once."""
        L = self.cfg.step_length
        wb = self.params.wheelbase
        s = self._steers
        tan = np.tan(s)
        straight = np.abs(tan) < 1e-6
        R = np.where(straight, 1.0, wb / np.where(straight, 1.0, tan))
        dtheta = np.where(straight, 0.0, L / R)
        nt = theta + dtheta
        nx = np.where(straight, x + L * np.cos(theta), x + R * (np.sin(nt) - np.sin(theta)))
        ny = np.where(straight, y + L * np.sin(theta), y - R * (np.cos(nt) - np.cos(theta)))
        return nx, ny, wrap_angle(nt), s, R, straight

    def _substep_poses(self, x, y, theta):
        """Intermediate poses along each primitive, for collision checking."""
        L = self.cfg.step_length
        wb = self.params.wheelbase
        s = self._steers
        tan = np.tan(s)
        straight = np.abs(tan) < 1e-6
        R = np.where(straight, 1.0, wb / np.where(straight, 1.0, tan))
        fracs = np.linspace(L / self.cfg.substeps, L, self.cfg.substeps)
        dth = np.where(straight, 0.0, 1.0 / R)[:, None] * fracs[None, :]
        th = theta + dth
        px = np.where(
            straight[:, None],
            x + fracs[None, :] * np.cos(theta),
            x + R[:, None] * (np.sin(th) - np.sin(theta)),
        )
        py = np.where(
            straight[:, None],
            y + fracs[None, :] * np.sin(theta),
            y - R[:, None] * (np.cos(th) - np.cos(theta)),
        )
        return px, py, th, fracs

    # -- main -------------------------------------------------------------
    def plan(
        self,
        grid: OccupancyGrid,
        start: EgoState,
        goal_xy,
        risk: Optional[RiskField] = None,
        previous: Optional[Path] = None,
    ) -> PlanResult:
        cfg = self.cfg
        t_start = time.perf_counter()
        goal = np.asarray(goal_xy, dtype=np.float64)
        sample = self._make_sampler(grid)
        heur = self._geodesic_heuristic(grid, goal)
        v_ref = max(float(start.v), cfg.plan_speed_floor)
        disc_clear = self._disc_radius + cfg.inflation
        n_disc = len(self._disc_offsets)
        prev_field = self._previous_path_field(grid, previous)

        def h_of(x: float, y: float) -> float:
            euc = float(np.hypot(goal[0] - x, goal[1] - y))
            if heur is None:
                return euc
            dist, cres, (ox, oy) = heur
            ix = int((x - ox) / cres)
            iy = int((y - oy) / cres)
            if not (0 <= ix < dist.shape[1] and 0 <= iy < dist.shape[0]):
                return euc
            g = dist[iy, ix]
            return euc if not np.isfinite(g) else max(float(g), euc)

        def theta_bin(th: float) -> int:
            return int(((th + np.pi) / (2 * np.pi) * cfg.n_theta)) % cfg.n_theta

        def key_of(x, y, th):
            return (
                int(np.floor(x / cfg.xy_resolution)),
                int(np.floor(y / cfg.xy_resolution)),
                theta_bin(th),
            )

        start_key = key_of(start.x, start.y, start.theta)
        # node -> (x, y, theta, delta, arc_length, parent_key)
        nodes: Dict[tuple, tuple] = {
            start_key: (start.x, start.y, start.theta, start.delta, 0.0, None)
        }
        g_score: Dict[tuple, float] = {start_key: 0.0}
        closed: set = set()
        counter = 0
        h0 = h_of(start.x, start.y)
        heap: List[tuple] = [(h0, counter, start_key)]
        best_key, best_h = start_key, h0
        goal_key = None
        iterations = 0
        deadline = (t_start + cfg.time_budget_ms / 1000.0
                    if cfg.time_budget_ms is not None else np.inf)

        while heap:
            iterations += 1
            if iterations > cfg.max_iterations:
                break
            if np.isfinite(deadline) and time.perf_counter() > deadline:
                break
            _, _, key = heapq.heappop(heap)
            if key in closed:
                continue
            closed.add(key)
            x, y, th, delta, arc, _ = nodes[key]
            g = g_score[key]

            if np.hypot(goal[0] - x, goal[1] - y) <= cfg.goal_tolerance:
                goal_key = key
                break

            nx_, ny_, nth, steers, _, _ = self._expand(x, y, th)
            px, py, pth, fracs = self._substep_poses(x, y, th)

            # One batched clearance query for every disc of every substep of
            # every primitive: (n_steer, substeps, n_disc).
            cx = px[:, :, None] + self._disc_offsets[None, None, :] * np.cos(pth)[:, :, None]
            cy = py[:, :, None] + self._disc_offsets[None, None, :] * np.sin(pth)[:, :, None]
            sd = sample(cx.ravel(), cy.ravel()).reshape(cfg.n_steer, cfg.substeps, n_disc)
            free = (sd >= disc_clear).all(axis=(1, 2))
            min_clear = sd.min(axis=(1, 2))

            if prev_field is not None:
                hyst = prev_field(px.ravel(), py.ravel()).reshape(
                    cfg.n_steer, cfg.substeps).mean(axis=1)
            else:
                hyst = np.zeros(cfg.n_steer)

            if risk is not None:
                t_at = (arc + fracs[None, :]) / v_ref
                rk = risk.lookup(px.ravel(), py.ravel(), np.broadcast_to(t_at, px.shape).ravel())
                rk = rk.reshape(cfg.n_steer, cfg.substeps).mean(axis=1)
            else:
                rk = np.zeros(cfg.n_steer)

            L = cfg.step_length
            step_cost = (
                cfg.w_length * L
                + cfg.w_steer * np.abs(steers) * L
                + cfg.w_steer_change * np.abs(steers - delta)
                + cfg.w_risk * rk * L
                + cfg.w_clearance * np.maximum(cfg.preferred_clearance - min_clear, 0.0) * L
                + cfg.w_hysteresis * hyst * L
            )

            for i in range(cfg.n_steer):
                if not free[i]:
                    continue
                nkey = key_of(nx_[i], ny_[i], nth[i])
                if nkey in closed:
                    continue
                ng = g + float(step_cost[i])
                if ng >= g_score.get(nkey, np.inf):
                    continue
                g_score[nkey] = ng
                nodes[nkey] = (
                    float(nx_[i]), float(ny_[i]), float(nth[i]),
                    float(steers[i]), arc + L, key,
                )
                hh = h_of(nx_[i], ny_[i])
                counter += 1
                heapq.heappush(heap, (ng + hh, counter, nkey))
                if hh < best_h:
                    best_h, best_key = hh, nkey

        elapsed = (time.perf_counter() - t_start) * 1e3
        end_key = goal_key if goal_key is not None else best_key
        success = goal_key is not None
        if not success and end_key == start_key:
            return PlanResult(
                Path(np.zeros((0, 4)), True), False, "no_expansion", False,
                iterations, elapsed, float("inf"),
            )

        # Reconstruct.
        chain = []
        k = end_key
        while k is not None:
            x, y, th, delta, _, parent = nodes[k]
            chain.append((x, y, th, np.tan(delta) / self.params.wheelbase))
            k = parent
        chain.reverse()
        return PlanResult(
            Path(np.array(chain), terminal_stop=not success),
            success,
            "goal_reached" if success else "budget_exhausted",
            not success,
            iterations,
            elapsed,
            g_score.get(end_key, float("inf")),
        )
