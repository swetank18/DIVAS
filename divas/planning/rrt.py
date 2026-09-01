"""Kinodynamic RRT -- the planner fallback.

Hybrid A* searches a fixed lattice.  That is exactly what you want most of the
time, and exactly what fails in the cases that matter here: a corridor that
pinches below the lattice's resolution, or a manoeuvre that needs a heading the
24 bins do not contain.  When the lattice search exhausts its budget without
reaching the goal, sampling can still find a way through, because it is not
constrained to a grid of poses.

**On the name.**  The deck says "Hybrid A* / RRT*".  This is honestly an RRT
with RRT*'s *choose-parent* step but without its *rewire* step.  Rewiring needs
an exact steering function -- given two poses, the optimal path between them --
and for a car that means Reeds-Shepp or Dubins curves.  Arc primitives cannot
land on an arbitrary pose exactly, so rewiring would be approximate and the
asymptotic-optimality guarantee that the star denotes would not hold.  Choosing
the best parent among near nodes recovers most of the practical benefit and
claims nothing that is not true.  Adding Dubins steering is the upgrade path,
and it is a small, well-defined piece of work.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from divas.planning.hybrid_astar import PlanResult
from divas.prediction.risk import RiskField
from divas.types import EgoState, OccupancyGrid, Path, VehicleParams, wrap_angle


@dataclass
class RRTConfig:
    max_iterations: int = 800
    time_budget_ms: Optional[float] = 90.0   # None -> iteration-bounded only
    step_length: float = 2.0
    n_steer: int = 7
    substeps: int = 3
    goal_bias: float = 0.15
    goal_tolerance: float = 3.0
    near_radius: float = 6.0
    connect_pos_tol: float = 0.5     # m, when re-parenting
    connect_ang_tol: float = 0.30    # rad
    inflation: float = 0.35
    heading_weight: float = 2.0      # m per rad, for the nearest-node metric
    w_steer: float = 0.6
    w_risk: float = 26.0
    plan_speed_floor: float = 3.0


class KinodynamicRRT:
    def __init__(
        self,
        params: Optional[VehicleParams] = None,
        config: Optional[RRTConfig] = None,
        seed: int = 0,
    ) -> None:
        self.params = params or VehicleParams()
        self.cfg = config or RRTConfig()
        self.rng = np.random.default_rng(seed)
        self._steers = np.linspace(
            -self.params.max_steer, self.params.max_steer, self.cfg.n_steer
        )
        self._discs, self._disc_r = self.params.footprint_discs()

    # -- geometry ---------------------------------------------------------
    def _arcs(self, x, y, th):
        """End poses and swept substeps for every steering angle."""
        L = self.cfg.step_length
        wb = self.params.wheelbase
        tan = np.tan(self._steers)
        straight = np.abs(tan) < 1e-6
        R = np.where(straight, 1.0, wb / np.where(straight, 1.0, tan))
        fr = np.linspace(L / self.cfg.substeps, L, self.cfg.substeps)
        dth = np.where(straight, 0.0, 1.0 / R)[:, None] * fr[None, :]
        pth = th + dth
        px = np.where(straight[:, None], x + fr[None, :] * np.cos(th),
                      x + R[:, None] * (np.sin(pth) - np.sin(th)))
        py = np.where(straight[:, None], y + fr[None, :] * np.sin(th),
                      y - R[:, None] * (np.cos(pth) - np.cos(th)))
        return px, py, pth

    def plan(
        self,
        grid: OccupancyGrid,
        start: EgoState,
        goal_xy,
        risk: Optional[RiskField] = None,
    ) -> PlanResult:
        cfg = self.cfg
        t0 = time.perf_counter()
        deadline = (t0 + cfg.time_budget_ms / 1000.0
                    if cfg.time_budget_ms is not None else np.inf)
        goal = np.asarray(goal_xy, dtype=np.float64)
        v_ref = max(float(start.v), cfg.plan_speed_floor)
        clear = self._disc_r + cfg.inflation

        sdf = grid.signed_distance_field()
        ox, oy = grid.origin
        res, nx, ny = grid.resolution, grid.nx, grid.ny

        def free(px, py, pth) -> np.ndarray:
            """Is every disc of every substep clear?

            Input is ``(..., substeps)``; the result reduces over both the
            substep axis and the disc axis, so it is ``(...,)`` -- one boolean
            per candidate arc, not per pose along it.
            """
            cx = px[..., None] + self._discs * np.cos(pth)[..., None]
            cy = py[..., None] + self._discs * np.sin(pth)[..., None]
            ix = ((cx - ox) / res).astype(np.int32)
            iy = ((cy - oy) / res).astype(np.int32)
            ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
            np.clip(ix, 0, nx - 1, out=ix)
            np.clip(iy, 0, ny - 1, out=iy)
            d = np.where(ok, sdf[iy, ix], -1.0)
            return (d >= clear).all(axis=(-2, -1))

        xs = [start.x]; ys = [start.y]; ths = [start.theta]
        dels = [start.delta]; costs = [0.0]; arcs = [0.0]; parents: List[Optional[int]] = [None]

        xmin, ymin, xmax, ymax = grid.bounds
        best_goal: Optional[int] = None
        best_goal_cost = np.inf
        it = 0

        while it < cfg.max_iterations:
            if np.isfinite(deadline) and time.perf_counter() >= deadline:
                break
            it += 1
            if self.rng.random() < cfg.goal_bias:
                sample = goal
            else:
                sample = np.array([self.rng.uniform(xmin, xmax),
                                   self.rng.uniform(ymin, ymax)])

            X = np.asarray(xs); Y = np.asarray(ys); TH = np.asarray(ths)
            # Heading-aware nearest: a node pointing the wrong way is not
            # actually near, whatever its coordinates say.
            bearing = np.arctan2(sample[1] - Y, sample[0] - X)
            d = np.hypot(sample[0] - X, sample[1] - Y)
            metric = d + cfg.heading_weight * np.abs(wrap_angle(bearing - TH))
            i = int(np.argmin(metric))

            px, py, pth = self._arcs(xs[i], ys[i], ths[i])
            ok = free(px, py, pth)
            if not ok.any():
                continue
            end = np.stack([px[:, -1], py[:, -1]], axis=1)
            dist = np.linalg.norm(end - sample, axis=1)
            dist[~ok] = np.inf
            k = int(np.argmin(dist))
            if not np.isfinite(dist[k]):
                continue

            nxp, nyp, nth = float(px[k, -1]), float(py[k, -1]), float(wrap_angle(pth[k, -1]))
            step = cfg.step_length
            seg = cfg.w_steer * abs(float(self._steers[k])) * step
            if risk is not None:
                t_at = (arcs[i] + np.linspace(step / cfg.substeps, step, cfg.substeps)) / v_ref
                seg += cfg.w_risk * float(risk.lookup(px[k], py[k], t_at).mean()) * step

            # RRT* choose-parent: among nearby nodes, is there one that can
            # reach this pose by a single arc more cheaply?
            best_p, best_c, best_arc = i, costs[i] + step + seg, arcs[i] + step
            near = np.where(np.hypot(np.asarray(xs) - nxp, np.asarray(ys) - nyp)
                            <= cfg.near_radius)[0]
            for j in near.tolist():
                if j == i or costs[j] + step >= best_c:
                    continue
                qx, qy, qth = self._arcs(xs[j], ys[j], ths[j])
                reach = (np.hypot(qx[:, -1] - nxp, qy[:, -1] - nyp) <= cfg.connect_pos_tol) & (
                    np.abs(wrap_angle(qth[:, -1] - nth)) <= cfg.connect_ang_tol)
                reach &= free(qx, qy, qth)
                if not reach.any():
                    continue
                m = int(np.argmax(reach))
                c = costs[j] + step + cfg.w_steer * abs(float(self._steers[m])) * step
                if c < best_c:
                    best_p, best_c, best_arc = j, c, arcs[j] + step

            xs.append(nxp); ys.append(nyp); ths.append(nth)
            dels.append(float(self._steers[k])); costs.append(best_c)
            arcs.append(best_arc); parents.append(best_p)

            if np.hypot(nxp - goal[0], nyp - goal[1]) <= cfg.goal_tolerance and best_c < best_goal_cost:
                best_goal, best_goal_cost = len(xs) - 1, best_c

        elapsed = (time.perf_counter() - t0) * 1e3
        if best_goal is None:
            if len(xs) == 1:
                return PlanResult(Path(np.zeros((0, 4)), True), False,
                                  "no_expansion", False, it, elapsed, np.inf)
            d = np.hypot(np.asarray(xs) - goal[0], np.asarray(ys) - goal[1])
            end, success = int(np.argmin(d)), False
        else:
            end, success = best_goal, True

        chain = []
        node: Optional[int] = end
        while node is not None:
            chain.append((xs[node], ys[node], ths[node],
                          np.tan(dels[node]) / self.params.wheelbase))
            node = parents[node]
        chain.reverse()
        return PlanResult(
            Path(np.array(chain), terminal_stop=not success),
            success,
            "goal_reached" if success else "budget_exhausted",
            not success, it, elapsed, costs[end],
        )
