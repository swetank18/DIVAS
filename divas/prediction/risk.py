"""The risk field -- how prediction uncertainty reaches the planner and the
controller.

This module is where the project's central claim becomes a number.  Predicted
actors are *not* handed downstream as hard obstacles; they are handed down as a
time-indexed field whose extent is governed by

    d_safe(t) = d0 + k_v * v_ego + lam * (1 - confidence(t))

Every term earns its place.  ``d0`` is the standstill buffer.  ``k_v * v_ego``
is the speed-dependent part every stack has.  ``lam * (1 - confidence(t))`` is
the contribution of this project: the buffer around a predicted actor *widens*
when the predictor is unsure and *tightens* when it is sure, at every step of
the horizon independently.  Setting ``lam = 0`` recovers the conventional fixed
policy, which is exactly the ablation Phase 4 has to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from divas.types import OccupancyGrid, TrajectorySet


@dataclass
class MarginParams:
    """Coefficients of the dynamic safety margin."""

    # Calibrated against the actor extents these are *added to*: the keep-out
    # already includes the actor's half-length plus the ego's radius, so a
    # motorcycle carries ~2.1 m before any margin at all.  An earlier setting
    # (k_v = 0.25, lam = 1.2, cap 4.0) produced ~5 m semi-axes, which blocks a
    # 10 m carriageway outright -- the stack was not cautious, it was blind.
    d0: float = 0.4          # m, standstill buffer
    k_v: float = 0.10        # s, speed-proportional term
    lam: float = 0.8         # m, full-uncertainty widening
    max_margin: float = 1.8  # m, cap so a hopeless prediction cannot freeze us

    @staticmethod
    def fixed(value: float = 1.0) -> "MarginParams":
        """Conventional fixed buffer -- the ablation baseline."""
        return MarginParams(d0=value, k_v=0.0, lam=0.0, max_margin=value)


class RiskField:
    """Spatiotemporal risk induced by a :class:`TrajectorySet`.

    Provides two access paths on purpose:

    * :meth:`risk_at` -- analytic, smooth, differentiable.  For the MPC, which
      queries a few hundred points per solve and needs gradients.
    * :meth:`rasterize` / :meth:`lookup` -- precomputed volume.  For the
      planner, which queries tens of thousands of points and needs speed far
      more than it needs smoothness.
    """

    def __init__(
        self,
        traj_set: TrajectorySet,
        ego_speed: float,
        ego_extent: Tuple[float, float],
        margin: Optional[MarginParams] = None,
    ) -> None:
        self.ts = traj_set
        self.margin = margin or MarginParams()
        # A single circumscribed radius for the ego is the same modelling
        # error as a disc for a bus: it charges the *lateral* keep-out for the
        # vehicle's full diagonal.  Summing the two boxes axis-wise is both
        # more correct and materially less conservative sideways, which is
        # the direction that decides whether a gap is passable.
        self.ego_half_length, self.ego_half_width = (float(v) for v in ego_extent)
        self.ego_radius = float(np.hypot(*ego_extent))
        self.ego_speed = float(ego_speed)
        self.dt = traj_set.dt
        self.horizon = traj_set.horizon
        self.n_steps = traj_set.n_steps

        # Flatten (trajectory, mode) into parallel arrays once; every query
        # below is then a single vectorised expression.
        pts, weights, semi_a, semi_b, margins = [], [], [], [], []
        for tr in traj_set:
            conf = tr.confidence_profile()                       # (T,)
            d_safe = (
                self.margin.d0
                + self.margin.k_v * self.ego_speed
                + self.margin.lam * (1.0 - conf)
            )
            d_safe = np.minimum(d_safe, self.margin.max_margin)
            w = tr.weights()
            for k, mode in enumerate(tr.modes):
                pts.append(mode.points)                              # (T, 2)
                weights.append(w[k])
                semi_a.append(tr.half_length + self.ego_half_length + d_safe)
                semi_b.append(tr.half_width + self.ego_half_width + d_safe)
                margins.append(d_safe)
        if pts:
            self.points = np.stack(pts)          # (M, T, 2)
            self.weights = np.array(weights)     # (M,)
            self.a = np.stack(semi_a)            # (M, T) along heading
            self.b = np.stack(semi_b)            # (M, T) across heading
            self.d_safe = np.stack(margins)      # (M, T)
            self.cos_h, self.sin_h = self._headings(self.points)
        else:
            self.points = np.zeros((0, self.n_steps, 2))
            self.weights = np.zeros(0)
            self.a = np.zeros((0, self.n_steps))
            self.b = np.zeros((0, self.n_steps))
            self.d_safe = np.zeros((0, self.n_steps))
            self.cos_h = np.zeros((0, self.n_steps))
            self.sin_h = np.zeros((0, self.n_steps))
        self.radii = np.maximum(self.a, self.b)  # circumscribed, for bboxes
        # Seconds elapsed since this prediction was made.  The controller
        # runs faster than the predictor, so without this every query between
        # updates reads the risk field one step too early.
        self.age: float = 0.0
        self._volume: Optional[np.ndarray] = None
        self._slice_times: Optional[np.ndarray] = None
        self._grid: Optional[OccupancyGrid] = None

    # -- reporting --------------------------------------------------------
    @staticmethod
    def _headings(points: np.ndarray):
        """Per-step heading of each mode, from finite differences."""
        d = np.diff(points, axis=1, prepend=points[:, :1] - np.diff(points[:, :2], axis=1))
        n = np.linalg.norm(d, axis=2)
        safe = n > 1e-6
        cos_h = np.where(safe, d[..., 0] / np.maximum(n, 1e-9), 1.0)
        sin_h = np.where(safe, d[..., 1] / np.maximum(n, 1e-9), 0.0)
        return cos_h, sin_h

    def margin_profile(self) -> np.ndarray:
        """``(M, T)`` of ``d_safe`` -- the margin itself, not the keep-out size.

        This is the quantity the Phase 4 ablation table reports.
        """
        return self.d_safe

    def mean_margin(self) -> float:
        return float(self.d_safe.mean()) if self.d_safe.size else 0.0

    # -- analytic ---------------------------------------------------------
    def _step_index(self, t) -> np.ndarray:
        return np.clip(
            np.round((np.asarray(t) + self.age) / self.dt).astype(int) - 1,
            0,
            self.n_steps - 1,
        )

    def risk_at(self, x, y, t) -> np.ndarray:
        """Risk in [0, inf) at world points ``(x, y)`` at time ``t``.

        Each (mode, step) contributes ``w * (1 - u)^2`` where ``u`` is the
        normalised elliptical distance to its keep-out (``u < 1`` is inside),
        and zero outside -- a smooth, compactly-supported bump.  Compact
        support matters: it keeps distant traffic from biasing the cost
        everywhere, which is how risk fields usually end up making a vehicle
        drive down the middle of an empty road for no reason.
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if self.points.shape[0] == 0:
            return np.zeros_like(x)
        i = np.broadcast_to(self._step_index(t), x.shape)
        u = self._ellipse_u(x, y, i)
        bump = np.clip(1.0 - u, 0.0, 1.0) ** 2
        return (self.weights.reshape((-1,) + (1,) * x.ndim) * bump).sum(axis=0)

    def _ellipse_u(self, x, y, i) -> np.ndarray:
        """Normalised elliptical distance: ``u < 1`` is inside the keep-out."""
        p = self.points[:, i, :]                       # (M, ..., 2)
        dx = x[None] - p[..., 0]
        dy = y[None] - p[..., 1]
        c, s = self.cos_h[:, i], self.sin_h[:, i]
        along = c * dx + s * dy
        across = -s * dx + c * dy
        return np.sqrt(
            (along / np.maximum(self.a[:, i], 1e-6)) ** 2
            + (across / np.maximum(self.b[:, i], 1e-6)) ** 2
        )

    def penetration(self, x, y, t) -> np.ndarray:
        """Depth inside the *most likely* violated keep-out, in metres.

        The MPC uses this rather than :meth:`risk_at` because a constraint
        wants a distance, not a score.
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        if self.points.shape[0] == 0:
            return np.zeros_like(x)
        i = np.broadcast_to(self._step_index(t), x.shape)
        u = self._ellipse_u(x, y, i)
        # Depth converted to metres through the *minor* axis: the conservative
        # choice, since that is the direction a lateral incursion happens in.
        depth = np.maximum(1.0 - u, 0.0) * self.b[:, i]
        return depth.max(axis=0)

    # -- rasterised -------------------------------------------------------
    def rasterize(self, grid: OccupancyGrid, n_slices: int = 12) -> np.ndarray:
        """Precompute ``(n_slices, ny, nx)`` risk for fast planner queries.

        Only cells within each bump's radius are touched, so cost scales with
        the traffic present rather than with the map size.
        """
        self._grid = grid
        self._slice_times = np.linspace(
            self.dt, self.horizon, n_slices, dtype=np.float64
        )
        vol = np.zeros((n_slices, grid.ny, grid.nx), dtype=np.float32)
        if self.points.shape[0] == 0:
            self._volume = vol
            return vol

        x0, y0 = grid.origin
        res = grid.resolution
        steps = self._step_index(self._slice_times)
        for si, step in enumerate(steps):
            for m in range(self.points.shape[0]):
                px, py = self.points[m, step]
                r = float(self.radii[m, step])
                w = float(self.weights[m])
                if r <= 0.0 or w <= 1e-4:
                    continue
                aa = float(self.a[m, step])
                bb = float(self.b[m, step])
                ch, sh = float(self.cos_h[m, step]), float(self.sin_h[m, step])
                ix0 = max(int(np.floor((px - r - x0) / res)), 0)
                ix1 = min(int(np.ceil((px + r - x0) / res)) + 1, grid.nx)
                iy0 = max(int(np.floor((py - r - y0) / res)), 0)
                iy1 = min(int(np.ceil((py + r - y0) / res)) + 1, grid.ny)
                if ix0 >= ix1 or iy0 >= iy1:
                    continue
                cx = x0 + (np.arange(ix0, ix1) + 0.5) * res
                cy = y0 + (np.arange(iy0, iy1) + 0.5) * res
                dx = cx[None, :] - px
                dy = cy[:, None] - py
                along = ch * dx + sh * dy
                across = -sh * dx + ch * dy
                u = np.sqrt((along / aa) ** 2 + (across / bb) ** 2)
                vol[si, iy0:iy1, ix0:ix1] += (
                    w * np.clip(1.0 - u, 0.0, 1.0) ** 2
                ).astype(np.float32)
        self._volume = vol
        return vol

    def lookup(self, x, y, t) -> np.ndarray:
        """Nearest-neighbour query into the rasterised volume."""
        if self._volume is None or self._grid is None:
            return self.risk_at(x, y, t)
        g = self._grid
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        t = np.broadcast_to(np.asarray(t, dtype=np.float64), x.shape)
        # Nearest slice, not searchsorted: searchsorted rounds *up* to the
        # next slice, which reads risk from a future the query did not ask
        # about and shifts every planner cost by half a slice.
        si = np.abs((t + self.age)[..., None] - self._slice_times).argmin(axis=-1)
        ok = g.in_bounds(x, y)
        ix, iy = g.world_to_cell(x, y)
        out = np.zeros(x.shape, dtype=np.float32)
        if np.any(ok):
            out[ok] = self._volume[si[ok], iy[ok], ix[ok]]
        return out
