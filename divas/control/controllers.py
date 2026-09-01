"""Stage 6 -- tracking controllers.

Two implementations, and the pair is the point:

``PurePursuitController`` is the Phase 1 baseline and the conventional arm of
the ablation.  Geometric lateral tracking, a curvature speed limit, and a
*fixed* braking buffer -- what a competent conventional stack does.

``SamplingMPC`` is the risk-aware nonlinear MPC.  It optimises steering and
acceleration jointly over a 2 s horizon on the kinematic bicycle model, with
the :class:`RiskField` entering the cost directly, so prediction uncertainty
reaches the actuators instead of stopping at the planner.

A note on the solver, because it is a real deviation from the plan.
``EXECUTION_PLAN.md`` Phase 4 specifies acados (SQP-RTI) or OSQP.  Neither is
installed in this environment, and neither is a drop-in: both want a smooth,
differentiable cost, while the risk field here is a *rasterised, non-convex*
volume with no useful gradients.  So this is a sampling-based MPC (MPPI):
perturb a nominal control sequence, roll every sample forward in parallel,
and take a softmax-weighted update.  It is a genuine receding-horizon
optimal controller, it handles the non-convex cost natively, and it needs no
solver dependency.  The trade-off is honest: no constraint guarantees and a
cost that scales with sample count.  The acados path stays open behind the
:class:`Controller` interface, and taking it means smoothing the risk field
into an analytic sum of ellipsoids -- which :meth:`RiskField.risk_at` already
provides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from divas.planning.hybrid_astar import HybridAStar
from divas.prediction.risk import RiskField
from divas.types import ControlCmd, EgoState, OccupancyGrid, Path, VehicleParams, wrap_angle


class Controller:
    """Interface for stage 6."""

    name = "controller"

    def reset(self) -> None:
        pass

    def step(
        self,
        path: Path,
        ego: EgoState,
        grid: Optional[OccupancyGrid] = None,
        risk: Optional[RiskField] = None,
        dt: float = 0.05,
    ) -> ControlCmd:
        raise NotImplementedError


def _grid_sampler(grid: OccupancyGrid):
    sdf = grid.signed_distance_field()
    x0, y0 = grid.origin
    res, nx, ny = grid.resolution, grid.nx, grid.ny

    def sample(px, py):
        ix = ((px - x0) / res).astype(np.int32)
        iy = ((py - y0) / res).astype(np.int32)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        np.clip(ix, 0, nx - 1, out=ix)
        np.clip(iy, 0, ny - 1, out=iy)
        return np.where(ok, sdf[iy, ix], -1.0)

    return sample


# --------------------------------------------------------------------------
# baseline
# --------------------------------------------------------------------------


@dataclass
class PurePursuitController(Controller):
    """Geometric tracking with a fixed safety buffer.  The ablation baseline."""

    params: VehicleParams = field(default_factory=VehicleParams)
    lookahead_gain: float = 0.55
    lookahead_min: float = 3.0
    lookahead_max: float = 14.0
    k_speed: float = 1.1
    fixed_buffer: float = 1.0    # m -- the conventional constant margin
    reaction_time: float = 0.6   # s
    name = "pure_pursuit"

    def step(self, path, ego, grid=None, risk=None, dt=0.05) -> ControlCmd:
        if path is None or len(path) < 2:
            return ControlCmd(accel=self.params.min_accel, steer=0.0)
        p = self.params
        xy = path.xy

        # -- lateral: pure pursuit -------------------------------------
        Ld = float(np.clip(self.lookahead_gain * ego.v + self.lookahead_min,
                           self.lookahead_min, self.lookahead_max))
        d = np.linalg.norm(xy - np.array([ego.x, ego.y]), axis=1)
        i0 = int(np.argmin(d))
        ahead = np.where(d[i0:] >= Ld)[0]
        tgt = xy[i0 + ahead[0]] if ahead.size else xy[-1]
        dx, dy = tgt[0] - ego.x, tgt[1] - ego.y
        alpha = wrap_angle(np.arctan2(dy, dx) - ego.theta)
        ld = max(float(np.hypot(dx, dy)), 1e-3)
        steer = float(np.arctan2(2.0 * p.wheelbase * np.sin(alpha), ld))

        # -- longitudinal ----------------------------------------------
        v_ref = self._speed_target(path, ego, i0, grid, risk)
        accel = float(np.clip(self.k_speed * (v_ref - ego.v),
                              p.min_accel, p.max_accel))
        return ControlCmd(accel=accel, steer=float(np.clip(steer, -p.max_steer, p.max_steer)))

    def _speed_target(self, path, ego, i0, grid, risk) -> float:
        p = self.params
        v = p.cruise_speed

        # curvature comfort limit over the next ~15 m
        s = path.arc_lengths()
        window = (s >= s[i0]) & (s <= s[i0] + 15.0)
        kappa = path.curvature_profile()[window]
        if kappa.size:
            k = float(np.percentile(kappa, 90))
            if k > 1e-3:
                v = min(v, float(np.sqrt(p.max_lat_accel / k)))

        # Taper to a stop only when the path really ends.
        if path.terminal_stop:
            remaining = float(s[-1] - s[i0])
            v = min(v, float(np.sqrt(max(2.0 * abs(p.min_accel) * max(remaining - 2.0, 0.0), 0.0))))

        # fixed-buffer braking: look along the path for anything predicted to
        # be inside a constant margin, and slow to keep a constant headway
        if risk is not None and len(path) > 1:
            v_plan = max(ego.v, 1.0)
            look = (s >= s[i0]) & (s <= s[i0] + max(ego.v * 3.0, 10.0))
            if look.any():
                px, py = path.xy[look, 0], path.xy[look, 1]
                t = (s[look] - s[i0]) / v_plan
                pen = risk.penetration(px, py, t)
                hit = np.where(pen > 0.0)[0]
                if hit.size:
                    gap = float(s[look][hit[0]] - s[i0]) - self.fixed_buffer
                    gap = max(gap - ego.v * self.reaction_time, 0.0)
                    v = min(v, float(np.sqrt(2.0 * abs(p.min_accel) * gap)))
        return float(np.clip(v, 0.0, p.max_speed))


# --------------------------------------------------------------------------
# risk-aware nonlinear MPC
# --------------------------------------------------------------------------


@dataclass
class MPCConfig:
    horizon: int = 20            # N steps
    dt: float = 0.1              # s  -> 2.0 s horizon
    n_samples: int = 512
    temperature: float = 0.5     # softmax lambda
    sigma_accel: float = 1.3     # m/s^2
    sigma_steer: float = 0.10    # rad
    n_iterations: int = 2        # refinement passes per control step
    # Noise correlated over this many horizon steps.  White noise on steering
    # is nearly useless here: averaged over a 20-step horizon its mean is
    # sigma/sqrt(20), so every sampled rollout ends up within a few
    # centimetres of the nominal one and the search explores no lateral
    # manoeuvre at all.
    smooth_window: int = 7
    offset_fraction: float = 0.7  # share of sigma given to a constant offset

    w_cross_track: float = 20.0
    w_heading: float = 2.5
    # Asymmetric on purpose.  Exceeding the reference is a safety concern and is
    # punished; falling below it is merely slow.  A symmetric term makes the
    # optimiser fight to reach cruise speed with roughly twenty times the
    # weight it gives to staying on the path, so it accelerates straight into
    # the obstacle the planner just routed around.
    w_speed_over: float = 4.0
    # Not too small.  If falling short of the reference is nearly free, the
    # cheapest trajectory is to stop moving: a stationary vehicle incurs no
    # cross-track, lateral-acceleration or risk cost at all.  The classic
    # freezing robot, and it is an artifact of the weights, not of caution.
    w_speed_under: float = 1.5
    w_accel: float = 0.04
    w_steer: float = 2.0
    w_slew: float = 6.0
    # Small: the jerk limit now lives in the dynamics (VehicleParams.max_jerk),
    # so this is a comfort preference, not the constraint.  At 3.0 it was
    # ~10,000 cost units for a change of mind and froze the controller.
    w_accel_slew: float = 0.06
    w_lat_accel: float = 8.0
    w_obstacle: float = 4000.0      # contact -- must never be worth it
    w_soft_clearance: float = 45.0  # approach -- cost rises before contact
    soft_band: float = 0.6          # m of margin the soft term defends
    w_risk: float = 150.0
    w_terminal: float = 3.0
    max_cross_track: float = 3.0    # m, cap on the tracking term
    # Reward for arc length covered along the reference over the horizon.
    # Without it, braking is always available as a cheap way to make the whole
    # horizon collision-free -- a stopped vehicle accrues no obstacle, risk or
    # lateral-acceleration cost at all -- and the optimiser takes it.  Every
    # penalty term needs something pulling the other way.
    w_progress: float = 34.0


@dataclass
class SamplingMPC(Controller):
    """Receding-horizon control by weighted sampling (MPPI)."""

    params: VehicleParams = field(default_factory=VehicleParams)
    cfg: MPCConfig = field(default_factory=MPCConfig)
    seed: int = 0
    name = "risk_mpc"

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.reset()
        self._discs, self._disc_r = self.params.footprint_discs()

    def reset(self) -> None:
        self.U = np.zeros((self.cfg.horizon, 2))  # nominal (accel, steer)
        self.last_accel = 0.0

    def _noise(self, K: int) -> np.ndarray:
        """Temporally correlated exploration noise.

        Two components: a smoothed sequence (sustained manoeuvres) and a
        constant per-sample offset (committed manoeuvres).  Both are needed --
        the offset finds the turn, the smoothed part shapes it.
        """
        cfg = self.cfg
        sig = np.array([cfg.sigma_accel, cfg.sigma_steer])
        w = max(int(cfg.smooth_window), 1)
        raw = self.rng.normal(size=(K, cfg.horizon, 2))
        smooth = uniform_filter1d(raw, size=w, axis=1, mode="nearest") * np.sqrt(w)
        offset = self.rng.normal(size=(K, 1, 2))
        f = cfg.offset_fraction
        return sig * (np.sqrt(1.0 - f) * smooth + np.sqrt(f) * offset)

    def _feedforward(self, path: Path, ego: EgoState) -> np.ndarray:
        """A geometric prior: pure pursuit rolled forward over the horizon.

        MPPI refines a nominal sequence; giving it a sensible one instead of
        zeros is the difference between refining a manoeuvre and searching
        for one.  It also bounds the downside -- the sampler is seeded with
        the conventional controller's answer, so it can improve on the
        geometric baseline but never fall below it by accident.
        """
        p = self.params
        cfg = self.cfg
        U = np.zeros((cfg.horizon, 2))
        x, y, th, v, delta = ego.x, ego.y, ego.theta, ego.v, ego.delta
        xy = path.xy
        s = path.arc_lengths()
        kprof = path.curvature_profile()
        for i in range(cfg.horizon):
            Ld = float(np.clip(0.55 * v + 3.0, 3.0, 14.0))
            d = np.linalg.norm(xy - np.array([x, y]), axis=1)
            i0 = int(np.argmin(d))
            ahead = np.where(d[i0:] >= Ld)[0]
            tgt = xy[i0 + ahead[0]] if ahead.size else xy[-1]
            alpha = wrap_angle(np.arctan2(tgt[1] - y, tgt[0] - x) - th)
            ld = max(float(np.hypot(tgt[0] - x, tgt[1] - y)), 1e-3)
            steer = float(np.clip(np.arctan2(2.0 * p.wheelbase * np.sin(alpha), ld),
                                  -p.max_steer, p.max_steer))
            v_ref = min(float(np.sqrt(p.max_lat_accel / max(kprof[i0], 1e-3))), p.cruise_speed)
            if path.terminal_stop:
                rem = float(s[-1] - s[i0])
                v_ref = min(v_ref, float(np.sqrt(max(2.0 * abs(p.min_accel) * max(rem - 2.0, 0.0), 0.0))))
            accel = float(np.clip(1.1 * (v_ref - v), p.min_accel, p.max_accel))
            U[i] = (accel, steer)
            md = p.max_steer_rate * cfg.dt
            delta += float(np.clip(steer - delta, -md, md))
            x += v * np.cos(th) * cfg.dt
            y += v * np.sin(th) * cfg.dt
            th += v / p.wheelbase * np.tan(delta) * cfg.dt
            v = float(np.clip(v + accel * cfg.dt, 0.0, p.max_speed))
        return U

    # -- rollout ----------------------------------------------------------
    def _rollout(self, ego: EgoState, U: np.ndarray) -> Tuple[np.ndarray, ...]:
        """Propagate every sample through the kinematic bicycle model.

        ``U`` is ``(K, N, 2)``.  Returns per-step state arrays of shape
        ``(K, N)``.  Everything is vectorised over samples; the loop is only
        over the horizon, which is what keeps a 400-sample solve at a few
        milliseconds in pure numpy.
        """
        p = self.params
        cfg = self.cfg
        K = U.shape[0]
        x = np.full(K, ego.x)
        y = np.full(K, ego.y)
        th = np.full(K, ego.theta)
        v = np.full(K, ego.v)
        delta = np.full(K, ego.delta)
        accel = np.full(K, ego.a)

        X = np.empty((K, cfg.horizon))
        Y = np.empty((K, cfg.horizon))
        TH = np.empty((K, cfg.horizon))
        V = np.empty((K, cfg.horizon))
        DEL = np.empty((K, cfg.horizon))
        for i in range(cfg.horizon):
            a = np.clip(U[:, i, 0], p.min_accel, p.max_accel)
            # Same jerk limit the vehicle has, so the rollout predicts what
            # the actuator will actually do rather than what was asked for.
            accel = np.clip(a, accel - p.max_jerk * cfg.dt, accel + p.max_jerk * cfg.dt)
            a = accel
            cmd = np.clip(U[:, i, 1], -p.max_steer, p.max_steer)
            # steering rate limit -- a controller that ignores it produces
            # beautiful trajectories the steering rack cannot execute
            delta = delta + np.clip(cmd - delta, -p.max_steer_rate * cfg.dt,
                                    p.max_steer_rate * cfg.dt)
            x = x + v * np.cos(th) * cfg.dt
            y = y + v * np.sin(th) * cfg.dt
            th = th + v / p.wheelbase * np.tan(delta) * cfg.dt
            v = np.clip(v + a * cfg.dt, 0.0, p.max_speed)
            X[:, i], Y[:, i], TH[:, i], V[:, i], DEL[:, i] = x, y, th, v, delta
        return X, Y, TH, V, DEL

    def _cost(self, X, Y, TH, V, DEL, U, path, grid, risk) -> np.ndarray:
        return self._cost_terms(X, Y, TH, V, DEL, U, path, grid, risk)[0]

    def _cost_terms(self, X, Y, TH, V, DEL, U, path, grid, risk):
        """Total cost, and the per-term breakdown that produced it.

        The breakdown exists because tuning a multi-term cost by nudging
        weights and watching the vehicle is guesswork.  Every pathology this
        controller has had -- freezing, crawling, grazing obstacles -- was one
        term quietly outweighing the rest by an order of magnitude, and each
        became obvious the moment the terms were printed side by side.
        """
        cfg = self.cfg
        p = self.params
        K, N = X.shape
        terms = {}

        # -- reference tracking: nearest point on the planned path
        ref = path.xy                       # (P, 2)
        dx = X[:, :, None] - ref[None, None, :, 0]
        dy = Y[:, :, None] - ref[None, None, :, 1]
        d2 = dx * dx + dy * dy
        idx = np.argmin(d2, axis=2)         # (K, N)
        cte2 = np.take_along_axis(d2, idx[:, :, None], axis=2)[:, :, 0]
        head_err = wrap_angle(TH - path.theta[idx])

        # Bounded, so that a rollout running past the end of a short path
        # cannot dominate every other term.  A partial plan is exactly when
        # the controller most needs its other costs to still matter.
        cte2 = np.minimum(cte2, cfg.max_cross_track**2)
        cost = cfg.w_cross_track * cte2
        terms["cross_track"] = cost.sum(axis=1).copy()
        t = cfg.w_heading * head_err**2
        terms["heading"] = t.sum(axis=1); cost = cost + t

        # -- speed reference from path curvature and remaining length
        v_ref = self._speed_reference(path, idx)
        t = (cfg.w_speed_over * np.maximum(V - v_ref, 0.0) ** 2
             + cfg.w_speed_under * np.maximum(v_ref - V, 0.0) ** 2)
        terms["speed"] = t.sum(axis=1); cost = cost + t

        # -- effort and smoothness
        t = cfg.w_accel * U[:, :, 0] ** 2 + cfg.w_steer * U[:, :, 1] ** 2
        terms["effort"] = t.sum(axis=1); cost = cost + t
        slew = np.diff(DEL, axis=1, prepend=DEL[:, :1])
        t = cfg.w_slew * (slew / cfg.dt) ** 2
        # Longitudinal jerk, measured from the control actually applied last
        # step rather than from the start of this horizon.  Without the
        # anchor the optimiser is free to jump the first command every cycle,
        # which is comfortable on a plot and violent in the vehicle.
        a_prev = np.full((K, 1), self.last_accel)
        a_slew = np.diff(U[:, :, 0], axis=1, prepend=a_prev)
        t = t + cfg.w_accel_slew * (a_slew / cfg.dt) ** 2
        terms["smoothness"] = t.sum(axis=1); cost = cost + t
        lat = V**2 * np.tan(DEL) / p.wheelbase
        t = cfg.w_lat_accel * np.maximum(np.abs(lat) - p.max_lat_accel, 0.0) ** 2
        terms["lat_accel"] = t.sum(axis=1); cost = cost + t

        # -- hard geometry: the footprint against the occupancy grid
        if grid is not None:
            sample = _grid_sampler(grid)
            cx = X[:, :, None] + self._discs[None, None, :] * np.cos(TH)[:, :, None]
            cy = Y[:, :, None] + self._discs[None, None, :] * np.sin(TH)[:, :, None]
            sd = sample(cx.ravel(), cy.ravel()).reshape(K, N, len(self._discs))
            worst = sd.min(axis=2)
            # Two tiers.  The hard term alone is not enough: at contact its
            # penetration is small, so its cost is comparable to a couple of
            # metres of cross-track error and the optimiser cheerfully trades
            # one for the other.  The soft term makes approaching expensive
            # long before touching is possible.
            t = cfg.w_obstacle * np.maximum(self._disc_r - worst, 0.0) ** 2
            terms["obstacle_hard"] = t.sum(axis=1); cost = cost + t
            t = cfg.w_soft_clearance * np.maximum(
                self._disc_r + cfg.soft_band - worst, 0.0) ** 2
            terms["obstacle_soft"] = t.sum(axis=1); cost = cost + t

        # -- predicted traffic, through the dynamic safety margin
        if risk is not None:
            t = (np.arange(1, N + 1) * cfg.dt)[None, :]
            r = risk.lookup(X.ravel(), Y.ravel(),
                            np.broadcast_to(t, X.shape).ravel()).reshape(K, N)
            t = cfg.w_risk * r
            terms["risk"] = t.sum(axis=1); cost = cost + t

        total = cost.sum(axis=1)
        t = cfg.w_terminal * cfg.w_cross_track * cte2[:, -1]
        terms["terminal"] = t; total = total + t
        s = path.arc_lengths()
        t = -cfg.w_progress * (s[idx[:, -1]] - s[idx[:, 0]])
        terms["progress"] = t; total = total + t
        return total, terms

    def explain(self, path, ego, grid=None, risk=None) -> dict:
        """Per-term cost of the current nominal control sequence.  Debug aid."""
        if path is None or len(path) < 2:
            return {}
        path = path.resample(0.4) if path.length > 1.0 else path
        U = self.U[None]
        X, Y, TH, V, DEL = self._rollout(ego, U)
        total, terms = self._cost_terms(X, Y, TH, V, DEL, U, path, grid, risk)
        out = {k: float(v[0]) for k, v in terms.items()}
        out["TOTAL"] = float(total[0])
        return out

    def _speed_reference(self, path: Path, idx: np.ndarray) -> np.ndarray:
        p = self.params
        kappa = path.curvature_profile()[idx]
        v_curve = np.minimum(
            np.sqrt(p.max_lat_accel / np.maximum(kappa, 1e-3)), p.cruise_speed
        )
        if not path.terminal_stop:
            return np.minimum(v_curve, p.max_speed)
        s = path.arc_lengths()
        remaining = s[-1] - s[idx]
        v_end = np.sqrt(np.maximum(2.0 * abs(p.min_accel) * np.maximum(remaining - 2.0, 0.0), 0.0))
        return np.minimum(np.minimum(v_curve, v_end), p.max_speed)

    # -- main -------------------------------------------------------------
    def step(self, path, ego, grid=None, risk=None, dt=0.05) -> ControlCmd:
        p = self.params
        cfg = self.cfg
        if path is None or len(path) < 2:
            self.reset()
            return ControlCmd(accel=p.min_accel, steer=0.0)

        path = path.resample(0.4) if path.length > 1.0 else path

        # Start from whichever nominal is better: the shifted previous
        # solution or the fresh geometric prior.  After a replan the warm
        # start can be tracking a path that no longer exists.
        seeds = np.stack([self.U, self._feedforward(path, ego)])
        Xs, Ys, THs, Vs, DELs = self._rollout(ego, seeds)
        seed_cost = self._cost(Xs, Ys, THs, Vs, DELs, seeds, path, grid, risk)
        U = seeds[int(np.argmin(seed_cost))].copy()

        for _ in range(cfg.n_iterations):
            eps = self._noise(cfg.n_samples)
            cand = U[None] + eps
            cand[:, :, 0] = np.clip(cand[:, :, 0], p.min_accel, p.max_accel)
            cand[:, :, 1] = np.clip(cand[:, :, 1], -p.max_steer, p.max_steer)
            X, Y, TH, V, DEL = self._rollout(ego, cand)
            S = self._cost(X, Y, TH, V, DEL, cand, path, grid, risk)
            # Softmax weighting.  Subtracting the minimum before the
            # exponent is not cosmetic: without it the exponent underflows to
            # all zeros and the update becomes NaN.
            #
            # The temperature is scaled by the spread of the *good* samples
            # (min to lower quartile), not by the standard deviation of all
            # of them.  With an obstacle in view a handful of samples score
            # in the thousands, which inflates the std, flattens the softmax
            # towards uniform, and cancels the update -- the optimiser goes
            # blind precisely when there is something to avoid.
            S0 = S.min()
            scale = float(np.percentile(S, 25) - S0)
            w = np.exp(-(S - S0) / max(cfg.temperature * scale, 1e-6))
            w /= w.sum()
            U = np.einsum("k,kij->ij", w, cand)

        U[:, 0] = np.clip(U[:, 0], p.min_accel, p.max_accel)
        U[:, 1] = np.clip(U[:, 1], -p.max_steer, p.max_steer)
        cmd = ControlCmd(accel=float(U[0, 0]), steer=float(U[0, 1]))
        self.last_accel = cmd.accel
        # Warm start: shift the horizon forward one step.
        self.U = np.roll(U, -1, axis=0)
        self.U[-1] = U[-1]
        return cmd
