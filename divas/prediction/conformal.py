"""Conformal calibration of the safety margin.

The project's headline mechanism is a buffer that widens when the predictor is
unsure::

    d_safe(t) = d0 + k_v * v_ego + lam * (1 - confidence(t))

``confidence`` there is a *heuristic*: the spatial dispersion of the
predictor's own modes (ADR-004).  The Phase 1 ablation ran the control arm that
almost no paper proposing an adaptive margin runs -- a fixed margin equal to
the dynamic arm's mean -- and the two scored identically, 0.90 success and 0.08
collisions.  So margin **size** explained the effect and margin **variation**
did not.

The diagnosis is not that adaptivity is worthless.  It is that mode dispersion
measures *the predictor disagreeing with itself*, which is not the same
quantity as *the predictor being wrong*.  A single-mode constant-velocity
prediction has zero dispersion and unbounded error; three modes that straddle
the truth have large dispersion and small error.  Nothing in the heuristic is
tied to observed accuracy, so there is no reason for it to track risk.

This module replaces it with the quantity the margin actually wants:

    d_safe(t) = d0 + k_v * v_ego + q_{1-alpha}(t)

where ``q_{1-alpha}(t)`` is the **conformal quantile of the predictor's own
recent errors at horizon offset t**.  The margin is no longer a tuned
coefficient multiplying a heuristic; it is a calibrated statement about how far
wrong this predictor has actually been, lately, at that lookahead.

**What that buys, precisely.**  Split conformal prediction gives, for
exchangeable residuals, a distribution-free finite-sample guarantee::

    P( ||actual - predicted|| <= q_{1-alpha} )  >=  1 - alpha

with no assumption about the predictor, the noise, or the traffic.  It holds
for constant velocity exactly as it holds for a transformer.  That is the
property a hand-tuned ``lam`` cannot have at any value.

**Where the guarantee is weaker than the textbook, and this is stated rather
than glossed.**  Closed-loop driving violates exchangeability in two ways: the
ego's own actions change the distribution of what it observes next, and traffic
is non-stationary.  Two things follow.  First, the window is rolling, so the
calibration tracks the current regime rather than the average of the episode.
Second, ``alpha`` *can* be adapted online in the manner of Gibbs & Candes
(2021), *Adaptive Conformal Inference Under Distribution Shift*, whose
guarantee is on long-run coverage and survives arbitrary shift.

It is **off by default**, and the reason is a measurement rather than a
preference.  With the rolling window at its default 240 the window itself
already tracks a shift, and ACI then becomes a second feedback loop correcting
an error the first has fixed: it overshoots, ``alpha_t`` settles near 0.13, and
coverage comes out *worse* than plain rolling conformal -- 0.844 against 0.880.
It only pays when the window is too long to adapt, which is what it was
designed for: at a 4000-sample window it recovers 0.771 to 0.814.  See
:class:`ConformalConfig`.

**The claim this module is allowed to support.**  That the margin is
*calibrated* -- that its empirical coverage matches its nominal target -- is a
measurable property, and :meth:`ConformalCalibrator.coverage` measures it.
Whether calibration also *reduces collisions* is a separate question that the
ablation answers, and it answers it with the same control arm that killed the
heuristic: a fixed margin equal to the conformal arm's own mean.  If that
control arm ties again, the honest conclusion is again that size beat
variation, and it must be reported the same way.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from divas.types import Track, TrajectorySet


@dataclass
class ConformalConfig:
    """Everything about the calibration that is a choice rather than a fact."""

    #: Target miss rate. 0.1 means the margin should contain the actor's true
    #: position 90% of the time, per horizon step.
    alpha: float = 0.1
    #: Residuals kept per horizon step. At 10 Hz prediction with a handful of
    #: actors this is a few seconds of history -- long enough for a stable
    #: quantile, short enough to track a change of regime.
    window: int = 240
    #: Below this many samples the quantile is not yet meaningful and the
    #: prior below is used instead. ceil((n+1)(1-alpha)) <= n requires
    #: n >= 1/alpha - 1, so 30 is comfortably past the floor for alpha=0.1.
    min_samples: int = 30
    #: Fallback margin while a horizon step is still cold, metres per second
    #: of lookahead. A constant-velocity predictor drifts roughly with the
    #: actor's own speed error, so a linear prior in lookahead time is the
    #: right shape to start from.
    prior_rate: float = 0.55
    #: Adaptive conformal inference (Gibbs & Candes 2021). **Off by default,
    #: on a measurement rather than on the citation.** Post-shift coverage,
    #: scoring each residual against the margin that was actually in force:
    #:
    #:     window   plain    ACI     delta
    #:        240   0.880   0.844   -0.036
    #:       1200   0.848   0.834   -0.014
    #:       4000   0.771   0.814   +0.043
    #:
    #: At the default window the rolling quantile already tracks the shift,
    #: and ACI is a second, slower feedback loop correcting an error the first
    #: one has already fixed -- it overshoots, alpha_t settles near 0.13, and
    #: the margin systematically under-covers. It earns its place only when
    #: the window is too long to adapt on its own, which is the case it was
    #: designed for. Turn it on with a long window, not with a short one.
    adaptive: bool = False
    #: ACI step size. Gibbs & Candes use 0.005-0.05; larger tracks shift
    #: faster and is noisier.
    gamma: float = 0.02
    #: Hard cap, metres. A margin larger than the carriageway does not make the
    #: vehicle careful, it makes it stuck -- the same trap ADR-003 records for
    #: the tuned version.
    max_margin: float = 2.5


class ConformalCalibrator:
    """Rolling conformal quantiles of prediction error, per horizon step.

    Used once per prediction cycle, in this order::

        cal.observe(tracks, t)      # score predictions whose target time arrived
        ts = predictor.predict(...)
        cal.record(ts, t)           # store the new ones for later scoring
        risk = RiskField(..., conformal=cal)

    ``observe`` before ``record`` matters: a prediction made *now* for
    ``now + 0`` would otherwise be scored against the observation that produced
    it, which is a residual of exactly zero and would drag every quantile down.
    """

    def __init__(self, n_steps: int, dt: float = 0.1,
                 cfg: Optional[ConformalConfig] = None) -> None:
        self.cfg = cfg or ConformalConfig()
        self.n_steps = int(n_steps)
        self.dt = float(dt)
        #: Nonconformity scores per horizon step, newest last.
        self._scores: List[Deque[float]] = [
            deque(maxlen=self.cfg.window) for _ in range(self.n_steps)
        ]
        #: Outstanding predictions: actor id -> list of
        #: ``(target_t, step, xy, margin_in_force, range_at_prediction)``.
        self._pending: Dict[int, List[Tuple[float, int, np.ndarray, float]]] = {}
        #: Cached quantile per step, cleared when that step's window changes.
        #: Without it ``observe`` re-partitions the whole window for every
        #: residual it scores, which is quadratic in the window size and turns
        #: a long calibration sweep into an overnight job.
        self._q_cache: List[Optional[float]] = [None] * self.n_steps
        #: ACI state. alpha_t is the *effective* level; cfg.alpha is the target.
        self._alpha_t = float(self.cfg.alpha)
        self._hits = 0
        self._total = 0
        self._per_step_hits = np.zeros(self.n_steps, dtype=np.int64)
        self._per_step_total = np.zeros(self.n_steps, dtype=np.int64)
        #: ``(step, range_at_prediction, error)`` per scored residual, kept
        #: only when :attr:`keep_samples` is set -- off in the control loop,
        #: where it would grow without bound, and on in the measurement
        #: harness, where it is what lets the margin be conditioned on range.
        self.samples: List[Tuple[int, float, float]] = []
        self.keep_samples = False

    # -- calibration ------------------------------------------------------
    def observe(self, tracks: List[Track], t: float) -> int:
        """Score every pending prediction whose target time has arrived.

        Returns how many residuals were added, which is worth asserting on in
        a test: a calibrator that never scores anything reports a perfectly
        stable quantile forever, and looks like it is working.
        """
        if not tracks:
            self._expire(t)
            return 0
        actual = {int(tr.id): np.array([tr.x, tr.y], dtype=np.float64)
                  for tr in tracks}
        added = 0
        half = 0.5 * self.dt
        for aid, entries in list(self._pending.items()):
            keep: List[Tuple[float, int, np.ndarray]] = []
            for target_t, step, xy, q_then, rng_m in entries:
                if target_t > t + half:
                    keep.append((target_t, step, xy, q_then, rng_m))
                    continue
                here = actual.get(aid)
                # An actor that has left the scene is dropped, not scored.
                # Scoring it against its last known position would credit the
                # predictor for a disappearance it did not predict.
                if here is not None and t - target_t <= half:
                    err = float(np.linalg.norm(here - xy))
                    # Judged against the margin that was in force when the
                    # prediction was made, not one recomputed now -- which
                    # would contain this very residual and bias the answer
                    # towards success. This is the question the vehicle
                    # actually faced, and it is the honest one to score.
                    covered = err <= q_then
                    self._scores[step].append(err)
                    self._q_cache[step] = None
                    if self.keep_samples:
                        self.samples.append((step, rng_m, err))
                    self._hits += int(covered)
                    self._total += 1
                    self._per_step_hits[step] += int(covered)
                    self._per_step_total[step] += 1
                    self._update_alpha(covered)
                    added += 1
            if keep:
                self._pending[aid] = keep
            else:
                self._pending.pop(aid, None)
        return added

    def record(self, traj_set: TrajectorySet, t: float,
               ego_xy: Optional[Tuple[float, float]] = None) -> None:
        """Store this cycle's predictions so they can be scored when due.

        The *mean* path is stored rather than the modes. The margin is a single
        radius applied to a single keep-out, so the quantity being calibrated
        has to be the error of the single path the risk field actually places.

        The margin recorded alongside is the one **as applied** -- from
        :meth:`margins`, so capped -- not the raw quantile. Coverage then
        answers the question that matters: was the keep-out the vehicle
        actually carried wide enough? When the cap binds, as it does on the
        conformal arm at 2.45 m of a 2.5 m ceiling, the honest answer is no,
        and scoring against the uncapped quantile would hide exactly that.
        It is also one computation per cycle rather than one per horizon step
        per actor.
        """
        in_force = self.margins()
        ego = np.asarray(ego_xy, dtype=np.float64) if ego_xy is not None else None
        for tr in traj_set:
            path = tr.mean_path()
            n = min(len(path), self.n_steps)
            entries = self._pending.setdefault(int(tr.track_id), [])
            # Range at the moment of prediction. Recorded because the sensor
            # noise this margin must cover grows with it, so a quantile pooled
            # over a 60 m track range describes no actor in particular.
            rng_m = (float(np.linalg.norm(np.asarray(path[0]) - ego))
                     if ego is not None and n else 0.0)
            for k in range(n):
                entries.append((t + (k + 1) * self.dt, k,
                                np.asarray(path[k], dtype=np.float64),
                                float(in_force[k]), rng_m))

    def _expire(self, t: float) -> None:
        """Drop predictions whose moment passed with nothing to score against."""
        half = 0.5 * self.dt
        for aid, entries in list(self._pending.items()):
            keep = [e for e in entries if e[0] > t + half]
            if keep:
                self._pending[aid] = keep
            else:
                self._pending.pop(aid, None)

    def _update_alpha(self, covered: bool) -> None:
        """Adaptive conformal inference, Gibbs & Candes (2021).

        ``alpha_{t+1} = alpha_t + gamma * (alpha - err_t)`` with ``err_t`` the
        realised miss. Under-coverage pushes alpha down, which widens the
        interval; over-coverage pushes it up and tightens it. Clamped away from
        both ends: at alpha <= 0 the quantile is +inf and the vehicle freezes,
        and at alpha >= 1 there is no margin at all.
        """
        if not self.cfg.adaptive:
            return
        err = 0.0 if covered else 1.0
        before = self._alpha_t
        self._alpha_t += self.cfg.gamma * (self.cfg.alpha - err)
        self._alpha_t = float(min(max(self._alpha_t, 0.01), 0.5))
        if self._alpha_t != before:      # every quantile depends on alpha
            self._q_cache = [None] * self.n_steps

    # -- the margin -------------------------------------------------------
    def _quantile_at(self, step: int) -> float:
        cached = self._q_cache[step]
        if cached is not None:
            return cached
        q = self._compute_quantile(step)
        self._q_cache[step] = q
        return q

    def _compute_quantile(self, step: int) -> float:
        s = self._scores[step]
        n = len(s)
        prior = self.cfg.prior_rate * (step + 1) * self.dt
        if n < self.cfg.min_samples:
            return prior
        # Split-conformal finite-sample correction: the ceil((n+1)(1-alpha))-th
        # smallest score, not the plain empirical quantile. With the plain one
        # the guarantee is off by a factor of (n+1)/n, which matters exactly
        # when the window is short -- that is, when it is being trusted least.
        level = 1.0 - self._alpha_t
        idx = int(math.ceil((n + 1) * level)) - 1
        if idx >= n:                       # level too high for this many samples
            return float(max(max(s), prior))
        return float(np.partition(np.fromiter(s, dtype=np.float64, count=n),
                                  idx)[idx])

    def margins(self) -> np.ndarray:
        """Conformal margin per horizon step, metres, ``(n_steps,)``."""
        q = np.array([self._quantile_at(k) for k in range(self.n_steps)],
                     dtype=np.float64)
        return np.minimum(q, self.cfg.max_margin)

    # -- reporting --------------------------------------------------------
    @property
    def alpha_t(self) -> float:
        """Current effective level. Drifting far from ``cfg.alpha`` means the
        residuals are not exchangeable and ACI is doing real work."""
        return self._alpha_t

    def coverage(self) -> Optional[float]:
        """Realised coverage over the episode, or ``None`` before any score.

        This is the number that decides whether the method did what it claims.
        A conformal margin whose empirical coverage misses its nominal target
        is not calibrated, whatever else it achieved.
        """
        return (self._hits / self._total) if self._total else None

    def coverage_profile(self) -> np.ndarray:
        """Per-horizon-step coverage, ``(n_steps,)``; NaN where unscored.

        Aggregate coverage can sit on target while the near horizon is
        over-covered and the far horizon under-covered, which is the failure
        that matters -- the far horizon is where the planner commits.
        """
        out = np.full(self.n_steps, np.nan)
        seen = self._per_step_total > 0
        out[seen] = (self._per_step_hits[seen] / self._per_step_total[seen])
        return out

    @property
    def n_scored(self) -> int:
        return int(self._total)
