"""Tests for the conformally calibrated safety margin.

The claim this module makes is statistical -- that the margin contains the
actor ``1 - alpha`` of the time -- so most of these tests are statistical too:
they drive the calibrator over thousands of synthetic residuals and check the
realised coverage against the nominal target. A unit test that only checked
"the number came out positive" would pass for a margin that is calibrated to
nothing at all, which is exactly the failure the heuristic it replaces had.
"""

from __future__ import annotations

import numpy as np
import pytest

from divas.prediction.conformal import ConformalCalibrator, ConformalConfig
from divas.prediction.risk import MarginParams, RiskField
from divas.types import (
    PredictedTrajectory,
    Track,
    TrajectoryMode,
    TrajectorySet,
)

DT, STEPS = 0.1, 10


def _one(points):
    return TrajectorySet(
        [PredictedTrajectory(track_id=1,
                             modes=[TrajectoryMode(points=points, probability=1.0)],
                             cls="car")],
        dt=DT, horizon=STEPS * DT,
    )


def drive(cal, n=4000, sigma=0.25, seed=0, shift_at=None, shift=3.0):
    """Random-walking actor against a persistence forecast.

    Error at horizon k is then a k-step random-walk displacement, growing like
    sqrt(k). That matters: a synthetic setup where every horizon step has the
    same error cannot tell a per-step calibration from a single global one.
    """
    rng = np.random.default_rng(seed)
    p = np.zeros(2)
    t = 0.0
    for c in range(n):
        s = sigma * (shift if (shift_at is not None and c > shift_at) else 1.0)
        cal.observe([Track(id=1, x=p[0], y=p[1], vx=0.0, vy=0.0, cls="car")], t)
        cal.record(_one(np.repeat(p[None, :], STEPS, axis=0)), t)
        p = p + rng.normal(0, s, 2)
        t = round(t + DT, 3)
    return cal


# --------------------------------------------------------------------------
# the guarantee
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_realised_coverage_matches_the_nominal_target(alpha):
    """The whole point of the method, and the one thing a tuned margin
    cannot offer at any value of its coefficient."""
    cal = drive(ConformalCalibrator(STEPS, DT, ConformalConfig(
        alpha=alpha, adaptive=False, window=240, min_samples=30)))
    assert cal.coverage() == pytest.approx(1.0 - alpha, abs=0.02)


def test_calibration_is_per_horizon_step_not_merely_on_average():
    """Aggregate coverage can sit on target while the near horizon is
    over-covered and the far horizon under-covered -- and the far horizon is
    where the planner commits, so that failure is the one that matters."""
    cal = drive(ConformalCalibrator(STEPS, DT, ConformalConfig(
        alpha=0.1, adaptive=False, window=240, min_samples=30)))

    prof = cal.coverage_profile()
    assert np.all(np.abs(prof - 0.9) < 0.05)

    m = cal.margins()
    assert m[-1] > m[0] * 1.5              # errors grow with lookahead
    assert np.all(np.diff(m) > -1e-9)      # and the margin follows, monotonically


def test_the_margin_is_a_quantile_not_a_mean():
    """A mean margin would be exceeded roughly half the time. The distinction
    is the reason this is a safety quantity rather than an accuracy one."""
    cal = drive(ConformalCalibrator(STEPS, DT, ConformalConfig(
        alpha=0.1, adaptive=False, window=100000, min_samples=30)), n=3000)
    scores = np.fromiter(cal._scores[STEPS - 1], dtype=float)
    assert cal.margins()[STEPS - 1] > np.mean(scores)


# --------------------------------------------------------------------------
# adaptive conformal inference
# --------------------------------------------------------------------------


def _shift_coverage(adaptive, window):
    cal = ConformalCalibrator(STEPS, DT, ConformalConfig(
        alpha=0.1, adaptive=adaptive, window=window, min_samples=30))
    drive(cal, n=6000, shift_at=2500, seed=1)
    return cal.coverage()


def test_aci_helps_only_when_the_window_is_too_long_to_adapt():
    """Why ACI is OFF by default, stated as a measurement.

    It is the citation everyone reaches for, and at this project's default
    window it makes coverage *worse*. The rolling quantile already tracks the
    shift, so ACI becomes a second feedback loop correcting an error the first
    one has fixed; it overshoots and the margin under-covers. It pays only
    where it was designed to, with a window too long to adapt on its own.
    """
    assert _shift_coverage(True, 4000) > _shift_coverage(False, 4000) + 0.02


def test_aci_is_not_free_at_a_short_window():
    """The measurement that set the default. Pinned so that turning ACI back
    on silently -- because the paper is good -- fails loudly."""
    plain, aci = _shift_coverage(False, 240), _shift_coverage(True, 240)
    assert plain > aci
    assert ConformalConfig().adaptive is False


def test_aci_moves_alpha_in_the_right_direction():
    """Under-coverage must widen the interval, not narrow it. A sign error
    here is silent: the margin still moves, just always the wrong way."""
    cal = ConformalCalibrator(STEPS, DT, ConformalConfig(
        alpha=0.1, gamma=0.05, adaptive=True))
    start = cal.alpha_t
    for _ in range(50):
        cal._update_alpha(covered=False)           # persistent misses
    assert cal.alpha_t < start                     # lower alpha = wider interval
    for _ in range(200):
        cal._update_alpha(covered=True)
    assert cal.alpha_t > start


def test_alpha_stays_off_both_ends():
    """At alpha <= 0 the quantile is unbounded and the vehicle freezes; at
    alpha >= 1 there is no margin at all."""
    cal = ConformalCalibrator(STEPS, DT, ConformalConfig(
        alpha=0.1, gamma=0.5, adaptive=True))
    for _ in range(500):
        cal._update_alpha(covered=False)
    assert cal.alpha_t >= 0.01
    for _ in range(2000):
        cal._update_alpha(covered=True)
    assert cal.alpha_t <= 0.5


# --------------------------------------------------------------------------
# the bookkeeping that makes the number mean anything
# --------------------------------------------------------------------------


def test_observing_before_recording_is_what_makes_the_residual_real():
    """Score-then-predict, never predict-then-score.

    A prediction made now for now is exactly the observation that produced it,
    so scoring it yields a residual of zero. Do that every cycle and every
    quantile collapses towards zero -- reporting a beautifully calibrated
    margin that has measured nothing.
    """
    cal = ConformalCalibrator(STEPS, DT)
    pts = np.zeros((STEPS, 2))
    track = Track(id=1, x=0.0, y=0.0, vx=0.0, vy=0.0, cls="car")

    cal.record(_one(pts), 0.0)
    assert cal.observe([track], 0.0) == 0       # nothing is due yet at t=0
    assert cal.observe([track], DT) == 1        # the first step is due at t=dt


def test_an_actor_that_leaves_is_dropped_rather_than_scored():
    """Scoring a departed actor against its last known position would credit
    the predictor for a disappearance it never predicted."""
    cal = ConformalCalibrator(STEPS, DT)
    cal.record(_one(np.zeros((STEPS, 2))), 0.0)
    assert cal.observe([], DT) == 0
    assert cal.n_scored == 0


def test_a_cold_horizon_step_uses_the_prior_not_a_bad_quantile():
    """Below the sample floor a quantile is not a quantile. ceil((n+1)(1-a))
    exceeds n for small n, which would silently return the largest sample."""
    cfg = ConformalConfig(alpha=0.1, min_samples=30, prior_rate=0.55)
    cal = ConformalCalibrator(STEPS, DT, cfg)
    m = cal.margins()
    assert m[0] == pytest.approx(0.55 * DT)
    assert m[-1] == pytest.approx(0.55 * STEPS * DT)


def test_the_margin_is_capped():
    """An honest quantile can exceed the road. A margin wider than the
    carriageway does not make the vehicle careful, it makes it stuck."""
    cal = ConformalCalibrator(STEPS, DT, ConformalConfig(
        alpha=0.1, adaptive=False, max_margin=0.9, min_samples=30))
    drive(cal, n=3000, sigma=2.0)
    assert np.all(cal.margins() <= 0.9 + 1e-9)


# --------------------------------------------------------------------------
# the seam into the risk field
# --------------------------------------------------------------------------


def test_the_risk_field_uses_the_calibrator_when_given_one():
    ts = _one(np.zeros((STEPS, 2)))
    cal = drive(ConformalCalibrator(STEPS, DT, ConformalConfig(
        alpha=0.1, adaptive=False, min_samples=30)), n=2000)

    conformal = RiskField(ts, 8.0, (1.95, 0.85),
                          MarginParams.fixed(1.0), conformal=cal)
    assert conformal.d_safe[0][-1] > conformal.d_safe[0][0]     # grows with lookahead


def test_a_single_mode_predictor_makes_the_heuristic_margin_constant():
    """The diagnosis behind ADR-010, as an assertion.

    Confidence is the spatial dispersion of the modes. A constant-velocity
    prediction has one mode, so dispersion is identically zero, confidence is
    identically one, and the 'dynamic' margin is a constant -- for every
    actor, at every horizon step, in every scenario.
    """
    ts = _one(np.zeros((STEPS, 2)))
    heuristic = RiskField(ts, 8.0, (1.95, 0.85), MarginParams())
    row = heuristic.d_safe[0]
    assert np.allclose(row, row[0])
