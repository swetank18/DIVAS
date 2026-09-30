"""Checks for divas/eval/safety_analysis.py.

Each test pins a number the module's docstring claims, so the explanation and
the arithmetic cannot drift apart.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from divas.eval.safety_analysis import (
    ellipse_signed_distance,
    exact_penetration,
    penetration_error_profile,
    rate_difference_interval,
    rate_summary,
    runs_needed,
    sequential_coverage,
    wilson_interval,
)
from divas.prediction.risk import MarginParams, RiskField
from divas.types import PredictedTrajectory, TrajectoryMode, TrajectorySet, VehicleParams


# -- rates -----------------------------------------------------------------

def test_wilson_matches_reference_values():
    lo, hi = wilson_interval(43, 48)
    assert lo == pytest.approx(0.778, abs=1e-3)
    assert hi == pytest.approx(0.955, abs=1e-3)


def test_wilson_does_not_collapse_at_zero_events():
    # Wald would say [0, 0]: "never crashes". Wilson keeps an upper bound.
    lo, hi = wilson_interval(0, 48)
    assert lo == 0.0
    assert hi == pytest.approx(0.074, abs=1e-3)


def test_small_rate_differences_are_not_resolvable_at_48_runs():
    d, lo, hi = rate_difference_interval(43, 48, 41, 48)
    assert d == pytest.approx(2 / 48)
    assert lo < 0.0 < hi


def test_runs_needed_for_a_four_point_collision_gain():
    assert 540 <= runs_needed(0.08, 0.04) <= 560
    with pytest.raises(ValueError):
        runs_needed(0.1, 0.1)


def test_rate_summary_reads_run_metrics():
    class R:
        def __init__(self, ok):
            self.success, self.collision, self.timeout = ok, not ok, False

    s = rate_summary([R(True)] * 9 + [R(False)])
    est, lo, hi = s["success_rate"]
    assert s["runs"] == 10 and est == 0.9 and lo < 0.9 < hi


# -- exact ellipse distance ------------------------------------------------

def _brute_force(px, py, a, b, n=200_001):
    th = np.linspace(0.0, 2.0 * np.pi, n)
    d = np.hypot(a * np.cos(th) - px, b * np.sin(th) - py).min()
    return -d if (px / a) ** 2 + (py / b) ** 2 < 1.0 else d


def test_ellipse_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(60):
        a, b = rng.uniform(0.3, 8.0, 2)
        px, py = rng.uniform(-2.0, 2.0, 2) * np.array([a, b])
        assert float(ellipse_signed_distance(px, py, a, b)) == pytest.approx(
            _brute_force(px, py, a, b), abs=1e-6)


@pytest.mark.parametrize("d", [0.0, 1.0, 3.0, 6.9, 7.5, 9.0])
def test_ellipse_distance_on_the_major_axis(d):
    # The on-axis case has its own closed form; pin it separately.
    a, b = 7.8, 2.6
    assert float(ellipse_signed_distance(d, 0.0, a, b)) == pytest.approx(
        _brute_force(d, 0.0, a, b), abs=1e-6)


def test_minor_axis_approximation_is_exact_and_major_axis_falls_to_b_over_a():
    prof = penetration_error_profile(6.0, 2.0)
    assert np.allclose(prof["minor"]["ratio"], 1.0, atol=1e-9)
    assert prof["major"]["ratio"].min() == pytest.approx(2.0 / 6.0, abs=1e-3)


def test_exact_penetration_is_never_below_the_approximation():
    # A single truck heading +x; query points all around its keep-out.
    pts = np.stack([np.linspace(0.0, 15.0, 30), np.zeros(30)], axis=1)
    ts = TrajectorySet([PredictedTrajectory(1, [TrajectoryMode(1.0, pts)], cls="truck")],
                       dt=0.1, horizon=3.0)
    rf = RiskField(ts, 8.0, VehicleParams().half_extent, MarginParams())
    xs, ys = np.meshgrid(np.linspace(-8.0, 8.0, 41), np.linspace(-4.0, 4.0, 21))
    approx = rf.penetration(xs.ravel(), ys.ravel(), 0.1)
    exact = exact_penetration(rf, xs.ravel(), ys.ravel(), 0.1)
    assert np.all(exact >= approx - 1e-9)
    assert np.all((exact > 0) == (approx > 0))       # same inside/outside set
    assert exact.max() == pytest.approx(approx.max(), abs=1e-6)  # centre: both = b


# -- conformal coverage ----------------------------------------------------

def test_out_of_sample_coverage_meets_nominal_and_in_sample_is_optimistic():
    errs = np.random.default_rng(1).exponential(1.0, 5000)
    rep = sequential_coverage(errs, alpha=0.1, window=60, min_samples=30)
    assert rep.out_of_sample >= 0.9 - 0.015
    assert rep.in_sample > rep.out_of_sample
    assert rep.optimism > 0.005


def test_coverage_with_too_few_samples_reports_nothing():
    rep = sequential_coverage([1.0] * 10, min_samples=30)
    assert rep.n_scored == 0 and math.isnan(rep.out_of_sample)
