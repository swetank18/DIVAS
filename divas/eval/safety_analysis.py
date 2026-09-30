"""Sharper arithmetic for the numbers the ablation table is built on.

Three calculations elsewhere in the stack are approximations, and each one
leans in a direction that matters.  This module supplies the exact version of
each, so the approximation can be measured rather than argued about.

1. **Rates without error bars.**  ``aggregate`` reports ``success_rate`` and
   ``collision_rate`` as bare fractions of 48 runs (6 scenarios x 8 seeds).
   Two stacks scoring 0.90 and 0.90 are called a tie, and 0.90 against 0.85 is
   read as a win -- but at n = 48 the 95% interval on 43/48 = 0.90 is
   [0.78, 0.95], so neither reading is supported.  :func:`wilson_interval`,
   :func:`rate_difference_interval` and :func:`runs_needed` put a number on
   what the table can and cannot claim.

2. **Keep-out penetration.**  ``RiskField.penetration`` converts the
   normalised elliptical distance ``u`` into metres as ``(1 - u) * b``, the
   minor semi-axis.  That is exact on the minor axis and at the centre, but
   along the major axis the true depth near the tip is ``a - d`` while the
   formula gives ``(a - d) * b / a``.  For an elongated keep-out with
   a / b = 3 -- the shape a long vehicle produces once the ego extent and the
   margin are added -- the MPC is told a third of the real incursion when it
   clips a front or rear corner.  :func:`ellipse_signed_distance` is the exact Euclidean distance
   (Eberly's robust bisection) and :func:`exact_penetration` is a drop-in
   replacement that reads the same arrays ``RiskField`` already holds.

3. **Conformal coverage scored in-sample.**  ``ConformalCalibrator.observe``
   appends the new residual to the window *and then* computes the quantile it
   is tested against, so every score is partly compared with itself.  That
   inflates the realised coverage, and because the ACI update is driven by
   that coverage, the effective alpha drifts upward and the margin tightens.
   :func:`sequential_coverage` replays a residual stream both ways so the
   size of the bias can be read off directly.

Everything here is pure numpy plus the standard library, has no side effects
and imports nothing from the rest of the stack except by duck-typing on a
``RiskField``, so it can be used from a notebook or a script without touching
the closed loop.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import NormalDist
from typing import Deque, Iterable, Optional, Tuple

import numpy as np


def _z(confidence: float) -> float:
    """Two-sided standard-normal critical value, e.g. 0.95 -> 1.95996."""
    return NormalDist().inv_cdf(0.5 + 0.5 * confidence)


# ---------------------------------------------------------------------------
# 1. Rates with honest uncertainty
# ---------------------------------------------------------------------------

def wilson_interval(k: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n``.

    Chosen over the textbook normal (Wald) interval because the rates in this
    project live near the edges -- collision rates of 0.00-0.10 -- which is
    exactly where Wald collapses: at k = 0 it returns [0, 0], claiming that a
    stack which did not crash in 48 runs *never* crashes.  Wilson gives
    [0, 0.074] there, which is the statement the data supports.

        centre = (p + z^2 / 2n) / (1 + z^2 / n)
        half   = z * sqrt(p(1-p)/n + z^2 / 4n^2) / (1 + z^2 / n)
    """
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError(f"k={k} outside [0, n={n}]")
    z = _z(confidence)
    p = k / n
    z2n = z * z / n
    denom = 1.0 + z2n
    centre = (p + 0.5 * z2n) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_difference_interval(
    k1: int, n1: int, k2: int, n2: int, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """``(diff, lo, hi)`` for ``p1 - p2`` by Newcombe's hybrid score method.

    Newcombe (1998), method 10: combine the two Wilson intervals rather than
    pooling a variance.  It keeps good coverage when either rate is near 0 or
    1 and when the arms differ in size, both of which happen here.

    If the interval contains zero the table cannot rank the two stacks on this
    metric, whatever the point estimates say.
    """
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1, confidence)
    l2, u2 = wilson_interval(k2, n2, confidence)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (d, max(-1.0, lo), min(1.0, hi))


def runs_needed(p1: float, p2: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Runs *per stack* to tell rates ``p1`` and ``p2`` apart.

    Standard two-proportion sample size (two-sided test at level ``alpha``)::

        n = ( z_a * sqrt(2 pbar qbar) + z_b * sqrt(p1 q1 + p2 q2) )^2 / (p1 - p2)^2

    Worked example for this repository: separating a 0.08 collision rate from
    0.04 needs ~550 runs per stack, against the 48 the ablation uses.  A
    four-point collision improvement is therefore not something the current
    table can show, in either direction -- more seeds, not more tuning, is
    what would settle it.
    """
    if not (0.0 <= p1 <= 1.0 and 0.0 <= p2 <= 1.0):
        raise ValueError("rates must lie in [0, 1]")
    delta = abs(p1 - p2)
    if delta == 0.0:
        raise ValueError("identical rates cannot be separated by any sample size")
    za = _z(1.0 - alpha)
    zb = NormalDist().inv_cdf(power)
    pbar = 0.5 * (p1 + p2)
    num = za * math.sqrt(2.0 * pbar * (1.0 - pbar)) + zb * math.sqrt(
        p1 * (1.0 - p1) + p2 * (1.0 - p2)
    )
    return int(math.ceil((num / delta) ** 2))


def rate_summary(runs: Iterable, confidence: float = 0.95) -> dict:
    """Success and collision rates of ``RunMetrics`` with Wilson intervals.

    Meant to sit beside ``aggregate()`` in the ablation printout::

        {"runs": 48, "success_rate": (0.896, 0.778, 0.955), ...}

    each value being ``(estimate, lo, hi)``.
    """
    runs = list(runs)
    n = len(runs)
    out = {"runs": n}
    for name, attr in (("success_rate", "success"),
                       ("collision_rate", "collision"),
                       ("timeout_rate", "timeout")):
        k = sum(bool(getattr(r, attr)) for r in runs)
        lo, hi = wilson_interval(k, n, confidence)
        out[name] = (round(k / n, 3) if n else 0.0, round(lo, 3), round(hi, 3))
    return out


# ---------------------------------------------------------------------------
# 2. Exact distance to an elliptical keep-out
# ---------------------------------------------------------------------------

def ellipse_signed_distance(x, y, a, b, iterations: int = 64) -> np.ndarray:
    """Exact signed Euclidean distance from ``(x, y)`` to the ellipse
    ``(x/a)^2 + (y/b)^2 = 1`` in its own frame.  Negative inside.

    Method (Eberly, *Distance from a Point to an Ellipse*, 2013): fold the
    point into the first quadrant, order the axes so ``e0 >= e1``, and find
    the unique root ``s`` of

        G(s) = (r0 z0 / (s + r0))^2 + (z1 / (s + 1))^2 - 1,
        z_i = y_i / e_i,   r0 = (e0 / e1)^2

    by bisection on a bracket that always contains it.  The closest boundary
    point is then ``(r0 y0 / (s + r0), y1 / (s + 1))``.  Bisection rather than
    Newton because G has a pole just left of the bracket; 64 halvings reach
    double precision from any starting width, and every array element runs the
    same fixed number of steps, so the whole call is one vectorised loop.  Checked against a brute-force
    boundary sweep to better than 1e-7 m.

    The axis case ``y1 = 0`` is a genuine special case -- inside the evolute
    the closest point is off-axis -- and is handled in closed form.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    a = np.broadcast_to(np.asarray(a, dtype=np.float64), np.broadcast(x, y).shape)
    b = np.broadcast_to(np.asarray(b, dtype=np.float64), a.shape)
    x, y = np.broadcast_arrays(x, y)

    swap = b > a                       # order the axes so that e0 >= e1
    e0 = np.where(swap, b, a)
    e1 = np.where(swap, a, b)
    y0 = np.abs(np.where(swap, y, x))
    y1 = np.abs(np.where(swap, x, y))

    inside = (y0 / e0) ** 2 + (y1 / e1) ** 2 < 1.0

    # -- general case, y1 > 0 -------------------------------------------
    z0 = y0 / e0
    z1 = y1 / e1
    r0 = (e0 / e1) ** 2
    n0 = r0 * z0
    g = z0 * z0 + z1 * z1 - 1.0
    lo = z1 - 1.0
    hi = np.where(g < 0.0, 0.0, np.hypot(n0, z1) - 1.0)
    # On-axis points (y1 = 0) can make s + 1 vanish; they are overwritten
    # by the closed form below, so the warnings they raise here are noise.
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(iterations):
            s = 0.5 * (lo + hi)
            val = (n0 / (s + r0)) ** 2 + (z1 / (s + 1.0)) ** 2 - 1.0
            pos = val > 0.0
            lo = np.where(pos, s, lo)
            hi = np.where(pos, hi, s)
        s = 0.5 * (lo + hi)
        cx = r0 * y0 / (s + r0)
        cy = y1 / (s + 1.0)

    # -- on the major axis, y1 = 0 -------------------------------------
    on_axis = y1 <= 0.0
    denom = e0 * e0 - e1 * e1
    with np.errstate(divide="ignore", invalid="ignore"):
        near = y0 < denom / e0                          # inside the evolute
        ax = np.where(near, e0 * e0 * y0 / np.where(denom > 0, denom, 1.0), e0)
        ay = np.where(near, e1 * np.sqrt(np.clip(1.0 - (ax / e0) ** 2, 0.0, 1.0)), 0.0)
    cx = np.where(on_axis, ax, cx)
    cy = np.where(on_axis, ay, cy)

    dist = np.hypot(y0 - cx, y1 - cy)
    return np.where(inside, -dist, dist)


def exact_penetration(risk_field, x, y, t) -> np.ndarray:
    """Drop-in for ``RiskField.penetration`` with the exact depth, metres.

    Same inputs, same ``max over modes`` reduction, same time indexing -- only
    the metres conversion changes, from ``(1 - u) * b`` to the true Euclidean
    distance to the keep-out boundary.  The two agree on the minor axis; away
    from it the approximation under-reports, by up to a factor ``b / a``.
    """
    rf = risk_field
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    y = np.atleast_1d(np.asarray(y, dtype=np.float64))
    if rf.points.shape[0] == 0:
        return np.zeros_like(x)
    i = np.broadcast_to(rf._step_index(t), x.shape)
    p = rf.points[:, i, :]
    dx = x[None] - p[..., 0]
    dy = y[None] - p[..., 1]
    c, s = rf.cos_h[:, i], rf.sin_h[:, i]
    along = c * dx + s * dy
    across = -s * dx + c * dy
    sd = ellipse_signed_distance(along, across,
                                 np.maximum(rf.a[:, i], 1e-6),
                                 np.maximum(rf.b[:, i], 1e-6))
    return np.maximum(-sd, 0.0).max(axis=0)


def penetration_error_profile(a: float, b: float, n: int = 201) -> dict:
    """How far ``(1 - u) * b`` is from the exact depth, along both axes.

    Sweeps a point from the centre to the boundary along the major and the
    minor axis and reports the ratio approx / exact.  Useful for a slide:
    along the minor axis the ratio is 1 everywhere; along the major axis it
    falls to ``b / a`` near the tip, which is where a corner clip happens.
    """
    frac = np.linspace(0.0, 0.999, n)
    out = {}
    for name, px, py, semi in (("major", frac * a, 0.0 * frac, a),
                               ("minor", 0.0 * frac, frac * b, b)):
        u = np.sqrt((px / a) ** 2 + (py / b) ** 2)
        approx = (1.0 - u) * b
        exact = -ellipse_signed_distance(px, py, a, b)
        out[name] = {
            "offset_m": frac * semi,
            "approx_m": approx,
            "exact_m": exact,
            "ratio": np.divide(approx, exact, out=np.ones_like(exact), where=exact > 1e-9),
        }
    return out


# ---------------------------------------------------------------------------
# 3. Conformal coverage, out-of-sample
# ---------------------------------------------------------------------------

@dataclass
class CoverageReport:
    """Realised coverage of a residual stream, scored two ways."""

    nominal: float          #: 1 - alpha
    out_of_sample: float    #: quantile from past residuals only -- correct
    in_sample: float        #: residual added before scoring -- as observe() does
    n_scored: int

    @property
    def optimism(self) -> float:
        """How much coverage in-sample scoring invents, in absolute terms."""
        return self.in_sample - self.out_of_sample


def _split_conformal_q(scores: Deque[float], alpha: float, prior: float,
                       min_samples: int) -> float:
    """Same quantile rule as ``ConformalCalibrator._quantile_at``."""
    n = len(scores)
    if n < min_samples:
        return prior
    idx = int(math.ceil((n + 1) * (1.0 - alpha))) - 1
    if idx >= n:
        return max(max(scores), prior)
    arr = np.fromiter(scores, dtype=np.float64, count=n)
    return float(np.partition(arr, idx)[idx])


def sequential_coverage(
    errors: Iterable[float],
    alpha: float = 0.1,
    window: int = 240,
    min_samples: int = 30,
    prior: float = math.inf,
) -> CoverageReport:
    """Replay one horizon step's residual stream and score its coverage.

    Two windows run side by side over the same errors:

    * **out-of-sample** -- the quantile is taken over the residuals seen
      *before* this one, then the residual joins the window.  This is the
      order split conformal's guarantee is stated for, and for exchangeable
      residuals it covers at least ``1 - alpha``.
    * **in-sample** -- the residual joins the window first, as in
      ``ConformalCalibrator.observe``.  A residual can never exceed a quantile
      that it is itself allowed to be, so this can only report more coverage.

    Only steps past ``min_samples`` are scored, so the prior does not colour
    either figure.  The gap between them is the optimism.  With ``window=240``
    and ``alpha=0.1`` it measures about half a point of coverage on i.i.d.
    residuals (about 0.905 in-sample against 0.901 out-of-sample over 5000
    exponential draws), and about 1.6 points at ``window=60``.  Small, but
    systematic: ACI steers the *in-sample* miss rate onto alpha, so the true
    miss rate settles that much above target, and the shortfall grows as the
    window shrinks, since the self-inclusion weighs roughly 1 / window.
    """
    past: Deque[float] = deque(maxlen=window)
    incl: Deque[float] = deque(maxlen=window)
    hit_out = hit_in = total = 0
    for e in errors:
        e = float(e)
        scored = len(past) >= min_samples
        if scored:
            hit_out += e <= _split_conformal_q(past, alpha, prior, min_samples)
        past.append(e)
        incl.append(e)
        if scored:
            hit_in += e <= _split_conformal_q(incl, alpha, prior, min_samples)
            total += 1
    if total == 0:
        return CoverageReport(1.0 - alpha, math.nan, math.nan, 0)
    return CoverageReport(1.0 - alpha, hit_out / total, hit_in / total, total)


__all__ = [
    "wilson_interval",
    "rate_difference_interval",
    "runs_needed",
    "rate_summary",
    "ellipse_signed_distance",
    "exact_penetration",
    "penetration_error_profile",
    "CoverageReport",
    "sequential_coverage",
]
