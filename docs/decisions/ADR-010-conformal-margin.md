# ADR-010: The safety margin is calibrated, not tuned

**Status:** accepted · **Phase:** 4 · **Evidence:** `tests/test_conformal.py`, `scripts/required_margin.py`, ADR-003, ADR-004, ADR-009

## Decision

`d_safe` may be sourced from a **conformal quantile of the predictor's own
recent error** instead of from the heuristic confidence term:

```
tuned       d_safe(t) = d0 + k_v·v_ego + λ·(1 − confidence(t))
calibrated  d_safe(t) = d0 + k_v·v_ego + q_{1−α}(t)
```

`q_{1−α}(t)` is the (1−α) quantile of the displacement error this predictor has
actually made at lookahead `t`, over a rolling window, with the split-conformal
finite-sample correction. The swap is total rather than a blend: two
uncertainty terms added together are two things to tune and one number nobody
can attribute.

It is a **new ablation arm**, not a replacement of the default. Every
Phase 1 number was measured with the tuned margin, and an arm that changes two
things at once cannot be read.

## Why — the heuristic was measuring the wrong quantity

ADR-004 defines confidence as the spatial dispersion of the predictor's modes.
That is *the predictor disagreeing with itself*, which is not the same
quantity as *the predictor being wrong*. The two come apart at both ends: a
single-mode constant-velocity prediction has zero dispersion and unbounded
error, while three modes straddling the truth have large dispersion and small
error. `test_a_single_mode_predictor_makes_the_heuristic_margin_constant` pins
the first case — with the CV predictor the "dynamic" margin is *identically
constant*, for every actor at every step.

But dispersion is not useless, and ADR-009 measured that: error by confidence
quintile runs **7.01, 5.59, 4.47, 3.90, 2.66 m**, correlation −0.27. The signal
is real and weak.

**The decisive problem is scale, not sign.** The quintile spread is 4.35 m of
error. λ is 0.8 m and the margin is capped at 1.8 m (ADR-003, because ~5 m
keep-outs block a 10 m carriageway outright). So the margin's entire dynamic
range is about a fifth of the variation it is responding to. That is why the
Phase 1 control arm — a fixed margin at the dynamic arm's mean — tied it at
0.90/0.08: with a range that small, only the mean can matter.

## Why conformal specifically

Split conformal prediction gives, for exchangeable residuals, a
**distribution-free finite-sample guarantee**:

```
P( ‖actual − predicted‖ ≤ q_{1−α} ) ≥ 1 − α
```

with no assumption about the predictor, the noise or the traffic. It holds for
constant velocity exactly as it would for a transformer. That is a property a
tuned coefficient cannot have at any value, and it is the honest answer to
"how would you make λ principled?" — which is the question the closest prior
art invites. Fisac et al. (RSS 2018) derive the same *shape* of behaviour from
Bayesian model confidence plus Hamilton–Jacobi reachability; the mechanism here
is weaker in guarantee-per-step and stronger in assumptions: it needs none.

Verified rather than asserted: realised coverage is 0.948 / 0.900 / 0.801
against nominal 0.95 / 0.90 / 0.80, and flat across every horizon step.

## Adaptive conformal, and when it earns its place

Closed-loop driving breaks exchangeability twice over — the ego's actions
change what it observes next, and traffic is non-stationary. `alpha` is
therefore adapted online after Gibbs & Candès (2021), whose guarantee is on
long-run coverage and survives arbitrary shift.

It is on by default because of a measurement, not a citation. Post-shift
coverage under a 3× error jump:

| window | plain rolling | with ACI |
|---|---|---|
| 240 | 0.890 | 0.895 |
| 1200 | 0.832 | 0.882 |
| 4000 | 0.700 | 0.860 |

At the default window the rolling quantile adapts on its own and ACI is within
noise; it is insurance against a window that is too long for the regime, and it
costs nothing.

## The uncomfortable part, again

Calibration does not create accuracy, it reveals its absence. The first live
run of the conformal arm **saturated its cap**: mean `d_safe` 2.45 m against a
2.5 m ceiling, and coverage still only 0.854 rather than 0.90.

Read that plainly. **An honest 90% keep-out for constant-velocity prediction at
a 3 s lookahead does not fit on the road.** No margin policy — fixed, tuned or
calibrated — can cover an error wider than the carriageway. What conformal
calibration buys is that this is now a *number on a chart* instead of a
mis-tuned coefficient nobody can see.

Two consequences follow, and they are the Phase 4 work:

1. **Required Safety Margin** (`scripts/required_margin.py`) becomes a predictor
   metric denominated in metres of carriageway rather than in ADE/FDE. A
   predictor change that does not reduce RSM has bought the planner nothing.
2. **The prediction horizon becomes a design variable.** Past the lookahead at
   which an honestly-calibrated keep-out stops fitting on the road, the
   prediction is not actionable, and `StackConfig.horizon` exists so the
   ablation can test whether a shorter, well-covered horizon beats a longer,
   under-covered one.

## What this ADR does not claim

That the conformal margin reduces collisions. That is a separate question, it
is answered by the ablation, and it is answered with the same control arm that
killed the heuristic: a fixed margin equal to the conformal arm's own mean. If
that control arm ties again, the conclusion is again that size beat variation,
and it gets reported the same way.
