# ADR-004: Confidence is spatial dispersion, not probability entropy

**Status:** accepted · **Phase:** 1

## Decision
`PredictedTrajectory.confidence_profile()` returns, per horizon step,

```
confidence(t) = exp( -spread(t) / SPREAD_SCALE )
```

where `spread(t)` is the probability-weighted standard deviation of the modes' positions at `t`, and `SPREAD_SCALE = 3 m`.

## Why
The first implementation used the normalised entropy of the mode probabilities. That is the wrong measure, and the test that catches it is simple: three equally likely modes that all predict nearly the same path describe a *certain* future, and entropy scores them as maximally uncertain.

What the controller needs to know is not how many hypotheses there are — it is **how many metres the prediction could be wrong by**. Dispersion answers that question in the units the safety margin is expressed in.

## Consequences
- Confidence discriminates the way it should: a bus scores ~0.86, a motorcycle at the same range ~0.32. That gap is the entire dynamic-margin mechanism; without it `d_safe` is a constant wearing a costume.
- A predictor that emits one mode is maximally confident by construction. This is correct — and it is exactly why `ConstantVelocityPredictor` is the overconfident baseline the ablation is built to expose.
