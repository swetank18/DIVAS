# ADR-002: Sampling-based MPC (MPPI) instead of acados/OSQP

**Status:** accepted, revisit in Phase 4 hardware bring-up · **Phase:** 1

## Decision
Stage 6's `SamplingMPC` optimises by sampling: perturb a nominal control sequence, roll every sample through the kinematic bicycle model in parallel, take a softmax-weighted update.

## Why
`EXECUTION_PLAN.md` Phase 4 specifies acados (SQP-RTI) or OSQP. Neither is installed here, and more importantly neither is a drop-in: both want a smooth differentiable cost, while the risk field is a **rasterised, non-convex** volume with no useful gradients. Sampling handles that natively and needs no solver dependency.

Measured: ~19 ms per solve at 512 samples over a 20-step horizon, in pure numpy. That meets the 20–50 Hz control-rate spec.

## Trade-off, stated plainly
No constraint *guarantees* — input limits are enforced by clamping, and obstacle avoidance is a cost, not a constraint. Cost scales with sample count. A gradient-based QP would give guarantees and be faster.

## Migration path
The `Controller` interface is the seam. Taking the acados path means smoothing the risk field into an analytic sum of ellipsoids, which `RiskField.risk_at` already provides — it exists precisely so that this option stays open.

## What made it work
Two non-obvious things, both of which cost real debugging time:
1. **Temporally correlated noise.** White noise on steering averages to σ/√N over the horizon, so every sampled rollout lands within centimetres of the nominal and the search explores no manoeuvre at all. The noise is now smoothed over ~7 steps plus a constant per-sample offset.
2. **A geometric feedforward prior.** The nominal is seeded from a pure-pursuit rollout, so the sampler refines a manoeuvre instead of searching for one — and cannot accidentally do worse than the conventional controller.
