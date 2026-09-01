# ADR-009: The obstacle force is disabled; the social force is retained but unproven

**Status:** accepted · **Phase:** 1 · **Evidence:** `scripts/tune_predictor.py`, `scripts/calibration.py`

## Decision
`SocialForcePredictor.obstacle_strength` defaults to **0.0**. `social_strength` is reduced from 2.1 to **0.6**. The predictor is given a **static-only** occupancy grid; the planner keeps the full one.

## Why — the measurement
Open-loop prediction error over the scenario suite, mean over a 3 s horizon. Constant-velocity baseline: **3.54 m**.

| obstacle_strength | mean error | | social_strength | mean error |
|---|---|---|---|---|
| 0.0 | **3.45 m** | | 0.0 | 3.45 m |
| 1.5 | 4.41 m | | 0.6 | 3.46 m |
| 6.0 | 4.55 m | | 2.1 | 3.48 m |

The obstacle force was the single largest source of error in the stack. It pushes predicted agents away from the road boundary in ways the actual agents do not move, costing ~1.1 m of accuracy — enough to make the "interaction-aware" predictor **less accurate than plain constant velocity**, and through the risk field to cost 11 points of closed-loop success rate (0.92 → 0.81) and triple the collision rate (0.06 → 0.17).

Disabling it costs nothing operationally. A prediction that strays off the drivable area creates a keep-out in a region the occupancy grid already blocks, so the ego was never going to drive there.

A second, smaller error was structural: the occupancy grid **contains the actors**, so the predictor applied a grid obstacle force *and* a pairwise social force to the same agents — repulsion counted twice. The predictor now receives a static-only grid, which is what a real fusion stage produces anyway (static layer plus a dynamic object list).

## The uncomfortable part
The social term is neutral on this benchmark — 3.45 m at zero strength, 3.48 m at 2.1. That is not evidence that interaction-awareness does not work. It is evidence that **this benchmark cannot test it**: the scenario actors follow scripted policies and do not react to one another, so there is no interaction for an interaction-aware model to exploit.

So the term is kept at a small, physically motivated value rather than tuned or removed. Tuning it against a benchmark that cannot reward it would be fitting noise.

**This is now the top benchmark task: the scenario actors need reactive policies** (yield, follow, gap-accept) before any claim about interaction-aware prediction can be supported or refuted. Until then, the deck should claim the *multi-modal, calibrated uncertainty* — which is measured and does hold — and not the interaction modelling, which is untested.

## What does hold
Confidence is genuinely calibrated. Mean prediction error by confidence quintile: 7.01, 5.59, 4.47, 3.90, 2.66 m; correlation with error −0.27. The predictor knows when it is unreliable, which is the property the dynamic safety margin depends on, and it is a property constant velocity does not have at all (it reports confidence 1.0 always, by construction).
