# ADR-006: The route goal must be further away than the control horizon

**Status:** accepted · **Phase:** 1

## Decision
`goal_lookahead` (28 m) > MPC horizon reach (`cruise_speed` × `horizon` × `dt` ≈ 18 m), and the costmap half-extent (32 m) exceeds both. The cross-track cost is additionally capped at 3 m.

## Why
This was the single most expensive bug in the Phase 1 build, and it presented as something else entirely.

With an 18 m route lookahead, the planner produced ~16 m paths while the MPC's 2 s horizon reached 20 m at cruise speed. Every rollout ran off the end of the path, where cross-track error grows without bound, and the terminal cost made it worse. The controller's cheapest available response was to **brake until its horizon fit inside the path** — so the vehicle crawled at 1.8 m/s on a completely empty road with the nearest actor 25 m away.

Every symptom pointed somewhere else: it looked like over-conservative risk weighting, then like a prediction problem, then like MPPI instability. It was a geometry mismatch between two horizons that were never checked against each other.

## Consequences
- The three horizons are now a stated invariant: **costmap ⊃ route lookahead > control horizon reach.** Any change to `cruise_speed`, MPC `horizon`, or `goal_lookahead` must preserve it.
- Cross-track cost is capped, so a genuinely short path (blocked route, partial plan) degrades gracefully instead of dominating every other term — which is exactly when the other terms matter most.
