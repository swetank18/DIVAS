# ADR-007: The fallback planner is an RRT with choose-parent, not RRT\*

**Status:** accepted · **Phase:** 1

## Decision
`divas/planning/rrt.py` implements a kinodynamic RRT with RRT\*'s **choose-parent** step and *without* its **rewire** step. `FallbackPlanner` runs it only when Hybrid A\* fails to reach the goal.

## Why not the full RRT\*
Rewiring requires an **exact steering function**: given two poses, the optimal path connecting them. For a car that means Reeds–Shepp or Dubins curves. This planner extends by arc primitives, which cannot land on an arbitrary pose exactly — so rewiring would be approximate, and the asymptotic-optimality guarantee that the star denotes would not actually hold.

Choosing the best parent among nearby nodes recovers most of the practical path-quality benefit and claims nothing untrue. The deck currently says "Hybrid A\* / RRT\*"; either add Dubins steering (a small, well-defined piece of work) or soften the claim before judging.

## Why have a fallback at all
Hybrid A\* searches a fixed lattice, which is what you want most of the time — deterministic, repeatable, fast enough to replan at 4 Hz. It fails in exactly the cases this project is about: a corridor that pinches below the lattice resolution, or a manoeuvre needing a heading the 24 bins do not contain. Sampling is not constrained to a grid of poses and can still find a way through.

Measured: Hybrid A\* solves the open-road, pothole and narrow-corridor cases in 15–100 ms; the RRT solves all three too, using its full 90 ms budget. Both correctly fail on a genuinely walled road, and the fallback then returns the longer of the two partial paths with `terminal_stop=True` so the controller brakes rather than driving into the dead end.

## Consequences
- The common case costs nothing: the sampler runs only after a lattice failure, and `FallbackPlanner` counts `fallback_calls` / `fallback_rescues` so its value is measurable rather than assumed.
- Sampling is seeded, so runs stay repeatable and the ablation still compares planners rather than luck.
