# ADR-008: Evaluation is iteration-bounded, deployment is clock-bounded

**Status:** accepted · **Phase:** 1 · **Invalidates:** every ablation number produced before it

## Decision
`PlannerConfig.time_budget_ms` and `RRTConfig.time_budget_ms` accept `None`. The evaluation runner sets both to `None`, leaving `max_iterations` as the only termination condition. The wall-clock deadline remains the default for deployment.

## Why
A wall-clock deadline is the right behaviour on the vehicle: overrunning the control period is worse than returning a mediocre path, so the planner returns whatever it has reached when the clock runs out.

It is the *wrong* behaviour in an experiment, because "whatever it has reached" depends on how loaded the machine is. This was not a theoretical concern:

| run | stacks × seeds × scenarios | `full_dynamic_margin` success | collisions |
|---|---|---|---|
| A | 5 × 8 × 6 = 240 sims, 12 workers | 0.62 | 0.27 |
| B | 3 × 8 × 6 = 144 sims, 12 workers | 0.79 | 0.12 |

Identical code, identical seeds, identical configuration. The only difference was CPU contention. Run A's planner was getting fewer expansions per replan than run B's, and the resulting paths were worse.

Any conclusion drawn across rows measured under different load was measuring the host, not the algorithm — and the difference is comparable in size to the effects the ablation exists to detect. **Every ablation number produced before this ADR should be discarded.**

## Consequences
- Results are now bit-reproducible: the same stack and seed produce identical progress, speed and outcome on repeated runs. Verified.
- `max_iterations` reduced to 4000 (Hybrid A\*) and 800 (RRT) so the iteration bound binds well before any plausible deadline.
- Evaluation plan times are ~113 ms p95 and are *not* a real-time claim. The real-time claim needs a separate measurement with the deployment budget enabled, ideally on the target hardware — that is Phase 4 work, and it is now listed there.
- General lesson for this project: anything that terminates on wall-clock time cannot appear in a benchmark without this switch.
