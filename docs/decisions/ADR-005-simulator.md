# ADR-005: A built-in 2-D world for Phase 1, simulator choice still open

**Status:** accepted, temporary · **Phase:** 1 · **Blocks:** open decision #1

## Decision
`divas.sim.world` is a minimal 2-D simulator that produces ground-truth occupancy grids and tracks in the stage-contract types. Phase 1 runs on it. It depends on neither CARLA nor MATLAB.

## Why
Open decision #1 (CARLA-primary vs MATLAB-primary) is unresolved and is the user's call — it depends on whether SIH26037's sponsoring organisation expects MathWorks deliverables. Waiting for it would have blocked the entire vertical slice, and the slice is what de-risks everything else.

Everything built against this simulator talks only through `divas.types`, so the answer to #1 changes one file and nothing else.

## What it deliberately is not
No sensor models, no images, no dynamics beyond a kinematic bicycle. It cannot validate perception, and it is not evidence about the real world. It exists to let stages 4–6 be built, closed-loop and measured before stages 1–3 exist.

## Consequences
- Scenario obstacles must be placed in the *road's* frame, not world coordinates — otherwise a pothole meant for the tarmac lands in a field on a curving road, silently turning a navigation test into an impossible one.
- When #1 lands, this file is replaced by a CARLA or Simulink bridge publishing the same types.
