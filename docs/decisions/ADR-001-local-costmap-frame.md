# ADR-001: The local costmap is ego-centred but world-aligned

**Status:** accepted · **Phase:** 1 · **Supersedes:** the "40 × 20 m @ 0.2 m ego-frame grid" in `PROJECT_OVERVIEW.md` §3

## Decision
Stage 3 publishes a **64 × 64 m rolling window at 0.25 m/cell**, centred on the ego and aligned to the odom frame. It translates with the vehicle; it does not rotate with it.

## Why
The original spec had the grid in the ego *body* frame. That forces a rotation into every consumer, and the rotation is time-varying: the planner runs at 4 Hz and the controller at 20 Hz, so a path produced in one body frame is consumed in four later ones. Every one of those transforms is a chance to be subtly wrong, and the failure mode is a path that drifts sideways as the vehicle turns.

A rolling window in a fixed frame is what real local costmaps do (ROS `costmap_2d`), and it makes the plan and the control step share one frame with nothing to go stale in between.

Square rather than forward-biased because the window no longer rotates: a 40 × 20 m window aligned to the world is nearly useless when the road turns 45°.

## Consequences
- Tracks are also published world-aligned, for the same reason.
- 65k cells instead of 20k. Grid construction and the EDT cost a few ms at 10 Hz — measured, acceptable.
- The window must exceed the route lookahead (see ADR-006), which forced 32 m half-extent.
