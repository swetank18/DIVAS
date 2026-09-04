# MATLAB replay — a real closed-loop run, animated

Distinct from `sim/matlab/`, which cross-validates the bicycle and
longitudinal models against Python. This folder does something else:
plays back a **real recorded closed-loop episode** — two stacks, same
scenario, same seed, side by side — the MATLAB equivalent of
`scripts/make_comparison.py`'s output, animated instead of a static PNG.

```bash
python3 scripts/export_replay.py --scenario pedestrian_crossing   # writes docs/replay-pedestrian_crossing.json, if it isn't already there
matlab -batch "cd('sim/MATLABS'); replay_animation('../../docs/replay-pedestrian_crossing.json')"
```

Or interactively, from the MATLAB prompt, after `cd sim/MATLABS`:

```matlab
replay_animation                                            % default scenario
replay_animation('../../docs/replay-mixed_traffic.json')
replay_animation('../../docs/replay-pedestrian_crossing.json', true)   % also writes an .avi
```

No toolboxes — `jsondecode` and `VideoWriter` are both base MATLAB.

## What it shows

Road corridor, static obstacles (potholes, parked vehicles), actors as
class-coloured oriented boxes, the ego vehicle, its planned path, and a
ring showing the safety margin actually in force that frame — all read
straight from the JSON `scripts/export_replay.py` already writes for the
web replay. **Not a re-enactment.** It is the literal recorded trace of a
real `divas.eval.runner.run` call. If this animation and the Python
comparison PNG for the same scenario/seed ever disagree, one of the two
has a bug — they read the same source of truth.

## What it does not show

Nothing here re-simulates anything. It is a player, not a solver — if you
change the planner, the predictor, or a scenario, you have to re-export
the replay JSON before this shows the new behaviour. It also inherits
whatever the closed-loop runner already does and doesn't claim — see
`CONTEXT.md` and `STATUS.md` for what the underlying numbers do and don't
prove (in particular: whether a given scenario's two arms actually
diverge, or both just complete the same way).

## Status: written, not yet run on real MATLAB

This machine has no MATLAB install (see `STATUS.md`'s MATLAB section —
same situation as `sim/matlab/`). The code is reviewed carefully —
including the two known MATLAB `jsondecode` gotchas it works around
(a JSON array of mixed-type elements, like `[x, y, theta, "pedestrian"]`,
decodes to a cell array rather than a numeric matrix; a JSON array of
objects with differing field sets decodes to a cell array of structs
rather than a struct array — `idx_any()` and `numel_rows()` handle both
without assuming which one you got) — but **it has not actually been run**.
Treat it the same way `sim/matlab/README.md` asks you to treat the .m
files there: reviewed, not verified, until someone with MATLAB runs it
once and reports back. The first run is also the first real check on
whether the `VideoWriter('Motion JPEG AVI')` choice (made because
`'MPEG-4'` isn't available on Linux, where this team's MATLAB lives) is
actually correct on your installation.
