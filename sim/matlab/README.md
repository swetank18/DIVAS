# MATLAB cross-validation

An independent implementation, in a second tool, of the two models every
published DIVAS number rests on: the kinematic bicycle the built-in simulator
integrates, and the longitudinal controller identified against a live CARLA
server.

```bash
python3 scripts/export_for_matlab.py          # writes reference/
matlab -batch "cd('sim/matlab'); validate_against_python"
```

Needs base MATLAB only — no toolboxes. `divas_wrap_angle.m` exists precisely so
that `wrapToPi` (Mapping Toolbox) is not required, because a cross-validation
that needs a licence the reviewer lacks is one nobody runs.

## What it establishes

The bicycle integrator and the pedal controller are not artefacts of one
codebase. Written twice, in two languages, from the same equations, they agree
to floating-point noise. The input sequence is chosen to exercise the parts
that are easy to get wrong — a steer reversal, a steer step larger than the
rate limit, a jump from full braking to full throttle, and braking at a
standstill — because a smooth sinusoid would agree between any two
implementations and prove nothing.

## What it does not establish

It does not touch the planner, the predictor, the risk field or the perception
stubs; none of them are reimplemented here. It cannot catch an error *shared*
by both implementations, since both were written from the same equations. It is
a check on transcription and arithmetic — which is the class of bug that
actually happens — not a proof that the model is the right model.

## Not yet done

RoadRunner is the reason the MATLAB licence matters beyond this: CARLA's stock
maps are Western roads *with* lane markings, and the stack ignores lane
markings by design, so free-space navigation still tests honestly — but the
road *texture* will not be Indian until a custom map is built. That is a
multi-day asset job, not a script, and nothing in the deck should claim it yet.
