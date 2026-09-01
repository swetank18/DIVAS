# ADR-003: Keep-outs are oriented ellipses, not discs

**Status:** accepted · **Phase:** 1

## Decision
The keep-out around each predicted actor is an ellipse oriented along its predicted heading, with semi-axes formed by summing the actor's and the ego's box half-extents axis-wise, plus `d_safe`:

```
a = actor.half_length + ego.half_length + d_safe     (along heading)
b = actor.half_width  + ego.half_width  + d_safe     (across heading)
```

## Why
A disc must circumscribe the object, so it charges the *lateral* keep-out for the object's full diagonal. A bus circumscribed by a disc has a 5.65 m radius and blocks 11 m of lateral space — wider than most of the carriageways in the scenario suite. Every scenario with a bus became unsolvable, and the stack looked cautious when it was actually blind.

The same error applied to the ego and was fixed the same way: its 1.07 m footprint-disc radius was charging lateral clearance for a 3.9 m long vehicle.

## Consequences
- Lateral gaps are judged on lateral geometry, which is the direction that decides whether a gap is passable.
- Heading comes from finite differences of the predicted mode, so the ellipse turns with the prediction.
- Still conservative when ego and actor headings differ a lot (an axis-wise box sum is not a true Minkowski sum for rotated boxes). Acceptable, and conservative in the safe direction.
