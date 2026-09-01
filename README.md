# DIVAS — Adaptive Path Planning & Collision Avoidance for Unstructured Indian Roads

**SIH26037 · Team DIVAS · SRMIST · Smart India Hackathon 2026**

A free-space-centric autonomy stack. Where a conventional stack asks *"where is my lane?"*, this one asks *"where can I physically go, and who is about to get in the way?"* — because on Indian roads there often is no lane.

- **[`PROJECT_OVERVIEW.md`](PROJECT_OVERVIEW.md)** — what the project is and why
- **[`EXECUTION_PLAN.md`](EXECUTION_PLAN.md)** — phased build plan, ownership, open decisions
- **[`docs/decisions/`](docs/decisions/)** — architecture decision records

## Status

Phases 0 and 1 are complete: the full six-stage pipeline runs closed-loop, all three rates are honoured, and the metrics harness reports. Stages 1–3 (perception, projection, fusion) are ground-truth stubs served by the simulator; stages 4–6 (prediction, planning, control) are real.

The [CARLA bridge](divas/sim/carla_bridge.py) is written and unit-tested against a stand-in for the `carla` module, so the same runner and the same metrics drive either simulator — but it has not yet been run against a real server. See [`STATUS.md`](STATUS.md).

## Quick start

```bash
python3 -m pytest tests/ -q                      # 56 tests, no ROS, GPU or CARLA needed
python3 scripts/run_demo.py --scenario mixed_traffic   # run one scenario, write a PNG
python3 scripts/run_ablation.py --seeds 8 --jobs 8     # the ablation table
python3 scripts/make_comparison.py --scenario mixed_traffic \
    --left baseline_conventional --right cv_pred_fixed_margin   # jury video + slide
python3 scripts/calibration.py                        # is the confidence calibrated?
python3 scripts/tune_predictor.py                     # open-loop predictor accuracy
python3 scripts/verify_idd.py --root ~/IDD        # Phase 0 dataset check
python3 scripts/run_carla.py --check                  # CARLA bridge self-test (needs a server)
```

## The pipeline

```
 [camera] ──▶ 1. PERCEPTION ──▶ 2. BEV PROJECTION ──▶ 3. FUSION ──┐   stub
 [radar]   ───────────────────────────────────────────────────────┤   today
 [LiDAR]   ───────────────────────────────────────────────────────┤
                                                                  ▼
        6. CONTROL ◀── 5. PLANNING ◀── 4. PREDICTION ◀── occupancy + tracks
             │              real            real
             ▼
      (steer, accel)
```

Every stage boundary is a typed contract in [`divas/types.py`](divas/types.py), mirrored by the ROS 2 messages in [`msgs/`](msgs/). Algorithm code never imports `rclpy`, so any stage can be unit-tested — and any stub swapped for a real module — without a ROS environment.

| Rate | Stage |
|---|---|
| 10 Hz | perception, fusion, prediction |
| 4 Hz | planning (Hybrid A\*, RRT fallback) |
| 20 Hz | control (MPC) |

They are kept separate on purpose. Collapsing them would make every result optimistic, because the controller would never once have to act on a stale plan — which on the real vehicle is the normal case.

## Where the idea becomes a number

Predicted actors are not passed downstream as hard obstacles. They become a time-indexed risk field whose extent is governed by

```
d_safe(t) = d0 + k_v · v_ego + λ · (1 − confidence(t))
```

The buffer around a predicted actor **widens when the predictor is unsure and tightens when it is sure**, at every step of the horizon independently. Setting `λ = 0` recovers the conventional fixed policy — which is exactly the ablation. See [`divas/prediction/risk.py`](divas/prediction/risk.py).

## Layout

```
divas/
  types.py          the stage contracts — everything else depends on this
  sim/              2-D world serving ground-truth stubs (see ADR-005),
                    with scripted and reactive (IDM + overtaking) traffic
  perception/       IDD dataset pipeline + drivable-area remapping
  fusion/           (Phase 3)
  prediction/       constant-velocity baseline, social-force, risk field
  planning/         Hybrid A* over free space, kinodynamic RRT fallback
  control/          pure-pursuit baseline, risk-aware sampling MPC
  eval/             metrics, scenario suite, closed-loop runner
msgs/               ROS 2 interface definitions
scripts/            demo, ablation, dataset verification
tests/              unit + end-to-end
```

Requires numpy, scipy, matplotlib, opencv. PyTorch is needed only for Phase 2.
