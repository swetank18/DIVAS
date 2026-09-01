"""The scenario suite.

Five scenarios from ``EXECUTION_PLAN.md`` Phase 1, plus a combined one.  Each
targets a specific assumption that a lane-centric stack makes and Indian roads
break.  They are scripted rather than reactive so that an ablation compares
planners rather than luck; ``seed`` varies actor noise and the erratic
policies, so repeated runs still sample a distribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from divas.sim.world import Actor, Circle, Rect, Road, World
from divas.types import EgoState, VehicleParams


@dataclass
class Scenario:
    name: str
    description: str
    build: Callable[[int], World]
    goal_progress: float = 110.0   # m of centreline to cover for success
    time_limit: float = 32.0       # s
    tests: str = ""                # which assumption this breaks


def _ego(v: float = 8.0) -> EgoState:
    return EgoState(x=0.0, y=0.0, theta=0.0, v=v)


def unmarked_road(seed: int) -> World:
    rng = np.random.default_rng(seed)
    road = Road.winding(half_width=4.5, amplitude=7.0, pinch=1.6)
    s0 = road.progress(0.0, 0.0)
    statics = []
    for i in range(5):
        p = road.offset_point(s0 + 35.0 + 18.0 * i, float(rng.uniform(-2.2, 2.2)))
        statics.append(Circle(p[0], p[1], float(rng.uniform(0.5, 0.9)), "pothole"))
    return World(road=road, statics=statics, actors=[], ego=_ego(9.0), seed=seed)


def pothole_slalom(seed: int) -> World:
    rng = np.random.default_rng(seed)
    statics = []
    for i in range(9):
        x = 20.0 + 12.0 * i
        y = float(rng.uniform(-2.5, 2.5))
        statics.append(Circle(x, y, float(rng.uniform(0.7, 1.3)), "pothole"))
    statics.append(Rect(70.0, 3.0, 4.2, 1.8, 0.0, "parked_car"))
    statics.append(Rect(105.0, -3.2, 4.2, 1.8, 0.05, "parked_car"))
    return World(road=Road.straight(half_width=5.0), statics=statics,
                 actors=[], ego=_ego(9.0), seed=seed)


def two_wheeler_cutin(seed: int) -> World:
    rng = np.random.default_rng(seed)
    actors = [
        Actor(1, "motorcycle", 26.0, 3.2, 0.0, 7.5, "cutin",
              {"trigger_gap": 15.0, "heading_change": -0.75, "duration": 1.4}),
        Actor(2, "motorcycle", 55.0, -3.0, 0.0, 8.0, "cutin",
              {"trigger_gap": 13.0, "heading_change": 0.7, "duration": 1.2}),
        Actor(3, "car", 90.0, 1.0, 0.0, 7.0, "stop_and_go", {"cruise": 7.0, "period": 7.0}),
    ]
    return World(road=Road.straight(half_width=5.0), statics=[],
                 actors=actors, ego=_ego(10.0), seed=seed)


def pedestrian_crossing(seed: int) -> World:
    actors = [
        Actor(1, "pedestrian", 30.0, -5.5, 1.5708, 0.0, "wait_then_go",
              {"cruise": 1.4, "start_time": 1.2}),
        Actor(2, "pedestrian", 32.0, -6.2, 1.5708, 0.0, "wait_then_go",
              {"cruise": 1.1, "start_time": 1.9}),
        Actor(3, "animal", 72.0, 2.5, -1.5708, 0.0, "wait_then_go",
              {"cruise": 0.9, "start_time": 6.0}),
    ]
    return World(road=Road.straight(half_width=6.0), statics=[],
                 actors=actors, ego=_ego(9.0), seed=seed)


def oncoming_negotiation(seed: int) -> World:
    """Shared narrow carriageway: an oncoming truck, parked vehicles pinching
    the gap.  There is no lane to yield into -- only free space to share."""
    statics = [
        Rect(45.0, 2.6, 4.4, 1.9, 0.0, "parked_car"),
        Rect(80.0, -2.8, 4.4, 1.9, 0.03, "parked_car"),
    ]
    actors = [
        Actor(1, "truck", 95.0, 2.2, np.pi, 7.0, "constant"),
        Actor(2, "autorickshaw", 40.0, -2.0, 0.0, 4.5, "erratic",
              {"sigma": 0.05, "vmin": 3.0, "vmax": 6.0}),
    ]
    return World(road=Road.straight(half_width=4.6), statics=statics,
                 actors=actors, ego=_ego(8.0), seed=seed)


def mixed_traffic(seed: int) -> World:
    """Everything at once -- the actual road."""
    rng = np.random.default_rng(seed)
    road = Road.winding(half_width=5.2, amplitude=5.0, pinch=1.0)
    s0 = road.progress(0.0, 0.0)
    p1 = road.offset_point(s0 + 48.0, 1.2)
    p2 = road.offset_point(s0 + 96.0, 3.0)
    statics = [Circle(p1[0], p1[1], 0.9, "pothole"),
               Rect(p2[0], p2[1], 4.2, 1.8, 0.0, "parked_car")]
    actors = [
        Actor(1, "motorcycle", 24.0, 2.6, 0.0, 7.0, "erratic",
              {"sigma": 0.08, "vmin": 5.0, "vmax": 10.0}),
        Actor(2, "autorickshaw", 38.0, -1.8, 0.0, 5.0, "stop_and_go",
              {"cruise": 5.5, "period": 8.0}),
        Actor(3, "pedestrian", 60.0, -5.0, 1.5708, 0.0, "wait_then_go",
              {"cruise": 1.3, "start_time": 4.0}),
        Actor(4, "bus", 78.0, 1.5, 0.0, 6.0, "constant"),
        Actor(5, "motorcycle", 52.0, -3.4, 0.0, 8.5, "cutin",
              {"trigger_gap": 14.0, "heading_change": 0.6, "duration": 1.3}),
        Actor(6, "animal", 110.0, 2.0, -1.5708, 0.0, "wait_then_go",
              {"cruise": 0.7, "start_time": 9.0}),
    ]
    return World(road=road, statics=statics, actors=actors, ego=_ego(9.0), seed=seed)


def reactive_overtaking(seed: int) -> World:
    """A slow bus with reactive traffic streaming around it.

    Built specifically to be unpredictable by extrapolation.  Each overtaking
    vehicle travels straight right up to the moment it pulls out, so a
    constant-velocity model sees no warning at all -- while a model that knows
    the vehicle is closing on a slow leader with room to one side can see it
    coming.  If interaction-aware prediction is worth anything, it is worth
    something here.
    """
    actors = [
        Actor(1, "bus", 45.0, 0.5, 0.0, 4.5, "reactive", {"v0": 4.5}),
        Actor(2, "motorcycle", 30.0, 0.0, 0.0, 9.0, "reactive", {"v0": 10.0}),
        Actor(3, "autorickshaw", 22.0, -1.2, 0.0, 6.5, "reactive", {"v0": 7.5}),
        Actor(4, "car", 12.0, 1.0, 0.0, 8.0, "reactive", {"v0": 9.5}),
        Actor(5, "motorcycle", 70.0, -0.8, 0.0, 8.5, "reactive", {"v0": 10.5}),
        Actor(6, "truck", 100.0, 0.8, 0.0, 5.0, "reactive", {"v0": 5.5}),
    ]
    return World(road=Road.straight(half_width=6.0), statics=[],
                 actors=actors, ego=_ego(9.0), seed=seed)


def reactive_dense(seed: int) -> World:
    """Dense reactive mixed traffic through a corridor that narrows.

    Everything reacts to everything, including the ego, so who yields to whom
    is genuinely uncertain -- which is the property the confidence signal and
    the dynamic margin are supposed to exploit.
    """
    rng = np.random.default_rng(seed)
    road = Road.winding(half_width=6.0, amplitude=4.0, pinch=1.4)
    s0 = road.progress(0.0, 0.0)
    kinds = ["motorcycle", "autorickshaw", "car", "motorcycle",
             "bus", "motorcycle", "autorickshaw", "car"]
    actors = []
    for i, k in enumerate(kinds):
        p = road.offset_point(s0 + 18.0 + 13.0 * i, float(rng.uniform(-2.5, 2.5)))
        tangent = road.tangents()[road.station_of(p[0], p[1])]
        actors.append(Actor(i + 1, k, float(p[0]), float(p[1]),
                            float(np.arctan2(tangent[1], tangent[0])),
                            float(rng.uniform(4.0, 8.0)), "reactive"))
    p = road.offset_point(s0 + 90.0, 1.0)
    return World(road=road, statics=[Circle(float(p[0]), float(p[1]), 0.9, "pothole")],
                 actors=actors, ego=_ego(9.0), seed=seed)


SCENARIOS: Dict[str, Scenario] = {
    s.name: s
    for s in [
        Scenario("unmarked_road", "Winding corridor, no markings, scattered potholes",
                 unmarked_road, tests="lane detection has nothing to detect"),
        Scenario("pothole_slalom", "Potholes and parked vehicles pinching the corridor",
                 pothole_slalom, tests="road surface is not uniform"),
        Scenario("two_wheeler_cutin", "Two-wheelers cutting across with ~1 s warning",
                 two_wheeler_cutin, tests="a fixed safety buffer is enough"),
        Scenario("pedestrian_crossing", "Pedestrians and cattle entering from the verge",
                 pedestrian_crossing, tests="agents stay in their lane"),
        Scenario("oncoming_negotiation", "Shared carriageway, oncoming truck, parked cars",
                 oncoming_negotiation, tests="a lane graph defines legal routes"),
        Scenario("mixed_traffic", "All of the above at once",
                 mixed_traffic, goal_progress=120.0, time_limit=38.0,
                 tests="everything"),
        Scenario("reactive_overtaking", "Reactive traffic overtaking a slow bus",
                 reactive_overtaking, goal_progress=110.0, time_limit=34.0,
                 tests="agents move at constant velocity"),
        Scenario("reactive_dense", "Dense reactive mixed traffic, narrowing corridor",
                 reactive_dense, goal_progress=110.0, time_limit=38.0,
                 tests="who yields to whom is knowable in advance"),
    ]
}


def get(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise KeyError(f"unknown scenario {name!r}; have {sorted(SCENARIOS)}")
    return SCENARIOS[name]
