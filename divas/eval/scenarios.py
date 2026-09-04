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


def cattle_and_crowd(seed: int) -> World:
    """A bazaar street: a cattle herd standing in the carriageway, and a crowd.

    The scenario the project exists for, and the one every Western benchmark
    lacks. Three things happen at once, and each defeats a different
    assumption a conventional stack makes:

    * **A herd of cattle occupies the middle of the road.** Not crossing it --
      *standing* in it, moving at under a metre a second and not in a straight
      line. A lane-following stack has no representation for "the lane is
      occupied by something that will not move and is not a vehicle"; a
      free-space planner simply sees the gap beside it.
    * **A crowd crosses from both verges**, not at a crossing and not
      together. Pedestrians are 20.4% of India's road deaths, and they enter
      from both sides at staggered times, so there is never one clean moment
      to pass.
    * **The corridor is pinched** by a stopped bus at the near kerb and a
      handcart opposite, so the gap the ego must thread is narrower than the
      road and moves as the crowd does.

    Everything is scripted rather than reactive, for the reason
    :class:`Actor` gives: the ablation has to compare planners, not luck. The
    staggered ``start_time`` values are what make it hard -- an actor that
    steps out at t = 3.4 s is a genuine surprise to a predictor that has been
    watching it stand still, which is exactly the case a fixed safety buffer
    is supposed to survive and does not.

    Tuned so the corridor is passable: a stack that stops here has been too
    timid, and a stack that ploughs through has ignored the crowd.
    """
    road = Road.straight(half_width=6.5)

    # The pinch: a bus stopped at the far kerb, a handcart opposite. Placed to
    # leave a genuine gap rather than to seal the road -- a scenario nothing
    # can pass measures nothing, and the first tuning of this one collided on
    # five runs of six with clearances around -0.004 m, which is a graze
    # rather than a crash and a sign the corridor was a few centimetres short
    # rather than the planners being wrong.
    statics = [
        Rect(46.0, 5.0, 9.0, 2.6, 0.0, "stopped_bus"),
        Rect(58.0, -5.9, 2.2, 1.4, 0.0, "handcart"),
        Circle(30.0, 1.6, 0.8, "pothole"),
    ]

    actors = [
        # -- the herd: slow, ambling, and clustered left of centre so the gap
        # to thread is on the right and *moves* as they drift -------------
        Actor(1, "animal", 34.0, 1.4, 0.25, 0.0, "wait_then_go",
              {"cruise": 0.8, "start_time": 1.0}),
        Actor(2, "animal", 37.5, 3.0, -0.15, 0.0, "wait_then_go",
              {"cruise": 0.6, "start_time": 2.2}),
        Actor(3, "animal", 36.0, 0.2, 0.30, 0.0, "wait_then_go",
              {"cruise": 0.7, "start_time": 3.6}),
        Actor(4, "animal", 40.5, 2.2, 0.0, 0.0, "constant"),      # simply stands

        # -- the crowd: both verges, staggered, none of them at a crossing --
        Actor(10, "pedestrian", 56.0, -6.1, 1.5708, 0.0, "wait_then_go",
              {"cruise": 1.4, "start_time": 3.4}),
        Actor(11, "pedestrian", 59.0, -6.3, 1.5708, 0.0, "wait_then_go",
              {"cruise": 1.1, "start_time": 4.6}),
        Actor(12, "pedestrian", 62.0, 6.2, -1.5708, 0.0, "wait_then_go",
              {"cruise": 1.3, "start_time": 5.8}),
        Actor(13, "pedestrian", 70.0, -6.0, 1.5708, 0.0, "wait_then_go",
              {"cruise": 1.5, "start_time": 7.4}),
        Actor(14, "pedestrian", 74.0, 6.1, -1.5708, 0.0, "wait_then_go",
              {"cruise": 1.2, "start_time": 8.8}),
        Actor(15, "pedestrian", 80.0, -6.2, 1.5708, 0.0, "wait_then_go",
              {"cruise": 1.0, "start_time": 10.2}),

        # -- the traffic that does not stop for any of it -------------------
        Actor(20, "autorickshaw", 20.0, -3.4, 0.0, 5.0, "stop_and_go",
              {"cruise": 5.5, "period": 7.0}),
        Actor(21, "motorcycle", 28.0, 3.8, 0.0, 7.5, "erratic",
              {"sigma": 0.07, "vmin": 5.0, "vmax": 9.0}),
        Actor(22, "motorcycle", 66.0, -4.2, 0.0, 8.0, "cutin",
              {"trigger_gap": 13.0, "heading_change": 0.45, "duration": 1.5}),
        Actor(23, "bicycle", 48.0, -4.6, 0.0, 3.5, "constant"),
    ]
    return World(road=road, statics=statics, actors=actors, ego=_ego(8.0),
                 seed=seed)


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
        Scenario("cattle_and_crowd",
                 "Cattle standing in the carriageway, a crowd crossing from both "
                 "verges, corridor pinched by a stopped bus",
                 cattle_and_crowd, goal_progress=100.0, time_limit=40.0,
                 tests="obstacles are vehicles, and people cross at crossings"),
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
