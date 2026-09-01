"""The simulator seam.

Everything above stage 3 talks to a simulator only through this protocol, so
swapping the built-in 2-D world for CARLA changes one object and nothing else.
``ADR-005`` promised this; here it is made explicit rather than implied by duck
typing, because the CARLA bridge is written against it and a mismatch should
fail loudly at review time rather than quietly at run time.

Anything implementing :class:`SimWorld` can be handed to
``divas.eval.runner.run`` unchanged.
"""

from __future__ import annotations

from typing import List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from divas.types import EgoState, OccupancyGrid, Track, VehicleParams


@runtime_checkable
class RouteSource(Protocol):
    """Whatever provides the global route.  A road centreline, or CARLA's
    waypoint graph -- the stack only ever asks how far along it has come."""

    def progress(self, x: float, y: float) -> float:
        """Arc length travelled along the route, metres."""


@runtime_checkable
class SimWorld(Protocol):
    """What the runner needs from a simulator."""

    ego: EgoState
    params: VehicleParams
    t: float
    road: RouteSource

    def step(self, dt: float, accel: float, steer: float) -> None:
        """Apply a control and advance by ``dt``."""

    def ground_truth_grids(
        self, half_extent: float = 32.0, resolution: float = 0.25
    ) -> Tuple[OccupancyGrid, OccupancyGrid]:
        """``(static, full)`` local costmaps -- without and with actors.

        Two, not one: the planner needs actors as obstacles now, while the
        predictor must not see them in the grid at all or it double-counts its
        own social repulsion.
        """

    def ground_truth_tracks(self) -> List[Track]:
        """Dynamic actors in the odom frame, with measurement noise."""

    def local_goal(self, lookahead: float = 28.0) -> np.ndarray:
        """Route target ahead of the ego, odom frame."""

    def collision(self) -> Optional[str]:
        """A description of what was hit, or ``None``."""

    def clearance_to_actors(self) -> float:
        """Smallest surface-to-surface gap to any actor, metres."""

    def time_to_collision(self) -> float:
        """Constant-velocity TTC against the nearest closing actor."""


def check(world) -> List[str]:
    """Report which parts of the protocol an object is missing.

    Used by the CARLA bridge's self-test.  ``isinstance`` against a
    ``runtime_checkable`` Protocol only checks that attributes exist, which is
    exactly the shallow check that lets a typo reach a two-hour simulation run.
    """
    required = [
        "step", "ground_truth_grids", "ground_truth_tracks", "local_goal",
        "collision", "clearance_to_actors", "time_to_collision",
    ]
    missing = [n for n in required if not callable(getattr(world, n, None))]
    for attr in ("ego", "params", "t", "road"):
        if not hasattr(world, attr):
            missing.append(attr)
    if hasattr(world, "road") and not callable(getattr(world.road, "progress", None)):
        missing.append("road.progress")
    return missing
