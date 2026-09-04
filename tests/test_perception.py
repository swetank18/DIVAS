"""Tests for stage 1 (drivable-area segmentation) and stage 2 (BEV projection).

These run without torch and without the dataset. The two things worth pinning
here are geometric rather than statistical: a label map that composites in the
wrong order, and a ground-plane projection with a sign error, both produce
output that looks entirely plausible and is wrong in a way no accuracy metric
would reveal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from divas.perception.bev import (
    BevSpec, Camera, free_space_to_grid, horizon_row, max_reliable_range,
    min_visible_range, project, reachable_goal,
)
from divas.perception.datasets.idd_polygons import (
    IGNORE, OTHER, ROAD, SHOULDER, class_of, drivable_from_mask, rasterize,
)


def _rec(objects, w=100, h=60):
    return {"imgWidth": w, "imgHeight": h, "objects": objects}


def _poly(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


# --------------------------------------------------------------------------
# label mapping
# --------------------------------------------------------------------------


def test_the_drivable_set_comes_from_names_not_ids():
    """IDD ships polygons carrying label *names*, so the fragile numeric-id
    hypothesis in idd.py is not needed at all."""
    assert class_of("road") == ROAD
    assert class_of("drivable fallback") == SHOULDER
    assert class_of("non-drivable fallback") == OTHER
    assert class_of("out of roi") == IGNORE
    assert class_of("motorcycle") == OTHER


def test_parking_is_not_a_drivable_class_because_idd_has_no_such_label():
    """idd.py names the drivable set as (road, parking, drivable fallback).
    Scanning 400 annotation files found no `parking` label anywhere in IDD, so
    anything named that falls through to not-drivable rather than silently
    widening the free space."""
    assert class_of("parking") == OTHER


def test_shoulder_is_separate_so_including_it_stays_a_decision():
    mask = np.array([[ROAD, SHOULDER, OTHER]], dtype=np.uint8)
    assert drivable_from_mask(mask, include_shoulder=True).tolist() == [[True, True, False]]
    assert drivable_from_mask(mask, include_shoulder=False).tolist() == [[True, False, False]]


# --------------------------------------------------------------------------
# rasterisation
# --------------------------------------------------------------------------


def test_later_polygons_occlude_earlier_ones():
    """Draw order *is* the annotation: objects are listed back to front.

    Composite them any other way -- by class priority, say -- and a motorcycle
    standing on the road stays road. The model then learns that vehicles are
    transparent, which is the most dangerous label bug available here.
    """
    rec = _rec([
        {"label": "road", "polygon": _poly(0, 0, 100, 60)},
        {"label": "motorcycle", "polygon": _poly(40, 20, 60, 40)},
    ])
    m = rasterize(rec)
    assert m[30, 50] == OTHER          # the motorcycle, drawn last
    assert m[30, 10] == ROAD           # road either side of it


def test_polygons_are_scaled_before_filling_not_after():
    """Resizing a finished label map with nearest-neighbour erodes thin
    structures and invents boundary pixels between classes that were never
    adjacent. Scaling the coordinates first cannot do either."""
    rec = _rec([{"label": "road", "polygon": _poly(0, 0, 50, 60)}])
    full = rasterize(rec)
    half = rasterize(rec, size=(50, 30))

    assert full.shape == (60, 100)
    assert half.shape == (30, 50)
    # the road covers the left half in both, to within a pixel
    assert abs((full == ROAD).mean() - (half == ROAD).mean()) < 0.02


def test_deleted_objects_are_skipped():
    rec = _rec([
        {"label": "road", "polygon": _poly(0, 0, 100, 60)},
        {"label": "car", "polygon": _poly(0, 0, 100, 60), "deleted": 1},
    ])
    assert (rasterize(rec) == ROAD).all()


def test_degenerate_polygons_do_not_raise():
    """Annotation files contain two-point and empty polygons; a loader that
    dies on one loses the whole epoch."""
    rec = _rec([
        {"label": "road", "polygon": [[1, 1], [2, 2]]},
        {"label": "road", "polygon": []},
    ])
    assert rasterize(rec).shape == (60, 100)


# --------------------------------------------------------------------------
# ground-plane projection
# --------------------------------------------------------------------------


def _cam():
    return Camera.from_fov(512, 288, fov_deg=90.0, height=1.35,
                           pitch=math.radians(5.0))


def test_projection_round_trips_to_the_centimetre():
    """The forward map and the range inverse must agree, or the flat-ground
    limit drawn on every figure is decoration."""
    cam = _cam()
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    for x in (10.0, 20.0, 40.0):
        _u, v, ok = project(cam, np.array([x]), np.array([0.0]))
        assert ok[0]
        dv = (v[0] - cam.cy) / cam.fy
        recovered = cam.height * (c - dv * s) / (dv * c + s) + cam.x_offset
        assert recovered == pytest.approx(x, abs=0.01)


def test_further_ground_is_higher_in_the_image_and_left_is_left():
    cam = _cam()
    _u10, v10, _ = project(cam, np.array([10.0]), np.array([0.0]))
    _u25, v25, _ = project(cam, np.array([25.0]), np.array([0.0]))
    assert v25[0] < v10[0]                       # further = nearer the horizon

    u_left, _v, _ = project(cam, np.array([10.0]), np.array([3.0]))
    assert u_left[0] < cam.cx                    # +y is left, and left is left


def test_points_behind_the_camera_are_invalid_not_merely_wrong():
    """A point behind the lens has negative depth and projects to a perfectly
    plausible pixel in front of it. It has to be masked, not divided by."""
    cam = _cam()
    _u, _v, ok = project(cam, np.array([-5.0]), np.array([0.0]))
    assert not ok[0]


def test_max_reliable_range_is_finite_and_beyond_the_horizon_is_not():
    """The inverse had its signs the other way round at first, which made the
    denominator negative for every row above the horizon and reported the
    reliable range as infinity -- 'trust everything'."""
    cam = _cam()
    reach = max_reliable_range(cam, 288)
    assert 20.0 < reach < 120.0
    assert horizon_row(cam) < cam.cy             # pitched down, horizon is up


def test_unseen_ground_is_occupied_rather_than_free():
    """'No evidence this is drivable' and 'this is drivable' are different
    statements. A planner that confuses them routes through the blind spot
    beside the vehicle."""
    cam = _cam()
    nothing_drivable = np.zeros((288, 512), dtype=bool)
    grid = free_space_to_grid(nothing_drivable, cam, BevSpec(forward=20, lateral=8))
    occ = grid.occupied_mask()
    # Everything is occupied *except* the near-field corridor the vehicle is
    # standing on, which is bridged deliberately -- see
    # test_the_near_field_blind_spot_is_bridged. That exception is small: a
    # corridor about a vehicle wide out to the nearest visible ground.
    assert occ.mean() > 0.92
    # The corridor runs from the rear of the grid to the nearest visible
    # ground: the vehicle drove through the ground behind it, so that is
    # evidence too.
    spec = BevSpec(forward=20, lateral=8)
    expected = 2 * spec.near_field_halfwidth * (min_visible_range(cam, 288) + spec.behind)
    free_area = (~occ).sum() * grid.resolution ** 2
    assert free_area < expected + 1.0

    all_drivable = np.ones((288, 512), dtype=bool)
    grid = free_space_to_grid(all_drivable, cam, BevSpec(forward=20, lateral=8))
    occ = grid.occupied_mask()
    assert not occ.all()                          # the frustum is free
    assert occ.any()                              # outside it is not


def test_the_grid_is_ego_centred_with_room_behind():
    """The ego's own footprint has to exist in the grid or the planner starts
    inside an obstacle."""
    cam = _cam()
    grid = free_space_to_grid(np.ones((288, 512), dtype=bool), cam,
                              BevSpec(forward=24, behind=4, lateral=10))
    assert grid.origin[0] == pytest.approx(-4.0, abs=0.3)
    assert grid.origin[1] == pytest.approx(-10.0, abs=0.3)


def test_the_near_field_blind_spot_is_bridged():
    """A forward camera cannot see the road under its own bumper.

    Unseen ground is occupied, so without a bridge the ego starts inside an
    obstacle and *every* plan fails -- which is exactly what the first run of
    the demo did: every frame came back partial with no path drawn. A still
    image has no previous frame to carry forward, so the corridor the vehicle
    is standing on is taken as drivable.
    """
    cam = _cam()
    near = min_visible_range(cam, 288)
    assert 1.0 < near < 12.0                     # the wedge is real and bounded

    grid = free_space_to_grid(np.zeros((288, 512), dtype=bool), cam,
                              BevSpec(forward=20, behind=3, lateral=8))
    occ = grid.occupied_mask()

    # directly ahead, inside the blind spot: bridged
    iy, ix = grid.world_to_cell(0.5 * near, 0.0)[::-1]
    assert not occ[iy, ix]
    # the same distance but well off to the side: still occupied, because the
    # bridge is a corridor and not a blanket clearing
    iy, ix = grid.world_to_cell(0.5 * near, 6.0)[::-1]
    assert occ[iy, ix]


def test_the_bridge_does_not_reach_past_what_the_camera_sees():
    """Beyond the nearest visible ground the evidence is real, so the bridge
    must stop -- otherwise it would paint free space over a detected obstacle."""
    cam = _cam()
    near = min_visible_range(cam, 288)
    grid = free_space_to_grid(np.zeros((288, 512), dtype=bool), cam,
                              BevSpec(forward=20, behind=3, lateral=8))
    iy, ix = grid.world_to_cell(near + 4.0, 0.0)[::-1]
    assert grid.occupied_mask()[iy, ix]


def _grid_with_corridor(gap_m: float):
    """A grid whose only route forward is a corridor ``gap_m`` wide."""
    cam = Camera.from_fov(512, 288, fov_deg=90.0, height=1.35, pitch=math.radians(5))
    free = np.zeros((288, 512), dtype=bool)
    free[150:, :] = True                       # ground ahead is drivable
    grid = free_space_to_grid(free, cam, BevSpec(forward=25, behind=3, lateral=10))
    occ = grid.occupied_mask()
    # wall across the corridor at ~12 m, with a gap of the given width
    ix, _iy = grid.world_to_cell(12.0, 0.0)
    occ[:, int(ix):int(ix) + 2] = True
    half = max(int(round(0.5 * gap_m / grid.resolution)), 1)
    _jx, iy = grid.world_to_cell(12.0, 0.0)
    occ[int(iy) - half:int(iy) + half, int(ix):int(ix) + 2] = False
    grid.data = occ.astype(np.float32)
    grid.invalidate_cache()
    return grid


def test_a_goal_is_only_offered_if_a_vehicle_could_reach_it():
    """Cells connect through a one-cell gap; a 3.9 x 1.7 m vehicle does not.

    Flood-filling the raw grid offered goals that were formally connected and
    physically unreachable -- on two IDD frames the planner got 2 m from a
    start whose goal the fill had blessed. The fill runs on configuration
    space instead.
    """
    wide = reachable_goal(_grid_with_corridor(4.0), 25.0, clearance=1.2)
    narrow = reachable_goal(_grid_with_corridor(0.5), 25.0, clearance=1.2)

    assert wide[0] > 12.0            # a 4 m gap is passable, so go past it
    assert narrow[0] <= 12.5         # a 0.5 m gap is not; stop short of it


def test_a_goal_is_returned_even_when_the_ego_does_not_fit_where_it_stands():
    """Normal in a narrow frame, and not a reason to return nothing -- the
    caller needs *some* goal or the demo has no query to make."""
    cam = _cam()
    grid = free_space_to_grid(np.zeros((288, 512), dtype=bool), cam,
                              BevSpec(forward=20, behind=3, lateral=8))
    goal = reachable_goal(grid, 20.0, clearance=1.2)
    assert np.isfinite(goal).all() and goal[0] > 0.0
