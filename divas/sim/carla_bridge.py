"""CARLA bridge -- implements :class:`divas.sim.interface.SimWorld`.

Open decision #1 resolved to CARLA, so this replaces ``divas.sim.world`` as the
simulator without anything above stage 3 changing. That was the whole point of
ADR-005 and of the stage contracts.

**Hardware.** CARLA is Unreal Engine and wants a dedicated GPU with 6 GB or
more of VRAM. It will not run usefully on integrated graphics -- expect it to
refuse to start or to render at a couple of frames per second, which makes a
20 Hz control loop meaningless. Everything in this module is written to run on
such a machine; nothing in it needs one to be *reviewed* or unit tested, which
is why the CARLA calls are kept apart from the logic below.

**Version.** Written against the CARLA 0.9.16 Python API -- the last Unreal
Engine 4 release, so it keeps the 6 GB VRAM floor that 0.10/1.0 raise to 8 GB+,
*and* it is the first with a cp312 wheel. 0.9.15 ships clients for Python 2.7
and 3.7 only, which on a Python 3.12 system means either a second interpreter
for the client or no client at all.

The client package is version-locked to the server: install the ``carla`` wheel
matching the simulator build you run, not whatever pip resolves to.

Layout of this file, deliberately:

* pure functions at the top -- rasterisation, control conversion, class
  mapping. No CARLA import reaches them, and the tests exercise them directly.
* :class:`CarlaWorld` at the bottom -- the part that talks to the simulator.

Untested bridge code is where projects like this die: the perception work
lands, the bridge has a sign error in the steering conversion, and two days go
into debugging the planner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from divas.sim.geometry import (
    first_hit as _first_hit,
    footprint_discs as _footprint_discs,
    min_clearance as _min_clearance,
    range_noise_sigmas as _range_noise_sigmas,
    time_to_collision as _ttc,
)
from divas.types import (
    CLASS_EXTENT,
    EgoState,
    OccupancyGrid,
    Track,
    VehicleParams,
)

try:                                            # pragma: no cover
    import carla                                # type: ignore
    HAVE_CARLA = True
except ImportError:                             # pragma: no cover
    carla = None
    HAVE_CARLA = False


# --------------------------------------------------------------------------
# pure logic -- no CARLA needed, and unit tested
# --------------------------------------------------------------------------

#: CARLA blueprint id fragments mapped to the classes this stack reasons about.
#: Order matters: the first fragment found in the blueprint id wins, so the
#: specific two-wheelers must be tested before the generic vehicle fallback.
BLUEPRINT_CLASS: Sequence[Tuple[str, str]] = (
    ("walker", "pedestrian"),
    ("harley", "motorcycle"),
    ("yamaha", "motorcycle"),
    ("kawasaki", "motorcycle"),
    ("vespa", "motorcycle"),
    ("gazelle", "bicycle"),
    ("diamondback", "bicycle"),
    ("crossbike", "bicycle"),
    ("carlamotors", "truck"),
    ("firetruck", "truck"),
    ("ambulance", "truck"),
    ("sprinter", "truck"),
    ("volkswagen.t2", "truck"),
    ("fusorosa", "bus"),
    ("bus", "bus"),
    ("truck", "truck"),
    ("vehicle", "car"),
)


def classify_blueprint(blueprint_id: str) -> str:
    """CARLA blueprint id -> one of our classes.

    CARLA has no auto-rickshaw, which is the single most consequential gap
    between its actor set and an Indian road. Until a custom asset exists,
    two-wheelers and small cars stand in for it, and the deck should not claim
    otherwise.
    """
    bid = blueprint_id.lower()
    for fragment, cls in BLUEPRINT_CLASS:
        if fragment in bid:
            return cls
    return "unknown"


def carla_to_odom(x: float, y: float, yaw_deg: float) -> Tuple[float, float, float]:
    """CARLA world coordinates -> this stack's odom frame.

    CARLA is **left-handed**: x forward, **y to the right**, z up, with yaw in
    degrees increasing clockwise seen from above. This stack is right-handed
    with y to the left and theta counter-clockwise in radians.

    So the conversion is a y-flip and a yaw negation. It is three lines and it
    is the single most likely place for this bridge to be silently wrong: get
    it backwards and everything still runs, the vehicle just steers the wrong
    way into obstacles that appear mirrored. Hence a dedicated function with a
    dedicated test rather than an inline expression at each call site.
    """
    return x, -y, -math.radians(yaw_deg)


def odom_to_carla(x: float, y: float, theta: float) -> Tuple[float, float, float]:
    """Inverse of :func:`carla_to_odom`. Returns ``(x, y, yaw_deg)``."""
    return x, -y, -math.degrees(theta)


def carla_vector_to_odom(vx: float, vy: float) -> Tuple[float, float]:
    """Velocity or any free vector, same handedness flip."""
    return vx, -vy


def steer_from_control(steer_norm: float, max_steer_deg: float) -> float:
    """Inverse of the steering half of :func:`control_from_command`.

    The MPC seeds each rollout with the road-wheel angle the vehicle is
    currently holding, so that its steering *rate* limit means something. That
    angle has to come back out of CARLA's normalised command in our sign
    convention, not CARLA's.
    """
    return float(-steer_norm * math.radians(max(max_steer_deg, 1e-3)))


def normalised_steer(steer_rad: float, max_steer_deg: float) -> float:
    """Our road-wheel angle -> CARLA's normalised ``VehicleControl.steer``.

    Split out of :func:`control_from_command` so the sign convention lives in
    exactly one place: :class:`LongitudinalTracker` owns the pedals now, but
    the steering half of the conversion is unchanged and still the most
    dangerous line in this module.  See :func:`control_from_command`.
    """
    max_steer_rad = math.radians(max(max_steer_deg, 1e-3))
    return float(np.clip(-steer_rad / max_steer_rad, -1.0, 1.0))


def control_from_command(
    accel: float,
    steer_rad: float,
    params: VehicleParams,
    max_steer_deg: float,
) -> Tuple[float, float, float]:
    """``(accel, steer)`` from stage 6 -> CARLA ``(throttle, steer, brake)``.

    CARLA takes a normalised steering command, not a road-wheel angle, and the
    normalisation constant is a property of the spawned vehicle -- read it from
    ``get_physics_control().wheels[0].max_steer_angle`` rather than assuming,
    because it differs by several degrees between blueprints and a wrong
    constant looks exactly like a badly tuned controller.

    **The sign flips.** ``VehicleControl.steer`` is positive to the *right* --
    CARLA is left-handed, and its own ``manual_control.py`` decrements the
    steer cache for the left-arrow key. Our road-wheel angle is positive to the
    *left*, counter-clockwise, because that is what the bicycle model in
    :mod:`divas.types` integrates. Passing our angle through unnegated is the
    mirrored-world bug this module's docstring warns about: the vehicle drives,
    tracks nothing, and steers into every obstacle it was avoiding.

    **The longitudinal half is open-loop and is no longer what drives the
    ego.** Throttle and brake here are simply proportional to the requested
    acceleration, which ignores drag entirely: a zero acceleration command
    gives zero throttle, so the vehicle coasts down instead of holding speed.
    With no integral action it settles wherever the proportional command
    balances drag -- a standing error of ``resistance(v) / (1.1 * authority)``,
    a few tenths of a metre per second, and larger wherever the pedal
    authority it assumes fails to arrive.  :class:`LongitudinalTracker`
    replaced it.  Note the size: this is a correctness defect, not an
    explanation for the 4.56 m/s *mean* of the first live Town10 run, which is
    mostly traffic and a curvature-limited reference.

    Kept because it is the honest baseline for the comparison in
    ``scripts/calibrate_longitudinal.py``, and because it is the one mapping
    with no state, which makes it the right thing to reach for in a test that
    is about steering.
    """
    steer = normalised_steer(steer_rad, max_steer_deg)
    if accel >= 0.0:
        throttle = float(np.clip(accel / max(params.max_accel, 1e-3), 0.0, 1.0))
        brake = 0.0
    else:
        throttle = 0.0
        brake = float(np.clip(-accel / max(abs(params.min_accel), 1e-3), 0.0, 1.0))
    return throttle, steer, brake


@dataclass(frozen=True)
class ResistanceModel:
    """Coast-down deceleration of the ego, m/s^2, as ``c0 + c1 v + c2 v^2``.

    Nominally rolling resistance, speed-proportional driveline loss and
    aerodynamic drag.  In CARLA it is dominated by none of those: the default
    vehicle physics carry
    ``damping_rate_zero_throttle_clutch_engaged = 2.0`` and an autobox that
    downshifts hard on lift-off, so a Micra coasting from 9 m/s loses speed at
    **about 4 m/s^2** -- an order of magnitude more than a real car, and
    comparable to braking.  The quadratic is therefore an empirical fit to
    CARLA's engine braking, not a drag model, which is why the coefficients
    here are measured by :mod:`scripts.calibrate_longitudinal` rather than
    taken from a textbook.  Do not reason about them as physical drag; a real
    vehicle would need refitting, not rescaling.

    Because engine braking is gear-dependent and the downshift points are
    hysteretic, the samples are genuinely noisy and the fit residual is a
    few tenths of a m/s^2.  That error is what the integral trim absorbs.

    Why this exists at all: the 2-D bicycle model in :mod:`divas.sim.world`
    integrates ``v += a dt`` with no losses, so a commanded acceleration of
    zero holds speed exactly.  CARLA is a real vehicle dynamics model, where
    zero throttle means coasting down.  A controller tuned against the former
    and pedalled open-loop into the latter can never hold its reference; it
    settles a standing error below it.
    """

    # Measured on vehicle.nissan.micra in CARLA 0.9.16, Town10HD_Opt, by
    # ``scripts/calibrate_longitudinal.py``: six coast-down segments from 3 to
    # 13 m/s, RMS residual 0.77 m/s^2.  The quadratic term fitted negative and
    # was rejected by the curve check, so this is the validated linear fit --
    # 4.08 m/s^2 of coast-down at 9 m/s.
    c0: float = 2.3915      # m/s^2
    c1: float = 0.18802     # m/s^2 per m/s
    c2: float = 0.0         # m/s^2 per (m/s)^2 -- not supported by the data

    def __call__(self, v: float) -> float:
        vv = max(float(v), 0.0)
        return float(self.c0 + self.c1 * vv + self.c2 * vv * vv)


class LongitudinalTracker:
    """Commanded acceleration -> ``(throttle, brake)``, closing the loop.

    Stage 6 emits an *acceleration*, and CARLA takes *pedal positions*.  The
    original mapping was proportional and open-loop -- ``throttle =
    accel / max_accel`` -- which has one structural property: the MPC's
    command tends to zero as the ego approaches its speed reference, so the
    throttle also tends to zero, and the vehicle coasts down against drag
    until the residual error is large enough to ask for acceleration again.
    It settles wherever those two effects balance, always below the reference,
    and the symptom reads as a badly tuned planner rather than a missing pedal
    model.  It also assumes the full ``max_accel`` is available at any speed,
    which no engine curve delivers, and it has no way to notice when the
    acceleration it asked for did not arrive.

    Two pieces fix it:

    * **Feedforward.** The pedal is asked for ``a_cmd + resistance(v)``, so a
      zero acceleration command holds speed instead of decaying.  This is the
      part that does the work; it is a model, and it is measured.
    * **Measured pedal gains.** ``throttle_gain`` and ``brake_gain`` are the
      accelerations one unit of pedal actually delivers, m/s^2.  The original
      mapping normalised by ``params.max_accel`` and ``params.min_accel``,
      which are the *planner's comfort limits* -- 2.0 and -4.0 m/s^2 -- not
      anything the vehicle is capable of.  Conflating a comfort bound with an
      actuator limit means the pedal is wrong by whatever ratio separates
      them, and on this ego it is more than double.  Both are measured by
      ``scripts/calibrate_longitudinal.py``.
    * **Integral trim** on the *acceleration* tracking error.  The feedforward
      is a three-parameter fit to a vehicle with an engine curve, a gearbox
      and load transfer, so it is wrong everywhere by a little and wrong on a
      gradient by a lot.  This is the term that makes the loop indifferent to
      how much pedal authority actually arrived.  It absorbs that without
      needing a speed reference the bridge does not have -- stage 6 sends an
      acceleration, not a target speed, so this loop is deliberately built to
      track acceleration and nothing else.

    Anti-windup is not optional here.  Braking saturates constantly in traffic
    and an unclamped integrator would spend the following seconds unwinding a
    term it accumulated while the pedal was already on the floor.

    The tracker is stateful, so :meth:`reset` runs at the start of every
    episode; carrying a trim across a teleport would corrupt the first second
    of the next run's speed trace, and the metrics average over exactly that.
    """

    def __init__(
        self,
        params: VehicleParams,
        resistance: Optional[ResistanceModel] = None,
        ki: float = 0.60,
        i_limit: float = 3.0,
        coast_band: float = 0.15,
        stop_speed: float = 0.2,
        throttle_gain: float = 6.47,
        brake_gain: float = 3.44,
    ) -> None:
        self.params = params
        self.resistance = resistance if resistance is not None else ResistanceModel()
        self.ki = float(ki)
        self.i_limit = float(i_limit)
        self.coast_band = float(coast_band)
        self.stop_speed = float(stop_speed)
        self.throttle_gain = float(max(throttle_gain, 1e-3))
        self.brake_gain = float(max(brake_gain, 1e-3))
        self._i = 0.0

    @property
    def trim(self) -> float:
        """Current integral term, m/s^2. Exposed for logging, not for control.

        Worth watching: a trim that sits at its clamp means the feedforward is
        wrong by more than the loop can absorb, and the fit needs redoing --
        the pedal is compensating for a model error rather than a gradient.
        """
        return self._i

    def reset(self) -> None:
        self._i = 0.0

    def update(self, a_cmd: float, v: float, a_meas: float, dt: float) -> Tuple[float, float]:
        """One control step. Returns ``(throttle, brake)``, both in ``[0, 1]``.

        ``a_meas`` is the acceleration the vehicle actually achieved over the
        previous step, which the caller already computes for the MPC's rate
        limits -- the loop is closed on that, not on a re-differentiated speed.
        """
        p = self.params
        v = max(float(v), 0.0)
        a_cmd = float(np.clip(a_cmd, p.min_accel, p.max_accel))

        # Standstill hold.  ``resistance(0)`` is 2.4 m/s^2 on this ego, so the
        # feedforward asks for real throttle even at rest, and a vehicle told
        # to hold a stop would creep off the line without this branch -- and a
        # held brake is also what keeps it there on a gradient.
        if v < self.stop_speed and a_cmd <= 0.0:
            self._i = 0.0
            return 0.0, 1.0

        a_req = a_cmd + self.resistance(v) + self._i

        if a_req >= 0.0:
            raw = a_req / self.throttle_gain
            throttle, brake = float(np.clip(raw, 0.0, 1.0)), 0.0
            saturated = raw > 1.0
        elif a_req > -self.coast_band:
            # A deceleration this small is what coasting already delivers.
            # Pedalling it would chatter between throttle and brake every
            # step at cruise, which is both unphysical and uncomfortable.
            throttle, brake, saturated = 0.0, 0.0, False
        else:
            raw = -a_req / self.brake_gain
            throttle, brake = 0.0, float(np.clip(raw, 0.0, 1.0))
            saturated = raw > 1.0

        # Integrate last, and only when the pedal has room to act on it.
        err = a_cmd - float(a_meas)
        if not (saturated and err * a_req > 0.0):
            self._i = float(np.clip(
                self._i + self.ki * err * max(float(dt), 0.0),
                -self.i_limit, self.i_limit,
            ))
        return throttle, brake


@dataclass
class DrivableRaster:
    """A whole town's drivable area, rasterised once.

    Querying ``Map.get_waypoint`` per cell per tick is hopeless -- a 64 x 64 m
    window at 0.25 m is 65k queries every 100 ms. The road network does not
    move, so it is rasterised once at connect time and the rolling window is
    then a crop.
    """

    origin: Tuple[float, float]
    resolution: float
    mask: np.ndarray                    # (ny, nx) bool, True = drivable

    @staticmethod
    def from_points(
        points: np.ndarray,
        widths: np.ndarray,
        resolution: float = 0.25,
        pad: float = 12.0,
    ) -> "DrivableRaster":
        """Rasterise from waypoint centres and their lane widths.

        ``points`` is (N, 2) in world coordinates; ``widths`` is (N,) full lane
        widths. Each waypoint paints a disc of radius ``width / 2``, which
        scallops the edges slightly at coarse sampling and is why the caller
        should generate waypoints at half the resolution or finer.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        widths = np.asarray(widths, dtype=np.float64).reshape(-1)
        if points.size == 0:
            raise ValueError("no waypoints: is the map loaded?")
        x0 = float(points[:, 0].min() - pad)
        y0 = float(points[:, 1].min() - pad)
        nx = int(np.ceil((points[:, 0].max() + pad - x0) / resolution))
        ny = int(np.ceil((points[:, 1].max() + pad - y0) / resolution))
        mask = np.zeros((ny, nx), dtype=bool)

        rad_cells = np.maximum((widths / 2.0) / resolution, 1.0)
        ix = ((points[:, 0] - x0) / resolution).astype(np.int32)
        iy = ((points[:, 1] - y0) / resolution).astype(np.int32)
        # Painted with a stamp per distinct radius rather than per waypoint:
        # a town is ~20k waypoints and a Python loop over them is seconds.
        for r in np.unique(np.round(rad_cells).astype(np.int32)):
            sel = np.round(rad_cells).astype(np.int32) == r
            if not sel.any():
                continue
            k = int(r)
            dy, dx = np.mgrid[-k:k + 1, -k:k + 1]
            stamp = (dx * dx + dy * dy) <= k * k
            oy, ox = np.nonzero(stamp)
            oy -= k
            ox -= k
            for cx, cy in zip(ix[sel], iy[sel]):
                yy = cy + oy
                xx = cx + ox
                ok = (yy >= 0) & (yy < ny) & (xx >= 0) & (xx < nx)
                mask[yy[ok], xx[ok]] = True
        return DrivableRaster((x0, y0), resolution, mask)

    def carve(self, boxes: Sequence[Tuple[float, float, float, float, float]]) -> int:
        """Clear oriented rectangles out of the drivable mask, in place.

        ``boxes`` are ``(x, y, theta, half_length, half_width)`` in the odom
        frame.  Returns how many cells were cleared, which is worth checking:
        zero means the boxes missed the raster entirely, and the usual reason
        for that is a frame conversion, not an empty town.

        This is what makes "free space" mean it.  The mask is painted from the
        lane graph, and the lane graph does not know that a lane has a parked
        car standing in it -- Town10HD carries 48 parked vehicles baked into
        the map as static meshes rather than actors, so nothing spawns them and
        ``ground_truth_tracks`` cannot report them.  Left alone they are marked
        *drivable*, and the only thing that would notice a planner routing
        through one is the collision sensor.

        **Measured effect on Town10HD: almost none, and say so.**  Of the 32
        that fall inside the rasterised area, *zero* have their centre on a
        driving lane -- the median sits 3.38 m clear of one, because this is a
        US-style town that parks in bays.  One cell gets carved, from a corner
        overlap.  The mechanism is kept because it is the correct definition of
        free space and because the maps this project is actually about do not
        park in bays: a lorry stopped in the carriageway is the normal case on
        an Indian road, and it is the case a lane-graph raster gets silently
        wrong.  Do not present this as a Town10 result.
        """
        ny, nx = self.mask.shape
        cleared = 0
        for x, y, theta, hl, hw in boxes:
            reach = math.hypot(hl, hw) + self.resolution
            x0 = int(max(math.floor((x - reach - self.origin[0]) / self.resolution), 0))
            x1 = int(min(math.ceil((x + reach - self.origin[0]) / self.resolution), nx))
            y0 = int(max(math.floor((y - reach - self.origin[1]) / self.resolution), 0))
            y1 = int(min(math.ceil((y + reach - self.origin[1]) / self.resolution), ny))
            if x1 <= x0 or y1 <= y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            wx = self.origin[0] + (gx + 0.5) * self.resolution
            wy = self.origin[1] + (gy + 0.5) * self.resolution
            c, s = math.cos(-theta), math.sin(-theta)
            lx = (wx - x) * c - (wy - y) * s
            ly = (wx - x) * s + (wy - y) * c
            inside = (np.abs(lx) <= hl) & (np.abs(ly) <= hw)
            sub = self.mask[y0:y1, x0:x1]
            cleared += int((sub & inside).sum())
            sub[inside] = False
        return cleared

    def contains(self, x, y) -> np.ndarray:
        """Is each world point on the drivable area?  Off the raster is not.

        The off-road check samples the ego's footprint disc *centres*, exactly
        as the built-in world samples its road polygon, so "off_road" means the
        same thing in both simulators.
        """
        sx = np.floor((np.asarray(x, dtype=np.float64) - self.origin[0])
                      / self.resolution).astype(np.int32)
        sy = np.floor((np.asarray(y, dtype=np.float64) - self.origin[1])
                      / self.resolution).astype(np.int32)
        inb = ((sx >= 0) & (sx < self.mask.shape[1])
               & (sy >= 0) & (sy < self.mask.shape[0]))
        sx = np.clip(sx, 0, self.mask.shape[1] - 1)
        sy = np.clip(sy, 0, self.mask.shape[0] - 1)
        return np.where(inb, self.mask[sy, sx], False)

    def window(
        self, cx: float, cy: float, half_extent: float, resolution: float
    ) -> OccupancyGrid:
        """Crop an ego-centred, world-aligned rolling window (ADR-001).

        Anything outside the rasterised area is occupied, which is correct:
        off the map is not drivable.
        """
        grid = OccupancyGrid.empty(
            (cx - half_extent, cx + half_extent),
            (cy - half_extent, cy + half_extent),
            resolution,
        )
        gx, gy = np.meshgrid(np.arange(grid.nx), np.arange(grid.ny))
        wx, wy = grid.cell_to_world(gx, gy)
        sx = ((wx - self.origin[0]) / self.resolution).astype(np.int32)
        sy = ((wy - self.origin[1]) / self.resolution).astype(np.int32)
        inb = ((sx >= 0) & (sx < self.mask.shape[1])
               & (sy >= 0) & (sy < self.mask.shape[0]))
        np.clip(sx, 0, self.mask.shape[1] - 1, out=sx)
        np.clip(sy, 0, self.mask.shape[0] - 1, out=sy)
        drivable = np.where(inb, self.mask[sy, sx], False)
        grid.data = (~drivable).astype(np.float32)
        grid.invalidate_cache()
        return grid


def stamp_actors(
    grid: OccupancyGrid,
    boxes: Sequence[Tuple[float, float, float, float, float]],
) -> OccupancyGrid:
    """Paint oriented actor footprints ``(x, y, yaw, half_len, half_wid)``."""
    out = grid.copy()
    if not boxes:
        return out
    gx, gy = np.meshgrid(np.arange(out.nx), np.arange(out.ny))
    wx, wy = out.cell_to_world(gx, gy)
    data = out.data
    for x, y, yaw, hl, hw in boxes:
        c, s = math.cos(-yaw), math.sin(-yaw)
        dx, dy = wx - x, wy - y
        lx = c * dx - s * dy
        ly = s * dx + c * dy
        data[(np.abs(lx) <= hl) & (np.abs(ly) <= hw)] = 1.0
    out.data = data
    out.invalidate_cache()
    return out


@dataclass
class Route:
    """A polyline route with arc-length progress. Satisfies ``RouteSource``.

    **On ``windowed``.** By default the nearest route point is found by
    searching the whole polyline, which is correct only while the route does
    not come near itself. Town10HD is about 400 x 230 m, so any route longer
    than roughly a kilometre has to revisit streets -- and then a global search
    is actively wrong: as the ego comes back past an earlier stretch, the
    nearest point jumps backwards, ``progress`` collapses, and ``point_ahead``
    hands the planner a goal on the wrong branch of the route. That is
    ADR-006's failure mode arriving by a different door: the goal leaves the
    costmap, every MPC rollout runs off the end of the path, and the vehicle
    brakes for no visible reason.

    ``windowed=True`` searches only a band of route around a cursor that
    advances with the vehicle, which makes progress monotonic and keeps the
    lookahead on the branch actually being driven. It is opt-in rather than
    the default because every published number was produced with the global
    search on a non-looping route, and a silent change to how progress is
    measured would quietly move all of them.
    """

    points: np.ndarray                       # (N, 2)
    windowed: bool = False
    #: How far ahead of the cursor the forward walk may look, metres. It has
    #: to exceed the goal lookahead, or ``point_ahead`` runs off the end of
    #: the searched span and the goal stops advancing. There is deliberately
    #: no *backward* window; see :meth:`_nearest`.
    forward_window: float = 90.0
    #: Consecutive route points that may get *further* away before the forward
    #: walk gives up. A polyline is not smooth, so a corner can worsen for a
    #: couple of points before improving; at the 2 m route spacing this is
    #: about 10 m of tolerance, which is far short of the distance to a
    #: parallel carriageway further along the route.
    patience: int = 5

    def __post_init__(self) -> None:
        self.points = np.asarray(self.points, dtype=np.float64).reshape(-1, 2)
        seg = np.linalg.norm(np.diff(self.points, axis=0), axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(seg)])
        self._cursor = 0

    def reset(self) -> None:
        """Send the cursor back to the start. Only meaningful when windowed."""
        self._cursor = 0

    def _nearest(self, x: float, y: float) -> int:
        if not self.windowed:
            return int(np.argmin(np.linalg.norm(self.points - np.array([x, y]),
                                                axis=1)))
        # Walk forward to the *first local minimum* of distance, rather than
        # taking the argmin over the whole band.
        #
        # An argmin over a band is not enough, and the failure is not subtle.
        # The band spans arc length, but "near" is measured in space, and on a
        # two-way street the opposite carriageway is 3.5 m away spatially
        # while being tens of metres away along the route. The argmin then
        # jumps to the return leg and the cursor teleports. Measured: a 25 s
        # run reported 1808 m of progress, or 72 m/s.
        #
        # Following the route continuously cannot do that -- it stops at the
        # first minimum, long before reaching a leg that only *looks* close.
        # The walk starts *at* the cursor, never behind it. Allowing it to
        # start 20 m back looks harmless and breaks the same two-way street:
        # from there the walk re-finds the outbound carriageway -- which is
        # genuinely the nearest local minimum, 3.5 m away -- and the cursor
        # never follows the route onto the return leg. Measured: an
        # out-and-back route stalled at 67 m of 123 m.
        pts, cum = self.points, self._cum
        target = np.array([x, y], dtype=np.float64)
        lo = self._cursor
        limit = cum[lo] + self.forward_window

        best_i = lo
        best_d = float(np.linalg.norm(pts[lo] - target))
        i, worsening = lo, 0
        while i + 1 < len(pts) and cum[i] <= limit:
            i += 1
            d = float(np.linalg.norm(pts[i] - target))
            if d < best_d:
                best_d, best_i, worsening = d, i, 0
            else:
                # Patience, because a polyline is not smooth: a couple of
                # points can worsen across a corner before improving again.
                worsening += 1
                if worsening >= self.patience:
                    break
        self._cursor = max(self._cursor, best_i)
        return best_i

    def progress(self, x: float, y: float) -> float:
        """Arc length travelled along the route, metres.

        When windowed this ratchets: it reports the cursor, which never
        retreats, rather than the instantaneous nearest point. The nearest
        point may legitimately sit up to ``back_window`` behind -- the ego
        swings wide of the line to pass a parked lorry and the closest route
        point is briefly one it already went by -- and reporting that as lost
        progress would show up as a 20 m backward jump in the metric, and as a
        stall to the runner's stuck detector. The vehicle cannot reverse
        (``v`` is clamped at zero in both simulators), so a ratchet cannot hide
        real backward motion.
        """
        i = self._nearest(x, y)
        return float(self._cum[self._cursor if self.windowed else i])

    def point_ahead(self, x: float, y: float, distance: float) -> np.ndarray:
        """Route point ``distance`` ahead. Uses the *nearest* point, not the
        ratcheted cursor: the lookahead has to be measured from where the
        vehicle actually is, or it grows by however far the ratchet has run
        ahead and the goal drifts off the costmap."""
        i = self._nearest(x, y)
        j = int(np.searchsorted(self._cum, self._cum[i] + distance))
        return self.points[min(j, len(self.points) - 1)].copy()

    @property
    def length(self) -> float:
        return float(self._cum[-1])


# Weather presets.  The rain and fog entries are not decoration: the deck's own
# challenge table names "fusion accuracy in rain/dust/glare" as a risk, and
# these are how that claim gets tested rather than asserted.
WEATHER_PRESETS: Dict[str, dict] = {
    "clear_noon": dict(cloudiness=5, precipitation=0, precipitation_deposits=0,
                       sun_altitude_angle=70, fog_density=0, wetness=0),
    "clear_sunset": dict(cloudiness=10, precipitation=0, precipitation_deposits=0,
                         sun_altitude_angle=8, fog_density=0, wetness=0),
    "hard_rain": dict(cloudiness=90, precipitation=80, precipitation_deposits=70,
                      sun_altitude_angle=30, fog_density=10, wetness=80),
    "wet_dusk": dict(cloudiness=70, precipitation=0, precipitation_deposits=60,
                     sun_altitude_angle=3, fog_density=5, wetness=70),
    "fog": dict(cloudiness=60, precipitation=0, precipitation_deposits=0,
                sun_altitude_angle=40, fog_density=65, fog_distance=8, wetness=10),
    "dust_haze": dict(cloudiness=35, precipitation=0, precipitation_deposits=0,
                      sun_altitude_angle=20, fog_density=25, fog_distance=25,
                      wetness=0),
    "night": dict(cloudiness=30, precipitation=0, precipitation_deposits=0,
                  sun_altitude_angle=-12, fog_density=8, wetness=15),
}


# --------------------------------------------------------------------------
# sensors -- still pure: specs are data, decoders take buffers
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorSpec:
    """One sensor to attach to the ego.

    Kept as data rather than as code inside :class:`CarlaWorld` so Phase 2 can
    add a second camera or move the LiDAR without touching the world at all.
    ``xyz`` and ``pitch`` are in CARLA's own frame, because that is the frame
    the spawn call wants and converting twice is how sign errors are born.
    """

    name: str
    blueprint: str
    xyz: Tuple[float, float, float] = (1.5, 0.0, 1.7)
    pitch: float = 0.0
    attributes: Dict[str, str] = field(default_factory=dict)


def default_sensor_rig(fixed_delta_seconds: float = 0.05) -> Dict[str, SensorSpec]:
    """The Phase 2 rig.

    The LiDAR's ``rotation_frequency`` must equal the tick rate or every frame
    is a partial sweep: CARLA spins the laser in real time and hands over
    whatever it swept since the last tick, so a 10 Hz LiDAR ticked at 20 Hz
    delivers half a revolution per frame and the point cloud looks like a fan.
    """
    hz = 1.0 / max(fixed_delta_seconds, 1e-3)
    return {
        s.name: s
        for s in (
            SensorSpec("rgb", "sensor.camera.rgb",
                       attributes={"image_size_x": "960", "image_size_y": "540",
                                   "fov": "90"}),
            # Third person, for video capture only -- nothing upstream of
            # stage 1 may read it.  Behind and above the roof so the ego's own
            # body gives the viewer a sense of scale and of what it cleared,
            # which a bonnet camera cannot show.
            SensorSpec("chase", "sensor.camera.rgb", xyz=(-6.5, 0.0, 3.4),
                       pitch=-13.0,
                       attributes={"image_size_x": "1280", "image_size_y": "720",
                                   "fov": "95"}),
            # Ground truth for the drivable-area head.  This is the reason
            # CARLA makes Phase 2 tractable: free per-pixel road labels, in
            # the same frame as the RGB the model actually consumes.
            SensorSpec("semantic", "sensor.camera.semantic_segmentation",
                       attributes={"image_size_x": "960", "image_size_y": "540",
                                   "fov": "90"}),
            SensorSpec("depth", "sensor.camera.depth",
                       attributes={"image_size_x": "960", "image_size_y": "540",
                                   "fov": "90"}),
            SensorSpec("lidar", "sensor.lidar.ray_cast", xyz=(0.0, 0.0, 2.4),
                       attributes={"channels": "32", "range": "60",
                                   "points_per_second": "600000",
                                   "rotation_frequency": f"{hz:.1f}",
                                   "upper_fov": "10", "lower_fov": "-30"}),
            SensorSpec("radar", "sensor.other.radar", xyz=(2.0, 0.0, 1.0),
                       pitch=5.0,
                       attributes={"horizontal_fov": "35", "vertical_fov": "12",
                                   "range": "70", "points_per_second": "3000"}),
        )
    }


#: CARLA's semantic tags, in the red channel of the segmentation image.
#: 1 = Roads, 24 = RoadLines, 25 = Ground (unpaved).
DRIVABLE_TAGS: Tuple[int, ...] = (1, 24)
#: Indian carriageways routinely include the unpaved shoulder, and a stack that
#: refuses to use it will refuse the gap a rickshaw just took.  Offered as a
#: separate constant rather than folded into the default, because widening the
#: definition of "drivable" is a claim that has to be made deliberately.
DRIVABLE_TAGS_WITH_SHOULDER: Tuple[int, ...] = (1, 24, 25)


def decode_camera(raw: bytes, height: int, width: int) -> np.ndarray:
    """CARLA image buffer -> ``(H, W, 4)`` uint8, BGRA as CARLA stores it."""
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)


def semantic_tags(raw: bytes, height: int, width: int) -> np.ndarray:
    """Per-pixel semantic tag from a raw segmentation image.

    The tag is in the **red** channel of the raw buffer -- index 2 of BGRA.
    Read the wrong channel and every pixel comes back zero, which looks
    exactly like a camera that failed to attach.
    """
    return decode_camera(raw, height, width)[:, :, 2]


def semantic_drivable_mask(
    raw: bytes, height: int, width: int, tags: Sequence[int] = DRIVABLE_TAGS
) -> np.ndarray:
    """Boolean drivable-area ground truth from the segmentation camera."""
    return np.isin(semantic_tags(raw, height, width), np.asarray(tags))


def decode_lidar(raw: bytes) -> np.ndarray:
    """Ray-cast LiDAR buffer -> ``(N, 4)`` of ``(x, y, z, intensity)``.

    Still in CARLA's left-handed sensor frame; flip y with
    :func:`carla_vector_to_odom` before comparing against anything of ours.
    """
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 4).copy()


def decode_radar(raw: bytes) -> np.ndarray:
    """Radar buffer -> ``(N, 4)`` of ``(velocity, azimuth, altitude, depth)``.

    Velocity is *along the beam* and signed towards the sensor, so a
    stationary wall read while the ego closes on it returns a positive number.
    """
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 4).copy()


# --------------------------------------------------------------------------
# vehicle parameters read from the spawned blueprint
# --------------------------------------------------------------------------


def params_from_physics(
    base: VehicleParams,
    length: float,
    width: float,
    wheelbase: float,
    max_steer_deg: float,
) -> VehicleParams:
    """Fill the geometric fields of :class:`VehicleParams` from the real car.

    The defaults describe a compact Indian hatchback; the blueprint CARLA
    actually spawned may be half a metre longer, and the collision metrics are
    computed from these numbers.  Leaving the defaults in place would mean
    measuring clearance for one vehicle while driving another.

    ``rear_overhang`` assumes the axles sit symmetrically inside the body,
    which is within a few centimetres for every passenger blueprint and is not
    worth an extra RPC to do better.
    """
    overhang = max((length - wheelbase) / 2.0, 0.0)
    return replace(
        base,
        length=float(length),
        width=float(width),
        wheelbase=float(max(wheelbase, 0.5)),
        rear_overhang=float(overhang),
        max_steer=float(min(base.max_steer, math.radians(max(max_steer_deg, 1.0)))),
    )


# --------------------------------------------------------------------------
# the world
# --------------------------------------------------------------------------


@dataclass
class ActorSnapshot:
    """One traffic actor as everything above stage 3 sees it, odom frame.

    Deliberately the same attribute names as :class:`divas.sim.world.Actor`
    so the runner's trace recording, the demo renderer and the metrics work
    against either simulator without a branch.
    """

    id: int
    cls: str
    x: float
    y: float
    theta: float
    v: float
    half_length: float
    half_width: float
    alive: bool = True

    @property
    def extent(self) -> Tuple[float, float]:
        return (self.half_length, self.half_width)


@dataclass
class _Spawned:
    """What we keep about an actor we own: the handle, and its geometry.

    Geometry is read **once** at spawn.  Reading a bounding box is an RPC, and
    at 20 Hz with forty actors that is 800 round trips a second spent
    re-learning constants.
    """

    handle: object
    cls: str
    half_length: float
    half_width: float


@dataclass(frozen=True)
class ObstructionSpec:
    """An unstructured-traffic scene laid along the ego's own route.

    CARLA's traffic manager produces orderly Western traffic: vehicles keep
    their lanes, pedestrians keep to the pavement, and nothing stands in the
    carriageway. That is the opposite of the case this project exists for, and
    it is why the stock CARLA episode does not discriminate between the
    ablation arms -- there is no conflict in it.

    This places the conflict deliberately, anchored to the route rather than
    scattered across the map, so the ego actually meets it.

    **On the cattle.** CARLA 0.9.16 ships no animal blueprint -- 214
    blueprints, and the only match for "cow", "goat" or "buffalo" is
    ``static.prop.doghouse``. The herd is therefore represented by *standing
    pedestrians in the carriageway*, which is a substitution and is labelled
    as one wherever this scene is shown. Structurally it poses the planner the
    same question: a non-vehicle obstruction, stationary, occupying the lane,
    that no amount of lane-following can resolve. It is not a picture of
    cattle and must not be presented as one.
    """

    #: Standing in the carriageway, no AI controller, so they do not walk away.
    standing: int = 8
    #: Crossing from the verges, with controllers aimed at the far side.
    crossing: int = 10
    #: Stopped vehicles, partially in lane, pinching the corridor.
    blockers: int = 3
    #: Where along the route the scene begins and how far it runs, metres.
    start_m: float = 25.0
    span_m: float = 70.0
    #: Lateral spread of the standing group about the route centre, metres.
    lateral: float = 3.0


@dataclass
class CarlaConfig:
    """Everything about a CARLA session that is a choice rather than a fact."""

    host: str = "127.0.0.1"
    port: int = 2000
    timeout: float = 30.0
    town: Optional[str] = None            # None = use whatever is loaded
    # 20 Hz, matching ``RunnerConfig.sim_dt``.  Sync mode plus a fixed delta is
    # what makes a CARLA run repeatable; without it the simulator advances by
    # however long the last frame took and the seed buys you nothing.
    fixed_delta_seconds: float = 0.05
    ego_blueprint: str = "vehicle.nissan.micra"
    n_vehicles: int = 40
    n_walkers: int = 25
    walker_cross_factor: float = 0.5      # fraction willing to cross off-crossing
    weather: str = "clear_noon"
    seed: int = 0
    tm_port: int = 8000
    route_length: float = 400.0
    #: Let the route cross itself, for drives longer than the town. Implies a
    #: windowed :class:`Route`; see :func:`route_from_waypoint`.
    long_route: bool = False
    #: Start the ego where the pedestrians actually are. CARLA's walker
    #: navigation mesh covers only part of a town -- in Town10HD it is a
    #: district around ``y = +75..+82`` -- and walkers cannot be spawned or
    #: sent anywhere else, because one placed off the mesh is dead by the next
    #: tick. Rather than move the crowd to the ego, this moves the ego to the
    #: crowd, which is the only direction that works.
    spawn_in_crowd: bool = False
    #: Populate the ego's own route with an unstructured-traffic scene --
    #: standing obstructions in the carriageway and a crowd crossing it.
    #: ``None`` leaves the town as the traffic manager arranges it, which is
    #: orderly Western traffic and is not what this project is about.
    obstruction: Optional["ObstructionSpec"] = None
    #: Stop at red lights, as a keep-out in the static costmap. On by default:
    #: a stack that drives through reds in a signalled town gets T-boned in
    #: dense traffic, which is what the first long run did. Ground truth from
    #: the simulator, standing in for the perception that would report it --
    #: see :meth:`CarlaWorld.red_light_keepouts`.
    obey_traffic_lights: bool = True
    stop_line_depth: float = 1.2          # m, half-depth of the keep-out
    waypoint_spacing: float = 0.5         # for the drivable raster
    route_spacing: float = 2.0
    raster_resolution: float = 0.25
    track_range: float = 60.0             # m; beyond this we do not report actors
    track_noise: float = 1.0
    sensors: Tuple[str, ...] = ()         # names from :func:`default_sensor_rig`
    render: bool = False                  # False -> no_rendering_mode, much faster
    # Coast-down fit for the ego blueprint, feeding
    # :class:`LongitudinalTracker`.  Override per blueprint from
    # ``scripts/calibrate_longitudinal.py --write``: a Micra and a Tesla do
    # not share a drag area, and the feedforward is the term that holds
    # cruise speed.
    resistance: ResistanceModel = field(default_factory=ResistanceModel)
    throttle_ki: float = 0.60             # integral trim gain, 1/s
    # Accelerations one unit of pedal delivers, m/s^2 -- the vehicle's real
    # authority, not the planner's comfort limits (2.0 and -4.0, which are
    # comfort bounds and are both wrong by more than a factor of one and a
    # half here).  Measured on the Micra; see the script.
    throttle_gain: float = 6.47
    brake_gain: float = 3.44


class CarlaWorld:
    """A CARLA session that satisfies :class:`divas.sim.interface.SimWorld`.

    Use it as a context manager, always::

        with CarlaWorld(CarlaConfig(seed=1)) as world:
            ...

    **Leaked actors are the failure mode of this class.** CARLA keeps every
    actor a disconnected client spawned, so a run that dies without cleaning up
    leaves forty vehicles parked across the map, and the *next* run spawns into
    a town that is already full -- it does not crash, it just quietly produces
    worse numbers. Hence ``__exit__``, ``close()`` being idempotent, and the
    ``finally`` in :func:`divas.eval.runner.run`.
    """

    def __init__(self, cfg: Optional[CarlaConfig] = None,
                 params: Optional[VehicleParams] = None) -> None:
        if not HAVE_CARLA:                              # pragma: no cover
            raise RuntimeError(
                "the carla Python package is not installed. Install the wheel "
                "that ships with your simulator build -- the client is "
                "version-locked to the server, so `pip install carla` without "
                "a version is how you get a silent protocol mismatch."
            )
        self.cfg = cfg or CarlaConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.t = 0.0
        self._closed = False
        self._collisions: List[str] = []
        self._spawned: Dict[int, _Spawned] = {}
        self._sensors: Dict[str, object] = {}
        self._walker_controllers: List[object] = []
        self.frames: Dict[str, object] = {}
        self._actor_cache: Optional[List[ActorSnapshot]] = None

        self.client = carla.Client(self.cfg.host, self.cfg.port)
        self.client.set_timeout(self.cfg.timeout)
        self.world = (self.client.load_world(self.cfg.town) if self.cfg.town
                      else self.client.get_world())
        self.map = self.world.get_map()

        self._original_settings = self.world.get_settings()
        self._enter_sync_mode()
        self._set_weather(self.cfg.weather)

        self.params = params or VehicleParams()
        self._spawn_ego()
        self._build_route()
        self._build_raster()
        self._start_traffic_manager()
        self._spawn_vehicles()
        self._spawn_walkers()
        # After the route exists: the scene is anchored to it, not scattered.
        self._spawn_obstruction()
        self._attach_sensors()

        # One tick so physics settles and every sensor has delivered a frame
        # before the first control step reads them.
        self.world.tick()
        self._pump_sensors()
        self._sync_ego(self.cfg.fixed_delta_seconds)
        self.ego.v = 0.0
        self.ego.a = 0.0
        self.t = self.ego.t = 0.0

    # -- setup ------------------------------------------------------------
    def _enter_sync_mode(self) -> None:
        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = self.cfg.fixed_delta_seconds
        # ``no_rendering_mode`` turns off the whole rendering pipeline, which is
        # a large speed-up on a laptop GPU and is exactly what an evaluation
        # batch wants -- but it also means every camera and LiDAR returns an
        # empty frame. Requesting a sensor therefore overrides it, rather than
        # handing Phase 2 a training set of black images.
        s.no_rendering_mode = not (self.cfg.render or bool(self.cfg.sensors))
        self.world.apply_settings(s)

    def _set_weather(self, name: str) -> None:
        preset = WEATHER_PRESETS.get(name)
        if preset is None:
            raise ValueError(
                f"unknown weather {name!r}; have {sorted(WEATHER_PRESETS)}"
            )
        self.world.set_weather(carla.WeatherParameters(**preset))

    def _blueprint(self, wanted: str):
        """Find a blueprint, falling back rather than dying on a renamed asset.

        Blueprint ids move between CARLA releases (``vehicle.audi.a2`` is gone
        in some builds). A missing ego blueprint should degrade to a different
        car, not end the run twenty minutes into a batch.
        """
        lib = self.world.get_blueprint_library()
        try:
            return lib.find(wanted)
        except (IndexError, RuntimeError):
            options = list(lib.filter("vehicle.*"))
            if not options:                             # pragma: no cover
                raise RuntimeError("no vehicle blueprints in this build")
            return options[int(self.rng.integers(len(options)))]

    def _spawn_ego(self) -> None:
        bp = self._blueprint(self.cfg.ego_blueprint)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "hero")
        spawn_points = list(self.map.get_spawn_points())
        if not spawn_points:                            # pragma: no cover
            raise RuntimeError("map has no spawn points")
        if self.cfg.spawn_in_crowd:
            order = self._spawn_points_by_crowd(spawn_points)
        else:
            order = self.rng.permutation(len(spawn_points))
        vehicle = None
        for i in order:
            vehicle = self.world.try_spawn_actor(bp, spawn_points[int(i)])
            if vehicle is not None:
                self._ego_spawn = spawn_points[int(i)]
                break
        if vehicle is None:                             # pragma: no cover
            raise RuntimeError("could not spawn the ego anywhere on this map")
        self.vehicle = vehicle

        physics = vehicle.get_physics_control()
        wheels = list(physics.wheels)
        self._max_steer_deg = float(wheels[0].max_steer_angle) if wheels else 70.0
        # Wheel positions come back in centimetres, in world coordinates.
        if len(wheels) >= 3:
            front = np.array([wheels[0].position.x, wheels[0].position.y,
                              wheels[0].position.z])
            rear = np.array([wheels[2].position.x, wheels[2].position.y,
                             wheels[2].position.z])
            wheelbase = float(np.linalg.norm(front - rear)) / 100.0
        else:                                           # pragma: no cover
            wheelbase = self.params.wheelbase
        ext = vehicle.bounding_box.extent
        self.params = params_from_physics(
            self.params, 2 * ext.x, 2 * ext.y, wheelbase, self._max_steer_deg
        )
        # Built here rather than in __init__ because it closes over the
        # *measured* actuation limits, which are only known once CARLA has
        # told us which blueprint it actually spawned.
        self._long = LongitudinalTracker(
            self.params, self.cfg.resistance, ki=self.cfg.throttle_ki,
            throttle_gain=self.cfg.throttle_gain,
            brake_gain=self.cfg.brake_gain,
        )

        tf = vehicle.get_transform()
        x, y, theta = carla_to_odom(tf.location.x, tf.location.y, tf.rotation.yaw)
        self.ego = EgoState(x=x, y=y, theta=theta, v=0.0)

        col_bp = self.world.get_blueprint_library().find("sensor.other.collision")
        self._collision_sensor = self.world.spawn_actor(
            col_bp, carla.Transform(), attach_to=vehicle
        )
        self._collision_sensor.listen(self._on_collision)

    def _spawn_points_by_crowd(self, spawn_points, samples: int = 300):
        """Spawn-point indices ordered by nearness to the walker navigation mesh.

        Sampling the mesh rather than assuming where it is: its extent is a
        property of the map's build, not of anything readable from the API,
        and it differs between towns. Three hundred samples is enough to find
        the centroid of a district and cheap enough to do at connect time.

        Ordered rather than reduced to a single best point, because the caller
        still has to try them in turn -- a spawn point can be occupied.
        """
        pts = []
        for _ in range(samples):
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                pts.append((loc.x, loc.y))
        if not pts:                                     # pragma: no cover
            return self.rng.permutation(len(spawn_points))
        centre = np.mean(np.asarray(pts, dtype=np.float64), axis=0)
        d = [float(np.hypot(sp.location.x - centre[0], sp.location.y - centre[1]))
             for sp in spawn_points]
        return np.argsort(d)

    def _build_route(self) -> None:
        start = self.map.get_waypoint(self._ego_spawn.location)
        self.road = route_from_waypoint(
            start, self.cfg.route_length, self.rng, self.cfg.route_spacing,
            allow_revisit=self.cfg.long_route,
        )

    def _build_raster(self) -> None:
        """Rasterise every driving lane in the town, once.

        *Every* lane, including the oncoming ones. That is not a bug: this
        stack navigates free space and does not model lane discipline, and
        neither does the traffic it is built for. Restricting the drivable set
        to the ego's own lane would smuggle the assumption back in through the
        map.
        """
        wps = self.map.generate_waypoints(self.cfg.waypoint_spacing)
        pts, widths = [], []
        for wp in wps:
            loc = wp.transform.location
            x, y, _ = carla_to_odom(loc.x, loc.y, 0.0)
            pts.append((x, y))
            widths.append(float(wp.lane_width))
        self._raster = DrivableRaster.from_points(
            np.asarray(pts), np.asarray(widths), self.cfg.raster_resolution
        )
        self._static_boxes = self._map_obstacle_boxes()
        self._carved_cells = self._raster.carve(self._static_boxes)

    #: Static map meshes that stand in the carriageway and are *not* actors, so
    #: nothing spawns them and ``ground_truth_tracks`` cannot report them.
    #: Parked vehicles are the ones that matter -- they sit in a lane, which is
    #: exactly where the lane graph says the ego may drive.
    STATIC_OBSTACLE_LABELS: Tuple[str, ...] = ("Car", "Truck", "Bus", "Motorcycle",
                                               "Bicycle")

    def _map_obstacle_boxes(self):
        """Oriented footprints of the town's baked-in obstacles, odom frame.

        ``get_environment_objects`` returns map geometry, not actors, so this
        is the only way to see a parked car in Town10HD.  Anything CARLA does
        not label in this build is skipped rather than guessed at: the label
        enum has changed before -- the semantic tag numbering shifted at
        0.9.14 -- and a wrong label here would carve holes out of the road at
        random.
        """
        boxes = []
        for label in self.STATIC_OBSTACLE_LABELS:
            tag = getattr(carla.CityObjectLabel, label, None)
            if tag is None:                              # pragma: no cover
                continue
            for obj in self.world.get_environment_objects(tag):
                bb = obj.bounding_box
                x, y, theta = carla_to_odom(bb.location.x, bb.location.y,
                                            bb.rotation.yaw)
                boxes.append((x, y, theta, float(bb.extent.x), float(bb.extent.y)))
        return tuple(boxes)

    def _start_traffic_manager(self) -> None:
        tm = self.client.get_trafficmanager(self.cfg.tm_port)
        tm.set_synchronous_mode(True)
        # Without this the traffic manager is the one unseeded thing left in
        # the loop, and two runs of the same seed diverge for reasons that
        # look like the planner.
        tm.set_random_device_seed(int(self.cfg.seed))
        self.tm = tm

    def _spawn_vehicles(self) -> None:
        lib = self.world.get_blueprint_library()
        options = [b for b in lib.filter("vehicle.*")]
        points = list(self.map.get_spawn_points())
        order = self.rng.permutation(len(points))
        spawned = 0
        for i in order:
            if spawned >= self.cfg.n_vehicles:
                break
            bp = options[int(self.rng.integers(len(options)))]
            if bp.has_attribute("role_name"):
                bp.set_attribute("role_name", "autopilot")
            actor = self.world.try_spawn_actor(bp, points[int(i)])
            if actor is None:
                continue                # occupied spawn point; try the next
            actor.set_autopilot(True, self.cfg.tm_port)
            self._register(actor)
            spawned += 1

    def _spawn_walkers(self) -> None:
        """Pedestrians, and the controllers that make them walk.

        A walker with no ``controller.ai.walker`` attached simply stands
        still, which produces a demo where nobody ever steps into the road --
        the one thing this project exists to survive.
        """
        lib = self.world.get_blueprint_library()
        options = [b for b in lib.filter("walker.pedestrian.*")]
        if not options:                                 # pragma: no cover
            return
        if hasattr(self.world, "set_pedestrians_seed"):
            self.world.set_pedestrians_seed(int(self.cfg.seed))
        if hasattr(self.world, "set_pedestrians_cross_factor"):
            self.world.set_pedestrians_cross_factor(self.cfg.walker_cross_factor)
        ctrl_bp = lib.find("controller.ai.walker")

        for _ in range(self.cfg.n_walkers):
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            bp = options[int(self.rng.integers(len(options)))]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker = self.world.try_spawn_actor(bp, carla.Transform(loc))
            if walker is None:
                continue
            controller = self.world.try_spawn_actor(
                ctrl_bp, carla.Transform(), attach_to=walker
            )
            if controller is None:                      # pragma: no cover
                walker.destroy()
                continue
            controller.start()
            controller.go_to_location(
                self.world.get_random_location_from_navigation()
            )
            controller.set_max_speed(float(self.rng.uniform(0.9, 1.8)))
            self._walker_controllers.append(controller)
            self._register(walker)

    def _register(self, actor) -> None:
        cls = classify_blueprint(actor.type_id)
        try:
            ext = actor.bounding_box.extent
            hl, hw = float(ext.x), float(ext.y)
        except AttributeError:                          # pragma: no cover
            hl, hw = CLASS_EXTENT.get(cls, CLASS_EXTENT["unknown"])
        self._spawned[int(actor.id)] = _Spawned(actor, cls, hl, hw)

    def _attach_sensors(self) -> None:
        rig = default_sensor_rig(self.cfg.fixed_delta_seconds)
        lib = self.world.get_blueprint_library()
        for name in self.cfg.sensors:
            spec = rig.get(name)
            if spec is None:
                raise ValueError(f"unknown sensor {name!r}; have {sorted(rig)}")
            bp = lib.find(spec.blueprint)
            for k, v in spec.attributes.items():
                if bp.has_attribute(k):
                    bp.set_attribute(k, v)
            tf = carla.Transform(
                carla.Location(x=spec.xyz[0], y=spec.xyz[1], z=spec.xyz[2]),
                carla.Rotation(pitch=spec.pitch),
            )
            sensor = self.world.spawn_actor(bp, tf, attach_to=self.vehicle)
            sensor.listen(self._sensor_callback(name))
            self._sensors[name] = sensor

    # -- callbacks --------------------------------------------------------
    def _sensor_callback(self, name: str):
        def _cb(data) -> None:
            # Latest frame only. Queueing every frame and never draining it is
            # the standard way a CARLA client's memory grows without bound.
            self.frames[name] = data
        return _cb

    def _on_collision(self, event) -> None:
        other = getattr(event, "other_actor", None)
        type_id = getattr(other, "type_id", "") or "unknown"
        if type_id.startswith("vehicle") or type_id.startswith("walker"):
            label = f"actor:{classify_blueprint(type_id)}#{getattr(other, 'id', -1)}"
        else:
            label = f"static:{type_id}"
        self._collisions.append(label)

    def _pump_sensors(self) -> None:
        """Nothing to drain: the callbacks already store the latest frame.

        Kept as a named no-op so the tick sequence reads in the order it
        actually happens, and so a future queue-based rig has one place to go.
        """

    # -- SimWorld ---------------------------------------------------------
    def step(self, dt: float, accel: float, steer: float) -> None:
        """Apply a control and advance ``dt`` seconds of simulated time.

        ``dt`` must be a whole number of server ticks. Silently rounding it
        would make the control rate differ from the one the results claim, and
        that discrepancy is invisible in every plot.
        """
        fds = self.cfg.fixed_delta_seconds
        n = int(round(dt / fds))
        if n < 1 or abs(n * fds - dt) > 1e-9:
            raise ValueError(
                f"dt={dt} is not a whole number of {fds}s server ticks; set "
                f"CarlaConfig.fixed_delta_seconds to match RunnerConfig.sim_dt"
            )
        steer_norm = normalised_steer(steer, self._max_steer_deg)
        # ``self.ego.a`` is the acceleration actually achieved over the
        # previous step -- the loop is closed on that.  See
        # :class:`LongitudinalTracker` for why an open-loop pedal map made the
        # ego crawl at half its cruise speed.
        throttle, brake = self._long.update(accel, self.ego.v, self.ego.a, dt)
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=throttle, steer=steer_norm, brake=brake)
        )
        for _ in range(n):
            self.world.tick()
        self._pump_sensors()
        self._actor_cache = None
        self._sync_ego(n * fds, steer_norm)

    def _sync_ego(self, dt: float, steer_norm: float = 0.0) -> None:
        tf = self.vehicle.get_transform()
        x, y, theta = carla_to_odom(tf.location.x, tf.location.y, tf.rotation.yaw)
        vel = self.vehicle.get_velocity()
        vx, vy = carla_vector_to_odom(vel.x, vel.y)
        # Signed forward speed, not |v|: a vehicle sliding backwards after a
        # hard stop must not read as making progress.
        v = float(vx * math.cos(theta) + vy * math.sin(theta))
        prev_v = self.ego.v
        self.ego.x, self.ego.y, self.ego.theta = x, y, theta
        self.ego.v = v
        # The MPC seeds its rollouts with the *current* steer angle and
        # acceleration to respect its rate limits, so these have to be what the
        # vehicle is really doing, not what was last commanded.
        self.ego.delta = steer_from_control(steer_norm, self._max_steer_deg)
        self.ego.a = float((v - prev_v) / max(dt, 1e-6))
        self.t += dt
        self.ego.t = self.t

    def _route_frame(self, s_m: float):
        """``(x, y, tangent_unit)`` at arc length ``s_m`` along the route, odom.

        The tangent is what makes a lateral offset mean "across the road"
        rather than "along the world's y axis" -- the route turns, and an
        obstruction placed by world axes ends up on the pavement half the
        time.
        """
        pts = self.road.points
        cum = self.road._cum
        i = int(np.searchsorted(cum, s_m))
        i = int(np.clip(i, 1, len(pts) - 1))
        p0, p1 = pts[i - 1], pts[i]
        t = p1 - p0
        n = float(np.linalg.norm(t))
        t = t / n if n > 1e-6 else np.array([1.0, 0.0])
        return float(p1[0]), float(p1[1]), t

    def _spawn_obstruction(self) -> None:
        """Lay the unstructured-traffic scene along the route.

        See :class:`ObstructionSpec` for what the standing group stands in for
        and why it is not cattle.

        **Two CARLA rules this obeys, both learned the hard way.**

        *Walkers must come from the navigation mesh.* Spawning one at an
        arbitrary point on a carriageway appears to succeed -- ``try_spawn_actor``
        returns a handle -- and the walker is dead by the next tick, which
        shows up much later as a scene with no pedestrians in it. So candidate
        positions are drawn from ``get_random_location_from_navigation`` and
        kept only when they land near the route.

        *Vehicles must come from the lane graph.* A blocker is placed on a
        waypoint beside the route rather than at a computed offset, because a
        computed offset lands in a wall as often as on tarmac.
        """
        spec = self.cfg.obstruction
        if spec is None:
            return
        lib = self.world.get_blueprint_library()
        walker_bps = list(lib.filter("walker.pedestrian.*"))
        ctrl_bp = lib.find("controller.ai.walker")
        vehicle_bps = [b for b in lib.filter("vehicle.*")
                       if b.get_attribute("number_of_wheels").as_int() == 4]
        if not walker_bps:                               # pragma: no cover
            return
        if hasattr(self.world, "set_pedestrians_seed"):
            self.world.set_pedestrians_seed(int(self.cfg.seed))

        base = self.road.progress(self.ego.x, self.ego.y)
        lo, hi = base + spec.start_m, base + spec.start_m + spec.span_m

        def route_xy(s_m):
            x, y, _t = self._route_frame(s_m)
            return np.array([x, y])

        anchors = [route_xy(s) for s in np.linspace(lo, hi, 24)]

        def near_route(loc, limit):
            ox, oy, _ = carla_to_odom(loc.x, loc.y, 0.0)
            p = np.array([ox, oy])
            return min(float(np.linalg.norm(p - a)) for a in anchors) <= limit

        def sample_nav(limit, tries=400):
            for _ in range(tries):
                loc = self.world.get_random_location_from_navigation()
                if loc is not None and near_route(loc, limit):
                    return loc
            return None

        # -- standing in and beside the carriageway, no controller ---------
        standing = 0
        for _ in range(spec.standing):
            loc = sample_nav(spec.lateral + 2.0)
            if loc is None:
                continue
            bp = walker_bps[int(self.rng.integers(len(walker_bps)))]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            actor = self.world.try_spawn_actor(bp, carla.Transform(loc))
            if actor is not None:
                self._register(actor)                    # no controller: it stands
                standing += 1

        # -- crossing, aimed at the far side of the road -------------------
        crossing = 0
        for _ in range(spec.crossing):
            loc = sample_nav(12.0)
            if loc is None:
                continue
            bp = walker_bps[int(self.rng.integers(len(walker_bps)))]
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            actor = self.world.try_spawn_actor(bp, carla.Transform(loc))
            if actor is None:
                continue
            controller = self.world.try_spawn_actor(
                ctrl_bp, carla.Transform(), attach_to=actor)
            if controller is None:                       # pragma: no cover
                actor.destroy()
                continue
            target = sample_nav(12.0) or \
                self.world.get_random_location_from_navigation()
            controller.start()
            controller.go_to_location(target)
            controller.set_max_speed(float(self.rng.uniform(0.7, 1.5)))
            self._walker_controllers.append(controller)
            self._register(actor)
            crossing += 1

        # -- stopped vehicles, on the lane graph, pinching the corridor ----
        blockers = 0
        for k in range(spec.blockers):
            if not vehicle_bps:                          # pragma: no cover
                break
            s_off = spec.start_m + spec.span_m * (0.25 + 0.22 * k)
            x, y, _t = self._route_frame(base + s_off)
            cx, cy, _ = odom_to_carla(x, y, 0.0)
            wp = self.map.get_waypoint(carla.Location(x=cx, y=cy, z=0.5),
                                       project_to_road=True,
                                       lane_type=carla.LaneType.Driving)
            if wp is None:
                continue
            side = wp.get_right_lane() if k % 2 == 0 else wp.get_left_lane()
            target = side if (side is not None and
                              side.lane_type == carla.LaneType.Driving) else wp
            tf = target.transform
            tf.location.z += 0.4
            bp = vehicle_bps[int(self.rng.integers(len(vehicle_bps)))]
            actor = self.world.try_spawn_actor(bp, tf)
            if actor is not None:
                actor.set_autopilot(False)
                actor.apply_control(carla.VehicleControl(hand_brake=True))
                self._register(actor)
                blockers += 1

        self._obstruction_counts = {"standing": standing, "crossing": crossing,
                                    "blockers": blockers}
        # One tick so the spawns are real before anything reads a transform:
        # in synchronous mode a fresh actor reports (0, 0) until the server
        # has processed it, which reads as "everything spawned at the origin".
        self.world.tick()

    def lane_context(self) -> Dict[str, object]:
        """Where the ego is in CARLA's road network, for telemetry only.

        Nothing upstream of stage 1 may read this. The stack navigates free
        space and does not model lanes at all -- that is the whole design --
        so this exists to *measure* what the free-space planner did, not to
        inform it. A lane change here is an observation about the trajectory,
        not a manoeuvre the planner selected.

        ``lane_id`` sign encodes side of the road in OpenDRIVE, and ids are
        only unique within a ``road_id``, so a change means the pair changed.
        Junctions are reported separately because lane ids are meaningless
        inside one and every junction traversal would otherwise read as a
        flurry of lane changes.
        """
        wp = self.map.get_waypoint(self.vehicle.get_transform().location,
                                   project_to_road=True,
                                   lane_type=carla.LaneType.Driving)
        return {
            "road_id": int(wp.road_id),
            "lane_id": int(wp.lane_id),
            "is_junction": bool(wp.is_junction),
            "lane_width": float(wp.lane_width),
        }

    def _actors(self) -> List[ActorSnapshot]:
        """Traffic in the odom frame, from one world snapshot.

        One ``get_snapshot()`` instead of a ``get_transform()`` RPC per actor:
        forty actors at 20 Hz is 800 round trips a second, and that alone can
        halve the frame rate on a laptop GPU.
        """
        if self._actor_cache is not None:
            return self._actor_cache
        snap = self.world.get_snapshot()
        ego_xy = (self.ego.x, self.ego.y)
        out: List[ActorSnapshot] = []
        for aid, info in self._spawned.items():
            found = snap.find(aid)
            if found is None:
                continue                # destroyed, or not yet in this frame
            tf = found.get_transform()
            x, y, theta = carla_to_odom(
                tf.location.x, tf.location.y, tf.rotation.yaw
            )
            if math.hypot(x - ego_xy[0], y - ego_xy[1]) > self.cfg.track_range:
                continue
            vel = found.get_velocity()
            vx, vy = carla_vector_to_odom(vel.x, vel.y)
            out.append(ActorSnapshot(
                id=aid, cls=info.cls, x=x, y=y, theta=theta,
                v=float(math.hypot(vx, vy)),
                half_length=info.half_length, half_width=info.half_width,
            ))
        self._actor_cache = out
        return out

    @property
    def actors(self) -> List[ActorSnapshot]:
        """Named to match the built-in world, so the trace recorder and the
        demo renderer work against either simulator unchanged."""
        return self._actors()

    def red_light_keepouts(self):
        """Keep-out boxes across the stop line when the ego faces a red.

        **Why a keep-out and not a signal input.** This stack plans over free
        space; it has no signal semantics anywhere, and adding a "traffic light
        state" to the stage contracts would be inventing an interface for one
        simulator. A red light is a region the vehicle may not enter, and a
        region the vehicle may not enter is exactly what an occupancy grid is
        for. So the constraint arrives the same way a parked lorry does, and
        the planner needs no new concept to obey it.

        The box spans **every** lane the light stops, not just the ego's. A
        free-space planner asked to avoid a barrier across one lane will
        cheerfully drive round it into the oncoming one, which is a worse
        outcome than the red light it was obeying.

        This is a ground-truth stub, exactly like the tracks and the drivable
        area -- stages 1--3 are stubbed and served by the simulator, and in a
        real vehicle the light state would come from perception. It is not a
        contribution of this project and the deck should not present it as one.
        """
        if not self.cfg.obey_traffic_lights:
            return ()
        light = self.vehicle.get_traffic_light()
        if light is None:                       # not inside any trigger volume
            return ()
        if light.get_state() != carla.TrafficLightState.Red:
            return ()
        boxes = []
        for wp in light.get_stop_waypoints():
            tf = wp.transform
            x, y, theta = carla_to_odom(tf.location.x, tf.location.y,
                                        tf.rotation.yaw)
            # Thin along the lane, full lane width across it: a line to stop
            # at, not a block to squeeze past.
            boxes.append((x, y, theta, self.cfg.stop_line_depth,
                          0.5 * float(wp.lane_width)))
        return tuple(boxes)

    def ground_truth_grids(
        self, half_extent: float = 32.0, resolution: float = 0.25
    ) -> Tuple[OccupancyGrid, OccupancyGrid]:
        static = self._raster.window(self.ego.x, self.ego.y, half_extent, resolution)
        # Red lights belong in the *static* grid: they are scenery, not agents.
        # Putting them in the dynamic one would hand the predictor a stationary
        # "actor" to extrapolate, and a risk field around a stop line is not a
        # thing.
        keepouts = self.red_light_keepouts()
        if keepouts:
            static = stamp_actors(static, keepouts)
        boxes = [(a.x, a.y, a.theta, a.half_length, a.half_width)
                 for a in self._actors()]
        return static, stamp_actors(static, boxes)

    def ground_truth_tracks(self) -> List[Track]:
        tracks: List[Track] = []
        for a in self._actors():
            rng_m = math.hypot(a.x - self.ego.x, a.y - self.ego.y)
            sigma_p, sigma_v = _range_noise_sigmas(rng_m, self.cfg.track_noise)
            vx = a.v * math.cos(a.theta)
            vy = a.v * math.sin(a.theta)
            x, y = a.x, a.y
            if self.cfg.track_noise > 0.0:
                x += self.rng.normal(0.0, sigma_p)
                y += self.rng.normal(0.0, sigma_p)
                vx += self.rng.normal(0.0, sigma_v)
                vy += self.rng.normal(0.0, sigma_v)
            tracks.append(Track(
                id=a.id, x=float(x), y=float(y), vx=float(vx), vy=float(vy),
                cls=a.cls,
                cov=np.diag([sigma_p**2, sigma_p**2, sigma_v**2, sigma_v**2]),
                half_length=a.half_length, half_width=a.half_width,
            ))
        return tracks

    def local_goal(self, lookahead: float = 28.0) -> np.ndarray:
        return self.road.point_ahead(self.ego.x, self.ego.y, lookahead)

    def _metric_boxes(self):
        return [(a.x, a.y, a.theta, a.half_length, a.half_width, a.v,
                 f"actor:{a.cls}#{a.id}") for a in self._actors()]

    def collision(self) -> Optional[str]:
        """What we hit, or ``None``.

        The collision sensor is authoritative -- it is the physics engine's own
        answer. The geometric check behind it exists because the sensor reports
        *contact*, and the built-in simulator reports *footprint overlap*: a
        graze that CARLA resolves without a contact impulse would otherwise
        count as a collision in one simulator and not the other, and the whole
        point of :mod:`divas.sim.geometry` is that the two tables compare.
        """
        if self._collisions:
            return self._collisions[0]
        hit = _first_hit(self.ego, self.params, self._metric_boxes())
        if hit is not None:
            return hit
        centres, _r = _footprint_discs(self.ego, self.params)
        if not self._raster.contains(centres[:, 0], centres[:, 1]).all():
            return "off_road"
        return None

    def clear_collisions(self) -> None:
        self._collisions.clear()

    def clearance_to_actors(self) -> float:
        return _min_clearance(self.ego, self.params, self._metric_boxes())

    def time_to_collision(self) -> float:
        return _ttc(self.ego, self.params, self._metric_boxes())

    # -- lifecycle --------------------------------------------------------
    def self_test(self) -> List[str]:
        """Names this object is missing from the ``SimWorld`` protocol."""
        from divas.sim.interface import check
        return check(self)

    def close(self) -> None:
        """Destroy everything we spawned and hand the server back as we found it.

        Idempotent, and every stage is wrapped: a half-finished cleanup that
        raises on the first dead actor leaves the rest of the town littered,
        which is worse than the error it is reporting.

        Three things here were learned from a live server, and all three look
        like fussiness until you have seen the core dump:

        1. **Stop the sensors before destroying anything.** A listening sensor
           whose callback fires into a half-torn-down client is an exception
           thrown on CARLA's own thread, which no ``try`` of ours can catch --
           it reaches ``std::terminate`` and the process aborts.
        2. **Destroy walker controllers before their walkers.** A controller
           attached to an already-destroyed walker is itself already dead, and
           destroying it warns and then throws.
        3. **Destroy in one server-side batch, with the tick cue set.** In
           synchronous mode the server only *processes* a destruction on the
           next tick. Destroy-then-disconnect leaves the commands queued and
           the actors alive, which is the leak this method exists to prevent.
        """
        if self._closed:
            return
        self._closed = True

        sensors = [s for s in list(self._sensors.values())
                   + [getattr(self, "_collision_sensor", None)] if s is not None]
        for sensor in sensors:
            try:
                sensor.stop()
            except (RuntimeError, AttributeError):
                pass
        for controller in self._walker_controllers:
            try:
                controller.stop()
            except (RuntimeError, AttributeError):
                pass

        # Controllers first, then sensors, then traffic, then the ego.
        doomed = (list(self._walker_controllers) + sensors
                  + [info.handle for info in self._spawned.values()]
                  + [getattr(self, "vehicle", None)])
        try:
            self.client.apply_batch_sync(
                [carla.command.DestroyActor(a) for a in doomed if a is not None],
                True,                       # due_tick_cue: process it this tick
            )
        except (RuntimeError, AttributeError):
            # Fall back to one-by-one rather than leaving a town full of actors.
            for actor in doomed:
                try:
                    actor.destroy()
                except (RuntimeError, AttributeError):
                    pass
        self._sensors.clear()
        self._walker_controllers.clear()
        self._spawned.clear()

        # Back to async mode last, and only after the destruction has been
        # ticked through. Leaving the server synchronous with no client ticking
        # it freezes it for the next person who connects, and that person is
        # usually you, ten minutes later, wondering why CARLA hangs.
        try:
            self.tm.set_synchronous_mode(False)
        except (RuntimeError, AttributeError):
            pass
        try:
            self.world.apply_settings(self._original_settings)
        except (RuntimeError, AttributeError):
            pass

    def __enter__(self) -> "CarlaWorld":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def route_from_waypoint(start_wp, length: float, rng, spacing: float = 2.0,
                       allow_revisit: bool = False) -> Route:
    """Follow lanes from ``start_wp`` for ``length`` metres, seeded at junctions.

    Deliberately *not* ``agents.navigation.GlobalRoutePlanner``: that lives in
    CARLA's ``PythonAPI/carla`` example package rather than in the ``carla``
    wheel, so depending on it means depending on a path on disk. Following the
    lane graph with a seeded choice at each junction is twenty lines, is
    reproducible, and is all a route needs to be here -- routing is not this
    project's contribution.

    ``allow_revisit`` lifts the stop-on-loop rule, which is what any route
    longer than the town needs: Town10HD is about 400 x 230 m, so a two
    kilometre drive is necessarily a circuit. It returns a **windowed**
    :class:`Route`, because a self-crossing route measured by a global nearest
    point search reports nonsense -- see that class. The two go together and
    are set together here rather than left to the caller to remember.
    """
    pts: List[Tuple[float, float]] = []
    wp = start_wp
    travelled = 0.0
    seen = set()
    while travelled <= length:
        loc = wp.transform.location
        x, y, _ = carla_to_odom(loc.x, loc.y, 0.0)
        pts.append((x, y))
        key = (round(x, 1), round(y, 1))
        if key in seen and len(pts) > 4 and not allow_revisit:
            break                       # the route has looped; stop here
        seen.add(key)
        nxt = list(wp.next(spacing))
        if not nxt:
            break
        wp = nxt[0] if len(nxt) == 1 else nxt[int(rng.integers(len(nxt)))]
        travelled += spacing
    if len(pts) < 2:                                    # pragma: no cover
        raise RuntimeError("route is a single point: is the ego on a driving lane?")
    return Route(np.asarray(pts, dtype=np.float64), windowed=allow_revisit)
