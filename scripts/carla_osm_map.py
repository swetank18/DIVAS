#!/usr/bin/env python3
"""Drive a real Indian road network in CARLA, imported from OpenStreetMap.

CARLA's stock towns are American: lane markings, orderly junctions, a grid
laid out by a level designer. No amount of extra traffic makes Town10HD an
Indian street, and presenting it as one invites the obvious question.

This imports the **actual road geometry of a real place**. OpenStreetMap has
India mapped in detail; ``carla.Osm2Odr`` converts a bounding box of it to
OpenDRIVE, and ``generate_opendrive_world`` builds a drivable map from that.
The junction angles, the road widths, the way five streets meet at one point
without a roundabout -- all of it is the real network rather than a designer's
idea of one.

    python3 scripts/carla_osm_map.py --fetch --bbox 12.9680 77.5920 12.9760 77.6010
    python3 scripts/carla_osm_map.py --osm bengaluru.osm --load

**Be precise about what this buys and what it does not.** The *geometry* is
authentic: this is the road network of the place named. The *scenery* is not --
a procedurally generated OpenDRIVE world has road surface and nothing else, no
buildings, no roadside encroachment, no textures. So it looks bare, and it is
honest in exactly the dimension the stock towns are dishonest in. Show it for
the network; show Town10HD for photorealism; and say which is which.

Default bounding box is central Bengaluru around MG Road.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path as FsPath

sys.path.insert(0, str(FsPath(__file__).resolve().parents[1]))

from divas.sim.carla_bridge import HAVE_CARLA

if HAVE_CARLA:                                        # pragma: no cover
    import carla

OVERPASS = "https://overpass-api.de/api/interpreter"


def fetch(bbox, out: FsPath, timeout: int = 180) -> FsPath:
    """Pull every ``highway`` way in the bounding box from Overpass.

    ``highway`` rather than everything: the converter only reads roads, and a
    full extract of a dense Indian city centre is tens of megabytes of
    buildings that would be discarded anyway.
    """
    south, west, north, east = bbox
    query = (f'[out:xml][timeout:{timeout}];'
             f'(way["highway"]({south},{west},{north},{east});>;);out body;')
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-s", "--max-time", str(timeout), "-X", "POST", OVERPASS,
         "--data-urlencode", f"data={query}", "-o", str(out)],
        check=True,
    )
    return out


def convert(osm_path: FsPath, xodr_path: FsPath, lane_width: float = 3.2,
            lights: bool = False) -> FsPath:
    """OSM -> OpenDRIVE.

    ``default_lane_width`` matters more here than it looks. OSM rarely tags
    lane widths on Indian roads, so every unmarked way inherits this value,
    and it becomes the width of the carriageway the planner sees. 3.2 m is a
    realistic Indian urban lane; CARLA's own default of 4.0 m quietly widens
    every street in the city.

    Traffic lights are off by default: the converter generates them at every
    junction it can, which on a dense OSM extract produces hundreds of signals
    that do not exist in reality.
    """
    settings = carla.Osm2OdrSettings()
    settings.default_lane_width = float(lane_width)
    settings.generate_traffic_lights = bool(lights)
    settings.center_map = True
    xodr = carla.Osm2Odr.convert(osm_path.read_text(encoding="utf-8"), settings)
    xodr_path.parent.mkdir(parents=True, exist_ok=True)
    xodr_path.write_text(xodr, encoding="utf-8")
    return xodr_path


def load(client, xodr_path: FsPath, timeout: float = 240.0):
    """Build the world from OpenDRIVE and report what came out.

    ``wall_height`` at zero and a generous ``additional_width``: the default
    walls a procedural world puts along every road edge would show up in the
    drivable raster as a canyon, and the extra width is what makes an
    OSM-derived carriageway wide enough for two-way mixed traffic.
    """
    params = carla.OpendriveGenerationParameters(
        vertex_distance=2.0,
        max_road_length=500.0,
        wall_height=0.0,
        additional_width=0.6,
        smooth_junctions=True,
        enable_mesh_visibility=True,
    )
    client.set_timeout(timeout)
    return client.generate_opendrive_world(
        xodr_path.read_text(encoding="utf-8"), params)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbox", nargs=4, type=float,
                    default=[12.9680, 77.5920, 12.9760, 77.6010],
                    metavar=("SOUTH", "WEST", "NORTH", "EAST"),
                    help="default: central Bengaluru around MG Road")
    ap.add_argument("--osm", default="docs/maps/bengaluru.osm")
    ap.add_argument("--xodr", default="docs/maps/bengaluru.xodr")
    ap.add_argument("--lane-width", type=float, default=3.2)
    ap.add_argument("--lights", action="store_true")
    ap.add_argument("--fetch", action="store_true", help="download the OSM extract")
    ap.add_argument("--load", action="store_true", help="build it in a running server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    args = ap.parse_args()

    osm = FsPath(args.osm)
    xodr = FsPath(args.xodr)

    if args.fetch or not osm.exists():
        print(f"fetching OSM for bbox {args.bbox} ...")
        fetch(args.bbox, osm)
        print(f"  wrote {osm}  ({osm.stat().st_size/1024:.0f} kB)")

    if not HAVE_CARLA:
        print("the carla package is needed to convert or load", file=sys.stderr)
        return 2

    print("converting to OpenDRIVE ...")
    convert(osm, xodr, args.lane_width, args.lights)
    print(f"  wrote {xodr}  ({xodr.stat().st_size/1024:.0f} kB)")

    if not args.load:
        print("\n--load to build it in a running server")
        return 0

    client = carla.Client(args.host, args.port)
    print("building the world (this takes a while for a dense extract) ...")
    world = load(client, xodr)
    cmap = world.get_map()
    wps = cmap.generate_waypoints(2.0)
    spawns = cmap.get_spawn_points()
    xs = [w.transform.location.x for w in wps]
    ys = [w.transform.location.y for w in wps]
    print(f"\nmap        {cmap.name}")
    print(f"waypoints  {len(wps)} at 2 m spacing")
    print(f"extent     {max(xs)-min(xs):.0f} x {max(ys)-min(ys):.0f} m")
    print(f"spawns     {len(spawns)}")
    print("\nThe geometry is the real network. The scenery is not: a procedural "
          "OpenDRIVE world\nhas road surface and nothing else. Say which is which.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
