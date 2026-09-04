"""Render the cattle-and-crowd scenario as an Indian street, in Blender.

Run headless::

    ~/blender-4.2.1-linux-x64/blender -b -P scripts/blender/indian_street.py -- \
        --replay docs/replay-cattle_and_crowd.json --out demo/frames --arm 1

**Why render this ourselves rather than in a simulator.** No off-the-shelf
driving simulator ships the assets that make an Indian road look Indian --
CARLA's 214 blueprints contain no cattle, no auto-rickshaw, no handcart, and
every one of its towns is lane-marked. Modelling the scene here means every
asset is a choice: the road has no markings because Indian roads often do not,
the cattle have the shoulder hump of a zebu because that is the animal that
stands in the carriageway, and the auto-rickshaw is the shape everyone in the
room will recognise instantly.

**What is real and what is dressing.** The *trajectories* are real: ego pose,
every actor's pose and the planned path come from
``scripts/export_replay.py``, which drives the same closed-loop runner that
produced the ablation. Nothing here moves because an animator moved it. The
*appearance* is dressing -- shop fronts, trees, dust -- and carries no claim.
"""

import bpy
import bmesh
import json
import math
import sys
from mathutils import Vector

# ---------------------------------------------------------------- arguments
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default

REPLAY = arg("--replay", "docs/replay-cattle_and_crowd.json")
OUT = arg("--out", "demo/frames")
ARM = int(arg("--arm", "1"))
FPS = int(arg("--fps", "30"))
SAMPLES = int(arg("--samples", "16"))
RES_X, RES_Y = int(arg("--width", "1600")), int(arg("--height", "900"))

data = json.load(open(REPLAY))
arm = data["arms"][ARM]
frames = arm["frames"]
SIM_DT = frames[1]["t"] - frames[0]["t"] if len(frames) > 1 else 0.05

# ------------------------------------------------------------------ helpers
def clear():
    bpy.ops.wm.read_factory_settings(use_empty=True)

ASSETS = "assets"


def tex_mat(name, base, normal=None, rough_map=None, scale=7.0, tint=(1, 1, 1)):
    """A PBR material from Poly Haven maps, tiled over the surface.

    Flat colours are what made the first render read as a toy. A real diffuse
    map with a normal and a roughness map costs almost nothing in EEVEE and is
    most of the distance between "blocks" and "surfaces".
    """
    import os
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Scale"].default_value = (scale, scale, scale)
    nt.links.new(coord.outputs["Object"], mapping.inputs["Vector"])

    def img(path, non_color=False):
        if not path or not os.path.exists(path):
            return None
        n = nt.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(path, check_existing=True)
        if non_color:
            n.image.colorspace_settings.name = "Non-Color"
        nt.links.new(mapping.outputs["Vector"], n.inputs["Vector"])
        return n

    d = img(base)
    if d:
        if tint != (1, 1, 1):
            mix = nt.nodes.new("ShaderNodeMixRGB")
            mix.blend_type = "MULTIPLY"
            mix.inputs["Fac"].default_value = 1.0
            mix.inputs["Color2"].default_value = (*tint, 1.0)
            nt.links.new(d.outputs["Color"], mix.inputs["Color1"])
            nt.links.new(mix.outputs["Color"], bsdf.inputs["Base Color"])
        else:
            nt.links.new(d.outputs["Color"], bsdf.inputs["Base Color"])
    r = img(rough_map, non_color=True)
    if r:
        nt.links.new(r.outputs["Color"], bsdf.inputs["Roughness"])
    n = img(normal, non_color=True)
    if n:
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(n.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    return m


def bevel_edges(obj, width=0.045, segments=2):
    """Catch a highlight on every edge.

    A perfectly sharp 90-degree edge does not exist and reads instantly as
    computer graphics. Faces stay flat on purpose: subdivision was tried and
    rounded every primitive into its own blob -- a car with glass shards and
    cattle like balloon animals.
    """
    m = obj.modifiers.new("bevel", "BEVEL")
    m.width = width
    m.segments = segments
    m.limit_method = "ANGLE"
    m.angle_limit = math.radians(50)
    return obj


def mat(name, rgb, rough=0.8, metal=0.0, emit=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1.0)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*rgb, 1.0)
        b.inputs["Emission Strength"].default_value = emit
    return m

def box(name, size, loc, material, rot_z=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    o = bpy.context.object
    o.name = name
    # primitive_cube_add(size=1.0) is a unit cube -- side 1, extent +/-0.5 --
    # so the scale IS the full dimension. Halving it here (the obvious
    # instinct, since the extent is half the side) built the entire scene at
    # 50%: a car narrower than its own wheelbase, and a road that read as a
    # runway.
    o.scale = (size[0], size[1], size[2])
    o.rotation_euler = (0, 0, rot_z)
    o.data.materials.append(material)
    return o

def cyl(name, r, h, loc, material, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=h, location=loc, vertices=16)
    o = bpy.context.object
    o.name = name
    o.rotation_euler = rot
    o.data.materials.append(material)
    return o

def join(objs, name, bevel=0.045):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    for o in objs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.join()
    o = bpy.context.object
    o.name = name
    if bevel:
        bevel_edges(o, width=bevel)
    return o

# ------------------------------------------------------------------ palette
M = {}
def build_materials():
    import os
    tex = os.path.join(ASSETS, "tex")
    def t(n, m):
        return os.path.join(tex, f"{n}_{m}.jpg")
    if os.path.exists(t("asphalt_02", "Diffuse")):
        M["tarmac"] = tex_mat("tarmac", t("asphalt_02", "Diffuse"),
                              t("asphalt_02", "nor_gl"), t("asphalt_02", "Rough"),
                              scale=7.0, tint=(0.78, 0.76, 0.73))
        # brown_mud_dry, not a rocky/grassy map: a green verge is the single
        # thing that makes a dusty Indian roadside read as an English one.
        M["dust"] = tex_mat("dust", t("dirt", "Diffuse"), t("dirt", "nor_gl"),
                            t("dirt", "Rough"), scale=6.0, tint=(1.10, 1.0, 0.86))
    else:
        M["tarmac"] = mat("tarmac", (0.10, 0.10, 0.11), rough=0.95)
        M["dust"] = mat("dust", (0.52, 0.42, 0.28), rough=1.0)
    M["kerb"] = mat("kerb", (0.55, 0.53, 0.50), rough=0.9)
    M["ego"] = mat("ego", (0.85, 0.86, 0.88), rough=0.35, metal=0.4)
    M["glass"] = mat("glass", (0.06, 0.09, 0.11), rough=0.15, metal=0.6)
    M["cow"] = mat("cow", (0.80, 0.76, 0.70), rough=0.9)
    M["cow2"] = mat("cow2", (0.42, 0.30, 0.20), rough=0.9)
    M["auto_y"] = mat("auto_y", (0.94, 0.72, 0.08), rough=0.5)
    M["auto_g"] = mat("auto_g", (0.05, 0.35, 0.18), rough=0.5)
    M["bus"] = mat("bus", (0.72, 0.20, 0.16), rough=0.6)
    M["bike"] = mat("bike", (0.12, 0.12, 0.14), rough=0.5)
    M["skin"] = mat("skin", (0.62, 0.44, 0.30), rough=0.85)
    M["cloth"] = [mat(f"cloth{i}", c, rough=0.9) for i, c in enumerate(
        [(0.80, 0.20, 0.25), (0.15, 0.35, 0.70), (0.95, 0.85, 0.30),
         (0.20, 0.55, 0.35), (0.85, 0.45, 0.15), (0.75, 0.75, 0.78)])]
    M["shop"] = [mat(f"shop{i}", c, rough=0.85) for i, c in enumerate(
        [(0.75, 0.35, 0.25), (0.30, 0.50, 0.65), (0.85, 0.72, 0.35),
         (0.45, 0.60, 0.40), (0.70, 0.55, 0.70)])]
    M["awning"] = mat("awning", (0.85, 0.30, 0.20), rough=0.8)
    M["tyre"] = mat("tyre", (0.05, 0.05, 0.06), rough=0.95)
    M["trunk"] = mat("trunk", (0.28, 0.20, 0.13), rough=0.95)
    M["leaf"] = mat("leaf", (0.16, 0.32, 0.12), rough=0.9)
    M["path"] = mat("path", (0.95, 0.42, 0.18), rough=0.4, emit=1.5)

# ------------------------------------------------------------------- models
def make_cow(name, dark=False):
    """A zebu: shoulder hump, dewlap, horns. The hump is what makes it read
    as Indian cattle rather than a Friesian, and it is one extra box."""
    skin = M["cow2"] if dark else M["cow"]
    parts = [
        box(f"{name}_body", (1.75, 0.62, 0.72), (0, 0, 0.86), skin),
        box(f"{name}_hump", (0.50, 0.46, 0.30), (0.32, 0, 1.30), skin),
        box(f"{name}_neck", (0.55, 0.40, 0.42), (0.98, 0, 1.02), skin),
        box(f"{name}_head", (0.52, 0.34, 0.34), (1.38, 0, 1.02), skin),
        box(f"{name}_dewlap", (0.40, 0.16, 0.30), (1.00, 0, 0.66), skin),
    ]
    for sx in (0.62, -0.62):
        for sy in (0.22, -0.22):
            parts.append(cyl(f"{name}_leg", 0.075, 0.86, (sx, sy, 0.44), skin))
    for sy in (0.16, -0.16):
        parts.append(cyl(f"{name}_horn", 0.035, 0.30, (1.46, sy, 1.26), skin,
                         rot=(math.radians(70) * (1 if sy > 0 else -1), 0, 0)))
    parts.append(cyl(f"{name}_tail", 0.035, 0.70, (-0.88, 0, 0.72), skin,
                     rot=(0, math.radians(12), 0)))
    return join(parts, name)

def make_auto(name):
    """Auto-rickshaw: one wheel in front, two behind, canopy, yellow over
    green. The silhouette is the point -- nobody needs it labelled."""
    parts = [
        box(f"{name}_lower", (2.45, 1.28, 0.62), (0, 0, 0.44), M["auto_g"]),
        box(f"{name}_cab", (1.70, 1.22, 0.78), (-0.20, 0, 1.14), M["auto_y"]),
        box(f"{name}_roof", (1.95, 1.30, 0.10), (-0.15, 0, 1.56), M["auto_y"]),
        box(f"{name}_front", (0.70, 0.55, 0.60), (1.10, 0, 0.78), M["auto_y"]),
        cyl(f"{name}_w0", 0.30, 0.16, (1.02, 0, 0.30), M["tyre"],
            rot=(math.radians(90), 0, 0)),
    ]
    for sy in (0.62, -0.62):
        parts.append(cyl(f"{name}_w", 0.32, 0.18, (-0.85, sy, 0.32), M["tyre"],
                         rot=(math.radians(90), 0, 0)))
    return join(parts, name)

def make_bike(name):
    parts = [
        box(f"{name}_body", (1.55, 0.28, 0.34), (0, 0, 0.62), M["bike"]),
        box(f"{name}_rider", (0.42, 0.46, 0.80), (-0.10, 0, 1.28), M["cloth"][1]),
        box(f"{name}_head", (0.26, 0.26, 0.26), (-0.10, 0, 1.80), M["skin"]),
    ]
    for sx in (0.66, -0.66):
        parts.append(cyl(f"{name}_w", 0.31, 0.10, (sx, 0, 0.31), M["tyre"],
                         rot=(math.radians(90), 0, 0)))
    return join(parts, name)

def make_person(name, i=0):
    c = M["cloth"][i % len(M["cloth"])]
    parts = [
        cyl(f"{name}_legs", 0.15, 0.86, (0, 0, 0.43), M["cloth"][(i + 2) % 6]),
        box(f"{name}_torso", (0.36, 0.26, 0.62), (0, 0, 1.16), c),
        cyl(f"{name}_head", 0.115, 0.24, (0, 0, 1.60), M["skin"]),
    ]
    return join(parts, name)

def make_bus(name):
    parts = [
        box(f"{name}_body", (9.0, 2.55, 2.55), (0, 0, 1.55), M["bus"]),
        box(f"{name}_glass", (8.2, 2.60, 0.72), (0, 0, 2.35), M["glass"]),
        box(f"{name}_roof", (9.0, 2.55, 0.14), (0, 0, 2.90), M["auto_y"]),
    ]
    for sx in (3.0, -3.0):
        for sy in (1.20, -1.20):
            parts.append(cyl(f"{name}_w", 0.50, 0.28, (sx, sy, 0.50), M["tyre"],
                             rot=(math.radians(90), 0, 0)))
    return join(parts, name)

def make_ego(name, length, width):
    parts = [
        box(f"{name}_body", (length, width, 0.72), (0, 0, 0.62), M["ego"]),
        box(f"{name}_cabin", (length * 0.52, width * 0.90, 0.60),
            (-length * 0.06, 0, 1.24), M["ego"]),
        box(f"{name}_glass", (length * 0.46, width * 0.94, 0.42),
            (-length * 0.06, 0, 1.30), M["glass"]),
    ]
    for sx in (length * 0.33, -length * 0.33):
        for sy in (width * 0.46, -width * 0.46):
            parts.append(cyl(f"{name}_w", 0.32, 0.20, (sx, sy, 0.32), M["tyre"],
                             rot=(math.radians(90), 0, 0)))
    return join(parts, name)

MAKERS = {
    "animal": make_cow,
    "autorickshaw": make_auto,
    "motorcycle": make_bike,
    "bicycle": make_bike,
    "pedestrian": make_person,
    "bus": make_bus,
}


# -------------------------------------------------------------- environment
def build_world(road_poly, statics, length_hint):
    """Road, verges and roadside. Deliberately without lane markings.

    The absence is the point and it is not laziness: an Indian carriageway
    often has no usable markings, which is the premise the whole stack rests
    on. Painting them here would contradict the argument the video is making.
    """
    xs = [p[0] for p in road_poly]
    ys = [p[1] for p in road_poly]
    x0, x1 = min(xs) - 20, max(xs) + 20
    half = max(abs(min(ys)), abs(max(ys)))

    # ground: dust either side, tarmac in the middle, a low kerb between
    bpy.ops.mesh.primitive_plane_add(size=1, location=((x0 + x1) / 2, 0, -0.02))
    g = bpy.context.object
    g.scale = ((x1 - x0) / 2, 60, 1)
    g.name = "ground"
    g.data.materials.append(M["dust"])

    road = box("road", (x1 - x0, 2 * half, 0.04), ((x0 + x1) / 2, 0, 0.0), M["tarmac"])
    for obj in (g, road):
        mod = obj.modifiers.new("subd", "SUBSURF")
        mod.subdivision_type = "SIMPLE"
        mod.levels = mod.render_levels = 2
    for sy in (half, -half):
        box("kerb", (x1 - x0, 0.5, 0.22), ((x0 + x1) / 2, sy, 0.11), M["kerb"])

    # roadside: shop fronts on one side, trees and poles on the other, so the
    # frame has depth without pretending to be a modelled city
    # Roadside sizes are in metres and were re-tuned after the unit-cube fix
    # doubled everything: shop fronts had grown to 13 m wide and were leaning
    # over the carriageway.
    rng = 1234
    for i in range(int((x1 - x0) / 7)):
        x = x0 + 5 + i * 7
        rng = (rng * 1103515245 + 12345) % 2147483648
        h = 3.2 + (rng % 220) / 100.0
        w = 5.0 + (rng % 180) / 100.0
        box(f"shop{i}", (w, 6.0, h), (x, half + 6.2, h / 2),
            M["shop"][i % len(M["shop"])])
        box(f"awn{i}", (w * 0.9, 1.8, 0.10), (x, half + 2.5, 2.6), M["awning"])
        for k in range(2):
            cyl(f"awnpole{i}_{k}", 0.045, 2.6, (x + (w * 0.38) * (1 if k else -1),
                                                half + 1.7, 1.3), M["kerb"])
    for i in range(int((x1 - x0) / 9)):
        x = x0 + 7 + i * 9
        cyl(f"trunk{i}", 0.18, 3.6, (x, -half - 4.0, 1.8), M["trunk"])
        bpy.ops.mesh.primitive_ico_sphere_add(radius=1.7, subdivisions=2,
                                              location=(x, -half - 4.0, 4.2))
        c = bpy.context.object
        c.name = f"canopy{i}"
        c.scale = (1.0, 1.0, 0.72)
        c.data.materials.append(M["leaf"])
        cyl(f"pole{i}", 0.07, 5.5, (x + 4.0, half + 1.2, 2.75), M["kerb"])

    for s in statics:
        if s["kind"] == "pothole":
            cyl("pothole", s["r"], 0.14, (s["x"], s["y"], 0.02), M["tarmac"])
        else:
            if s["l"] > 7.0:
                o = make_bus("stopped_bus")
                o.location = (s["x"], s["y"], 0)
                o.rotation_euler = (0, 0, s["theta"])
            else:
                box("handcart", (s["l"], s["w"], 1.1), (s["x"], s["y"], 0.55),
                    M["trunk"], rot_z=s["theta"])


def build_lighting():
    """Warm, high, hazy -- Indian midday, not an overcast European afternoon."""
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 60))
    sun = bpy.context.object
    sun.data.energy = 2.6
    # A small angular diameter gives hard-edged shadows. Indian midday sun is
    # harsh, and soft shadows read as an overcast European afternoon.
    sun.data.angle = math.radians(0.9)
    sun.data.color = (1.0, 0.95, 0.86)
    sun.rotation_euler = (math.radians(52), 0, math.radians(28))

    w = bpy.context.scene.world
    if w is None:
        w = bpy.data.worlds.new("World")
        bpy.context.scene.world = w
    w.use_nodes = True
    nt = w.node_tree
    bg = nt.nodes["Background"]
    import os
    hdri = os.path.join(ASSETS, "hdri", "sky.hdr")
    if os.path.exists(hdri):
        # A captured sky. Its value is the *lighting*, not the backdrop: a
        # hemisphere of measured radiance gives every surface a plausible
        # ambient term, which one flat colour cannot at any strength.
        env = nt.nodes.new("ShaderNodeTexEnvironment")
        env.image = bpy.data.images.load(hdri, check_existing=True)
        mp = nt.nodes.new("ShaderNodeMapping")
        cd = nt.nodes.new("ShaderNodeTexCoord")
        mp.inputs["Rotation"].default_value = (0, 0, math.radians(150))
        nt.links.new(cd.outputs["Generated"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], env.inputs["Vector"])
        nt.links.new(env.outputs["Color"], bg.inputs["Color"])
        bg.inputs["Strength"].default_value = 0.85
    # Warm, slightly dusty sky rather than clean blue: the haze is what puts
    # distance in a flat scene with no atmospheric perspective of its own.
    bg.inputs["Color"].default_value = (0.74, 0.76, 0.76, 1.0)
    bg.inputs["Strength"].default_value = 1.35

    # Depth haze, so the far end of the street recedes instead of staying as
    # crisp as the foreground.
    scene = bpy.context.scene
    scene.world.mist_settings.use_mist = False
    try:
        scene.eevee.use_volumetric_fog = False
    except AttributeError:
        pass


# --------------------------------------------------------------- animation
def key_pose(obj, frame, x, y, theta, visible=True):
    obj.location = (x, y, obj.location.z)
    obj.rotation_euler = (obj.rotation_euler.x, obj.rotation_euler.y, theta)
    obj.keyframe_insert("location", frame=frame)
    obj.keyframe_insert("rotation_euler", frame=frame)
    obj.hide_viewport = obj.hide_render = not visible
    obj.keyframe_insert("hide_viewport", frame=frame)
    obj.keyframe_insert("hide_render", frame=frame)


def animate():
    veh = data["vehicle"]
    ego = make_ego("ego", veh["length"], veh["width"])

    # One pool per class, sized to the busiest frame. Actors come and go, so
    # a pool with visibility keys is cheaper and steadier than creating and
    # deleting objects mid-animation.
    need = {}
    for f in frames:
        c = {}
        for a in f["actors"]:
            c[a[3]] = c.get(a[3], 0) + 1
        for k, v in c.items():
            need[k] = max(need.get(k, 0), v)

    pools = {}
    for cls, n in need.items():
        maker = MAKERS.get(cls, make_person)
        pools[cls] = []
        for i in range(n):
            o = maker(f"{cls}_{i}", **({"dark": i % 3 == 1} if cls == "animal"
                                       else ({"i": i} if cls == "pedestrian" else {}))) \
                if cls in ("animal", "pedestrian") else maker(f"{cls}_{i}")
            o.location = (0, 0, o.location.z)
            pools[cls].append(o)

    # the planned path, as a glowing ribbon on the road
    path_obj = box("plan", (1.0, 0.35, 0.02), (0, 0, 0.06), M["path"])

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 34.0
    cam = bpy.data.objects.new("cam", cam_data)
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = max(1, int(round(frames[-1]["t"] * FPS)))

    for f in frames:
        fr = max(1, int(round(f["t"] * FPS)))
        ex, ey, eth = f["ego"]
        key_pose(ego, fr, ex, ey, eth)

        used = {k: 0 for k in pools}
        for a in f["actors"]:
            cls = a[3] if a[3] in pools else "pedestrian"
            if cls not in pools:
                continue
            i = used[cls]
            if i >= len(pools[cls]):
                continue
            key_pose(pools[cls][i], fr, a[0], a[1], a[2])
            used[cls] += 1
        for cls, objs in pools.items():
            for o in objs[used.get(cls, 0):]:
                key_pose(o, fr, 0, 0, 0, visible=False)

        # the plan: a flat bar laid from the ego towards the path's far end
        p = f["path"]
        if len(p) >= 2:
            tx, ty = p[-1]
            dx, dy = tx - ex, ty - ey
            d = math.hypot(dx, dy)
            path_obj.location = (ex + dx / 2, ey + dy / 2, 0.06)
            path_obj.scale = (max(d, 0.1) / 2, 0.35 / 2, 0.02 / 2)
            path_obj.rotation_euler = (0, 0, math.atan2(dy, dx))
            path_obj.hide_viewport = path_obj.hide_render = False
        else:
            path_obj.hide_viewport = path_obj.hide_render = True
        for prop in ("location", "scale", "rotation_euler",
                     "hide_viewport", "hide_render"):
            path_obj.keyframe_insert(prop, frame=fr)

        # chase camera: behind, above, and looking a little ahead of the ego
        back, up, ahead = 12.0, 4.6, 15.0
        cx = ex - back * math.cos(eth)
        cy = ey - back * math.sin(eth)
        cam.location = (cx, cy, up)
        tx = ex + ahead * math.cos(eth)
        ty = ey + ahead * math.sin(eth)
        direction = Vector((tx - cx, ty - cy, 1.1 - up))
        cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        cam.keyframe_insert("location", frame=fr)
        cam.keyframe_insert("rotation_euler", frame=fr)

    # linear between samples: the trajectory is already at 20 Hz, and Bezier
    # smoothing would overshoot corners the vehicle did not actually cut.
    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            for fc in obj.animation_data.action.fcurves:
                for kp in fc.keyframe_points:
                    kp.interpolation = "LINEAR"


def configure_render():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x = RES_X
    scene.render.resolution_y = RES_Y
    scene.render.fps = FPS
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = OUT + "/f_"
    scene.eevee.taa_render_samples = SAMPLES
    try:
        scene.eevee.use_shadows = True
    except AttributeError:
        pass
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - High Contrast"
    scene.view_settings.exposure = -0.3


def main():
    global ASSETS
    ASSETS = arg("--assets", "assets")
    clear()
    build_materials()
    build_world(data["road"], data["statics"], 120.0)
    build_lighting()
    animate()
    configure_render()
    print(f"[divas] scene ready: {bpy.context.scene.frame_end} frames at {FPS} fps")
    bpy.ops.render.render(animation=True)
    print("[divas] render complete")


main()
