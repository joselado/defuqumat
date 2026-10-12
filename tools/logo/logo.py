"""logo.py: the package logo, built in Blender from compute.py's files and rendered headless.

    B=/u/40/ladovj1/unix/apps/blender-5.2.2-linux-x64/blender
    $B -b --factory-startup --python tools/logo/logo.py -- render --mesh docs/logo/mesh \\
        --out docs/logo                 # the four committed PNGs, Cycles, about 4 min
    $B -b --factory-startup --python tools/logo/logo.py -- still --mesh docs/logo/mesh \\
        --out draft.png --camera icon --engine BLENDER_EEVEE --samples 16   # a draft, 2 s
    $B -b --factory-startup --python tools/logo/logo.py -- build --mesh docs/logo/mesh \\
        --blend logo.blend              # the scene, to open by hand

``render`` writes ``defuqumat-icon.png`` (``--icon-size``, 1024), ``defuqumat-banner.png``
and ``defuqumat-banner-dark.png`` (four times ``--size`` by ``--size``, 2400 by 600, the
wordmark dark for a light background and light for a dark one); the 256 and 64 px
icons are pillow downsamples of the first. The defaults of every dial are the
values that made the committed logo, so a bare ``render`` regenerates it.

The scene is a deterministic function of the files in ``--mesh`` (``surface_m.ply``,
``surface_m_low.ply``, ``arrows.npz``, ``atoms.json``, written by ``compute.py mesh``)
and of the dials below: the Cr t2g shell as an isosurface of |m| of the CrI3
monolayer (plans/logo.md), the Cr moment as one arrow, on each iodine a
translucent cloud of the antiparallel density with a small arrow of the field,
the atoms and the Cr to I bonds, and the wordmark. The icon is one CrI6
octahedron (centred on a Cr, a disc of 6.5 bohr, no wordmark, since at 64 px the
name is noise); the banner the honeycomb ring of six (a hexagon centre, 11.5
bohr) beside the wordmark. One Blender unit is ten bohr.

Dials: ``--camera icon|banner``, ``--arrows atoms+field|all|atoms|field|surface|none``
with ``--field-sign neg|both``, ``--colour mz|tilt``, ``--text 0|1`` and
``--text-colour dark|light``, ``--shell 0|1``, ``--surface m|n``, ``--centre``,
``--radius``, ``--cr-radius``, ``--elev-icon``, ``--elev-banner``, ``--cloud-reach``,
``--lobe-reach``, ``--engine``, ``--samples``, ``--view Standard|AgX``, ``--size``.
"""
import argparse
import json
import math
import os
import sys

import bpy
import bmesh
import numpy as np
from mathutils import Matrix, Vector

SCALE = 0.1  # Blender units per bohr
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

PALETTE = {
    "surface": (0.93, 0.30, 0.06, 1.0),   # the lobes of |m|: warm orange
    "surface_n": (0.75, 0.78, 0.85, 1.0),  # the total density, if drawn: pale
    "Cr": (0.20, 0.22, 0.30, 1.0),         # dark steel
    "I": (0.55, 0.30, 0.75, 1.0),          # iodine violet
    "bond": (0.45, 0.45, 0.50, 1.0),
    "shell": (0.25, 0.45, 0.95, 1.0),      # the antiparallel cloud on the iodine
    "text": (0.10, 0.10, 0.12, 1.0),       # the wordmark, as an emission (flat) colour
    "arrow_lo": (0.10, 0.35, 0.90, 1.0),   # m_z down, or no tilt
    "arrow_hi": (0.95, 0.15, 0.10, 1.0),   # m_z up, or the largest tilt
}
RADIUS = {"Cr": 0.10, "I": 0.075}  # sphere radii, Blender units (one unit is ten bohr)
BOND_RADIUS = 0.028


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _set_input(node, name, value):
    """Set a Principled input by name, tolerating the names that moved between versions."""
    aliases = {
        "Transmission Weight": ("Transmission Weight", "Transmission"),
        "Subsurface Weight": ("Subsurface Weight", "Subsurface"),
        "Coat Weight": ("Coat Weight", "Clearcoat"),
        "Emission Color": ("Emission Color", "Emission"),
    }
    for key in aliases.get(name, (name,)):
        if key in node.inputs:
            node.inputs[key].default_value = value
            return
    print(f"  (no input {name} on {node.name})")


def material(name, color, roughness=0.4, metallic=0.0, transmission=0.0,
             subsurface=0.0, emission=0.0, vertex_color=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    bsdf = nodes["Principled BSDF"]
    _set_input(bsdf, "Base Color", color)
    _set_input(bsdf, "Roughness", roughness)
    _set_input(bsdf, "Metallic", metallic)
    if transmission:
        _set_input(bsdf, "Transmission Weight", transmission)
    if subsurface:
        _set_input(bsdf, "Subsurface Weight", subsurface)
    if emission:
        _set_input(bsdf, "Emission Color", color)
        _set_input(bsdf, "Emission Strength", emission)
    if vertex_color:
        vc = nodes.new("ShaderNodeVertexColor")
        vc.layer_name = vertex_color
        mat.node_tree.links.new(vc.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def new_object(name, mesh, mat=None, collection=None):
    ob = bpy.data.objects.new(name, mesh)
    (collection or bpy.context.scene.collection).objects.link(ob)
    if mat is not None:
        if mesh.materials:
            mesh.materials[0] = mat
        else:
            mesh.materials.append(mat)
    return ob


def smooth(mesh):
    for p in mesh.polygons:
        p.use_smooth = True


def rotation_to(vector):
    """A 3x3 matrix taking +z to ``vector`` (unit), numpy."""
    v = np.asarray(vector, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.eye(3)
    v = v / n
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, v))
    if c > 1 - 1e-9:
        return np.eye(3)
    if c < -1 + 1e-9:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(z, v)
    s = np.linalg.norm(axis)
    axis /= s
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + s * k + (1 - c) * k @ k


# ---------------------------------------------------------------------------
# the pieces of the scene
# ---------------------------------------------------------------------------

def load_geometry(mesh_dir, surface):
    with open(os.path.join(mesh_dir, "atoms.json")) as f:
        atoms = json.load(f)
    arrows = np.load(os.path.join(mesh_dir, "arrows.npz"))
    ply = os.path.join(mesh_dir, f"surface_{surface}.ply")
    low = os.path.join(mesh_dir, f"surface_{surface}_low.ply")
    return atoms, arrows, ply, (low if os.path.exists(low) else None)


def centre_of(atoms, where):
    """The clip centre in bohr: a hexagon centre of the honeycomb (a lattice
    point of the tiled patch) or the middle of the patch."""
    at = np.array(atoms["at"])
    t = atoms["tile"]
    if where == "patch":
        frac = np.array([t / 2, t / 2, 0.5])
    elif where == "cr":
        # the Cr nearest the middle of the patch: one octahedron fills the frame
        mid = np.array([t / 2, t / 2, 0.5]) @ at
        cr = [np.array(a["position"]) for a in atoms["atoms"] if a["symbol"].startswith("Cr")]
        return min(cr, key=lambda p: np.linalg.norm(p - mid))
    else:
        frac = np.array([(t + 1) // 2, (t + 1) // 2, 0.5])
    return frac @ at


def add_surface(ply, centre, radius, color, name="surface", faint=False, anchors=None,
                reach=0.5):
    """Import one isosurface, centre and scale it, clip it to the disc. The main
    surface is opaque in one colour; the faint one (``faint``) is a translucent
    shell whose colour is the vertex colour the PLY carries, the cosine of m to
    z, so the iodine's antiparallel moment reads as the other colour."""
    bpy.ops.wm.ply_import(filepath=ply)
    ob = bpy.context.selected_objects[0]
    ob.name = name
    me = ob.data
    # move to the centre, scale, and clip to the disc
    bm = bmesh.new()
    bm.from_mesh(me)
    for v in bm.verts:
        v.co = (v.co - Vector(centre)) * SCALE
    r2 = (radius * SCALE) ** 2
    outside = [f for f in bm.faces if f.calc_center_median().xy.length_squared > r2]
    bmesh.ops.delete(bm, geom=outside, context="FACES")
    if anchors is not None and len(anchors):
        # keep only the faces within ``reach`` of one of the anchors (the iodines
        # drawn), so a cloud of an atom outside the disc leaves no crescent
        bm.faces.ensure_lookup_table()
        cent = np.array([f.calc_center_median()[:] for f in bm.faces])
        pts = np.asarray(anchors)
        near = np.zeros(len(cent), dtype=bool)
        for p in pts:
            near |= ((cent - p) ** 2).sum(axis=1) < reach ** 2
        bmesh.ops.delete(bm, geom=[f for f, ok in zip(bm.faces, near) if not ok],
                         context="FACES")
    # the normals: marching cubes hands over vertex normals whose sign depends on
    # its gradient convention, and Cycles shades a surface with inward custom
    # normals black; so the custom normals are dropped and the face normals
    # recomputed consistently outward from the geometry
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(me)
    bm.free()
    for attr in ("custom_normal", "sharp_edge"):
        if attr in me.attributes:  # the importer's copy of the file's normals
            me.attributes.remove(me.attributes[attr])
    smooth(me)
    if faint:
        # a translucent cloud in one colour (the sign is already in which part
        # of the surface compute.py kept); the vertex colour stays on the mesh
        layer = me.color_attributes[0].name if len(me.color_attributes) else None
        mat = material(name, color, roughness=0.6)
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        _set_input(bsdf, "Alpha", 0.45)
        mat.surface_render_method = "BLENDED"
        mat.use_backface_culling = True
    else:
        mat = material(name, color, roughness=0.3, metallic=0.1)
    if me.materials:
        me.materials[0] = mat
    else:
        me.materials.append(mat)
    print(f"  {name}: {len(me.vertices)} vertices inside the disc"
          + (f", colour layer {layer}" if faint else ""))
    return ob


def add_atoms(atoms, centre, radius):
    at = np.array(atoms["at"])
    spheres = {}
    for sym, rad in RADIUS.items():
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=rad)
        me = bpy.data.meshes.new(f"sphere_{sym}")
        bm.to_mesh(me)
        bm.free()
        smooth(me)
        me.materials.append(material(f"atom_{sym}", PALETTE[sym], roughness=0.35,
                                     metallic=0.6 if sym == "Cr" else 0.1))
        spheres[sym] = me
    placed = {}
    for a in atoms["atoms"]:
        p = (np.array(a["position"]) - centre) * SCALE
        if np.hypot(p[0], p[1]) > radius * SCALE:
            continue
        ob = new_object(f"{a['symbol']}_{a['index']}_{a['tile'][0]}{a['tile'][1]}",
                        spheres[a["symbol"]])
        ob.location = p.tolist()
        placed[(a["index"], tuple(a["tile"]))] = p
    # bonds: every Cr-I pair closer than the cutoff, between the placed atoms
    bond_mat = material("bond", PALETTE["bond"], roughness=0.5)
    bm = bmesh.new()
    keys = list(placed)
    pos = np.array([placed[k] for k in keys])
    syms = {k: next(a["symbol"] for a in atoms["atoms"]
                    if a["index"] == k[0] and tuple(a["tile"]) == k[1]) for k in keys}
    cutoff = max(b["d"] for b in atoms["bonds"]) * SCALE * 1.02 if atoms["bonds"] else 0
    n_bonds = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            if syms[keys[i]] == syms[keys[j]]:
                continue
            d = pos[j] - pos[i]
            length = np.linalg.norm(d)
            if length > cutoff:
                continue
            rot = Matrix(rotation_to(d).tolist()).to_4x4()
            mid = Matrix.Translation(Vector(((pos[i] + pos[j]) / 2).tolist()))
            bmesh.ops.create_cone(bm, cap_ends=False, segments=16, radius1=BOND_RADIUS,
                                  radius2=BOND_RADIUS, depth=float(length), matrix=mid @ rot)
            n_bonds += 1
    me = bpy.data.meshes.new("bonds")
    bm.to_mesh(me)
    bm.free()
    smooth(me)
    new_object("bonds", me, bond_mat)
    print(f"  atoms placed: {len(placed)}, bonds: {n_bonds}")
    return {sym: np.array([placed[k] for k in keys if syms[k] == sym])
            for sym in set(syms.values())}


def arrow_mesh(length, shaft=0.022, head=0.055, head_frac=0.38, segments=16):
    """Vertices and faces of one arrow along +z, numpy; the shaft from 0 to
    (1 - head_frac) length, the head to ``length``."""
    L1 = length * (1 - head_frac)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    v = [np.array([0.0, 0.0, 0.0])]
    v += [np.array([x * shaft, y * shaft, 0.0]) for x, y in ring]
    v += [np.array([x * shaft, y * shaft, L1]) for x, y in ring]
    v += [np.array([x * head, y * head, L1]) for x, y in ring]
    v.append(np.array([0.0, 0.0, length]))
    verts = np.array(v)
    faces = []
    tip = len(verts) - 1
    for k in range(segments):
        k2 = (k + 1) % segments
        faces.append((0, 1 + k2, 1 + k))                              # base cap
        faces.append((1 + k, 1 + k2, 1 + segments + k2, 1 + segments + k))   # shaft
        faces.append((1 + segments + k, 1 + segments + k2,
                      1 + 2 * segments + k2, 1 + 2 * segments + k))         # head base
        faces.append((1 + 2 * segments + k, 1 + 2 * segments + k2, tip))     # head
    return verts, faces


def add_arrows(arrows, centre, radius, which, colour_by, length_scale, atom_scale=2.4,
               field_sign="neg"):
    groups = []
    if which in ("all", "surface"):
        groups.append(("surface", arrows["surface_points"], arrows["surface_m"], 1.0))
    if which in ("all", "field", "atoms+field"):
        pts, m = arrows["grid_points"], arrows["grid_m"]
        if field_sign == "neg":
            # only where m points against the Cr moment: the induced iodine
            # moment, the one place the field arrows say something the atom
            # arrows do not; the parallel ones near the Cr are clutter
            pts, m = pts[m[:, 2] < 0], m[m[:, 2] < 0]
        groups.append(("field", pts, m, 1.0))
    if which in ("all", "atoms", "atoms+field"):
        groups.append(("atoms", arrows["atom_points"], arrows["atom_m"], 1.0))
    if not groups:
        return
    all_m = np.concatenate([g[2] for g in groups]) if groups else np.zeros((0, 3))
    m_max = max(np.linalg.norm(all_m, axis=1).max(), 1e-12)
    tilt_max = 1e-12
    for _, _, m, _ in groups:
        nrm = np.linalg.norm(m, axis=1) + 1e-12
        tilt_max = max(tilt_max, float(np.degrees(np.arccos(np.clip(np.abs(m[:, 2]) / nrm, -1, 1))).max()))
    print(f"  arrows: |m| max {m_max:.4f}, largest tilt from z {tilt_max:.2f} degrees")
    lo, hi = np.array(PALETTE["arrow_lo"]), np.array(PALETTE["arrow_hi"])
    for name, pts, m, _ in groups:
        pts = (np.asarray(pts) - centre) * SCALE
        inside = np.hypot(pts[:, 0], pts[:, 1]) < radius * SCALE
        pts, m = pts[inside], np.asarray(m)[inside]
        if len(pts) == 0:
            continue
        verts_all, faces_all, cols_all = [], [], []
        base = 0
        for p, mv in zip(pts, m):
            nrm = np.linalg.norm(mv)
            if nrm < 1e-9:
                continue
            length = length_scale * (0.5 + 0.5 * nrm / m_max)
            if name == "atoms":
                length = length_scale * atom_scale * nrm / m_max
            v, f = arrow_mesh(length)
            R = rotation_to(mv)
            v = v @ R.T + p
            if colour_by == "tilt":
                tilt = math.degrees(math.acos(min(1.0, abs(mv[2]) / nrm)))
                w = tilt / tilt_max
            else:
                w = 0.5 * (1 + mv[2] / nrm)
            col = lo * (1 - w) + hi * w
            verts_all.append(v)
            faces_all += [tuple(i + base for i in face) for face in f]
            cols_all.append(np.tile(col, (len(v), 1)))
            base += len(v)
        verts = np.concatenate(verts_all)
        cols = np.concatenate(cols_all)
        me = bpy.data.meshes.new(f"arrows_{name}")
        me.from_pydata(verts.tolist(), [], faces_all)
        me.update()
        attr = me.color_attributes.new(name="col", type="FLOAT_COLOR", domain="POINT")
        for i, c in enumerate(cols):
            attr.data[i].color = c.tolist()
        smooth(me)
        new_object(f"arrows_{name}", me,
                   material(f"arrows_{name}", (1, 1, 1, 1), roughness=0.3,
                            metallic=0.2, vertex_color="col"))
        print(f"  arrows {name}: {len(pts)} inside the disc")


def add_text(text, camera, distance, offset, size):
    curve = bpy.data.curves.new("wordmark", type="FONT")
    curve.body = text
    curve.size = size
    curve.extrude = 0.02 * size
    curve.bevel_depth = 0.004 * size
    curve.align_x = "LEFT"
    curve.align_y = "CENTER"
    if os.path.exists(FONT):
        curve.font = bpy.data.fonts.load(FONT)
    mat = material("text", (0, 0, 0, 1), roughness=1.0)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    _set_input(bsdf, "Emission Color", PALETTE["text"])   # flat, lit by itself
    _set_input(bsdf, "Emission Strength", 1.0)
    ob = new_object("wordmark", curve, mat)
    # billboard: the camera's orientation, placed in its frame (the camera's
    # matrix_world is not evaluated yet in a scene built this way, so it is
    # assembled from the location and the rotation set above)
    ob.rotation_euler = camera.rotation_euler
    cam_matrix = Matrix.Translation(camera.location) @ camera.rotation_euler.to_matrix().to_4x4()
    local = Vector((offset[0], offset[1], -distance))
    ob.location = cam_matrix @ local
    return ob


def add_lights_and_world():
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.9, 0.9, 0.92, 1.0)
    bg.inputs[1].default_value = 0.35
    specs = [("key", "SUN", (3, -4, 6), 2.2, 3.0), ("fill", "AREA", (-5, -2, 4), 90, 0),
             ("rim", "AREA", (2, 5, 3), 70, 0)]
    for name, kind, loc, energy, angle in specs:
        light = bpy.data.lights.new(name, kind)
        light.energy = energy
        if kind == "SUN":
            light.angle = math.radians(angle)
        else:
            light.size = 6.0
        ob = bpy.data.objects.new(name, light)
        bpy.context.scene.collection.objects.link(ob)
        ob.location = loc
        direction = Vector((0, 0, 0)) - Vector(loc)
        ob.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def add_camera(kind, extent, elev_icon=58.0, elev_banner=40.0, with_text=True):
    """An orthographic camera whose frame is computed from the clip radius: the
    icon is a square 2.4 extents wide with the structure in its middle; the banner
    is four times as wide as high, the structure in its left square and the
    wordmark in the rest. Returns the camera, its distance and the frame (w, h)."""
    cam = bpy.data.cameras.new("camera")
    cam.type = "ORTHO"
    ob = bpy.data.objects.new("camera", cam)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.scene.camera = ob
    h = 2.4 * extent
    if kind == "icon":
        h = 2.2 * extent
        elev, w, shift = math.radians(elev_icon), h, 0.0
    else:
        elev, w, shift = math.radians(elev_banner), 4 * h, (4 * h) / 2 - h / 2
    cam.ortho_scale = w
    dist = 12.0 * extent
    azim = math.radians(-90)
    target = Vector((0, 0, 0.1))
    forward = -Vector((math.cos(elev) * math.cos(azim),
                       math.cos(elev) * math.sin(azim), math.sin(elev)))
    loc = target - dist * forward
    ob.rotation_euler = forward.to_track_quat("-Z", "Y").to_euler()
    rot = ob.rotation_euler.to_matrix()
    right, up = rot @ Vector((1, 0, 0)), rot @ Vector((0, 1, 0))
    # the banner's structure sits in the left square (the camera moves right);
    # the icon's sits above the wordmark (the camera moves down a little)
    if kind == "icon" and with_text:
        loc = loc - 0.11 * h * up
    ob.location = loc + shift * right
    cam.clip_end = 100 * extent
    return ob, dist, (w, h)


def place_text(cam, dist, frame, kind, extent):
    """The wordmark in the camera's frame: under the structure for the icon,
    in the right three quarters for the banner."""
    w, h = frame
    if kind == "banner":
        size = 0.40 * h
        x = -w / 2 + 1.06 * h
        y = -0.08 * h
    else:
        size = 0.17 * h
        x = -0.5 * w + 0.06 * h
        y = -0.5 * h + 0.14 * h
    add_text("defuqumat", cam, dist, (x, y), size)


# ---------------------------------------------------------------------------
# scene and render
# ---------------------------------------------------------------------------

def resolve_kind(args, kind):
    """The icon is one octahedron (centred on a Cr, a small disc, the lobes of
    that Cr alone); the banner the honeycomb (a hexagon centre, a wide disc).
    ``auto`` picks by kind; an explicit --centre or --radius overrides."""
    args.camera = kind
    if args.text is None:
        args.text = 0 if kind == "icon" else 1
    if args.centre == "auto":
        args.centre = "cr" if kind == "icon" else "lattice"
    if args.radius is None:
        args.radius = args.icon_radius if kind == "icon" else args.banner_radius
    if args.lobe_reach is None:
        args.lobe_reach = 0.4 if kind == "icon" else 0.0
    return args


def build_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    atoms, arrows, ply, low = load_geometry(args.mesh, args.surface)
    RADIUS["Cr"], RADIUS["I"] = args.cr_radius, args.i_radius
    centre = centre_of(atoms, args.centre)
    print(f"centre (bohr) {np.round(centre, 3)}, clip radius {args.radius} bohr")
    placed = add_atoms(atoms, centre, args.radius)
    add_surface(ply, centre, args.radius,
                PALETTE["surface" if args.surface == "m" else "surface_n"],
                anchors=placed.get("Cr") if args.lobe_reach > 0 else None,
                reach=args.lobe_reach)
    if low is not None and args.shell:
        add_surface(low, centre, args.radius, PALETTE["shell"], name="shell", faint=True,
                    anchors=placed.get("I"), reach=args.cloud_reach)
    add_arrows(arrows, centre, args.radius, args.arrows, args.colour, args.arrow_length,
               args.atom_arrow, args.field_sign)
    add_lights_and_world()
    extent = args.radius * SCALE
    cam, dist, frame = add_camera(args.camera, extent, args.elev_icon, args.elev_banner,
                                  bool(args.text))
    if args.text:
        PALETTE["text"] = {"dark": (0.10, 0.10, 0.12, 1.0),
                           "light": (0.88, 0.88, 0.90, 1.0)}[args.text_colour]
        place_text(cam, dist, frame, args.camera, extent)
    scene.render.film_transparent = True
    scene.view_settings.view_transform = args.view
    scene.view_settings.look = "None"
    return scene


def configure_render(scene, args, width, height, path):
    scene.render.engine = args.engine
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "16"
    scene.render.filepath = path
    if args.engine == "CYCLES":
        scene.cycles.device = "CPU"
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
    else:
        scene.eevee.taa_render_samples = args.samples


def render(scene, args, width, height, path):
    configure_render(scene, args, width, height, path)
    import time
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print(f"rendered {path} ({width}x{height}, {args.engine}, {args.samples} samples) "
          f"in {time.time() - t:.1f} s")


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("verb", choices=("still", "render", "build"))
    p.add_argument("--mesh", required=True, help="directory written by compute.py mesh")
    p.add_argument("--out", default="draft.png", help="a PNG (still) or a directory (render)")
    p.add_argument("--blend", default="logo.blend")
    p.add_argument("--surface", choices=("m", "n"), default="m")
    p.add_argument("--camera", choices=("icon", "banner"), default="icon")
    p.add_argument("--arrows", choices=("all", "atoms", "atoms+field", "field", "surface", "none"),
                   default="atoms+field")
    p.add_argument("--field-sign", choices=("neg", "both"), default="neg",
                   help="field arrows only where m is antiparallel to the Cr moment, or everywhere")
    p.add_argument("--colour", choices=("mz", "tilt"), default="mz")
    p.add_argument("--arrow-length", type=float, default=0.26, help="Blender units")
    p.add_argument("--centre", choices=("auto", "lattice", "patch", "cr"), default="auto",
                   help="a hexagon centre of the honeycomb, the patch middle, or one Cr "
                        "(auto: one Cr for the icon, a hexagon centre for the banner)")
    p.add_argument("--radius", type=float, default=None,
                   help="clip disc, bohr (default: --icon-radius or --banner-radius by camera)")
    p.add_argument("--icon-radius", type=float, default=6.5)
    p.add_argument("--banner-radius", type=float, default=11.5)
    p.add_argument("--text", type=int, default=None,
                   help="draw the wordmark (default: on the banner, not on the icon)")
    p.add_argument("--text-colour", choices=("dark", "light"), default="dark",
                   help="dark grey for a light background, light grey for a dark one")
    p.add_argument("--shell", type=int, default=1,
                   help="draw the faint low-level surface coloured by cos(m, z)")
    p.add_argument("--atom-arrow", type=float, default=2.4,
                   help="length of the atom arrows in units of --arrow-length, at the largest |m|")
    p.add_argument("--lobe-reach", type=float, default=None,
                   help="if > 0, keep the lobe surface within this distance of a drawn Cr "
                        "(default 0.4 for the icon, off for the banner)")
    p.add_argument("--cloud-reach", type=float, default=0.45,
                   help="keep the iodine cloud within this distance of a drawn iodine, Blender units")
    p.add_argument("--cr-radius", type=float, default=0.05, help="Cr sphere, Blender units")
    p.add_argument("--i-radius", type=float, default=0.075, help="I sphere, Blender units")
    p.add_argument("--elev-icon", type=float, default=55.0, help="camera elevation, degrees")
    p.add_argument("--elev-banner", type=float, default=50.0, help="camera elevation, degrees")
    p.add_argument("--engine", default="CYCLES")
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--size", type=int, default=600, help="the banner's height (a still: its side)")
    p.add_argument("--icon-size", type=int, default=1024, help="the icon's side in render")
    p.add_argument("--view", default="Standard")
    args = p.parse_args(argv)

    if args.verb == "build":
        scene = build_scene(resolve_kind(args, args.camera))
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.blend))
        print(f"saved {args.blend}")
    elif args.verb == "still":
        scene = build_scene(resolve_kind(args, args.camera))
        w = args.size * (4 if args.camera == "banner" else 1)
        render(scene, args, w, args.size, os.path.abspath(args.out))
    else:
        # the icon and the banner, each its own scene (its own centre and disc)
        os.makedirs(args.out, exist_ok=True)
        import copy
        for kind, name, w, h, colour in (
                ("icon", "defuqumat-icon.png", args.icon_size, args.icon_size, "dark"),
                ("banner", "defuqumat-banner.png", 4 * args.size, args.size, "dark"),
                ("banner", "defuqumat-banner-dark.png", 4 * args.size, args.size, "light")):
            a = resolve_kind(copy.copy(args), kind)
            a.text_colour = colour
            scene = build_scene(a)
            render(scene, a, w, h, os.path.abspath(os.path.join(args.out, name)))


if __name__ == "__main__":
    main()
