"""logo.py: the package logo, built in Blender from compute.py's files and rendered headless.

    B=/u/40/ladovj1/unix/apps/blender-5.2.2-linux-x64/blender
    $B -b --factory-startup --python tools/logo/logo.py -- render --mesh docs/logo/mesh \\
        --out docs/logo                 # the four committed PNGs, Cycles, about 4 min
    $B -b --factory-startup --python tools/logo/logo.py -- still --mesh docs/logo/mesh \\
        --out draft.png --camera icon --engine BLENDER_EEVEE --samples 16   # a draft, 2 s
    $B -b --factory-startup --python tools/logo/logo.py -- build --mesh docs/logo/mesh \\
        --blend logo.blend              # the scene, to open by hand

``render`` writes ``defuqumat-icon.png`` (``--icon-size``, 1024) and
``defuqumat-banner.png`` (four times ``--size`` by ``--size``, 2400 by 600); the 256 and
64 px icons are pillow downsamples of the first. The defaults of every dial are the
values that made the committed logo, so a bare ``render`` regenerates it.

The scene is a deterministic function of the files in ``--mesh`` (``surface_m.ply``,
``surface_m_low.ply``, ``arrows.npz``, ``atoms.json``, written by ``compute.py mesh``)
and of the dials below: the Cr t2g shell as an isosurface of |m| of the CrI3
monolayer (plans/logo.md), the Cr moment as one arrow, on each iodine a
translucent cloud of the antiparallel density, a field of direction arrows where
``m`` leans more than ``--field-tilt`` degrees off the axis (the region around each
iodine where it turns over), the atoms and the Cr to I bonds, and the wordmark.
The icon is one CrI6 octahedron (centred on a Cr, a disc of 6.5 bohr, no wordmark,
since at 64 px the name is noise). The banner is a strip of the honeycomb with the
wordmark in front of it, pearl bevelled type lit by the scene and the honeycomb
softened by the camera's depth of field: the line "density functional quantum
materials" on one baseline, each word's first letters at the full size and the rest
at half of it in italic, so the big letters read ``defuqumat`` through the line (``--wordmark acronym``; ``plain``
is the name alone; ``--banner-layout beside`` puts the ring of six to the left
instead). One Blender unit is ten bohr.

Dials: ``--camera icon|banner``, ``--arrows atoms+field|all|atoms|field|surface|none``
with ``--field-sign``, ``--field-style direction|magnitude``, ``--field-tilt``,
``--field-thin``, ``--colour mz|tilt``, ``--text 0|1``, ``--text-style glossy|metal|flat``,
``--font`` and ``--font-small`` (presets or files), ``--wordmark``, ``--banner-layout``,
``--dof`` (f-stop), ``--shell 0|1``, ``--surface m|n``, ``--centre``, ``--radius``,
``--cr-radius``, ``--elev-icon``, ``--elev-banner``, ``--cloud-reach``, ``--lobe-reach``,
``--engine``, ``--samples``, ``--view Standard|AgX``, ``--size``, ``--icon-size``.
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
FONTS = {
    "dejavu": "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "garamond": "/usr/share/fonts/truetype/ebgaramond/EBGaramond12-Regular.ttf",
    "garamond-italic": "/usr/share/fonts/truetype/ebgaramond/EBGaramond12-Italic.ttf",
    "garamond-bold": "/usr/share/fonts/truetype/ebgaramond/EBGaramond12-Bold.ttf",
    "gentium-italic": "/usr/share/fonts/truetype/gentiumplus/GentiumBookPlus-BoldItalic.ttf",
    "besley": "/usr/share/fonts/truetype/besley/Besley-Bold.ttf",
    "comfortaa": "/usr/share/fonts/truetype/comfortaa/Comfortaa-Bold.ttf",
}
FONT = FONTS["garamond-italic"]

PALETTE = {
    "surface": (0.93, 0.30, 0.06, 1.0),   # the lobes of |m|: warm orange
    "surface_n": (0.75, 0.78, 0.85, 1.0),  # the total density, if drawn: pale
    "Cr": (0.20, 0.22, 0.30, 1.0),         # dark steel
    "I": (0.55, 0.30, 0.75, 1.0),          # iodine violet
    "bond": (0.45, 0.45, 0.50, 1.0),
    "shell": (0.25, 0.45, 0.95, 1.0),      # the antiparallel cloud on the iodine
    "text": (0.10, 0.10, 0.12, 1.0),       # the wordmark when flat (an emission colour)
    "text_glossy": (0.93, 0.93, 0.96, 1.0), # the wordmark as pearl, lit by the scene
    "arrow_mid": (0.86, 0.86, 0.88, 1.0),  # an arrow in the plane
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


def arrow_mesh(length, shaft=0.011, head=0.030, head_frac=0.34, segments=24):
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


def _thin_points(points, spacing):
    """A greedy subset of ``points`` no two of which are closer than ``spacing``."""
    keep, cells = [], {}
    inv = 1.0 / spacing
    for idx, p in enumerate(points):
        key = tuple(np.floor(p * inv).astype(int))
        close = any(np.linalg.norm(points[q] - p) < spacing
                    for d1 in (-1, 0, 1) for d2 in (-1, 0, 1) for d3 in (-1, 0, 1)
                    for q in cells.get((key[0] + d1, key[1] + d2, key[2] + d3), ()))
        if not close:
            keep.append(idx)
            cells.setdefault(key, []).append(idx)
    return np.array(keep, dtype=int)


def add_arrows(arrows, centre, radius, which, colour_by, length_scale, atom_scale=2.4,
               field_sign="neg", field_style="direction", i_arrow=0.6, field_thin=0.0,
               field_tilt=15.0):
    groups = []
    if which in ("all", "surface"):
        groups.append(("surface", arrows["surface_points"], arrows["surface_m"], 1.0))
    if which in ("all", "field", "atoms+field"):
        pts, m = arrows["grid_points"], arrows["grid_m"]
        if field_tilt > 0:
            # only where m leans away from the axis by more than field_tilt
            # degrees: the arrows that show the turning, not the parallel ones
            nrm = np.linalg.norm(m, axis=1) + 1e-30
            lean = np.degrees(np.arccos(np.clip(np.abs(m[:, 2]) / nrm, 0, 1)))
            pts, m = pts[lean > field_tilt], m[lean > field_tilt]
        if field_thin > 0:
            keep = _thin_points((np.asarray(pts) - centre) * SCALE, field_thin)
            pts, m = pts[keep], m[keep]
        if field_sign == "neg":
            # only where m points against the Cr moment
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
    lo, mid, hi = (np.array(PALETTE["arrow_lo"]), np.array(PALETTE["arrow_mid"]),
                   np.array(PALETTE["arrow_hi"]))
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
            if name == "field":
                # a direction field: every arrow the same length, so the eye
                # reads where m turns rather than how small it is there
                length = length_scale * (0.55 if field_style == "direction"
                                         else 0.5 + 0.5 * nrm / m_max)
            elif name == "atoms":
                is_cr = nrm > 0.5 * m_max
                length = (length_scale * atom_scale * nrm / m_max if is_cr
                          else length_scale * i_arrow)   # the iodine's at a fixed length
            else:
                length = length_scale * (0.5 + 0.5 * nrm / m_max)
            v, f = arrow_mesh(length)
            R = rotation_to(mv)
            v = v @ R.T + p
            if colour_by == "tilt":
                tilt = math.degrees(math.acos(min(1.0, abs(mv[2]) / nrm)))
                w = tilt / tilt_max
            else:
                w = 0.5 * (1 + mv[2] / nrm)
            col = (lo * (1 - 2 * w) + mid * 2 * w) if w < 0.5 else (mid * (2 - 2 * w) + hi * (2 * w - 1))
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
                   material(f"arrows_{name}", (1, 1, 1, 1), roughness=0.65,
                            metallic=0.0, vertex_color="col"))   # matte, not mirror-like
        print(f"  arrows {name}: {len(pts)} inside the disc")


def text_material(style, size_hint=1.0):
    if style == "flat":
        mat = material("text", (0, 0, 0, 1), roughness=1.0)
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        _set_input(bsdf, "Emission Color", PALETTE["text"])   # flat, lit by itself
        _set_input(bsdf, "Emission Strength", 1.0)
        return mat
    # pearl letters with a bevelled edge, lit by the scene's lights like
    # everything else in it, so they belong to the picture
    mat = material("text", PALETTE["text_glossy"], roughness=0.22,
                   metallic=0.15 if style == "glossy" else 0.9)
    _set_input(mat.node_tree.nodes["Principled BSDF"], "Coat Weight", 1.0)
    return mat


def text_curve(name, body, size, font, style, align="LEFT"):
    curve = bpy.data.curves.new(name, type="FONT")
    curve.body = body
    curve.size = size
    curve.align_x = align
    curve.align_y = "CENTER"
    if os.path.exists(font):
        curve.font = bpy.data.fonts.load(font, check_existing=True)
    if style == "flat":
        curve.extrude, curve.bevel_depth = 0.02 * size, 0.004 * size
    else:
        curve.extrude, curve.bevel_depth, curve.bevel_resolution = 0.07 * size, 0.018 * size, 4
    return curve


def text_width(body, size, font):
    """The advance width of ``body`` at ``size`` in ``font``, from a throwaway
    text object's evaluated bounding box."""
    curve = text_curve("measure", body, size, font, "flat")
    ob = bpy.data.objects.new("measure", curve)
    bpy.context.scene.collection.objects.link(ob)
    bpy.context.view_layer.update()
    width = ob.dimensions.x
    bpy.data.objects.remove(ob)
    bpy.data.curves.remove(curve)
    return width


# the name read as its words: one line, each word's first letters big and the
# rest small, so the big letters spell the name
WORDS = [("de", "nsity", "up"), ("fu", "nctional", "down"),
         ("qu", "antum", "up"), ("mat", "erials", "down")]


def add_text(text, camera, distance, offset, size, style="glossy", align="LEFT",
             wordmark="acronym", font_small=None):
    cam_matrix = Matrix.Translation(camera.location) @ camera.rotation_euler.to_matrix().to_4x4()
    root = bpy.data.objects.new("wordmark", None)
    bpy.context.scene.collection.objects.link(root)
    root.rotation_euler = camera.rotation_euler
    # billboard: the camera's orientation, placed in its frame (the camera's
    # matrix_world is not evaluated yet in a scene built this way, so it is
    # assembled from the location and the rotation set above)
    root.location = cam_matrix @ Vector((offset[0], offset[1], -distance))
    mat = text_material(style)
    if wordmark == "plain":
        ob = new_object("wordmark_text", text_curve("wordmark", text, size, FONT, style, align), mat)
        ob.parent = root
        return root
    small_font = font_small or FONT
    small = 0.50 * size
    gap = 0.22 * size
    # one line, "density functional quantum materials", each word's first
    # letters big and the rest small on the same baseline, so the big letters
    # read "defuqumat" through the line
    pieces, x = [], 0.0
    for word, rest, _ in WORDS:
        pieces.append((word, size, FONT, x)); x += text_width(word, size, FONT) + 0.02 * size
        pieces.append((rest, small, small_font, x)); x += text_width(rest, small, small_font) + gap
    total = x - gap
    x0 = -total / 2 if align == "CENTER" else 0.0
    for i, (body, sz, font, xi) in enumerate(pieces):
        curve = text_curve(f"wordmark_{i}", body, sz, font, style)
        curve.align_y = "BOTTOM_BASELINE"
        ob = new_object(f"wordmark_{i}", curve, mat)
        ob.parent = root
        ob.location = (x0 + xi, -0.3 * size, 0.0)
    return root


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


def add_camera(kind, extent, elev_icon=58.0, elev_banner=40.0, with_text=True,
               layout="behind"):
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
    elif layout == "behind":
        # the disc fills the width, the camera sees a strip of it behind the letters
        w = 2.0 * extent
        h = w / 4
        elev, shift = math.radians(elev_banner), 0.0
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


def place_text(cam, dist, frame, kind, extent, style="glossy", layout="behind",
               wordmark="acronym", font_small=None):
    """The wordmark in the camera's frame: under the structure for the icon;
    for the banner either in the right three quarters (``beside``) or across the
    middle in front of the honeycomb (``behind``), a little nearer the camera
    than the structure so it stands in front of it."""
    w, h = frame
    big = 0.46 * h if wordmark == "acronym" else 0.62 * h
    if kind == "banner" and layout == "behind":
        add_text("defuqumat", cam, dist - 1.2 * extent, (0.0, -0.06 * h), big,
                 style, align="CENTER", wordmark=wordmark, font_small=font_small)
    elif kind == "banner":
        add_text("defuqumat", cam, dist, (-w / 2 + 1.06 * h, -0.08 * h), 0.36 * h, style,
                 wordmark=wordmark, font_small=font_small)
    else:
        add_text("defuqumat", cam, dist, (-0.5 * w + 0.06 * h, -0.5 * h + 0.14 * h),
                 0.15 * h, style, wordmark=wordmark, font_small=font_small)


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
        args.centre = ("cr" if kind == "icon"
                       else "patch" if args.banner_layout == "behind" else "lattice")
    if args.radius is None:
        args.radius = args.icon_radius if kind == "icon" else args.banner_radius
    if kind == "banner" and args.banner_layout == "behind" and args.centre == "patch":
        pass
    if args.lobe_reach is None:
        args.lobe_reach = 0.4 if kind == "icon" else 0.0
    if args.field_thin is None:
        args.field_thin = 0.0 if kind == "icon" else 0.22
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
               args.atom_arrow, args.field_sign, args.field_style, args.i_arrow,
               args.field_thin, args.field_tilt)
    add_lights_and_world()
    extent = args.radius * SCALE
    cam, dist, frame = add_camera(args.camera, extent, args.elev_icon, args.elev_banner,
                                  bool(args.text), args.banner_layout)
    if args.dof > 0 and args.camera == "banner" and args.banner_layout == "behind":
        # focus on the wordmark's plane, in front of the structure, and let the
        # honeycomb soften behind it
        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance = dist - 1.2 * extent
        cam.data.dof.aperture_fstop = args.dof
    if args.text:
        global FONT
        FONT = FONTS.get(args.font, args.font)
        PALETTE["text"] = {"dark": (0.10, 0.10, 0.12, 1.0),
                           "light": (0.88, 0.88, 0.90, 1.0)}[args.text_colour]
        place_text(cam, dist, frame, args.camera, extent, args.text_style, args.banner_layout,
                   args.wordmark, FONTS.get(args.font_small, args.font_small))
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
    p.add_argument("--field-sign", choices=("neg", "both"), default="both",
                   help="field arrows only where m is antiparallel to the Cr moment, or everywhere")
    p.add_argument("--field-style", choices=("direction", "magnitude"), default="direction",
                   help="field arrows all one length (the direction of m) or scaled by |m|")
    p.add_argument("--field-thin", type=float, default=None,
                   help="minimum distance between field arrows, Blender units "
                        "(default none for the icon, 0.22 for the banner)")
    p.add_argument("--field-tilt", type=float, default=15.0,
                   help="draw a field arrow only where m leans more than this from z, degrees")
    p.add_argument("--dof", type=float, default=2.4,
                   help="f-stop of the banner camera's depth of field, 0 for none")
    p.add_argument("--i-arrow", type=float, default=0.6,
                   help="length of an iodine's atom arrow in units of --arrow-length")
    p.add_argument("--font", default="garamond",
                   help="the big letters: a preset (%s) or a font file" % ", ".join(FONTS))
    p.add_argument("--font-small", default="garamond-italic",
                   help="the words' completions, a preset or a font file")
    p.add_argument("--wordmark", choices=("acronym", "plain"), default="plain",
                   help="the name with its words' completions above and below, or alone")
    p.add_argument("--text-style", choices=("glossy", "metal", "flat"), default="glossy")
    p.add_argument("--banner-layout", choices=("behind", "beside"), default="behind",
                   help="the wordmark in front of the honeycomb, or beside it")
    p.add_argument("--colour", choices=("mz", "tilt"), default="mz")
    p.add_argument("--arrow-length", type=float, default=0.26, help="Blender units")
    p.add_argument("--centre", choices=("auto", "lattice", "patch", "cr"), default="auto",
                   help="a hexagon centre of the honeycomb, the patch middle, or one Cr "
                        "(auto: one Cr for the icon, a hexagon centre for the banner)")
    p.add_argument("--radius", type=float, default=None,
                   help="clip disc, bohr (default: --icon-radius or --banner-radius by camera)")
    p.add_argument("--icon-radius", type=float, default=6.5)
    p.add_argument("--banner-radius", type=float, default=18.0,
                   help="bohr; 18 for the behind layout (the 3x3 patch is 39 wide), 11.5 beside")
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
        for kind, name, w, h in (
                ("icon", "defuqumat-icon.png", args.icon_size, args.icon_size),
                ("banner", "defuqumat-banner.png", 4 * args.size, args.size)):
            a = resolve_kind(copy.copy(args), kind)
            scene = build_scene(a)
            render(scene, a, w, h, os.path.abspath(os.path.join(args.out, name)))


if __name__ == "__main__":
    main()
