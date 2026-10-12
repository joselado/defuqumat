"""compute.py: the physics behind the package logo (plans/logo.md).

Two steps, so the SCF runs once, on D22's CPU, and the geometry is cut from its
saved density anywhere::

    python3 -u tools/logo/compute.py scf  --input docs/logo/cri3.in \\
        --pseudo-dir tests/data/pseudo --out docs/logo/data/cri3.npz   # on D22, 35 min
    python3 tools/logo/compute.py mesh --npz docs/logo/data/cri3.npz --out docs/logo/mesh

The defaults of ``mesh`` are the values that made the committed logo (an
isosurface of |m| at 10 per cent of its maximum, the antiparallel part of the
one at 0.4 per cent, a 2x refinement cropped to 6 angstrom either side of the Cr
plane, a 3x3 patch, field arrows on a 1.4 angstrom grid where |m| is above 0.4
per cent); run bare, it regenerates ``docs/logo/mesh/`` in about 20 s.

``scf`` runs the input through ``Calculator``, checks the geometry it parsed (the
Cr-I and Cr-Cr distances), prints the size estimate, runs the SCF, checks the
integrals of the four density channels (the valence count, the moment along
``z``) and saves the density ``(nspin_mag, n1, n2, n3)`` with the cell and the
atoms to an ``.npz``.

``mesh`` interpolates each channel to a finer grid exactly, by zero-padding in
reciprocal space (the density is a trigonometric polynomial on the ``ecutrho``
sphere), tiles it in the plane so the honeycomb shows as a patch, closes the
patch with one periodic layer, runs marching cubes on ``|m|`` (or on ``n``) and
writes a PLY with normals, a JSON with the atoms and bonds, and the arrows: ``m``
at a thinned subset of the surface vertices, at a coarse grid in the vacuum where
``|m|`` is above a floor, and one per atom from ``m`` integrated in a sphere. The
Blender script ``tools/logo/logo.py`` reads these files and nothing else.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def _minimum_image_distance(r_i, r_j, at):
    """Shortest |r_i - r_j| over the periodic images of the cell ``at`` (rows)."""
    best = np.inf
    for n1 in (-1, 0, 1):
        for n2 in (-1, 0, 1):
            for n3 in (-1, 0, 1):
                shift = n1 * at[0] + n2 * at[1] + n3 * at[2]
                best = min(best, float(np.linalg.norm(r_i - r_j + shift)))
    return best


def _bonds(positions, symbols, at, cutoff):
    """Pairs (i, j, image shift) closer than ``cutoff`` bohr, one entry per image."""
    out = []
    for i in range(len(positions)):
        for j in range(len(positions)):
            if symbols[i] == symbols[j]:
                continue
            for n1 in (-1, 0, 1):
                for n2 in (-1, 0, 1):
                    shift = n1 * at[0] + n2 * at[1]
                    d = float(np.linalg.norm(positions[i] - positions[j] - shift))
                    if d < cutoff:
                        out.append((i, j, (n1, n2, 0), d))
    return out


def _as_array(x):
    return np.asarray(x() if callable(x) else x, dtype=float)


# ---------------------------------------------------------------------------
# step 1: the SCF, on the CPU of D22
# ---------------------------------------------------------------------------

def run_scf(args):
    from defuqumat import Calculator
    from defuqumat.sizing import estimate_size

    calc = Calculator.from_file(args.input, pseudo_dir=args.pseudo_dir)
    system = calc.system
    at = _as_array(system.cell.at)
    if at.max() < 3.0:  # ``at`` in units of alat rather than bohr
        at = at * float(system.cell.alat)
    pos = _as_array(system.structure.positions)
    sym = tuple(system.structure.symbols() if callable(system.structure.symbols)
                else system.structure.symbols)
    print(f"cell (bohr):\n{at}")
    print(f"atoms: {sym}")

    # the geometry check: every I bridges two Cr at the same distance
    cr = [i for i, s in enumerate(sym) if s.startswith("Cr")]
    io = [i for i, s in enumerate(sym) if s.startswith("I")]
    for i in io:
        d = sorted(_minimum_image_distance(pos[i], pos[j], at) for j in cr)
        print(f"  I{i}: Cr neighbours at {d[0] * BOHR_TO_ANGSTROM:.3f} "
              f"and {d[1] * BOHR_TO_ANGSTROM:.3f} angstrom")
    if len(cr) == 2:
        d = _minimum_image_distance(pos[cr[0]], pos[cr[1]], at) * BOHR_TO_ANGSTROM
        print(f"  Cr-Cr: {d:.3f} angstrom")

    print(estimate_size(system, calc.pseudos))
    sys.stdout.flush()

    t0 = time.time()
    # one k-point in flight keeps the Davidson buffers under D22's 8 GB cap
    # (the estimate above sizes them for the whole axis); verbose prints the
    # iterations as they happen, so the log says whether the run is alive
    res = calc.get_scf(k_batch=args.k_batch, verbose=True)
    wall = time.time() - t0
    print(f"SCF: converged={res.converged} iterations={res.iterations} "
          f"wall={wall:.1f} s ({wall / max(res.iterations, 1):.1f} s per iteration, "
          f"compile included)")
    print(f"total energy {res.total_energy:.8f} Ry")
    print(f"magnetization_vector {res.magnetization_vector} "
          f"absolute {res.absolute_magnetization}")
    print(f"site_moments {res.site_moments}")

    rho = np.asarray(res.density, dtype=float)  # (nspin_mag, n1, n2, n3), e/bohr^3
    volume = float(_as_array(system.cell.volume))
    npts = rho[0].size
    integrals = rho.reshape(rho.shape[0], -1).sum(axis=1) * volume / npts
    valence = sum(float(p.z_valence if hasattr(p, "z_valence") else p.zp)
                  for p in (calc.pseudos[t] for t in system.structure.types))
    print(f"density shape {rho.shape}; integrals over the cell: {integrals}; "
          f"valence electrons in the cell: {valence:.1f}")
    if rho.shape[0] != 4:
        raise SystemExit("the logo wants the four channels (n, m_x, m_y, m_z)")
    if abs(integrals[0] - valence) > 1e-3 * valence:
        raise SystemExit("channel 0 does not integrate to the valence count; "
                         "the layout or the normalisation was read wrong")

    site_moments = (np.asarray(res.site_moments, dtype=float)
                    if res.site_moments is not None else np.zeros((len(sym), 3)))
    mag_vec = (np.asarray(res.magnetization_vector, dtype=float)
               if res.magnetization_vector is not None else np.zeros(3))
    np.savez(args.out, density=rho, at=at, positions=pos, symbols=np.array(sym),
             integrals=integrals, total_energy=res.total_energy,
             iterations=res.iterations, wall=wall, site_moments=site_moments,
             magnetization_vector=mag_vec,
             absolute_magnetization=float(res.absolute_magnetization or 0.0),
             eigenvalues=np.asarray(res.eigenvalues, dtype=float),
             fermi_energy=float(res.fermi_energy or 0.0))
    print(f"saved {args.out}")
    print("LOGO_SCF_DONE")


# ---------------------------------------------------------------------------
# step 2: the surface and the arrows, anywhere with scikit-image
# ---------------------------------------------------------------------------

def _fourier_refine(field, factor):
    """``field`` on an (n1, n2, n3) periodic grid, interpolated exactly to
    (factor n1, factor n2, factor n3) by zero-padding its Fourier coefficients."""
    n = np.array(field.shape)
    m = n * factor
    coeff = np.fft.fftn(field)
    out = np.zeros(m, dtype=complex)
    # copy each coefficient to the same (signed) index of the larger grid
    for axis_slices in _signed_blocks(n, m):
        src, dst = axis_slices
        out[dst] = coeff[src]
    return np.fft.ifftn(out).real * (m.prod() / n.prod())


def _signed_blocks(n, m):
    """The eight (or fewer) corner blocks that map the signed frequencies of an
    n-grid into an m-grid, as (source, destination) slice tuples."""
    halves = []
    for ni in n:
        lo = (ni + 1) // 2  # non-negative frequencies 0 .. lo-1
        hi = ni - lo        # negative frequencies -hi .. -1
        parts = [(slice(0, lo), slice(0, lo))]
        if hi:
            parts.append((slice(ni - hi, ni), slice(-hi, None)))
        halves.append(parts)
    for a in halves[0]:
        for b in halves[1]:
            for c in halves[2]:
                yield ((a[0], b[0], c[0]), (a[1], b[1], c[1]))


def _write_ply(path, verts, normals, faces, colours=None):
    """ASCII PLY with normals and, if given, uchar vertex colours, which Blender's
    importer turns into the colour attribute ``Col``."""
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property float nx\nproperty float ny\nproperty float nz\n")
        if colours is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        if colours is None:
            colours = [None] * len(verts)
        for v, nrm, col in zip(verts, normals, colours):
            f.write(f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f} "
                    f"{nrm[0]:.4f} {nrm[1]:.4f} {nrm[2]:.4f}"
                    + ("" if col is None else f" {col[0]} {col[1]} {col[2]}") + "\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")


def _thin(points, spacing):
    """A greedy subset of ``points`` no two of which are closer than ``spacing``."""
    keep = []
    cells = {}
    inv = 1.0 / spacing
    for idx, p in enumerate(points):
        key = tuple(np.floor(p * inv).astype(int))
        close = False
        for d1 in (-1, 0, 1):
            for d2 in (-1, 0, 1):
                for d3 in (-1, 0, 1):
                    for q in cells.get((key[0] + d1, key[1] + d2, key[2] + d3), ()):
                        if np.linalg.norm(points[q] - p) < spacing:
                            close = True
                            break
                    if close:
                        break
                if close:
                    break
            if close:
                break
        if not close:
            keep.append(idx)
            cells.setdefault(key, []).append(idx)
    return np.array(keep, dtype=int)


def _contour(field, level, N, z0, z1, at, tile):
    """Marching cubes on one cell of ``field`` (the fine grid, cropped to the
    planes ``z0:z1``), closed with one wrapped layer in the plane, then the mesh
    translated ``tile x tile`` times; returns cartesian vertices, normals, faces
    and the fractional coordinates of the vertices (for sampling the field)."""
    from skimage import measure
    cell = field[:, :, z0:z1]
    closed = np.pad(cell, ((0, 1), (0, 1), (0, 0)), mode="wrap")
    verts_idx, faces, normals_idx, _ = measure.marching_cubes(closed, level=level)
    frac = verts_idx / N + np.array([0.0, 0.0, z0 / N[2]])
    B = at / N[:, None]                      # index -> cartesian
    normals = normals_idx @ np.linalg.inv(B).T
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    v_all, n_all, f_all, fr_all = [], [], [], []
    offset = 0
    for i1 in range(tile):
        for i2 in range(tile):
            shift = np.array([i1, i2, 0.0])
            fr = frac + shift
            v_all.append(fr @ at)
            n_all.append(normals)
            f_all.append(faces + offset)
            fr_all.append(fr)
            offset += len(frac)
    return (np.concatenate(v_all), np.concatenate(n_all), np.concatenate(f_all),
            np.concatenate(fr_all))


def run_mesh(args):
    from scipy.ndimage import map_coordinates

    data = np.load(args.npz)
    rho, at = data["density"], data["at"]
    pos, sym = data["positions"], [str(s) for s in data["symbols"]]
    site_moments = data["site_moments"] if "site_moments" in data else None

    # exact refinement of each channel, then |m|; the crop to the layer keeps
    # the arrays at a few hundred MB (the reviewer's sizing: a 3x3 tiling of a
    # refined dense grid would be tens of GB, so the tiling is of the mesh)
    fine = np.stack([_fourier_refine(c, args.fine) for c in rho])
    N = np.array(fine.shape[1:])
    n_f, m_f = fine[0], fine[1:]
    mag_f = np.linalg.norm(m_f, axis=0)
    c_len = np.linalg.norm(at[2])
    half = args.zcrop / BOHR_TO_ANGSTROM / c_len
    z0 = max(int(np.floor((0.5 - half) * N[2])), 0)
    z1 = min(int(np.ceil((0.5 + half) * N[2])) + 1, N[2])
    print(f"grid {rho.shape[1:]} -> {tuple(int(x) for x in N)}, z planes {z0}:{z1} kept; "
          f"max n {n_f.max():.4f}, max |m| {mag_f.max():.4f} e/bohr^3")
    mz_frac_outside = mag_f[:, :, :z0].max() if z0 > 0 else 0.0
    print(f"  largest |m| outside the crop: {max(mz_frac_outside, mag_f[:, :, z1:].max() if z1 < N[2] else 0):.2e}")

    def sample(channels, points_frac):
        coords = (points_frac % 1.0).T * N[:, None]
        return np.stack([map_coordinates(c, coords, order=1, mode="wrap")
                         for c in channels]).T

    def cosine_colour(points_frac):
        m = sample(m_f, points_frac)
        cos = m[:, 2] / (np.linalg.norm(m, axis=1) + 1e-12)
        w = 0.5 * (1 + cos)
        lo, hi = np.array([25, 90, 230]), np.array([240, 40, 25])
        return (lo * (1 - w[:, None]) + hi * w[:, None]).astype(np.uint8), cos

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t = args.tile
    field = n_f if args.surface == "n" else mag_f
    surfaces = [(args.surface, args.level)]
    if args.level2 > 0:
        surfaces.append((args.surface + "_low", args.level2))
    main_verts = main_frac = None
    for name, level in surfaces:
        verts, normals, faces, frac = _contour(field, level * field.max(), N, z0, z1, at, t)
        colours, cos = cosine_colour(frac)
        if name.endswith("_low") and args.shell_sign == "neg":
            # keep the antiparallel part alone: the induced moment on the iodine,
            # which is the magnetic content; the parallel part is a larger copy
            # of the Cr lobes and would hide them
            keep_face = (cos[faces] < 0).all(axis=1)
            used = np.unique(faces[keep_face])
            renumber = -np.ones(len(verts), dtype=int)
            renumber[used] = np.arange(len(used))
            verts, normals, colours = verts[used], normals[used], colours[used]
            faces = renumber[faces[keep_face]]
            frac, cos = frac[used], cos[used]
        _write_ply(out / f"surface_{name}.ply", verts, normals, faces, colours)
        print(f"surface {name} at {level:.3f} of the maximum: {len(verts)} vertices, "
              f"{len(faces)} faces (tiled {t}x{t}); cos(m, z) on it from "
              f"{cos.min():.3f} to {cos.max():.3f}")
        if main_verts is None:
            main_verts, main_frac = verts, frac

    # arrows on the main surface: a thinned subset of its vertices
    keep = _thin(main_verts, args.arrow_spacing / BOHR_TO_ANGSTROM)
    surf_pts, surf_m = main_verts[keep], sample(m_f, main_frac[keep])

    # arrows on a coarse grid in and around the layer where |m| is above a floor
    step = args.grid_spacing / BOHR_TO_ANGSTROM
    g1 = np.arange(0, t, step / np.linalg.norm(at[0]))
    g2 = np.arange(0, t, step / np.linalg.norm(at[1]))
    g3 = np.arange(z0 / N[2], z1 / N[2], step / c_len)
    grid_frac = np.array(np.meshgrid(g1, g2, g3, indexing="ij")).reshape(3, -1).T
    grid_m = sample(m_f, grid_frac)
    strong = np.linalg.norm(grid_m, axis=1) > args.floor * mag_f.max()
    grid_pts, grid_m = grid_frac[strong] @ at, grid_m[strong]

    # one arrow per atom: the result's own site moments (report_mag's spheres),
    # with a sphere integral on the fine grid printed beside them as a check
    dv = abs(np.linalg.det(at)) / n_f.size
    sl = m_f[:, :, :, z0:z1]
    idx = np.indices(sl.shape[1:]).reshape(3, -1).T + np.array([0, 0, z0])
    cart = (idx / N) @ at
    inv_at = np.linalg.inv(at)
    sphere = []
    for p, s in zip(pos, sym):
        d = cart - p
        d -= np.round(d @ inv_at) @ at
        radius = (args.radius_cr if s.startswith("Cr") else args.radius_i) / BOHR_TO_ANGSTROM
        inside = np.linalg.norm(d, axis=1) < radius
        sphere.append(sl.reshape(3, -1)[:, inside].sum(axis=1) * dv)
    sphere = np.array(sphere)
    moments = site_moments if site_moments is not None and site_moments.any() else sphere
    for s, ms, mr in zip(sym, sphere, moments):
        print(f"  {s}: sphere integral {ms.round(3)}, site moment used {mr.round(3)} mu_B")
    atom_pts, atom_m, atom_sym = [], [], []
    for i1 in range(t):
        for i2 in range(t):
            for p, s, mm in zip(pos, sym, moments):
                atom_pts.append(p + i1 * at[0] + i2 * at[1])
                atom_m.append(mm)
                atom_sym.append(s)
    np.savez(out / "arrows.npz", surface_points=surf_pts, surface_m=surf_m,
             grid_points=grid_pts, grid_m=grid_m, atom_points=np.array(atom_pts),
             atom_m=np.array(atom_m), atom_symbols=np.array(atom_sym))
    print(f"arrows: {len(surf_pts)} on the surface, {len(grid_pts)} in the field, "
          f"{len(atom_pts)} on atoms")

    # atoms and bonds of the patch, in bohr, for Blender
    bonds = _bonds(pos, sym, at, args.bond_cutoff / BOHR_TO_ANGSTROM)
    atoms = []
    for i1 in range(t):
        for i2 in range(t):
            for k, (p, s) in enumerate(zip(pos, sym)):
                atoms.append({"index": k, "tile": [i1, i2], "symbol": s,
                              "position": (p + i1 * at[0] + i2 * at[1]).tolist()})
    with open(out / "atoms.json", "w") as f:
        json.dump({"at": at.tolist(), "tile": t, "bohr_to_angstrom": BOHR_TO_ANGSTROM,
                   "atoms": atoms,
                   "bonds": [{"i": i, "j": j, "shift": list(sh), "d": d}
                             for i, j, sh, d in bonds],
                   "surface": args.surface, "level": args.level, "level2": args.level2,
                   "max_n": float(n_f.max()), "max_m": float(mag_f.max()),
                   "site_moments": np.asarray(moments).tolist()}, f, indent=1)
    print(f"wrote {out}/")


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="step", required=True)
    s = sub.add_parser("scf", help="run the SCF and save the density")
    s.add_argument("--input", required=True)
    s.add_argument("--pseudo-dir", required=True)
    s.add_argument("--out", default="cri3.npz")
    s.add_argument("--k-batch", type=int, default=1, help="k-points in flight at once")
    s.set_defaults(func=run_scf)
    g = sub.add_parser("mesh", help="cut the surface and the arrows from a saved density")
    g.add_argument("--npz", required=True)
    g.add_argument("--out", default="docs/logo/mesh")
    g.add_argument("--surface", choices=("m", "n"), default="m",
                   help="isosurface of |m| (default) or of the total density n")
    g.add_argument("--level", type=float, default=0.10,
                   help="isolevel as a fraction of the field's maximum")
    g.add_argument("--level2", type=float, default=0.004,
                   help="a second, faint isosurface at this fraction (0 for none), "
                        "coloured by cos(m, z): the iodine's antiparallel moment")
    g.add_argument("--shell-sign", choices=("neg", "both"), default="neg",
                   help="the second surface keeps only its antiparallel part (neg) or all of it")
    g.add_argument("--fine", type=int, default=2, help="refinement factor of the grid")
    g.add_argument("--zcrop", type=float, default=6.0,
                   help="keep the density within this distance of the Cr plane, angstrom")
    g.add_argument("--tile", type=int, default=3, help="cells per side of the patch")
    g.add_argument("--arrow-spacing", type=float, default=1.5,
                   help="minimum distance between surface arrows, angstrom")
    g.add_argument("--grid-spacing", type=float, default=1.4,
                   help="spacing of the field arrows, angstrom")
    g.add_argument("--floor", type=float, default=0.004,
                   help="field arrows only where |m| exceeds this fraction of its maximum")
    g.add_argument("--radius-cr", type=float, default=1.2, help="sphere for the Cr moment, angstrom")
    g.add_argument("--radius-i", type=float, default=1.2, help="sphere for the I moment, angstrom")
    g.add_argument("--bond-cutoff", type=float, default=3.0, help="Cr-I bond cutoff, angstrom")
    g.set_defaults(func=run_mesh)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
