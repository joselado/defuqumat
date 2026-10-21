"""I117: a noncollinear seed on a compensated (antiferromagnetic) reference.

Unit cell: two hydrogens 5.5 bohr apart along x in a (11, 5.5, 5.5) bohr box,
noncollinear, starting moments +z and -z (two species on one UPF), which
converges to the collinear antiferromagnet. Ultracell (n, 1, 1), kgrid (1, 2, 2),
seeded with the Neel vector turning 360/n per cell about z,
s(x) = (cos 2 pi x/n, sin 2 pi x/n, 0), on the Kramers-closed default basis.
Reference: the explicit 2n-atom supercell seeded the same way per species
(atom A of cell c at angle2 = 360 c/n, atom B at 360 (c + 1/2)/n + 180, both
angle1 = 90), which is what R(+z -> s) does to +z and -z sublattices.

Compared: the energy per unit cell and the site moments (the magnetization
integrated over a sphere of radius R_SITE about each atom, the same function on
both boxes), which are sign-sensitive where the Fourier norm is not. The
control is the same ultracell seeded with -s (the other sign convention of the
Neel axis): same energy, site moments reversed.

The driver still calls ``seed.reference_axis``, which refuses a compensated
reference (the item is partial), so this harness puts ``seed.neel_axis`` in its
place, with the unit cell's atoms' grid points as the sites, which is what the
lifted driver did (commit 8c7bf8e5).

Usage: i117.py <out.json> [n] [nbnd list] [control: yes|no] [supercell nbnd]
"""
import json, sys, time, warnings
from pathlib import Path
import tempfile

import numpy as np
import jax

from defuqumat import Calculator
from defuqumat.ultracell import run_ultracell
import defuqumat.ultracell.driver as ultracell_driver
from defuqumat.ultracell.seed import neel_axis as reference_axis, SEED_COLLINEAR_TOL

out = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
NBNDS = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "8,16").split(",")]
CONTROL = (sys.argv[4] if len(sys.argv) > 4 else "yes") == "yes"
SUPER_NBND = int(sys.argv[5]) if len(sys.argv) > 5 else 12
A = 5.5
R_SITE = 1.5
KGRID = (1, 2, 2)
PSEUDO = str(Path("tests/data/pseudo").resolve())

TEMPLATE = """&control
 calculation='scf'
/
&system
 ibrav=0, celldm(1)=5.5, nat={nat}, ntyp={nat}, ecutwfc=15.0,
 nosym=.true., noinv=.true.,
 noncolin=.true.,
{magnetism}
 occupations='smearing', smearing='gaussian', degauss=0.02
/
&electrons
 conv_thr=1.0d-11
 mixing_beta=0.3
/
CELL_PARAMETERS alat
 {length:.10f} 0.0 0.0
 0.0 1.0 0.0
 0.0 0.0 1.0
ATOMIC_SPECIES
{species}
ATOMIC_POSITIONS crystal
{atoms}
K_POINTS automatic
 {k0} {k1} {k2} 0 0 0
"""


def write(tmp, tag, angles, kgrid):
    """``angles`` is one (angle1, angle2) per atom, atoms evenly along x."""
    nat = len(angles)
    names = [f"H{i + 1}" for i in range(nat)]
    species = "\n".join(f" {name} 1.008 H.pz-vbc.UPF" for name in names)
    atoms = "\n".join(" %s %.10f 0.0 0.0" % (names[i], i / nat) for i in range(nat))
    magnetism = "\n".join(
        f" starting_magnetization({i + 1})=0.8, angle1({i + 1})={a1:.8f}, "
        f"angle2({i + 1})={a2:.8f}," for i, (a1, a2) in enumerate(angles))
    path = tmp / f"{tag}.in"
    path.write_text(TEMPLATE.format(nat=nat, magnetism=magnetism, length=float(nat),
                                    species=species, atoms=atoms, k0=kgrid[0],
                                    k1=kgrid[1], k2=kgrid[2]))
    return Calculator.from_file(path, pseudo_dir=PSEUDO, announce=False)


def site_moments(magnetization, lengths, sites):
    """``(nsite, 3)``: m integrated over a sphere about each site (cartesian bohr)."""
    m = np.asarray(magnetization)
    grid = m.shape[1:]
    axes = [np.arange(g) / g * L for g, L in zip(grid, lengths)]
    r = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    dv = np.prod(lengths) / np.prod(grid)
    out = []
    for site in sites:
        d = r - np.asarray(site)
        d -= np.round(d / lengths) * lengths
        mask = (d ** 2).sum(axis=-1) < R_SITE ** 2
        out.append((m[:, mask].sum(axis=1)) * dv)
    return np.asarray(out)


def fingerprint(moments):
    """In-plane angle of each site, its cone off the plane, and successive angles."""
    m = np.asarray(moments)
    planar = np.degrees(np.arctan2(m[:, 1], m[:, 0]))
    cone = np.degrees(np.arcsin(np.clip(m[:, 2] / np.linalg.norm(m, axis=1), -1, 1)))
    unit = m / np.linalg.norm(m, axis=1)[:, None]
    steps = np.degrees(np.arccos(np.clip(
        (unit * np.roll(unit, -1, axis=0)).sum(axis=1), -1, 1)))
    return planar.tolist(), cone.tolist(), steps.tolist()


tmp = Path(tempfile.mkdtemp())
t0 = time.time()
records = dict(n=N, kgrid=KGRID, r_site=R_SITE, runs=[])
shape = (N, 1, 1)
folded = tuple(a * b for a, b in zip(shape, KGRID))
unit = write(tmp, "afm_unit", [(0.0, 0.0), (180.0, 0.0)], folded)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scf = unit.get_scf(conv_thr=1e-11, nbnd=8, max_iterations=300)
rho = np.asarray(scf.density)
flat = rho[1:].reshape(3, -1)
values = np.linalg.eigvalsh(flat @ flat.T)
volume = float(unit.system.cell.volume)
net = np.linalg.norm(flat.sum(axis=1)) * volume / flat.shape[1]
absolute = np.sqrt((flat ** 2).sum(axis=0)).sum() * volume / flat.shape[1]
cell_grid = np.asarray(rho.shape[1:])
sites_idx = (np.round(np.asarray(unit.system.structure.positions_crystal(unit.system.cell))
                      * cell_grid) % cell_grid).astype(int)
axis = reference_axis(rho[1:], volume, sites=sites_idx)
ultracell_driver.reference_axis = (
    lambda m, v, sites=None: reference_axis(m, v, sites=sites_idx))
records["reference"] = dict(
    converged=bool(scf.converged), iterations=int(scf.iterations),
    total_energy=float(scf.total_energy), net=float(net), absolute=float(absolute),
    lambda_ratio=float(values[1] / values[2]), tol=SEED_COLLINEAR_TOL,
    axis=axis.tolist(), seconds=time.time() - t0)
print(json.dumps(records["reference"]), flush=True)
if records["reference"]["lambda_ratio"] > SEED_COLLINEAR_TOL or not scf.converged:
    with open(out, "w") as handle:
        json.dump(records, handle, indent=1)
    sys.exit("the reference is not a converged collinear antiferromagnet")

# The supercell: 2N atoms, A of cell c at x = c (unit-cell crystal), B at c + 1/2.
angles = []
for c in range(N):
    angles.append((90.0, 360.0 * c / N))
    angles.append((90.0, 360.0 * (c + 0.5) / N + 180.0))
t1 = time.time()
supercell = write(tmp, "afm_super", angles, KGRID)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    exact = supercell.get_scf(conv_thr=1e-11, nbnd=SUPER_NBND, max_iterations=400)
lengths = np.array([2 * A * N, A, A])
sites = [np.array([A * j, 0.0, 0.0]) for j in range(2 * N)]
theirs = site_moments(np.asarray(exact.density)[1:], lengths, sites)
exact_energy = float(exact.total_energy) / N
records["supercell"] = dict(
    converged=bool(exact.converged), iterations=int(exact.iterations),
    energy=exact_energy, sites=theirs.tolist(), fingerprint=fingerprint(theirs),
    seconds=time.time() - t1)
print(json.dumps(records["supercell"]), flush=True)
jax.clear_caches()


def helix(x, sign=1.0):
    phase = 2 * np.pi * x[..., 0] / N
    return sign * np.stack([np.cos(phase), np.sin(phase), np.zeros_like(phase)], axis=-1)


arms = [(nbnd, 1.0) for nbnd in NBNDS]
if CONTROL:
    arms.append((NBNDS[0], -1.0))
for nbnd, sign in arms:
    t2 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = run_ultracell(
            unit.system, unit.pseudos, scf, shape, KGRID, nbnd=nbnd,
            seed_magnetization=lambda x, s=sign: helix(x, s), conv_thr=1e-10,
            states_conv_thr=1e-8, mixing_beta=0.3, max_iterations=300)
    ours = site_moments(np.asarray(result.magnetization), lengths, sites)
    scale = np.linalg.norm(theirs, axis=1).max()
    rec = dict(nbnd=nbnd, sign=sign, converged=bool(result.converged),
               iterations=int(result.iterations), energy=float(result.total_energy),
               above=float(result.total_energy) - exact_energy,
               site_error=float(np.abs(ours - theirs).max() / scale),
               site_error_reversed=float(np.abs(ours + theirs).max() / scale),
               sites=ours.tolist(), fingerprint=fingerprint(ours),
               net=np.asarray(result.cell_moments()).sum(axis=0).tolist(),
               kramers=bool(result.kramers_ranks is not None),
               seconds=time.time() - t2)
    records["runs"].append(rec)
    print(json.dumps(rec), flush=True)
    with open(out, "w") as handle:
        json.dump(records, handle, indent=1)
    jax.clear_caches()
records["seconds"] = time.time() - t0
with open(out, "w") as handle:
    json.dump(records, handle, indent=1)
print("# done", records["seconds"])
