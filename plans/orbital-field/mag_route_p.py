"""Route P (step 1) acceptance probes, one item per call. Run from the worktree root on D22.

    python mag_route_p.py landau p q n1 n2 nbnd [two-gauge|kinetic|identity] [chern|keep]   item 1
    python mag_route_p.py herm                                               item 2
    python mag_route_p.py sphere ecut_ry nbnd                                item 3
    python mag_route_p.py strong                                             item 4
    python mag_route_p.py local q [outdir]                                   items 5 and 6

Every number printed is in Ry unless said otherwise.
"""
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np

from defuqumat import Calculator
from defuqumat.magnetic.blochgrid import (
    MagneticGridHamiltonian, chemical_copies, dense_matrix, estimate_bytes, landau_problem,
    reference_matrix, sphere_local_levels, wavefunction_box,
)
from defuqumat.magnetic.conventions import MagneticCell
from defuqumat.magnetic.landau import free_landau_levels
from defuqumat.units import RY_TO_EV, TPI
from defuqumat.workflows.magnetic_bands import local_potential_of, run_magnetic_bands

SCF = "plans/orbital-field/graphene-scf.npz"


def graphene(ecut_ry=50.0, kmesh=(12, 12, 1)):
    return Calculator.from_keywords(
        lattice="hexagonal", a=2.46, c=15.0,
        atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
        pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
        ecut=ecut_ry * RY_TO_EV, kmesh=kmesh, smearing=0.01 * RY_TO_EV, smearing_type="mv")


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def landau(p, q, n1, n2, nbnd, precond="two-gauge", chern=False, unconverged="fallback"):
    calc = graphene()
    system = calc.system
    t0 = time.perf_counter()
    bands = run_magnetic_bands(system, calc.pseudos, None, flux=(p, q), kmesh=(n1, n2), nbnd=nbnd,
                               free=True, chern=chern, preconditioner=precond, unconverged=unconverged)
    wall = time.perf_counter() - t0
    mcell = MagneticCell.build(system.cell, p, q)
    levels, mult, labels = free_landau_levels(mcell, nmax=12, g3max=3)
    ref = np.repeat(levels, p)[:nbnd]
    lab = np.repeat(labels, p, axis=0)[:nbnd]
    e = bands.eigenvalues
    print(f"grid {bands.grid} box {bands.box} ndim {bands.ndim} B_ph {bands.b_ph!r} "
          f"({bands.field_tesla:.1f} T) memory estimate {bands.memory_estimate / 1e9:.3f} GB")
    print("reference (n, m, E):", [(int(a), int(b), round(float(x), 10)) for (a, b), x in zip(lab, ref)])
    print("levels at k0:", np.round(e[0], 12).tolist())
    dev = np.abs(e - ref[None, :])
    print(f"max |E - (2n+1) B_ph - (2 pi m/c)^2| = {dev.max():.3e} Ry (per band {np.round(dev.max(0), 13).tolist()})")
    spread = e.max(0) - e.min(0)
    print(f"flatness: max over bands of max_k - min_k = {spread.max():.3e} Ry")
    print(f"Davidson [{precond}] steps per k {bands.steps.tolist()} unsettled {bands.unsettled.tolist()} "
          f"solver {sorted(set(bands.solver))}")
    if bands.chern is not None:
        print("chern groups (first, last, C, max plaquette flux):", bands.chern)
        for first, last, c, mf in bands.chern:
            print(f"  bands {first}-{last} labels {lab[first:last + 1].tolist()} C = {c:.12f}")
    print(f"timings {json.dumps({k: v for k, v in bands.timings.items() if k != 'per_k_s'})} "
          f"wall {wall:.1f} s peak RSS {peak_gb():.2f} GB")


def herm():
    calc = graphene()
    cell = calc.system.cell
    pot = local_potential_of(SCF)
    rng = np.random.default_rng(7)
    for q, grid in ((10, None), (2, (4, 4, 8))):
        h = MagneticGridHamiltonian.build(cell, 1, q, 50.0, [[0.37, 0.29, 0.0], [0.0, 0.0, 0.0]], pot,
                                          grid=grid)
        n = h.ndim
        v = rng.normal(size=(2, n)) + 1j * rng.normal(size=(2, n))
        import jax.numpy as jnp
        hv = np.asarray(h.apply(jnp.asarray(v), 0))
        a = np.vdot(v[0], hv[1])
        b = np.vdot(hv[0], v[1])
        scale = np.linalg.norm(v[0]) * np.linalg.norm(hv[1])
        print(f"q = {q} box {h.box} ndim {n}: |<v|Hw> - <Hv|w>| = {abs(a - b):.3e}, relative {abs(a - b) / scale:.3e}")
        if n <= 1024:
            for ik in (0, 1):
                m = dense_matrix(h, ik)
                r = reference_matrix(h, ik)
                x = rng.normal(size=n) + 1j * rng.normal(size=n)
                applied = np.asarray(h.apply(jnp.asarray(x[None]), ik))[0]
                print(f"  ik {ik}: max|apply-built - Kronecker reference| / max|ref| = "
                      f"{np.abs(m - r).max() / np.abs(r).max():.3e}; |H x - R x|/|R x| = "
                      f"{np.linalg.norm(applied - r @ x) / np.linalg.norm(r @ x):.3e}; "
                      f"max|R - R^H|/max|R| = {np.abs(r - r.conj().T).max() / np.abs(r).max():.3e}")


def sphere(ecut, nbnd):
    from defuqumat.system.kpoints import KPoints

    calc = graphene(ecut_ry=ecut, kmesh=(1, 1, 1))
    cell = calc.system.cell
    pot = local_potential_of(SCF)
    gamma = calc.with_kpoints(KPoints.from_crystal(np.zeros((1, 3)), np.ones(1), cell))
    calculation = gamma.calculation
    t0 = time.perf_counter()
    es, st, un = sphere_local_levels(calculation, pot, nbnd)
    ts = time.perf_counter() - t0
    t0 = time.perf_counter()
    bands = run_magnetic_bands(calc.system, calc.pseudos, SCF, flux=(0, 1), kpoints=[[0, 0, 0]], nbnd=nbnd)
    tb = time.perf_counter() - t0
    eb = bands.eigenvalues[0]
    print(f"ecut {ecut} Ry: sphere npwx {calculation.basis.planewaves.npwx} dense grid "
          f"{calculation.basis.dense.grid}; box {bands.box} ndim {bands.ndim}")
    print("sphere:", np.round(es[0], 10).tolist(), f"steps {st.tolist()} unsettled {un.tolist()} {ts:.1f} s")
    print("box   :", np.round(eb, 10).tolist(), f"steps {bands.steps.tolist()} {tb:.1f} s")
    d = eb - es[0]
    print(f"box - sphere per level: {np.round(d, 10).tolist()}; max |diff| {np.abs(d).max():.3e} Ry")


def strong():
    calc = graphene()
    q = 3
    t0 = time.perf_counter()
    bands = run_magnetic_bands(calc.system, calc.pseudos, SCF, flux=(1, q), kmesh=(2, 6), nbnd=4 * q + 8,
                               keep_states=True)
    print(f"solve {time.perf_counter() - t0:.1f} s steps {bands.steps.tolist()}")
    e = bands.eigenvalues.reshape(2, 6, -1)
    worst = 0.0
    for i in range(2):
        for j in range(6):
            jj = (j - 2) % 6          # kappa2 - 1/3 = kappa2 - 2/6
            worst = max(worst, np.abs(e[i, j] - e[i, jj]).max())
    print(f"max over the mesh |eps(kappa2) - eps(kappa2 - 1/3)| = {worst:.3e} Ry")
    nocc = 4 * q
    gaps = bands.eigenvalues[:, nocc] - bands.eigenvalues[:, nocc - 1]
    print(f"gap above band {nocc} per k: min {gaps.min():.4e} Ry")
    # a smooth weight of a fixed mu between the occupied and empty levels, so that a
    # multiplet at the cut weighs the same at k and at its partner
    mu = 0.5 * (bands.eigenvalues[:, nocc - 1].max() + bands.eigenvalues[:, nocc].min())
    sigma = 0.01
    from scipy.special import erfc
    weights = 0.5 * erfc((bands.eigenvalues - mu) / sigma)
    rho = np.einsum("kn,knr->r", weights, np.abs(bands.states) ** 2).reshape(bands.box)
    copies = chemical_copies(rho, q)
    mean = copies.mean(axis=0)
    spread = np.abs(copies - mean[None]).max() / np.abs(mean).max()
    print(f"mu {mu:.6f} Ry sigma {sigma}; density over the closed mesh, spread over the {q} copies "
          f"(max |copy - mean| / max mean) = {spread:.3e}")
    # the open-mesh contrast: only the representatives kappa2 in {0, 1/6}
    rep = np.array([[i * 6 + j for j in (0, 1)] for i in range(2)]).reshape(-1)
    rho_open = np.einsum("kn,knr->r", weights[rep], np.abs(bands.states[rep]) ** 2).reshape(bands.box)
    c2 = chemical_copies(rho_open, q)
    print(f"same at the representatives alone (an open mesh): {np.abs(c2 - c2.mean(0)[None]).max() / np.abs(c2.mean(0)).max():.3e}")


def local(q, outdir="plans/orbital-field"):
    calc = graphene()
    nbnd = 4 * q + 8
    t0 = time.perf_counter()
    bands = run_magnetic_bands(calc.system, calc.pseudos, SCF, flux=(1, q), kmesh=(2, 2 * q), nbnd=nbnd)
    wall = time.perf_counter() - t0
    e = bands.eigenvalues.reshape(2, 2 * q, nbnd)
    rep = e[:, :2, :].reshape(-1)
    full = np.sort(bands.eigenvalues.reshape(-1))
    copies = np.sort(np.repeat(rep, q))
    diff = np.abs(full - copies).max()
    print(f"q = {q}: box {bands.box} ndim {bands.ndim} nbnd {nbnd} nk {len(bands.kpoints)}; "
          f"full-mesh set against q copies of the representatives' set: {diff:.3e} Ry")
    print(f"steps per k {bands.steps.tolist()} unsettled {bands.unsettled.tolist()} solver {sorted(set(bands.solver))}")
    print(f"lowest levels at k0 (Ry): {np.round(bands.eigenvalues[0, :8], 8).tolist()}")
    print(f"memory estimate {bands.memory_estimate / 1e9:.3f} GB, peak RSS {peak_gb():.3f} GB, wall {wall:.1f} s, "
          f"build {bands.timings['build_s']:.2f} s, solve {bands.timings['solve_s']:.1f} s, "
          f"first k {bands.timings['per_k_s'][0]:.1f} s, later k mean {np.mean(bands.timings['per_k_s'][1:]):.1f} s")
    out = Path(outdir) / f"route-p-q{q}-local.npz"
    np.savez(out, sorted_levels=full, eigenvalues=bands.eigenvalues, kpoints=bands.kpoints,
             flux=np.array(bands.flux), box=np.array(bands.box), grid=np.array(bands.grid),
             representatives_sorted=np.sort(rep), steps=bands.steps,
             memory_estimate=np.int64(bands.memory_estimate), wall_s=np.float64(wall),
             description=np.array("route P step 1: pi^2 + V_loc of graphene-scf.npz, flux 1/q per "
                                  "chemical cell, magnetic mesh (2, 2q) kappa = (i/2, j/(2q)), Ry"))
    print(f"saved {out}")


if __name__ == "__main__":
    what = sys.argv[1]
    a = sys.argv[2:]
    if what == "landau":
        landau(int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]),
               a[5] if len(a) > 5 else "two-gauge", len(a) > 6 and a[6] == "chern",
               "keep" if len(a) > 6 and a[6] == "keep" else "fallback")
    elif what == "herm":
        herm()
    elif what == "sphere":
        sphere(float(a[0]), int(a[1]))
    elif what == "strong":
        strong()
    elif what == "local":
        local(int(a[0]), *a[1:])
    else:
        raise SystemExit(__doc__)
