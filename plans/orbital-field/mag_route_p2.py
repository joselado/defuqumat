"""Route P (step 2) acceptance probes, one item per call. Run from the worktree root on D22.

    python mag_route_p2.py item1 [ecut_ry] [nbnd]        item 1: a = 0, q = 1 graphene, box against sphere
    python mag_route_p2.py si                            the projector part alone on two-atom silicon
    python mag_route_p2.py covariance                    Hermiticity, T(a1), time reversal, bare against dressed
    python mag_route_p2.py chern n                       item 2: Chern numbers at n_Phi = 1, kmesh (n, n)
    python mag_route_p2.py shift n                       item 5: the band shift, kmesh (n, n)
    python mag_route_p2.py sets q [outdir]               item 6: graphene full H at 1/q, kmesh (2, 2q)
    python mag_route_p2.py fold q [m] [conv_thr]         item 3: the SCF by the fold at 1/q, kmesh (2, q m)
    python mag_route_p2.py diamond p [ecut_ry] [n1,n2,n3]  item 4: one of E(+), E(0), E(-) on Lu's cell
    python mag_route_p2.py chi                           item 4: chi from the three diamond npz files

Every number printed is in Ry unless said otherwise.
"""
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from defuqumat import Calculator
from defuqumat.units import RY_TO_EV

SCF = "plans/orbital-field/graphene-scf.npz"


def graphene(ecut_ry=50.0, kmesh=(12, 12, 1), positions=None):
    atoms = [("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))] if positions is None else positions
    return Calculator.from_keywords(
        lattice="hexagonal", a=2.46, c=15.0, atoms=atoms,
        pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
        ecut=ecut_ry * RY_TO_EV, kmesh=kmesh, smearing=0.01 * RY_TO_EV, smearing_type="mv")


def silicon():
    return Calculator.from_keywords(
        lattice="fcc", a=10.2 * 0.529177210903,
        atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
        pseudos={"Si": "Si.pz-vbc.UPF"}, pseudo_dir="tests/data/pseudo",
        ecut=12 * RY_TO_EV, kmesh=(2, 2, 2))


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def projector_part(calc, potential, kcrys, nbnd, label):
    """Sphere Davidson of the full H at kcrys, then the box H's Rayleigh quotients on its states.

    The box at p = 0, q = 1 holds the sphere; on a sphere state the kinetic and the two-grid
    local term are exact Galerkin matrix elements, so the Rayleigh quotient against the sphere
    eigenvalue is the grid projectors' error alone, and becp is compared directly with vkb's.
    """
    import jax.numpy as jnp

    from defuqumat.magnetic.blochgrid import (
        MagneticGridHamiltonian, sphere_levels, sphere_state_on_box,
    )
    from defuqumat.system.kpoints import KPoints

    cell = calc.system.cell
    sub = calc.with_kpoints(KPoints.from_crystal(np.asarray([kcrys], float), np.ones(1), cell))
    calculation = sub.calculation
    t0 = time.perf_counter()
    es, vs, st, un = sphere_levels(calculation, potential, nbnd)
    print(f"[{label}] sphere full H at k = {kcrys}: steps {st}, unsettled {un}, "
          f"{time.perf_counter() - t0:.1f} s", flush=True)
    h = MagneticGridHamiltonian.build(cell, 0, 1, calc.system.ecutwfc, [kcrys], potential,
                                      structure=calc.system.structure, pseudos=calc.pseudos)
    pr = h.projectors
    print(f"[{label}] box {h.box}, fine {h.potential_fine.shape}, natom {pr.natom}, nproj {pr.nproj}, "
          f"points per atom {pr.index.shape[1]}, radius {pr.radius[0]:.3f} bohr, "
          f"dropped norm fraction {pr.dropped:.2e}", flush=True)
    u = sphere_state_on_box(calculation, vs[0], 0, h.box)
    norms = np.linalg.norm(u, axis=1)
    hu = np.asarray(h.apply(jnp.asarray(u), 0))
    rq = np.real(np.einsum("bi,bi->b", u.conj(), hu)) / norms ** 2
    print(f"[{label}] state norms on the box: {np.max(np.abs(norms - 1)):.2e} from one")
    x = h.interpolate(jnp.asarray(u).reshape((-1,) + h.box), h.kpoints[0])
    becp = np.asarray(pr.project(x.reshape(x.shape[0], -1), h.kpoints[0]))
    vkb_becp = np.asarray(calculation.projectors.project(jnp.asarray(vs[0]), 0))
    tau = np.asarray(calc.system.structure.positions)
    kc =2 * np.pi * np.asarray(kcrys) @ np.asarray(jnp.linalg.inv(cell.at).T)
    nper = pr.nproj
    grid_becp = (becp * np.exp(1j * (tau @ kc))[None, :, None]).reshape(becp.shape[0], -1)
    # vkb's channel order: atoms outermost, then that species' channels (same as here for one species)
    nch = vkb_becp.shape[1] // len(tau)
    grid_becp = grid_becp.reshape(becp.shape[0], len(tau), nper)[:, :, :nch].reshape(becp.shape[0], -1)
    db = np.abs(grid_becp - vkb_becp)
    dij = np.asarray(calculation.projectors.dij)
    e_vkb = np.real(np.einsum("bi,ij,bj->b", vkb_becp.conj(), dij, vkb_becp))
    e_grid = np.real(np.einsum("bi,ij,bj->b", grid_becp.conj(), dij, grid_becp))
    print(f"[{label}] max |becp_grid - becp_vkb| {db.max():.3e} (max |becp| {np.abs(vkb_becp).max():.3e})")
    for n in range(len(rq)):
        print(f"[{label}]   band {n}: sphere {es[0, n]: .10f}  box Rayleigh {rq[n]: .10f}  "
              f"diff {rq[n] - es[0, n]: .3e}   <V_NL> vkb {e_vkb[n]: .8f} grid {e_grid[n]: .8f} "
              f"diff {e_grid[n] - e_vkb[n]: .3e}", flush=True)
    return es[0], rq, e_vkb, e_grid


def item1(ecut=50.0, nbnd=8):
    from defuqumat.workflows.magnetic_bands import local_potential_of, run_magnetic_bands

    calc = graphene(ecut, kmesh=(1, 1, 1))
    pot = local_potential_of(SCF)
    es, rq, e_vkb, e_grid = projector_part(calc, pot, [0.0, 0.0, 0.0], nbnd, f"graphene {ecut} Ry")
    t0 = time.perf_counter()
    bands = run_magnetic_bands(calc.system, calc.pseudos, SCF, flux=(0, 1), kpoints=[[0, 0, 0]],
                               nbnd=nbnd, local_only=False)
    t = time.perf_counter() - t0
    eb = bands.eigenvalues[0]
    print(f"box full H at a = 0: steps {bands.steps}, unsettled {bands.unsettled}, {t:.1f} s, "
          f"solver {bands.solver}")
    for n in range(nbnd):
        print(f"  level {n}: box {eb[n]: .10f}  sphere {es[n]: .10f}  box - sphere {eb[n] - es[n]: .3e}")
    print(f"max |box - sphere| over {nbnd} levels: {np.max(np.abs(eb - es)):.3e} Ry; "
          f"projector part alone (Rayleigh) max {np.max(np.abs(rq - es)):.3e} Ry; peak {peak_gb():.2f} GB")


def si():
    calc = silicon()
    scf = calc.get_scf(conv_thr=1e-10)
    pot = np.asarray(scf.potential)[0]
    print(f"silicon scf: {scf.total_energy:.10f} Ry, {scf.iterations} iterations, grid {pot.shape}")
    worst = 0.0
    for k in ([0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.0, 0.25]):
        es, rq, e_vkb, e_grid = projector_part(calc, pot, k, 4, f"silicon k={k}")
        worst = max(worst, np.max(np.abs(rq - es)), np.max(np.abs(e_grid - e_vkb)))
    print(f"silicon: worst |<psi|V_NL|psi>_grid - vkb| or Rayleigh difference over 4 bands, 3 k: {worst:.3e} Ry")


def chern(n, nbnd=8):
    from defuqumat.workflows.magnetic_bands import _chern_numbers, level_groups, run_magnetic_bands

    calc = graphene(kmesh=(1, 1, 1))
    t0 = time.perf_counter()
    bands = run_magnetic_bands(calc.system, calc.pseudos, SCF, flux=(1, 1), kmesh=(n, n), nbnd=nbnd,
                               local_only=False, keep_states=True)
    print(f"n_Phi = 1, kmesh ({n},{n}): {bands.field_tesla:.1f} T, ndim {bands.ndim}, steps "
          f"{bands.steps.min()} to {bands.steps.max()}, solver {set(bands.solver)}, "
          f"{time.perf_counter() - t0:.1f} s", flush=True)
    e = bands.eigenvalues
    gaps = np.min(e[:, 1:] - e[:, :-1], axis=0)
    print("smallest gap above each band over the mesh (Ry):", " ".join(f"{g:.3e}" for g in gaps))
    print("band minima/maxima (Ry):", " ".join(f"[{e[:, i].min():.5f},{e[:, i].max():.5f}]" for i in range(nbnd)))
    groups = [(0, 0), (1, 1), (2, 2), (3, 3), (2, 3), (0, 3)] + level_groups(e, 1e-6)
    for first, last, c, flux in _chern_numbers(bands.states, (n, n), bands.box, groups):
        print(f"  bands {first} to {last}: C = {c:+.10f}, largest plaquette flux {flux:.3f}")
    np.savez(Path(OUT) / f"chern-{n}.npz", eigenvalues=e, kpoints=bands.kpoints)


def shift(n, nbnd=12):
    from defuqumat.magnetic.fold import magnetic_occupations
    from defuqumat.workflows.magnetic_bands import local_potential_of, run_magnetic_bands

    calc = graphene(kmesh=(1, 1, 1))
    a1 = np.asarray(calc.system.cell.at)[0]
    pos = np.asarray(calc.system.structure.positions)
    moved = calc.with_positions(pos + 0.5 * a1)
    frac = np.asarray(moved.system.structure.positions) @ np.linalg.inv(np.asarray(moved.system.cell.at))
    print(f"moved atoms, crystal: {np.round(frac, 6).tolist()}")
    pot = local_potential_of(SCF)
    rolled = np.roll(pot, pot.shape[0] // 2, axis=0)
    runs = {}
    for label, c, v, p in (("original", calc, pot, 1), ("moved", moved, rolled, 1),
                           ("original, -B", calc, pot, -1)):
        t0 = time.perf_counter()
        b = run_magnetic_bands(c.system, c.pseudos, {"potential": v[None]}, flux=(p, 1),
                               kmesh=(n, n), nbnd=nbnd, local_only=False)
        wg, ef, ts = magnetic_occupations(b.eigenvalues, np.full(len(b.kpoints), 2.0 / len(b.kpoints)),
                                          8.0, calc.system)
        eband = float(np.sum(wg * b.eigenvalues))
        runs[label] = (b, eband, ts)
        print(f"[{label}] p = {p}: steps {b.steps.min()} to {b.steps.max()}, eband {eband:.12f} Ry, "
              f"-TS {ts:.3e}, ef {ef:.6f}, {time.perf_counter() - t0:.1f} s", flush=True)
    b0, e0, _ = runs["original"]
    b1, e1, _ = runs["moved"]
    b2, e2, _ = runs["original, -B"]
    print(f"eps(k) at the same named k, original against moved: max |difference| "
          f"{np.max(np.abs(b0.eigenvalues - b1.eigenvalues)):.4e} Ry")
    print(f"sorted level set over the ({n},{n}) mesh, original against moved: "
          f"{np.max(np.abs(b0.sorted_levels - b1.sorted_levels)):.3e} Ry; band energy {abs(e0 - e1):.3e} Ry")
    print(f"time reversal on the grid: sorted set at +B against -B {np.max(np.abs(b0.sorted_levels - b2.sorted_levels)):.3e} Ry; "
          f"band energy {abs(e0 - e2):.3e} Ry")


def sets(q, outdir="plans/orbital-field"):
    from defuqumat.workflows.magnetic_bands import run_magnetic_bands

    calc = graphene(kmesh=(1, 1, 1))
    nbnd = 4 * q + 8
    t0 = time.perf_counter()
    b = run_magnetic_bands(calc.system, calc.pseudos, SCF, flux=(1, q), kmesh=(2, 2 * q), nbnd=nbnd,
                           local_only=False)
    wall = time.perf_counter() - t0
    rep = np.isin(np.round(b.kpoints[:, 1] * 2 * q).astype(int), [0, 1])
    copies = np.sort(np.tile(b.eigenvalues[rep].reshape(-1), q))
    print(f"q = {q}: ndim {b.ndim}, nk {len(b.kpoints)}, nbnd {nbnd}, steps {b.steps.min()} to "
          f"{b.steps.max()}, memory estimate {b.memory_estimate / 1e9:.3f} GB, solve {b.timings['solve_s']:.0f} s, "
          f"{np.mean(b.timings['per_k_s']):.1f} s a k-point, peak RSS {peak_gb():.2f} GB")
    print(f"q = {q}: full mesh against q copies of the representatives: "
          f"{np.max(np.abs(b.sorted_levels - copies)):.3e} Ry")
    np.savez(Path(outdir) / f"route-p-q{q}-full.npz", eigenvalues=b.eigenvalues, kpoints=b.kpoints,
             sorted_levels=b.sorted_levels, steps=b.steps, flux=np.array(b.flux), box=np.array(b.box),
             field_tesla=b.field_tesla, wall_s=wall,
             description=np.array("route P full H (GIPAW projectors), graphene 50 Ry, two-grid, "
                                  "centred window, shared zero-field potential; Ry"))


def fold(q, m=2, conv_thr=1e-8, check=True):
    from defuqumat.magnetic.fold import fold_check, run_magnetic_scf

    calc = graphene()
    nbnd = 4 * q + 8
    with np.load(SCF) as data:
        rho0 = np.asarray(data["density"])
    t0 = time.perf_counter()
    r = run_magnetic_scf(calc.system, calc.pseudos, SCF, flux=(1, q), kmesh=(2, q * m), nbnd=nbnd,
                         conv_thr=conv_thr)
    wall = time.perf_counter() - t0
    volume = float(calc.system.cell.volume)
    drho = r.density - rho0
    print(f"SCF 1/{q}, kmesh (2,{q * m}), {len(r.kpoints)} representatives: converged {r.converged} in "
          f"{r.iterations} iterations, dr2 {r.accuracy:.2e}, E {r.total_energy:.10f} Ry per chemical cell "
          f"(zero-field sphere SCF -22.664820820312105), {wall:.0f} s, memory estimate "
          f"{r.memory_estimate / 1e9:.3f} GB, peak RSS {peak_gb():.2f} GB", flush=True)
    print(f"terms: {json.dumps(r.energy_terms)}")
    print(f"density change from the zero-field one: max |drho| {np.max(np.abs(drho)):.3e} e/bohr^3 "
          f"(max rho {np.max(rho0):.3e}), int |drho| {np.sum(np.abs(drho)) * volume / drho[0].size:.4e} e "
          f"of {np.sum(rho0) * volume / rho0[0].size:.6f}")
    np.savez(Path(OUT) / f"scf-{q}.npz", density=r.density, potential=r.potential,
             eigenvalues=r.eigenvalues, kpoints=r.kpoints, total_energy=r.total_energy,
             history=np.array(json.dumps(r.history)))
    if check:
        density, levels, t = fold_check(r, calc.system, calc.pseudos)
        print(f"fold check: folded density of the representatives against the whole closed mesh, "
              f"max relative difference {density:.3e}; sorted levels (q copies) {levels:.3e} Ry; {t:.0f} s")


def diamond_calc(ecut_ry=60.0, kmesh=(2, 6, 6)):
    a = 3.56
    base = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
    sites = [(x, y, z) for x, y, z in base] + [(x + .25, y + .25, z + .25) for x, y, z in base]
    atoms = [("C", ((x + j) / 3.0, y, z)) for j in range(3) for x, y, z in sites]
    return Calculator.from_keywords(
        lattice=[[3 * a, 0, 0], [0, a, 0], [0, 0, a]], atoms=atoms,
        pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
        ecut=ecut_ry * RY_TO_EV, kmesh=kmesh)


def diamond0(ecut=60.0, kmesh=(2, 6, 6)):
    calc = diamond_calc(ecut, kmesh)
    t0 = time.perf_counter()
    scf = calc.get_scf(conv_thr=1e-11)
    print(f"diamond 3x1x1 zero-field sphere SCF: {scf.total_energy:.10f} Ry, {scf.iterations} iterations, "
          f"grid {np.asarray(scf.density).shape}, {time.perf_counter() - t0:.0f} s", flush=True)
    np.savez(Path(OUT) / f"diamond-scf-{int(ecut)}.npz", density=np.asarray(scf.density),
             potential=np.asarray(scf.potential), total_energy=scf.total_energy)


def diamond(p, ecut=60.0, kmesh=(2, 6, 6), nbnd=52):
    from defuqumat.magnetic.blochgrid import estimate_bytes, wavefunction_box
    from defuqumat.magnetic.fold import run_magnetic_scf

    calc = diamond_calc(ecut, kmesh)
    grid = wavefunction_box(calc.system.cell, calc.system.ecutwfc)
    nk = int(np.prod(kmesh))
    est = estimate_bytes(1, nbnd + 4, 4, grid=grid, kept=nk * (nbnd + 4), fine_grid=(36, 36, 100))
    print(f"diamond 3x1x1 at {ecut} Ry: box {grid}, ndim {int(np.prod(grid))}, {nk} k, nbnd {nbnd}: "
          f"memory estimate {est / 1e9:.2f} GB", flush=True)
    t0 = time.perf_counter()
    r = run_magnetic_scf(calc.system, calc.pseudos, Path(OUT) / f"diamond-scf-{int(ecut)}.npz",
                         flux=(p, 1), kmesh=kmesh, nbnd=nbnd, conv_thr=1e-8, calculation=calc.calculation)
    print(f"diamond p = {p}: {r.field_tesla:.1f} T, converged {r.converged} in {r.iterations} iterations, "
          f"dr2 {r.accuracy:.2e}, E {r.total_energy:.12f} Ry, {time.perf_counter() - t0:.0f} s, "
          f"peak RSS {peak_gb():.2f} GB")
    print(f"terms: {json.dumps(r.energy_terms)}")
    np.savez(Path(OUT) / f"diamond-{int(ecut)}-p{p}.npz", total_energy=r.total_energy,
             field_tesla=r.field_tesla, eigenvalues=r.eigenvalues, kpoints=r.kpoints,
             converged=r.converged, history=np.array(json.dumps(r.history)), natoms=24)


def chi(ecut=60.0):
    from defuqumat.magnetic.fold import molar_magnetizability

    e = {}
    for p in (0, 1, -1):
        with np.load(Path(OUT) / f"diamond-{int(ecut)}-p{p}.npz") as d:
            e[p] = float(d["total_energy"])
            if p == 1:
                b = float(d["field_tesla"])
    print(f"E(0) {e[0]:.12f}  E(+) {e[1]:.12f}  E(-) {e[-1]:.12f} Ry; E(+) - E(-) {e[1] - e[-1]:.3e} Ry; "
          f"E(+) - E(0) {e[1] - e[0]:.6e} Ry; dB {b:.2f} T")
    print(f"chi per C2 (12 primitive cells): {molar_magnetizability(e[1], e[0], e[-1], b, 12):.4f} "
          f"1e-6 cm^3/mol (Lu -11.16, LDA); per C atom {molar_magnetizability(e[1], e[0], e[-1], b, 24):.4f}")


OUT = "."

if __name__ == "__main__":
    import os

    OUT = os.environ.get("MAG_OUT", ".")
    what = sys.argv[1]
    args = sys.argv[2:]
    t0 = time.perf_counter()
    if what == "item1":
        item1(float(args[0]) if args else 50.0, int(args[1]) if len(args) > 1 else 8)
    elif what == "si":
        si()
    elif what == "chern":
        chern(int(args[0]) if args else 12)
    elif what == "shift":
        shift(int(args[0]) if args else 6)
    elif what == "sets":
        sets(int(args[0]), *(args[1:2]))
    elif what == "fold":
        fold(int(args[0]), int(args[1]) if len(args) > 1 else 2,
             float(args[2]) if len(args) > 2 else 1e-8, check=(len(args) <= 3 or args[3] != "nocheck"))
    elif what == "diamond0":
        diamond0(float(args[0]) if args else 60.0)
    elif what == "diamond":
        diamond(int(args[0]), float(args[1]) if len(args) > 1 else 60.0)
    elif what == "chi":
        chi(float(args[0]) if args else 60.0)
    else:
        raise SystemExit(f"unknown item {what}")
    print(f"# wall {time.perf_counter() - t0:.1f} s, peak RSS {peak_gb():.2f} GB")
