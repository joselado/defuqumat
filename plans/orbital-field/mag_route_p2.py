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


if __name__ == "__main__":
    what = sys.argv[1]
    args = sys.argv[2:]
    t0 = time.perf_counter()
    if what == "item1":
        item1(float(args[0]) if args else 50.0, int(args[1]) if len(args) > 1 else 8)
    elif what == "si":
        si()
    else:
        raise SystemExit(f"unknown item {what}")
    print(f"# wall {time.perf_counter() - t0:.1f} s, peak RSS {peak_gb():.2f} GB")
