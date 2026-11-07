"""Route U with a projected trial basis on the free Landau problem (scratch probe, fable).

The hybrid states are replaced by the projected trials P_N g_j: at every k of the N1 x q
x nk set the lowest N free bands are solved, and the basis function of trial j is
phi_j(k) = sum_m psi_m <psi_m|g_j> (no Loewdin step, the overlap S carries the
non-orthogonality; the span is nested in J by construction).  g_j is the Hermite-Gaussian
H_j(xi1/sigma) exp(-xi1^2/2 sigma^2) in the crystal coordinate xi1 = b1 . r (cells) times
the plane wave of the reduced k (G2 = G3 = 0).  Everything after hybrid_set is the
committed code: MagneticUltracell.build with h0 = "box", form = "symmetric".

Usage: probe_projected.py --N 16 --sigma 0.5 --J 1,2,3 --N1 10 [--window 3,4] [--scale -1]
"""
import argparse
import json
import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from numpy.polynomial.hermite import hermval

from defuqumat import Calculator
from defuqumat.units import RY_TO_EV
from defuqumat.magnetic.hybrid import TransportGauge, free_electron, hybrid_set
from defuqumat.magnetic.ultracell_field import MagneticUltracell, assemble, hermiticity, solve
from defuqumat.scf.driver import Calculation
from defuqumat.system.kpoints import KPoints, for_spin as kpoints_for_spin
from defuqumat.topology.states import build_plane_wave_states
from defuqumat.workflows.nscf import fixed_density_ethr, fixed_density_hamiltonians
from defuqumat.workflows.landau_ultracell import ultracell_kpoints

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=int, default=1)
ap.add_argument("--q", type=int, default=10)
ap.add_argument("--N", type=int, default=16)
ap.add_argument("--sigma", type=float, default=0.5)
ap.add_argument("--J", default="1,2,3")
ap.add_argument("--N1", type=int, default=10)
ap.add_argument("--kmesh", default="2,2")
ap.add_argument("--window", default=None)
ap.add_argument("--scale", type=float, default=1.0)
ap.add_argument("--ecut", type=float, default=50.0)
ap.add_argument("--c", type=float, default=15.0)
ap.add_argument("--chirp", action="store_true", help="dress the j=0 trial with the gauge phase exp(-i c1 (b1.b2)/(2|b1|^2) xi1^2) of the analytic Landau state (u1 0b)")
ap.add_argument("--conv", type=float, default=1e-10)
ap.add_argument("--drop", type=float, default=1e-8)
ap.add_argument("--tol", type=float, default=1e-8)
args = ap.parse_args()
np.set_printoptions(linewidth=160)

calc = Calculator.from_keywords(
    lattice="hexagonal", a=2.46, c=args.c,
    atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
    pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=args.ecut * RY_TO_EV, kmesh=(12, 12, 1), smearing=0.01 * RY_TO_EV, smearing_type="mv")
system, pseudos = calc.system, calc.pseudos
p, q, N, N1 = args.p, args.q, args.N, args.N1
n1m, n2m = (int(x) for x in args.kmesh.split(","))
kappas = [j / (n2m * q) for j in range(n2m)]
window = None if args.window is None else tuple(float(x) for x in args.window.split(","))

# --- the zero-field (free) states, N bands ---------------------------------------------
t0 = time.perf_counter()
points = ultracell_kpoints(q, N1, kappas)
cell = system.cell
kpoints = KPoints.from_crystal(points, np.ones(len(points)), cell)
system_k = eqx.tree_at(lambda s: s.kpoints, system, kpoints_for_spin(kpoints, system.nspin))
calculation = Calculation(system_k, pseudos).for_bands(N)
density = jnp.full((1,) + tuple(calculation.basis.dense.grid),
                   float(calculation.nelec) / float(cell.volume), dtype=cell.precision.real)
hamiltonians = tuple(free_electron(h) for h in fixed_density_hamiltonians(calculation, density))
ethr = fixed_density_ethr(args.conv, calculation.nelec)
eigenvalues, wavefunctions, steps, notcnv = calculation.diagonalize(hamiltonians, N, None, ethr,
                                                                    return_steps=True)
eigenvalues = np.asarray(eigenvalues[0])
wfc = wavefunctions[0]
jax.block_until_ready(wfc)
states = build_plane_wave_states(calculation, wfc)
print(f"NSCF {len(points)} k, N {N}: {time.perf_counter() - t0:.1f} s, davidson steps max "
      f"{int(np.max(steps))}, not converged {int(np.sum(notcnv))}", flush=True)

from defuqumat.magnetic.conventions import MagneticCell
if p == 0:
    b_ph = 1.0
    print('zero flux: levels printed in Ry minus (2 pi kappa2 |b2|)^2, n<6', flush=True)
else:
    mc = MagneticCell.build(cell, p, q, scale=args.scale)
    b_ph = float(mc.b_ph)
    print(f"B_ph {b_ph!r} Ry, field {mc.field_tesla:.1f} T", flush=True)


def trial_coefficients(k, J, sigma):
    """(J, npwx): the projected trial's coefficients before projection, on the sphere of k."""
    miller = np.asarray(states.miller[k])
    valid = np.asarray(states.valid[k])
    k1 = points[k, 0]
    kap = 2.0 * np.pi * (k1 + miller[:, 0])          # xi1-momentum in 1/cells
    # the lowest free band's stripe at the reduced kappa': G2 = -round(kappa'), so that the
    # trial is periodic in kappa' as a crystal's Bloch sum of a localized orbital is (the code
    # reuses the reduced state for X and X +- q with the Miller relabelling; a G2 = 0 stripe at
    # every reduced point is not periodic and cuts the lattice at the magnetic cell boundary)
    k2 = points[k, 1]
    g2 = 0 if (k2 % 1.0) < 0.5 else -1
    stripe = valid & (miller[:, 1] == g2) & (miller[:, 2] == 0)
    out = np.zeros((J, miller.shape[0]), dtype=complex)
    bsc = np.linalg.inv(np.asarray(cell.at, dtype=float)).T
    fcoef = (2.0 * np.pi * p / q) * args.scale * (bsc[0] @ bsc[1]) / (bsc[0] @ bsc[0]) / 2.0
    for j in range(J):
        c = np.zeros(j + 1); c[j] = 1.0
        if args.chirp and j == 0:
            a = 1.0 / (2.0 * sigma ** 2) + 1j * fcoef      # exp(-a xi^2) = Gaussian times exp(-i f)
            g = np.exp(-kap ** 2 / (4.0 * a))
        else:
            g = (-1j) ** j * hermval(sigma * kap, c) * np.exp(-0.5 * sigma ** 2 * kap ** 2)
        out[j] = np.where(stripe, g, 0.0)
    return out


def projected_gauge(sl, J, sigma):
    """TransportGauge whose rotation (N1, N, J) is A_i = <psi_m|g_j> at each k of the loop."""
    rot = []
    for k in range(sl.start, sl.stop):
        psi = np.asarray(states.coefficients[k])[:N]
        g = trial_coefficients(k, J, sigma)
        a = psi.conj() @ g.T                      # (N, J)
        rot.append(a)
    rot = np.stack(rot)
    # the weight of each trial outside the window, 1 - |A|^2/|g|^2, the jump size
    outside = []
    for k in range(sl.start, sl.stop):
        g = trial_coefficients(k, J, sigma)
        outside.append(1.0 - np.sum(np.abs(rot[k - sl.start]) ** 2, axis=0) / np.sum(np.abs(g) ** 2, axis=1))
    return TransportGauge(rotation=rot, phases=np.zeros(J), centres=np.zeros(J), singular=1.0), np.max(outside)


ref = (2 * np.arange(8) + 1) * b_ph
for J in [int(x) for x in args.J.split(",")]:
    t0 = time.perf_counter()
    hybrids, worst = [], 0.0
    for jk in range(n2m):
        row = []
        for m in range(q):
            sl = slice((jk * q + m) * N1, (jk * q + m + 1) * N1)
            gauge, outside = projected_gauge(sl, J, args.sigma)
            worst = max(worst, float(outside))
            row.append(hybrid_set(wfc[sl], eigenvalues[sl], states.miller[sl], states.valid[sl],
                                  gauge, kappas[jk] + m / q))
        hybrids.append(row)
    ultra = MagneticUltracell.build(cell, (p, q), hybrids, kappas, hamiltonians[0].grid, nbnd=J, n1=N1,
                                    potential=np.zeros(hamiltonians[0].grid), h0="box", form="symmetric",
                                    scale=args.scale, window=window)
    w_measured, weights, widths = ultra.spread(args.tol)
    cap = (N1 - 2) // 2
    if window is not None:
        w = ultra.w
    else:
        w = cap if w_measured is None else min(w_measured, cap)
    ultra = ultra.with_width(int(w))
    outside_by_width = np.max(weights.reshape(-1, len(widths)), axis=0)
    print(f"\n== J {J} sigma {args.sigma} chirp {args.chirp} N {N} N1 {N1} window {window} scale {args.scale}: w {w} measured "
          f"{w_measured}; trial weight outside the N-band window (max over k) {worst:.3e}", flush=True)
    print("outside weight by width:", np.array2string(outside_by_width, precision=3))
    rows = []
    for jk in range(n2m):
        out = ultra.blocks(jk)
        B, S = np.asarray(out[0]), np.asarray(out[1])
        B_out = out[2]
        dropped = float("nan") if B_out is None else float(np.max(np.abs(np.asarray(B_out))) / np.max(np.abs(B[:, w])))
        if J == 1:
            for X in (0, 1):
                print(f"blocks kappa2 {kappas[jk]:.3f} bra X={X}: Delta -2..2  B/|b1|^2:",
                      np.array2string(B[X, int(w) - 2:int(w) + 3, 0, 0] / 0.061703, precision=5),
                      " S:", np.array2string(S[X, int(w) - 2:int(w) + 3, 0, 0], precision=5), flush=True)
        for i1 in range(n1m):
            kappa1 = i1 / n1m
            h, s = assemble(kappa1, B, S, int(w), q)
            herm = hermiticity(h)
            eps, rank, d_min, d_all = solve(h, s, args.drop)
            nl = min(6, len(eps))
            err = eps[:nl] / b_ph - (2 * np.arange(nl) + 1) if p else eps[:nl] - (2*np.pi*kappas[jk])**2 * float(np.linalg.inv(np.asarray(cell.at, dtype=float)).T[1] @ np.linalg.inv(np.asarray(cell.at, dtype=float)).T[1])
            rows.append((kappa1, kappas[jk], herm, rank, d_all, dropped, err))
            print(f"k ({kappa1:.2f}, {kappas[jk]:.3f}) herm {herm:.2e} rank {rank}/{q * J} d_all {d_all:.2e} "
                  f"dropped {dropped:.1e}  (E/B_ph - (2n+1)), n<{nl}: {np.array2string(err, precision=2, formatter={'float_kind': lambda x: f'{x:+.2e}'})}",
                  flush=True)
    errs = np.array([r[6][:min(3, J * q)] for r in rows])
    print(f"J {J}: max |herm| {max(r[2] for r in rows):.2e}; (n, m=0) n<3 error over k, in Ry: "
          f"min {np.min(errs) * b_ph:+.3e} max {np.max(errs) * b_ph:+.3e}; wall {time.perf_counter() - t0:.1f} s", flush=True)
