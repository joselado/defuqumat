"""Probe for the nspin = 2 sum-over-states chi_0 (notes/incompat/I7.md).

Run from the repository root. Prints the two identities' numbers:

A. unpolarized silicon at nspin = 2 against nspin = 1, the whole matrix;
B. the magnetic H3 chain, RPA eps_M(0) from the sum over states against the
   Sternheimer solve at nspin = 2, at several nbnd and response cutoffs.
"""

import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation, run_scf
from defuqumat.system import build_system
from defuqumat.tddft import alda_matrix, independent_response, solve_dyson
from defuqumat.workflows.nscf import fixed_density_states
from defuqumat.workflows.tddft import run_absorption

CASES = Path("tests/data/qe")
PSEUDO = Path("tests/data/pseudo")


def load(case, **spin):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    if spin:
        system = system.with_spin(**spin)
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos


def chi_of(system, pseudos, nbnd, ecut):
    calculation = Calculation(system, pseudos)
    scf = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                  max_iterations=80)
    calculation, system, eigenvalues, wavefunctions = fixed_density_states(
        system, pseudos, scf.density, nbnd=nbnd, conv_thr=1e-10)
    potential = calculation.potential(jnp.asarray(scf.density))
    chi = independent_response(
        calculation, wavefunctions, eigenvalues, potential.v_scf,
        np.array([0.0, 0.1 + 0.01j]), ecut_response=ecut, broadening=0.0)
    return scf, calculation, chi


def identity_a():
    t = time.time()
    one = chi_of(*load("si-epsilon-unshifted-nosym"), 60, 8.0)
    two = chi_of(*load("si-epsilon-unshifted-nosym", nspin=2, tot_magnetization=0),
                 60, 8.0)
    x1, x2 = np.asarray(one[2].x), np.asarray(two[2].x)
    print("A: shapes", x1.shape, x2.shape, "occupied", one[2].occupied, two[2].occupied)
    print("A: max|x2-x1| =", np.abs(x2 - x1).max(), " max|x1| =", np.abs(x1).max())
    print("A: head x1[0] diag", np.diag(x1[0, :3, :3]).real)
    print("A: head x2[0] diag", np.diag(x2[0, :3, :3]).real)
    for kernel in ("rpa", "alda"):
        eps = []
        for scf, calculation, chi in (one, two):
            context = {}
            if kernel == "alda":
                context["alda_matrix"] = alda_matrix(
                    calculation, jnp.asarray(scf.density), chi.sphere)
            sol = solve_dyson(chi, kernel, context, static_index=0)
            eps.append(np.real(np.diag(np.asarray(sol.epsilon)[0])).mean())
        print(f"A: {kernel} eps_M(0) nspin=1 {eps[0]:.10f} nspin=2 {eps[1]:.10f} "
              f"diff {eps[1] - eps[0]:.3e}")
    print(f"A: took {time.time() - t:.0f} s", flush=True)


def identity_b(nbnds, ecuts):
    system, pseudos = load("h3-chain-lsda-fixed")
    calculation = Calculation(system, pseudos)
    scf = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                  max_iterations=150)
    print("B: scf converged", getattr(scf, "converged", None), "iterations", getattr(scf, "iterations", None),
          "total mag", getattr(scf, "total_magnetization", None),
          "abs mag", getattr(scf, "absolute_magnetization", None))
    eig = np.asarray(scf.eigenvalues_by_spin)
    print("B: eigenvalues (Ry) up min/max per band", eig[0].min(0), eig[0].max(0))
    print("B: eigenvalues (Ry) dn min/max per band", eig[1].min(0), eig[1].max(0))
    print("B: cut_gap", getattr(scf, "cut_gap", None), flush=True)
    for ecut in ecuts:
        for nbnd in nbnds:
            t = time.time()
            spectrum = run_absorption(
                system, pseudos, scf.density, np.array([0.1]), kernel="rpa",
                nbnd=nbnd, ecut_response=ecut, broadening=0.01,
                calculation=calculation)
            stern = spectrum.static_rpa - spectrum.static_residual
            print(f"B: ecut {ecut} nbnd {nbnd} nm {spectrum.nm} "
                  f"sos {spectrum.static_rpa:.8f} sternheimer {stern:.8f} "
                  f"residual {spectrum.static_residual:+.3e} "
                  f"rel(eps-1) {spectrum.static_residual / (stern - 1):+.3e} "
                  f"({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "ab"
    if "b" in which:
        identity_b((20, 40, 80), (4.0, 8.0))
    if "a" in which:
        identity_a()
