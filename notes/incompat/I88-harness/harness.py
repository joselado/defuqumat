"""I88: the elastic constants of a smeared metal against a five-point energy difference.

Run from a checkout root (slot.sh cds there). Prints every number the record uses.
Usage: harness.py [al|si|all]
"""
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.forces.energy import FrozenState, energy_at
from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response import elastic
from defuqumat.response.elastic import RY_TO_GPA, elastic_constants
from defuqumat.response.electrostriction import refined_states
from defuqumat.response.sternheimer import SternheimerSolver
from defuqumat.response.strain import strain_response, strain_tangent
from defuqumat.scf import Calculation, run_scf
from defuqumat.system import build_system

CASES = Path("tests/data/qe")
PSEUDO = Path("tests/data/pseudo")
IDENTITY = {"tr2": 1.0e-24, "threshold": 1.0e-12}
T0 = time.time()


def say(*args):
    print(f"[{time.time() - T0:7.1f} s]", *args, flush=True)


def converged(case):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=100)
    eigenvalues, psi = refined_states(calculation, result)
    return system, pseudos, calculation, result, eigenvalues, psi


def components(tensor):
    return {f"C{a}{b}{c}{d}": round(float(tensor[a, b, c, d]) * RY_TO_GPA, 6)
            for (a, b, c, d) in ((0, 0, 0, 0), (0, 1, 0, 1), (0, 0, 1, 1), (2, 2, 2, 2))}


def aluminium():
    case = "al2-metal"
    system, pseudos, calculation, result, eigenvalues, psi = converged(case)
    say("al2 SCF total energy", result.total_energy, "Ry; terms smearing",
        result.energy_terms.get("smearing"), "; pw.x reference.out.al2-metal:342 -8.33210380")
    volume = calculation.system.cell.volume
    density = jnp.asarray(result.density)
    nelec = float(jnp.sum(density[0])) * volume / density[0].size
    say("volume", volume, "electrons", nelec, "nbnd", psi.shape[2], "nk", psi.shape[1])
    e3 = jnp.asarray(eigenvalues)
    e3 = e3 if e3.ndim == 3 else e3[None]
    weights, levels = calculation.occupations(e3)
    wk = np.asarray(calculation.system.kpoints.weights)
    say("sum wk", wk.sum(), "max wg/wk band 0",
        float(np.max(np.asarray(weights)[0, :, 0] / wk)), "ef", levels.get("fermi_energy"))

    response = strain_response(calculation, psi, eigenvalues, density, result.becsum, **IDENTITY)
    say("strain response converged", response.converged, "iterations", len(response.history),
        "last", response.history[-1])
    element = volume / density[0].size
    for pair in ((0, 0), (0, 1), (1, 1)):
        total = float(jnp.sum(response.drho[pair][0])) * element
        states = float(jnp.sum((response.drho[pair] - response.moved_drho[pair])[0])) * element
        expect = -nelec * float(np.trace(np.asarray(strain_tangent(*pair))))
        say(f"integral of drho{pair}: total {total:.6e} (expect -N Tr T = {expect:.6e}),"
            f" states' half {states:.3e} (expect 0)")
    analytic = elastic_constants(calculation, psi, eigenvalues, density, response)
    say("analytic (split, shifted)", components(analytic.tensor))
    say("voigt GPa\n", np.array2string(analytic.voigt, precision=4, suppress_small=True))

    # A/B 1: the weight split off (the insulator's single block at wg).
    original = elastic._split_weights
    elastic._split_weights = lambda c, p, w, e: (p, w, e, None)
    try:
        unsplit = elastic_constants(calculation, psi, eigenvalues, density, response)
    finally:
        elastic._split_weights = original
    say("A/B unsplit assembly (wg single block), shift on", components(unsplit.tensor))

    # A/B 2: the Fermi-level shift off, split on.
    shift = SternheimerSolver.fermi_level_shift
    SternheimerSolver.fermi_level_shift = (
        lambda self, drho, levels=None: (drho, jnp.zeros(())))
    try:
        response_off = strain_response(calculation, psi, eigenvalues, density,
                                       result.becsum, **IDENTITY)
    finally:
        SternheimerSolver.fermi_level_shift = shift
    say("no-shift response converged", response_off.converged, len(response_off.history))
    noshift = elastic_constants(calculation, psi, eigenvalues, density, response_off,
                                allow_unconverged=True)
    say("A/B no Fermi shift, split on", components(noshift.tensor))

    # The reference: five-point second differences of the free energy over
    # re-converged cells at the same frozen sphere, at three steps.
    base = 2.0e-3
    for pair in ((0, 0), (0, 1)):
        tangent = strain_tangent(*pair)
        energies = {0: float(result.total_energy)}
        for m in (1, 2, 4, 8):
            for sign in (1, -1):
                moved = Calculation(system, pseudos).at_strain(sign * m * base * tangent)
                run = run_scf(system, pseudos, calculation=moved, conv_thr=1e-12,
                              max_iterations=200, starting_density=result.density,
                              starting_wavefunctions=result.wavefunctions)
                energies[sign * m] = float(run.total_energy)
                say(f"  E({pair}, {sign * m * base:+.4f}) = {run.total_energy:.12f}"
                    f" conv {run.converged}")
        a, b = pair
        ours = float(analytic.tensor[a, b, a, b]) * RY_TO_GPA
        other = {"unsplit": float(unsplit.tensor[a, b, a, b]) * RY_TO_GPA,
                 "no-shift": float(noshift.tensor[a, b, a, b]) * RY_TO_GPA}
        for m in (1, 2, 4):
            h = m * base
            second = (-energies[2 * m] + 16 * energies[m] - 30 * energies[0]
                      + 16 * energies[-m] - energies[-2 * m]) / (12 * h ** 2)
            three = (energies[m] - 2 * energies[0] + energies[-m]) / h ** 2
            reference = second / volume * RY_TO_GPA
            say(f"FD {pair} h={h:.0e}: five-point {reference:.6f} GPa, three-point "
                f"{three / volume * RY_TO_GPA:.6f}; analytic {ours:.6f}; relative "
                f"{(ours - reference) / reference:+.3e}; "
                + "; ".join(f"{k} {(v - reference) / reference:+.3e}" for k, v in other.items()))


def silicon():
    """The split form on an insulator must reduce to the single block exactly."""
    case = "si-electrostriction"
    system, pseudos, calculation, result, eigenvalues, psi = converged(case)
    density = jnp.asarray(result.density)
    response = strain_response(calculation, psi, eigenvalues, density, result.becsum,
                               **IDENTITY)
    single = elastic_constants(calculation, psi, eigenvalues, density, response)
    say("si single block", components(single.tensor))
    e3 = jnp.asarray(eigenvalues)
    e3 = e3 if e3.ndim == 3 else e3[None]
    weights, _ = calculation.occupations(e3)
    doubled, split, energies, block = elastic._split_weights(
        calculation, jnp.asarray(psi), jnp.asarray(weights), e3)

    def energy(strain, states):
        return energy_at(calculation.at_strain(strain),
                         FrozenState(wavefunctions=states, weights=split,
                                     eigenvalues=energies))

    gradient = jax.grad(energy, argnums=0)
    volume = calculation.system.cell.volume
    for pair in ((0, 0), (0, 1)):
        column = jax.jvp(gradient, (jnp.zeros((3, 3)), doubled),
                         (strain_tangent(*pair), block(response.dpsi[pair])))[1]
        column = 0.5 * (np.asarray(column) + np.asarray(column).T) / volume
        a, b = pair
        say(f"si split {pair}: {column[a, b] * RY_TO_GPA:.10f} GPa against single "
            f"{single.tensor[a, b, a, b] * RY_TO_GPA:.10f}, difference "
            f"{(column[a, b] - single.tensor[a, b, a, b]) * RY_TO_GPA:.3e}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("si", "all"):
        silicon()
    if which in ("al", "all"):
        aluminium()
    say("done")
