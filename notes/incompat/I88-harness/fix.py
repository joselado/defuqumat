"""I88 fix round: the strain route's smearing cutoff, on and off, against a five-point F difference.

Usage: fix.py <input.in> [full|guard]
Run from a checkout root (slot.sh cds there). ``full`` (the default) computes the
analytic elastic constants at the strain route's cutoff (STRAIN_SMEARING_SMALL =
1e-12), at 1e-8, and at QE's 6.96e-5 (the new term off), then five-point
differences of F at h = 2e-3 and 4e-3 along (0,0), (0,1) and a random symmetric
strain. ``guard`` only calls strain_response and prints what it raises.
"""
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response import strain as strain_module
from defuqumat.response.elastic import RY_TO_GPA, elastic_constants
from defuqumat.response.electrostriction import refined_states
from defuqumat.response.sternheimer import _SMALL
from defuqumat.response.strain import strain_response, strain_tangent
from defuqumat.scf import Calculation, run_scf
from defuqumat.system import build_system

PSEUDO = Path("tests/data/pseudo")
IDENTITY = {"tr2": 1.0e-24, "threshold": 1.0e-12}
T0 = time.time()


def say(*args):
    print(f"[{time.time() - T0:7.1f} s]", *args, flush=True)


def tangents():
    rng = np.random.default_rng(7)
    m = rng.normal(size=(3, 3))
    m = 0.5 * (m + m.T)
    m = m / np.max(np.abs(m))
    return {"00": np.asarray(strain_tangent(0, 0)), "01": np.asarray(strain_tangent(0, 1)),
            "rand": m}


def values(tensor):
    return {k: float(np.einsum("ij,ijkl,kl->", t, np.asarray(tensor), t)) * RY_TO_GPA
            for k, t in tangents().items()}


def main(path, mode):
    system = build_system(read_pw_input(Path(path)))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=200)
    eigenvalues, psi = refined_states(calculation, result)
    volume = calculation.system.cell.volume
    density = jnp.asarray(result.density)
    e3 = jnp.asarray(eigenvalues)
    e3 = e3 if e3.ndim == 3 else e3[None]
    wg, lv = calculation.occupations(e3)
    wk = np.asarray(calculation.system.kpoints.weights)
    occ = np.asarray(wg)[0] / wk[:, None]
    say(path, "F", result.total_energy, "converged", result.converged, "nk", psi.shape[1],
        "nbnd", psi.shape[2], system.smearing, system.degauss, "ef", lv.get("fermi_energy"),
        "max |occ| top band", float(np.max(np.abs(occ[:, -1]))))

    if mode == "guard":
        try:
            strain_response(calculation, psi, eigenvalues, density, result.becsum, **IDENTITY)
            say("GUARD DID NOT FIRE")
        except NotImplementedError as error:
            say("refused:", str(error))
        return

    analytic = {}
    for label, small in (("new 1e-12", 1.0e-12), ("1e-8", 1.0e-8), ("off (QE 6.96e-5)", _SMALL)):
        strain_module.STRAIN_SMEARING_SMALL = small
        response = strain_response(calculation, psi, eigenvalues, density, result.becsum,
                                   **IDENTITY)
        tensor = elastic_constants(calculation, psi, eigenvalues, density, response).tensor
        t = np.asarray(tensor)
        analytic[label] = values(tensor)
        say(f"analytic {label}: converged {response.converged} passes {len(response.history)}",
            {k: round(v, 6) for k, v in analytic[label].items()},
            "major-symmetry residue (GPa)",
            float(np.max(np.abs(t - t.transpose(2, 3, 0, 1)))) * RY_TO_GPA)
    strain_module.STRAIN_SMEARING_SMALL = 1.0e-12

    base = 1.0e-3
    for name, t in tangents().items():
        energies = {0: float(result.total_energy)}
        for m in (2, 4, 8):
            for sign in (1, -1):
                moved = Calculation(system, pseudos).at_strain(jnp.asarray(sign * m * base * t))
                run = run_scf(system, pseudos, calculation=moved, conv_thr=1e-12,
                              max_iterations=200, starting_density=result.density,
                              starting_wavefunctions=result.wavefunctions)
                energies[sign * m] = float(run.total_energy)
                if not run.converged:
                    say(f"  NOT CONVERGED {name} {sign * m}")
        for s in (2, 4):
            h = s * base
            second = (-energies[2 * s] + 16 * energies[s] - 30 * energies[0]
                      + 16 * energies[-s] - energies[-2 * s]) / (12 * h ** 2)
            reference = second / volume * RY_TO_GPA
            say(f"FD {name} h={h:.0e}: five-point {reference:.6f} GPa; "
                + "; ".join(f"{label} {v[name]:.6f} rel {(v[name] - reference) / reference:+.3e}"
                            for label, v in analytic.items()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "full")
    say("done")
