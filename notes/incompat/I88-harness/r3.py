"""I88, third fix round: the error of the analytic elastic tensor against the
top-band occupation.

Usage (from a checkout root, under slot.sh):
  r3.py curve <smearing>:<degauss>:<nbnd> [...]   identity at h = 1e-3, 2e-3, 4e-3
  r3.py scan <smearing>:<degauss>:<nbnd>:<celldm>:<grid> [...]   SCF only, top-band occupation
  r3.py path <input> [...]                         top-band occupation of a committed cell
  r3.py guardpath <input> [...]                    the same, then call strain_response
  r3.py guard <smearing>:<degauss>:<nbnd>          call strain_response, print refusal or not

The cell is al2-metal.in (a = 7.50, 4x4x2, nosym) with the smearing line and
nbnd replaced, unless a scan names another celldm and grid. The reference is
the five-point second difference of the free energy F on the frozen sphere
(at_strain), each strained cell re-converged at conv_thr = 1e-12. The analytic
tensor is the strain route at its own cutoff (STRAIN_SMEARING_SMALL = 1e-12)
and tr2 = 1e-24, threshold = 1e-12. An error flat in h is a systematic; one
rising by 16 per doubling of h is the stencil's own h^4.
"""
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response.elastic import RY_TO_GPA, elastic_constants
from defuqumat.response.electrostriction import refined_states
from defuqumat.response.strain import strain_response, strain_tangent
from defuqumat.scf import Calculation, run_scf
from defuqumat.scf.occupations import smearing_order, wgauss
from defuqumat.system import build_system

PSEUDO = Path("tests/data/pseudo")
IDENTITY = {"tr2": 1.0e-24, "threshold": 1.0e-12}
OUT = Path(os.environ.get("R3_OUT", "/l/ladovj1/review/incompat/I88/r3"))
FULL = os.environ.get("FULL", "0") == "1"
T0 = time.time()

TEMPLATE = """ &control
    calculation='scf'
 /
 &system
    ibrav=0, celldm(1)={celldm},
    nat=2, ntyp=1,{nbnd}
    ecutwfc=15.0, nosym=.true., noinv=.true.,
    occupations='smearing', smearing='{smearing}', degauss={degauss}
 /
 &electrons
    conv_thr = 1.0d-12
 /
CELL_PARAMETERS alat
  -0.50  0.00  0.50
   0.00  0.50  0.50
  -1.00  1.00  0.00
ATOMIC_SPECIES
 Al  26.98 Al.pz-vbc.UPF
ATOMIC_POSITIONS (alat)
 Al  0.00  0.00  0.00
 Al -0.50  0.50  0.00
K_POINTS automatic
 {grid} 0 0 0
"""


def say(*args):
    print(f"[{time.time() - T0:7.1f} s]", *args, flush=True)


def write_input(smearing, degauss, nbnd, celldm="7.50", grid="4x4x2"):
    nb = "" if nbnd in ("", "default") else f" nbnd={nbnd},"
    path = OUT / f"{smearing[:2]}{degauss}-n{nbnd or 'def'}-{celldm}-{grid}.in"
    path.write_text(TEMPLATE.format(celldm=celldm, nbnd=nb, smearing=smearing,
                                    degauss=degauss, grid=grid.replace("x", " ")))
    return path


def tangents():
    rng = np.random.default_rng(11)
    m = rng.normal(size=(3, 3))
    m = 0.5 * (m + m.T)
    m = m / np.max(np.abs(m))
    return {"00": np.asarray(strain_tangent(0, 0)), "01": np.asarray(strain_tangent(0, 1)),
            "rand": m}


def values(tensor):
    return {k: float(np.einsum("ij,ijkl,kl->", t, np.asarray(tensor), t)) * RY_TO_GPA
            for k, t in tangents().items()}


def ground(path):
    system = build_system(read_pw_input(Path(path)))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     diago_full_acc=FULL, max_iterations=300)
    eigenvalues, psi = refined_states(calculation, result)
    e3 = jnp.asarray(eigenvalues)
    e3 = e3 if e3.ndim == 3 else e3[None]
    _, lv = calculation.occupations(e3)
    ef = float(lv.get("fermi_energy"))
    deg = float(system.degauss)
    e = np.asarray(e3)[0]
    occ = np.abs(np.asarray(wgauss((ef - e) / deg, smearing_order(system.smearing))))
    top = float(occ[:, -1].max())
    say(f"{path.name}: F {result.total_energy!r} converged {result.converged} nk {e.shape[0]} "
        f"nbnd {e.shape[1]} {system.smearing} {system.degauss} ef {ef:.10f}")
    say(f"  TOP {top:.3e} (band {e.shape[1]}, x_min {(e[:, -1].min() - ef) / deg:.3f}); "
        f"band nbnd-1 max occ {float(occ[:, -2].max()):.3e}; "
        f"min gap to the band below {float(np.min(e[:, -1] - e[:, -2])):.3e} Ry")
    return system, pseudos, calculation, result, eigenvalues, psi, top


def identity(state, names=("00", "01", "rand")):
    system, pseudos, calculation, result, eigenvalues, psi, top = state
    density = jnp.asarray(result.density)
    volume = calculation.system.cell.volume
    response = strain_response(calculation, psi, eigenvalues, density, result.becsum, **IDENTITY)
    tensor = elastic_constants(calculation, psi, eigenvalues, density, response).tensor
    t = np.asarray(tensor)
    an = values(tensor)
    say(f"  analytic: converged {response.converged} passes {len(response.history)} "
        f"{ {k: round(x, 6) for k, x in an.items()} } major-symmetry residue "
        f"{float(np.max(np.abs(t - t.transpose(2, 3, 0, 1)))) * RY_TO_GPA:.2e} GPa")
    base = 1.0e-3
    for name in names:
        tg = tangents()[name]
        energies = {0: float(result.total_energy)}
        for m in (1, 2, 4, 8):
            for sign in (1, -1):
                moved = Calculation(system, pseudos).at_strain(jnp.asarray(sign * m * base * tg))
                run = run_scf(system, pseudos, calculation=moved, conv_thr=1e-12,
                              diago_full_acc=FULL, max_iterations=300,
                              starting_density=result.density,
                              starting_wavefunctions=result.wavefunctions)
                energies[sign * m] = float(run.total_energy)
                if not run.converged:
                    say(f"  NOT CONVERGED {name} {sign * m}")
        row = []
        for s in (1, 2, 4):
            h = s * base
            second = (-energies[2 * s] + 16 * energies[s] - 30 * energies[0]
                      + 16 * energies[-s] - energies[-2 * s]) / (12 * h ** 2)
            reference = second / volume * RY_TO_GPA
            row.append(f"h={h:.0e} ref {reference:.6f} rel {(an[name] - reference) / reference:+.3e}")
        say(f"  CURVE top {top:.3e} {name} analytic {an[name]:.6f}: " + "; ".join(row))


def guard(state):
    system, pseudos, calculation, result, eigenvalues, psi, _ = state
    try:
        strain_response(calculation, psi, eigenvalues, jnp.asarray(result.density),
                        result.becsum, **IDENTITY)
        say("  GUARD DID NOT FIRE")
    except NotImplementedError as error:
        say("  refused:", str(error)[:400])


def main():
    mode, specs = sys.argv[1], sys.argv[2:]
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        if mode in ("path", "guardpath"):
            state = ground(Path(spec))
            if mode == "guardpath":
                guard(state)
        else:
            parts = spec.split(":")
            path = write_input(*parts) if mode == "scan" else write_input(*parts[:3])
            state = ground(path)
            if mode == "curve":
                identity(state)
            elif mode == "guard":
                guard(state)
        jax.clear_caches()


if __name__ == "__main__":
    main()
    say("done")
