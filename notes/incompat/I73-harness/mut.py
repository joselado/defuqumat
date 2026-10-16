"""I73 verifier: do the two slow tests still pass when Newton's F drops the field?

Mutation: driver.make_residual receives a shallow copy of the calculation with
magnetic_field = None, so ScfResidual.step builds a field-free potential. The
mixing loop that runs after Newton is untouched. Each slow test's assertions
are evaluated and printed, plus the loop iteration count after the hand-over.
argv: any of field, atomic, angle, total; prefix 'clean-' to run unmutated.

Written by the I73 verifier; the fix round added the loop-iteration checks the
two slow tests now carry, so the mutant should FAIL exactly those.
"""
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
import defuqumat.scf.driver as driver  # noqa: E402
from defuqumat.io import read_qe_output  # noqa: E402
from defuqumat.io.pwin import parse_pw_input, read_pw_input  # noqa: E402
from defuqumat.pseudo import read_upf  # noqa: E402
from defuqumat.scf import Calculation, run_scf  # noqa: E402
from defuqumat.scf.residual import make_residual as real_make_residual  # noqa: E402
from defuqumat.system import build_system  # noqa: E402

PSEUDO = Path("tests/data/pseudo")
QE = Path(os.environ["DEFUQUMAT_QE_ROOT"]) / "test-suite" / "pw_noncolin"
TOTAL_ENERGY_RY = None
import tests.regression.test_scf_solvers as tss  # noqa: E402
TOTAL_ENERGY_RY = tss.TOTAL_ENERGY_RY
CONSTRAINED_RY = 3e-7

calls = {"n": 0}


def dropping_make_residual(calculation, nbnd, ethr):
    calls["n"] += 1
    free = copy.copy(calculation)
    free.magnetic_field = None
    return real_make_residual(free, nbnd, ethr)


def show(tag, r):
    s = r.solver
    print(f"{tag}: conv={r.converged} E={r.total_energy:.10f} m={r.magnetization} "
          f"mvec={r.magnetization_vector} e_field={r.field_energy} "
          f"e_con={r.constraint_energy} loop_iterations={r.iterations}"
          + ("" if s is None else f" | newton conv={s.converged} acc={s.accuracy:.3e} "
             f"outer={s.iterations} steps={s.steps}"), flush=True)


def check(name, ok):
    print(f"   assert {name}: {'PASS' if ok else 'FAIL'}", flush=True)


for arg in sys.argv[1:]:
    mutate = not arg.startswith("clean-")
    case = arg.removeprefix("clean-")
    driver.make_residual = dropping_make_residual if mutate else real_make_residual
    calls["n"] = 0
    tag = f"{'MUTANT' if mutate else 'CLEAN'} {case}"
    t = time.perf_counter()
    if case == "field":
        system, pseudos = tss._iron_under_a_field(tss.FIELD_EDITS)
        mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                        conv_thr=1e-10, max_iterations=300, mixing_beta=0.3)
        show(f"{tag} mixed", mixed)
        newton = run_scf(
            system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-10,
            max_iterations=300, mixing_beta=0.3, scf_solver="newton-krylov",
            scf_solver_options={"forcing": 0.5, "warmup": 6},
        )
        show(f"{tag} newton", newton)
        check("both converged", mixed.converged and newton.converged)
        check("solver converged", newton.solver is not None and newton.solver.converged)
        check("E", abs(newton.total_energy - mixed.total_energy) <= TOTAL_ENERGY_RY)
        check("e_field", abs(newton.field_energy - mixed.field_energy) <= TOTAL_ENERGY_RY)
        check("m", abs(newton.magnetization - mixed.magnetization) <= 1e-4)
        check("newton loop iterations <= 2", newton.iterations <= 2)
        print(f"   dE={newton.total_energy - mixed.total_energy:.3e} "
              f"dm={newton.magnetization - mixed.magnetization:.3e}", flush=True)
    else:
        name = {"atomic": "noncolin-constrain_atomic.in",
                "angle": "noncolin-constrain_angle.in",
                "total": "noncolin-constrain_total.in"}[case]
        system = build_system(read_pw_input(QE / name))
        pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
        result = run_scf(
            system, pseudos, conv_thr=1e-13, max_iterations=200, mixing_beta=0.2,
            scf_solver="newton-krylov",
            scf_solver_options={"forcing": 0.5, "warmup": 6, "max_iterations": 40},
        )
        show(f"{tag}", result)
        ref_path = Path("tests/data/qe") / f"reference.out.pw_noncolin-{Path(name).stem}"
        reference = read_qe_output(ref_path)
        check("converged", result.converged)
        check("solver converged", result.solver is not None and result.solver.converged)
        check("loop iterations <= 2", result.iterations <= 2)
        check("E vs pw.x", abs(result.total_energy - reference.total_energy) <= CONSTRAINED_RY)
        check("mvec", np.allclose(np.asarray(result.magnetization_vector),
                                  np.asarray(reference.magnetization_vector), atol=6e-3, rtol=0))
        print(f"   E-pw.x={result.total_energy - reference.total_energy:.3e}", flush=True)
        if case == "atomic":
            printed = [float(line.split("=")[1]) for line in ref_path.read_text().splitlines()
                       if "constraint energy" in line]
            check("e_con", abs(result.constraint_energy - printed[-1]) <= 1e-6)
    print(f"{tag}: make_residual calls={calls['n']} time={time.perf_counter() - t:.1f}s",
          flush=True)
