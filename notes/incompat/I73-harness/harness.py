"""I73: Newton-Krylov under a field or a penalty constraint, against pw.x and the mixer."""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from defuqumat.io import read_qe_output  # noqa: E402
from defuqumat.io.pwin import parse_pw_input, read_pw_input  # noqa: E402
from defuqumat.pseudo import read_upf  # noqa: E402
from defuqumat.scf import Calculation, run_scf  # noqa: E402
from defuqumat.scf.driver import default_nbnd  # noqa: E402
from defuqumat.scf.residual import make_residual  # noqa: E402
from defuqumat.system import build_system  # noqa: E402

which = sys.argv[1:] or ["A", "B", "C"]
PSEUDO = Path("tests/data/pseudo")
QE = Path(os.environ["DEFUQUMAT_QE_ROOT"]) / "test-suite" / "pw_noncolin"


def pseudos_of(system):
    return tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)


def show(tag, r, t):
    s = r.solver
    print(f"{tag}: converged={r.converged} E={r.total_energy:.10f} "
          f"e_field={r.field_energy} e_constraint={r.constraint_energy} "
          f"m={r.magnetization} mvec={r.magnetization_vector} "
          f"loop_iterations={getattr(r, 'iterations', None)} time={t:.1f}s"
          + ("" if s is None else
             f" | newton converged={s.converged} acc={s.accuracy:.3e} "
             f"outer={s.iterations} steps={s.steps} jvps={s.jvps}"), flush=True)


def fe_cell(replacements):
    text = Path("tests/data/qe/fe-fsm.in").read_text()
    for old, new in replacements:
        assert old in text, old
        text = text.replace(old, new)
    return build_system(parse_pw_input(text))


FIELD = [("constrained_magnetization = 'fsm'", "b_field(3) = 0.01"),
         ("lambda = 0.02", ""), ("fixed_magnetization(3) = 2.0", "")]
PENALTY = [("constrained_magnetization = 'fsm'", "constrained_magnetization = 'atomic'"),
           ("lambda = 0.02", "lambda = 0.05"), ("fixed_magnetization(3) = 2.0", "")]


def pair(tag, system, pseudos, conv_thr=1e-10):
    t = time.perf_counter()
    mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                    conv_thr=conv_thr, max_iterations=300, mixing_beta=0.3)
    show(f"{tag} mixing", mixed, time.perf_counter() - t)
    t = time.perf_counter()
    newton = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                     conv_thr=conv_thr, max_iterations=300, mixing_beta=0.3,
                     scf_solver="newton-krylov",
                     scf_solver_options={"forcing": 0.5, "warmup": 6})
    show(f"{tag} newton", newton, time.perf_counter() - t)
    print(f"{tag} dE={newton.total_energy - mixed.total_energy:.3e} "
          f"d e_field={(newton.field_energy or 0) - (mixed.field_energy or 0):.3e} "
          f"d e_constraint={(newton.constraint_energy or 0) - (mixed.constraint_energy or 0):.3e} "
          f"dm={newton.magnetization - mixed.magnetization:.3e}", flush=True)


if "B" in which:
    system = fe_cell(FIELD)
    pair("B field", system, pseudos_of(system))
    system = fe_cell(PENALTY)
    pair("B penalty", system, pseudos_of(system))

if "C" in which:
    # The Jacobian action under the field: autodiff against a central difference.
    system = fe_cell(PENALTY)
    pseudos = pseudos_of(system)
    calculation = Calculation(system, pseudos)
    nbnd = system.nbnd or default_nbnd(
        calculation.nelec, system.occupations, calculation.nelup, calculation.neldw)
    residual = make_residual(calculation, nbnd, 1e-11)
    converged = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-10,
                        max_iterations=300, mixing_beta=0.3)
    x = residual.pack(converged.density, converged.becsum)
    _, psi = residual.residual(x, converged.wavefunctions)
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        v = rng.normal(size=x.shape)
        v /= np.linalg.norm(v)
        ad, _ = residual.jvp(x, v, psi)
        fd = residual.jvp_finite_difference(x, v, psi)
        print(f"C seed {seed}: |ad-fd|/|fd| = "
              f"{np.linalg.norm(ad - fd) / np.linalg.norm(fd):.3e}", flush=True)

if "A" in which:
    # pw.x's constrained-atomic iron, noncollinear, 70 k-points.
    pwin = read_pw_input(QE / "noncolin-constrain_atomic.in")
    system = build_system(pwin)
    pseudos = pseudos_of(system)
    reference = read_qe_output(
        Path("tests/data/qe/reference.out.pw_noncolin-noncolin-constrain_atomic"))
    print("A: pw.x total energy", reference.total_energy, flush=True)
    t = time.perf_counter()
    newton = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                     conv_thr=1e-13, max_iterations=200, mixing_beta=0.2,
                     scf_solver="newton-krylov",
                     scf_solver_options={"forcing": 0.5, "warmup": 6,
                                         "max_iterations": 40})
    show("A newton", newton, time.perf_counter() - t)
    print(f"A newton - pw.x = {newton.total_energy - reference.total_energy:.3e} Ry",
          flush=True)
