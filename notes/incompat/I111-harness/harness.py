"""I111: the fixed-spin-moment field under spin-orbit coupling, against the penalty.

Usage: harness.py <case> <run> [<run> ...]
  case: 'sym'  -- bcc Fe, fully-relativistic dataset, K_POINTS gamma, target (0, 0, 2)
        'asym' -- the same cell at fe-fsm.in's one generic k-point, target
                  2 (sin 30, 0, cos 30), a direction no symmetry selects
  run:  'fsm' (elk rule, SOC), 'fsm-nosoc', 'free' (no constraint, SOC),
        'pen<lambda>' (the 'total' penalty, SOC), 'fsm-tight' (FSM_TOLERANCE 1e-5)

For the penalty the equivalent uniform field is B = -2 lambda (M - M_fix), the
field whose Zeeman potential equals the penalty's gradient (fields.py:
field_energy is -B.M, constraint_energy is lambda |M - M_fix|^2).
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import defuqumat  # noqa: F401  (x64)
import defuqumat.scf.fields as fields
from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import run_scf
from defuqumat.system.builder import build_system

PSEUDO = Path(__file__).resolve().parents[3] / "tests" / "data" / "pseudo"

TEMPLATE = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 3, celldm(1) = 5.217, nat = 1, ntyp = 1,
    ecutwfc = 25.0, ecutrho = 200.0,
    occupations = 'smearing', smearing = 'marzari-vanderbilt', degauss = 0.05
    noncolin = .true.
    nosym = .true.
    noinv = .true.
    lspinorb = {soc}
    starting_magnetization(1) = 0.5
    angle1(1) = {theta}
    angle2(1) = 0.0
{constraint}
 /
 &electrons
    conv_thr = 1.0d-10
    mixing_beta = 0.3
 /
ATOMIC_SPECIES
 Fe 55.847 Fe.rel-pbe-spn-rrkjus_psl.0.2.1.UPF
ATOMIC_POSITIONS (alat)
 Fe 0.0 0.0 0.0
{kpoints}
"""

CASES = {
    "sym": dict(theta=0.0, target=(0.0, 0.0, 2.0), kpoints="K_POINTS gamma"),
    "asym": dict(
        theta=30.0,
        target=(2.0 * math.sin(math.radians(30.0)), 0.0,
                2.0 * math.cos(math.radians(30.0))),
        kpoints="K_POINTS\n 1\n   0.1875000  0.1875000  0.4375000   1.00",
    ),
}


def text(case, run):
    c = CASES[case]
    soc = ".false." if run == "fsm-nosoc" else ".true."
    lines = []
    if run.startswith("fsm"):
        lines.append("    constrained_magnetization = 'fsm'")
        lines.append("    lambda = 0.02")
    elif run.startswith("pen"):
        lines.append("    constrained_magnetization = 'total'")
        lines.append(f"    lambda = {float(run[3:])}")
    if run != "free":
        for i, v in enumerate(c["target"], 1):
            lines.append(f"    fixed_magnetization({i}) = {v:.12f}")
    return TEMPLATE.format(soc=soc, theta=c["theta"], constraint="\n".join(lines),
                           kpoints=c["kpoints"])


def main():
    case, runs = sys.argv[1], sys.argv[2:]
    target = np.asarray(CASES[case]["target"])
    unit = target / np.linalg.norm(target)
    for run in runs:
        fields.FSM_TOLERANCE = 1.0e-5 if run == "fsm-tight" else 1.0e-3
        system = build_system(parse_pw_input(text(case, run)))
        pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
        t0 = time.time()
        result = run_scf(system, pseudos, conv_thr=1e-10, max_iterations=400,
                         mixing_beta=0.3)
        m = np.asarray(result.magnetization_vector, dtype=float)
        if run.startswith("pen"):
            b = -2.0 * float(run[3:]) * (m - target)
        elif run.startswith("fsm"):
            b = np.asarray(result.magnetic_field.uniform, dtype=float)
        else:
            b = np.zeros(3)
        bpar = float(b @ unit)
        bperp = b - bpar * unit
        trace = [list(map(float, h["magnetization_vector"])) for h in result.history
                 if "magnetization_vector" in h]
        out = dict(case=case, run=run, converged=bool(result.converged),
                   iterations=int(result.iterations),
                   total_energy=float(result.total_energy),
                   constraint_energy=float(result.constraint_energy or 0.0),
                   accuracy=float(result.accuracy), moment=m.tolist(),
                   moment_error=(m - target).tolist(), field=b.tolist(),
                   field_parallel=bpar, field_perp=bperp.tolist(),
                   field_perp_norm=float(np.linalg.norm(bperp)),
                   seconds=time.time() - t0,
                   trace_every_10=trace[::10] + trace[-1:])
        print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
