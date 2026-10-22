"""I120: Newton-Krylov with jvp='sternheimer' through the lifted guard, against mixing.

    python3 newton.py tb09|ferro [jvp] [conv_thr]

The same cell converged by the mixing loop and by Newton-Krylov with the
Sternheimer Jacobian, both at conv_thr = 1e-10; printed are the two totals, the
largest eigenvalue difference and the solver's counts.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jvp import ROOT, edited  # noqa: E402

from defuqumat.io.pwin import parse_pw_input  # noqa: E402
from defuqumat.pseudo import read_upf  # noqa: E402
from defuqumat.scf.driver import Calculation, run_scf  # noqa: E402
from defuqumat.system import build_system  # noqa: E402

CELLS = {
    "tb09": dict(case="si2-tb09", kmesh="2 2 2 1 1 1", ecut=15),
    "ferro": dict(case="h-fcc-spiral-scan", kmesh="2 2 2 1 1 1", sub="ferro"),
}
cell = CELLS[sys.argv[1]]
jvp = sys.argv[2] if len(sys.argv) > 2 else "sternheimer"
conv = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-10
text = edited(cell["case"], cell.get("kmesh"), cell.get("ecut"), None, cell.get("sub"))
system = build_system(parse_pw_input(text))
pseudos = tuple(read_upf(ROOT / "tests/data/pseudo" / s.pseudo_file)
                for s in system.structure.species)
mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                conv_thr=conv, max_iterations=200)
newton = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                 conv_thr=conv, max_iterations=200, scf_solver="newton-krylov",
                 scf_solver_options={"jvp": jvp, "forcing": 0.5, "warmup": 3})
solver = newton.solver
print("RESULT", sys.argv[1], jvp, conv, "converged", mixed.converged, newton.converged,
      "mixing", repr(mixed.total_energy), "newton", repr(newton.total_energy),
      "dE", abs(newton.total_energy - mixed.total_energy),
      "deig", float(np.max(np.abs(np.asarray(newton.eigenvalues)
                                  - np.asarray(mixed.eigenvalues)))),
      "steps", getattr(solver, "steps", None), "jvps", getattr(solver, "jvps", None),
      "iterations", getattr(solver, "iterations", None),
      "accuracy", mixed.accuracy, newton.accuracy, flush=True)
