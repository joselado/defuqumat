import sys
import time
from pathlib import Path
WT = Path(sys.argv[1])
sys.path.insert(0, str(WT / "notes/incompat/I120-harness"))
sys.path.insert(0, str(WT))
from jvp import ROOT, edited  # noqa
from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import Calculation, run_scf
from defuqumat.system import build_system
text = edited("h-sheet-noncolin")
system = build_system(parse_pw_input(text))
pseudos = tuple(read_upf(ROOT / "tests/data/pseudo" / s.pseudo_file) for s in system.structure.species)
t = time.time()
mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-10,
                max_iterations=300)
print("mixed", mixed.converged, mixed.total_energy, time.time() - t, flush=True)
for jvp in ("sternheimer", "finite-difference"):
    t = time.time()
    n = run_scf(system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-10,
                max_iterations=300, scf_solver="newton-krylov",
                scf_solver_options={"jvp": jvp, "forcing": 0.5, "warmup": 3})
    print(jvp, "converged", n.converged, "steps", n.solver.steps,
          "dE", abs(n.total_energy - mixed.total_energy), time.time() - t, flush=True)
