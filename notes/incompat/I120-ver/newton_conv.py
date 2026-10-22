import sys
from pathlib import Path
WT = Path(sys.argv[1])
sys.path.insert(0, str(WT / "notes/incompat/I120-harness"))
sys.path.insert(0, str(WT))
from jvp import ROOT, edited  # noqa
from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import Calculation, run_scf
from defuqumat.system import build_system
text = edited("si2-tb09", "2 2 2 1 1 1", 15)
system = build_system(parse_pw_input(text))
pseudos = tuple(read_upf(ROOT / "tests/data/pseudo" / s.pseudo_file) for s in system.structure.species)
mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-12,
                max_iterations=200)
for conv in (1e-11, 1e-13):
    store = "device"
    mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos), conv_thr=conv, max_iterations=300)
    n = run_scf(system, pseudos, calculation=Calculation(system, pseudos), conv_thr=conv,
                max_iterations=300, wfc_store=store, scf_solver="newton-krylov",
                scf_solver_options={"jvp": "sternheimer", "forcing": 0.5, "warmup": 3})
    print(conv, mixed.converged, "converged", n.converged, "steps", n.solver.steps,
          "dE", abs(n.total_energy - mixed.total_energy), flush=True)
