"""I73: the alternative -- the same iron cell with no field, from the mixer."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
from defuqumat.io.pwin import parse_pw_input  # noqa: E402
from defuqumat.pseudo import read_upf  # noqa: E402
from defuqumat.scf import run_scf  # noqa: E402
from defuqumat.system import build_system  # noqa: E402

text = Path("tests/data/qe/fe-fsm.in").read_text()
for old in ("constrained_magnetization = 'fsm'", "lambda = 0.02", "fixed_magnetization(3) = 2.0"):
    assert old in text
    text = text.replace(old, "")
system = build_system(parse_pw_input(text))
pseudos = tuple(read_upf(Path("tests/data/pseudo") / s.pseudo_file)
                for s in system.structure.species)
t = time.perf_counter()
free = run_scf(system, pseudos, conv_thr=1e-10, max_iterations=300, mixing_beta=0.3)
print(f"free mixing: converged={free.converged} E={free.total_energy:.10f} "
      f"m={free.magnetization} iterations={free.iterations} "
      f"time={time.perf_counter() - t:.1f}s", flush=True)
