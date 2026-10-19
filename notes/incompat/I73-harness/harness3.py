"""I73: Newton on one of QE's constrained iron inputs, at one conv_thr, against pw.x."""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
from defuqumat.io import read_qe_output  # noqa: E402
from defuqumat.io.pwin import read_pw_input  # noqa: E402
from defuqumat.pseudo import read_upf  # noqa: E402
from defuqumat.scf import run_scf  # noqa: E402
from defuqumat.system import build_system  # noqa: E402

name, conv_thr = sys.argv[1], float(sys.argv[2])
QE = Path(os.environ["DEFUQUMAT_QE_ROOT"]) / "test-suite" / "pw_noncolin"
system = build_system(read_pw_input(QE / name))
pseudos = tuple(read_upf(Path("tests/data/pseudo") / s.pseudo_file)
                for s in system.structure.species)
stem = Path(name).stem
reference_path = Path(f"tests/data/qe/reference.out.pw_noncolin-{stem}")
reference = read_qe_output(reference_path)
printed = [float(line.split("=")[1]) for line in reference_path.read_text().splitlines()
           if "constraint energy" in line]
t = time.perf_counter()
r = run_scf(system, pseudos, conv_thr=conv_thr, max_iterations=200, mixing_beta=0.2,
            scf_solver="newton-krylov",
            scf_solver_options={"forcing": 0.5, "warmup": 6, "max_iterations": 40})
s = r.solver
print(f"{name} conv_thr={conv_thr:g} scheme={system.constrained_magnetization}: "
      f"converged={r.converged} E={r.total_energy:.10f} pw.x={reference.total_energy} "
      f"dE={r.total_energy - reference.total_energy:.3e} "
      f"e_constraint={r.constraint_energy} pw.x_last={printed[-1] if printed else None} "
      f"mvec={r.magnetization_vector} pw.x_m={reference.magnetization_vector} "
      f"loop_iterations={r.iterations} newton converged={s.converged} acc={s.accuracy:.3e} "
      f"outer={s.iterations} steps={s.steps} jvps={s.jvps} time={time.perf_counter() - t:.1f}s",
      flush=True)
