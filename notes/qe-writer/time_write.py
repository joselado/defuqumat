import sys, time, statistics, tempfile, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from defuqumat import Calculator
from defuqumat.io.qe_density import write_qe_save
from defuqumat.io.qe_gvectors import qe_gvector_order
SAVE = REPO / "tests/data/qe/save/si-paw-444"
calc = Calculator.from_text((SAVE / "pw.in").read_text(), pseudo_dir=REPO / "tests/data/pseudo", announce=False)
r = calc.get_scf(**calc.get_qe_seed(SAVE))
d = Path(tempfile.mkdtemp())
write_qe_save(d, r, calc.calculation)  # warm-up (numba compile)
t = []
for _ in range(5):
    t0 = time.perf_counter(); write_qe_save(d, r, calc.calculation); t.append(time.perf_counter() - t0)
g = []
cell = calc.calculation.system.cell
for _ in range(5):
    t0 = time.perf_counter(); qe_gvector_order(cell.at, cell.alat, calc.calculation.basis.dense.ecut); g.append(time.perf_counter() - t0)
print(f"write_qe_save median {statistics.median(t)*1e3:.1f} ms of 5; G order {statistics.median(g)*1e3:.1f} ms; ngm {len(qe_gvector_order(cell.at, cell.alat, calc.calculation.basis.dense.ecut))}")
