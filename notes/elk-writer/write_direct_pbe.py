import sys, shutil, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from defuqumat import Calculator
from defuqumat.io.elk_allelectron import write_elk_state_from
ecut = 30.0
text = f"""&control
 calculation='scf'
/
&system
 ibrav=0, nat=2, ntyp=1, ecutwfc={ecut}, ecutrho={8*ecut}
/
&electrons
 conv_thr=1d-11
/
CELL_PARAMETERS bohr
 0.0 5.10 5.10
 5.10 0.0 5.10
 5.10 5.10 0.0
ATOMIC_SPECIES
 Si 28.086 Si.pbe-n-kjpaw_psl.0.1.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 4 4 4 0 0 0
"""
calc = Calculator.from_text(text, pseudo_dir=REPO / "tests/data/pseudo", announce=False)
res = calc.get_scf()
template = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
t = time.perf_counter()
write_elk_state_from(out, res, calc.calculation, template)
print(f"written in {time.perf_counter() - t:.2f} s")
for f in ("GEOMETRY.OUT", "Si.in"):
    shutil.copy(template / f, out / f)
(out / "elk.in").write_text((template / "elk.in").read_text().replace("tasks\n  0", "tasks\n  1").replace("tasks\n  9006", "tasks\n  1"))
