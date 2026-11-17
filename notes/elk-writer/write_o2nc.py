import sys, shutil, time, re
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from defuqumat import Calculator
text = """&control
 calculation='scf'
/
&system
 ibrav=1, celldm(1)=8.0, nat=2, ntyp=1, 
 ecutwfc=25.0, ecutrho=100.0,
 occupations='smearing', smearing='gaussian', degauss=0.005, nbnd=16,
 noncolin=.true., starting_magnetization(1)=0.5, angle1(1)=135.0, angle2(1)=0.0
/
&electrons
 mixing_beta=0.3, conv_thr=1d-10
/
ATOMIC_SPECIES
 O 15.99994 O.pz-kjpaw.UPF
ATOMIC_POSITIONS bohr
 O 0.0 0.0 0.0
 O 0.0 0.0 3.4
K_POINTS tpiba
 1
 0.0 0.0 0.0 1.0
"""
calc = Calculator.from_text(text, pseudo_dir=REPO / "tests/data/pseudo", announce=False)
res = calc.get_scf()
print("defuqumat", res.iterations, res.total_energy, "M", res.magnetization)
template = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
t = time.perf_counter(); calc.write_elk_state(out, template, res); print(f"written in {time.perf_counter()-t:.2f} s")
shutil.copy(template / "GEOMETRY.OUT", out); shutil.copy(template / "O.in", out)
e = (template / "elk.in").read_text().replace("tasks\n  0\n", "tasks\n  1\n")
e = re.sub(r"bfieldc\n.*\n", "bfieldc\n  2.0d-6  0.0  0.0\n", e)
e = re.sub(r"reducebf\n.*\n", "", e)
(out / "elk.in").write_text(e)
