"""Al fcc under PBE with a Gaussian smearing, a metal: the PAW all-electron state for Elk.

usage: python3 write_al.py <template> <out>   (template: runs/al-pbe.elk.in run
with maxscl 1 and nwrite 1, so that Elk writes STATE.OUT after one loop)
"""
import sys, shutil, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from defuqumat import Calculator
from defuqumat.io.elk_allelectron import write_elk_state_from
ecut = 30.0
text = f"""&control
 calculation='scf'
/
&system
 ibrav=0, nat=1, ntyp=1, ecutwfc={ecut}, ecutrho={8*ecut},
 occupations='smearing', smearing='gaussian', degauss=0.01
/
&electrons
 conv_thr=1d-11
/
CELL_PARAMETERS bohr
 0.0 3.825 3.825
 3.825 0.0 3.825
 3.825 3.825 0.0
ATOMIC_SPECIES
 Al 26.98 Al.pbe-n-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
 Al 0.00 0.00 0.00
K_POINTS automatic
 8 8 8 0 0 0
"""
calc = Calculator.from_text(text, pseudo_dir=REPO / "tests/data/pseudo", announce=False)
res = calc.get_scf()
print(f"defuqumat: {res.iterations} iterations, converged {res.converged}")
template = Path(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
t = time.perf_counter()
write_elk_state_from(out, res, calc.calculation, template)
print(f"written in {time.perf_counter() - t:.2f} s")
for f in ("GEOMETRY.OUT", "Al.in"):
    shutil.copy(template / f, out / f)
elk_in = (template / "elk.in").read_text().replace("tasks\n  0", "tasks\n  1")
elk_in = elk_in.split("\nmaxscl")[0] + "\n"
(out / "elk.in").write_text(elk_in)
