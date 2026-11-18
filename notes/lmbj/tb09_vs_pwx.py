"""Si2 TB09 three ways in defuqumat against pw.x 7.2's tb09 (libxc 6.2.2), same cell."""
import sys, json, re, warnings
from pathlib import Path
REPO = Path("/u/40/ladovj1/data/Documents/programs/claude/defuqumat"); sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")
import numpy as np
import defuqumat.xc.functional as F
from defuqumat import Calculator
from defuqumat.io import qe_density as qd
from defuqumat.units import RY_TO_EV
here = Path(__file__).parent
variant = sys.argv[1]
base = (REPO / "notes/qe-writer/inputs/si2-tb09.in").read_text()
text = base
if variant == "bj06":
    text = base.replace("input_dft = 'tb09'", "input_dft = 'tb09', mbj_c = 1.0")
elif variant == "qe-emulation":
    text = base.replace("input_dft = 'tb09'", "input_dft = 'tb09', mbj_c = 1.0")
    original = F.tb09_potential
    F.tb09_potential = lambda rho, sigma, lap, tau, c: original(rho, sigma, 0.0 * lap, tau, c)
assert variant == "tb09" or text != base
calc = Calculator.from_text(text, pseudo_dir=REPO / "tests/data/pseudo", announce=False)
res = calc.get_scf()
e = np.asarray(res.eigenvalues) * RY_TO_EV
homo, lumo = float(e[:, 3].max()), float(e[:, 4].min())
out = here / f"save-{variant}"
qd.write_qe_save(out, res, calc.calculation)
_, _, m1, r1 = qd.read_rhog(here / "pwx72-charge-density.dat")
_, _, m2, r2 = qd.read_rhog(out / "charge-density.dat")
assert np.array_equal(m1, m2)
pw = (here / "pwx72.out").read_text()
ho, lu = map(float, re.search(r"highest occupied, lowest unoccupied level \(ev\):\s+([-0-9.]+)\s+([-0-9.]+)", pw).groups())
print(json.dumps(dict(variant=variant, iterations=res.iterations, converged=bool(res.converged),
    mesh_gap_eV=round(lumo - homo, 4), pwx_mesh_gap_eV=round(lu - ho, 4),
    density_diff_from_pwx=float(np.abs(r1 - r2).max()), density_peak=float(np.abs(r1).max()))))
