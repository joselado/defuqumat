import sys, json, warnings
sys.path.insert(0, "/u/40/ladovj1/data/Documents/programs/claude/defuqumat")
warnings.filterwarnings("ignore")
import numpy as np
from defuqumat import Calculator
from defuqumat.units import RY_TO_EV
base = open("/u/40/ladovj1/data/Documents/programs/claude/defuqumat/notes/qe-writer/inputs/si2-tb09.in").read()
variants = {
    "tb09": base,
    "lmbj-limit": base.replace("input_dft = 'tb09'", "input_dft = 'lmbj', lmbj_sigma = 1000.0, lmbj_rs = 1000.0"),
    "lmbj": base.replace("input_dft = 'tb09'", "input_dft = 'lmbj'"),
    "lmbj-koller": base.replace("input_dft = 'tb09'", "input_dft = 'lmbj', lmbj_parameters = 'koller'"),
}
out = {}
for name, text in variants.items():
    if name not in sys.argv[1:]:
        continue
    calc = Calculator.from_text(text, pseudo_dir="/u/40/ladovj1/data/Documents/programs/claude/defuqumat/tests/data/pseudo", announce=False)
    res = calc.get_scf()
    e = np.asarray(res.eigenvalues)
    f = res.meta_c_field
    out[name] = dict(iterations=res.iterations, converged=bool(res.converged), c=res.meta_c,
                     c_spread=None if f is None else float(np.ptp(f)),
                     mesh_gap_eV=float((e[:, 4].min() - e[:, 3].max()) * RY_TO_EV))
    np.save(f"eig-{name}.npy", e)
    print(name, json.dumps(out[name]), flush=True)
