import sys, time, json, warnings
sys.path.insert(0, "/u/40/ladovj1/data/Documents/programs/claude/defuqumat")
warnings.filterwarnings("ignore")
import numpy as np
from defuqumat import Calculator
base = open("/u/40/ladovj1/data/Documents/programs/claude/defuqumat/notes/qe-writer/inputs/si2-tb09.in").read()
texts = {"tb09": base, "lmbj": base.replace("input_dft = 'tb09'", "input_dft = 'lmbj'")}
out = {}
for name, text in texts.items():
    per_iter, iters = [], None
    for rep in range(4):
        calc = Calculator.from_text(text, pseudo_dir="/u/40/ladovj1/data/Documents/programs/claude/defuqumat/tests/data/pseudo", announce=False)
        t = time.perf_counter(); res = calc.get_scf(); dt = time.perf_counter() - t
        if rep > 0:
            per_iter.append(dt / res.iterations); iters = res.iterations
    out[name] = dict(iterations=iters, median_s_per_iteration=float(np.median(per_iter)), samples=len(per_iter))
print(json.dumps(out))
