"""I114: the Hellmann-Feynman slope dE/ds at s = 0, from the s = 0 self-consistent
states of tetragonal cobalt, along x and z (frozen_expectation on the converged
four-component s = 0 density, turned rigidly; at s = 0 the functional is
spin-rotation invariant, so the turned density is the s = 0 state along x).

Usage: hf.py <out.json>
"""
import dataclasses, json, sys, time, warnings
from defuqumat import Calculator
from defuqumat.scf.driver import run_scf
from defuqumat.workflows.anisotropy import frozen_expectation

out = sys.argv[1]
PSEUDO = "tests/data/pseudo"
calc = Calculator.from_file("tests/data/qe/co-tetragonal-relaxed-mae.in",
                            pseudo_dir=PSEUDO, announce=False)
system = dataclasses.replace(calc.system, noncollinear_gga="kubler")
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    scf = run_scf(system.with_soc_scale(0.0), calc.pseudos, conv_thr=1.0e-12,
                  max_iterations=400)
    rec = dict(total_energy_s0=float(scf.total_energy), converged=bool(scf.converged))
    for axis, direction in (("z", (0.0, 0.0, 1.0)), ("x", (1.0, 0.0, 0.0))):
        rec[f"E1_{axis}_Ry"] = float(frozen_expectation(
            system, calc.pseudos, scf.density, direction=direction, conv_thr=1.0e-12))
rec["seconds"] = time.time() - t0
print(json.dumps(rec))
with open(out, "w") as handle:
    json.dump(rec, handle)
