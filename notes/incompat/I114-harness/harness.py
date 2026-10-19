"""I114: one self-consistent direction of tetragonal cobalt (ultrasoft,
fully relativistic) at one soc_scale.

Usage: harness.py <x|z> <s> <conv_thr> <out.json> [old|qqfull]

``old`` puts back the pre-P118 becsum (the soc_scale = 0 map at every scale
below 1, while dvan_so, qq_so and newd_so blend): the alternative hypothesis,
a functional whose Hamiltonian is not its derivative between the ends.
``qqfull`` switches the entry's term off: qq_so is the coupled overlap at every
scale while the other three sandwiches blend.
"""
import dataclasses, json, sys, time, warnings
from defuqumat import Calculator
from defuqumat.workflows.anisotropy import run_relaxed_direction

axis, scale, thr, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
mode = sys.argv[5] if len(sys.argv) > 5 else ""
old = mode == "old"
if mode == "qqfull":
    from defuqumat.pseudo.spinorbit import SpinOrbitCoupling

    def qq_so(self, qq):
        return self._sandwich({(0, 0): qq, (1, 1): qq})

    SpinOrbitCoupling.qq_so = qq_so
if old:
    import defuqumat.scf.driver as driver
    original = driver.becsum_transform

    def becsum_transform(fcoef, becsum_nc, nspin_mag, real=True, soc_scale=1.0):
        return original(fcoef, becsum_nc, nspin_mag, real=real,
                        soc_scale=1.0 if soc_scale == 1.0 else 0.0)

    driver.becsum_transform = becsum_transform

PSEUDO = "tests/data/pseudo"
calc = Calculator.from_file("tests/data/qe/co-tetragonal-relaxed-mae.in",
                            pseudo_dir=PSEUDO, announce=False)
system = dataclasses.replace(calc.system, noncollinear_gga="kubler")
direction = {"x": (1.0, 0.0, 0.0), "z": (0.0, 0.0, 1.0)}[axis]
t0 = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    r = run_relaxed_direction(system, calc.pseudos, direction=direction,
                              soc_scale=scale, conv_thr=thr, max_iterations=400)
rec = dict(axis=axis, soc_scale=scale, conv_thr=thr, old_becsum=old, mode=mode,
           total_energy=r.total_energy,
           converged=r.converged, iterations=r.iterations, accuracy=r.accuracy,
           moment=r.moment, drift=r.drift, seconds=time.time() - t0)
print(json.dumps(rec))
with open(out, "w") as handle:
    json.dump(rec, handle)
