"""Timing pair (1), route P's side: the SCF at p/q = 1/6 on graphene, twice in one process.

The first call pays the compilation; the second is the number (never time a first call).
Both start from the shared zero-field density, kmesh (2, 12), the four representatives,
nbnd 32, conv_thr 1e-8, as fold6.log. Usage: python mag_time_routep.py [q]
"""
import resource
import sys
import time

from defuqumat import Calculator
from defuqumat.magnetic.fold import run_magnetic_scf
from defuqumat.units import RY_TO_EV

SCF = "plans/orbital-field/graphene-scf.npz"
q = int(sys.argv[1]) if len(sys.argv) > 1 else 6
calc = Calculator.from_keywords(
    lattice="hexagonal", a=2.46, c=15.0, atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
    pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo", ecut=50 * RY_TO_EV,
    kmesh=(12, 12, 1), smearing=0.01 * RY_TO_EV, smearing_type="mv")
for call in ("first", "second"):
    t0 = time.perf_counter()
    r = run_magnetic_scf(calc.system, calc.pseudos, SCF, flux=(1, q), kmesh=(2, 2 * q),
                         nbnd=4 * q + 8, conv_thr=1e-8, verbose=False)
    wall = time.perf_counter() - t0
    steps = [h.get("davidson_steps") for h in r.history]
    secs = [round(h.get("seconds", float("nan")), 1) for h in r.history]
    print(f"[{call}] 1/{q}: converged {r.converged} in {r.iterations} iterations, "
          f"E {r.total_energy:.10f} Ry per chemical cell, {wall:.1f} s; per iteration {secs} s; "
          f"Davidson steps {steps}; peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)
