"""#17: the ultracell density pass with and without the zero-weight states.

u17.py save N STATE_BATCH OUT   one run, the result arrays saved to OUT (.npz)
u17.py time N STATE_BATCH REPS  both arms in this process, interleaved, warm
                                (only in the tree that has _weighted_states)
"""
import sys
import time
import warnings

import numpy as np

import defuqumat
from defuqumat.calculator import Calculator
from defuqumat.ultracell import run_ultracell
import defuqumat.ultracell.driver as driver
import defuqumat.ultracell.density as density

warnings.simplefilter("ignore")
SILICON = """&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=10.20, nat=2, ntyp=1, ecutwfc=12.0,
 nosym=.true., noinv=.true.
/
&electrons
 conv_thr=1.0d-12
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 {n} 2 2 0 0 0
"""
mode, cells, batch = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
batch = None if batch == 0 else batch
print("package:", defuqumat.__file__, "cells", cells, "state_batch", batch, flush=True)
calc = Calculator.from_text(SILICON.format(n=cells), "tests/data/pseudo", announce=False)
scf = calc.get_scf()
shape, kgrid = (cells, 1, 1), (1, 2, 2)

# count the columns the density kernel is handed, per call
columns = []
kernel = density._ultracell_density


def counting(coefficients, vectors, weights, *rest, **kw):
    columns.append((int(vectors.shape[1]), int(np.count_nonzero(np.asarray(weights)))))
    return kernel(coefficients, vectors, weights, *rest, **kw)


density._ultracell_density = counting


def run():
    return run_ultracell(
        calc.system, calc.pseudos, scf, shape, kgrid, nbnd=24,
        external=lambda x: 0.05 * np.cos(2 * np.pi * x[..., 0] / cells),
        conv_thr=1e-10, state_batch=batch)


if mode == "save":
    result = run()
    calls = len(columns)
    print("iterations", result.iterations, "converged", result.converged,
          "density calls", calls, "columns a call", sorted(set(c for c, _ in columns)),
          "weighted a call", sorted(set(w for _, w in columns)))
    print("transforms in the density pass:", sum(c for c, _ in columns),
          "of which weighted:", sum(w for _, w in columns))
    np.savez(sys.argv[4], density=np.asarray(result.density),
             levels=np.asarray(result.eigenvalues), occupations=np.asarray(result.occupations),
             energy=np.asarray(float(result.total_energy)))
    print("E = %.12f Ry" % float(result.total_energy))
else:
    reps = int(sys.argv[4])
    truncating = driver._weighted_states
    arms = {"every state": lambda occupations, smearing: None, "weighted only": truncating}
    times = {name: [] for name in arms}
    results = {}
    for name, rule in arms.items():       # one discarded call of each arm
        driver._weighted_states = rule
        run()
    for _ in range(reps):
        for name, rule in arms.items():
            driver._weighted_states = rule
            columns.clear()
            start = time.perf_counter()
            results[name] = run()
            times[name].append(time.perf_counter() - start)
    for name in arms:
        t = np.array(times[name])
        print(f"{name:14s} median {np.median(t):.3f} s of {reps}: "
              + ", ".join(f"{x:.3f}" for x in t), "iterations", results[name].iterations)
    a, b = (np.asarray(results[n].density) for n in arms)
    print("density, max |difference| between the arms:", float(np.abs(a - b).max()),
          "bitwise equal:", bool(np.array_equal(a, b)))
    print("load", open("/proc/loadavg").read().split()[:3])
