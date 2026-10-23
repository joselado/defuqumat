"""OPEN.md Part III M5 gate: one SCF, its peak memory, total and per-k Davidson steps.

Run through slot.sh from the worktree (cwd = the worktree), compiled-kernel cache off:

    DEFUQUMAT_CACHE_DIR=off python3 notes/speed/m5-harness/scf_m5.py <input> <out.json>

``<input>`` is a pw.x input (tests/data/qe/bi10-soc.in, benchmarks/si8-us-1k.in, ...).
This is the computation of ``tests/regression/test_ten_site.py::_converged`` (a
``Calculation`` and ``run_scf`` at conv_thr 1e-10, 200 iterations), without the slow test.
The per-k step arrays are caught off the one ``jax.device_get`` that carries them.
"""

import json
import os
import resource
import sys
import time
from pathlib import Path

import jax
import numpy as np

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import Calculation, run_scf
from defuqumat.system import build_system

path, out = Path(sys.argv[1]), Path(sys.argv[2])
pseudo_dir = Path("tests/data/pseudo")

steps_log = []
real_get = jax.device_get


def spy(tree):
    value = real_get(tree)
    # the iteration's scalars fetch: (steps, unsettled, accuracy_terms, iteration_terms, ...)
    if isinstance(value, tuple) and len(value) == 5:
        first = value[0]
        if hasattr(first, "dtype") and np.issubdtype(np.asarray(first).dtype, np.integer):
            steps_log.append(np.asarray(first).tolist())
    return value


jax.device_get = spy

system = build_system(read_pw_input(path))
pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
start = time.perf_counter()
calculation = Calculation(system, pseudos)
result = run_scf(system, pseudos, calculation=calculation, conv_thr=1.0e-10,
                 max_iterations=200)
elapsed = time.perf_counter() - start

peak_cgroup = None
try:
    cg = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().strip().split("::")[-1].lstrip("/")
    peak_cgroup = int((cg / "memory.peak").read_text())
except Exception:
    pass

record = dict(
    input=str(path),
    converged=bool(result.converged),
    total_energy=repr(float(result.total_energy)),
    iterations=int(result.iterations),
    davidson_iterations=[float(h["davidson_iterations"]) for h in result.history],
    steps_per_k=steps_log,
    ru_maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    cgroup_peak_gb=None if peak_cgroup is None else peak_cgroup / 1e9,
    seconds_including_compile=elapsed,
    cache=os.environ.get("DEFUQUMAT_CACHE_DIR"),
)
out.write_text(json.dumps(record, indent=1))
print(json.dumps({k: v for k, v in record.items() if k != "steps_per_k"}, indent=1))
