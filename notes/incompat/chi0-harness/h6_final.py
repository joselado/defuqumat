"""The slow test bodies of this unit on the final commit, printing every number they assert."""
import sys
import time
import traceback

sys.path.insert(0, ".")
import jax
import numpy as np

from tests.regression import test_tddft_metal as metal
from tests.regression import test_tddft_ultrasoft as ultrasoft

t0 = time.time()


def wrap(module, name, show):
    inner = getattr(module, name)

    def outer(*args, **kwargs):
        value = inner(*args, **kwargs)
        print(f"[{time.time() - t0:8.1f} s]   {name}{args}{kwargs or ''}: {show(value)}", flush=True)
        return value

    setattr(module, name, outer)


wrap(ultrasoft, "residual", lambda v: f"eps_M(0) {v[0]:.4f} residual {v[1]:+.4f}")
wrap(metal, "fsum", lambda v: f"total {v[0]:.6f} inter {v[1]:.6f} dia {v[2]:.6f} "
                              f"ratio {v[0] / v[2]:.5f} inter/dia {v[1] / v[2]:.5f}")
wrap(metal, "drude_identity", lambda v: f"max|here - there| {np.max(np.abs(v[0] - v[1])):.3e} "
                                        f"wp^2 {float(np.ravel(v[2])[0]):.6f}")
fixed = {}


def show_silicon(value):
    if "ref" not in fixed:
        fixed["ref"] = value
        return "fixed reference"
    return f"max|eps - fixed| {np.max(np.abs(value - fixed['ref'])):.3e}"


wrap(metal, "smeared_silicon", show_silicon)

names = sys.argv[1].split(",")
for module in (metal, ultrasoft):
    for name in sorted(dir(module)):
        if name.startswith("test_") and any(n in name for n in names):
            try:
                getattr(module, name)()
                print(f"[{time.time() - t0:8.1f} s] PASS {name}", flush=True)
            except Exception:
                print(f"[{time.time() - t0:8.1f} s] FAIL {name}", flush=True)
                traceback.print_exc()
            jax.clear_caches()
