"""Final check: the slow test bodies of this unit, called directly (no pytest -m slow)."""
import sys
import time
import traceback

sys.path.insert(0, ".")
import jax

from tests.regression import test_tddft_metal as metal
from tests.regression import test_tddft_ultrasoft as ultrasoft

t0 = time.time()
names = sys.argv[1].split(",")
for module in (metal, ultrasoft):
    for name in dir(module):
        if name.startswith("test_") and any(n in name for n in names):
            try:
                getattr(module, name)()
                print(f"[{time.time() - t0:8.1f} s] PASS {name}", flush=True)
            except Exception:
                print(f"[{time.time() - t0:8.1f} s] FAIL {name}", flush=True)
                traceback.print_exc()
            jax.clear_caches()
