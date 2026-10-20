"""I96 harness, third run: the f-sum rule of aluminium against nbnd, at a k-converged smearing."""
import sys
import time

sys.path.insert(0, ".")
import jax

from tests.regression import test_tddft_metal as t

t0 = time.time()
nk = int(sys.argv[1])
for nbnd in [int(x) for x in sys.argv[2].split(",")]:
    total, inter, dia = t.fsum(nk, nbnd=nbnd, smearing="gaussian", degauss=0.2)
    print(f"[{time.time() - t0:8.1f} s] gaussian 0.2 nk={nk} nbnd={nbnd}: total {total:.6f} "
          f"inter {inter:.6f} drude {total - inter:.6f} dia {dia:.6f} Ry^2 "
          f"ratio {total / dia:.5f} inter/dia {inter / dia:.5f}", flush=True)
    jax.clear_caches()
print(f"[{time.time() - t0:8.1f} s] done")
