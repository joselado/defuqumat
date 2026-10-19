"""I96 harness, second run: the f-sum rule of aluminium against the k-mesh, at two smearings."""
import sys
import time

sys.path.insert(0, ".")
import jax

from tests.regression import test_tddft_metal as t

t0 = time.time()
smearing = sys.argv[1]
degauss = float(sys.argv[2])
meshes = [int(x) for x in sys.argv[3].split(",")]
for nk in meshes:
    total, inter, dia = t.fsum(nk, smearing=smearing, degauss=degauss)
    print(f"[{time.time() - t0:8.1f} s] {smearing} {degauss} nk={nk}: total {total:.6f} "
          f"inter {inter:.6f} drude {total - inter:.6f} dia {dia:.6f} Ry^2 "
          f"ratio {total / dia:.5f} inter/dia {inter / dia:.5f}", flush=True)
    jax.clear_caches()
print(f"[{time.time() - t0:8.1f} s] done")
