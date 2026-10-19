"""I95/I96 harness: the numbers the slow tests in tests/regression/test_tddft_metal.py assert."""
import sys
import time

sys.path.insert(0, ".")
import numpy as np

from tests.regression import test_tddft_metal as t

np.set_printoptions(precision=6, linewidth=160)
t0 = time.time()


def stamp(msg):
    print(f"[{time.time() - t0:8.1f} s] {msg}", flush=True)


part = sys.argv[1] if len(sys.argv) > 1 else "all"

if part in ("all", "drude"):
    here, there, plasma2, z = t.drude_identity()
    stamp(f"drude identity nk=4 nbnd=12: max|eps_chi0 - eps_sigma| = {np.max(np.abs(here - there)):.3e}")
    stamp(f"  wp^2 = {plasma2:.6f} Ry^2, hbar wp = {np.sqrt(plasma2) * 13.605693:.4f} eV")
    print("  eps_chi0 xx:", here[:, 0])
    print("  eps_sig  xx:", there[:, 0])
    off, _, _, _ = t.drude_identity(drude=False)
    stamp(f"  control (no Drude): max|eps_chi0 - eps_sigma| = {np.max(np.abs(off - there)):.3e}")
    stamp(f"  (off - here) vs wp^2/z^2: max rel = {np.max(np.abs((off - here) / (plasma2 / z[:, None] ** 2) - 1)):.3e}")
    import jax
    jax.clear_caches()

if part in ("all", "fsum"):
    for nk, nbnd in ((4, 16), (4, 24), (6, 16), (8, 16)):
        total, inter, dia = t.fsum(nk, nbnd=nbnd)
        stamp(f"fsum nk={nk} nbnd={nbnd}: total {total:.6f} inter {inter:.6f} dia {dia:.6f} Ry^2 "
              f"ratio {total / dia:.5f} inter/dia {inter / dia:.5f}")
        import jax
        jax.clear_caches()

if part in ("all", "silicon"):
    fixed = t.smeared_silicon(None)
    stamp(f"silicon fixed eps_M: {fixed}")
    for d in (0.02, 0.01, 0.005):
        on = t.smeared_silicon(d)
        off = t.smeared_silicon(d, drude=False)
        stamp(f"silicon degauss={d}: max|eps - fixed| = {np.max(np.abs(on - fixed)):.3e}, "
              f"without Drude {np.max(np.abs(off - fixed)):.3e}, Drude alone {np.max(np.abs(on - off)):.3e}")
stamp("done")
