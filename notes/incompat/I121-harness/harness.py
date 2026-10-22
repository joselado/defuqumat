"""I121 harness: dE/dq on a tabulated augmentation table against the stored one."""
import os, sys, time
sys.path.insert(0, os.getcwd())
import jax, jax.numpy as jnp, numpy as np
from pathlib import Path
from defuqumat.forces.energy import state_from_result
from defuqumat.forces.spiral import compute_spiral_gradient, spiral_energy
from defuqumat.io.pwin import parse_pw_input
from defuqumat.scf import run_scf
from defuqumat.scf.driver import Calculation
from defuqumat.system import build_system
import defuqumat.pseudo.augmentation as aug
from tests.regression.test_spiral_relaxation import AUGMENTED_SILICON, AUGMENTED_DATASETS, _pseudos

dataset = sys.argv[1] if len(sys.argv) > 1 else "us"
dqs = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.04, 0.02, 0.01]
pseudo_dir = Path("tests/data/pseudo")
if dataset in AUGMENTED_DATASETS:
    text = AUGMENTED_SILICON.format(pseudo=AUGMENTED_DATASETS[dataset], q3=0.3)
else:
    text = Path(f"tests/data/qe/{dataset}.in").read_text()
KB = None if os.environ.get("KB", "none") == "none" else int(os.environ["KB"])
system = build_system(parse_pw_input(text))
pseudos = _pseudos(system, pseudo_dir)
t0 = time.time()
calc = Calculation(system, pseudos)
result = run_scf(system, pseudos, calculation=calc, conv_thr=1e-11, mixing_beta=0.3,
                 max_iterations=200)
assert result.converged
state = state_from_result(result)
_d = np.asarray(result.density); print(f"[{dataset}] sum abs m_i / N: {np.abs(_d[1:]).sum(axis=(1,2,3)) / _d[0].sum()}", flush=True)
print(f"[{dataset}] SCF {time.time()-t0:.0f}s E={result.total_energy:.12f}", flush=True)
stored = compute_spiral_gradient(calc, result, k_batch=KB)
g_stored = np.asarray(stored.gradient)
print(f"[{dataset}] stored grad {g_stored!r}", flush=True)

# The missing term switched off: the cross table held at q0 under the traced q.
original = Calculation.at_spiral_q
def frozen_table(self, q, rebuild_basis=True, widths=None):
    moved = original(self, q, rebuild_basis=rebuild_basis, widths=widths)
    if not rebuild_basis:
        moved.cross_augmentation = self.cross_augmentation
    return moved
Calculation.at_spiral_q = frozen_table
off = Calculation(system, pseudos)
g_off = np.asarray(compute_spiral_gradient(off, state, k_batch=KB).gradient)
Calculation.at_spiral_q = original
print(f"[{dataset}] dQ/dq term off: grad {g_off!r}  |off-stored|={np.abs(g_off-g_stored).max():.3e}", flush=True)
del off
jax.clear_caches()

os.environ["DEFUQUMAT_AUG_MAX_BYTES"] = "1"
q = np.asarray(system.spiral_q, dtype=float)
for dq in dqs:
    aug.AUG_DQ = dq
    jax.clear_caches()
    tab = Calculation(system, pseudos)
    assert isinstance(tab.cross_augmentation, aug.TabulatedAugmentation)
    g_tab = np.asarray(compute_spiral_gradient(tab, state, k_batch=KB).gradient)
    e0 = float(spiral_energy(tab, jnp.asarray(q), state))
    h = 1e-5
    p, m = q.copy(), q.copy(); p[2] += h; m[2] -= h
    fd = (float(spiral_energy(tab, jnp.asarray(p), state))
          - float(spiral_energy(tab, jnp.asarray(m), state))) / (2 * h)
    print(f"[{dataset}] AUG_DQ={dq}: grad {g_tab!r}  |tab-stored|={np.abs(g_tab-g_stored).max():.3e}"
          f"  E_tab-E_scf={e0-result.total_energy:.3e}  fd_z-grad_z={fd-g_tab[2]:.3e}", flush=True)
    del tab
print(f"[{dataset}] total {time.time()-t0:.0f}s")
