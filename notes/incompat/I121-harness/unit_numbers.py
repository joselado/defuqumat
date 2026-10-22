import os, sys, time
sys.path.insert(0, os.getcwd())
import numpy as np, jax, jax.numpy as jnp
from pathlib import Path
import tests.unit.test_spiral_tabulated_augmentation as t
import defuqumat.pseudo.augmentation as aug
t0 = time.time()
import warnings; from defuqumat.calculator import Calculator; c = Calculator.from_text(t.SILICON_US, Path("tests/data/pseudo"), announce=False).calculation; q = jnp.asarray(t.Q0); ext = float(np.linalg.norm(t.Q0))
print("build", time.time() - t0)
st = np.asarray(jax.grad(t._scalar(c, None, None))(q)); print("stored", st, time.time() - t0)
for dq in (0.04, 0.02, 0.01, 0.005):
    aug.AUG_DQ = dq; jax.clear_caches()
    g = np.asarray(jax.grad(t._scalar(c, 1, ext))(q))
    print(dq, g, np.abs(g - st).max(), time.time() - t0, flush=True)
