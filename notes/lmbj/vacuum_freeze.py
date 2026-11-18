"""Experiment: the local mBJ with its meta potential held at the clean tail's value in vacuum.

usage: python3 vacuum_freeze.py <height bohr> <rho_freeze>

Below ``rho_freeze`` (total density) the meta exchange potential is replaced by
its mean over the shell ``rho_freeze < n < 10 rho_freeze``, where the density
is still an exponential tail and the Becke-Johnson term is near its asymptote
``(1/pi) sqrt(5/12) kappa``. A measurement only (a wrapper around
``scf.potential.meta_exchange``), not the code's behaviour.
"""
import sys, json, warnings
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")
import numpy as np
import jax.numpy as jnp
import defuqumat.scf.potential as P
height, rho_f = float(sys.argv[1]), float(sys.argv[2])
original = P.meta_exchange

def frozen(density_r, density_g, tau_r, gvectors, cell, functional):
    v, c = original(density_r, density_g, tau_r, gvectors, cell, functional)
    total = jnp.real(density_r[0]) if density_r.shape[0] == 1 else jnp.sum(jnp.real(density_r), axis=0)
    shell = (total > rho_f) & (total < 10.0 * rho_f)
    below = total <= rho_f
    v_tail = jnp.sum(jnp.where(shell[None], v, 0.0), axis=(1, 2, 3)) / jnp.maximum(jnp.sum(shell), 1)
    return jnp.where(below[None], v_tail[:, None, None, None], v), c

P.meta_exchange = frozen
sys.argv = [sys.argv[0], "lmbj", str(height), "1.0d-11"]
exec(compile((Path(__file__).parent / "vacuum_check.py").read_text().replace(
    'tag = f"vac-{dft}-{int(height)}"', 'tag = f"vac-freeze{rho_f:g}-{int(height)}"'), "vacuum_check.py", "exec"))
