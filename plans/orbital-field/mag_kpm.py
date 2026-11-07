"""Foundation if-time item (b): the KPM DOS of the Peierls operator against its exact levels.

The nearest-neighbour model at q = 7900 (9.99 T), kappa (0.11, 0.23): PeierlsOperator.apply
under kpm.moments.recursion with the Jackson kernel and random-phase start vectors, against
the Gaussian-broadened shift-invert levels within E_D +- 0.3 eV. The Jackson kernel's
resolution at the band centre is a Gaussian of sigma = pi half / N, so N is chosen for
sigma = 20 meV and the exact levels are broadened by the same sigma. The levels are flat in
kappa (round-off), so one kappa's levels are the DOS of the whole magnetic zone.
Usage: python mag_kpm.py <outdir> [q] [n_random]
"""
import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.kpm.kernels import get_kernel
from defuqumat.kpm.moments import random_phases, recursion
from defuqumat.kpm.profile import chebyshev_profile
from defuqumat.magnetic.conventions import MagneticCell
from defuqumat.magnetic.peierls import PeierlsTorus, honeycomb_nn_bonds, levels_near
from defuqumat.units import RY_TO_EV

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
q = int(sys.argv[2]) if len(sys.argv) > 2 else 7900
n_random = int(sys.argv[3]) if len(sys.argv) > 3 else 100
bonds = honeycomb_nn_bonds()
torus = PeierlsTorus(MagneticCell.build(bonds.lattice, 1, q), bonds)
kappa = (0.11, 0.23)
h = torus.matrix(kappa)
op = torus.operator(kappa)
half = 1.01 * 3 * 2.7 / RY_TO_EV          # |E| <= 3 |t| for the NN model
sigma = 0.020 / RY_TO_EV
n_moments = int(math.ceil(math.pi * half / sigma))
t0 = time.perf_counter()
key = jax.random.PRNGKey(0)
start = random_phases(key, jnp.ones((1, torus.dim), bool)[0], n_random, op.blocks.dtype)
moments = recursion(lambda v: op.apply(v) / half, start, n_moments // 2)   # (R, 2 (M + 1))
moments = np.asarray(moments)[:, :n_moments].mean(axis=0) * torus.dim       # a trace estimate
t_kpm = time.perf_counter() - t0
factors = get_kernel("jackson").factors(n_moments)
energies = np.linspace(-0.3, 0.3, 601) / RY_TO_EV
dos, _ = chebyshev_profile(moments, factors, 0.0, half, energies)
dos = np.asarray(dos) / torus.dim                                           # per Ry per state

t0 = time.perf_counter()
w = levels_near(h, 0.0, 80, vectors=False)
t_si = time.perf_counter() - t0
inside = w[np.abs(w) * RY_TO_EV <= 0.36]
assert np.max(np.abs(w)) * RY_TO_EV > 0.36, "widen the level window"
exact = np.sum(np.exp(-0.5 * ((energies[:, None] - inside[None, :]) / sigma) ** 2), axis=1) \
    / (sigma * math.sqrt(2 * math.pi)) / torus.dim
l2 = float(np.sqrt(np.trapezoid((dos - exact) ** 2, energies) / np.trapezoid(exact ** 2, energies)))
record = {"q": q, "dim": torus.dim, "n_moments": n_moments, "n_random": n_random, "half_ry": half,
          "sigma_ev": 0.020, "relative_L2": l2, "kpm_s": t_kpm, "shift_invert_s": t_si,
          "levels_in_window": int(len(inside))}
print("kpm:", json.dumps(record), flush=True)
np.savez(out / f"nn-kpm-{q}.npz", energies_ev=energies * RY_TO_EV, kpm=dos, exact=exact)
with open(out / f"nn-kpm-{q}.json", "w") as f:
    json.dump(record, f, indent=1)
