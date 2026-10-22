"""I120: the jvp of the Tran-Blaha potential in (rho, tau) against a central difference.

    python3 dv.py

A floor in the Sternheimer action that the step does not move could come from
the potential's own tangent; this isolates it on the converged si2-tb09 state.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jvp import ROOT, edited  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from defuqumat.io.pwin import parse_pw_input  # noqa: E402
from defuqumat.pseudo import read_upf  # noqa: E402
from defuqumat.scf.driver import Calculation, run_scf  # noqa: E402
from defuqumat.system import build_system  # noqa: E402

text = edited("si2-tb09", "2 2 2 1 1 1", 15)
system = build_system(parse_pw_input(text))
pseudos = tuple(read_upf(ROOT / "tests/data/pseudo" / s.pseudo_file)
                for s in system.structure.species)
calc = Calculation(system, pseudos)
r = run_scf(system, pseudos, calculation=calc, conv_thr=1e-12, max_iterations=200)
rho, tau = jnp.asarray(r.density), jnp.asarray(r.tau)
rng = np.random.default_rng(1)
dr = jnp.asarray(rng.normal(size=rho.shape)) * rho
dt = jnp.asarray(rng.normal(size=tau.shape)) * tau
if len(sys.argv) > 1 and sys.argv[1] == "symmetric":
    # A direction with the crystal's symmetry keeps |grad rho| = 0 where the
    # group forces it, which is where c's |grad rho| has its cone.
    dr, dt = calc.symmetrize(dr), calc.symmetrize(dt)


def f(q, t):
    return calc.potential(q, tau=t).v_scf


_, jv = jax.jvp(f, (rho, tau), (dr, dt))
for e in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
    fd = (f(rho + e * dr, tau + e * dt) - f(rho - e * dr, tau - e * dt)) / (2 * e)
    print("dv", e, float(jnp.linalg.norm(fd - jv) / jnp.linalg.norm(jv)), flush=True)


def fc(q, t):
    return calc.potential(q, tau=t).meta_c


_, jc = jax.jvp(fc, (rho, tau), (dr, dt))
for e in [1e-3, 1e-5]:
    print("c", e, float(jc),
          float((fc(rho + e * dr, tau + e * dt) - fc(rho - e * dr, tau - e * dt)) / (2 * e)))
