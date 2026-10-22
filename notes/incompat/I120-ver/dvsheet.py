"""Is the vacuum cell's non-convergence in the potential's own tangent?"""
import sys
from pathlib import Path
import numpy as np
WT = Path(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "random"
sys.path.insert(0, str(WT / "notes/incompat/I120-harness"))
sys.path.insert(0, str(WT))
from jvp import ROOT, edited  # noqa
import jax
import jax.numpy as jnp
from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import Calculation, run_scf
from defuqumat.system import build_system

text = edited("h-sheet-noncolin")
system = build_system(parse_pw_input(text))
pseudos = tuple(read_upf(ROOT / "tests/data/pseudo" / s.pseudo_file) for s in system.structure.species)
calc = Calculation(system, pseudos)
r = run_scf(system, pseudos, calculation=calc, conv_thr=1e-12, max_iterations=300)
rho = jnp.asarray(r.density)
n = np.asarray(rho[0]); m = np.linalg.norm(np.asarray(rho[1:]), axis=0)
print("converged", r.converged, "shape", rho.shape, "min n", n.min(), "max n", n.max())
for t in (1e-10, 1e-8, 1e-6, 1e-4):
    print(f"frac n<{t}", float(np.mean(n < t)))
print("frac |m|>=n", float(np.mean(m >= n)), "frac |m|>0.999n", float(np.mean(m > 0.999 * np.abs(n))))
print("frac n<0", float(np.mean(n < 0)))
rng = np.random.default_rng(7)
d = rng.normal(size=rho.shape)
if mode == "relative":
    d = d * np.asarray(rho)
if mode == "charge":
    d[1:] = 0
d = jnp.asarray(d / np.linalg.norm(d))


def f(q):
    return calc.potential(q).v_scf


_, jv = jax.jvp(f, (rho,), (d,))
size = max(1.0, float(jnp.linalg.norm(rho)))
for e in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
    h = e * size
    fd = (f(rho + h * d) - f(rho - h * d)) / (2 * h)
    print("dv", e, float(jnp.linalg.norm(fd - jv) / jnp.linalg.norm(jv)), flush=True)
