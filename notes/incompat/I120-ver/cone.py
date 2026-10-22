import sys
from pathlib import Path
import numpy as np
WT = Path(sys.argv[1])
sys.path.insert(0, str(WT / "notes/incompat/I120-harness"))
sys.path.insert(0, str(WT))
from jvp import ROOT, edited  # noqa
import jax.numpy as jnp
import defuqumat.xc.functional as fmod
from defuqumat.xc import mgga
from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import Calculation, run_scf
from defuqumat.system import build_system

kmesh = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "full" else None
ecut = float(sys.argv[3]) if len(sys.argv) > 3 else None
text = edited("si2-tb09", kmesh, ecut)
system = build_system(parse_pw_input(text))
pseudos = tuple(read_upf(ROOT / "tests/data/pseudo" / s.pseudo_file) for s in system.structure.species)
calc = Calculation(system, pseudos)
r = run_scf(system, pseudos, calculation=calc, conv_thr=1e-10, max_iterations=200)
print("converged", r.converged, "E", r.total_energy, "c", float(r.meta_c) if hasattr(r, "meta_c") else None)
cap = {}
import jax
cls = type(calc.functional)
orig = cls.meta_c
def rec(self, rho, grad_rho):
    cap["rho"], cap["g"] = np.asarray(rho), np.asarray(grad_rho)
    return orig(self, rho, grad_rho)
cls.meta_c = rec
with jax.disable_jit():
    p = calc.potential(jnp.asarray(r.density), tau=jnp.asarray(r.tau))
cls.meta_c = orig
rho, g = cap["rho"], cap["g"]
sigma = np.sum(g ** 2, axis=0)
print("grid", sigma.shape, "min sigma", sigma.min())
s = np.sort(sigma.ravel())
print("smallest 12 sigma", s[:12])
print("count <= 1e-20", int(np.sum(sigma <= 1e-20)), "count < 1e-12", int(np.sum(sigma < 1e-12)))
new = float(mgga.tb09_coefficient(rho, g))
mag = np.sqrt(sigma)
act = rho > mgga.META_RHO_THRESHOLD
old_ratio = np.where(act, mag / np.where(act, rho, 1), 0)
alpha, beta = None, None
print("c new", repr(new), "c in potential", repr(float(p.meta_c)))
# old formula: identical except magnitude = sqrt(sigma)
import inspect
src = inspect.getsource(mgga.tb09_coefficient)
print("mean ratio new-old (abs)", float(np.abs(np.mean(np.where(act, np.where(sigma > 1e-20, mag, 0)/np.where(act, rho, 1), 0)) - np.mean(old_ratio))))
