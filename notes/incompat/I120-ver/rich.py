"""Verifier harness: action vs central differences and their Richardson extrapolation.

python3 rich.py <case> [jvp.py options] --steps ... --refethr 1e-13 --seed 7 --vmode
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

WT = Path(sys.argv.pop(1))
sys.path.insert(0, str(WT / "notes/incompat/I120-harness"))
sys.path.insert(0, str(WT))
from jvp import ROOT, edited  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from defuqumat.io.pwin import parse_pw_input  # noqa: E402
from defuqumat.pseudo import read_upf  # noqa: E402
from defuqumat.scf.driver import Calculation, run_scf  # noqa: E402
from defuqumat.scf.residual import SternheimerJacobian, make_residual  # noqa: E402
from defuqumat.system import build_system  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("--kmesh", default=None)
ap.add_argument("--ecut", type=float, default=None)
ap.add_argument("--extra", default=None)
ap.add_argument("--sub", default=None)
ap.add_argument("--buffer", type=int, default=4)
ap.add_argument("--vmode", default="random")
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--steps", default="1e-3,1e-4,1e-5,1e-6")
ap.add_argument("--refethr", type=float, default=1e-13)
ap.add_argument("--conv", type=float, default=1e-12)
ap.add_argument("--guard", action="store_true")
ap.add_argument("--store", default=None)
args = ap.parse_args()
t0 = time.time()
text = edited(args.case, args.kmesh, args.ecut, args.extra, args.sub)
system = build_system(parse_pw_input(text))
pseudos = tuple(read_upf(ROOT / "tests" / "data" / "pseudo" / s.pseudo_file)
                for s in system.structure.species)
calculation = Calculation(system, pseudos)
kw = {} if args.store is None else {"wfc_store": args.store}
result = run_scf(system, pseudos, calculation=calculation, conv_thr=args.conv,
                 max_iterations=300, **kw)
from defuqumat.scf.streaming import is_host_store
print("host store:", is_host_store(result.wavefunctions), type(result.wavefunctions), flush=True)
print("converged", result.converged, flush=True)
nbnd = int(np.asarray(result.eigenvalues_by_spin).shape[-1])
residual = make_residual(calculation, nbnd, float(result.ethr))
tau = result.tau if residual.tau_shape is not None else None
x = np.asarray(residual.pack(result.density, tuple(result.becsum or ()), None, tau))
print(f"scf {time.time()-t0:.1f}s nk={len(system.kpoints.weights)} nbnd={nbnd} size={x.size} "
      f"nspin_mag={calculation.nspin_mag} tau={residual.tau_shape}", flush=True)
rng = np.random.default_rng(args.seed)
v = rng.normal(size=x.shape)
if args.vmode in ("relative", "symmetric"):
    v = v * x
if args.vmode == "symmetric":
    rho_v, becsum_v, _, tau_v = residual.unpack(v)
    rho_v = calculation.symmetrize(rho_v)
    tau_v = None if tau_v is None else calculation.symmetrize(tau_v)
    v = np.array(residual.flatten(rho_v, becsum_v, None, tau_v))
if args.vmode == "scalarrel":
    # one random factor per grid point shared by (n, m): zeta does not move
    rho_x, becsum_x, _, tau_x = residual.unpack(x)
    g = rng.normal(size=np.asarray(rho_x).shape[1:])
    rho_v = jnp.asarray(rho_x) * g[None]
    v = np.array(residual.flatten(rho_v, becsum_x, None, tau_x))
if args.vmode == "charge":
    # only the charge channel: zero magnetization rows
    rho_v, becsum_v, _, tau_v = residual.unpack(v)
    rho_v = jnp.asarray(rho_v).at[1:].set(0.0)
    v = np.array(residual.flatten(rho_v, becsum_v, None, tau_v))
v /= np.linalg.norm(v)
size = max(1.0, float(np.linalg.norm(x)))
if args.guard:
    jac = residual.sternheimer_jacobian(x, result.wavefunctions)
else:
    rho, becsum_, _, tau_x = residual.unpack(x)
    potential = calculation.potential(rho, tau=tau_x)
    _, ddd = calculation.onecenter(becsum_, None if tau_x is None else potential.meta_c)
    ham = calculation.hamiltonian(potential.v_scf, ddd)
    eig, psi = calculation.diagonalize(ham, nbnd, result.wavefunctions, residual.ethr)
    jac = SternheimerJacobian(residual, x, eig, psi, None, built=(potential, ham))
jv = jac(v)
print("CG", int(np.max(np.asarray(jac.last_iterations))), flush=True)
exact = make_residual(calculation, nbnd + args.buffer, args.refethr)
D = {}
for step in [float(s) for s in args.steps.split(",")]:
    D[step] = np.asarray(exact.jvp_finite_difference(x, v, None, epsilon=step * size))
    scale = np.linalg.norm(D[step] + v)
    print(json.dumps({"step": step, "err": float(np.linalg.norm(jv - D[step]) / scale)}),
          flush=True)
steps = sorted(D, reverse=True)
for a, b in zip(steps, steps[1:]):
    r = (b / a) ** 2
    R = (D[b] - r * D[a]) / (1 - r)
    scale = np.linalg.norm(R + v)
    print(json.dumps({"richardson": [a, b], "err": float(np.linalg.norm(jv - R) / scale),
                      "D_a_vs_D_b": float(np.linalg.norm(D[a] - D[b]) / scale)}), flush=True)
print("done", time.time() - t0, flush=True)
