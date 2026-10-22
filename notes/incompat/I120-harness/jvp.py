"""I120: the Sternheimer Jacobian against a cold central difference of F.

    python3 jvp.py <case> [--kmesh "2 2 2 0 0 0"] [--ecut 20] [--buffer 4]
                          [--vmode random|relative] [--steps 1e-3,1e-4,1e-5,1e-6]

The action is built by constructing SternheimerJacobian directly, so the
residual's guard is not in the way while the regime is being measured. The
reference is make_residual(calculation, nbnd + buffer, 1e-13), cold (psi0 None),
as P150's p150_jvp.py. Printed: |J v - ref| / |ref + v| per step, the same with
the term under study switched off (meta: the tau block of v zeroed on input,
which is the action without dH/dtau; noncollinear: the magnetization
components of dV zeroed), and the CG step count.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import jax.numpy as jnp  # noqa: E402

from defuqumat.io.pwin import parse_pw_input  # noqa: E402
from defuqumat.pseudo import read_upf  # noqa: E402
from defuqumat.scf.driver import Calculation, run_scf  # noqa: E402
from defuqumat.scf.residual import SternheimerJacobian, make_residual  # noqa: E402
from defuqumat.system import build_system  # noqa: E402

CASES = ROOT / "tests" / "data" / "qe"

#: ``--extra`` names: lines put into ``&system`` of the committed input.
EXTRA = {
    "tb09": "input_dft = 'tb09', nbnd = 8",
}

#: ``--sub`` names: text replacements in the committed input.
SUBS = {
    # h-fcc-spiral-scan.in at q = 0 with the spiral switched off: a bulk
    # ferromagnet in the spinor formalism, nspin_mag = 4, no vacuum.
    "ferro": [("    spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = 0.0\n", ""),
              ("degauss = 0.01", "degauss = 0.05")],
    # The same cell without a moment: a nonmagnetic spinor metal, the control.
    "nonmag": [("    spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = 0.0\n", ""),
               ("degauss = 0.01", "degauss = 0.05"),
               ("starting_magnetization(1) = 0.9", "starting_magnetization(1) = 0.0")],
}


def edited(case, kmesh=None, ecut=None, extra=None, sub=None):
    text = (CASES / f"{case}.in").read_text()
    for old, new in SUBS.get(sub, []) if sub else []:
        assert old in text, old
        text = text.replace(old, new)
    lines = text.splitlines()
    if kmesh is not None:
        i = next(n for n, l in enumerate(lines) if l.strip().upper().startswith("K_POINTS"))
        lines[i + 1] = " " + kmesh
    if ecut is not None:
        lines = [re.sub(r"ecutwfc\s*=\s*[0-9.dDeE+-]+", f"ecutwfc={ecut}", l)
                 for l in lines]
    if extra:
        extra = EXTRA.get(extra, extra)
        i = next(n for n, l in enumerate(lines) if l.strip().lower() == "&system")
        lines.insert(i + 1, "    " + extra)
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("--kmesh", default=None)
    ap.add_argument("--ecut", type=float, default=None)
    ap.add_argument("--extra", default=None)
    ap.add_argument("--sub", default=None)
    ap.add_argument("--buffer", type=int, default=4)
    ap.add_argument("--vmode", default="random")
    ap.add_argument("--steps", default="1e-3,1e-4,1e-5,1e-6")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--conv", type=float, default=1e-12)
    args = ap.parse_args()
    t0 = time.time()
    text = edited(args.case, args.kmesh, args.ecut, args.extra, args.sub)
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(ROOT / "tests" / "data" / "pseudo" / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=args.conv,
                     max_iterations=200)
    assert result.converged
    nbnd = int(np.asarray(result.eigenvalues_by_spin).shape[-1])
    residual = make_residual(calculation, nbnd, float(result.ethr))
    tau = None
    if residual.tau_shape is not None:
        tau = result.tau
    x = np.asarray(residual.pack(result.density, tuple(result.becsum or ()), None, tau))
    print(f"scf done {time.time()-t0:.1f}s nk={len(system.kpoints.weights)} nbnd={nbnd} "
          f"size={x.size} nspin_mag={calculation.nspin_mag} tau={residual.tau_shape}",
          flush=True)
    r0, _ = residual.residual(x, result.wavefunctions)
    print(f"|r(x)|/|x| = {np.linalg.norm(np.asarray(r0))/np.linalg.norm(x):.2e}", flush=True)

    rng = np.random.default_rng(7)
    v = rng.normal(size=x.shape)
    if args.vmode in ("relative", "symmetric"):
        v = v * x
    if args.vmode == "symmetric":
        # The crystal's symmetry, field by field: Tran-Blaha's c averages
        # |grad rho|, a cone where the group forces grad rho = 0, so F has a
        # derivative there only along a direction that keeps it zero.
        rho_v, becsum_v, _, tau_v = residual.unpack(v)
        rho_v = calculation.symmetrize(rho_v)
        tau_v = None if tau_v is None else calculation.symmetrize(tau_v)
        v = np.array(residual.flatten(rho_v, becsum_v, None, tau_v))
    v /= np.linalg.norm(v)
    size = max(1.0, float(np.linalg.norm(x)))

    rho, becsum_, _, tau_x = residual.unpack(x)
    potential = calculation.potential(rho, tau=tau_x)
    _, ddd = calculation.onecenter(becsum_, None if tau_x is None else potential.meta_c)
    ham = calculation.hamiltonian(potential.v_scf, ddd)
    eig, psi = calculation.diagonalize(ham, nbnd, result.wavefunctions, residual.ethr)
    jac = SternheimerJacobian(residual, x, eig, psi, args.threshold,
                              built=(potential, ham))
    t1 = time.time()
    jv = jac(v)
    cg = int(np.max(np.asarray(jac.last_iterations)))
    print(f"action {time.time()-t1:.1f}s CG {cg}", flush=True)

    off = None
    n_tau = 0 if residual.tau_shape is None else int(np.prod(residual.tau_shape))
    if n_tau:
        vt = v.copy()
        vt[x.size - n_tau:] = 0.0
        # The action without dH/dtau: the tau block of v no longer moves H,
        # while the identity's -v still sees it.
        off = jac(vt) - (v - vt)
    elif calculation.nspin_mag == 4:
        orig = jac._dv

        def dv_charge_only(drho, dtau=None):
            dv, dc = orig(drho, dtau)
            return jnp.asarray(dv).at[1:].set(0.0), dc

        jac._dv = dv_charge_only
        off = jac(v)
        jac._dv = orig

    exact = make_residual(calculation, nbnd + args.buffer, 1e-13)
    rows = []
    for step in [float(s) for s in args.steps.split(",")]:
        t2 = time.time()
        ref = np.asarray(exact.jvp_finite_difference(x, v, None, epsilon=step * size))
        scale = np.linalg.norm(ref + v)
        row = {"step": step, "err": float(np.linalg.norm(jv - ref) / scale)}
        if off is not None:
            row["err_off"] = float(np.linalg.norm(off - ref) / scale)
        if n_tau:
            sl = slice(x.size - n_tau, None)
            row["err_tau_rows"] = float(np.linalg.norm((jv - ref)[sl])
                                        / np.linalg.norm((ref + v)[sl]))
        rows.append(row)
        print(json.dumps(row), f"{time.time()-t2:.1f}s", flush=True)
    errs = [r["err"] for r in rows]
    print("RESULT", json.dumps({
        "case": args.case, "kmesh": args.kmesh, "ecut": args.ecut, "extra": args.extra, "sub": args.sub,
        "median_err": float(np.median(errs)), "best_err": float(min(errs)),
        "rows": rows, "cg": cg, "seconds": time.time() - t0}), flush=True)


if __name__ == "__main__":
    main()
