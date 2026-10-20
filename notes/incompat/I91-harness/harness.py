"""I91 harness: the augmented shift current's second derivative, two checks.

Run on D22 through slot.sh with the worktree as cwd: ``python harness_i91.py <cell> [nbnd...]``.

1. ``L^{ac}`` (VelocityOperator.augmentation_second) against the second
   difference in ``k'`` of the overlap primitive ``<psi|S(k, k')|psi>`` at
   frozen coefficients, ``q_ij(b)`` from ``augmentation_at_q``; and ``K^a``
   against the first difference, as the control.
2. ``r^{c;a}`` from the sum rule with the generalised ``v`` and ``w`` against a
   parallel-transport difference of the generalised dipole whose phases come
   from the same primitive, on one frozen sphere, swept in ``nbnd`` to the
   complete basis; with the new term switched off (bare ``w``) beside it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "tests"))

from exact_reference import exact_eigenpairs_all  # noqa: E402

from defuqumat.io.pwin import read_pw_input  # noqa: E402
from defuqumat.pseudo.upf import read_upf  # noqa: E402
from defuqumat.response.photocurrent import dipole_matrix, generalized_derivative  # noqa: E402
from defuqumat.response.velocity import (  # noqa: E402
    VelocityOperator, generalised_second, with_connections,
)
from defuqumat.scf.driver import Calculation, run_scf  # noqa: E402
from defuqumat.system.builder import build_system  # noqa: E402
from defuqumat.system.kpoints import KPoints  # noqa: E402
from defuqumat.topology.augmentation import augmentation_at_q  # noqa: E402

CASES = ROOT / "tests" / "data" / "qe"
PSEUDO = ROOT / "tests" / "data" / "pseudo"
GENERIC_K = np.array([0.11, 0.23, 0.31])


def setup(name):
    system = build_system(read_pw_input(CASES / name))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    result = run_scf(system, pseudos, conv_thr=1.0e-10)
    kp = KPoints.from_crystal(GENERIC_K.reshape(1, 3), np.ones(1), system.cell,
                              precision=system.kpoints.precision)
    calc = Calculation(system, pseudos).at_kpoints(kp)
    v_scf = calc.potential(result.density).v_scf
    ddd = calc.onecenter(result.becsum)[1] if calc.is_paw else None
    k0 = np.asarray(kp.cartesian(system.cell))
    return calc, v_scf, ddd, k0


def states_at(calc, v_scf, ddd, kcart, nbnd):
    moved = calc.at_kcart(jnp.asarray(kcart))
    e, psi = exact_eigenpairs_all(moved.hamiltonian(v_scf, ddd, None)[0], nbnd)
    return np.asarray(e)[0], np.asarray(psi)[0]


def operator(calc, v_scf, ddd, kcart):
    # The calculation itself moved to ``kcart``: ``augmentation_connection``
    # reads the calculation's own projectors, so an operator handed only a
    # ``kcart`` would mix ``vkb`` at two k-points.
    return VelocityOperator(calc.at_kcart(jnp.asarray(kcart).reshape(1, 3)), v_scf, ddd)


def pieces(calc, v_scf, ddd, kcart, nbnd):
    e, psi = states_at(calc, v_scf, ddd, kcart, nbnd)
    op = operator(calc, v_scf, ddd, kcart)
    p = jnp.asarray(psi)[None, None]
    ee = jnp.asarray(e)[None, None]
    tangents = op.tangent_elements(p, ee)
    k = op.connections(p)
    v = with_connections(tangents, ee, k)
    h2 = op.second_matrix_elements(p)
    lsec = op.augmentation_seconds(p)
    w = generalised_second(h2, v, k, lsec, ee)
    sq = lambda x: None if x is None else np.asarray(x)[..., 0, 0, :, :]  # drop spin, k
    return e, psi, sq(v), sq(w), sq(h2), sq(k), sq(lsec)


def closed_curvature(calc, v_scf, ddd, kcart, psi, h=1.0e-4):
    """``D_a K^c - K^{a dag} K^c`` at frozen ``psi``, ``(3, 3, nb, nb)``, by central differences."""
    p = jnp.asarray(psi)[None, None]
    eye = np.eye(3)
    k0 = operator(calc, v_scf, ddd, kcart).connections(p)
    k0 = np.asarray(k0)[:, 0, 0]
    out = np.zeros((3, 3) + k0.shape[1:], complex)
    for a in range(3):
        kp = np.asarray(operator(calc, v_scf, ddd, kcart + h * eye[a]).connections(p))[:, 0, 0]
        km = np.asarray(operator(calc, v_scf, ddd, kcart - h * eye[a]).connections(p))[:, 0, 0]
        derivative = (kp - km) / (2 * h)
        for c in range(3):
            out[a, c] = derivative[c] - np.conj(k0[a].T) @ k0[c]
    return out


def primitive(calc, psi_left, psi_right, k_left, k_right):
    """``<psi_l(k)|S(k, k')|psi_r(k')>`` on the frozen sphere, [n, m]."""
    vl = np.asarray(calc.at_kcart(jnp.asarray(k_left).reshape(1, 3)).projectors.vkb)[0]
    vr = np.asarray(calc.at_kcart(jnp.asarray(k_right).reshape(1, 3)).projectors.vkb)[0]
    q = augmentation_at_q(calc, (np.asarray(k_right) - np.asarray(k_left)).reshape(3))
    bl = vl.conj().T @ psi_left.T  # (nkb, nb)
    br = vr.conj().T @ psi_right.T
    plain = psi_left.conj() @ psi_right.T
    return plain if q is None else plain + bl.conj().T @ np.asarray(q) @ br


def check_l(calc, v_scf, ddd, k0, nb=8):
    e, psi, v, w, h2, kk, lsec = pieces(calc, v_scf, ddd, k0, nb)
    if kk is None:
        print("norm-conserving: no K, no L")
        return
    eye = np.eye(3)
    for h in (2e-3, 1e-3):
        def m(da, dc):
            return primitive(calc, psi, psi, k0[0], k0[0] + da + dc)
        k_err, l_err = 0.0, 0.0
        for a in range(3):
            d1 = (m(h * eye[a], 0) - m(-h * eye[a], 0)) / (2 * h)
            k_err = max(k_err, float(np.max(np.abs(d1 - kk[a]))))
            for c in range(3):
                d2 = (m(h * eye[a], h * eye[c]) - m(h * eye[a], -h * eye[c])
                      - m(-h * eye[a], h * eye[c]) + m(-h * eye[a], -h * eye[c])) / (4 * h * h)
                l_err = max(l_err, float(np.max(np.abs(d2 - lsec[a, c]))))
        print(f"h={h:.1e}  |K - dM| = {k_err:.3e} (max|K| {np.abs(kk).max():.3e})  "
              f"|L - d2M| = {l_err:.3e} (max|L| {np.abs(lsec).max():.3e})", flush=True)
    # the quadrupole's share of L, for scale
    print("asymmetry of w before symmetrisation is not reported; w sym by construction")


def transported(ref_psi, psi, e, v, comp, overlap):
    phase = np.diagonal(overlap)
    phase = phase / np.abs(phase)
    r = np.asarray(dipole_matrix(e[None], v[:, None]))[comp, 0]
    return r * (phase[:, None] * np.conj(phase)[None, :])


def check_sum_rule(calc, v_scf, ddd, k0, sweeps, a=0, c=1, delta=1e-3):
    npw = int(calc.basis.npwx)
    print(f"npwx = {npw}", flush=True)
    step = np.zeros(3)
    step[a] = delta
    base = pieces(calc, v_scf, ddd, k0, npw)
    plus = pieces(calc, v_scf, ddd, k0 + step, npw)
    minus = pieces(calc, v_scf, ddd, k0 - step, npw)
    e, psi = base[0], base[1]
    o_p = primitive(calc, psi, plus[1], k0[0], k0[0] + step)
    o_m = primitive(calc, psi, minus[1], k0[0], k0[0] - step)
    diff = (transported(psi, plus[1], plus[0], plus[2], c, o_p)
            - transported(psi, minus[1], minus[0], minus[2], c, o_m)) / (2 * delta)
    window = diff[:8, :8]
    large = np.abs(window) > 1e-2 * np.abs(window).max()
    print(f"large elements in window: {int(large.sum())}, max {np.abs(window).max():.3e}")
    v, w, h2 = base[2], base[3], base[4]
    kk, lsec = base[5], base[6]
    norm = None
    closed = None
    if kk is not None:
        moved = calc.at_kcart(jnp.asarray(k0))
        smat = np.asarray(moved.hamiltonian(v_scf, ddd, None)[0].overlap_matrix(0))
        norm = float(np.max(np.abs(np.conj(psi) @ smat @ psi.T - np.eye(len(e)))))
        print(f"max |psi^dag S psi - 1| = {norm:.2e}", flush=True)
        closed = closed_curvature(calc, v_scf, ddd, k0, psi)
        print(f"max|L_closed - L_PAW| = {np.abs(closed - lsec).max():.3e}, "
              f"asymmetry of L_closed {np.abs(closed - np.swapaxes(closed, 0, 1)).max():.3e}",
              flush=True)

    def residue(bands, which):
        if which == "full":
            ww = w[..., :bands, :bands]
            if kk is not None:
                ww = np.asarray(generalised_second(
                    jnp.asarray(h2[..., :bands, :bands]), jnp.asarray(v[..., :bands, :bands]),
                    jnp.asarray(kk[..., :bands, :bands]), jnp.asarray(lsec[..., :bands, :bands]),
                    jnp.asarray(e[:bands])))
        elif which == "closed":
            ww = np.asarray(generalised_second(
                jnp.asarray(h2[..., :bands, :bands]), jnp.asarray(v[..., :bands, :bands]),
                jnp.asarray(kk[..., :bands, :bands]), jnp.asarray(closed[..., :bands, :bands]),
                jnp.asarray(e[:bands])))
        elif which == "bare":
            ww = h2[..., :bands, :bands]
        elif which == "no-L":
            ww = np.asarray(generalised_second(
                jnp.asarray(h2[..., :bands, :bands]), jnp.asarray(v[..., :bands, :bands]),
                jnp.asarray(kk[..., :bands, :bands]), 0.0 * jnp.asarray(lsec[..., :bands, :bands]),
                jnp.asarray(e[:bands])))
        analytic = np.asarray(generalized_derivative(
            e[None, :bands], v[:, None, :bands, :bands], ww[:, :, None]))[a, c, 0][:8, :8]
        return float(np.max(np.abs(analytic - window)[large] / np.abs(window)[large]))

    for bands in list(sweeps) + [npw]:
        line = f"nbnd={bands:4d}  full {residue(bands, 'full'):.3e}"
        if kk is not None:
            line += (f"  closed {residue(bands, 'closed'):.3e}"
                     f"  bare-w {residue(bands, 'bare'):.3e}  no-L {residue(bands, 'no-L'):.3e}")
        print(line, flush=True)


if __name__ == "__main__":
    name = sys.argv[1]
    sweeps = [int(x) for x in sys.argv[2:]]
    t0 = time.time()
    calc, v_scf, ddd, k0 = setup(name)
    print(f"{name}: scf {time.time() - t0:.1f} s; ultrasoft={calc.is_ultrasoft} paw={calc.is_paw}",
          flush=True)
    import os

    deltas = [float(x) for x in os.environ.get("DELTAS", "1e-3").split(",")]
    if os.environ.get("SKIP_L") != "1":
        check_l(calc, v_scf, ddd, k0)
    for index, delta in enumerate(deltas):
        print(f"--- delta = {delta:.2e}", flush=True)
        check_sum_rule(calc, v_scf, ddd, k0, sweeps if index == 0 else [], delta=delta)
    print(f"total {time.time() - t0:.1f} s")
