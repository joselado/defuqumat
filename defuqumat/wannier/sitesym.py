"""Symmetry-adapted Wannier functions: Wannier90's site-symmetry engine.

Sakuma, "Symmetry-adapted Wannier functions in the maximal localization
procedure", PRB 87, 235109 (2013), arXiv:1306.0032; Wannier90 3.1.0's
``src/sitesym.F90`` and its hooks in ``disentangle.F90``, ``overlap.F90`` and
``wannierise.F90`` (``lsitesymmetry``). The constraint is

    U(g k) = d(g, k) U(k) D(g, k)^dagger,

with ``d_mn(g, k) = <psi_m,gk|g psi_nk>`` the operation on the Bloch states
(``d_matrix_band``) and ``D(g, k)`` the operation on the trial orbitals
(``d_matrix_wann``), so that ``g w_n = sum_m w_m D_mn``: the Wannier functions
transform under each operation of the space group exactly as the trial
orbitals do. Only one k-point of each star (``ir2ik``) is optimised; the rest
of the mesh is filled in through the star. Both matrices come from a ``.dmn``
file (``pw2wannier90.x`` ``compute_dmn``) or from
:mod:`defuqumat.wannier.dmatrix`.

This is the user's own NumPy port of ``sitesym.F90`` (the ``wannierpy``
project, ``wannier90/_engine/sitesym.py``), which P162 left behind when it
copied the engine; it is brought back here with its hooks kept as optional
``sym=None`` arguments of the engine's routines. Validated against
``wannier90.x`` on Wannier90's own test fixtures, ``PLAN.md`` P164.

Not carried, as in Wannier90 (``disentangle.F90:104``, "frozen window not
implemented yet in symmetry-adapted mode"): a frozen window. The frozen
columns of ``u_matrix_opt`` are one-hot, and :func:`symmetrize_u_matrix`
mixes them with the free ones.

Indices are 0-based; the ``.dmn`` file's are 1-based and are shifted once, in
:func:`from_dmn`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import warnings

import numpy as np

__all__ = ["SymmetryData", "from_dmn", "slim_d_matrix_band", "symmetrize_u_matrix",
           "symmetrize_gradient", "symmetrize_rotation", "symmetrize_zmatrix",
           "dis_extract_symmetry", "dis_extract_symmetric", "wannier_gauge"]


@dataclass
class SymmetryData:
    nsymmetry: int
    nkptirr: int
    num_kpts: int
    ik2ir: np.ndarray  # (num_kpts,) 0-based into 0..nkptirr-1
    ir2ik: np.ndarray  # (nkptirr,) 0-based into 0..num_kpts-1
    kptsym: np.ndarray  # (nsymmetry, nkptirr): the index of g k, 0-based
    d_matrix_wann: np.ndarray  # (num_wann, num_wann, nsymmetry, nkptirr)
    d_matrix_band: np.ndarray  # (num_bands, num_bands, nsymmetry, nkptirr)
    symmetrize_eps: float = 1.0e-3
    #: Refuse a band set that cuts a degenerate multiplet (our own states), or
    #: only warn as ``wannier90.x`` runs on regardless (a ``.dmn`` read from a
    #: file: the H3S fixture's 20 bands cut one at Gamma, |d^+ d - 1| = 0.97).
    refuse_cut: bool = True


def from_dmn(dmn: dict, num_wann: int, symmetrize_eps: float = 1.0e-3) -> SymmetryData:
    """A :class:`SymmetryData` from :func:`defuqumat.wannier.files.read_dmn`'s
    dictionary, whose index tables are still 1-based."""
    if dmn["d_matrix_wann"].shape[0] != num_wann:
        raise ValueError(f"the .dmn file's D-matrices are for num_wann = "
                         f"{dmn['d_matrix_wann'].shape[0]}, not {num_wann}")
    return SymmetryData(
        nsymmetry=dmn["nsymmetry"], nkptirr=dmn["nkptirr"], num_kpts=dmn["num_kpts"],
        ik2ir=dmn["ik2ir"] - 1, ir2ik=dmn["ir2ik"] - 1, kptsym=dmn["kptsym"] - 1,
        d_matrix_wann=dmn["d_matrix_wann"].copy(), d_matrix_band=dmn["d_matrix_band"].copy(),
        symmetrize_eps=float(symmetrize_eps), refuse_cut=False)


def wannier_gauge(sym: SymmetryData) -> SymmetryData:
    """``sitesym_replace_d_matrix_band``: once the states are ``num_wann`` wide,
    the operation on them is ``D`` on both sides."""
    return replace(sym, d_matrix_band=sym.d_matrix_wann.copy())


def slim_d_matrix_band(sym: SymmetryData, lwindow: np.ndarray) -> SymmetryData:
    """``sitesym_slim_d_matrix_band``: ``d`` from absolute to window-relative rows.

    Defined at each star's representative only; every point of a star must
    have the same window, which is asserted here (the Fortran assumes it): an
    edge at round-off from a band can put it inside at ``k`` and outside at
    ``g k``.
    """
    out = np.zeros_like(sym.d_matrix_band)
    for ir in range(sym.nkptirr):
        ik = sym.ir2ik[ir]
        nindx = np.nonzero(lwindow[:, ik])[0]
        for isym in range(sym.nsymmetry):
            if not np.array_equal(lwindow[:, sym.kptsym[isym, ir]], lwindow[:, ik]):
                raise ValueError(
                    "the outer window holds different bands at k and g k for a star of the "
                    f"mesh (irreducible point {ir}, operation {isym}): a window edge sits on "
                    "a band, which the site-symmetry mode cannot treat; move dis_win")
        nb = len(nindx)
        out[:nb, :nb, :, ir] = sym.d_matrix_band[np.ix_(nindx, nindx)][:, :, :, ir]
        check_band_unitarity(out[:nb, :nb, :, ir], ik, "the outer window", sym.refuse_cut)
    return replace(sym, d_matrix_band=out)


#: The band representation of a set that cuts no degenerate multiplet is
#: unitary to the eigensolver's accuracy (1e-14 measured on silicon); above
#: this the set is cut.
UNITARITY_TOLERANCE = 1.0e-4


def check_band_unitarity(d: np.ndarray, ik: int, what: str, refuse: bool = True) -> float:
    """Refuse a band set that cuts a degenerate multiplet at ``k``: ``d`` is then
    not unitary, and no ``U`` satisfies the constraint. ``d`` is ``(n, n, nsym)``.
    With ``refuse=False`` the first such operation is a warning instead."""
    n = d.shape[0]
    worst = 0.0
    for s in range(d.shape[2]):
        err = float(np.abs(d[:, :, s].conj().T @ d[:, :, s] - np.eye(n)).max()) if n else 0.0
        worst = max(worst, err)
        if err > UNITARITY_TOLERANCE:
            message = (
                f"the band representation is not unitary at k-point {ik} under operation {s} "
                f"(|d^+ d - 1| = {err:.2e}): {what} ({n} bands there) cuts a degenerate "
                "multiplet; move the window edge (or num_bands) into a gap at every k")
            if refuse:
                raise ValueError(message)
            warnings.warn(message, stacklevel=3)
            return err
    return worst


def _orthogonalize(u: np.ndarray, n: int) -> np.ndarray:
    """``orthogonalize_u``: Lowdin on ``u[:n, :]`` by an SVD."""
    m = u.shape[1]
    if n < m:
        raise ValueError("orthogonalize_u: n < m")
    Z, _, Vh = np.linalg.svd(u[:n, :], full_matrices=True)
    out = u.copy()
    out[:n, :] = Z[:, :m] @ Vh
    return out


def symmetrize_ukirr(ir: int, umat: np.ndarray, sym: SymmetryData, n: int,
                     niter: int = 100) -> np.ndarray:
    """``symmetrize_ukirr``: ``U(k) <- (1/N) sum_{g k = k} d^dagger U(k) D``, made
    unitary again, until ``U(k)`` is invariant under the little group of ``k``."""
    ik = sym.ir2ik[ir]
    little_group = np.nonzero(sym.kptsym[:, ir] == ik)[0]
    ngk = len(little_group)
    num_wann = umat.shape[1]
    if ngk == 1:
        return _orthogonalize(umat, n)
    u = umat.copy()
    for iteration in range(niter):
        usum = np.zeros_like(u)
        cmat2 = ngk * np.eye(num_wann, dtype=complex)
        for isym in little_group:
            cmat_sub = u[:n, :] @ sym.d_matrix_wann[:, :, isym, ir]
            cmat = sym.d_matrix_band[:n, :n, isym, ir].conj().T @ cmat_sub
            usum[:n, :] += cmat
            cmat2 -= u[:n, :].conj().T @ cmat
        diff = float(np.sum(np.abs(cmat2)))
        if diff < sym.symmetrize_eps:
            break
        if iteration == niter - 1:
            raise ValueError(
                f"symmetrize_ukirr: not converged (diff = {diff}, symmetrize_eps = "
                f"{sym.symmetrize_eps}): either the tolerance is too small or the trial "
                "orbitals do not carry the representation the bands carry at this k")
        usum[:n, :] /= ngk
        u = _orthogonalize(usum, n)
    return u


def symmetrize_u_matrix(umat: np.ndarray, sym: SymmetryData,
                        lwindow: np.ndarray | None = None) -> np.ndarray:
    """``sitesym_symmetrize_u_matrix``: symmetrise ``U`` at each representative
    and set ``U(g k) = d U(k) D^dagger`` on the rest of its star.

    With ``lwindow`` the rows are window-relative (``d`` slimmed by
    :func:`slim_d_matrix_band`); without, ``U`` is ``num_wann`` square and
    ``sym`` must be :func:`wannier_gauge`'s.
    """
    num_kpts = umat.shape[2]
    out = umat.copy()
    done = np.zeros(num_kpts, dtype=bool)
    for ir in range(sym.nkptirr):
        ik = sym.ir2ik[ir]
        done[ik] = True
        n = int(np.count_nonzero(lwindow[:, ik])) if lwindow is not None else out.shape[0]
        out[:, :, ik] = symmetrize_ukirr(ir, out[:, :, ik], sym, n)
        for isym in range(1, sym.nsymmetry):
            irk = sym.kptsym[isym, ir]
            if done[irk]:
                continue
            done[irk] = True
            cmat = sym.d_matrix_band[:n, :n, isym, ir] @ out[:n, :, ik]
            out[:n, :, irk] = cmat @ sym.d_matrix_wann[:, :, isym, ir].conj().T
    if not np.all(done):
        raise ValueError("symmetrize_u_matrix: a k-point is in no star of the mesh")
    return out


def symmetrize_gradient(imode: int, grad: np.ndarray, sym: SymmetryData) -> np.ndarray:
    """``sitesym_symmetrize_gradient``. Mode 1 (the raw gradient of ``wann_domega``)
    folds every k-point of a star onto its representative and zeroes the rest;
    mode 2 (the search direction) only averages over each little group, which
    both modes end with."""
    num_kpts = grad.shape[2]
    out = grad.copy()
    if imode == 1:
        found = np.zeros(num_kpts, dtype=bool)
        for ir in range(sym.nkptirr):
            ik = sym.ir2ik[ir]
            total = out[:, :, ik].copy()
            found[ik] = True
            for isym in range(1, sym.nsymmetry):
                irk = sym.kptsym[isym, ir]
                if found[irk]:
                    continue
                found[irk] = True
                cmat2 = sym.d_matrix_wann[:, :, isym, ir].conj().T @ out[:, :, irk]
                total = total + cmat2 @ sym.d_matrix_wann[:, :, isym, ir]
            out[:, :, ik] = total
        for ik in range(num_kpts):
            if sym.ir2ik[sym.ik2ir[ik]] != ik:
                out[:, :, ik] = 0.0
    for ir in range(sym.nkptirr):
        ik = sym.ir2ik[ir]
        little_group = np.nonzero(sym.kptsym[:, ir] == ik)[0]
        ngk = len(little_group)
        if ngk == 1:
            continue
        total = out[:, :, ik].copy()
        for isym in little_group:
            if isym == 0:
                continue
            cmat2 = out[:, :, ik] @ sym.d_matrix_wann[:, :, isym, ir]
            total = total + sym.d_matrix_wann[:, :, isym, ir].conj().T @ cmat2
        out[:, :, ik] = total / ngk
    return out


def symmetrize_rotation(urot: np.ndarray, sym: SymmetryData) -> np.ndarray:
    """``sitesym_symmetrize_rotation``: ``urot(g k) = D urot(k) D^dagger`` from
    each representative to the rest of its star."""
    num_kpts = urot.shape[2]
    out = urot.copy()
    done = np.zeros(num_kpts, dtype=bool)
    for ir in range(sym.nkptirr):
        ik = sym.ir2ik[ir]
        done[ik] = True
        for isym in range(1, sym.nsymmetry):
            irk = sym.kptsym[isym, ir]
            if irk == ik or done[irk]:
                continue
            done[irk] = True
            cmat2 = out[:, :, ik] @ sym.d_matrix_wann[:, :, isym, ir].conj().T
            out[:, :, irk] = sym.d_matrix_wann[:, :, isym, ir] @ cmat2
    if not np.all(done):
        raise ValueError("symmetrize_rotation: a k-point is in no star of the mesh")
    return out


def symmetrize_zmatrix(czmat: np.ndarray, sym: SymmetryData, lwindow: np.ndarray) -> np.ndarray:
    """``sitesym_symmetrize_zmatrix``: fold the star's other members and the
    little group into the representative's ``Z``."""
    num_kpts = czmat.shape[2]
    out = czmat.copy()
    found = np.zeros(num_kpts, dtype=bool)
    for ir in range(sym.nkptirr):
        ik = sym.ir2ik[ir]
        nd = int(np.count_nonzero(lwindow[:, ik]))
        found[ik] = True
        for isym in range(1, sym.nsymmetry):
            irk = sym.kptsym[isym, ir]
            if found[irk]:
                continue
            found[irk] = True
            cmat1 = out[:nd, :nd, irk] @ sym.d_matrix_band[:nd, :nd, isym, ir]
            out[:nd, :nd, ik] += sym.d_matrix_band[:nd, :nd, isym, ir].conj().T @ cmat1
        snapshot = out[:nd, :nd, ik].copy()
        for isym in range(1, sym.nsymmetry):
            if sym.kptsym[isym, ir] != ik:
                continue
            cmat1 = snapshot @ sym.d_matrix_band[:nd, :nd, isym, ir]
            out[:nd, :nd, ik] += sym.d_matrix_band[:nd, :nd, isym, ir].conj().T @ cmat1
        out[:nd, :nd, ik] /= int(np.count_nonzero(sym.kptsym[:, ir] == ik))
    return out


def _dlaev2(a: float, b: float, c: float):
    """LAPACK ``dlaev2``: ``rt1`` (the larger in modulus), ``rt2`` and the unit
    eigenvector ``(cs1, sn1)`` of ``rt1`` for ``[[a, b], [b, c]]``."""
    sm, df = a + c, a - c
    adf, tb = abs(df), b + b
    ab = abs(tb)
    acmx, acmn = (a, c) if abs(a) > abs(c) else (c, a)
    if adf > ab:
        rt = adf * np.sqrt(1.0 + (ab / adf) ** 2)
    elif adf < ab:
        rt = ab * np.sqrt(1.0 + (adf / ab) ** 2)
    else:
        rt = ab * np.sqrt(2.0)
    if sm < 0.0:
        rt1, sgn1 = 0.5 * (sm - rt), -1
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b
    elif sm > 0.0:
        rt1, sgn1 = 0.5 * (sm + rt), 1
        rt2 = (acmx / rt1) * acmn - (b / rt1) * b
    else:
        rt1, rt2, sgn1 = 0.5 * rt, -0.5 * rt, 1
    if df >= 0.0:
        cs, sgn2 = df + rt, 1
    else:
        cs, sgn2 = df - rt, -1
    if abs(cs) > ab:
        ct = -tb / cs
        sn1 = 1.0 / np.sqrt(1.0 + ct * ct)
        cs1 = ct * sn1
    elif ab == 0.0:
        cs1, sn1 = 1.0, 0.0
    else:
        tn = -cs / tb
        cs1 = 1.0 / np.sqrt(1.0 + tn * tn)
        sn1 = tn * cs1
    if sgn1 == sgn2:
        cs1, sn1 = -sn1, cs1
    return rt1, rt2, cs1, sn1


def _solve_2x2_generalized_eigh(H: np.ndarray, S: np.ndarray):
    """``H v = w S v`` for a 2x2 pair, as LAPACK's ``zhpgvx`` returns it.

    The phase of each eigenvector is not fixed by the problem, and
    :func:`dis_extract_symmetry` mixes the columns it builds through ``D``, so
    the phase reaches the subspace: on Wannier90's H3S fixture a random phase
    moves ``Omega_I`` by 1.3e-6 A^2 after ten steps, and ``numpy.linalg.eigh``'s
    convention left the total 2.1e-7 A^2 from ``wannier90.x``. So the chain of
    ``zhpgvx`` for ``n = 2`` is followed: ``zpptrf`` (``S = U^H U``),
    ``zhpgst`` (``C = U^-H H U^-1``), ``zhptrd`` (one reflector makes ``C_12``
    real), ``zsteqr`` (one ``dlaev2`` rotation, then an ascending sort) and
    ``ztpsv`` (``v = U^-1 z``).
    """
    u11 = np.sqrt(S[0, 0].real)
    u12 = S[0, 1] / u11
    u22 = np.sqrt(S[1, 1].real - abs(u12) ** 2)
    U = np.array([[u11, u12], [0.0, u22]], dtype=complex)
    Uinv = np.linalg.inv(U)
    C = Uinv.conj().T @ H @ Uinv
    d1, d2, a12 = C[0, 0].real, C[1, 1].real, C[0, 1]
    if a12.imag == 0.0:
        q, e = 1.0 + 0.0j, a12.real
    else:
        beta = -np.copysign(abs(a12), a12.real)
        q, e = a12 / beta, beta
    rt1, rt2, c, s = _dlaev2(d1, e, d2)
    Z = np.array([[c * q, -s * q], [s, c]], dtype=complex)
    w = np.array([rt1, rt2])
    if rt2 < rt1:
        w, Z = w[::-1], Z[:, ::-1]
    return w, Uinv @ Z


def dis_extract_symmetry(ik: int, n: int, zmat: np.ndarray, umat: np.ndarray, sym: SymmetryData,
                         niter: int = 50):
    """``sitesym_dis_extract_symmetry``: the Z-matrix step at a representative.

    In place of the eigenvectors of ``Z``, each column is moved within the plane
    of itself and its steepest-ascent step to the largest Rayleigh quotient (a
    2x2 generalised eigenproblem), and the block is made symmetric again after
    every pass. Returns ``(U, lambda)``, ``lambda = U^dagger Z U``.
    """
    num_bands, num_wann = umat.shape
    u = umat.copy()
    lam = np.zeros((num_wann, num_wann), dtype=complex)
    for _ in range(niter):
        ZU = zmat[:, :n] @ u[:n, :]
        ZU_full = np.zeros((num_bands, num_wann), dtype=complex)
        ZU_full[:n, :] = ZU
        lam = u[:n, :].conj().T @ ZU
        deltaU = np.zeros((num_bands, num_wann), dtype=complex)
        deltaU[:n, :] = ZU - u[:n, :] @ lam
        if np.sum(np.abs(deltaU[:n, :])) < 1.0e-10:
            return u, lam
        u_new = u.copy()
        for i in range(num_wann):
            ui, zui, dui = u[:n, i], ZU_full[:n, i], deltaU[:n, i]
            sp3 = float(np.vdot(dui, dui).real)
            if abs(sp3) < 1.0e-10:
                continue
            H = np.array([
                [np.vdot(ui, zui).real, np.vdot(zui, dui)],
                [np.conj(np.vdot(zui, dui)), float(np.vdot(dui, zmat[:n, :n] @ dui).real)],
            ], dtype=complex)
            S = np.array([
                [np.vdot(ui, ui).real, np.vdot(ui, dui)],
                [np.conj(np.vdot(ui, dui)), sp3],
            ], dtype=complex)
            _, V = _solve_2x2_generalized_eigh(H, S)
            u_new[:n, i] = V[0, 1] * ui + V[1, 1] * dui
        u = symmetrize_ukirr(sym.ik2ir[ik], u_new, sym, n)
    return u, lam


def dis_extract_symmetric(u_matrix_opt_init: np.ndarray, M_slim: np.ndarray, windows,
                          nnlist: np.ndarray, wb: np.ndarray, wbtot: float, num_wann: int,
                          dis_num_iter: int, dis_mix_ratio: float, dis_conv_tol: float,
                          dis_conv_window: int, sym: SymmetryData, lwindow: np.ndarray):
    """``dis_extract`` under ``lsitesymmetry`` (``disentangle.F90:1719-2005``).

    Only the representatives are extracted (:func:`dis_extract_symmetry`), the
    rest of each star follows from :func:`symmetrize_u_matrix`, and ``Z`` is
    kept as one zero-padded ``(num_bands, num_bands, num_kpts)`` stack so that
    :func:`symmetrize_zmatrix` can fold it. No frozen states (refused by the
    caller). ``sym`` is already window-sliced.
    """
    from defuqumat.wannier.disentangle import _womegai, _zmatrix

    num_bands, _, num_kpts = u_matrix_opt_init.shape
    u_matrix_opt = u_matrix_opt_init.copy()

    def zmatrix_stack(u):
        Z = np.zeros((num_bands, num_bands, num_kpts), dtype=complex)
        for k in range(num_kpts):
            nd = windows.ndimwin[k]
            Z[:nd, :nd, k] = _zmatrix(k, u, M_slim, windows, nnlist, wb, num_wann)
        return Z

    z_in = symmetrize_zmatrix(zmatrix_stack(u_matrix_opt), sym, lwindow)
    z_out = None
    history = []
    converged = False
    for iteration in range(1, dis_num_iter + 1):
        if iteration > 1:
            for ir in range(sym.nkptirr):
                ik = sym.ir2ik[ir]
                nd = windows.ndimwin[ik]
                mixed = dis_mix_ratio * z_out[:nd, :nd, ik] + (1.0 - dis_mix_ratio) * z_in[:nd, :nd, ik]
                z_in[:nd, :nd, ik] = 0.5 * (mixed + mixed.conj().T)
        womegai1 = 0.0
        for ir in range(sym.nkptirr):
            ik = sym.ir2ik[ir]
            nd = windows.ndimwin[ik]
            ngk = int(np.count_nonzero(sym.kptsym[:, ir] == ik))
            star_size = sym.nsymmetry / ngk
            u_new, lam = dis_extract_symmetry(ik, nd, z_in[:nd, :nd, ik], u_matrix_opt[:, :, ik], sym)
            u_matrix_opt[:, :, ik] = u_new
            # disentangle.F90:1767-1769 and :1814-1816: the star weight multiplies
            # num_wann wbtot alone, and lambda is subtracted once (wannierpy weighted
            # both). It is the convergence test's estimate only.
            womegai1 += star_size * num_wann * wbtot - np.sum(np.real(np.diag(lam)))
        womegai1 /= num_kpts
        u_matrix_opt = symmetrize_u_matrix(u_matrix_opt, sym, lwindow)
        womegai = _womegai(u_matrix_opt, M_slim, windows, nnlist, wb, wbtot, num_wann)
        delta = womegai1 / womegai - 1.0
        z_out = symmetrize_zmatrix(zmatrix_stack(u_matrix_opt), sym, lwindow)
        history.append(delta)
        if len(history) > dis_conv_window:
            history.pop(0)
        if iteration >= dis_conv_window and all(abs(h) < dis_conv_tol for h in history):
            converged = True
            break
    return u_matrix_opt, converged
