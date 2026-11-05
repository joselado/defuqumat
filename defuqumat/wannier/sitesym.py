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

**A frozen set** (P164b), which Wannier90 refuses (``disentangle.F90:104``,
"frozen window not implemented yet in symmetry-adapted mode"): its frozen
columns of ``u_matrix_opt`` are one-hot, and :func:`symmetrize_u_matrix`
mixes them with the free ones. Here the column basis is changed at each
representative instead. A unitary ``V(k)`` on the trial-orbital index
block-diagonalises ``D(g, k)`` over the little group into
``diag(d_F(g, k), D_free(g, k))``, with ``d_F`` the band representation on
the frozen rows, so that ``U V = [E_F | U_free]`` keeps the one-hot frozen
columns exactly (``d^+ E_F d_F = E_F``) and only ``U_free`` is optimised, by
the same 2x2 steps under ``(d_free, D_free)``
(:func:`frozen_blocks`, :func:`frozen_start`,
:func:`dis_extract_symmetric`). ``V`` exists only if every irrep of the
little group in the frozen set appears in ``D`` at least as often, and
``D_free`` fits in the free bands; both are rank tests of a generic
intertwiner (Schur's lemma) and refused by name. Every consumer of ``U``
downstream sees ``U U^+`` or is covariant under ``U -> U W``, so the frozen
columns need not be one-hot in the trial-orbital basis the star copy uses.

Indices are 0-based; the ``.dmn`` file's are 1-based and are shifted once, in
:func:`from_dmn`.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import warnings

import numpy as np

__all__ = ["SymmetryData", "from_dmn", "slim_d_matrix_band", "symmetrize_u_matrix",
           "symmetrize_gradient", "symmetrize_rotation", "symmetrize_zmatrix",
           "dis_extract_symmetry", "dis_extract_symmetric", "wannier_gauge",
           "FrozenBlock", "check_frozen_star", "frozen_blocks", "frozen_start",
           "intertwiner_rank"]


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
                    "a band, which the site-symmetry mode cannot treat; move dis_win (with "
                    "projections='auto' the outer set is chosen star by star, so this is a "
                    "defect of that selection)")
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


# --- a frozen set under site symmetry (P164b) ----------------------------------------------

#: A generic intertwiner whose smallest singular value is below this fraction of
#: its largest has lost rank: the representations do not fit (Schur's lemma).
INTERTWINER_RANK_TOLERANCE = 1.0e-6
#: The projections of the frozen states onto the trial orbitals fix ``V_F`` when
#: their averaged intertwiner is at least this well conditioned; otherwise a random one does.
PROJECTION_CONDITION = 1.0e-3


@dataclass
class FrozenBlock:
    """The column basis of one representative ``ik``: ``V = [V_F | V_free]``.

    ``sym_free`` is a one-point :class:`SymmetryData` (``nkptirr = 1``) on the free
    block: ``d_matrix_band[:, :, s, 0]`` is ``d_win[ix_(free rows at g k, free rows
    at k)]`` for every operation, ``d_matrix_wann[:, :, s, 0]`` is ``D_free`` (only
    the little group is meaningful there). ``source`` says what fixed ``V_F``.
    """

    ir: int
    ik: int
    nfroz: int
    V: np.ndarray
    sym_free: SymmetryData
    source: str


def _polar(x: np.ndarray) -> np.ndarray:
    Z, _, Vh = np.linalg.svd(x, full_matrices=False)
    return Z @ Vh


def intertwiner_rank(left: np.ndarray, right: np.ndarray, rng, little_group) -> tuple[int, float]:
    """Rank and relative smallest singular value of ``(1/|G|) sum_g left(g)^+ X right(g)``
    for a random ``X``.

    By Schur's lemma its generic rank is ``sum_i dim_i min(m_L(i), m_R(i))``, so it is
    ``min(n, m)`` exactly when every irrep of the smaller representation appears in the
    other at least as often. ``left`` and ``right`` are ``(n, n, nsym)`` and
    ``(m, m, nsym)``. The character inner product is not such a test (it counts
    products of multiplicities).
    """
    n, m = left.shape[0], right.shape[0]
    if n == 0 or m == 0:
        return min(n, m), 1.0
    X = rng.normal(size=(n, m)) + 1j * rng.normal(size=(n, m))
    Y = sum(left[:, :, s].conj().T @ X @ right[:, :, s] for s in little_group) / len(little_group)
    sv = np.linalg.svd(Y, compute_uv=False)
    rel = sv / sv[0] if sv[0] > 0 else sv
    return int(np.count_nonzero(rel > INTERTWINER_RANK_TOLERANCE)), float(rel[-1])


def check_frozen_star(windows, sym: SymmetryData) -> None:
    """The frozen set must hold the same window rows at ``k`` and ``g k`` and close
    its multiplets (``d`` unitary on the frozen rows), at every representative.
    ``sym`` is window-sliced (:func:`slim_d_matrix_band`)."""
    for ir in range(sym.nkptirr):
        ik = int(sym.ir2ik[ir])
        nd, nf = int(windows.ndimwin[ik]), int(windows.ndimfroz[ik])
        froz = windows.lfrozen[:nd, ik]
        for s in range(sym.nsymmetry):
            jk = sym.kptsym[s, ir]
            if not np.array_equal(windows.lfrozen[:windows.ndimwin[jk], jk], froz):
                raise ValueError(
                    "the frozen set holds different bands at k and g k for a star of the mesh "
                    f"(irreducible point {ir}, operation {s}): a frozen edge sits on a band; "
                    "move dis_froz (with projections='auto' the frozen set is chosen star by "
                    "star, so this is a defect of that selection)")
        f = windows.indxfroz[:nf, ik]
        check_band_unitarity(sym.d_matrix_band[np.ix_(f, f)][:, :, :, ir], ik,
                             "the frozen set", sym.refuse_cut)


def frozen_blocks(sym: SymmetryData, windows, A_matrix: np.ndarray, num_wann: int,
                  seed: int = 0) -> list[FrozenBlock]:
    """``V(k)``, ``D_free`` and ``d_free`` at every representative, with the two refusals.

    ``sym`` is window-sliced; ``A_matrix`` is on the absolute band index.
    ``V_F = polar((1/|G_k|) sum_g D(g) Y d_F(g)^+)`` is an intertwiner,
    ``D V_F = V_F d_F``, so ``V^+ D V`` is block diagonal with ``d_F`` in the
    frozen block. ``Y`` is the projection of the frozen states onto the trial
    orbitals, ``A^+ E_F``, which is already an intertwiner when ``A`` is covariant;
    a random ``Y`` replaces it when the average is ill-conditioned (``source``).
    Only the subspace (``U U^+``) is fixed by the extraction, so the choice of
    ``V_F`` changes the start and the gauge, not the set of symmetric subspaces.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    for ir in range(sym.nkptirr):
        ik = int(sym.ir2ik[ir])
        nd, nf = int(windows.ndimwin[ik]), int(windows.ndimfroz[ik])
        ndk, nfree = nd - nf, num_wann - nf
        little_group = np.nonzero(sym.kptsym[:, ir] == ik)[0]
        f = windows.indxfroz[:nf, ik]
        d_F = sym.d_matrix_band[np.ix_(f, f)][:, :, :, ir]
        D = sym.d_matrix_wann[:, :, :, ir]
        rank, rel = intertwiner_rank(d_F, D, rng, little_group)
        if rank < nf:
            raise ValueError(
                f"the frozen states at k-point {ik} carry an irrep the trial orbitals do not "
                f"(or more often than the trial orbitals do): a generic intertwiner has rank "
                f"{rank} of {nf} (smallest relative singular value {rel:.1e}), so no U keeps "
                "them frozen and the symmetry together. Lower the frozen edge or change the "
                "projections")
        if nf == 0:
            V = np.eye(num_wann, dtype=complex)
            source = "no frozen state"
        else:
            rows = windows.indxwin[:nd, ik]
            Y = A_matrix[rows[f], :num_wann, ik].conj().T             # A^+ E_F, (num_wann, nf)
            Y = sum(D[:, :, s] @ Y @ d_F[:, :, s].conj().T for s in little_group) / len(little_group)
            sv = np.linalg.svd(Y, compute_uv=False)
            source = "projections"
            if sv[-1] <= PROJECTION_CONDITION * sv[0]:
                source = "random"
                X = rng.normal(size=(num_wann, nf)) + 1j * rng.normal(size=(num_wann, nf))
                Y = sum(D[:, :, s] @ X @ d_F[:, :, s].conj().T for s in little_group) / len(little_group)
            V_F = _polar(Y)
            Q = np.eye(num_wann) - V_F @ V_F.conj().T
            w, vecs = np.linalg.eigh(0.5 * (Q + Q.conj().T))
            V = np.concatenate([V_F, vecs[:, w > 0.5]], axis=1)
            if V.shape[1] != num_wann:
                raise ValueError(f"the frozen column basis at k-point {ik} is not square")
        D_free = np.einsum("ia,ijs,jb->abs", V[:, nf:].conj(), D, V[:, nf:])
        # d on the free rows: target rows at g k, source rows at k, every operation.
        free = windows.indxnfroz[:ndk, ik]
        d_free = np.zeros((ndk, ndk, sym.nsymmetry), dtype=complex)
        for s in range(sym.nsymmetry):
            target = windows.indxnfroz[:ndk, sym.kptsym[s, ir]]
            d_free[:, :, s] = sym.d_matrix_band[np.ix_(target, free)][:, :, s, ir]
        rank2, rel2 = intertwiner_rank(d_free, D_free, rng, little_group)
        if rank2 < nfree:
            raise ValueError(
                f"the bands left free at k-point {ik} do not carry the irreps the trial orbitals "
                f"need beside the frozen ones (rank {rank2} of {nfree}, smallest relative "
                f"singular value {rel2:.1e}): raise the outer window or num_bands, or change "
                "the projections")
        sym_free = SymmetryData(
            nsymmetry=sym.nsymmetry, nkptirr=1, num_kpts=sym.num_kpts,
            ik2ir=np.zeros(sym.num_kpts, dtype=np.int64), ir2ik=np.array([ik], dtype=np.int64),
            kptsym=sym.kptsym[:, ir:ir + 1].copy(), d_matrix_wann=D_free[:, :, :, None],
            d_matrix_band=d_free[:, :, :, None], symmetrize_eps=sym.symmetrize_eps,
            refuse_cut=sym.refuse_cut)
        blocks.append(FrozenBlock(ir=ir, ik=ik, nfroz=nf, V=V, sym_free=sym_free, source=source))
    return blocks


def _onehot(windows, k: int) -> np.ndarray:
    nd, nf = int(windows.ndimwin[k]), int(windows.ndimfroz[k])
    e = np.zeros((nd, nf), dtype=complex)
    e[windows.indxfroz[:nf, k], np.arange(nf)] = 1.0
    return e


def _assemble(windows, block: FrozenBlock, u_free: np.ndarray) -> np.ndarray:
    """``[E_F | U_free] V^+`` on the window rows; ``u_free`` is on the free rows only."""
    ik = block.ik
    nd = int(windows.ndimwin[ik])
    ndk = nd - block.nfroz
    full = np.zeros((nd, u_free.shape[1]), dtype=complex)
    full[windows.indxnfroz[:ndk, ik]] = u_free[:ndk]
    return np.concatenate([_onehot(windows, ik), full], axis=1) @ block.V.conj().T


def _free_part(windows, block: FrozenBlock, u: np.ndarray) -> np.ndarray:
    """The free block of ``u[:nd, :, ik] V`` on the free rows, ``(ndk, num_wann - nf)``."""
    ik, nf = block.ik, block.nfroz
    nd = int(windows.ndimwin[ik])
    up = u[:nd, :, ik] @ block.V
    return up[windows.indxnfroz[:nd - nf, ik], nf:]


def frozen_start(A_matrix: np.ndarray, windows, blocks: list[FrozenBlock], sym: SymmetryData,
                 lwindow: np.ndarray, num_wann: int) -> np.ndarray:
    """The symmetric start with the frozen states in: ``U_free = polar(Q A V_free)``.

    ``Q`` zeroes the frozen rows. ``Q A V_free`` is an intertwiner between
    ``d_free`` and ``D_free`` when ``A`` is covariant, so its polar part is one
    too; the little-group average of :func:`symmetrize_ukirr` fixes what ``A``
    lacks, and :func:`symmetrize_u_matrix` fills the stars. ``dis_proj_froz`` is
    not used: the top eigenvectors of ``Q P_s Q`` need not carry the irreps of
    ``D``.
    """
    num_bands, num_kpts = lwindow.shape
    u = np.zeros((num_bands, num_wann, num_kpts), dtype=complex)
    for b in blocks:
        ik, nf = b.ik, b.nfroz
        nd = int(windows.ndimwin[ik])
        ndk = nd - nf
        if num_wann > nf:
            rows = windows.indxwin[:nd, ik]
            free = windows.indxnfroz[:ndk, ik]
            X = A_matrix[rows[free], :num_wann, ik] @ b.V[:, nf:]
            uc = symmetrize_ukirr(0, _polar(X), b.sym_free, ndk)
        else:
            uc = np.zeros((ndk, 0), dtype=complex)
        u[:nd, :, ik] = _assemble(windows, b, uc)
    return symmetrize_u_matrix(u, sym, lwindow)


def _fold_free(z: np.ndarray, windows, blocks: list[FrozenBlock], sym: SymmetryData) -> np.ndarray:
    """:func:`symmetrize_zmatrix` on the free rows, ``d_free`` in place of ``d``."""
    out = z.copy()
    found = np.zeros(z.shape[2], dtype=bool)
    for b in blocks:
        ik, ir = b.ik, b.ir
        ndk = int(windows.ndimwin[ik]) - b.nfroz
        d = b.sym_free.d_matrix_band[:, :, :, 0]
        found[ik] = True
        for s in range(1, sym.nsymmetry):
            jk = sym.kptsym[s, ir]
            if found[jk]:
                continue
            found[jk] = True
            out[:ndk, :ndk, ik] += d[:, :, s].conj().T @ out[:ndk, :ndk, jk] @ d[:, :, s]
        snapshot = out[:ndk, :ndk, ik].copy()
        for s in range(1, sym.nsymmetry):
            if sym.kptsym[s, ir] != ik:
                continue
            out[:ndk, :ndk, ik] += d[:, :, s].conj().T @ snapshot @ d[:, :, s]
        out[:ndk, :ndk, ik] /= int(np.count_nonzero(sym.kptsym[:, ir] == ik))
    return out


def dis_extract_symmetric_frozen(u_matrix_opt_init, M_slim, windows, nnlist, wb, wbtot,
                                 num_wann, dis_num_iter, dis_mix_ratio, dis_conv_tol,
                                 dis_conv_window, sym, lwindow, blocks, trace=None):
    """The symmetric iteration with frozen states (P164b).

    At each representative the free block ``U_free`` of ``U V`` is extracted by
    :func:`dis_extract_symmetry` under ``(d_free, D_free)`` from the free ``Z``
    (``disentangle._zmatrix``, folded over the star with ``d_free``), the one-hot
    frozen columns are kept, and ``U = [E_F | U_free] V^+`` is copied to the star
    with ``(d, D)``.

    The convergence estimate is ``dis_extract``'s with the star weights of
    :func:`dis_extract_symmetric`: ``womegai1 = (1/N) sum_ir [n_star (num_wann wbtot
    - r_F(k)) - tr lambda]``, ``r_F`` the frozen rows' overlap with the current
    neighbours. There is no Fortran counterpart for this route; the estimate was
    checked against the exact ``_womegai`` on the H3S fixture (``PLAN.md`` P164b);
    ``trace``, a list, receives ``(womegai1, womegai)`` at every step for that.
    """
    from defuqumat.wannier.disentangle import _womegai, _zmatrix

    num_bands, _, num_kpts = u_matrix_opt_init.shape
    u = u_matrix_opt_init.copy()

    def zstack(u):
        Z = np.zeros((num_bands, num_bands, num_kpts), dtype=complex)
        for k in range(num_kpts):
            ndk = windows.ndimwin[k] - windows.ndimfroz[k]
            if ndk:
                Z[:ndk, :ndk, k] = _zmatrix(k, u, M_slim, windows, nnlist, wb, num_wann)
        return Z

    def frozen_overlap(k, u):
        nf = int(windows.ndimfroz[k])
        if nf == 0:
            return 0.0
        rows = windows.indxfroz[:nf, k]
        total = 0.0
        for nn in range(len(wb)):
            k2 = int(nnlist[k, nn]) - 1
            nd2 = windows.ndimwin[k2]
            total += wb[nn] * float(np.sum(np.abs(M_slim[rows, :nd2, nn, k] @ u[:nd2, :, k2]) ** 2))
        return total

    z_in = _fold_free(zstack(u), windows, blocks, sym)
    z_out = None
    history = []
    converged = False
    for iteration in range(1, dis_num_iter + 1):
        if iteration > 1:
            for b in blocks:
                ndk = int(windows.ndimwin[b.ik]) - b.nfroz
                mixed = (dis_mix_ratio * z_out[:ndk, :ndk, b.ik]
                         + (1.0 - dis_mix_ratio) * z_in[:ndk, :ndk, b.ik])
                z_in[:ndk, :ndk, b.ik] = 0.5 * (mixed + mixed.conj().T)
        # the frozen rows' term from the neighbours before this step, as dis_extract
        womegai1 = sum(sym.nsymmetry / int(np.count_nonzero(sym.kptsym[:, b.ir] == b.ik))
                       * (num_wann * wbtot - frozen_overlap(b.ik, u)) for b in blocks)
        for b in blocks:
            ik, nf = b.ik, b.nfroz
            ndk = int(windows.ndimwin[ik]) - nf
            if num_wann > nf:
                uc, lam = dis_extract_symmetry(ik, ndk, z_in[:ndk, :ndk, ik],
                                               _free_part(windows, b, u), b.sym_free)
                womegai1 -= float(np.sum(np.real(np.diag(lam))))
            else:
                uc = np.zeros((ndk, 0), dtype=complex)
            u[:int(windows.ndimwin[ik]), :, ik] = _assemble(windows, b, uc)
        womegai1 /= num_kpts
        u = symmetrize_u_matrix(u, sym, lwindow)
        womegai = _womegai(u, M_slim, windows, nnlist, wb, wbtot, num_wann)
        delta = womegai1 / womegai - 1.0
        if trace is not None:
            trace.append((womegai1, womegai))
        z_out = _fold_free(zstack(u), windows, blocks, sym)
        history.append(delta)
        if len(history) > dis_conv_window:
            history.pop(0)
        if iteration >= dis_conv_window and all(abs(h) < dis_conv_tol for h in history):
            converged = True
            break
    return u, converged
