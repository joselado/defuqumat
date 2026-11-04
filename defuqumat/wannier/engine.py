"""The minimisation half of Wannier90: disentanglement, then the spread.

This is ``wannier_run`` of ``src/wannier_lib.F90`` with the file input taken
away: the overlaps ``M``, the projections ``A`` and the eigenvalues go in as
arrays, and the unitary matrices, the centres and the spreads come out. The
call graph is the Fortran one,

    kmesh_get -> dis_main (if num_bands > num_wann) or overlap_project -> wann_main

and the routines it calls are the user's own NumPy port of Wannier90 3.1.0
(the ``wannierpy`` project, ``wannier90/_engine/``), copied here and trimmed of
site symmetry. That port was validated against ``wannier90.x`` on GaAs, diamond
and silicon; the numbers for this copy are in ``PLAN.md``, P162.

The engine is **unit-agnostic**: lengths are whatever the reciprocal lattice
was given in (defuqumat passes bohr, so spreads come out in bohr^2) and the
window energies are in the unit of ``eigenvalues`` (defuqumat passes eV, the
unit Wannier90 users write windows in). It runs on the host in NumPy: it is a
small dense optimisation over ``num_kpts`` matrices of size ``num_bands``,
done once after the SCF, like the symmetry search.

Wannier90's defaults are kept, including ``conv_window = -1`` meaning that all
``num_iter`` steps are taken with no convergence test (wannierpy replaced it by
5, which stops earlier than ``wannier90.x`` does).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from defuqumat.wannier.disentangle import dis_main
from defuqumat.wannier.kmesh import KmeshResult, kmesh_get
from defuqumat.wannier.overlap import overlap_project
from defuqumat.wannier.wannierise import SpreadTerms, wann_main

__all__ = ["EngineOptions", "EngineResult", "run_engine", "kmesh_get", "KmeshResult"]


@dataclass(frozen=True)
class EngineOptions:
    """Wannier90's keywords that steer the minimisation, with its defaults.

    Window energies are in the unit of the eigenvalues handed to
    :func:`run_engine`. ``dis_froz_max = None`` means no frozen window.
    """

    dis_win_min: float | None = None
    dis_win_max: float | None = None
    dis_froz_min: float | None = None
    dis_froz_max: float | None = None
    dis_num_iter: int = 200
    dis_mix_ratio: float = 0.5
    dis_conv_tol: float = 1.0e-10
    dis_conv_window: int = 3
    num_iter: int = 100
    num_cg_steps: int = 5
    conv_tol: float = 1.0e-10
    conv_window: int = -1
    trial_step: float = 2.0
    fixed_step: float | None = None
    guiding_centres: bool = False
    num_no_guide_iter: int = 0
    num_guide_cycles: int = 1
    precond: bool = False


@dataclass(frozen=True)
class EngineResult:
    u_matrix: np.ndarray        # (num_wann, num_wann, num_kpts)
    u_matrix_opt: np.ndarray    # (num_bands, num_wann, num_kpts), zero outside the window
    lwindow: np.ndarray         # (num_bands, num_kpts) bool
    centres: np.ndarray         # (num_wann, 3), cartesian
    spreads: np.ndarray         # (num_wann,)
    terms: SpreadTerms          # om_i, om_d, om_od, om_tot
    iterations: int
    disentangled: bool
    dis_converged: bool
    converged: bool


def run_engine(mmn: np.ndarray, amn: np.ndarray, eigenvalues: np.ndarray, kmesh: KmeshResult,
               num_wann: int, options: EngineOptions = EngineOptions(), *,
               kpt_frac: np.ndarray | None = None, real_lattice: np.ndarray | None = None,
               mp_grid=None, site_cart: np.ndarray | None = None) -> EngineResult:
    """Disentangle (when ``num_bands > num_wann``) and minimise the spread.

    ``mmn`` is ``(num_bands, num_bands, nntot, num_kpts)``, ``amn``
    ``(num_bands, num_wann, num_kpts)`` and ``eigenvalues``
    ``(num_bands, num_kpts)``, all in Wannier90's layout. ``site_cart``
    (``(num_wann, 3)``) is needed for ``guiding_centres``; ``kpt_frac``
    (``(3, num_kpts)``), ``real_lattice`` (rows) and ``mp_grid`` for
    ``precond``.
    """
    o = options
    num_bands, num_kpts = eigenvalues.shape
    disentangled = num_bands > num_wann
    dis_converged = True
    if disentangled:
        win_min = float(eigenvalues.min()) if o.dis_win_min is None else o.dis_win_min
        win_max = float(eigenvalues.max()) if o.dis_win_max is None else o.dis_win_max
        frozen = o.dis_froz_max is not None
        froz_min = win_min if o.dis_froz_min is None else o.dis_froz_min
        u_opt, u, lwindow, m_wann, dis_converged = dis_main(
            amn, mmn, eigenvalues, kmesh.nnlist, kmesh.wb, num_wann,
            win_min, win_max, frozen, froz_min, 0.0 if o.dis_froz_max is None else o.dis_froz_max,
            o.dis_num_iter, o.dis_mix_ratio, o.dis_conv_tol, o.dis_conv_window)
        if not dis_converged:
            warnings.warn(f"the disentanglement did not converge in dis_num_iter = {o.dis_num_iter} "
                          "steps", stacklevel=2)
    else:
        u, m_wann = overlap_project(amn, mmn, kmesh.nnlist)
        u_opt = np.tile(np.eye(num_wann, dtype=complex)[:, :, None], (1, 1, num_kpts))
        lwindow = np.ones((num_bands, num_kpts), dtype=bool)

    if o.guiding_centres and site_cart is None:
        raise ValueError("guiding_centres needs the projection sites")
    u_final, centres, spreads, terms, iterations, converged = wann_main(
        u, m_wann, kmesh.nnlist, kmesh.bk, kmesh.wb,
        num_iter=o.num_iter, num_cg_steps=o.num_cg_steps, conv_tol=o.conv_tol,
        conv_window=o.conv_window, trial_step=o.trial_step, guiding_centres=o.guiding_centres,
        bka=kmesh.bka, neigh=kmesh.neigh,
        proj_site_cart=None if site_cart is None else np.asarray(site_cart, dtype=float).T,
        num_no_guide_iter=o.num_no_guide_iter, num_guide_cycles=o.num_guide_cycles,
        fixed_step=o.fixed_step, precond=o.precond, kpt_latt=kpt_frac,
        real_lattice=None if real_lattice is None else np.asarray(real_lattice, dtype=float),
        mp_grid=None if mp_grid is None else np.asarray(mp_grid, dtype=np.int64))
    if not converged:
        warnings.warn(f"the spread did not converge in num_iter = {o.num_iter} steps "
                      f"(conv_window = {o.conv_window})", stacklevel=2)
    return EngineResult(u_matrix=u_final, u_matrix_opt=u_opt, lwindow=lwindow,
                        centres=np.asarray(centres).T, spreads=np.asarray(spreads), terms=terms,
                        iterations=int(iterations), disentangled=disentangled,
                        dis_converged=bool(dis_converged), converged=bool(converged))
