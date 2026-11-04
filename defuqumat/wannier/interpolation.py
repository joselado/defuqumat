"""Wannier interpolation: ``H_mn(R)`` on the Wigner-Seitz supercell, and bands at any k.

The Hamiltonian in the Wannier gauge at the mesh points is
``H(k) = V(k)^H diag(e(k)) V(k)``, ``V = U_opt U``, and its Fourier transform

    H_mn(R) = (1/N_k) sum_k e^{-i k.R} H_mn(k)

is the hopping between Wannier function ``m`` in the home cell and ``n`` in cell
``R``, the tight-binding model the minimisation hands over. At any other ``k``,

    H_mn(k) = sum_R e^{i k.R} H_mn(R) / ndegen(R),

over the lattice points of the Wigner-Seitz cell of the ``N_k`` supercell,
each weighted by how many supercell images share it (``hamiltonian_get_hr``,
``hamiltonian_wigner_seitz`` and ``plot_interpolate_bands`` of Wannier90
3.1.0, ``src/hamiltonian.F90`` and ``src/plot.F90``).

``use_ws_distance`` (Wannier90's default since 3.0, ``src/ws_distance.F90``)
moves each pair ``(m, n)`` at each ``R`` to the image ``R + T`` of the supercell
that minimises ``|R + T + tau_n - tau_m|``, with ``tau`` the Wannier centres,
and splits it evenly over a tie. It is what makes the interpolation of a
model whose centres are not on the lattice points as smooth as the functions
are localized.

Host-side NumPy. ``nR`` is about ``N_k`` and the table is
``nR num_wann^2`` complex numbers.
"""

from __future__ import annotations

import numpy as np

from defuqumat.wannier.ws_vectors import wigner_seitz_vectors

__all__ = ["wannier_hamiltonian", "ws_distance_shifts", "interpolate"]

_WS_DISTANCE_TOL = 1.0e-5
_WS_SEARCH = (2, 2, 2)


def wannier_hamiltonian(v: np.ndarray, energies: np.ndarray, kpt_frac: np.ndarray,
                        mp_grid, real_lattice: np.ndarray):
    """``(irvec, ndegen, H(R))`` from ``v`` ``(nk, num_bands, num_wann)``.

    ``energies`` is ``(nk, num_bands)`` and ``kpt_frac`` ``(nk, 3)``; the unit of
    ``H`` is that of ``energies``. ``real_lattice`` rows set the metric of the
    Wigner-Seitz search and are in any length unit.
    """
    irvec, ndegen, _ = wigner_seitz_vectors(np.asarray(mp_grid), np.asarray(real_lattice, dtype=float),
                                            ws_search_size=_WS_SEARCH,
                                            ws_distance_tol=_WS_DISTANCE_TOL)
    hk = np.einsum("kbm,kb,kbn->kmn", v.conj(), energies, v)
    phase = np.exp(-2j * np.pi * (np.asarray(kpt_frac) @ irvec.T))  # (nk, nR)
    hr = np.einsum("kr,kmn->rmn", phase, hk) / len(kpt_frac)
    return irvec, ndegen, hr


def ws_distance_shifts(irvec: np.ndarray, centres: np.ndarray, mp_grid,
                       real_lattice: np.ndarray):
    """``ws_translate_dist``: for each ``(R, m, n)`` the images ``R + T`` that
    minimise ``|R + tau_n - tau_m|``, as a list per ``(R, m, n)``.

    Returns ``(shifted, count)``: ``shifted`` ``(nR, nw, nw, 8, 3)`` integer
    lattice vectors and ``count`` ``(nR, nw, nw)`` the number of them.
    ``centres`` (``(nw, 3)``) and ``real_lattice`` share one length unit.
    """
    real_lattice = np.asarray(real_lattice, dtype=float)
    recip = np.linalg.inv(real_lattice)  # frac = cart @ recip
    mp = np.asarray(mp_grid)
    search = [np.arange(-s - 1, s + 2) for s in _WS_SEARCH]
    trans = np.stack(np.meshgrid(*search, indexing="ij"), axis=-1).reshape(-1, 3) * mp  # (nT, 3)
    nr, nw = len(irvec), len(centres)
    shifted = np.zeros((nr, nw, nw, 8, 3), dtype=np.int64)
    count = np.zeros((nr, nw, nw), dtype=np.int64)
    tol = _WS_DISTANCE_TOL
    tcart = trans @ real_lattice                                  # (nT, 3)
    pair = centres[None, :, :] - centres[:, None, :]              # (m, n): tau_n - tau_m
    for ir in range(nr):
        r_in = irvec[ir] @ real_lattice + pair                    # (nw, nw, 3)
        # First sweep: the strict minimum, the Fortran's first in loop order
        # (argmin returns the first of a tie; the untranslated point wins a tie
        # with any translation, as the strict '<' does).
        cand = r_in[:, :, None, :] + tcart[None, None]            # (nw, nw, nT, 3)
        d2 = np.einsum("mnti,mnti->mnt", cand, cand)
        first = np.argmin(d2, axis=2)
        best2_t = np.take_along_axis(d2, first[..., None], axis=2)[..., 0]
        here2 = np.einsum("mni,mni->mn", r_in, r_in)
        moved = best2_t < here2
        shift = np.where(moved[..., None], trans[first], 0)       # (nw, nw, 3)
        best = r_in + shift @ real_lattice
        best2 = np.einsum("mni,mni->mn", best, best)
        # Second sweep around the minimum: every image as close, within tol.
        cand = best[:, :, None, :] + tcart[None, None]
        dist = np.sqrt(np.einsum("mnti,mnti->mnt", cand, cand))
        tie = np.abs(dist - np.sqrt(best2)[..., None]) < tol
        origin = best2 < tol ** 2
        for m in range(nw):
            for n in range(nw):
                if origin[m, n]:
                    count[ir, m, n] = 1
                    shifted[ir, m, n, 0] = irvec[ir] + shift[m, n]
                    continue
                hits = trans[tie[m, n]]
                if len(hits) > 8:
                    raise ValueError("a Wigner-Seitz degeneracy larger than 8")
                count[ir, m, n] = len(hits)
                shifted[ir, m, n, :len(hits)] = irvec[ir] + shift[m, n] + hits
    return shifted, count


def interpolate(kpt_frac: np.ndarray, irvec, ndegen, hr, shifted=None, count=None):
    """``(nk, nw)`` eigenvalues and ``(nk, nw, nw)`` eigenvectors at ``kpt_frac``."""
    kpt_frac = np.atleast_2d(np.asarray(kpt_frac, dtype=float))
    if shifted is None:
        phase = np.exp(2j * np.pi * (kpt_frac @ irvec.T)) / ndegen[None, :]
        hk = np.einsum("kr,rmn->kmn", phase, hr)
    else:
        # sum over the images of each (R, m, n), each 1/(ndegen count)
        # (one k at a time: the phase table is nR nw^2 8 per k)
        valid = np.arange(8)[None, None, None, :] < count[..., None]
        weight = valid / (ndegen[:, None, None, None] * np.maximum(count, 1)[..., None])
        weighted = weight * hr[..., None]
        hk = np.stack([np.einsum("rmnd,rmnd->mn", np.exp(2j * np.pi * (shifted @ k)), weighted)
                       for k in kpt_frac])
    hk = 0.5 * (hk + np.conj(np.swapaxes(hk, 1, 2)))
    return np.linalg.eigh(hk)
