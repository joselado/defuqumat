"""The exact references of the orbital-field work: free Landau levels and the Dirac ladder.

Two closed forms that share no machinery with any route, and the velocity the second one
needs, read from zero-field bands.

- Free electrons, ``pi^2`` on a cell periodic along ``a3`` (Rydberg units, ``m = 1/2``):
  ``E = (2 n + 1) B_ph + (k3 + G3)^2``, ``n >= 0``, ``p`` states per level per magnetic
  ``k`` per ``G3`` branch, ``B_ph = 2 pi n_phi / area`` (:mod:`defuqumat.magnetic.conventions`).
- A Dirac cone in a field: ``E_n = sgn(n) v_F sqrt(2 e hbar B |n|)``, with the two zero
  modes per valley pair of graphene; for the nearest-neighbour model ``v_F = 3 |t| a_cc /
  (2 hbar)``, ``a_cc = a / sqrt(3)`` (8.74e5 m/s at ``t = 2.7`` eV, ``a = 2.46`` A, and
  ``E_1 = 0.10021`` eV at 9.991 T, the review).

The papers: Lu et al., PRL 135, 196401 (2025), arXiv:2505.06585 (the Landau problem as
their first check); Wang et al., arXiv:2609.01053 (graphene's ladder at laboratory fields).
"""

from __future__ import annotations

import numpy as np

from defuqumat.units import BOHR_RADIUS_SI, ELECTRON_SI, H_PLANCK_SI, RYDBERG_SI, TPI

__all__ = ["free_landau_levels", "dirac_levels", "dirac_point", "dirac_velocity_from_bands",
           "landau_degeneracy", "linear_deviation", "HBAR_SI"]

HBAR_SI = H_PLANCK_SI / TPI


def free_landau_levels(cell, nmax: int, kappa3: float = 0.0, g3max: int = 2):
    """``(levels, multiplicity, labels)`` of ``pi^2`` on ``cell`` (a :class:`MagneticCell`), Ry.

    ``levels`` sorted, ``multiplicity`` ``p`` each (per magnetic ``k``), ``labels``
    ``(n, m)`` with ``G3 = 2 pi m / |a3|``. ``a3`` must be normal to ``a1`` and ``a2``,
    which is what makes the in-plane and the out-of-plane motion separate.
    """
    lattice = np.asarray(cell.lattice, dtype=float)
    a3 = lattice[2]
    if abs(a3 @ lattice[0]) > 1e-8 * np.linalg.norm(a3) * np.linalg.norm(lattice[0]) or \
            abs(a3 @ lattice[1]) > 1e-8 * np.linalg.norm(a3) * np.linalg.norm(lattice[1]):
        raise NotImplementedError("the free Landau ladder is written for a3 normal to the plane "
                                  "of a1 and a2")
    b_ph = float(cell.b_ph)
    c = float(np.linalg.norm(a3))
    out = []
    for n in range(nmax + 1):
        for m in range(-g3max, g3max + 1):
            out.append(((2 * n + 1) * b_ph + (TPI * (kappa3 + m) / c) ** 2, n, m))
    out.sort()
    levels = np.array([x[0] for x in out])
    labels = np.array([(x[1], x[2]) for x in out])
    return levels, np.full(len(levels), cell.p), labels


def dirac_levels(B_tesla: float, v_F: float, nmax: int) -> np.ndarray:
    """``E_n = sgn(n) v_F sqrt(2 e hbar B |n|)`` in eV for ``n = -nmax .. nmax``; ``v_F`` in m/s."""
    n = np.arange(-nmax, nmax + 1)
    return np.sign(n) * v_F * np.sqrt(2.0 * ELECTRON_SI * HBAR_SI * B_tesla * np.abs(n)) / ELECTRON_SI


def _reciprocal(lattice):
    return np.linalg.inv(np.asarray(lattice, dtype=float)).T  # rows b_i, a_i . b_j = delta


def _min_image_length(kappa, b):
    best = np.inf
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            best = min(best, np.linalg.norm(TPI * (np.asarray(kappa[:2]) + (i, j)) @ b[:2]))
    return best


def dirac_point(lattice, bands, energy: float, *, delta: float = 1.0e-3,
                degenerate: float = 1.0e-3) -> dict:
    """``E_D`` and ``v_F`` of a hexagonal cell's Dirac cone from zero-field ``bands``.

    ``bands(kappa)`` returns ``(nk, nb)`` energies in Ry at crystal ``kappa`` ``(nk, 3)``.
    ``K`` is the corner of the zone (of ``(1/3, 1/3)`` and ``(1/3, 2/3)`` the one farther
    from Gamma), ``E_D`` the mean of the degenerate pair at ``K`` (split by less than
    ``degenerate`` Ry) nearest ``energy``, and ``v_F`` the slope of the upper band of the
    pair by a central difference: the points ``K +- delta |b1| u`` along the unit vector
    ``u`` from ``K`` to the nearest ``M``, the two one-sided slopes averaged. Also the
    lower band's slope (``v_F_hole``) and the cartesian step.
    """
    lattice = np.asarray(lattice, dtype=float)
    b = _reciprocal(lattice)
    candidates = [np.array([1 / 3, 1 / 3, 0.0]), np.array([1 / 3, 2 / 3, 0.0])]
    K = max(candidates, key=lambda k: _min_image_length(k, b))
    ms = [np.array(m, float) for m in ((0.5, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0), (0.5, -0.5, 0),
                                       (-0.5, 0, 0), (0, -0.5, 0), (-0.5, -0.5, 0), (-0.5, 0.5, 0),
                                       (1, 0.5, 0), (0.5, 1, 0))]
    M = min(ms, key=lambda m: np.linalg.norm(TPI * (m - K) @ b))
    u_cart = TPI * (M - K) @ b
    u_cart = u_cart / np.linalg.norm(u_cart)
    step = delta * np.linalg.norm(TPI * b[0])                     # 1/bohr
    dk_crys = (step * u_cart) @ lattice.T / TPI                     # crystal step
    e = np.asarray(bands(np.stack([K, K + dk_crys, K - dk_crys])), dtype=float)
    eK = e[0]
    pairs = [(i, abs(eK[i + 1] - eK[i])) for i in range(len(eK) - 1)
             if abs(eK[i + 1] - eK[i]) < degenerate]
    if not pairs:
        raise ValueError("no degenerate pair at K: not a Dirac cone at the zone corner")
    i = min(pairs, key=lambda x: abs(0.5 * (eK[x[0]] + eK[x[0] + 1]) - energy))[0]
    E_D = 0.5 * (eK[i] + eK[i + 1])
    slope_up = 0.5 * ((e[1, i + 1] - E_D) + (e[2, i + 1] - E_D)) / step    # Ry bohr
    slope_dn = 0.5 * ((E_D - e[1, i]) + (E_D - e[2, i])) / step
    to_si = RYDBERG_SI * BOHR_RADIUS_SI / HBAR_SI
    return {"E_D": float(E_D), "v_F": float(slope_up * to_si), "v_F_hole": float(slope_dn * to_si),
            "K": K, "M": M, "band": int(i), "splitting": float(abs(eK[i + 1] - eK[i])),
            "step": float(step)}


def dirac_velocity_from_bands(w, energy: float | None = None, **options) -> dict:
    """:func:`dirac_point` on a :class:`~defuqumat.workflows.wannier.WannierFunctions`' bands.

    ``energy`` defaults to ``w.edge`` (the Fermi level of a metal).
    """
    if energy is None:
        energy = w.edge
    if energy is None:
        raise ValueError("give the energy the Dirac point is looked for at")
    return dirac_point(w.lattice, w.bands, float(energy), **options)


def landau_degeneracy(values, p: int, tol: float):
    """``(levels, multiplicities, ok)``: clusters of ``values`` and whether each is a multiple of ``p``.

    Only the clusters strictly inside the window carry the count; the first and the
    last may be cut by the window and are not judged.
    """
    from defuqumat.magnetic.peierls import cluster_levels

    levels, counts = cluster_levels(values, tol)
    inner = counts[1:-1] if len(counts) > 2 else counts[:0]
    return levels, counts, bool(np.all(inner % p == 0))


def linear_deviation(ratios, n) -> tuple[float, float, float]:
    """Fit ``ratio = 1 - c n``: ``(c, c_free, intercept)``, the second with a free intercept."""
    ratios = np.asarray(ratios, dtype=float)
    n = np.asarray(n, dtype=float)
    c = -float(np.sum(n * (ratios - 1.0)) / np.sum(n * n))
    slope, intercept = np.polyfit(n, ratios, 1)
    return c, -float(slope), float(intercept)
