"""Landau levels of a Wannier model in a uniform orbital field, by Peierls substitution.

The entry point of the weak-field route (:mod:`defuqumat.magnetic.peierls`): P162's
Wannier set (or any object with its fields), a field in tesla or a flux ``p/q`` per
chemical cell, the magnetic torus (default) or a ribbon, and the levels nearest an energy
at a list of magnetic ``k``, with the weight of each level on each site taken as the
trace over its degenerate cluster (rule D4), the Hermiticity residual of every matrix, a
plaquette phase, and with ``dirac=True`` the comparison with the Dirac ladder at the
velocity read from the zero-field interpolated bands at ``K``.

The conventions are :mod:`defuqumat.magnetic.conventions` (Lu et al., PRL 135, 196401
(2025), arXiv:2505.06585; Wang et al., arXiv:2609.01053; Wang and Vafek, PRB 108, 245109
(2023), arXiv:2303.16347). Refused: a Zeeman term (``zeeman=True``, for now), one spin
channel of an ``nspin = 2`` set.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from defuqumat.magnetic.conventions import MagneticCell, minimum_image
from defuqumat.magnetic.landau import HBAR_SI, dirac_levels, dirac_point, linear_deviation
from defuqumat.magnetic.peierls import (
    PeierlsTorus, block_weight, bonds_from_wannier, cluster_levels, hermiticity, levels_near,
    plaquette_check,
)
from defuqumat.units import ELECTRON_SI, RY_TO_EV, rational_flux, tesla_from_flux

__all__ = ["LandauLevels", "run_landau_levels", "wannier_sites"]


@dataclass(frozen=True)
class LandauLevels:
    """The levels of a Peierls model in a field, Ry, and what checks them.

    ``levels`` ``(nk, nlevels)`` sorted, nearest ``energy``; ``weights`` ``(nk, nlevels,
    nsite)`` the weight of each level on each site, the trace over the level's degenerate
    cluster shared equally among its members (so a cluster's sum is the invariant);
    ``clusters`` per k the ``(level, multiplicity)`` list; ``hermiticity`` per k
    ``max |H - H^dagger|`` in Ry and ``relative`` that over ``max |H|``; ``plaquette`` the
    measured product around one cell of the first orbital with hops along ``a1`` and
    ``a2`` (``None`` when there are none). With ``dirac``: ``dirac`` holds ``E_D``,
    ``v_F`` (m/s), ``ratios`` ``{n: (E_n - E_D) / E_n^Dirac}`` for ``0 < |n| <= 10``, the
    zero modes' count and site traces, and the linear fit of each side.
    """

    flux: tuple
    field_tesla: float
    kappa: np.ndarray
    energy: float
    levels: np.ndarray
    weights: np.ndarray
    sites: np.ndarray
    clusters: tuple
    hermiticity: np.ndarray
    relative: np.ndarray
    plaquette: complex | None
    geometry: str
    dim: int
    nbonds: int
    dropped: float
    timings: dict
    vectors: tuple | None = field(default=None, repr=False)
    dirac: dict | None = None

    @property
    def levels_ev(self) -> np.ndarray:
        return self.levels * RY_TO_EV


def wannier_sites(centres, lattice, tol: float = 0.8):
    """``(sites, site_of)``: the centres grouped into sites within ``tol`` bohr (minimum image)."""
    centres = np.asarray(centres, dtype=float)
    sites, site_of = [], []
    for c in centres:
        for i, s in enumerate(sites):
            if np.linalg.norm(minimum_image(c - s, lattice)) < tol:
                site_of.append(i)
                break
        else:
            sites.append(c)
            site_of.append(len(sites) - 1)
    return np.array(sites), np.array(site_of)


def _first_plaquette(cell, bonds):
    """The loop ``0 -> a1 -> a1 + a2 -> a2`` of the orbital with the largest ``|t(a1)|``, if any."""
    known = {(int(b), int(k), tuple(int(x) for x in c)): abs(v)
             for b, k, c, v in zip(bonds.bra, bonds.ket, bonds.cells, bonds.values)}
    best = None
    for o in range(bonds.nw):
        needed = [(o, o, (-1, 0, 0)), (o, o, (0, -1, 0)), (o, o, (1, 0, 0)), (o, o, (0, 1, 0))]
        if all(key in known for key in needed):
            if best is None or known[(o, o, (-1, 0, 0))] > known[(best, best, (-1, 0, 0))]:
                best = o
    if best is None:
        return None
    loop = [(best, (0, 0, 0)), (best, (1, 0, 0)), (best, (1, 1, 0)), (best, (0, 1, 0))]
    return plaquette_check(cell, bonds, loop)


def run_landau_levels(
    wannier,
    *,
    B: float | None = None,
    flux=None,
    qmax: int = 20000,
    geometry: str = "torus",
    width: int | None = None,
    kappa=((0.0, 0.0),),
    nlevels: int = 40,
    energy: float | None = None,
    dirac: bool = False,
    zeeman: bool = False,
    cutoff: float = 0.0,
    degeneracy_tol: float = 1.0e-7,
    vectors: bool = False,
    bonds=None,
) -> LandauLevels:
    """The ``nlevels`` levels nearest ``energy`` of ``wannier`` at flux ``p/q`` (or ``B`` tesla).

    Args:
        wannier: a :class:`~defuqumat.workflows.wannier.WannierFunctions` (``centres``,
            ``rvectors``, ``degeneracy``, ``hoppings``, ``ws_shifts``, ``lattice``,
            ``bands``, ``edge``).
        B, flux: the field in tesla, rounded by :func:`~defuqumat.units.rational_flux`
            with ``qmax``, or ``(p, q)``; exactly one.
        geometry: ``"torus"`` (the magnetic cell with magnetic translations) or
            ``"ribbon"`` (open along ``a1``, ``width`` cells; ten magnetic lengths when
            ``None``).
        kappa: magnetic crystal momenta ``(kappa1, kappa2)``; on a ribbon only
            ``kappa2`` counts.
        energy: the centre of the window in Ry; the Dirac energy of the zero-field
            bands when ``dirac`` and otherwise ``wannier.edge``.
        cutoff: hoppings with ``|H_mn(R)| / (ndegen count) <= cutoff`` Ry are dropped
            (the largest dropped is reported).
        degeneracy_tol: Ry, the clustering of the levels into degenerate blocks.
        bonds: a :class:`~defuqumat.magnetic.peierls.Bonds` to use instead of
            ``wannier``'s (a model; ``wannier`` then only supplies ``bands`` and ``edge``,
            and may be ``None`` when ``dirac`` is off and ``energy`` given).
    """
    if zeeman:
        raise NotImplementedError("the Zeeman term is not implemented with the orbital field: "
                                  "zeeman=True is refused (levels hold one electron per spin)")
    if (B is None) == (flux is None):
        raise ValueError("give exactly one of B (tesla) and flux = (p, q)")
    if geometry not in ("torus", "ribbon"):
        raise ValueError(f"geometry is 'torus' or 'ribbon', not {geometry!r}")
    timings = {}
    t0 = time.perf_counter()
    if bonds is None:
        bonds = bonds_from_wannier(wannier, cutoff=cutoff)
    lattice = bonds.lattice
    area = abs(np.linalg.det(lattice)) / np.linalg.norm(lattice[2])
    if flux is not None:
        p, q = int(flux[0]), int(flux[1])
        field_tesla = tesla_from_flux(p, q, area)
    else:
        p, q, field_tesla = rational_flux(B, area, qmax)
    cell = MagneticCell.build(lattice, p, q)
    torus = PeierlsTorus(cell, bonds)
    if geometry == "ribbon":
        if width is None:
            l_b = math.sqrt(1.0 / float(cell.b_ph))                    # bohr, l_B^2 = 1/B_ph
            width = int(math.ceil(10.0 * l_b / np.linalg.norm(lattice[0])))
        torus = torus.ribbon(width)
    timings["bonds_s"] = time.perf_counter() - t0

    point = None
    if dirac:
        point = dirac_point(lattice, wannier.bands, float(wannier.edge if energy is None else energy))
    if energy is None:
        energy = point["E_D"] if point is not None else float(wannier.edge)
    sites, site_of = wannier_sites(bonds.centres, lattice)
    masks = np.stack([np.repeat((site_of == s)[None, :], torus.ncell, 0).ravel()
                      for s in range(len(sites))], axis=-1)

    kappa = np.atleast_2d(np.asarray(kappa, dtype=float))
    all_levels, all_weights, clusters, herm, rel, vecs = [], [], [], [], [], []
    t_build = t_solve = 0.0
    for kap in kappa:
        t1 = time.perf_counter()
        h = torus.matrix(kap)
        t_build += time.perf_counter() - t1
        residual = hermiticity(h)
        herm.append(residual)
        rel.append(residual / float(np.max(np.abs(h.data))))
        t1 = time.perf_counter()
        w, v = levels_near(h, energy, nlevels)
        t_solve += time.perf_counter() - t1
        lev, mult = cluster_levels(w, degeneracy_tol)
        clusters.append(tuple(zip(lev.tolist(), mult.tolist())))
        weights = np.zeros((len(w), len(sites)))
        start = 0
        for m in mult:
            block = v[:, start:start + m]
            for s in range(len(sites)):
                weights[start:start + m, s] = block_weight(block, masks[:, s]) / m
            start += m
        all_levels.append(w)
        all_weights.append(weights)
        if vectors:
            vecs.append(v)
    timings["build_s"] = t_build
    timings["solve_s"] = t_solve
    levels = np.array(all_levels)

    result_dirac = None
    if dirac:
        result_dirac = _dirac_comparison(levels, np.array(all_weights), clusters, point,
                                         field_tesla, p, degeneracy_tol)
    plaquette = _first_plaquette(cell, bonds) if bonds.nw else None
    return LandauLevels(
        flux=(p, q), field_tesla=float(field_tesla), kappa=kappa, energy=float(energy),
        levels=levels, weights=np.array(all_weights), sites=sites, clusters=tuple(clusters),
        hermiticity=np.array(herm), relative=np.array(rel), plaquette=plaquette,
        geometry=geometry, dim=torus.dim, nbonds=bonds.nb, dropped=bonds.dropped,
        timings=timings, vectors=tuple(vecs) if vectors else None, dirac=result_dirac)


#: Levels closer than this fraction of the Dirac ``E_1`` are one Landau level of the
#: comparison. The spacing at ``|n| = 10`` is ``0.155 E_1``; a Wannier set that breaks
#: a symmetry splits the two valleys of a level by 1e-3 ``E_1`` (the 8-function graphene
#: set at 10 T, 165 ueV at ``n = 1``), which the fine clustering keeps apart.
LANDAU_GROUPING = 0.02


def _dirac_comparison(levels, weights, clusters, point, field_tesla, p, tol):
    """Index the Landau levels about ``E_D`` as ``n`` and compare with the Dirac ladder.

    The levels are grouped into Landau levels at :data:`LANDAU_GROUPING` of ``E_1``,
    not at the degeneracy tolerance, so that a valley splitting is reported
    (``splittings_ev``) rather than counted as another ``n``. Each level's energy is the
    group's mean, and the site traces of the zero level are the traces over its group.
    """
    E_D, v_F = point["E_D"], point["v_F"]
    ladder = dirac_levels(field_tesla, v_F, 10)                       # eV, n = -10..10
    out = {"E_D": E_D, "v_F": v_F, "v_F_hole": point["v_F_hole"], "K": point["K"].tolist(),
           "E1_dirac_ev": float(ladder[11]), "per_k": []}
    group_tol = LANDAU_GROUPING * float(ladder[11]) / RY_TO_EV
    for ik in range(levels.shape[0]):
        lev, mult = cluster_levels(levels[ik], group_tol)
        order = np.sort(levels[ik])
        zero = int(np.argmin(np.abs(lev - E_D)))
        ratios, energies, splittings = {}, {}, {}
        for idx in range(len(lev)):
            n = idx - zero
            start = int(np.sum(mult[:idx]))
            members = order[start:start + mult[idx]]
            splittings[n] = float((members[-1] - members[0]) * RY_TO_EV)
            if n == 0 or abs(n) > 10:
                continue
            e_n = (lev[idx] - E_D) * RY_TO_EV
            energies[n] = float(e_n)
            ratios[n] = float(e_n / ladder[10 + n])
        start = int(np.sum(mult[:zero]))
        zero_weights = weights[ik, start:start + mult[zero]].sum(axis=0)
        entry = {"zero_level_ev": float((lev[zero] - E_D) * RY_TO_EV), "zero_multiplicity": int(mult[zero]),
                 "zero_site_traces": zero_weights.tolist(), "ratios": ratios, "energies_ev": energies,
                 "multiplicities": {int(i - zero): int(m) for i, m in enumerate(mult)},
                 "splittings_ev": splittings,
                 "fine_clusters": len(clusters[ik])}
        for side, sign in (("electron", 1), ("hole", -1)):
            ns = [n for n in range(1, 11) if sign * n in ratios]
            if len(ns) >= 2:
                entry[f"fit_{side}"] = linear_deviation([ratios[sign * n] for n in ns], ns)
        out["per_k"].append(entry)
    out["l_B_nm"] = float(np.sqrt(HBAR_SI / (ELECTRON_SI * field_tesla)) * 1e9)
    return out
