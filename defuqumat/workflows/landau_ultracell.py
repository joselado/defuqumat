"""Landau levels of a plane-wave crystal by the magnetic ultracell (route U, step 1).

The entry point of :mod:`defuqumat.magnetic.ultracell_field`: a converged zero-field state
(an ``SCFResult``, or anything with its ``density``), a flux ``p/q`` per chemical cell, a
band count and a magnetic k-mesh, and the levels per magnetic ``k`` of the Rayleigh-Ritz
problem on the zero-field hybrid Wannier basis of Wang and Vafek, PRB 108, 245109 (2023),
arXiv:2303.16347. The conventions are :mod:`defuqumat.magnetic.conventions` (Lu et al., PRL
135, 196401 (2025), arXiv:2505.06585; Wang et al., arXiv:2609.01053).

**What is solved.** Step 1 is local only: the zero-field states are eigenstates of ``T +
V_loc`` on the frozen potential of the state given (``local_only=True``), or of ``T`` alone
(``free=True``, the free Landau problem). The bare nonlocal term does not commute with the
magnetic translations, so it is refused here rather than dropped silently:
``local_only=False`` raises until the GIPAW-dressed term of step 2 exists.

**The k-points.** For each magnetic ``kappa2 = j/(n2 q)`` (``kmesh = (n1, n2)``, the
magnetic ``kappa1 = i/n1``), one NSCF at the ``N1 q`` chemical points ``(i/N1, kappa2 +
m/q, 0)``, all in ``[0, 1)``. The cost is ``N1 q n2`` zero-field k-points.

**Refused, by name** (:func:`~defuqumat.magnetic.conventions.refuse_regime`): spinors,
``nspin = 2``, ultrasoft and PAW datasets, spirals, DFT+U, meta-GGA, gamma-only storage,
and the nonlocal term (``local_only=False``). The field is along ``a3`` by the gauge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.magnetic.conventions import refuse_regime
from defuqumat.magnetic.hybrid import free_electron, hybrid_set, transport_gauge
from defuqumat.magnetic.hybrid import local_only as _local_only
from defuqumat.magnetic.ultracell_field import (
    MagneticUltracell, assemble, hermiticity, hermiticity_guard, solve,
)
from defuqumat.scf.driver import Calculation
from defuqumat.system.kpoints import KPoints, for_spin as kpoints_for_spin
from defuqumat.topology.states import build_plane_wave_states
from defuqumat.workflows.nscf import fixed_density_ethr, fixed_density_hamiltonians

__all__ = ["MagneticUltracellResult", "run_magnetic_ultracell", "ultracell_kpoints"]


@dataclass(frozen=True)
class MagneticUltracellResult:
    """The levels of the magnetic ultracell, Ry, and the numbers that check them.

    ``levels`` ``(nk, nlevels)`` sorted per magnetic ``k``; ``kappa`` ``(nk, 2)`` magnetic
    ``(kappa1, kappa2)``; ``rank`` and ``d_min`` (smallest kept eigenvalue of ``S`` over the
    largest) and ``d_all`` (smallest of all) per ``k``; ``hermiticity`` per ``k`` ``max |H -
    H^dagger| / max |H|``; ``w`` the half width used and ``w_measured`` the measured one
    (``None`` when no width below ``N1/2`` holds the weight to ``spread_tol``);
    ``outside`` the largest weight of any hybrid state at ``|xi1| > w``; ``dropped``
    ``max |B(w+1)| / max |B(0)|`` (``nan`` when the ket would leave half the box);
    ``centres`` ``(nkappa, q, nbnd)`` hybrid centres in cells; ``singular`` the smallest
    singular value of any transport link; ``eigenvalues`` ``(nkappa, q, N1, nbnd)`` the
    zero-field ones; ``timings`` in seconds.
    """

    flux: tuple
    field_tesla: float
    b_ph: float
    kappa: np.ndarray
    levels: np.ndarray
    rank: np.ndarray
    d_min: np.ndarray
    d_all: np.ndarray
    hermiticity: np.ndarray
    w: int
    w_measured: int | None
    outside: float
    outside_by_width: np.ndarray
    dropped: float
    n1: int
    nbnd: int
    h0: str
    centres: np.ndarray
    singular: float
    eigenvalues: np.ndarray
    davidson: dict
    timings: dict
    window: tuple | None = None
    form: str = "ket"
    blocks: tuple | None = field(default=None, repr=False)


def ultracell_kpoints(q: int, n1: int, kappas) -> np.ndarray:
    """``(len(kappas) q N1, 3)`` chemical crystal k: ``(i/N1, kappa + m/q, 0)``, ``i`` fastest."""
    return np.array([[i / n1, kappa + m / q, 0.0]
                     for kappa in kappas for m in range(q) for i in range(n1)])


def null_vector_cells(s, nbnd: int, top: int = 4):
    """The ``(j, n)`` pairs carrying most of the weight of the smallest eigenvector of ``S``."""
    d, u = np.linalg.eigh(0.5 * (s + s.conj().T))
    weight = np.abs(u[:, 0]) ** 2
    order = np.argsort(-weight)[:top]
    return [(int(i // nbnd), int(i % nbnd), round(float(weight[i]), 4)) for i in order]


def _density_of(scf):
    if scf is None:
        return None
    if hasattr(scf, "density"):
        return jnp.asarray(scf.density)
    if isinstance(scf, (str, bytes)) or hasattr(scf, "__fspath__"):
        return jnp.asarray(np.load(scf)["density"])
    return jnp.asarray(scf["density"])


def run_magnetic_ultracell(
    system,
    pseudos,
    scf,
    *,
    flux=(1, 1),
    nbnd: int,
    N1: int = 10,
    kmesh=(2, 2),
    local_only: bool = True,
    free: bool = False,
    nlevels: int | None = None,
    w: int | None = None,
    spread_tol: float = 1.0e-8,
    wmax: int | None = None,
    h0: str = "eigen",
    drop: float = 1.0e-8,
    conv_thr: float = 1.0e-10,
    k_batch="default",
    window=None,
    form: str = "ket",
    scale=1.0,
    strict: bool = True,
    keep_blocks: bool = False,
) -> MagneticUltracellResult:
    """The levels of ``system`` at flux ``p/q`` per chemical cell on the magnetic ultracell.

    Args:
        system, pseudos: the zero-field crystal (``Calculator.system``, ``.pseudos``).
        scf: the zero-field state whose ``density`` fixes the potential (an
            ``SCFResult``, a mapping or an ``npz`` path with ``density``); ``None`` only
            with ``free=True``.
        flux: ``(p, q)``, coprime, or ``(0, 1)`` for zero field.
        nbnd: zero-field bands per k-point, the basis has ``q nbnd`` per magnetic ``k``.
        N1: k-points along ``kappa1`` of the hybrid states (the box is ``N1`` cells).
        kmesh: the magnetic mesh ``(n1, n2)``: ``kappa1 = i/n1``, ``kappa2 = j/(n2 q)``.
        local_only, free: step 1's two operators (see the module docstring).
        nlevels: levels kept per ``k`` (all ``rank`` when ``None``).
        w: the half width of the band in ``Delta``; measured when ``None``: the smallest
            width with every hybrid weight outside it below ``spread_tol``, capped at
            ``wmax`` (default ``(N1 - 2) // 2``, so that ``N1 >= 2 w + 2``).
        h0: ``"eigen"`` (the zero-field operator through the eigenvalues on the bra) or
            ``"box"`` (applied on the box; exactly Hermitian), see
            :mod:`~defuqumat.magnetic.ultracell_field`.
        drop: the relative eigenvalue of ``S`` below which a direction is dropped.
        conv_thr: sets the Davidson threshold as an NSCF does
            (:func:`~defuqumat.workflows.nscf.fixed_density_ethr`).
        window: ``(inner, outer)`` in cells, the smooth window every basis function is
            multiplied by (:func:`~defuqumat.magnetic.ultracell_field.smooth_window`);
            needs ``h0 = "box"`` and fixes ``w``. ``None``: the bare hybrid states.
        form: ``"ket"`` (the design's ``2 a.p - i div a`` with ``p`` on the ket) or
            ``"symmetric"`` (``a.p + p.a`` with ``p`` on the ket and on the bra, exactly
            Hermitian on the grid); the same operator in the continuum.
        scale: a factor on the vector potential in the Hamiltonian only (traced, for
            ``d/dB`` at fixed ``p/q``); the basis keeps the momentum shifts of ``p/q``, so
            ``scale = -1`` is the field reversed against its basis, a control.
        strict: refuse (raise) when the assembled ``H`` is not Hermitian to 1e-8, and on
            the free problem when ``S`` drops a direction; ``False`` records and goes on.
    """
    refuse_regime(system, pseudos, "the magnetic ultracell")
    if not local_only and not free:
        raise NotImplementedError(
            "the magnetic ultracell carries the local potential only (step 1): the bare "
            "nonlocal term does not commute with the magnetic translations T(R), and the "
            "GIPAW-dressed term (step 2) is not implemented; pass local_only=True")
    p, q = int(flux[0]), int(flux[1])
    n1m, n2m = int(kmesh[0]), int(kmesh[1])
    kappas = [j / (n2m * q) for j in range(n2m)]
    N1 = int(N1)
    timings = {}

    # --- the zero-field states at the N1 q points of each magnetic kappa2 -----------------
    points = ultracell_kpoints(q, N1, kappas)
    cell = system.cell
    kpoints = KPoints.from_crystal(points, np.ones(len(points)), cell)
    system_k = eqx.tree_at(lambda s: s.kpoints, system, kpoints_for_spin(kpoints, system.nspin))
    t0 = time.perf_counter()
    calculation = Calculation(system_k, pseudos, k_batch=k_batch)
    calculation = calculation.for_bands(int(nbnd))
    density = _density_of(scf)
    if density is None:
        if not free:
            raise ValueError("a zero-field state is needed unless free=True")
        nelec = float(calculation.nelec)
        density = jnp.full((1,) + tuple(calculation.basis.dense.grid), nelec / float(cell.volume),
                           dtype=cell.precision.real)
    hamiltonians = fixed_density_hamiltonians(calculation, density)
    hamiltonians = tuple(free_electron(h) if free else _local_only(h) for h in hamiltonians)
    ethr = fixed_density_ethr(conv_thr, calculation.nelec)
    eigenvalues, wavefunctions, steps, notcnv = calculation.diagonalize(
        hamiltonians, int(nbnd), None, ethr, return_steps=True)
    eigenvalues = np.asarray(eigenvalues[0])
    jax.block_until_ready(wavefunctions)
    timings["nscf_s"] = time.perf_counter() - t0
    davidson = {"ethr": ethr, "steps_max": int(np.max(steps)), "notconverged": int(np.sum(notcnv))}

    # --- the parallel-transport gauge and the hybrid sets ---------------------------------
    t0 = time.perf_counter()
    states = build_plane_wave_states(calculation, wavefunctions[0])
    pairs = []
    for jk in range(n2m):
        for m in range(q):
            base = (jk * q + m) * N1
            pairs += [(base + i, base + i + 1, None) for i in range(N1 - 1)]
            pairs.append((base + N1 - 1, base, (1, 0, 0)))
    links = np.asarray(states.overlaps(pairs)).reshape(n2m, q, N1, int(nbnd), int(nbnd))
    hybrids, centres, singular = [], [], 1.0
    for jk in range(n2m):
        row = []
        for m in range(q):
            gauge = transport_gauge(links[jk, m])
            sl = slice((jk * q + m) * N1, (jk * q + m + 1) * N1)
            row.append(hybrid_set(wavefunctions[0][sl], eigenvalues[sl], states.miller[sl],
                                  states.valid[sl], gauge, kappas[jk] + m / q))
            singular = min(singular, gauge.singular)
        hybrids.append(row)
        centres.append([h.centres for h in row])
    timings["gauge_s"] = time.perf_counter() - t0

    potential = np.asarray(hamiltonians[0].potential) if h0 == "box" else None
    ultra = MagneticUltracell.build(cell, (p, q), hybrids, kappas, hamiltonians[0].grid,
                                    nbnd=int(nbnd), n1=N1, potential=potential, h0=h0,
                                    window=window, w=w, scale=scale, form=form)
    t0 = time.perf_counter()
    w_measured, weights, widths = ultra.spread(spread_tol)
    timings["spread_s"] = time.perf_counter() - t0
    cap = (N1 - 2) // 2 if wmax is None else int(wmax)
    if window is not None:
        w = ultra.w
    elif w is None:
        w = cap if w_measured is None else min(w_measured, cap)
    w = int(w)
    outside_by_width = np.max(weights.reshape(-1, len(widths)), axis=0)
    outside = float(outside_by_width[min(w, len(widths) - 1)])
    ultra = ultra.with_width(w)

    # --- the blocks, the assembly and the solve per magnetic k ---------------------------
    levels, ranks, dmins, dalls, herm, kap = [], [], [], [], [], []
    t_blocks, t_solve, dropped, kept = 0.0, 0.0, 0.0, []
    for jk in range(n2m):
        t0 = time.perf_counter()
        B, S, B_out, _ = ultra.blocks(jk)
        B = np.asarray(B)
        S = np.asarray(S)
        t_blocks += time.perf_counter() - t0
        if keep_blocks:
            kept.append((B, S))
        if B_out is not None:
            dropped = max(dropped, float(np.max(np.abs(np.asarray(B_out))) /
                                   np.max(np.abs(B[:, w]))))
        else:
            dropped = float("nan")
        t0 = time.perf_counter()
        for i1 in range(n1m):
            kappa1 = i1 / n1m
            h, s = assemble(kappa1, B, S, w, q)
            herm.append(hermiticity(h))
            if strict:
                hermiticity_guard(h)
            eps, rank, d_min, d_all = solve(h, s, drop)
            if strict and free and rank < s.shape[0]:
                raise ValueError(
                    f"S of the free problem drops {s.shape[0] - rank} of {s.shape[0]} "
                    f"directions at magnetic k ({kappa1}, {kappas[jk]}) (smallest eigenvalue "
                    f"{d_all:.3e} of the largest): a construction error; the null vector "
                    f"lives on (cell j, band n) = {null_vector_cells(s, int(nbnd))}")
            levels.append(eps if nlevels is None else eps[:nlevels])
            ranks.append(rank)
            dmins.append(d_min)
            dalls.append(d_all)
            kap.append((kappa1, kappas[jk]))
        t_solve += time.perf_counter() - t0
    timings["blocks_s"] = t_blocks
    timings["solve_s"] = t_solve
    nmin = min(len(x) for x in levels)
    magnetic = ultra.magnetic
    return MagneticUltracellResult(
        flux=(p, q),
        field_tesla=0.0 if magnetic is None else magnetic.field_tesla,
        b_ph=0.0 if magnetic is None else float(magnetic.b_ph),
        kappa=np.array(kap), levels=np.stack([x[:nmin] for x in levels]),
        rank=np.array(ranks), d_min=np.array(dmins), d_all=np.array(dalls),
        hermiticity=np.array(herm), w=w, w_measured=w_measured, outside=outside,
        outside_by_width=outside_by_width, dropped=dropped, n1=N1, nbnd=int(nbnd), h0=h0,
        centres=np.array(centres), singular=float(singular),
        eigenvalues=eigenvalues.reshape(n2m, q, N1, int(nbnd)), davidson=davidson,
        timings=timings, form=form, window=None if window is None else tuple(float(x) for x in window),
        blocks=tuple(kept) if keep_blocks else None)
