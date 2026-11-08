"""Landau levels of a plane-wave crystal by the magnetic ultracell (route U, steps 1 and 2).

The entry point of :mod:`defuqumat.magnetic.ultracell_field`: a converged zero-field state
(an ``SCFResult``, or anything with its ``density``), a flux ``p/q`` per chemical cell, a
band count and a magnetic k-mesh, and the levels per magnetic ``k`` of the Rayleigh-Ritz
problem on the zero-field hybrid Wannier basis of Wang and Vafek, PRB 108, 245109 (2023),
arXiv:2303.16347. The conventions are :mod:`defuqumat.magnetic.conventions` (Lu et al., PRL
135, 196401 (2025), arXiv:2505.06585; Wang et al., arXiv:2609.01053).

**What is solved.** Step 1 is local only: the zero-field states are eigenstates of ``T +
V_loc`` on the frozen potential of the state given (``local_only=True``), or of ``T`` alone
(``free=True``, the free Landau problem). Step 2 (``local_only=False``) takes the states of
the full zero-field Hamiltonian, unmodified, and adds the GIPAW-dressed nonlocal term of
:mod:`defuqumat.magnetic.ultracell_nonlocal`: with ``h0 = "eigen"``

    H = H_0 + Delta H + (V_NL^B - V_NL^0),

``V_NL^0`` being inside the eigenvalues and subtracted on the grid points it is dressed on.
``nonlocal_term`` chooses the term: ``"gipaw"`` (the default of step 2), ``"bare"`` (the
undressed term carried as if it commuted with ``T(R)``, a control) or ``"none"`` (step 1's
default). On a local-only basis ``"gipaw"`` and ``"bare"`` add the whole term.

**The k-points.** For each magnetic ``kappa2 = j/(n2 q)`` (``kmesh = (n1, n2)``, the
magnetic ``kappa1 = i/n1``), one NSCF at the ``N1 q`` chemical points ``(i/N1, kappa2 +
m/q, 0)``, all in ``[0, 1)``. The cost is ``N1 q n2`` zero-field k-points.

**Refused, by name** (:func:`~defuqumat.magnetic.conventions.refuse_regime`): spinors,
``nspin = 2``, ultrasoft and PAW datasets, spirals, DFT+U, meta-GGA, gamma-only storage;
a full zero-field basis with no nonlocal term, and a nonlocal term on the free problem. The
field is along ``a3`` by the gauge.

**The projected trial basis (step 3).** ``trials=`` replaces the transport gauge, which
refuses on every case it was run on (P176), by :func:`~defuqumat.magnetic.hybrid.projected_hybrid`:
the basis function of trial ``j`` is the trial projected on the lowest ``N`` zero-field bands
at every k of the loop, so the basis has ``J = len(trials)`` functions per cell and the result's
``nbnd`` is ``J``, while ``N`` (the band window, solved by the NSCF) is fixed by the
projectability and is not a convergence knob. ``trials`` is a
:class:`~defuqumat.magnetic.hybrid.StripeTrials` (the free problem), ``"pao"`` (every
pseudo-atomic orbital of the datasets' ``PP_PSWFC``), or a list of Wannier90 projection lines
and :class:`~defuqumat.wannier.pao.PAOProjection` objects in any mix; the spans are nested in
the order given. A run refuses when a trial's projectability falls below ``1 -
projectability_tol`` at any k (default 1e-4; fable's probe reads Hermiticity 2.5e-4 at 2.7e-3).
Its defaults are fable's: ``window = (3, 4)``, ``h0 = "box"``, ``form = "symmetric"``,
``N1 = 16`` (with any window), so the matrix is an exact Rayleigh-Ritz problem of the box operator and every level
is an upper bound that falls monotonically with ``J``. On a projected basis a direction of ``S``
dropped by ``drop`` is the conditioning of non-orthogonal trials, recorded in ``rank``, not a
construction error, so the free problem's rank refusal applies to the hybrid basis only.

**The zero-flux check, restated.** At ``p = 0`` the levels are the zero-field bands only inside
the span: with ``J = N`` trials whose projections are independent at every k (and no window,
odd ``N1``, ``w = (N1 - 1)/2``, ``kappa1`` on the ``N1`` mesh) they are the ``N`` bands to
round-off; with ``J < N`` they lie above the bands, by what the span of the trials misses.

**The projector check.** Whenever the grid projectors are built, the run reports
``nonlocal_error``: the largest difference, in Ry, between ``<h_{n,0}|V_NL^0|h_{n',0}>`` on
the box grid and the same block from the plane-wave projectors (``vkb``) through the
transport gauge, at cell 0 and every magnetic ``kappa2``. At zero flux this is the only
check that sees the normalisation and the ``e^{iK.r}`` of the projection, since there
``V_NL^B - V_NL^0`` vanishes whatever both are.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.magnetic.conventions import refuse_regime
from defuqumat.magnetic.hybrid import (
    StripeTrials, atomic_trials, free_electron, hybrid_set, projected_hybrid, stripe_trials,
    transport_gauge,
)
from defuqumat.magnetic.hybrid import local_only as _local_only
from defuqumat.magnetic.ultracell_field import (
    MagneticUltracell, _bra_fields, assemble, hermiticity, hermiticity_guard, solve,
)
from defuqumat.magnetic.ultracell_nonlocal import grid_projectors, nonlocal_block
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

    On a projected basis (``trials=``): ``nbnd`` is ``J``, the functions per cell; ``N`` the band
    window and ``eigenvalues`` ``(nkappa, q, N1, N)``; ``trials`` the trials' names;
    ``projectability`` ``(nkappa, q, N1, J)`` and ``projectability_deficit`` its largest ``1 -
    p``; ``singular`` the smallest singular value of the normalized projections ``A``.
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
    nonlocal_term: str = "none"
    nonlocal_mode: str = "none"
    nonlocal_error: float | None = None
    projector_radius: tuple | None = None
    projector_dropped: tuple | None = None
    region_weights: np.ndarray | None = None
    N: int | None = None
    trials: tuple | None = None
    projectability: np.ndarray | None = None
    projectability_deficit: float | None = None


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


def solve_vectors(h, s, drop: float = 1.0e-8):
    """``(eps, c)``: :func:`~defuqumat.magnetic.ultracell_field.solve` with its vectors.

    ``c`` ``(dim, rank)`` in the original basis, ``S``-orthonormal (``c^dagger S c = 1``).
    """
    h = 0.5 * (np.asarray(h) + np.asarray(h).conj().T)
    s = 0.5 * (np.asarray(s) + np.asarray(s).conj().T)
    d, u = np.linalg.eigh(s)
    keep = d > drop * d.max()
    x = u[:, keep] / np.sqrt(d[keep])[None, :]
    eps, y = np.linalg.eigh(x.conj().T @ h @ x)
    return eps, x @ y


def sublattice_masks(cell, structure, ultracell, groups) -> np.ndarray:
    """``(len(groups), npts)`` masks on the box: one where the nearest atom is in the group
    (a point equidistant from atoms of several groups is shared equally).

    ``groups`` lists atom indices of the chemical cell (graphene: ``[[0], [1]]``, A and
    B); the nearest atom is by the minimum-image distance over the chemical lattice, so the
    masks are periodic over the chemical cell and tile the box.
    """
    from defuqumat.magnetic.conventions import minimum_image

    at = np.asarray(cell.at, dtype=float)
    grid = ultracell.cell_grid
    idx = np.stack(np.meshgrid(*[np.arange(n) for n in grid], indexing="ij"), -1).reshape(-1, 3)
    r = (idx / np.asarray(grid, float)) @ at
    pos = np.asarray(structure.positions, dtype=float)
    dist = np.stack([np.linalg.norm(minimum_image(r - p, at), axis=-1) for p in pos], -1)
    # a point equidistant from several atoms is shared equally among them
    tied = dist <= dist.min(axis=-1, keepdims=True) + 1e-9
    share = tied / tied.sum(axis=-1, keepdims=True)
    masks = np.stack([share[:, list(g)].sum(axis=-1).reshape(grid) for g in groups])
    return np.asarray(ultracell.tile(masks)).reshape(len(groups), -1)


def _resolve_nonlocal(local_only: bool, free: bool, h0: str, nonlocal_term):
    """``(term, mode)``: the nonlocal term asked for and the one the blocks add."""
    if nonlocal_term is None:
        nonlocal_term = "gipaw" if not (local_only or free) else "none"
    if nonlocal_term not in ("none", "gipaw", "bare"):
        raise ValueError(f"nonlocal_term must be 'none', 'gipaw' or 'bare', got {nonlocal_term!r}")
    if free and nonlocal_term != "none":
        raise ValueError("the free Landau problem has no nonlocal term")
    full = not (local_only or free)
    if full and nonlocal_term == "none":
        raise ValueError("a full zero-field basis carries V_NL^0 in its eigenvalues: say what the "
                         "field does with it, nonlocal_term = 'gipaw' (dressed) or 'bare' "
                         "(the control)")
    if nonlocal_term == "none":
        return nonlocal_term, "none"
    if h0 == "eigen" and full:
        return nonlocal_term, "difference" if nonlocal_term == "gipaw" else "none"
    return nonlocal_term, nonlocal_term


def _nonlocal_check(ultra, projectors, hamiltonian, dij, wavefunctions, gauges, kappas, q, n1):
    """Largest ``|<h|V_NL^0|h'>_grid - <h|V_NL^0|h'>_vkb|`` (Ry) at cell 0, every kappa2.

    ``dij`` ``(nkb, nkb)`` the coefficients the grid projectors carry; the plane-wave side
    uses ``hamiltonian.projectors`` (``vkb``) with the same ``dij``.
    """
    gp = projectors
    worst = 0.0
    dij = np.asarray(dij)
    for jk, kappa in enumerate(kappas):
        hybrid = ultra.hybrids[jk][0]
        u, _ = _bra_fields(hybrid.values, ultra.bra_index(hybrid), ultra.grid)
        grid_v = np.asarray(nonlocal_block(u, u, gp.index, gp.beta, gp.d_b2, gp.phase, gp.dij,
                                           jnp.asarray(kappa, ultra.g_b2.dtype), mode="bare"))
        rotation = gauges[jk].rotation
        exact = 0.0
        for i in range(n1):
            ik = (jk * q) * n1 + i
            bec = np.asarray(hamiltonian.projectors.project(wavefunctions[ik], ik))   # (nbnd, nkb)
            v = bec.conj() @ dij @ bec.T
            exact = exact + rotation[i].conj().T @ v @ rotation[i] / n1
        worst = max(worst, float(np.max(np.abs(grid_v - exact))))
    return worst


def _density_of(scf):
    if scf is None:
        return None
    if hasattr(scf, "density"):
        return jnp.asarray(scf.density)
    if isinstance(scf, (str, bytes)) or hasattr(scf, "__fspath__"):
        return jnp.asarray(np.load(scf)["density"])
    return jnp.asarray(scf["density"])


def _resolve_trials(trials, system, pseudos):
    """``trials=`` as a :class:`StripeTrials` or a list of atom-centred projections."""
    if isinstance(trials, StripeTrials):
        return trials
    from defuqumat.wannier.pao import pao_projections
    from defuqumat.wannier.projections import parse_projections

    crystal = np.asarray(system.structure.positions_crystal(system.cell), dtype=float)
    if isinstance(trials, str):
        if trials != "pao":
            raise ValueError(f"trials must be 'pao', a StripeTrials or a list of projections, "
                             f"got {trials!r}")
        return pao_projections(pseudos, system.structure, crystal)
    out = []
    bg = np.asarray(system.cell.bg, dtype=float)
    symbols = list(system.structure.symbols)
    for t in trials:
        if isinstance(t, str):
            out.extend(parse_projections([t], symbols, crystal.T, bg))
        elif all(hasattr(t, a) for a in ("l", "m", "site_frac")):
            out.append(t)
        else:
            raise ValueError(f"a trial must be a Wannier90 projection line or a projection with l, m "
                             f"and site_frac, got {t!r}")
    if not out:
        raise ValueError("trials= is empty")
    return out


def _trial_names(trials):
    if isinstance(trials, StripeTrials):
        return trials.names
    return tuple(getattr(t, "name", None) or f"site {np.round(np.asarray(t.site_frac), 4).tolist()} "
                 f"l={t.l} mr={t.m}" for t in trials)


def run_magnetic_ultracell(
    system,
    pseudos,
    scf,
    *,
    flux=(1, 1),
    nbnd: int | None = None,
    N1: int | None = None,
    kmesh=(2, 2),
    local_only: bool = True,
    free: bool = False,
    nlevels: int | None = None,
    w: int | None = None,
    spread_tol: float = 1.0e-8,
    wmax: int | None = None,
    h0: str | None = None,
    drop: float = 1.0e-8,
    conv_thr: float = 1.0e-10,
    k_batch="default",
    window="default",
    form: str | None = None,
    scale=1.0,
    strict: bool = True,
    keep_blocks: bool = False,
    nonlocal_term: str | None = None,
    projector_cut: float = 1.0e-6,
    sublattices=None,
    trials=None,
    N: int | None = None,
    projectability_tol: float | None = 1.0e-4,
    kappa2=None,
) -> MagneticUltracellResult:
    """The levels of ``system`` at flux ``p/q`` per chemical cell on the magnetic ultracell.

    Args:
        system, pseudos: the zero-field crystal (``Calculator.system``, ``.pseudos``).
        scf: the zero-field state whose ``density`` fixes the potential (an
            ``SCFResult``, a mapping or an ``npz`` path with ``density``); ``None`` only
            with ``free=True``.
        flux: ``(p, q)``, coprime, or ``(0, 1)`` for zero field.
        nbnd: zero-field bands per k-point, the basis has ``q nbnd`` per magnetic ``k`` (the
            hybrid basis; a projected run has ``q J``).
        N1: k-points along ``kappa1`` of the hybrid states (the box is ``N1`` cells).
        kmesh: the magnetic mesh ``(n1, n2)``: ``kappa1 = i/n1``, ``kappa2 = j/(n2 q)``.
        local_only, free: the zero-field basis: ``T + V_loc`` (step 1, ``local_only``),
            ``T`` (``free``), or the full Hamiltonian (``local_only=False``, step 2).
        nonlocal_term: ``"gipaw"``, ``"bare"`` or ``"none"`` (module docstring); ``None``
            is ``"gipaw"`` on a full basis and ``"none"`` otherwise.
        projector_cut: the grid projector is cut where ``|beta|`` falls below this
            fraction of its largest value (the dropped norm is reported).
        sublattices: atom groups of the chemical cell (``[[0], [1]]`` for graphene's A
            and B): the weight of every level on the box region nearest each group is
            returned as ``region_weights`` ``(nk, nlevels, ngroups)``.
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
            the free problem's hybrid basis when ``S`` drops a direction; ``False`` records
            and goes on.
        trials: the projected trial basis (module docstring): a
            :class:`~defuqumat.magnetic.hybrid.StripeTrials`, ``"pao"``, or a list of Wannier90
            projection lines and projection objects. ``None``: the hybrid Wannier basis of
            the transport gauge (refused in practice, P176).
        N: the band window of a projected run, the zero-field bands solved and projected on;
            needed with ``trials`` and refused without it (``nbnd`` is then ``J``).
        projectability_tol: a projected run refuses when ``1 - p`` of any trial at any k
            exceeds it; ``None`` records the projectability and goes on.
        kappa2: explicit magnetic ``kappa2`` values in ``[0, 1/q)``, in place of the
            ``kmesh[1]`` values ``j/(n2 q)`` (``kmesh[0]`` still sets ``kappa1``).

    Defaults that depend on the basis: ``N1`` 16 with a window, 10 without; ``h0`` ``"eigen"``
    or ``"box"``; ``form`` ``"ket"`` or ``"symmetric"``; ``window`` ``None`` or ``(3, 4)``.
    """
    refuse_regime(system, pseudos, "the magnetic ultracell")
    projected = trials is not None
    if projected:
        if N is None:
            raise ValueError("a projected run (trials=) needs N, the zero-field bands the trials "
                             "are projected on")
        if nbnd is not None:
            raise ValueError("a projected run has J = len(trials) functions per cell; nbnd is not "
                             "one of its knobs, N (the band window) is")
        nsolve = int(N)
    else:
        if N is not None:
            raise ValueError("N is the band window of a projected run; give trials= with it")
        if nbnd is None:
            raise ValueError("the hybrid basis needs nbnd, the zero-field bands per k-point")
        nsolve = int(nbnd)
    h0 = ("box" if projected else "eigen") if h0 is None else h0
    form = ("symmetric" if projected else "ket") if form is None else form
    if isinstance(window, str):
        if window != "default":
            raise ValueError(f"window must be (inner, outer), None or 'default', got {window!r}")
        window = (3.0, 4.0) if projected else None
    # fable (2026-11-08): N1 follows the window, whose bound is ceil(2 outer) - 1 + 2 outer
    N1 = (16 if window is not None else 10) if N1 is None else int(N1)
    if window is not None:
        # MagneticUltracell.build's own bound, checked here so that it fires before the NSCF
        bound = int(np.ceil(2.0 * float(window[1]))) - 1 + 2.0 * float(window[1])
        if N1 < bound:
            raise ValueError(f"the window ({float(window[0])}, {float(window[1])}) needs N1 >= "
                             f"{bound:g} cells, got N1 = {N1}")
    nonlocal_term, nonlocal_mode = _resolve_nonlocal(local_only, free, h0, nonlocal_term)
    p, q = int(flux[0]), int(flux[1])
    n1m, n2m = int(kmesh[0]), int(kmesh[1])
    if kappa2 is None:
        kappas = [j / (n2m * q) for j in range(n2m)]
    else:
        kappas = [float(k) for k in kappa2]
        if not kappas or any(not 0.0 <= k < 1.0 / q for k in kappas):
            raise ValueError(f"kappa2 must be magnetic values in [0, 1/q) = [0, {1.0 / q:g}), got "
                             f"{kappa2!r}")
        n2m = len(kappas)
    timings = {}

    # --- the zero-field states at the N1 q points of each magnetic kappa2 -----------------
    points = ultracell_kpoints(q, N1, kappas)
    cell = system.cell
    kpoints = KPoints.from_crystal(points, np.ones(len(points)), cell)
    system_k = eqx.tree_at(lambda s: s.kpoints, system, kpoints_for_spin(kpoints, system.nspin))
    t0 = time.perf_counter()
    calculation = Calculation(system_k, pseudos, k_batch=k_batch)
    calculation = calculation.for_bands(nsolve)
    density = _density_of(scf)
    if density is None:
        if not free:
            raise ValueError("a zero-field state is needed unless free=True")
        nelec = float(calculation.nelec)
        density = jnp.full((1,) + tuple(calculation.basis.dense.grid), nelec / float(cell.volume),
                           dtype=cell.precision.real)
    hamiltonians = fixed_density_hamiltonians(calculation, density)
    full_hamiltonian = hamiltonians[0]
    if free:
        hamiltonians = tuple(free_electron(h) for h in hamiltonians)
    elif local_only:
        hamiltonians = tuple(_local_only(h) for h in hamiltonians)
    ethr = fixed_density_ethr(conv_thr, calculation.nelec)
    eigenvalues, wavefunctions, steps, notcnv = calculation.diagonalize(
        hamiltonians, nsolve, None, ethr, return_steps=True)
    eigenvalues = np.asarray(eigenvalues[0])
    jax.block_until_ready(wavefunctions)
    timings["nscf_s"] = time.perf_counter() - t0
    davidson = {"ethr": ethr, "steps_max": int(np.max(steps)), "notconverged": int(np.sum(notcnv))}

    # --- the parallel-transport gauge and the hybrid sets ---------------------------------
    t0 = time.perf_counter()
    states = build_plane_wave_states(calculation, wavefunctions[0])
    if projected:
        trials = _resolve_trials(trials, calculation.system, pseudos)
        if isinstance(trials, StripeTrials):
            # f of StripeTrials, the chirp of the analytic Landau state at this field (u1 0b)
            b = np.linalg.inv(np.asarray(cell.at, dtype=float)).T
            # of p/q only, never of scale: the basis is fixed by the flux (fable, 2026-11-08)
            chirp = 0.0 if p == 0 else (np.pi * p / q) * float(b[0] @ b[1]) / float(b[0] @ b[0])
            trial_values = stripe_trials(trials, points, states.miller, states.valid, chirp)
        else:
            trial_values = atomic_trials(states, trials, pseudos, float(cell.volume))
        nbnd = trial_values.shape[1]
    else:
        pairs = []
        for jk in range(n2m):
            for m in range(q):
                base = (jk * q + m) * N1
                pairs += [(base + i, base + i + 1, None) for i in range(N1 - 1)]
                pairs.append((base + N1 - 1, base, (1, 0, 0)))
        links = np.asarray(states.overlaps(pairs)).reshape(n2m, q, N1, nsolve, nsolve)
    nbnd = int(nbnd)
    hybrids, centres, singular, first_gauges, projectability = [], [], 1.0, [], []
    for jk in range(n2m):
        row = []
        for m in range(q):
            sl = slice((jk * q + m) * N1, (jk * q + m + 1) * N1)
            if projected:
                gauge = projected_hybrid(wavefunctions[0][sl], trial_values[sl], nsolve)
                projectability.append(gauge.projectability)
            else:
                gauge = transport_gauge(links[jk, m])
            if m == 0:
                first_gauges.append(gauge)
            row.append(hybrid_set(wavefunctions[0][sl], eigenvalues[sl], states.miller[sl],
                                  states.valid[sl], gauge, kappas[jk] + m / q))
            singular = min(singular, gauge.singular)
        hybrids.append(row)
        centres.append([h.centres for h in row])
    deficit = None
    if projected:
        projectability = np.stack(projectability).reshape(n2m, q, N1, nbnd)
        lost = 1.0 - projectability
        deficit = float(lost.max())
        if projectability_tol is not None and deficit > float(projectability_tol):
            jk, m, i, j = np.unravel_index(int(np.argmax(lost)), lost.shape)
            raise ValueError(
                f"trial {j} ({_trial_names(trials)[j]}) is not inside the {nsolve}-band window: "
                f"1 - projectability = {deficit:.3e} > {float(projectability_tol):.1e} at the "
                f"chemical k {np.round(points[(jk * q + m) * N1 + i], 6).tolist()}; its projection "
                "carries a jump in k and delocalizes the basis. Raise N, or choose trials that "
                "the lowest N bands hold (projectability_tol=None records and goes on)")
    timings["gauge_s"] = time.perf_counter() - t0

    potential = np.asarray(hamiltonians[0].potential) if h0 == "box" else None
    projectors = None
    nonlocal_error = None
    if nonlocal_term != "none":
        from defuqumat.magnetic.conventions import MagneticCell

        t0 = time.perf_counter()
        magnetic = None if p == 0 else MagneticCell.build(cell, p, q, scale=scale)
        # the coefficients of the full zero-field operator (a local-only basis zeroed them)
        dij = np.asarray(full_hamiltonian.coefficients)
        projectors = grid_projectors(cell, calculation.system.structure, calculation.pseudos, dij,
                                     full_hamiltonian.projectors.atom_of_channel,
                                     hamiltonians[0].grid, N1, magnetic, cut=projector_cut)
        timings["projectors_s"] = time.perf_counter() - t0
    regions = None
    if sublattices is not None:
        from defuqumat.ultracell.grid import Ultracell

        box = Ultracell.build((N1, 1, 1), hamiltonians[0].grid, cell.precision)
        regions = sublattice_masks(cell, calculation.system.structure, box, sublattices)
    ultra = MagneticUltracell.build(cell, (p, q), hybrids, kappas, hamiltonians[0].grid,
                                    nbnd=int(nbnd), n1=N1, potential=potential, h0=h0,
                                    window=window, w=w, scale=scale, form=form,
                                    projectors=projectors, nonlocal_mode=nonlocal_mode,
                                    regions=regions)
    if projectors is not None:
        t0 = time.perf_counter()
        nonlocal_error = _nonlocal_check(ultra, projectors, full_hamiltonian, dij, wavefunctions[0],
                                         first_gauges, kappas, q, N1)
        timings["nonlocal_check_s"] = time.perf_counter() - t0
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
    levels, ranks, dmins, dalls, herm, kap, weights_r = [], [], [], [], [], [], []
    t_blocks, t_solve, dropped, kept = 0.0, 0.0, 0.0, []
    for jk in range(n2m):
        t0 = time.perf_counter()
        out = ultra.blocks(jk, with_regions=regions is not None)
        B, S, B_out = np.asarray(out[0]), np.asarray(out[1]), out[2]
        R = np.asarray(out[4]) if regions is not None else None
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
            if R is not None:
                _, vec = solve_vectors(h, s, drop)
                rw = np.stack([np.real(np.einsum("in,ij,jn->n", vec.conj(),
                                                 assemble(kappa1, R[:, :, ir], S, w, q)[0], vec))
                               for ir in range(R.shape[2])], -1)
                weights_r.append(rw if nlevels is None else rw[:nlevels])
            if strict and free and not projected and rank < s.shape[0]:
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
        eigenvalues=eigenvalues.reshape(n2m, q, N1, nsolve), davidson=davidson,
        timings=timings, form=form, window=None if window is None else tuple(float(x) for x in window),
        blocks=tuple(kept) if keep_blocks else None,
        nonlocal_term=nonlocal_term, nonlocal_mode=nonlocal_mode, nonlocal_error=nonlocal_error,
        projector_radius=None if projectors is None else projectors.radius,
        projector_dropped=None if projectors is None else projectors.dropped,
        region_weights=None if not weights_r else np.stack([x[:nmin] for x in weights_r]),
        N=nsolve if projected else None, trials=_trial_names(trials) if projected else None,
        projectability=projectability if projected else None, projectability_deficit=deficit)
