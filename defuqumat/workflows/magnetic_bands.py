"""Magnetic Bloch bands of a crystal at flux ``p/q`` per cell: route P, step 1 (local potential only).

The entry point of :mod:`defuqumat.magnetic.blochgrid`, the plane-wave magnetic Bloch route
of Lu et al., PRL 135, 196401 (2025), arXiv:2505.06585: the levels of ``pi^2 + V_loc`` on the
magnetic cell ``q a1 x a2 x a3`` at each magnetic ``k`` of a closed ``(n1, n2)`` mesh, by
:func:`~defuqumat.solvers.davidson.davidson_eigensolver_all` with the operator's own
preconditioner, one k-point at a time (QE's ``c_bands`` loop; the working set of one solve is
sized and refused by number before anything runs), and with ``chern=True`` the Chern number
of each isolated group of levels by FHS (Fukui, Hatsugai and Suzuki) on the box states in the
atomic gauge of the magnetic cell, ``u_{k+b} = exp(-i b.r) u_k``.

The local potential is the zero-field SCF's (``SCFResult.potential[0]``, or the shared
``plans/orbital-field/graphene-scf.npz``), filtered to the wavefunction box and tiled; with
``free=True`` it is off, which is the Landau problem. The conventions are
:mod:`defuqumat.magnetic.conventions` (Wang et al., arXiv:2609.01053, and Wang and Vafek,
PRB 108, 245109 (2023), arXiv:2303.16347, for the magnetic translations and the labels).

``k`` is labelled ``kappa`` in the magnetic cell's reciprocal basis (``kappa1`` spans
``b1/q``, ``kappa2`` the chemical ``b2``); ``T(a1)`` sends ``kappa2 -> kappa2 - p/q``, so the
spectrum is periodic in ``kappa2`` with period ``1/q``. Never compare ``eps(k)`` at a named
``k`` across routes: compare :attr:`MagneticBands.sorted_levels`.

Refused by name: the regimes of :func:`~defuqumat.magnetic.conventions.refuse_regime`
(spinors, ``nspin = 2``, ultrasoft, PAW, spirals, DFT+U, meta-GGA, gamma-only storage); the
nonlocal term (``local_only=False``, the GIPAW-dressed projectors of step 2); a field not
along ``a3``; a memory estimate above the limit; a Chern number on a mesh other than the
closed plane mesh. A Davidson solve that has not converged after ``max_iterations`` falls back
to a dense diagonalisation for ``ndim <= 20000`` and raises above.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from defuqumat.magnetic.blochgrid import (
    DENSE_LIMIT, MagneticGridHamiltonian, dense_matrix, estimate_bytes, grid_positions,
    magnetic_cell, refuse_memory, start_block, wavefunction_box,
)
from defuqumat.magnetic.conventions import refuse_regime
from defuqumat.units import tesla_from_flux

__all__ = ["MagneticBands", "run_magnetic_bands", "level_groups", "local_potential_of"]


@dataclass(frozen=True)
class MagneticBands:
    """The magnetic Bloch levels, Ry, and what the solve took.

    ``eigenvalues`` ``(nk, nbnd)`` ascending; ``kpoints`` ``(nk, 3)`` magnetic crystal
    coordinates; ``steps`` and ``unsettled`` per k the Davidson step count and the bands still
    unsettled when it stopped; ``solver`` per k ``'davidson'`` or ``'dense'``;
    ``memory_estimate`` bytes of one k-point's solve; ``chern`` a tuple of ``(first, last,
    C)`` for each isolated group of bands (``None`` unless asked); ``states`` ``(nk, nkeep,
    ndim)`` when kept.
    """

    flux: tuple
    field_tesla: float
    b_ph: float
    grid: tuple
    box: tuple
    ndim: int
    kpoints: np.ndarray
    eigenvalues: np.ndarray
    steps: np.ndarray
    unsettled: np.ndarray
    solver: tuple
    preconditioner: str
    memory_estimate: int
    timings: dict
    chern: tuple | None = None
    states: np.ndarray | None = field(default=None, repr=False)

    @property
    def sorted_levels(self) -> np.ndarray:
        """Every level over the mesh, sorted: the quantity routes and geometries are compared by."""
        return np.sort(self.eigenvalues.reshape(-1))


def local_potential_of(scf) -> np.ndarray:
    """The total local potential of channel 0 from an ``SCFResult``, an npz path or a mapping."""
    if isinstance(scf, (str, bytes)) or hasattr(scf, "__fspath__"):
        with np.load(scf) as data:
            return np.asarray(data["potential"])[0]
    if hasattr(scf, "potential"):
        return np.asarray(scf.potential)[0]
    return np.asarray(scf["potential"])[0]


def level_groups(eigenvalues, gap: float = 1e-6) -> list[tuple[int, int]]:
    """Consecutive bands joined wherever ``min_k (e_{n+1} - e_n) <= gap``; the last group is dropped.

    The last group may be cut by ``nbnd`` (its partner above was not computed), so a Chern
    number is not taken over it.
    """
    e = np.asarray(eigenvalues)
    separation = np.min(e[:, 1:] - e[:, :-1], axis=0)
    groups, first = [], 0
    for n, s in enumerate(separation):
        if s > gap:
            groups.append((first, n))
            first = n + 1
    return groups


def _chern_numbers(states, kmesh, box, groups) -> tuple:
    from defuqumat.topology.berry import fhs_curvature
    from defuqumat.topology.mesh import plane_mesh
    from defuqumat.topology.states import ArrayStates

    import jax.numpy as jnp

    mesh = plane_mesh(tuple(kmesh), axis=2)
    positions = jnp.asarray(grid_positions(box))
    out = []
    for first, last in groups:
        subset = ArrayStates(coefficients=jnp.asarray(states[:, first:last + 1, :]),
                             orbital_positions=positions)
        curvature = fhs_curvature(subset, mesh)
        out.append((first, last, float(curvature.chern_number), float(curvature.max_flux)))
    return tuple(out)


def run_magnetic_bands(system, pseudos, scf=None, *, flux=(1, 1), kmesh=(2, 2), nbnd: int,
                       local_only: bool = True, free: bool = False, chern: bool = False,
                       david: int = 4, ethr: float = 1e-12, max_iterations: int = 200,
                       preconditioner: str = "two-gauge", kpoints=None, grid=None,
                       memory_limit: float | None = None, keep_states: bool = False,
                       field_axis: int = 2, chern_gap: float = 1e-6,
                       band_batch="default", unconverged: str = "fallback") -> MagneticBands:
    """The ``nbnd`` lowest levels of ``pi^2 + V_loc`` at flux ``p/q`` on a ``kmesh`` of the magnetic zone.

    Args:
        system, pseudos: the chemical cell's :class:`~defuqumat.system.builder.System` and its
            datasets (checked by :func:`~defuqumat.magnetic.conventions.refuse_regime`).
        scf: the zero-field ground state (``SCFResult``, npz path or mapping with
            ``potential``); ignored with ``free=True``.
        flux: ``(p, q)`` flux quanta per chemical cell, coprime; ``(0, 1)`` is zero field.
        kmesh: ``(n1, n2)``, the closed plane mesh ``kappa = (i/n1, j/n2, 0)`` in the order of
            :func:`~defuqumat.topology.mesh.plane_mesh`; ``kpoints`` overrides it.
        local_only: must be ``True`` (the nonlocal term is step 2, refused).
        free: the Landau problem, all potentials off.
        chern: the Chern number of each isolated group of levels (``chern_gap`` Ry).
        david: Davidson's subspace multiple; ``ethr``, ``max_iterations`` its threshold and
            budget (``david = 4``, QE's default: at 2 the free problem at ``q = 10`` leaves 8 of 12
            bands unsettled after 200 steps); ``preconditioner`` ``'two-gauge'`` (the hook's
            default), ``'kinetic'`` (the single-gauge rule) or ``'identity'``.
        grid: the chemical wavefunction box (default :func:`wavefunction_box` at ``ecutwfc``).
        memory_limit: bytes (default :data:`~defuqumat.magnetic.blochgrid.MEMORY_LIMIT`).
        keep_states: return the states ``(nk, nbnd, ndim)``.
        unconverged: ``'fallback'`` (a dense solve for ``ndim <= 20000``, a refusal above) or
            ``'keep'`` (the Davidson values as they stand, with ``unsettled`` saying so: for
            measuring a preconditioner, never for a result).
    """
    import jax.numpy as jnp

    from defuqumat.solvers.davidson import davidson_eigensolver_all

    refuse_regime(system, pseudos, "Magnetic Bloch bands (route P)")
    if not local_only:
        raise NotImplementedError(
            "route P with the nonlocal pseudopotential needs the GIPAW-dressed projectors "
            "on the magnetic box (Lu eq. 6), which is step 2 and not implemented; the bare "
            "nonlocal term breaks the magnetic translations and is not a substitute. Pass "
            "local_only=True for pi^2 + V_loc")
    if unconverged not in ("fallback", "keep"):
        raise ValueError(f"unconverged {unconverged!r}: 'fallback' or 'keep'")
    if field_axis != 2:
        raise NotImplementedError(
            f"a field along a{field_axis + 1}: the Landau gauge of the conventions puts B along "
            "a3; relabel the lattice vectors so that the field is along a3")
    if system.cell.precision.eps > 1e-10:
        raise NotImplementedError("route P is validated in double precision only")
    p, q = (int(x) for x in flux)
    cell = system.cell
    mcell = magnetic_cell(cell, p, q)
    grid = wavefunction_box(cell, system.ecutwfc) if grid is None else tuple(grid)
    if kpoints is None:
        from defuqumat.topology.mesh import plane_mesh

        kpts = plane_mesh(tuple(kmesh), axis=2).flat()
    else:
        if chern:
            raise ValueError("a Chern number needs the closed plane mesh: pass kmesh, not kpoints")
        kpts = np.asarray(kpoints, dtype=float)
        if kpts.shape[-1] == 2:
            kpts = np.concatenate([kpts, np.zeros((len(kpts), 1))], axis=-1)
    nk = len(kpts)
    ndim = q * int(np.prod(grid))
    kept = nk * nbnd if (chern or keep_states) else 0
    estimate = estimate_bytes(q, nbnd, david, grid=grid, kept=kept)
    refuse_memory(estimate, memory_limit, f"route P at p/q = {p}/{q}, nbnd {nbnd}, ndim {ndim}")
    potential = None if free else local_potential_of(scf)

    timings = {}
    t0 = time.perf_counter()
    hamiltonian = MagneticGridHamiltonian.build(cell, p, q, system.ecutwfc, kpts, potential,
                                                grid=grid, preconditioner=preconditioner,
                                                band_batch=band_batch)
    timings["build_s"] = time.perf_counter() - t0

    eigenvalues = np.zeros((nk, nbnd))
    steps = np.zeros(nk, dtype=int)
    unsettled = np.zeros(nk, dtype=int)
    solver = []
    states = np.zeros((nk, nbnd, ndim), dtype=hamiltonian.dtype) if kept else None
    t0 = time.perf_counter()
    per_k = []
    for ik in range(nk):
        t1 = time.perf_counter()
        start = start_block(hamiltonian, ik, nbnd)
        values, vectors, step, left = davidson_eigensolver_all(
            hamiltonian, nbnd, psi0=start[None], ethr=ethr, david=david,
            max_iterations=max_iterations, indices=np.array([ik]), return_steps=True)
        steps[ik], unsettled[ik] = int(step[0]), int(left[0])
        if unsettled[ik] == 0 or unconverged == "keep":
            eigenvalues[ik] = np.asarray(values[0])
            solver.append("davidson" if unsettled[ik] == 0 else "davidson-unsettled")
            if states is not None:
                states[ik] = np.asarray(vectors[0])
        elif ndim <= DENSE_LIMIT:
            matrix = dense_matrix(hamiltonian, ik)
            w, v = np.linalg.eigh(0.5 * (matrix + matrix.conj().T))
            eigenvalues[ik] = w[:nbnd]
            solver.append("dense")
            if states is not None:
                states[ik] = v[:, :nbnd].T
        else:
            raise RuntimeError(
                f"Davidson left {unsettled[ik]} of {nbnd} bands unsettled at k-point {ik} after "
                f"{steps[ik]} steps (preconditioner {preconditioner!r}), and ndim = {ndim} is "
                f"above the dense fallback's {DENSE_LIMIT}: refused rather than returned")
        per_k.append(time.perf_counter() - t1)
    timings["solve_s"] = time.perf_counter() - t0
    timings["per_k_s"] = per_k

    groups_chern = None
    if chern:
        t0 = time.perf_counter()
        groups_chern = _chern_numbers(states, kmesh, hamiltonian.box,
                                      level_groups(eigenvalues, chern_gap))
        timings["chern_s"] = time.perf_counter() - t0

    field_tesla = 0.0 if p == 0 else float(tesla_from_flux(p, q, float(mcell.area)))
    return MagneticBands(
        flux=(p, q), field_tesla=field_tesla, b_ph=float(mcell.b_ph), grid=tuple(grid),
        box=hamiltonian.box, ndim=ndim, kpoints=np.asarray(kpts), eigenvalues=eigenvalues,
        steps=steps, unsettled=unsettled, solver=tuple(solver), preconditioner=preconditioner,
        memory_estimate=estimate, timings=timings, chern=groups_chern,
        states=states if keep_states else None)
