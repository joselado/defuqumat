"""Deterministic KPM traces: the projected DOS and the local DOS at points.

These are traces of rank ``r``, ``sum_k w_k sum_a <v_a(k)|delta(E - H_k)|v_a(k)>``,
with one recursion per start vector and no random vectors, so they carry no
statistical error; what remains is the kernel's broadening.

* **The projected DOS.** The start vectors are the atomic orbitals of
  ``projwfc``, Loewdin-orthonormalised (:func:`~defuqumat.projwfc.projections.atomic_orbitals`),
  so ``<phi_a|delta(E - H)|phi_a> = sum_n |<phi_a|psi_n>|^2 delta(E - e_n)`` is
  exactly what ``run_pdos`` integrates, with every state of the basis in it, the
  empty ones up to the cutoff included. A single orbital's trace is not
  invariant under the point group (the orbitals of a shell rotate into each
  other, and atoms into their images), so on a symmetry-reduced wedge the trace
  runs on the whole Monkhorst-Pack grid of the run instead.
* **The local DOS at a point.** The start vector is ``delta(r - r0)`` in plane
  waves, ``<k+G|r0> = exp(-i (k+G) . r0) / sqrt(Omega)``, so
  ``<r0|delta(E - H)|r0> = sum_n |psi_n(r0)|^2 delta(E - e_n)``. Its norm is
  ``npw_k / Omega``, not one. This is the pseudo-LDOS, the smooth part, with no
  augmentation (norm-conserving only, as everything here). Integrated to the
  Fermi level it is the density at ``r0``, which is the check. A point does not
  self-average, so it is exact here rather than stochastic.

Memory: the start block ``(nk, r, ndim)`` complex is built whole (``r`` orbitals
or points, ``2 r`` for a spinor's points), plus the ``(nk, r, N)`` moments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.kpm.bounds import EDGE_MARGIN, LANCZOS_STEPS
from defuqumat.kpm.dos import (
    DEFAULT_BROADENING, DEFAULT_DELTA_E, _evaluate, interval_and_moments, kpm_setup,
)
from defuqumat.kpm.guard import check_moments
from defuqumat.kpm.kernels import get_kernel
from defuqumat.kpm.moments import chebyshev_moments, live_mask


@dataclass(frozen=True)
class KPMProjectedDOS:
    """``D_a(E)`` per atomic orbital from Chebyshev moments, states per Ry per cell.

    ``pdos`` is ``(nproj, nE)``, or ``(2, nproj, nE)`` for ``nspin = 2``; the
    channels are ``projwfc``'s (:class:`~defuqumat.projwfc.channels.AtomicChannel`).
    ``moments`` is ``(nspin, nproj, N)``, the k-weighted ``<phi_a|T_n(H')|phi_a>``.
    """

    energies: np.ndarray
    pdos: np.ndarray
    integrated: np.ndarray
    channels: tuple
    moments: np.ndarray
    centre: float
    half_width: float
    kernel: str
    n_moments: int
    broadening: float
    nspin: int
    projectors: str = "ortho-atomic"
    fermi_energy: float | None = None

    @property
    def pdos_by_spin(self) -> np.ndarray:
        return self.pdos if self.nspin == 2 else self.pdos[None]

    def select(self, atom=None, l=None, species=None, m=None, wfc=None) -> np.ndarray:
        """The channels matching every constraint, summed, as ``ProjectedDOS.select``."""
        from defuqumat.projwfc.channels import L_LABELS

        if isinstance(l, str):
            l = L_LABELS.index(l.lower())
        picked = [c.index for c in self.channels
                  if (atom is None or c.atom == atom) and (l is None or c.l == l)
                  and (species is None or c.species == species)
                  and (m is None or c.m == m) and (wfc is None or c.wfc == wfc)]
        chosen = self.pdos_by_spin[:, picked].sum(axis=1)
        return chosen if self.nspin == 2 else chosen[0]

    def charges_at(self, energy: float) -> np.ndarray:
        """``(nspin, nproj)``: the charge of each orbital below ``energy``, its Loewdin charge at the Fermi level."""
        _, count = _evaluate(self.moments, self.kernel, self.n_moments, self.centre,
                             self.half_width, np.array([energy]))
        return count[..., 0]


@dataclass(frozen=True)
class KPMLocalDOS:
    """``LDOS(r_j, E)`` at given points, states per Ry per bohr^3.

    ``ldos`` is ``(npoints, nE)``, or ``(2, npoints, nE)`` for ``nspin = 2``;
    ``points`` in crystal coordinates. ``moments`` is ``(nspin, npoints, N)``.
    """

    energies: np.ndarray
    ldos: np.ndarray
    points: np.ndarray
    moments: np.ndarray
    centre: float
    half_width: float
    kernel: str
    n_moments: int
    broadening: float
    nspin: int
    fermi_energy: float | None = None

    def density_at(self, energy: float) -> np.ndarray:
        """``(nspin, npoints)``: the charge density below ``energy`` at each point, electrons per bohr^3."""
        _, count = _evaluate(self.moments, self.kernel, self.n_moments, self.centre,
                             self.half_width, np.array([energy]))
        return count[..., 0]


def whole_grid_calculation(calculation):
    """``calculation`` on the whole Monkhorst-Pack grid of its run, if its k-set is a wedge."""
    from defuqumat.system.kpoints import KPoints, for_spin

    kpoints = calculation.system.kpoints
    if not kpoints.reduced:
        return calculation
    if kpoints.grid is None:
        raise ValueError(
            "a per-orbital or per-point trace on a symmetry-reduced k-set needs the "
            "whole grid, and this k-set does not say which grid it came from: pass "
            "the whole grid as kpoints")
    whole = KPoints.automatic(kpoints.grid, kpoints.shift or (0, 0, 0),
                              calculation.system.cell)
    return calculation.at_kpoints(for_spin(whole, calculation.system.nspin))


def _given_moments(hamiltonians, starts, centre, half, n_moments, weights, batch):
    """``(nspin, r, N)``: the k-weighted moments of the given start block, guarded."""
    key = jax.random.PRNGKey(0)  # unused by kind="given", a fixed argument
    out = []
    for hamiltonian in hamiltonians:
        raw = np.asarray(chebyshev_moments(hamiltonian, centre, half, key,
                                           n_moments // 2 - 1, starts.shape[1],
                                           kind="given", k_batch=batch, given=starts))
        check_moments(raw, centre, half)
        out.append(np.einsum("k,krn->rn", weights, raw))
    return np.stack(out)


def _common(system, pseudos, density, broadening, n_moments, kernel, emin, emax, delta_e,
            reference_energy, lanczos_steps, edge_margin, fermi_energy, k_batch, state,
            calculation, whole_grid):
    from defuqumat.workflows.nscf import threaded_calculation

    if n_moments is not None and n_moments % 2:
        raise ValueError(f"n_moments must be even, got {n_moments}")
    calculation, system, kpoints, k_batch = threaded_calculation(calculation, system, None,
                                                                 k_batch)
    if calculation is None:
        from defuqumat.scf.driver import Calculation

        calculation = Calculation(system, pseudos, k_batch=k_batch)
    if whole_grid:
        calculation = whole_grid_calculation(calculation)
    calculation, hamiltonians, batch = kpm_setup(
        system, pseudos, density, k_batch=k_batch, calculation=calculation, **state)
    interval = interval_and_moments(hamiltonians, batch, 0, lanczos_steps, edge_margin, emin,
                                    emax, reference_energy, fermi_energy, broadening,
                                    n_moments, kernel)
    return calculation, hamiltonians, batch, interval


def run_kpm_pdos(system, pseudos, density, *, projectors: str = "ortho-atomic",
                 broadening: float = DEFAULT_BROADENING, n_moments: int | None = None,
                 kernel: str = "jackson", emin=None, emax=None,
                 delta_e: float = DEFAULT_DELTA_E, reference_energy=None,
                 lanczos_steps: int = LANCZOS_STEPS, edge_margin: float = EDGE_MARGIN,
                 fermi_energy=None, k_batch="default", ns=None, tau=None, becsum=(),
                 field=None, field_scale=None, calculation=None) -> KPMProjectedDOS:
    """The projected density of states on the atomic orbitals, by Chebyshev moments.

    The arguments are :func:`~defuqumat.kpm.dos.run_kpm_dos`' (no random vectors
    here), plus ``projectors``, ``projwfc``'s orbital set (``"ortho-atomic"``,
    Loewdin's, by default). A wedge is replaced by the whole grid of the run.
    """
    from defuqumat.projwfc.projections import atomic_orbitals, calculation_channels

    state = dict(ns=ns, tau=tau, becsum=becsum, field=field, field_scale=field_scale)
    calculation, hamiltonians, batch, interval = _common(
        system, pseudos, density, broadening, n_moments, kernel, emin, emax, delta_e,
        reference_energy, lanczos_steps, edge_margin, fermi_energy, k_batch, state,
        calculation, whole_grid=True)
    centre, half, emin, emax, reference_energy, n_moments, broadening = interval
    channels = calculation_channels(calculation)
    if not channels:
        raise ValueError("none of the pseudopotentials carries an atomic orbital to project on")
    orbitals = jnp.swapaxes(atomic_orbitals(calculation, projectors), 1, 2)  # (nk, r, ndim)
    weights = np.asarray(calculation.system.kpoints.weights)
    moments = _given_moments(hamiltonians, orbitals, centre, half, n_moments, weights, batch)
    energies = np.arange(emin, emax + 0.5 * delta_e, delta_e)
    pdos, integrated = _evaluate(moments, kernel, n_moments, centre, half, energies)
    nspin = len(hamiltonians) if len(hamiltonians) == 2 else 1
    if nspin == 1:
        pdos, integrated = pdos[0], integrated[0]
    return KPMProjectedDOS(energies=energies, pdos=pdos, integrated=integrated,
                           channels=channels, moments=moments, centre=centre,
                           half_width=half, kernel=kernel, n_moments=n_moments,
                           broadening=broadening, nspin=nspin, projectors=projectors,
                           fermi_energy=fermi_energy)


def point_vectors(hamiltonian, kcrystal, points, volume):
    """``(nk, np, ndim)``: ``exp(-i (k+G) . r) / sqrt(Omega)`` on each k's sphere.

    ``points`` in crystal coordinates. The Miller index of each plane wave is
    the box coordinate of its ``fft_index`` taken in ``(-n/2, n/2]``, which is
    the vector itself as long as the box holds the sphere. A spinor gets two
    vectors per point, up then down, ``(nk, 2 np, 2 npwx)``.
    """
    n1, n2, n3 = hamiltonian.grid
    index = np.asarray(hamiltonian.fft_index)
    a, b, c = index // (n2 * n3), (index // n3) % n2, index % n3
    miller = np.stack([np.where(a > n1 // 2, a - n1, a), np.where(b > n2 // 2, b - n2, b),
                       np.where(c > n3 // 2, c - n3, c)], axis=-1)  # (nk, npwx, 3)
    frac = np.asarray(points, dtype=float)
    phase = 2.0 * math.pi * np.einsum("kgi,pi->kpg", miller + np.asarray(kcrystal)[:, None, :],
                                      frac)
    vectors = np.exp(-1j * phase) / math.sqrt(volume)
    mask = np.asarray(hamiltonian.mask)[:, None, :]
    vectors = np.where(mask, vectors, 0.0)
    if hasattr(hamiltonian, "_as_state"):
        zero = np.zeros_like(vectors)
        vectors = np.concatenate([np.concatenate([vectors, zero], axis=-1),
                                  np.concatenate([zero, vectors], axis=-1)], axis=1)
    return jnp.asarray(vectors, dtype=hamiltonian.dtype)


def run_kpm_ldos(system, pseudos, density, points, *,
                 broadening: float = DEFAULT_BROADENING, n_moments: int | None = None,
                 kernel: str = "jackson", emin=None, emax=None,
                 delta_e: float = DEFAULT_DELTA_E, reference_energy=None,
                 lanczos_steps: int = LANCZOS_STEPS, edge_margin: float = EDGE_MARGIN,
                 fermi_energy=None, k_batch="default", ns=None, tau=None, becsum=(),
                 field=None, field_scale=None, calculation=None) -> KPMLocalDOS:
    """The local density of states at ``points`` (crystal coordinates, ``(np, 3)``), by Chebyshev moments.

    The other arguments are :func:`run_kpm_pdos`'. A wedge is replaced by the
    whole grid of the run: the LDOS at a point is not invariant under the group.
    """
    state = dict(ns=ns, tau=tau, becsum=becsum, field=field, field_scale=field_scale)
    calculation, hamiltonians, batch, interval = _common(
        system, pseudos, density, broadening, n_moments, kernel, emin, emax, delta_e,
        reference_energy, lanczos_steps, edge_margin, fermi_energy, k_batch, state,
        calculation, whole_grid=True)
    centre, half, emin, emax, reference_energy, n_moments, broadening = interval
    points = np.atleast_2d(np.asarray(points, dtype=float))
    cell = calculation.system.cell
    kcrystal = np.asarray(calculation.system.kpoints.crystal(cell))
    vectors = point_vectors(hamiltonians[0], kcrystal, points, float(cell.volume))
    weights = np.asarray(calculation.system.kpoints.weights)
    moments = _given_moments(hamiltonians, vectors, centre, half, n_moments, weights, batch)
    if vectors.shape[1] == 2 * points.shape[0]:  # a spinor: up and down add
        moments = moments[:, :points.shape[0]] + moments[:, points.shape[0]:]
    energies = np.arange(emin, emax + 0.5 * delta_e, delta_e)
    ldos, _ = _evaluate(moments, kernel, n_moments, centre, half, energies)
    nspin = len(hamiltonians) if len(hamiltonians) == 2 else 1
    return KPMLocalDOS(energies=energies, ldos=ldos if nspin == 2 else ldos[0],
                       points=points, moments=moments, centre=centre, half_width=half,
                       kernel=kernel, n_moments=n_moments, broadening=broadening,
                       nspin=nspin, fermi_energy=fermi_energy)


@dataclass(frozen=True)
class KPMUnfoldedSpectrum:
    """``A(k_p, E)``: the spectral function of a supercell unfolded to primitive k-points.

    ``spectral`` is ``(nk, nE)``, or ``(2, nk, nE)`` for ``nspin = 2``, in states
    per Ry per spin channel at each primitive ``k_p`` (crystal coordinates of the
    primitive reciprocal lattice, ``kpoints``). In a crystal with the primitive
    period it is ``sum_n delta(E - e_n(k_p))`` broadened by the kernel; a defect
    or disorder broadens and shifts its weight. ``moments`` is ``(nspin, nk, N)``
    (the mean over the samples for a stochastic trace).
    """

    energies: np.ndarray
    spectral: np.ndarray
    kpoints: np.ndarray
    moments: np.ndarray
    centre: float
    half_width: float
    kernel: str
    n_moments: int
    broadening: float
    nspin: int
    trace: str
    subset_size: np.ndarray
    fermi_energy: float | None = None


def supercell_matrix(supercell_at, primitive_at) -> np.ndarray:
    """The integer ``M`` with ``A_super = M A_prim`` (rows are lattice vectors), or an error."""
    m = np.asarray(supercell_at, dtype=float) @ np.linalg.inv(np.asarray(primitive_at, dtype=float))
    rounded = np.rint(m)
    if np.max(np.abs(m - rounded)) > 1e-6:
        raise ValueError(
            "the cell is not a supercell of the primitive lattice given: A_super A_prim^-1 = "
            f"{np.round(m, 6).tolist()} is not an integer matrix")
    return rounded.astype(int)


def primitive_subset(hamiltonian, matrix) -> np.ndarray:
    """``(nk, ndim)``: which plane waves of each k's sphere are ``k_p + G_p``, ``G_p`` primitive.

    A supercell reciprocal vector with Miller indices ``m`` is a primitive one
    when ``M^-1 m`` is integer, since ``b_super_j . a_prim_i = 2 pi (M^-1)_ij``.
    The test is on Miller indices, never on cartesian lengths.
    """
    n1, n2, n3 = hamiltonian.grid
    index = np.asarray(hamiltonian.fft_index)
    a, b, c = index // (n2 * n3), (index // n3) % n2, index % n3
    miller = np.stack([np.where(a > n1 // 2, a - n1, a), np.where(b > n2 // 2, b - n2, b),
                       np.where(c > n3 // 2, c - n3, c)], axis=-1)
    prim = miller @ np.linalg.inv(matrix).T
    subset = np.all(np.abs(prim - np.rint(prim)) < 1e-8, axis=-1) & np.asarray(hamiltonian.mask)
    if hasattr(hamiltonian, "_as_state"):
        subset = np.concatenate([subset, subset], axis=-1)
    return subset


def run_kpm_unfolded(system, pseudos, density, primitive_lattice, kpoints, *,
                     trace: str = "exact", n_random: int = 8, seed: int = 0,
                     broadening: float = DEFAULT_BROADENING, n_moments: int | None = None,
                     kernel: str = "jackson", emin=None, emax=None,
                     delta_e: float = DEFAULT_DELTA_E, reference_energy=None,
                     lanczos_steps: int = LANCZOS_STEPS, edge_margin: float = EDGE_MARGIN,
                     fermi_energy=None, k_batch="default", ns=None, tau=None, becsum=(),
                     field=None, field_scale=None, calculation=None) -> KPMUnfoldedSpectrum:
    """The supercell's spectral function unfolded to primitive ``kpoints``, with no bands.

    ``primitive_lattice`` is ``(3, 3)``, the primitive lattice vectors as rows in
    bohr; the run's cell must be an integer supercell of it. ``kpoints`` is
    ``(nk, 3)`` in crystal coordinates of the primitive reciprocal lattice. The
    supercell Hamiltonian is taken at each ``k_p`` itself, and the trace runs over
    its plane waves ``k_p + G_p`` (``trace = "exact"``, one recursion each) or over
    random phases on them (``"stochastic"``, ``n_random`` vectors).
    """
    from defuqumat.system.kpoints import KPoints, for_spin
    from defuqumat.workflows.nscf import threaded_calculation

    if trace not in ("exact", "stochastic"):
        raise ValueError(f"trace must be 'exact' or 'stochastic', got {trace!r}")
    calculation, system, _, k_batch = threaded_calculation(calculation, system, None, k_batch)
    if calculation is None:
        from defuqumat.scf.driver import Calculation

        calculation = Calculation(system, pseudos, k_batch=k_batch)
    cell = calculation.system.cell
    matrix = supercell_matrix(np.asarray(cell.at), primitive_lattice)
    kpoints = np.atleast_2d(np.asarray(kpoints, dtype=float))
    primitive_bg = 2.0 * math.pi * np.linalg.inv(np.asarray(primitive_lattice, dtype=float)).T
    kcart = kpoints @ primitive_bg  # 1/bohr
    path = KPoints.from_cartesian(kcart / float(cell.tpiba), np.ones(len(kpoints)))
    calculation = calculation.at_kpoints(for_spin(path, calculation.system.nspin))
    state = dict(ns=ns, tau=tau, becsum=becsum, field=field, field_scale=field_scale)
    calculation, hamiltonians, batch = kpm_setup(system, pseudos, density, k_batch=k_batch,
                                                 calculation=calculation, **state)
    centre, half, emin, emax, reference_energy, n_moments, broadening = interval_and_moments(
        hamiltonians, batch, seed, lanczos_steps, edge_margin, emin, emax, reference_energy,
        fermi_energy, broadening, n_moments, kernel)
    subset = primitive_subset(hamiltonians[0], matrix)  # (nk, ndim)
    counts = subset.sum(axis=-1)
    nk, ndim = subset.shape
    if trace == "exact":
        width = int(counts.max())
        starts = np.zeros((nk, width, ndim))
        for ik in range(nk):
            columns = np.flatnonzero(subset[ik])
            starts[ik, np.arange(columns.size), columns] = 1.0
        scale = np.ones(nk)
    else:
        rng = np.random.default_rng(seed)
        angle = 2.0 * math.pi * rng.random((nk, n_random, ndim))
        starts = np.where(subset[:, None, :], np.exp(1j * angle), 0.0) / np.sqrt(
            counts)[:, None, None]
        scale = counts / n_random
    starts = jnp.asarray(starts, dtype=hamiltonians[0].dtype)
    key = jax.random.PRNGKey(0)
    moments = []
    for hamiltonian in hamiltonians:
        raw = np.asarray(chebyshev_moments(hamiltonian, centre, half, key, n_moments // 2 - 1,
                                           starts.shape[1], kind="given", k_batch=batch,
                                           given=starts))
        check_moments(raw, centre, half)
        moments.append(raw.sum(axis=1) * scale[:, None])
    moments = np.stack(moments)  # (nspin, nk, N)
    energies = np.arange(emin, emax + 0.5 * delta_e, delta_e)
    spectral, _ = _evaluate(moments, kernel, n_moments, centre, half, energies)
    nspin = len(hamiltonians) if len(hamiltonians) == 2 else 1
    return KPMUnfoldedSpectrum(
        energies=energies, spectral=spectral if nspin == 2 else spectral[0], kpoints=kpoints,
        moments=moments, centre=centre, half_width=half, kernel=kernel, n_moments=n_moments,
        broadening=broadening, nspin=nspin, trace=trace, subset_size=counts,
        fermi_energy=fermi_energy)
