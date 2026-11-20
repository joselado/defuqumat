"""The self-consistent field of route P, on the chemical cell's density, by the fold.

Lu, Chen, Wang, Dai, Fang, Gong, Wu and Xiang, *General first-principles approach to
crystals in finite magnetic fields*, PRL 135, 196401 (2025), arXiv:2505.06585 (route P;
the design's section 3, step 2, and section 12 item 8 of ``notes/ORBITAL-FIELD-NEXT.md``).
The operator is :class:`~defuqumat.magnetic.blochgrid.MagneticGridHamiltonian` with the
GIPAW-dressed projectors; the conventions are :mod:`defuqumat.magnetic.conventions`.

The density is chemical-periodic. A uniform field keeps the crystal's translations as
magnetic translations, and ``T(r a1)`` sends the magnetic ``kappa2`` to ``kappa2 - r p/q``.
So on a closed mesh ``(N1, N2[, N3])`` with ``N2 = q m`` the states at the ``q`` points
``kappa2 - r p/q`` are translates of one another, their densities are translates, and the
density of the whole mesh averaged over the ``q`` chemical copies (the fold) is the fold of
the ``m`` representatives ``kappa2 = j/N2, j < m``, each with weight ``q/(N1 N2 N3)`` (times
two for the spin, as the zero-field code). Only the representatives are diagonalised: the
SCF of a magnetic cell of ``q`` chemical cells costs ``1/q`` of the full mesh. Measured
against the closed mesh solved explicitly in ``plans/orbital-field/route-p.md``.

The density of a state is ``|u(r)|^2`` on the potential's (dense) grid, from the same
interpolation the two-grid local term uses (QE's ``sum_band`` order: the wavefunction on
the dense grid, then squared), which is exact for a band-limited state; the design's
squaring on the box and Fourier padding aliases the product and is not used. The fold is
the mean of the ``q`` blocks along ``xi1``. The chemical cell's
:meth:`~defuqumat.scf.driver.Calculation.potential` makes the Hartree and
exchange-correlation potential of it, the project's Anderson mixer (``scf/mixing``, with
``rho_ddot`` as its inner product) mixes it, and ``dr2`` is
:func:`~defuqumat.scf.potential.scf_accuracy`, the SCF's own test against ``conv_thr``.

The energy per chemical cell is the zero-field code's sum with the band term of the
magnetic cell: ``eband = (1/q) sum_k w_k f eps`` (``w`` summing to two over the
representatives), ``deband = -int rho_out v_Hxc[rho_in]`` over the chemical cell, the
Hartree and exchange-correlation energies at the output density, the Ewald energy (the
field does not reach the ions) and the smearing term over ``q``. The occupations are the
system's scheme over the magnetic levels with ``q nelec`` electrons: smearing by the
system's ``degauss`` and kind, fixed only across a gap, which is checked.

The magnetizability (:func:`run_magnetizability`) is Lu's finite difference: the energy at
``+1``, ``0`` and ``-1`` flux quantum per magnetic cell, all three on the same box (``p = 0``
is the magnetic operator at zero field, so the basis is the same in the three), ``chi =
-(E+ - 2 E0 + E-)/dB^2``, in cgs per mole (:func:`molar_magnetizability`).

Refused by name: the regimes of :func:`~defuqumat.magnetic.conventions.refuse_regime`; a mesh
whose ``N2`` is not a multiple of ``q``; occupations other than ``'smearing'`` and
``'fixed'``; fixed occupations across levels that cross; a band set whose top band still
holds charge; a memory estimate above the limit; a Davidson solve that does not settle in
``max_davidson`` steps.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.magnetic.blochgrid import (
    MagneticGridHamiltonian, estimate_bytes, magnetic_cell, refuse_memory, start_block,
    wavefunction_box,
)
from defuqumat.magnetic.conventions import refuse_regime
from defuqumat.units import tesla_from_flux

__all__ = ["representatives", "fold_to_chemical", "band_density", "magnetic_occupations",
           "MagneticSCF", "run_magnetic_scf", "fold_check", "molar_magnetizability",
           "Magnetizability", "run_magnetizability", "RY_ERG", "AVOGADRO"]

#: One Rydberg in erg (``RYDBERG_SI`` x 1e7): CODATA 2018, 2.1798723611035e-18 J.
RY_ERG = 2.1798723611035e-11
#: Avogadro's number (exact in SI 2019).
AVOGADRO = 6.02214076e23


def representatives(kmesh, q: int):
    """``(kpoints (nrep, 3), weights (nrep,))``: the representatives of a closed mesh at ``q``.

    ``kmesh`` ``(N1, N2)`` or ``(N1, N2, N3)``; ``kappa = (i/N1, j/N2, l/N3)`` with ``j <
    N2/q``, the last index fastest; each weight is ``2 q/(N1 N2 N3)``, so they sum to two.
    """
    kmesh = tuple(int(n) for n in kmesh)
    if len(kmesh) == 2:
        kmesh = kmesh + (1,)
    n1, n2, n3 = kmesh
    if n2 % int(q):
        raise ValueError(f"the fold needs N2 a multiple of q: N2 = {n2}, q = {q}")
    m = n2 // int(q)
    axes = [np.arange(n1) / n1, np.arange(m) / n2, np.arange(n3) / n3]
    kpts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return kpts, np.full(len(kpts), 2.0 * int(q) / (n1 * n2 * n3))


def fold_to_chemical(field, q: int):
    """The mean of the ``q`` chemical-cell blocks of a magnetic-cell field along axis 0."""
    field = jnp.asarray(field)
    n1 = field.shape[0] // int(q)
    return jnp.mean(field.reshape((int(q), n1) + field.shape[1:]), axis=0)


@jax.jit
def band_density(hamiltonian: MagneticGridHamiltonian, vectors, occupations, ik):
    """``sum_n f_n |u_n(r)|^2`` on the magnetic fine grid, electrons per bohr^3.

    ``vectors`` ``(nb, ndim)`` box states normalised to one over the magnetic cell,
    ``occupations`` ``(nb,)`` the weights ``w_k f_n``; one band at a time through the
    interpolation of :meth:`~defuqumat.magnetic.blochgrid.MagneticGridHamiltonian.
    interpolate`, whose ``x = c u(r_l)`` gives ``|psi|^2 = |x|^2 / (c^2 dV_box)``.
    """
    kappa = hamiltonian.kpoints[ik]
    real = hamiltonian.precision.real
    shape = hamiltonian.potential_fine.shape
    c = hamiltonian.grid_ratio
    # the metric is B B^T with B the magnetic cell's scaled reciprocal rows: det = 1/Omega^2
    volume = 1.0 / jnp.sqrt(jnp.abs(jnp.linalg.det(hamiltonian.metric)))

    def body(rho, item):
        u, f = item
        x = hamiltonian.interpolate(u.reshape((1,) + hamiltonian.box), kappa)[0]
        return rho + f * jnp.real(x * jnp.conj(x)), None

    rho, _ = jax.lax.scan(body, jnp.zeros(shape, dtype=real),
                          (vectors.astype(hamiltonian.dtype), occupations.astype(real)))
    dv = volume / hamiltonian.ndim
    return rho / (c * c * dv)


def magnetic_occupations(eigenvalues, weights, nelec: float, system):
    """``(wg (nk, nbnd), fermi, minus_ts)`` over the magnetic levels by the system's scheme.

    Two electrons per state (the weights sum to two), ``nelec`` the magnetic cell's count.
    ``'smearing'``: the system's ``degauss`` and kind (:func:`~defuqumat.scf.occupations.
    smeared_occupations`), refused when the top band holds more than 1e-8 of a state.
    ``'fixed'``: the lowest ``nelec/2`` at every k, refused unless the highest occupied
    level is below the lowest empty one over the whole mesh.
    """
    from defuqumat.scf.occupations import smeared_occupations, smearing_entropy

    e = np.asarray(eigenvalues, dtype=float)
    w = np.asarray(weights, dtype=float)
    scheme = getattr(system, "occupations", "fixed")
    if scheme == "smearing":
        wg, ef = smeared_occupations(e[None], w, nelec, system.degauss, system.smearing)
        wg = np.asarray(wg)[0]
        minus_ts = float(smearing_entropy(e[None], w, ef, system.degauss, system.smearing))
        top = np.max(wg[:, -1] / w)
        if top > 1e-8:
            raise ValueError(f"the top band of {e.shape[1]} still holds {top:.2e} of a state "
                             "under the smearing: raise nbnd")
        return wg, float(ef), minus_ts
    if scheme == "fixed":
        nocc = int(round(nelec / 2.0))
        if abs(nocc * 2.0 - nelec) > 1e-8:
            raise ValueError(f"fixed occupations need an even electron count, got {nelec}")
        if e.shape[1] <= nocc:
            raise ValueError(f"fixed occupations need nbnd > {nocc} to see the gap, got {e.shape[1]}")
        homo, lumo = float(np.max(e[:, nocc - 1])), float(np.min(e[:, nocc]))
        if lumo <= homo:
            raise ValueError(
                f"fixed occupations across crossing magnetic levels: the highest occupied "
                f"level {homo:.6f} Ry is above the lowest empty one {lumo:.6f} Ry; give the "
                "system a smearing")
        wg = np.zeros_like(e)
        wg[:, :nocc] = w[:, None]
        return wg, 0.5 * (homo + lumo), 0.0
    raise NotImplementedError(
        f"occupations {scheme!r} on the magnetic levels: route P implements 'smearing' and "
        "'fixed' (the tetrahedra need the mesh's connectivity, which the fold does not keep)")


@dataclass(frozen=True)
class MagneticSCF:
    """The self-consistent state of route P at flux ``p/q``, per chemical cell.

    ``density`` and ``potential`` ``(1, f1, F2, F3)`` on the chemical cell's dense grid (the
    total local potential, as ``SCFResult.potential``); ``kpoints`` the representatives and
    ``weights`` theirs (summing to two); ``eigenvalues`` and ``occupations`` ``(nrep, nbnd)``
    over the magnetic cell; ``total_energy`` and ``energy_terms`` per chemical cell (Ry);
    ``history`` one dict per iteration (``dr2``, energy, Davidson steps, seconds).
    """

    flux: tuple
    field_tesla: float
    kmesh: tuple
    kpoints: np.ndarray
    weights: np.ndarray
    eigenvalues: np.ndarray
    occupations: np.ndarray
    fermi_energy: float
    density: np.ndarray
    potential: np.ndarray
    total_energy: float
    energy_terms: dict
    converged: bool
    iterations: int
    accuracy: float
    history: list
    memory_estimate: int
    timings: dict
    box: tuple
    ndim: int
    states: np.ndarray | None = field(default=None, repr=False)
    hamiltonian: object = field(default=None, repr=False)


def _initial_density(scf0):
    if scf0 is None:
        return None
    if isinstance(scf0, (str, bytes)) or hasattr(scf0, "__fspath__"):
        with np.load(scf0) as data:
            return np.asarray(data["density"])
    if hasattr(scf0, "density"):
        return np.asarray(scf0.density)
    return np.asarray(scf0["density"])


def _solve(hamiltonian, ik, nbnd, nsolve, start, thresholds, david, max_davidson,
           residual_threshold):
    from defuqumat.solvers.davidson import davidson_eigensolver_all

    if residual_threshold is not None:
        # on the returned bands only: the buffer is converged loosely by design, and a
        # residual bound on it left one band of 24 unsettled after 400 steps at 1/3
        residual_threshold = jnp.asarray(np.concatenate(
            [np.full(nbnd, float(residual_threshold)), np.full(nsolve - nbnd, 1e30)]))
    values, vectors, step, left = davidson_eigensolver_all(
        hamiltonian, nsolve, psi0=start[None], ethr=thresholds, david=david,
        max_iterations=max_davidson, indices=np.array([ik]), return_steps=True,
        residual_threshold=residual_threshold)
    if int(left[0]):
        raise RuntimeError(
            f"Davidson left {int(left[0])} of {nsolve} bands unsettled at k-point {ik} after "
            f"{int(step[0])} steps with the nonlocal term in: refused rather than used")
    return np.asarray(values[0]), vectors[0], int(step[0])


def run_magnetic_scf(system, pseudos, scf0=None, *, flux=(1, 1), kmesh=(2, 2), nbnd: int,
                     conv_thr: float = 1e-8, mixing_beta: float = 0.7, max_iterations: int = 60,
                     calculation=None, grid=None, david: int = 4, buffer: int = 4,
                     buffer_ethr: float = 1e-5, ethr: float = 1e-8, max_davidson: int = 200,
                     memory_limit: float | None = None, keep_states: bool = False,
                     residual_threshold: float | None = None, verbose: bool = True,
                     history: int = 8) -> MagneticSCF:
    """The SCF of ``pi^2 + V_loc + V_NL`` at flux ``p/q``, on the chemical cell's density.

    Args:
        system, pseudos: the chemical cell and its datasets (norm-conserving, collinear
            unpolarized; refused otherwise by name).
        scf0: the starting density (``SCFResult``, npz path or mapping with ``density``),
            usually the zero-field ground state; required (no atomic superposition here).
        flux: ``(p, q)``, ``p`` may be negative (the field along ``-a3``) or zero.
        kmesh: ``(N1, N2[, N3])`` over the magnetic zone, ``N2`` a multiple of ``q``.
        nbnd: magnetic bands returned (and occupied over); ``buffer`` more are solved at
            ``buffer_ethr`` and dropped.
        conv_thr: on ``dr2`` (Ry), as the zero-field SCF; ``mixing_beta`` Anderson's step,
            ``history`` its depth.
        calculation: the chemical cell's :class:`~defuqumat.scf.driver.Calculation` (built if
            ``None``) for ``v_of_rho``, ``vltot``, the Ewald energy and the dense grid.
        ethr: the first iteration's Davidson threshold (the start is a converged zero-field
            density); later ones follow QE's schedule (:func:`~defuqumat.scf.driver.next_ethr`).
        residual_threshold: a bound on each state's residual on top of ``ethr`` (for a check
            on the states, as :func:`fold_check`).
    """
    from defuqumat.scf.driver import Calculation, next_ethr
    from defuqumat.scf.mixing import get_mixer
    from defuqumat.scf.potential import rho_ddot_vector, scf_accuracy

    refuse_regime(system, pseudos, "The route P SCF")
    p, q = (int(x) for x in flux)
    mcell = magnetic_cell(system.cell, p, q)
    rho = _initial_density(scf0)
    if rho is None:
        raise ValueError("run_magnetic_scf starts from a density: pass scf0 (the zero-field state)")
    calculation = Calculation(system, pseudos) if calculation is None else calculation
    cell = system.cell
    dense = calculation.basis.dense
    if tuple(rho.shape[1:]) != tuple(dense.grid):
        raise ValueError(f"the starting density's grid {rho.shape[1:]} is not the chemical "
                         f"cell's dense grid {tuple(dense.grid)}")
    grid = wavefunction_box(cell, system.ecutwfc) if grid is None else tuple(grid)
    kpts, weights = representatives(kmesh, q)
    nrep = len(kpts)
    nsolve = nbnd + int(buffer)
    nelec = float(calculation.nelec) * q
    estimate = estimate_bytes(q, nsolve, david, grid=grid, kept=nrep * nsolve,
                              fine_grid=tuple(dense.grid))
    refuse_memory(estimate, memory_limit, f"the route P SCF at p/q = {p}/{q}, nbnd {nbnd}, "
                  f"{nrep} representatives")
    vltot = jnp.asarray(calculation.vltot)
    real = cell.precision.real
    timings = {}
    t0 = time.perf_counter()
    potential_in = calculation.potential(jnp.asarray(rho, dtype=real))
    hamiltonian = MagneticGridHamiltonian.build(
        cell, p, q, system.ecutwfc, kpts, vltot + potential_in.v_scf[0], grid=grid,
        structure=system.structure, pseudos=pseudos)
    timings["build_s"] = time.perf_counter() - t0
    mixer = get_mixer("anderson", beta=mixing_beta, history=history)
    thresholds_of = lambda e: np.concatenate(
        [np.full(nbnd, e), np.full(int(buffer), max(e, float(buffer_ethr)))])
    states = [None] * nrep
    records = []
    volume = float(cell.volume)
    converged, accuracy = False, np.inf
    terms = {}
    t_loop = time.perf_counter()
    for iteration in range(1, int(max_iterations) + 1):
        t1 = time.perf_counter()
        eigenvalues = np.zeros((nrep, nsolve))
        steps = []
        for ik in range(nrep):
            start = start_block(hamiltonian, ik, nsolve) if states[ik] is None else states[ik]
            values, vectors, step = _solve(hamiltonian, ik, nbnd, nsolve, start,
                                           thresholds_of(ethr), david, max_davidson,
                                           residual_threshold)
            eigenvalues[ik], states[ik] = values, vectors
            steps.append(step)
        wg, ef, minus_ts = magnetic_occupations(eigenvalues[:, :nbnd], weights, nelec, system)
        rho_mag = None
        for ik in range(nrep):
            part = band_density(hamiltonian, states[ik][:nbnd], jnp.asarray(wg[ik]), ik)
            rho_mag = part if rho_mag is None else rho_mag + part
        rho_out = np.asarray(fold_to_chemical(rho_mag, q))[None]
        residual = jnp.asarray(rho_out - rho)
        accuracy = float(scf_accuracy(residual, dense, cell))
        eband = float(np.sum(wg * eigenvalues[:, :nbnd])) / q
        deband = -volume / rho_out[0].size * float(np.sum(rho_out * np.asarray(potential_in.v_scf)))
        converged = accuracy < conv_thr
        potential_used = potential_in
        if converged:
            potential_in = calculation.potential(jnp.asarray(rho_out, dtype=real))
        terms = {"one-electron": eband + deband, "hartree": float(potential_in.ehart),
                 "xc": float(potential_in.etxc), "ewald": float(calculation.ewald),
                 "smearing": minus_ts / q, "eband": eband, "deband": deband}
        total = sum(v for k, v in terms.items() if k not in ("eband", "deband"))
        records.append({"iteration": iteration, "dr2": accuracy, "total_energy": total,
                        "fermi_energy": ef, "ethr": ethr, "davidson_steps": steps,
                        "seconds": time.perf_counter() - t1,
                        "charge": float(np.sum(rho_out) * volume / rho_out[0].size)})
        if verbose:
            print(f"  route P SCF {p}/{q} iteration {iteration:3d}: dr2 {accuracy:.3e} "
                  f"E {total:.10f} Ry, ef {ef:.6f}, ethr {ethr:.1e}, Davidson {steps}, "
                  f"{records[-1]['seconds']:.1f} s", flush=True)
        if converged:
            final_potential = potential_used
            break
        fit = np.asarray(rho_ddot_vector(residual, dense, cell))
        rho = mixer.mix(rho, rho_out, fit=fit)
        potential_in = calculation.potential(jnp.asarray(rho, dtype=real))
        hamiltonian = hamiltonian.with_potential(vltot + potential_in.v_scf[0])
        ethr = next_ethr(ethr, accuracy, float(calculation.nelec), iteration + 1)
        final_potential = potential_in
    timings["scf_s"] = time.perf_counter() - t_loop
    if not converged:
        # the total is then not variational: Hartree and xc at the mixed input
        # density, the band energy from the output states
        warnings.warn(
            f"the route P SCF at flux {p}/{q} did not converge in {len(records)} "
            f"iterations (dr2 {accuracy:.3e} against conv_thr {conv_thr:.1e}); its "
            "total_energy is not the energy of a self-consistent state",
            RuntimeWarning, stacklevel=2)
    field_tesla = 0.0 if p == 0 else float(tesla_from_flux(abs(p), q, float(mcell.area))) * np.sign(p)
    return MagneticSCF(
        flux=(p, q), field_tesla=field_tesla, kmesh=tuple(kmesh), kpoints=kpts, weights=weights,
        eigenvalues=eigenvalues[:, :nbnd], occupations=wg, fermi_energy=ef, density=rho_out,
        potential=np.asarray(vltot + final_potential.v_scf[0])[None],
        total_energy=total, energy_terms=terms, converged=converged, iterations=len(records),
        accuracy=accuracy, history=records, memory_estimate=estimate, timings=timings,
        box=hamiltonian.box, ndim=hamiltonian.ndim,
        states=np.stack([np.asarray(s[:nbnd]) for s in states]) if keep_states else None,
        hamiltonian=hamiltonian)


def fold_check(result: MagneticSCF, system, pseudos, *, nbnd: int | None = None,
               residual_threshold: float = 1e-9, ethr: float = 1e-12, david: int = 4,
               buffer: int = 4, max_davidson: int = 400):
    """The folded density of the representatives against that of the whole closed mesh.

    At the potential the SCF's last diagonalisation used (``result.hamiltonian``), solve the
    representatives and then every ``kappa2`` of the mesh explicitly, both to ``ethr`` with a
    residual bound, occupy each set by the system's scheme (the full mesh with weights
    ``2/(N1 N2 N3)``), fold both, and return ``(max |rho_rep - rho_full| / max rho_full,
    max |eps_rep sorted q times - eps_full sorted|, seconds)``.
    """
    from defuqumat.workflows.magnetic_bands import magnetic_mesh

    h = result.hamiltonian
    p, q = result.flux
    nbnd = result.eigenvalues.shape[1] if nbnd is None else int(nbnd)
    nsolve = nbnd + int(buffer)
    nelec = float(np.sum(result.occupations))
    t0 = time.perf_counter()
    full = magnetic_mesh(result.kmesh)
    sets = {}
    for label, kpts, weights in (("rep", result.kpoints, result.weights),
                                 ("full", full, np.full(len(full), 2.0 / len(full)))):
        hk = eqx_replace_kpoints(h, kpts)
        thresholds = np.concatenate([np.full(nbnd, ethr), np.full(int(buffer), 1e-5)])
        values, vectors = [], []
        for ik in range(len(kpts)):
            v, s, _ = _solve(hk, ik, nbnd, nsolve, start_block(hk, ik, nsolve), thresholds,
                             david, max_davidson, residual_threshold)
            values.append(v[:nbnd])
            vectors.append(s[:nbnd])
        values = np.asarray(values)
        wg, _, _ = magnetic_occupations(values, weights, nelec, system)
        rho = None
        for ik in range(len(kpts)):
            part = band_density(hk, vectors[ik], jnp.asarray(wg[ik]), ik)
            rho = part if rho is None else rho + part
        sets[label] = (np.asarray(fold_to_chemical(rho, q)), values)
    rho_rep, e_rep = sets["rep"]
    rho_full, e_full = sets["full"]
    density = float(np.max(np.abs(rho_rep - rho_full)) / np.max(np.abs(rho_full)))
    levels = float(np.max(np.abs(np.sort(np.tile(e_rep.reshape(-1), q)) - np.sort(e_full.reshape(-1)))))
    return density, levels, time.perf_counter() - t0


def eqx_replace_kpoints(hamiltonian: MagneticGridHamiltonian, kpoints) -> MagneticGridHamiltonian:
    """The same operator at another k-list ``(nk, 3)`` magnetic crystal coordinates."""
    import equinox as eqx

    kpoints = jnp.asarray(np.asarray(kpoints, dtype=float), dtype=hamiltonian.precision.real)
    return eqx.tree_at(lambda h: h.kpoints, hamiltonian, kpoints)


def molar_magnetizability(e_plus: float, e_zero: float, e_minus: float, b_tesla: float,
                          units: float = 1.0) -> float:
    """``chi = -(E+ - 2 E0 + E-)/dB^2`` in 1e-6 cm^3/mol, per mole of ``units``.

    ``E`` in Ry for the cell the field was applied to (``units`` formula units in it, e.g.
    the number of two-atom primitive cells), ``b_tesla`` the field step in tesla. cgs:
    energy in erg (1 Ry = :data:`RY_ERG`), field in gauss (1 T = 1e4 G), times
    :data:`AVOGADRO`.
    """
    second = (e_plus - 2.0 * e_zero + e_minus) * RY_ERG
    gauss = b_tesla * 1.0e4
    return -second / gauss ** 2 * AVOGADRO / units * 1.0e6


@dataclass(frozen=True)
class Magnetizability:
    """Lu's finite-difference magnetizability: the three SCFs and ``chi`` (1e-6 cm^3/mol)."""

    field_tesla: float
    energies: dict
    chi_per_unit: float
    chi_per_atom: float
    units: float
    asymmetry: float | None
    results: dict = field(repr=False, default_factory=dict)


def run_magnetizability(system, pseudos, scf0, *, kmesh, nbnd: int, conv_thr: float = 1e-8,
                        atoms_per_unit: int = 2, time_reversal_check: bool = True,
                        **options) -> Magnetizability:
    """``chi`` from the route P energies at ``-1``, ``0`` and ``+1`` flux quantum per cell (``q = 1``).

    All three on the same magnetic box. ``atoms_per_unit`` sets the formula unit the molar
    value is per (two for diamond's primitive cell, Lu's "per C2"); the per-atom value is
    beside it. ``asymmetry`` is ``E(+) - E(-)``, zero by time reversal, which is the
    control of a defect that breaks ``H(-A) = H(A)^*``, such as the sign of the field in
    one of the two runs. A defect even in the flux (a wrong ``1/q``, the sign of
    ``deband``, a global sign of ``A``) cannot show there. On a box even along an axis the
    Nyquist slot breaks the relation as well: on two-atom silicon at 12 Ry, ``conv_thr =
    1e-10``, the default 8 x 8 x 8 box gives 2.1e-5 Ry and 9 x 9 x 9 gives 1.6e-12 (P177,
    audit 2026-11-15). ``time_reversal_check=False`` skips the
    ``-1`` run, takes ``E(-) = E(+)``, which time reversal makes exact in every regime the
    route admits, and leaves ``asymmetry`` at ``None``; it saves about 40 per cent of the
    call (7 of 18 SCF iterations on the diamond of ``PLAN.md`` P177). Every energy used
    must come from a converged SCF, or the call raises: an unconverged total is not
    variational, and the second difference divides its error by ``dB^2``.
    """
    fluxes = (0, 1, -1) if time_reversal_check else (0, 1)
    results = {}
    for p in fluxes:
        results[p] = run_magnetic_scf(system, pseudos, scf0, flux=(p, 1), kmesh=kmesh,
                                      nbnd=nbnd, conv_thr=conv_thr, **options)
        # At once, not after the three: the field runs on the large box are the costly
        # ones, and chi needs every energy (audit 2026-11-20 #29).
        if not results[p].converged:
            raise RuntimeError(
                f"the route P SCF did not converge at flux {[p]} (flux quanta per "
                "cell), so its total energy is not variational and chi, a second "
                "difference divided by dB^2, is not defined from it; raise "
                "max_iterations or lower mixing_beta")
    mcell = magnetic_cell(system.cell, 1, 1)
    b = float(tesla_from_flux(1, 1, float(mcell.area)))
    energies = {p: float(r.total_energy) for p, r in results.items()}
    asymmetry = energies[1] - energies[-1] if time_reversal_check else None
    if not time_reversal_check:
        energies[-1] = energies[1]
    natoms = len(system.structure.types)
    units = natoms / float(atoms_per_unit)
    chi_unit = molar_magnetizability(energies[1], energies[0], energies[-1], b, units)
    chi_atom = molar_magnetizability(energies[1], energies[0], energies[-1], b, natoms)
    return Magnetizability(field_tesla=b, energies=energies, chi_per_unit=chi_unit,
                           chi_per_atom=chi_atom, units=units,
                           asymmetry=asymmetry, results=results)
