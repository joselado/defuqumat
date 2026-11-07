"""The density of states by the kernel polynomial method, with no bands computed.

The DOS of the crystal is ``rho(E) = sum_k w_k Tr delta(E - H(k))``, the trace
taken in the plane-wave basis at each k-point. Expanding ``delta(E - H)`` in
Chebyshev polynomials of ``H' = (H - centre) / half`` turns the trace into the
moments ``Tr T_n(H')``, and those come from a recursion on a few random vectors
(:mod:`defuqumat.kpm.moments`), so nothing is diagonalised and the cost does not
depend on how many bands the energy window holds. The spectrum is the whole
basis, so the DOS integrates to ``sum_k w_k npw_k`` over the interval, and to the
number of electrons at the Fermi level. That holds exactly, sample by sample, for
the exact trace and for the plain random trace, whose ``mu_0`` is ``n_live`` for
every random-phase vector; with the control variate it holds on average only,
since each sample's ``mu_0`` gains ``sum_k w_k (natw - n_live sum_j |<y_j|r>|^2)``
from its ``natw`` Ritz vectors (Si2 at 12 Ry on 2x2x2, four samples: 365.43,
365.42, 365.09 and 365.92 against 364.75).

What the moments cost is set by the basis and not by the physics: the interval
reaches up to roughly ``ecutwfc``, so the number of moments for a broadening
``delta`` is ``N ~ sqrt(2 ln 2) pi half sqrt(1 - x^2) / delta``, with ``x`` the
rescaled reference energy (the Jackson width narrows towards the edges, where
the bands are). ``plans/kpm.md`` has the cost argument and the crossover with
Davidson.

Memory: one k-point in flight holds three blocks of ``(n_random, ndim)`` states,
``ndim = npwx`` (``2 npwx`` for a spinor), plus the moments ``(nk, n_random, N)``
in real numbers; the ``"exact"`` trace holds ``(ndim, ndim)`` per k-point in flight
and is for small cells.

Regimes: norm-conserving, ``nspin = 1`` and ``2`` (one Hamiltonian per channel),
and the spinor (``npol = 2``); ultrasoft and PAW at ``nspin = 1`` and ``2``
through ``S^(-1/2) H S^(-1/2)`` (:mod:`defuqumat.kpm.overlap`, P171). Refused by
name: a spinor with an augmented dataset, the half-sphere ``gamma_only``
storage, and a spin spiral.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.kpm.bounds import EDGE_MARGIN, LANCZOS_STEPS, spectral_interval
from defuqumat.kpm.control import FILTER_CUT, FILTER_DEGREE
from defuqumat.kpm.guard import check_moments
from defuqumat.kpm.kernels import get_kernel, moment_count
from defuqumat.kpm.moments import chebyshev_moments, live_mask
from defuqumat.kpm.profile import chebyshev_profile

#: Energy step of the default grid, Ry.
DEFAULT_DELTA_E = 0.002
#: Default broadening, Ry (half width at half maximum at the reference energy).
DEFAULT_BROADENING = 0.01
#: Default number of random vectors per k-point.
DEFAULT_RANDOM = 8


@dataclass(frozen=True)
class KPMDensityOfStates:
    """A density of states from Chebyshev moments, in Ry and states per Ry per cell.

    ``dos`` and ``integrated`` are ``(nE,)``, or ``(2, nE)`` for ``nspin = 2``,
    with the spin degeneracy in the k weights as for
    :class:`~defuqumat.workflows.dos.DensityOfStates`. ``dos_error`` is the
    standard error of the mean over the random vectors (zero for an exact trace),
    the noise to read a difference against. ``samples`` holds the moments of
    each random vector, already summed over k with the weights, ``(nspin, S, N)``;
    the trace is their mean. Keeping them makes another kernel or a coarser
    broadening a host-side operation with no new recursion
    (:meth:`reconstructed`).
    """

    energies: np.ndarray
    dos: np.ndarray
    integrated: np.ndarray
    dos_error: np.ndarray
    samples: np.ndarray
    centre: float
    half_width: float
    kernel: str
    n_moments: int
    broadening: float
    reference_energy: float
    trace: str
    n_random: int
    seed: int
    nspin: int
    nelec: float
    fermi_energy: float | None = None
    #: ``(3, nE)``: ``sum_k w_k Tr[sigma_i delta(E - H_k)]`` for a spinor, the
    #: spin density of states along x, y and z (``spin_resolved=True``), else ``None``.
    spin_dos: np.ndarray | None = None

    @property
    def moments(self) -> np.ndarray:
        """``(nspin, N)``: the weighted trace moments ``sum_k w_k Tr T_n(H'(k))``."""
        return self.samples.mean(axis=1)

    def broadening_at(self, energies) -> np.ndarray:
        """The kernel's half width at half maximum at ``energies``, in Ry."""
        kernel = get_kernel(self.kernel)
        if kernel.width_at is None:
            raise ValueError(f"the {self.kernel!r} kernel has no width formula")
        x = (np.asarray(energies) - self.centre) / self.half_width
        return self.half_width * kernel.width_at(self.n_moments, x)

    def count_at(self, energy: float) -> float:
        """The number of states below ``energy``, summed over the channels."""
        _, count = _evaluate(self.moments, self.kernel, self.n_moments, self.centre,
                             self.half_width, np.array([energy]))
        return float(np.sum(count))

    def fermi_level(self, nelec: float | None = None, tol: float = 1e-8) -> float:
        """The energy where :meth:`count_at` reaches ``nelec`` (the run's own by default), by bisection.

        No band is used, so this is the Fermi level of a cell too large to
        diagonalise; in a gap any energy with the right count is a solution and
        the bisection returns one of them. Its error follows from
        :meth:`count_error_at` divided by the DOS there.
        """
        target = self.nelec if nelec is None else float(nelec)
        low, high = self.centre - self.half_width, self.centre + self.half_width
        if not self.count_at(low) <= target <= self.count_at(high):
            raise ValueError(f"{target} states are outside the interval's count")
        while high - low > tol:
            middle = 0.5 * (low + high)
            if self.count_at(middle) < target:
                low = middle
            else:
                high = middle
        return 0.5 * (low + high)

    def count_error_at(self, energy: float) -> float:
        """The standard error of :meth:`count_at` over the random vectors (0 if exact)."""
        if self.trace != "stochastic" or self.samples.shape[1] < 2:
            return 0.0
        _, each = _evaluate(self.samples.sum(axis=0), self.kernel, self.n_moments,
                            self.centre, self.half_width, np.array([energy]))
        return float(each[:, 0].std(ddof=1) / math.sqrt(self.samples.shape[1]))

    def reconstructed(self, energies=None, kernel: str | None = None,
                      n_moments: int | None = None) -> "KPMDensityOfStates":
        """The same moments through another kernel, or cut to fewer of them.

        Fewer moments is a coarser broadening; more than were computed is
        refused, since those moments do not exist.
        """
        kernel = kernel or self.kernel
        n_moments = n_moments or self.n_moments
        if n_moments > self.samples.shape[-1]:
            raise ValueError(
                f"{n_moments} moments asked for, {self.samples.shape[-1]} computed")
        energies = self.energies if energies is None else np.asarray(energies)
        dos, integrated, error = _assemble(self.samples[..., :n_moments], kernel,
                                           n_moments, self.centre, self.half_width,
                                           energies, self.trace)
        width = get_kernel(kernel).width_at
        x = (self.reference_energy - self.centre) / self.half_width
        broadening = (float(self.half_width * width(n_moments, x)) if width else math.nan)
        return replace(self, energies=energies, dos=dos, integrated=integrated,
                       dos_error=error, kernel=kernel, n_moments=n_moments,
                       broadening=broadening)


def _evaluate(moments, kernel, n_moments, centre, half, energies):
    factors = get_kernel(kernel).factors(n_moments)
    dos, count = chebyshev_profile(moments[..., :n_moments], factors, centre, half,
                                   energies)
    return np.asarray(dos), np.asarray(count)


def _assemble(samples, kernel, n_moments, centre, half, energies, trace):
    """``(dos, integrated, error)`` from ``(nspin, S, N)`` samples."""
    dos, integrated = _evaluate(samples.mean(axis=1), kernel, n_moments, centre, half,
                                energies)
    if trace == "stochastic" and samples.shape[1] > 1:
        each, _ = _evaluate(samples, kernel, n_moments, centre, half, energies)
        error = each.std(axis=1, ddof=1) / math.sqrt(samples.shape[1])
    else:
        error = np.zeros_like(dos)
    if samples.shape[0] == 1:
        return dos[0], integrated[0], error[0]
    return dos, integrated, error


def kpm_moments(hamiltonians, weights, centre: float, half: float, n_moments: int,
                n_random: int = DEFAULT_RANDOM, seed: int = 0, trace: str = "stochastic",
                k_batch=None) -> np.ndarray:
    """``(nspin, S, N)``: the k-weighted trace moments of each sample.

    ``S`` is ``n_random`` for a stochastic trace, each sample an unbiased
    estimate of ``sum_k w_k Tr T_n(H'(k))``, and 1 for the exact trace. The
    moments are checked against ``mu_0`` before they are returned
    (:func:`~defuqumat.kpm.guard.check_moments`).
    """
    if n_moments % 2:
        raise ValueError(f"n_moments must be even (the doubling trick), got {n_moments}")
    if trace not in ("stochastic", "exact"):
        raise ValueError(f"trace must be 'stochastic' or 'exact', got {trace!r}")
    n_steps = n_moments // 2 - 1
    weights = np.asarray(weights, dtype=float)
    key = jax.random.PRNGKey(seed)
    channels = []
    for channel, hamiltonian in enumerate(hamiltonians):
        raw = chebyshev_moments(hamiltonian, centre, half, jax.random.fold_in(key, channel),
                                n_steps, n_random, kind=trace, k_batch=k_batch)
        raw = np.asarray(raw)
        check_moments(raw, centre, half)
        live = np.asarray(jnp.sum(live_mask(hamiltonian), axis=-1), dtype=float)
        if trace == "stochastic":
            samples = np.einsum("k,krn->rn", weights * live, raw)
        else:
            samples = np.einsum("k,krn->n", weights, raw)[None]
        channels.append(samples)
    return np.stack(channels)


def require_a_kpm_regime(calculation, hamiltonians) -> None:
    """Refuse what the moments of ``H`` alone do not describe.

    The regimes are tested in the order a caller can act on: a spin spiral and
    the half sphere are refused whatever the dataset, so an augmented spiral is
    told about the spiral, not about the overlap the DOS already folds in.
    """
    if calculation.spiral:
        raise NotImplementedError(
            "the KPM density of states does not run on a spin spiral, whose two "
            "spinor components live on different spheres"
        )
    if any(getattr(h, "gamma_only", False) for h in hamiltonians):
        raise NotImplementedError(
            "the KPM density of states does not run on gamma_only storage: the half "
            "sphere's inner product is 2 Re - (G = 0), so a random vector on it is "
            "not a random vector of the basis. Use K_POINTS automatic 1 1 1 0 0 0"
        )
    if any(h.has_overlap for h in hamiltonians):
        raise NotImplementedError(
            "this KPM trace is implemented for norm-conserving pseudopotentials "
            "only: with an ultrasoft or PAW dataset the states solve H x = e S x, so "
            "the polynomial must be in S^-1 H, and the moments of H alone give another "
            "spectrum. The total and projected density of states (get_kpm_dos, "
            "get_kpm_pdos) run on S^(-1/2) H S^(-1/2); the local DOS, the unfolding "
            "and the conductivity need the S metric in their start vectors or their "
            "operator, which is not implemented"
        )


def default_window(bottom: float, fermi_energy, broadening: float, delta_e: float):
    """``(emin, emax)``: from below the lowest band to 1 Ry above the Fermi level.

    Without a Fermi level, 2 Ry above the lowest band.
    """
    emin = bottom - 10.0 * broadening
    emax = (fermi_energy + 1.0) if fermi_energy is not None else bottom + 2.0
    return emin, emax


def run_kpm_dos(
    system,
    pseudos,
    density,
    *,
    kpoints=None,
    broadening: float = DEFAULT_BROADENING,
    n_moments: int | None = None,
    n_random: int = DEFAULT_RANDOM,
    seed: int = 0,
    kernel: str = "jackson",
    trace: str = "stochastic",
    emin: float | None = None,
    emax: float | None = None,
    delta_e: float = DEFAULT_DELTA_E,
    reference_energy: float | None = None,
    lanczos_steps: int = LANCZOS_STEPS,
    edge_margin: float = EDGE_MARGIN,
    fermi_energy: float | None = None,
    spin_resolved: bool = False,
    control_variate: bool | str = "auto",
    filter_degree: int | None = None,
    filter_cut: float | None = None,
    k_batch="default",
    ns=None,
    tau=None,
    becsum: tuple = (),
    field=None,
    field_scale=None,
    calculation=None,
) -> KPMDensityOfStates:
    """The density of states of a converged density, by Chebyshev moments.

    Args:
        system, pseudos, density: the converged run (``SCFResult.density``).
        kpoints: the k-set of the trace, if not the run's own. The DOS is a
            scalar trace, so a symmetry-reduced set with its weights completes it.
        broadening: the kernel's half width at half maximum, Ry, at
            ``reference_energy``; it sets the number of moments.
        n_moments: the number of moments, overriding ``broadening``; even.
        n_random: random vectors per k-point and channel.
        seed: of the random vectors, recorded on the result.
        kernel: ``"jackson"`` (default), ``"lorentz"`` or ``"fejer"``.
        trace: ``"stochastic"``, or ``"exact"``, which runs one recursion per
            plane wave and is for small cells.
        emin, emax, delta_e: the energy grid, Ry. The default window runs from
            below the lowest band to 1 Ry above ``fermi_energy``.
        reference_energy: where ``broadening`` holds, Ry. Defaults to the
            middle of the window.
        lanczos_steps, edge_margin: how the interval is estimated
            (:func:`~defuqumat.kpm.bounds.spectral_interval`).
        control_variate: subtract from each random sample its projection on
            Ritz pairs of the low-pass-filtered atomic orbitals and add back
            its mean (:mod:`defuqumat.kpm.control`): unbiased, with much less
            noise in the valence band, for ``(filter_degree + 1) natw``
            applications of ``H`` and a dense ``natw`` Rayleigh-Ritz per k-point.
            ``"auto"`` (default) uses it for random vectors when the datasets
            carry orbitals (:func:`~defuqumat.kpm.control.resolve_variate`).
        filter_degree, filter_cut: the low-pass filter, degree 16 and a cut at
            ``fermi_energy + 1`` Ry (the top of the window without a Fermi level)
            when not given.
        spin_resolved: for a spinor, also ``Tr[sigma_i delta(E - H)]`` on the
            same random vectors, as ``spin_dos``; refused for another regime.
        fermi_energy: recorded on the result and used for the default window;
            the highest occupied level for an insulator, as
            :meth:`~defuqumat.calculator.Calculator.get_kpm_dos` passes it.
        ns, tau, becsum, field, field_scale: the rest of the converged state,
            as for :func:`~defuqumat.workflows.nscf.run_nscf`.
        calculation: the SCF's own, moved to ``kpoints`` if given.
    """
    setup = kpm_setup(system, pseudos, density, kpoints=kpoints, k_batch=k_batch, ns=ns,
                      tau=tau, becsum=becsum, field=field, field_scale=field_scale,
                      calculation=calculation, symmetrize=True)
    calculation, hamiltonians, batch = setup
    (centre, half, emin, emax, reference_energy, n_moments,
     broadening) = interval_and_moments(hamiltonians, batch, seed, lanczos_steps, edge_margin,
                                        emin, emax, reference_energy, fermi_energy,
                                        broadening, n_moments, kernel)

    from defuqumat.kpm.control import resolve_variate

    # before the moments, so that a refused variate costs no recursion
    control_variate = resolve_variate(control_variate, trace, calculation)
    weights = np.asarray(calculation.system.kpoints.weights)
    samples = kpm_moments(hamiltonians, weights, centre, half, n_moments,
                          n_random=n_random, seed=seed, trace=trace, k_batch=batch)
    if control_variate:
        samples = samples + variate_corrections(
            calculation, hamiltonians, weights, centre, half, n_moments, n_random, seed,
            trace, (fermi_energy if fermi_energy is not None else emax)
            + (FILTER_CUT if filter_cut is None else filter_cut),
            FILTER_DEGREE if filter_degree is None else filter_degree)
    energies = np.arange(emin, emax + 0.5 * delta_e, delta_e)
    dos, integrated, error = _assemble(samples, kernel, n_moments, centre, half,
                                       energies, trace)
    spin = None
    if spin_resolved:
        spin = spin_resolved_dos(hamiltonians, weights, centre, half, n_moments, n_random,
                                 seed, kernel, energies, batch)
    return KPMDensityOfStates(
        energies=energies, dos=dos, integrated=integrated, dos_error=error, spin_dos=spin,
        samples=samples, centre=centre, half_width=half, kernel=kernel,
        n_moments=n_moments, broadening=broadening,
        reference_energy=float(reference_energy), trace=trace,
        n_random=n_random if trace == "stochastic" else 0, seed=seed,
        nspin=len(hamiltonians) if len(hamiltonians) == 2 else 1,
        nelec=float(calculation.nelec), fermi_energy=fermi_energy,
    )


def kpm_setup(system, pseudos, density, *, kpoints=None, k_batch="default", ns=None, tau=None,
              becsum=(), field=None, field_scale=None, calculation=None, symmetrize=False):
    """``(calculation, hamiltonians, k_batch)`` at a fixed density, the KPM regimes checked.

    ``symmetrize`` replaces an augmented Hamiltonian by ``S^(-1/2) H S^(-1/2)``
    (:mod:`defuqumat.kpm.overlap`), which a trace of the spectrum alone can use;
    without it an ultrasoft or PAW dataset is refused.
    """
    import equinox as eqx

    from defuqumat.batching import resolve_k_batch
    from defuqumat.scf.driver import Calculation
    from defuqumat.system.kpoints import for_spin
    from defuqumat.workflows.nscf import fixed_density_hamiltonians, threaded_calculation

    calculation, system, kpoints, k_batch = threaded_calculation(
        calculation, system, kpoints, k_batch)
    if kpoints is not None:
        system = eqx.tree_at(lambda s: s.kpoints, system, for_spin(kpoints, system.nspin))
    if calculation is None:
        calculation = Calculation(system, pseudos, k_batch=k_batch)
    hamiltonians = fixed_density_hamiltonians(
        calculation, density, ns=ns, tau=tau, becsum=becsum, field=field,
        field_scale=field_scale)
    batch = resolve_k_batch(calculation.k_batch)
    if symmetrize and any(h.has_overlap for h in hamiltonians):
        from defuqumat.kpm.overlap import symmetrized

        if getattr(calculation, "gamma_only", False) or calculation.spiral:
            require_a_kpm_regime(calculation, hamiltonians)
        hamiltonians = symmetrized(hamiltonians, k_batch=batch)
    require_a_kpm_regime(calculation, hamiltonians)
    return calculation, hamiltonians, batch


def interval_and_moments(hamiltonians, batch, seed, lanczos_steps, edge_margin, emin, emax,
                         reference_energy, fermi_energy, broadening, n_moments, kernel):
    """The interval, the energy window and the moment count, from what the caller gave."""
    centre, half, bottom = spectral_interval(hamiltonians, seed=seed, steps=lanczos_steps,
                                             margin=edge_margin, k_batch=batch)
    if emin is None or emax is None:
        low, high = default_window(bottom, fermi_energy, broadening, None)
        emin = low if emin is None else emin
        emax = high if emax is None else emax
    if reference_energy is None:
        reference_energy = 0.5 * (emin + emax)
    x_reference = (reference_energy - centre) / half
    chosen = get_kernel(kernel)
    if n_moments is None:
        n_moments = moment_count(chosen, broadening, half, x_reference)
    elif chosen.width_at is not None:
        broadening = float(half * chosen.width_at(n_moments, x_reference))
    else:
        broadening = math.nan
    return centre, half, emin, emax, float(reference_energy), n_moments, broadening


def spin_resolved_dos(hamiltonians, weights, centre, half, n_moments, n_random, seed, kernel,
                      energies, batch):
    """``(3, nE)``: the spin density of states of a spinor, ``sum_k w_k Tr[sigma_i delta(E - H)]``."""
    from defuqumat.kpm.moments import spin_moments

    if len(hamiltonians) != 1 or not hasattr(hamiltonians[0], "_as_state"):
        raise ValueError("spin_resolved needs a spinor (noncollinear) run; a collinear "
                         "run's spin channels are already the two rows of dos")
    hamiltonian = hamiltonians[0]
    key = jax.random.fold_in(jax.random.PRNGKey(seed), 0)
    raw = np.asarray(spin_moments(hamiltonian, centre, half, key, n_moments, n_random,
                                  k_batch=batch))  # (nk, 3, R, N)
    live = np.asarray(jnp.sum(live_mask(hamiltonian), axis=-1), dtype=float)
    moments = np.einsum("k,kirn->in", np.asarray(weights) * live, raw) / n_random
    dos, _ = _evaluate(moments, kernel, n_moments, centre, half, energies)
    return dos


def variate_corrections(calculation, hamiltonians, weights, centre, half, n_moments, n_random,
                        seed, trace, cut, degree):
    """``(nspin, R, N)``: the control variate's correction of every random sample."""
    from defuqumat.kpm.control import control_corrections
    from defuqumat.projwfc.projections import atomic_orbitals

    if trace != "stochastic":
        raise ValueError("control_variate reduces the noise of random vectors; the exact "
                         "trace has none")
    orbitals = jnp.swapaxes(atomic_orbitals(calculation), 1, 2)  # (nk, natw, ndim)
    key = jax.random.PRNGKey(seed)
    return np.stack([
        control_corrections(hamiltonian, orbitals, jax.random.fold_in(key, channel), centre,
                            half, n_moments, n_random, weights, cut, degree)[0]
        for channel, hamiltonian in enumerate(hamiltonians)])
