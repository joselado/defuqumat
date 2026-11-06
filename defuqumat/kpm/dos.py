"""The density of states by the kernel polynomial method, with no bands computed.

The DOS of the crystal is ``rho(E) = sum_k w_k Tr delta(E - H(k))``, the trace
taken in the plane-wave basis at each k-point. Expanding ``delta(E - H)`` in
Chebyshev polynomials of ``H' = (H - centre) / half`` turns the trace into the
moments ``Tr T_n(H')``, and those come from a recursion on a few random vectors
(:mod:`defuqumat.kpm.moments`), so nothing is diagonalised and the cost does not
depend on how many bands the energy window holds. The spectrum is the whole
basis, so the DOS integrates to ``sum_k w_k npw_k`` over the interval, and to the
number of electrons at the Fermi level.

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
and the spinor (``npol = 2``). Refused by name: ultrasoft and PAW (the polynomial
must be in ``S^-1 H``), the half-sphere ``gamma_only`` storage, and a spin
spiral.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.kpm.bounds import EDGE_MARGIN, LANCZOS_STEPS, spectral_interval
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
    """Refuse what the moments of ``H`` alone do not describe."""
    if any(h.has_overlap for h in hamiltonians):
        raise NotImplementedError(
            "the KPM density of states is implemented for norm-conserving "
            "pseudopotentials only: with an ultrasoft or PAW dataset the states "
            "solve H x = e S x, so the polynomial must be in S^-1 H with moments in "
            "the S inner product, and the moments of H alone give another spectrum"
        )
    if any(getattr(h, "gamma_only", False) for h in hamiltonians):
        raise NotImplementedError(
            "the KPM density of states does not run on gamma_only storage: the half "
            "sphere's inner product is 2 Re - (G = 0), so a random vector on it is "
            "not a random vector of the basis. Use K_POINTS automatic 1 1 1 0 0 0"
        )
    if calculation.spiral:
        raise NotImplementedError(
            "the KPM density of states does not run on a spin spiral, whose two "
            "spinor components live on different spheres"
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
        fermi_energy: recorded on the result and used for the default window.
        ns, tau, becsum, field, field_scale: the rest of the converged state,
            as for :func:`~defuqumat.workflows.nscf.run_nscf`.
        calculation: the SCF's own, moved to ``kpoints`` if given.
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
    require_a_kpm_regime(calculation, hamiltonians)
    batch = resolve_k_batch(calculation.k_batch)

    centre, half, bottom = spectral_interval(hamiltonians, seed=seed, steps=lanczos_steps,
                                             margin=edge_margin, k_batch=batch)
    if emin is None or emax is None:
        low, high = default_window(bottom, fermi_energy, broadening, delta_e)
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

    weights = np.asarray(calculation.system.kpoints.weights)
    samples = kpm_moments(hamiltonians, weights, centre, half, n_moments,
                          n_random=n_random, seed=seed, trace=trace, k_batch=batch)
    energies = np.arange(emin, emax + 0.5 * delta_e, delta_e)
    dos, integrated, error = _assemble(samples, kernel, n_moments, centre, half,
                                       energies, trace)
    return KPMDensityOfStates(
        energies=energies, dos=dos, integrated=integrated, dos_error=error,
        samples=samples, centre=centre, half_width=half, kernel=kernel,
        n_moments=n_moments, broadening=broadening,
        reference_energy=float(reference_energy), trace=trace,
        n_random=n_random if trace == "stochastic" else 0, seed=seed,
        nspin=len(hamiltonians) if len(hamiltonians) == 2 else 1,
        nelec=float(calculation.nelec), fermi_energy=fermi_energy,
    )
