"""The optical conductivity by two-index Chebyshev moments (P169).

The moments are checked against the dense route, which shares no machinery
with the recursion: every eigenpair of ``Hamiltonian.matrix`` and the velocity
matrix elements of those eigenvectors from
:meth:`~defuqumat.response.velocity.VelocityOperator.matrix_elements` (one
``jvp`` of the whole ``H(k)`` per direction, contracted inside its own k walk).
The conductivity is checked against the pair sum over the same eigenpairs,
and those elements against the band sum
:func:`~defuqumat.workflows.conductivity.run_conductivity` (nbnd = 80 of the 190
states, which the 0.6 Ry window does not reach past). Silicon at 12 Ry on a 2x2x2 mesh, as in
``test_kpm.py``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import defuqumat  # noqa: F401
from defuqumat import Calculator
from defuqumat.kpm import get_kernel
from defuqumat.kpm.bounds import spectral_interval
from defuqumat.kpm.conductivity import (
    conductivity_moments, run_kpm_conductivity, velocity_tangents,
)
from defuqumat.kpm.moments import live_mask
from defuqumat.response.velocity import VelocityOperator
from defuqumat.workflows.nscf import fixed_density_hamiltonians

pytestmark = pytest.mark.unit

SILICON = dict(
    lattice="fcc", a=5.398, pseudos={"Si": "Si.pz-vbc.UPF"},
    atoms=[("Si", (0.0, 0.0, 0.0)), ("Si", (0.25, 0.25, 0.25))],
    ecut=163.3, kmesh=(2, 2, 2),
)
AXES = np.eye(3)


@functools.lru_cache(maxsize=2)
def _silicon():
    from pathlib import Path

    pseudo = Path(__file__).parents[1] / "data" / "pseudo"
    calc = Calculator.from_keywords(pseudo_dir=pseudo, announce=False, conv_thr=1e-10,
                                    **SILICON)
    result = calc.get_scf()
    hamiltonians = fixed_density_hamiltonians(calc.calculation, result.density)
    return calc, result, hamiltonians


def _dense_moments(calc, result, hamiltonian, centre, half, n_moments):
    """``sum_k w_k Tr[v_a T_n v_b T_m]`` from every eigenpair and the velocity operator's elements."""
    calculation = calc.calculation
    mask = np.asarray(live_mask(hamiltonian))
    nk, ndim = mask.shape
    states = np.zeros((1, nk, ndim, ndim), dtype=complex)
    values = np.full((nk, ndim), np.nan)
    for ik in range(nk):
        live = mask[ik]
        matrix = np.asarray(hamiltonian.matrix(ik))[np.ix_(live, live)]
        e, u = np.linalg.eigh(matrix)
        values[ik, :live.sum()] = e
        states[0, ik, :live.sum(), live] = u
    v_scf = calculation.potential(jnp.asarray(result.density), 1.0, None).v_scf
    elements = np.asarray(VelocityOperator(calculation, v_scf).matrix_elements(
        jnp.asarray(states)))[:, 0]  # (3, nk, i, j) = <i|v_a|j>
    weights = np.asarray(calculation.system.kpoints.weights)
    n = np.arange(n_moments)
    total = np.zeros((3, 3, n_moments, n_moments), dtype=complex)
    for ik in range(nk):
        count = mask[ik].sum()
        x = (values[ik, :count] - centre) / half
        cheb = np.cos(n[:, None] * np.arccos(x)[None, :])  # (N, i)
        v = elements[:, ik, :count, :count]
        # Tr[v_a T_n v_b T_m] = sum_ij v_a[i, j] T_n(e_j) v_b[j, i] T_m(e_i)
        total += weights[ik] * np.einsum("aij,nj,bji,mi->abnm", v, cheb, v, cheb)
    return total


def test_the_exact_moments_are_the_dense_trace():
    """Every unit vector of the sphere against the eigenpairs and the velocity matrix, to round-off."""
    calc, result, hamiltonians = _silicon()
    centre, half, _ = spectral_interval(hamiltonians)
    n_moments = 12
    moving, tangents = velocity_tangents(calc.calculation, hamiltonians, AXES)
    weights = jnp.asarray(calc.calculation.system.kpoints.weights)
    mu, norms, _ = conductivity_moments(
        hamiltonians[0], moving[0], tuple(t[0] for t in tangents), centre, half,
        jax.random.PRNGKey(0), weights, n_moments, 1, kind="exact")
    dense = _dense_moments(calc, result, hamiltonians[0], centre, half, n_moments)
    scale = np.abs(dense).max()
    np.testing.assert_allclose(np.asarray(mu), dense, rtol=0, atol=1e-9 * scale)
    assert np.all(np.asarray(norms) <= 1.0 + 1e-9)


def _dense_states(calculation, density):
    """Every eigenpair of the dense ``H(k)`` and ``<i|v_a|j>`` from the velocity operator."""
    hamiltonian = fixed_density_hamiltonians(calculation, density)[0]
    mask = np.asarray(live_mask(hamiltonian))
    nk, ndim = mask.shape
    states = np.zeros((1, nk, ndim, ndim), dtype=complex)
    values = []
    for ik in range(nk):
        live = mask[ik]
        e, u = np.linalg.eigh(np.asarray(hamiltonian.matrix(ik))[np.ix_(live, live)])
        values.append(e)
        states[0, ik, :live.sum(), live] = u
    v_scf = calculation.potential(jnp.asarray(density), 1.0, None).v_scf
    elements = np.asarray(VelocityOperator(calculation, v_scf).matrix_elements(
        jnp.asarray(states)))[:, 0]
    return values, elements


def test_the_conductivity_is_the_dense_kubo_greenwood_sum():
    """``sigma'_xx(w)`` from the moments against the pair sum over every eigenpair.

    The reference is the same energy integral, ``(pi / (Omega w)) sum_k w_k
    sum_ij |v_ij|^2 int dE [f(E) - f(E + w)] D(E; e_i) D(E + w; e_j)``, with
    ``D`` the kernel's delta evaluated at the eigenvalues, so it shares the
    kernel and nothing else: not the recursion, not the moment contraction,
    not the grid arithmetic of ``E_i + w_j``. That the velocity elements are
    the band sum's own is :func:`test_the_dense_elements_are_the_band_sums`.
    """
    from defuqumat.kpm.traces import whole_grid_calculation

    calc, result, _ = _silicon()
    level = result.homo + 0.02
    n_moments, delta_e, window = 48, 0.01, 0.8
    kpm = run_kpm_conductivity(calc.system, calc.pseudos, result.density,
                               calculation=calc.calculation, fermi_energy=level,
                               directions=("x",), kernel="jackson", trace="exact",
                               n_moments=n_moments, window=window, delta_e=delta_e)
    calculation = whole_grid_calculation(calc.calculation)
    values, elements = _dense_states(calculation, result.density)
    weights = np.asarray(calculation.system.kpoints.weights)
    volume = float(calculation.system.cell.volume)
    factors = get_kernel("jackson").factors(n_moments)
    coefficients = factors * np.where(np.arange(n_moments) > 0, 2.0, 1.0)
    energies = kpm.emin + delta_e * np.arange(4000)
    energies = energies[energies < level + window + delta_e]
    x = (energies - kpm.centre) / kpm.half_width
    n = np.arange(n_moments)
    cheb_grid = np.cos(n[:, None] * np.arccos(x)[None, :])  # (N, nE)
    measure = np.pi * kpm.half_width * np.sqrt(1.0 - x**2)
    occupation = np.where(energies < level, 1.0, 0.0)
    w = kpm.frequencies[1:]
    shift = np.arange(1, len(kpm.frequencies))
    reference = np.zeros(len(w))
    for ik, e in enumerate(values):
        xe = (e - kpm.centre) / kpm.half_width
        cheb_e = np.cos(n[:, None] * np.arccos(xe)[None, :])  # (N, i)
        delta = (coefficients[:, None] * cheb_e).T @ cheb_grid / measure  # (i, nE)
        count = len(e)
        v2 = np.abs(elements[0, ik, :count, :count]) ** 2
        gamma = delta.T @ v2 @ delta  # (E, E') = sum_ij D(E; e_i) |v_ij|^2 D(E'; e_j)
        for jw, j in enumerate(shift):
            rows = np.arange(len(energies) - j)
            weight = occupation[rows] - occupation[rows + j]
            reference[jw] += weights[ik] * np.sum(weight * gamma[rows, rows + j])
    reference *= np.pi / volume * delta_e / w
    got = kpm.sigma[1:, 0, 0].real
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-8 * np.abs(reference).max())
    assert np.abs(kpm.sigma[1:, 0, 0].imag).max() < 1e-10 * np.abs(reference).max()


def test_the_dense_elements_are_the_band_sums():
    """The pair sum of the dense elements in the band sum's own form is ``run_conductivity``'s ``sigma``."""
    from defuqumat.kpm.traces import whole_grid_calculation

    calc, result, _ = _silicon()
    level = result.homo + 0.02
    eta = 0.08
    frequencies = np.linspace(0.05, 0.6, 12)
    band = calc.get_optical_conductivity(nbnd=80, broadening=eta, intraband=False,
                                         frequencies=frequencies, fermi_energy=level)
    calculation = whole_grid_calculation(calc.calculation)
    values, elements = _dense_states(calculation, result.density)
    weights = np.asarray(calculation.system.kpoints.weights)
    volume = float(calculation.system.cell.volume)

    def lorentzian(x):
        return eta / np.pi / (x * x + eta * eta)

    reference = np.zeros(len(frequencies))
    for ik, e in enumerate(values):
        count = len(e)
        occupied = e < level
        v2 = np.abs(elements[0, ik, :count, :count]) ** 2
        i, j = np.nonzero(occupied[:, None] & ~occupied[None, :])
        gap = e[j] - e[i]
        reference += weights[ik] * np.pi / volume * np.sum(
            v2[i, j] / gap * (lorentzian(frequencies[:, None] - gap)
                              + lorentzian(frequencies[:, None] + gap)), axis=1)
    np.testing.assert_allclose(band.sigma[:, 0, 0].real, reference, rtol=2e-4)


def test_a_second_call_compiles_nothing():
    """The moments are a module-level ``jit``; a second potential reuses the executable."""
    calc, result, hamiltonians = _silicon()
    centre, half, _ = spectral_interval(hamiltonians)
    moving, tangents = velocity_tangents(calc.calculation, hamiltonians, AXES[:1])
    tangents = (moving[0], tuple(t[0] for t in tangents))
    weights = jnp.asarray(calc.calculation.system.kpoints.weights)
    args = (centre, half, jax.random.PRNGKey(0), weights, 8, 2)
    conductivity_moments(hamiltonians[0], *tangents, *args)
    moved = fixed_density_hamiltonians(calc.calculation, 0.999 * result.density)
    before = conductivity_moments._cache_size()
    conductivity_moments(moved[0], *tangents, *args)
    assert conductivity_moments._cache_size() == before


def test_the_guard_sees_a_spectrum_outside_the_interval():
    """With half the interval, the Chebyshev vectors grow, and the norms the guard reads say so."""
    calc, _, hamiltonians = _silicon()
    centre, half, _ = spectral_interval(hamiltonians)
    moving, tangents = velocity_tangents(calc.calculation, hamiltonians, AXES[:1])
    weights = jnp.asarray(calc.calculation.system.kpoints.weights)
    args = (moving[0], tuple(t[0] for t in tangents))
    inside = np.asarray(conductivity_moments(hamiltonians[0], *args, centre, half,
                                             jax.random.PRNGKey(0), weights, 16, 2)[1])
    outside = np.asarray(conductivity_moments(hamiltonians[0], *args, centre, 0.5 * half,
                                              jax.random.PRNGKey(0), weights, 16, 2)[1])
    assert inside.max() <= 1.0 + 1e-9
    assert outside.max() > 10.0


def test_the_stochastic_trace_is_unbiased():
    """The mean over seeds of ``R = 2`` estimates against the exact moments, within its own error."""
    calc, _, hamiltonians = _silicon()
    centre, half, _ = spectral_interval(hamiltonians)
    moving, tangents = velocity_tangents(calc.calculation, hamiltonians, AXES[:1])
    tangents = (moving[0], tuple(t[0] for t in tangents))
    weights = jnp.asarray(calc.calculation.system.kpoints.weights)
    n_moments = 8
    exact = np.asarray(conductivity_moments(hamiltonians[0], *tangents, centre, half,
                                            jax.random.PRNGKey(0), weights, n_moments, 1,
                                            kind="exact")[0])[0, 0]
    samples = np.stack([
        np.asarray(conductivity_moments(hamiltonians[0], *tangents, centre, half,
                                        jax.random.PRNGKey(seed), weights, n_moments, 2)[0])[0, 0]
        for seed in range(16)])
    mean = samples.mean(axis=0)
    error = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    deviation = np.abs(mean - exact) / np.maximum(error, 1e-12 * np.abs(exact).max())
    assert np.median(deviation) < 2.0, np.median(deviation)
    assert deviation.max() < 5.0, deviation.max()


def test_an_ultrasoft_dataset_is_refused():
    """``H x = e S x`` is not what the moments of ``H`` describe."""
    from pathlib import Path

    pseudo = Path(__file__).parents[1] / "data" / "pseudo"
    keywords = {**SILICON, "pseudos": {"Si": "Si.pz-n-rrkjus_psl.0.1.UPF"},
                "kmesh": (1, 1, 1), "ecutrho": 4 * 163.3}
    calc = Calculator.from_keywords(pseudo_dir=pseudo, announce=False, **keywords)
    density = calc.get_scf().density
    with pytest.raises(NotImplementedError, match="S\\^-1 H"):
        run_kpm_conductivity(calc.system, calc.pseudos, density, fermi_energy=0.5,
                             calculation=calc.calculation, n_moments=8)
