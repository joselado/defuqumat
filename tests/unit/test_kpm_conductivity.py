"""The optical conductivity by two-index Chebyshev moments (P169).

The moments are checked against the dense route, which shares no machinery
with the recursion: every eigenpair of ``Hamiltonian.matrix`` and the velocity
matrix elements of those eigenvectors from
:meth:`~defuqumat.response.velocity.VelocityOperator.matrix_elements` (one
``jvp`` of the whole ``H(k)`` per direction, contracted inside its own k walk).
The conductivity is checked against the band sum
:func:`~defuqumat.workflows.conductivity.run_conductivity` at the broadening the
Lorentz kernel gives a transition. Silicon at 12 Ry on a 2x2x2 mesh, as in
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
    tangents = velocity_tangents(calc.calculation, hamiltonians, AXES)
    weights = jnp.asarray(calc.calculation.system.kpoints.weights)
    mu, norms, _ = conductivity_moments(
        hamiltonians[0], tuple(t[0] for t in tangents), centre, half,
        jax.random.PRNGKey(0), weights, n_moments, 1, kind="exact")
    dense = _dense_moments(calc, result, hamiltonians[0], centre, half, n_moments)
    scale = np.abs(dense).max()
    np.testing.assert_allclose(np.asarray(mu), dense, rtol=0, atol=1e-9 * scale)
    assert np.all(np.asarray(norms) <= 1.0 + 1e-9)


def test_the_conductivity_is_the_band_sum_at_twice_the_lorentz_width():
    """Exact trace, Lorentz kernel, against ``run_conductivity`` at ``eta = 2 lambda half / N``."""
    calc, result, _ = _silicon()
    gap_level = result.homo + 0.02
    kpm = run_kpm_conductivity(calc.system, calc.pseudos, result.density,
                               calculation=calc.calculation, fermi_energy=gap_level,
                               directions=("x",), kernel="lorentz", trace="exact",
                               broadening=0.01, window=0.8, delta_e=0.002)
    band = calc.get_optical_conductivity(
        nbnd=60, broadening=kpm.transition_broadening, intraband=False,
        frequencies=kpm.frequencies[1:], fermi_energy=gap_level)
    kpm_xx = kpm.sigma[1:, 0, 0].real
    band_xx = band.sigma[:, 0, 0].real
    window = (kpm.frequencies[1:] > 0.15) & (kpm.frequencies[1:] < 0.6)
    error = np.abs(kpm_xx - band_xx)[window].max() / band_xx[window].max()
    assert error < 0.05, error


def test_a_second_call_compiles_nothing():
    """The moments are a module-level ``jit``; a second potential reuses the executable."""
    calc, result, hamiltonians = _silicon()
    centre, half, _ = spectral_interval(hamiltonians)
    tangents = tuple(t[0] for t in velocity_tangents(calc.calculation, hamiltonians, AXES[:1]))
    weights = jnp.asarray(calc.calculation.system.kpoints.weights)
    args = (centre, half, jax.random.PRNGKey(0), weights, 8, 2)
    conductivity_moments(hamiltonians[0], tangents, *args)
    moved = fixed_density_hamiltonians(calc.calculation, 0.999 * result.density)
    before = conductivity_moments._cache_size()
    conductivity_moments(moved[0], tangents, *args)
    assert conductivity_moments._cache_size() == before


def test_the_stochastic_trace_is_unbiased():
    """The mean over seeds of ``R = 2`` estimates against the exact moments, within its own error."""
    calc, _, hamiltonians = _silicon()
    centre, half, _ = spectral_interval(hamiltonians)
    tangents = tuple(t[0] for t in velocity_tangents(calc.calculation, hamiltonians, AXES[:1]))
    weights = jnp.asarray(calc.calculation.system.kpoints.weights)
    n_moments = 8
    exact = np.asarray(conductivity_moments(hamiltonians[0], tangents, centre, half,
                                            jax.random.PRNGKey(0), weights, n_moments, 1,
                                            kind="exact")[0])[0, 0]
    samples = np.stack([
        np.asarray(conductivity_moments(hamiltonians[0], tangents, centre, half,
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
