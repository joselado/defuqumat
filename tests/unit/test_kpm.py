"""The kernel polynomial method's moments and density of states (P167).

Every check here is against something that shares no machinery with the
Chebyshev recursion: the eigenvalues of the dense Hamiltonian
(``Hamiltonian.matrix``, every state of the basis), the SCF's own Davidson
eigenvalues, the trace of ``H`` from its diagonal, and the electron count. The
cell is silicon at 12 Ry on a 2x2x2 mesh (``pw_scf/scf.in``'s physics), whose
sphere holds about 180 plane waves, so the exact trace is cheap.
"""

from __future__ import annotations

import functools

import jax
import numpy as np
import pytest

import defuqumat  # noqa: F401
from defuqumat import Calculator
from defuqumat.kpm import SpectralIntervalError, get_kernel, kpm_moments
from defuqumat.kpm.bounds import spectral_interval
from defuqumat.kpm.moments import live_mask
from defuqumat.kpm.profile import chebyshev_profile
from defuqumat.workflows.nscf import fixed_density_hamiltonians

pytestmark = pytest.mark.unit

SILICON = dict(
    lattice="fcc", a=5.398, pseudos={"Si": "Si.pz-vbc.UPF"},
    atoms=[("Si", (0.0, 0.0, 0.0)), ("Si", (0.25, 0.25, 0.25))],
    ecut=163.3, kmesh=(2, 2, 2),
)


@functools.lru_cache(maxsize=2)
def _silicon(noncolin: bool = False):
    from pathlib import Path

    pseudo = Path(__file__).parents[1] / "data" / "pseudo"
    keywords = dict(SILICON)
    if noncolin:
        keywords["qe"] = {"system": {"noncolin": True}}
    calc = Calculator.from_keywords(pseudo_dir=pseudo, announce=False, conv_thr=1e-10,
                                    **keywords)
    result = calc.get_scf()
    hamiltonians = fixed_density_hamiltonians(calc.calculation, result.density)
    return calc, result, hamiltonians


def _dense_spectrum(hamiltonian):
    """Every eigenvalue of ``H(k)`` in the live basis, per k-point, from the dense matrix."""
    mask = np.asarray(live_mask(hamiltonian))
    spectra = []
    for ik in range(mask.shape[0]):
        matrix = np.asarray(hamiltonian.matrix(ik))
        live = mask[ik]
        spectra.append(np.linalg.eigvalsh(matrix[np.ix_(live, live)]))
    return spectra


def _exact_moments(spectra, weights, centre, half, n_moments):
    total = np.zeros(n_moments)
    for values, w in zip(spectra, weights):
        x = (values - centre) / half
        total += w * np.cos(np.arange(n_moments)[:, None] * np.arccos(x)[None, :]).sum(axis=1)
    return total


def test_the_exact_trace_is_the_dense_spectrum():
    """The full-sphere recursion against ``sum_i T_n(e_i')`` of the dense matrix, to round-off."""
    calc, _, hamiltonians = _silicon()
    weights = np.asarray(calc.calculation.system.kpoints.weights)
    centre, half, _ = spectral_interval(hamiltonians)
    spectra = _dense_spectrum(hamiltonians[0])
    assert min(s.min() for s in spectra) > centre - half
    assert max(s.max() for s in spectra) < centre + half
    exact = _exact_moments(spectra, weights, centre, half, 40)
    moments = kpm_moments(hamiltonians, weights, centre, half, 40, trace="exact")[0, 0]
    np.testing.assert_allclose(moments, exact, rtol=0, atol=1e-9 * exact[0])


def test_the_stochastic_error_falls_as_one_over_root_r():
    """Seeded, ``R = 4`` against ``R = 64``: the error must fall by roughly four."""
    calc, _, hamiltonians = _silicon()
    weights = np.asarray(calc.calculation.system.kpoints.weights)
    centre, half, _ = spectral_interval(hamiltonians)
    exact = _exact_moments(_dense_spectrum(hamiltonians[0]), weights, centre, half, 40)
    errors = []
    for n_random in (4, 64):
        samples = kpm_moments(hamiltonians, weights, centre, half, 40, n_random=n_random,
                              seed=3)[0]
        errors.append(np.sqrt(np.mean((samples.mean(axis=0)[1:] - exact[1:]) ** 2)))
    assert errors[1] < errors[0] / 2.0, errors


def test_mu_one_is_the_trace_of_h():
    """``mu_1 = Tr H'`` from the diagonal of ``H``, which no recursion touches."""
    calc, _, hamiltonians = _silicon()
    hamiltonian = hamiltonians[0]
    weights = np.asarray(calc.calculation.system.kpoints.weights)
    centre, half, _ = spectral_interval(hamiltonians)
    mask = np.asarray(live_mask(hamiltonian))
    trace = sum(w * ((np.asarray(hamiltonian.diagonal(ik))[mask[ik]] - centre) / half).sum()
                for ik, w in enumerate(weights))
    moments = kpm_moments(hamiltonians, weights, centre, half, 4, trace="exact")[0, 0]
    assert abs(moments[1] - trace) < 1e-9 * abs(trace)


def test_the_guard_fires_on_a_short_interval():
    """An interval ten per cent too narrow is refused, not used."""
    calc, _, hamiltonians = _silicon()
    weights = np.asarray(calc.calculation.system.kpoints.weights)
    centre, half, _ = spectral_interval(hamiltonians)
    with pytest.raises(SpectralIntervalError):
        kpm_moments(hamiltonians, weights, centre, 0.9 * half, 200, n_random=2)


def test_the_occupied_dos_is_the_scf_bands_through_the_same_kernel():
    """Below the top of the computed bands, the KPM DOS is Davidson's eigenvalues smoothed alike."""
    calc, result, _ = _silicon()
    dos = calc.get_kpm_dos(broadening=0.02, n_random=16, emin=-0.5, emax=0.3)
    weights = np.asarray(calc.calculation.system.kpoints.weights)
    x = (np.asarray(result.eigenvalues) - dos.centre) / dos.half_width
    n = np.arange(dos.n_moments)
    reference_moments = np.einsum("k,knb->n", weights,
                                  np.cos(n[None, :, None] * np.arccos(x)[:, None, :]))
    reference, _ = chebyshev_profile(reference_moments, get_kernel("jackson").factors(dos.n_moments),
                                     dos.centre, dos.half_width, dos.energies)
    reference = np.asarray(reference)
    # far enough below the top computed band that the states above it do not reach
    top = np.asarray(result.eigenvalues).max()
    window = dos.energies < top - 6.0 * dos.broadening
    difference = np.abs(dos.dos - reference)[window]
    assert np.max(difference / (dos.dos_error[window] + 1e-3 * reference.max())) < 5.0


def test_the_gap_holds_the_electrons():
    """The integrated DOS in silicon's gap is the electron count, eight, within its noise.

    On three k-points of about 180 plane waves the count is noisy (7.88, 8.44 and
    8.04 at R = 4, 16 and 64 when measured), so the check is against the count's
    own standard error, and that error must itself be small at R = 64.
    """
    calc, _, hamiltonians = _silicon()
    dos = calc.get_kpm_dos(broadening=0.02, n_random=64)
    spectra = _dense_spectrum(hamiltonians[0])
    gap_centre = 0.5 * (max(s[3] for s in spectra) + min(s[4] for s in spectra))
    error = dos.count_error_at(gap_centre)
    assert 0.0 < error < 0.2
    assert abs(dos.count_at(gap_centre) - 8.0) < 4.0 * error


def test_a_second_call_compiles_nothing():
    """The moments kernel is a module-level jit over its arguments: a new potential reuses it."""
    import logging

    import equinox as eqx

    calc, _, hamiltonians = _silicon()
    weights = np.asarray(calc.calculation.system.kpoints.weights)
    centre, half, _ = spectral_interval(hamiltonians)
    kpm_moments(hamiltonians, weights, centre, half, 20, n_random=2)
    shifted = tuple(eqx.tree_at(lambda h: h.potential, h, h.potential + 0.01)
                    for h in hamiltonians)
    compiled = []
    handler = logging.Handler()
    handler.emit = lambda record: compiled.append(record.getMessage())
    logger = logging.getLogger("jax")
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    jax.config.update("jax_log_compiles", True)
    try:
        kpm_moments(shifted, weights, centre + 0.001, half, 20, n_random=2, seed=5)
    finally:
        jax.config.update("jax_log_compiles", False)
        logger.removeHandler(handler)
        logger.setLevel(previous)
    assert not [m for m in compiled if "chebyshev_moments" in m
                and ("Compiling" in m or "Finished XLA compilation" in m)], compiled


def test_the_spinor_trace_is_twice_the_scalar_one():
    """Without spin-orbit coupling or a moment, the spinor's exact moments are twice the scalar ones."""
    calc, _, hamiltonians = _silicon()
    spinor_calc, _, spinor = _silicon(noncolin=True)
    weights = np.asarray(calc.calculation.system.kpoints.weights)
    spinor_weights = np.asarray(spinor_calc.calculation.system.kpoints.weights)
    centre, half, _ = spectral_interval(hamiltonians)
    scalar = kpm_moments(hamiltonians, weights, centre, half, 20, trace="exact")[0, 0]
    double = kpm_moments(spinor, spinor_weights, centre, half, 20, trace="exact")[0, 0]
    # the scalar weights carry the factor two of the spin, the spinor's do not
    np.testing.assert_allclose(double, scalar, rtol=0, atol=1e-8 * scalar[0])


def test_ultrasoft_is_refused_by_name():
    from pathlib import Path

    pseudo = Path(__file__).parents[1] / "data" / "pseudo"
    keywords = {**SILICON, "pseudos": {"Si": "Si.pz-n-rrkjus_psl.0.1.UPF"},
                "kmesh": (1, 1, 1), "ecutrho": 4 * 163.3}
    calc = Calculator.from_keywords(pseudo_dir=pseudo, announce=False, **keywords)
    calc.get_scf()
    # the total DOS runs on S^(-1/2) H S^(-1/2) since P171 (test_kpm_augmented.py); a
    # trace with chosen start vectors does not
    with pytest.raises(NotImplementedError, match="S\\^-1 H"):
        calc.get_kpm_ldos([(0.1, 0.1, 0.1)], n_moments=20)


def test_two_spin_channels_add_up_to_the_unpolarized_trace():
    """``nspin = 2`` with no moment: the channels agree, and their sum is the scalar run's trace.

    Each channel carries weights summing to 1 where the scalar run's sum to 2, so the
    exact moments of the two channels add up to the scalar ones, to the converged
    residual of the magnetization.
    """
    from pathlib import Path

    calc, _, hamiltonians = _silicon()
    pseudo = Path(__file__).parents[1] / "data" / "pseudo"
    polarized = Calculator.from_keywords(
        pseudo_dir=pseudo, announce=False, conv_thr=1e-10,
        **{**SILICON, "qe": {"system": {"nspin": 2, "tot_magnetization": 0}}})
    result = polarized.get_scf()
    channels = fixed_density_hamiltonians(polarized.calculation, result.density)
    assert len(channels) == 2
    centre, half, _ = spectral_interval(hamiltonians)
    scalar = kpm_moments(hamiltonians, np.asarray(calc.calculation.system.kpoints.weights),
                         centre, half, 20, trace="exact")[0, 0]
    both = kpm_moments(channels, np.asarray(polarized.calculation.system.kpoints.weights),
                       centre, half, 20, trace="exact")[:, 0]
    np.testing.assert_allclose(both[0], both[1], rtol=0, atol=1e-7 * both[0, 0])
    np.testing.assert_allclose(both.sum(axis=0), scalar, rtol=0, atol=1e-7 * scalar[0])
    dos = polarized.get_kpm_dos(n_moments=40, n_random=2)
    assert dos.dos.shape == (2, dos.energies.size) and dos.nspin == 2


@pytest.mark.parametrize("x0", [0.0, -0.85])
@pytest.mark.parametrize("kernel", ["jackson", "lorentz"])
def test_the_kernel_width_is_the_width_of_the_peak(x0, kernel):
    """The half width ``width_at`` reports against the peak the series of one level gives."""
    from defuqumat.kpm.kernels import get_kernel as kernel_named

    jackson = kernel_named(kernel)
    n_moments = 800
    moments = np.cos(np.arange(n_moments) * np.arccos(x0))
    x = np.linspace(x0 - 0.05, x0 + 0.05, 20001)
    peak, _ = chebyshev_profile(moments, jackson.factors(n_moments), 0.0, 1.0, x)
    peak = np.asarray(peak)
    above = x[peak >= 0.5 * peak.max()]
    measured = 0.5 * (above[-1] - above[0])
    assert abs(measured / float(jackson.width_at(n_moments, x0)) - 1.0) < 0.02
