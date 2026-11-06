"""The control variate of the KPM trace (P170): unbiased, and much less noise in the valence band.

The reference is the exact DOS of the dense Hamiltonian (every eigenvalue of
``matrix(ik)`` on the live sphere) through the same Jackson kernel, so it shares
nothing with the recursion or the variate.
"""

from __future__ import annotations

import numpy as np
import pytest

import defuqumat  # noqa: F401
from defuqumat.kpm import get_kernel
from defuqumat.kpm.profile import chebyshev_profile
from tests.unit.test_kpm import _dense_spectrum, _silicon

pytestmark = pytest.mark.unit

BROADENING = 0.02


def _reference(calc, hamiltonians, dos):
    weights = np.asarray(calc.calculation.system.kpoints.weights)
    moments = np.zeros(dos.n_moments)
    for values, w in zip(_dense_spectrum(hamiltonians[0]), weights):
        x = (values - dos.centre) / dos.half_width
        moments += w * np.cos(np.arange(dos.n_moments)[:, None] * np.arccos(x)[None, :]).sum(axis=1)
    reference, _ = chebyshev_profile(moments, get_kernel("jackson").factors(dos.n_moments),
                                     dos.centre, dos.half_width, dos.energies)
    return np.asarray(reference)


def _valence(calc, result, dos):
    return dos.energies < float(np.asarray(result.eigenvalues).max())


def test_the_variate_cuts_the_noise_in_the_valence_band():
    """At R = 4, the valence-band error falls at least five-fold, and the reported error stays honest."""
    calc, result, hamiltonians = _silicon()
    plain = calc.get_kpm_dos(broadening=BROADENING, n_random=4, seed=1)
    varied = calc.get_kpm_dos(broadening=BROADENING, n_random=4, seed=1, control_variate=True)
    reference = _reference(calc, hamiltonians, plain)
    window = _valence(calc, result, plain)
    error = [np.sqrt(np.mean((d.dos - reference)[window] ** 2)) for d in (plain, varied)]
    assert error[1] < error[0] / 5.0, error
    z = np.abs(varied.dos - reference)[window] / (varied.dos_error[window] + 1e-12)
    assert 0.1 < np.median(z) < 2.0, np.median(z)


def test_the_variate_is_unbiased():
    """Over 16 seeds the mean DOS equals the exact one within four standard errors of the mean."""
    calc, result, hamiltonians = _silicon()
    runs = np.stack([calc.get_kpm_dos(broadening=BROADENING, n_random=2, seed=s,
                                      control_variate=True).dos for s in range(16)])
    reference = _reference(calc, hamiltonians,
                           calc.get_kpm_dos(broadening=BROADENING, n_random=2, seed=0))
    mean, spread = runs.mean(axis=0), runs.std(axis=0, ddof=1) / np.sqrt(len(runs))
    window = spread > 1e-6 * reference.max()
    assert np.max(np.abs(mean - reference)[window] / spread[window]) < 4.5


def test_the_exact_trace_refuses_the_variate():
    calc, _, _ = _silicon()
    with pytest.raises(ValueError, match="exact"):
        calc.get_kpm_dos(n_moments=20, trace="exact", control_variate=True)
