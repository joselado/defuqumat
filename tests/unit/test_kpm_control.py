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


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Free the compiled programs after each test (``.claude/rules/memory-design.md``).

    The KPM files compile a recursion per cell and per moment count, and in the
    one-process fast gate they sat in memory beside the files that follow; the
    gate was killed at its 6 GB cap two files later (2026-11-07).
    """
    yield
    import jax

    jax.clear_caches()

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
    plain = calc.get_kpm_dos(broadening=BROADENING, n_random=4, seed=1,
                           control_variate=False)
    varied = calc.get_kpm_dos(broadening=BROADENING, n_random=4, seed=1, control_variate=True)
    reference = _reference(calc, hamiltonians, plain)
    window = _valence(calc, result, plain)
    error = [np.sqrt(np.mean((d.dos - reference)[window] ** 2)) for d in (plain, varied)]
    assert error[1] < error[0] / 5.0, error
    z = np.abs(varied.dos - reference)[window] / (varied.dos_error[window] + 1e-12)
    assert 0.1 < np.median(z) < 2.0, np.median(z)


def test_the_variate_is_unbiased():
    """Over 16 seeds the mean error against the exact DOS is zero within 4.5 standard errors.

    **Each run against its own reference.** The seed also starts the Lanczos run
    of the interval, so the half width, and with it the kernel's width, differs
    from run to run (8e-4 Ry here); at filter degree 16 the variate is precise
    enough (1e-4 at the band bottom) that one reference for all runs read as a
    bias of 6e-4, at 5.9 standard errors, which a run's own reference removes
    (2.7 at 48 seeds).
    """
    calc, result, hamiltonians = _silicon()
    errors = []
    for seed in range(16):
        dos = calc.get_kpm_dos(broadening=BROADENING, n_random=2, seed=seed,
                               control_variate=True)
        errors.append(dos.dos - _reference(calc, hamiltonians, dos))
    errors = np.stack(errors)
    mean, spread = errors.mean(axis=0), errors.std(axis=0, ddof=1) / np.sqrt(len(errors))
    window = spread > 1e-9
    assert np.max(np.abs(mean)[window] / spread[window]) < 4.5


def test_the_exact_trace_refuses_the_variate():
    calc, _, _ = _silicon()
    with pytest.raises(ValueError, match="exact"):
        calc.get_kpm_dos(n_moments=20, trace="exact", control_variate=True)


def test_auto_takes_the_variate_only_where_it_applies(monkeypatch):
    """``"auto"``: on for random vectors with orbitals, off for the exact trace, off with a warning without orbitals."""
    import defuqumat.projwfc.projections as projections
    from defuqumat.kpm.control import resolve_variate

    calc, _, _ = _silicon()
    assert resolve_variate("auto", "stochastic", calc.calculation) is True
    assert resolve_variate("auto", "exact", calc.calculation) is False
    assert resolve_variate(False, "stochastic", calc.calculation) is False
    assert resolve_variate(True, "exact", calc.calculation) is True  # and the run raises
    monkeypatch.setattr(projections, "calculation_channels", lambda calculation: ())
    with pytest.warns(RuntimeWarning, match="PP_CHI"):
        assert resolve_variate("auto", "stochastic", calc.calculation) is False


@pytest.mark.parametrize("regime", ["spinor", "two channels"])
def test_the_variate_is_unbiased_in_the_other_spin_regimes(regime):
    """The mean of 12 seeds' moments with the variate equals the exact trace's, for a spinor and for ``nspin = 2``.

    These are the regimes ``"auto"`` turns the variate on in without P170 having
    looked: the orbitals are spinors in one, and each channel draws its own key in
    the other.
    """
    from pathlib import Path

    from defuqumat import Calculator
    from tests.unit.test_kpm import SILICON

    if regime == "spinor":
        calc = _silicon(True)[0]
    else:
        pseudo = Path(__file__).parents[1] / "data" / "pseudo"
        calc = Calculator.from_keywords(
            pseudo_dir=pseudo, announce=False, conv_thr=1e-10,
            **{**SILICON, "qe": {"system": {"nspin": 2, "tot_magnetization": 0}}})
        calc.get_scf()
    options = dict(n_moments=24, emin=-0.5, emax=1.0)
    exact = calc.get_kpm_dos(trace="exact", **options).samples[:, 0]  # (nspin, N)
    runs = np.stack([calc.get_kpm_dos(n_random=2, seed=s, control_variate=True,
                                      **options).samples.mean(axis=1) for s in range(12)])
    mean = runs.mean(axis=0)
    spread = runs.std(axis=0, ddof=1) / np.sqrt(len(runs))
    window = spread > 1e-9 * np.abs(exact).max()
    assert window.any()
    assert np.max(np.abs(mean - exact)[window] / spread[window]) < 4.5
