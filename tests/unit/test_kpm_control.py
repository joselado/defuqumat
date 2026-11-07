"""The control variate of the KPM trace (P170): unbiased, and much less noise in the valence band.

The reference is the exact DOS of the dense Hamiltonian (every eigenvalue of
``matrix(ik)`` on the live sphere) through the same Jackson kernel, so it shares
nothing with the recursion or the variate.
"""

from __future__ import annotations

import functools

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


@functools.lru_cache(maxsize=1)
def _two_channels():
    """Si2 at ``nspin = 2`` with no moment, converged: two channels, each with its own key."""
    from pathlib import Path

    from defuqumat import Calculator
    from tests.unit.test_kpm import SILICON

    pseudo = Path(__file__).parents[1] / "data" / "pseudo"
    calc = Calculator.from_keywords(
        pseudo_dir=pseudo, announce=False, conv_thr=1e-10,
        **{**SILICON, "qe": {"system": {"nspin": 2, "tot_magnetization": 0}}})
    return calc, calc.get_scf()


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


def test_true_without_orbitals_refuses(monkeypatch):
    """``control_variate=True`` with no ``PP_CHI`` anywhere raises, before any moment, where it used to run the plain trace silently."""
    import defuqumat.kpm.dos as dos
    import defuqumat.projwfc.projections as projections
    from defuqumat.kpm.control import resolve_variate

    calc, _, _ = _silicon()
    monkeypatch.setattr(projections, "calculation_channels", lambda calculation: ())
    with pytest.raises(ValueError, match="PP_CHI"):
        resolve_variate(True, "stochastic", calc.calculation)
    assert resolve_variate(True, "exact", calc.calculation) is True  # the caller names the trace
    assert resolve_variate(False, "stochastic", calc.calculation) is False

    def no_moments(*args, **kwargs):
        raise AssertionError("the moments ran before the variate was refused")

    monkeypatch.setattr(dos, "kpm_moments", no_moments)
    with pytest.raises(ValueError, match="PP_CHI"):
        calc.get_kpm_dos(n_moments=20, n_random=2, control_variate=True)


@pytest.mark.parametrize("regime", ["spinor", "two channels"])
def test_the_variate_is_unbiased_in_the_other_spin_regimes(regime):
    """The mean of 12 seeds' moments with the variate equals the exact trace's, for a spinor and for ``nspin = 2``.

    These are the regimes ``"auto"`` turns the variate on in without P170 having
    looked: the orbitals are spinors in one, and each channel draws its own key in
    the other.
    """
    calc = _silicon(True)[0] if regime == "spinor" else _two_channels()[0]
    options = dict(n_moments=24, emin=-0.5, emax=1.0)
    exact = calc.get_kpm_dos(trace="exact", **options).samples[:, 0]  # (nspin, N)
    runs = np.stack([calc.get_kpm_dos(n_random=2, seed=s, control_variate=True,
                                      **options).samples.mean(axis=1) for s in range(12)])
    mean = runs.mean(axis=0)
    spread = runs.std(axis=0, ddof=1) / np.sqrt(len(runs))
    window = spread > 1e-9 * np.abs(exact).max()
    assert window.any()
    assert np.max(np.abs(mean - exact)[window] / spread[window]) < 4.5


def test_each_channel_takes_the_variate_on_its_own_random_vectors():
    """``nspin = 2``: the correction of each channel is drawn on that channel's random vectors.

    Four seeds at R = 4, each against the exact trace at the same seed (so the same
    interval and moment count): the rms error of each channel's DOS in the valence
    window is below 1e-3 of the peak, measured 9e-6 and 6e-6 on D22. A correction
    drawn with channel 0's key for both channels leaves channel 0 as it is and
    gives channel 1 6.1e-2, a plain trace with extra noise; the unbiasedness test
    above passes that defect, since an uncorrelated correction is unbiased too.
    """
    calc, result = _two_channels()
    errors = []
    for seed in range(4):
        dos = calc.get_kpm_dos(broadening=BROADENING, n_random=4, seed=seed,
                               control_variate=True)
        exact = calc.get_kpm_dos(broadening=BROADENING, trace="exact", seed=seed)
        assert (dos.centre, dos.half_width, dos.n_moments) == (
            exact.centre, exact.half_width, exact.n_moments)
        window = dos.energies < float(result.homo)
        peak = np.abs(exact.dos).max()
        errors.append([np.sqrt(np.mean((dos.dos[c] - exact.dos[c])[window] ** 2)) / peak
                       for c in range(2)])
    errors = np.mean(errors, axis=0)
    assert np.all(errors < 1e-3), errors
