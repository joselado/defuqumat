"""The gap across a cut through the band index, and the SCF's warning on it.

A filling that ends inside a degenerate multiplet -- ``occupations = 'fixed'``
on an atom whose minority channel holds one electron in a threefold shell -- makes
everything built from the occupied set a function of the rotation the
eigensolver returned (rule D4). :func:`~defuqumat.scf.occupations.cut_gaps` is
the one arithmetic every caller shares (the Sternheimer refusal, the real-time
refusal, the ultracell and the three sum-over-states workflows' ``band_cut_gap``,
and the SCF's own :attr:`SCFResult.cut_gap`), and
:func:`~defuqumat.scf.driver.warn_about_a_cut_shell` is the SCF's outcome: a
warning and never a refusal, since with the point group on the symmetrised
density and the total energy are single-valued (measured to 6e-12 and 1e-13 Ry
on the oxygen atom, ``notes/DEGENERACY-NEXT.md``).

Everything here is on synthetic spectra; the oxygen atom itself is
``tests/regression/test_lsda.py``.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from defuqumat.scf.driver import warn_about_a_cut_shell
from defuqumat.scf.occupations import cut_gaps, occupied_counts
from defuqumat.solvers.davidson import EMPTY_ETHR_FLOOR, empty_band_threshold
from defuqumat.ultracell.hamiltonian import multiplet_cut

pytestmark = pytest.mark.unit


def _spectrum():
    """``(2, nk, nbnd)`` in Ry: two k-points, a 3-fold shell in channel 1 at k = 1.

    Channel 0 is gapped everywhere; channel 1 holds an exactly degenerate triplet
    at the second k-point in bands 1..3, and at the first k-point the same three
    bands are split by 0.02. The per-k **minimum** is what the cut reads, so a
    cut at 2 in channel 1 is 0.0 (k = 1) and not 0.02 (k = 0).
    """
    return np.array(
        [
            [[-0.60, -0.10, 0.20, 0.50, 0.90],
             [-0.55, -0.05, 0.25, 0.55, 0.95]],
            [[-0.30, 0.05, 0.07, 0.09, 1.00],
             [-0.35, 0.00, 0.00, 0.00, 1.10]],
        ]
    )


def test_gap_is_the_per_k_minimum_across_the_cut_per_channel():
    e = _spectrum()
    gaps = cut_gaps(e, (2, 2))
    assert gaps[0] == pytest.approx(0.30)        # band 2 - band 1, 0.30 at both k
    assert gaps[1] == pytest.approx(0.0)         # the exact triplet at k = 1 wins over 0.02 at k = 0
    # A cut below the shell in channel 1 reads the gap *into* it, which is real.
    assert cut_gaps(e, (2, 1))[1] == pytest.approx(0.35)
    # A cut above the shell reads the gap out of it, at the k-point where it is
    # smallest (0.91 at k = 0, 1.10 at k = 1).
    assert cut_gaps(e, (2, 4))[1] == pytest.approx(0.91)


def test_nothing_to_measure_is_nan_and_not_a_number():
    e = _spectrum()
    gaps = cut_gaps(e, (0, 5))
    assert np.isnan(gaps).all()
    # NaN compares false against any tolerance, so a caller that tests
    # ``gap < tol`` is silent on an unmeasured channel rather than wrong.
    assert not (gaps < EMPTY_ETHR_FLOOR).any()


def test_shape_contract():
    with pytest.raises(ValueError, match=r"\(nspin, nk, nbnd\)"):
        cut_gaps(_spectrum()[0], (2,))
    with pytest.raises(ValueError, match="cuts were given"):
        cut_gaps(_spectrum(), (2,))


def test_multiplet_cut_is_the_same_arithmetic_over_every_leading_axis():
    """The ultracell's truncation cut folds spin blocks, k and cells into one k axis."""
    e = _spectrum()
    assert multiplet_cut(e, 2) == pytest.approx(min(cut_gaps(e, (2, 2))))
    stacked = np.stack([e, e + 0.5])            # a fourth leading axis
    assert multiplet_cut(stacked, 4) == pytest.approx(min(cut_gaps(e, (4, 4))))
    with pytest.raises(ValueError, match="one band more"):
        multiplet_cut(e, 5)


def test_occupied_counts_is_one_rule_for_every_regime():
    unpolarized = SimpleNamespace(nspin=1, noncolin=False, nelec=8.0, nelup=4.0, neldw=4.0)
    spinor = SimpleNamespace(nspin=1, noncolin=True, nelec=8.0, nelup=4.0, neldw=4.0)
    polarized = SimpleNamespace(nspin=2, noncolin=False, nelec=6.0, nelup=4.0, neldw=2.0)
    half = SimpleNamespace(nspin=2, noncolin=False, nelec=7.0, nelup=4.5, neldw=2.5)
    assert occupied_counts(unpolarized) == (4,)
    assert occupied_counts(spinor) == (8,)      # a spinor band holds one electron
    assert occupied_counts(polarized) == (4, 2)
    assert occupied_counts(half) == (5, 3)      # NINT rounds a half away from zero


def test_the_warning_fires_on_a_cut_shell_and_names_the_channel():
    """The oxygen atom's shape: channel 1 fills 2 of 6 bands into an exact triplet."""
    ethr = 1.0e-12
    gap = (0.67, 4.9e-12)
    with pytest.warns(RuntimeWarning, match="channel 1 fills 2 of 6 bands") as record:
        warn_about_a_cut_shell(gap, ethr, (4, 2), 6)
    message = str(record[0].message)
    assert "smearing" in message
    assert "channel 0" not in message


def test_the_warning_is_silent_on_a_gapped_filling_and_on_nan():
    ethr = 1.0e-12
    tolerance = empty_band_threshold(ethr)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Gapped in both channels, by a margin over the tolerance.
        warn_about_a_cut_shell((0.67, 0.46), ethr, (4, 4), 6)
        # One channel with no band above its cut: unmeasured, and silent.
        warn_about_a_cut_shell((0.67, float("nan")), ethr, (4, 6), 6)
        # No cut at all (a smearing run).
        warn_about_a_cut_shell(None, ethr, (4, 4), 6)
        # Just above the tolerance is a gap as far as this run can tell.
        warn_about_a_cut_shell((0.67, 1.01 * tolerance), ethr, (4, 2), 6)
    # ...and just below it is not.
    with pytest.warns(RuntimeWarning):
        warn_about_a_cut_shell((0.67, 0.99 * tolerance), ethr, (4, 2), 6)


def test_the_tolerance_is_the_empty_band_promise_not_ethr():
    """A pair split by 3e-7 -- what an SCF leaves on a symmetry-degenerate empty
    pair at the default ``ethr`` (``OPEN.md`` B1) -- is degenerate here, because
    the band above the cut was converged only to ``max(5 ethr, 1e-5)``."""
    with pytest.warns(RuntimeWarning):
        warn_about_a_cut_shell((3.0e-7,), 1.0e-10, (4,), 8)
