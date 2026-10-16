"""``BandStructure.gap`` on synthetic eigenvalues: the filling rule per regime.

None of these needs an SCF, so the file belongs in the fast gate. The numbers
against ``pw.x`` are in ``tests/regression/test_lsda.py``
(``test_the_band_gap_reads_both_channels_as_pw_x_does``).
"""

import numpy as np
import pytest

from defuqumat.units import RY_TO_EV
from defuqumat.workflows.bands import BandStructure

pytestmark = pytest.mark.unit


def _lsda():
    """Two channels, two k-points, four bands, in Ry.

    The majority fills three bands and its gap is 0.5 Ry; the minority fills
    one, and its empty band 2 sits *below* the majority's band 4, so the
    fundamental gap crosses channels: majority top (0.2, at k1) to minority
    bottom (0.3, at k0).
    """
    up = np.array([[-1.0, -0.5, 0.1, 0.7], [-1.0, -0.4, 0.2, 0.8]])
    down = np.array([[-0.8, 0.3, 0.4, 0.9], [-0.7, 0.35, 0.45, 0.95]])
    return BandStructure(kpoints=None, eigenvalues=np.stack([up, down]), nspin=2)


def test_a_spin_polarized_gap_without_a_filling_per_channel_is_refused():
    bands = BandStructure(kpoints=None, eigenvalues=np.zeros((2, 1, 6)), nspin=2)
    with pytest.raises(NotImplementedError, match="tot_magnetization"):
        bands.gap(6)


def test_a_fractional_split_between_the_channels_is_refused():
    with pytest.raises(ValueError, match="whole number"):
        _lsda().gap(4, tot_magnetization=1.5)


def test_the_lsda_gap_runs_across_the_channels():
    bands = _lsda()
    homo, lumo = bands.band_edges(4, tot_magnetization=2)
    assert homo == pytest.approx(0.2 * RY_TO_EV, abs=1e-12)
    assert lumo == pytest.approx(0.3 * RY_TO_EV, abs=1e-12)
    assert bands.gap(4, 2) == pytest.approx(0.1 * RY_TO_EV, abs=1e-12)
    # The same filling given band by band.
    assert bands.gap(occupied=(3, 1)) == bands.gap(4, 2)
    # Each channel's own gap is larger than the fundamental one.
    assert 0.5 * RY_TO_EV > bands.gap(4, 2)


def test_an_unpolarized_run_in_two_channels_gives_the_one_channel_gap():
    one = np.array([[-1.0, -0.5, 0.1, 0.7], [-1.0, -0.4, 0.2, 0.8]])
    single = BandStructure(kpoints=None, eigenvalues=one, nspin=1)
    double = BandStructure(kpoints=None, eigenvalues=np.stack([one, one]), nspin=2)
    assert double.gap(6, 0) == single.gap(6)
    assert single.gap(6) == pytest.approx(0.5 * RY_TO_EV, abs=1e-12)


def test_a_spinor_band_holds_one_electron():
    levels = np.array([[-1.0, -0.5, 0.1, 0.7]])
    bands = BandStructure(kpoints=None, eigenvalues=levels, nspin=4)
    assert bands.gap(3) == pytest.approx(0.6 * RY_TO_EV, abs=1e-12)


def test_a_filling_with_no_empty_band_above_it_is_refused():
    with pytest.raises(ValueError, match="nbnd"):
        _lsda().gap(occupied=(4, 4))
