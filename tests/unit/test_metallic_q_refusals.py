"""P132, I4: where the metallic phonon at ``q`` is refused, checked without a ground state.

A metal at ``q = 0`` runs through this route since I4, which carries ``ph.x``'s
``ef_shift`` there (``tests/regression/test_phonons_at_q.py`` has its number).
At a *nonzero* reciprocal lattice vector it is still refused: the Fermi level
moves there too, and ``ef_shift_wfc`` would add a state on the ``k`` sphere to
a ``dpsi`` on the ``k + G`` one. The electron-phonon coupling is refused at
every ``q = G``, zero included. Each refusal is a function of the cell and
``q`` alone, so it is checked here at gate speed;
``tests/regression/test_electron_phonon.py`` checks it through the calculator.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from defuqumat.response.phononq import _require_a_metallic_q, at_zone_centre
from defuqumat.system.cell import Cell

FCC = Cell.from_ibrav(2, [7.5, 0.0, 0.0, 0.0, 0.0, 0.0])


def _cartesian(crystal):
    return np.asarray(FCC.k_to_cartesian(np.asarray(crystal, dtype=float))) * FCC.tpiba


@pytest.mark.parametrize("crystal", [(1.0, 0.0, 0.0), (-1.0, 2.0, 1.0)])
def test_a_metal_at_a_reciprocal_lattice_vector_is_refused(crystal):
    with pytest.raises(NotImplementedError, match="ef_shift"):
        _require_a_metallic_q(FCC, _cartesian(crystal))


@pytest.mark.parametrize("crystal", [(0.0, 0.0, 0.0), (0.0, 0.0, 0.5),
                                     (0.25, 0.0, 0.0), (1.0, 0.0, 1e-6)])
def test_a_metal_away_from_a_nonzero_lattice_vector_runs(crystal):
    _require_a_metallic_q(FCC, _cartesian(crystal))


@pytest.mark.parametrize("crystal, centre", [
    ((0.0, 0.0, 0.0), True), ((1e-10, 0.0, 0.0), True), ((1.0, 0.0, 0.0), False),
    ((1e-6, 0.0, 0.0), False), ((0.5, 0.0, 0.0), False),
])
def test_the_fermi_level_shifts_at_q_zero_alone(crystal, centre):
    """``lmetq0``'s ``lgamma``: ``q = 0`` and nothing else, ``G`` excluded."""
    assert at_zone_centre(FCC, _cartesian(crystal)) is centre


@pytest.mark.parametrize("crystal", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)])
def test_the_coupling_at_a_reciprocal_lattice_vector_is_refused(crystal):
    """The guard fires before anything reads the ground state."""
    from defuqumat.response.elph import electron_phonon_at_q

    calculation = SimpleNamespace(
        system=SimpleNamespace(occupations="smearing", cell=FCC, nspin=1))
    with pytest.raises(NotImplementedError, match="ef_shift"):
        electron_phonon_at_q(calculation, None, None, None, q=crystal)
