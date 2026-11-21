"""A PAW third derivative on a symmetry-reduced k-set is refused (audit 2026-11-20c #2).

The second-order energy ``F`` builds PAW's one-centre ``D`` from the raw ``becsum`` of
the k-points the run holds, and so does the operator ``db`` is solved with. On a wedge
that sum is not the ground state's ``becsum``, and its derivative is a wedge sum that the
average of the assembled tensor does not complete. Measured on
``si-epsilon-paw-unshifted.in`` against its ``-nosym`` twin on the same 32^3 grid: the
Raman tensor 1.8e-5 off and ``d(eps)/dx`` 6.5e-4, on both routes
(:func:`~defuqumat.response.electrostriction.require_a_whole_grid_for_paw` has the
numbers and the fix).

The refusal is checked on calculations alone, so no SCF is run: the entry points are
handed ``result=None`` and refuse before they touch it. The regimes beside it pass the
guard: the same PAW cell run whole, the same cell at Gamma alone with symmetry on (a
whole grid), and an ultrasoft wedge. Their end-to-end third derivatives are slow tests:
PAW silicon whole in ``tests/regression/test_nonlinear.py`` and ``test_streamed_third.py``,
a norm-conserving wedge in ``test_nonlinear.py`` (``alas-raman-wedge``) and
``test_electrostriction.py`` (``si-strain-wedge``).
"""

from pathlib import Path

import pytest

from defuqumat import Calculator
from defuqumat.response.electrostriction import (
    electrostriction,
    require_a_whole_grid_for_paw,
)
from defuqumat.response.nonlinear import raman_tensors
from defuqumat.response.spectra import vibrational_spectrum

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"
REFUSAL = "PAW dataset on a symmetry-reduced k-set"


def _calculation(case: str, edit=lambda text: text):
    text = edit((CASES / f"{case}.in").read_text())
    return Calculator.from_text(text, pseudo_dir=PSEUDO, announce=False).calculation


@pytest.fixture(scope="module")
def paw_wedge():
    calculation = _calculation("si-epsilon-paw-unshifted")
    assert calculation.paw is not None and calculation.use_symmetry
    assert len(calculation.system.kpoints.weights) == 3, "the wedge of the 2x2x2 grid"
    return calculation


def test_the_raman_tensors_refuse_a_paw_wedge(paw_wedge):
    with pytest.raises(NotImplementedError, match=REFUSAL):
        raman_tensors(paw_wedge, None)
    with pytest.raises(NotImplementedError, match=REFUSAL):
        vibrational_spectrum(paw_wedge, None)


@pytest.mark.parametrize("elastic", [False, True])
def test_the_electrostriction_refuses_a_paw_wedge(paw_wedge, elastic):
    """With ``elastic`` on, this refusal comes before the elastic one, whose message
    would send the caller to ``elastic=False``, which a PAW wedge refuses too."""
    with pytest.raises(NotImplementedError, match=REFUSAL):
        electrostriction(paw_wedge, None, elastic=elastic)


def test_the_paw_cell_run_whole_passes():
    calculation = _calculation("si-epsilon-paw-unshifted-nosym")
    assert calculation.paw is not None and not calculation.use_symmetry
    require_a_whole_grid_for_paw(calculation, "the Raman tensors")


def test_a_whole_grid_with_symmetry_on_passes():
    """Gamma alone is a whole grid, so the raw ``becsum`` is the full-zone one."""
    calculation = _calculation(
        "si-epsilon-paw-unshifted",
        lambda text: text.replace(" 2 2 2 0 0 0", " 1 1 1 0 0 0"))
    assert calculation.paw is not None
    assert calculation._becsum_symmetry is not None
    assert len(calculation.system.kpoints.weights) == 1
    require_a_whole_grid_for_paw(calculation, "the Raman tensors")


def test_an_ultrasoft_wedge_passes():
    """Ultrasoft has no one-centre ``D``: its ``becsum`` enters ``F`` only through
    the augmentation charge, which the screening term's split already averages."""
    calculation = _calculation(
        "si-us-nosym",
        lambda text: text.replace("nosym = .true.", "").replace("noinv = .true.", ""))
    assert calculation.paw is None and calculation.is_ultrasoft
    assert calculation._becsum_symmetry is not None
    assert len(calculation.system.kpoints.weights) < 8, "a wedge"
    require_a_whole_grid_for_paw(calculation, "the Raman tensors")
