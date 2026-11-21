"""The relaxed geometry of a pw.x output, read as ``output_tau`` writes it.

Two defects of the reference parser (audit 2026-11-20c #11 and #12), each
checked on a committed QE benchmark that shows it and on a short fixture:

* ``output_tau`` prints the final cell in the input card's units, ``(alat=
  ...)``, ``(bohr)`` or ``(angstrom)``. Only the first was read, so a vc-relax
  in bohr or angstrom gave ``final_cell = None`` and its crystal positions were
  scaled by the *starting* cell.
* A row is ``(A6, 3X, 3F20.10)``. A search for floats read the digit of a
  label such as ``Ti1`` as the first coordinate.

The expected numbers are transcribed by hand from the files, as in
``test_qeref.py``.
"""

import numpy as np
import pytest

from defuqumat.io import read_qe_output
from defuqumat.units import ANGSTROM_TO_BOHR

pytestmark = pytest.mark.unit


def test_a_final_cell_in_angstrom_is_read(committed_benchmark):
    """``paw-vcbfgs``: ``CELL_PARAMETERS (angstrom)`` then crystal positions."""
    ref = read_qe_output(committed_benchmark("pw_pawatom", "paw-vcbfgs.in"))
    a = 2.892189013                     # the printed final rows, in angstrom
    printed = np.array([[0.0, a, a], [a, 0.0, a], [a, a, 0.0]])
    assert ref.final_cell is not None
    assert np.allclose(ref.final_cell, printed * ANGSTROM_TO_BOHR, rtol=0, atol=1e-12)
    # the block's own "new unit-cell volume = 326.51761 a.u.^3"
    assert abs(np.linalg.det(ref.final_cell)) == pytest.approx(326.51761, abs=1e-5)
    # Ge at (1/4, 1/4, 1/4) of the *final* cell; the starting cell
    # (2.893335939 angstrom) puts it 1.1e-3 bohr away on each axis
    expected = 0.25 * printed.sum(axis=0) * ANGSTROM_TO_BOHR
    assert np.allclose(ref.final_positions[1], expected, rtol=0, atol=1e-10)
    assert np.allclose(ref.final_positions[0], 0.0, atol=1e-12)


_HEADER = """\
     lattice parameter (alat)  =      10.0000  a.u.
     crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000   0.000000   0.000000 )
               a(2) = (   0.000000   1.000000   0.000000 )
               a(3) = (   0.000000   0.000000   1.000000 )
"""


def _output(tmp_path, final_block):
    path = tmp_path / "relax.out"
    path.write_text(_HEADER + "\nBegin final coordinates\n" + final_block
                    + "End final coordinates\n")
    return read_qe_output(path)


def test_a_final_cell_in_bohr_is_read(tmp_path):
    ref = _output(tmp_path, """\
     new unit-cell volume =   1200.00000 a.u.^3 (   177.82254 Ang^3 )

CELL_PARAMETERS (bohr)
  10.000000000   0.000000000   0.000000000
   0.000000000  10.000000000   0.000000000
   0.000000000   0.000000000  12.000000000

ATOMIC_POSITIONS (crystal)
Si            0.0000000000        0.0000000000        0.5000000000
""")
    assert np.allclose(ref.final_cell, np.diag([10.0, 10.0, 12.0]), atol=0)
    # on the starting cell (10 bohr along z) this is 5.0, not 6.0
    assert np.allclose(ref.final_positions, [[0.0, 0.0, 6.0]], atol=1e-12)


def test_a_cell_printed_in_units_it_cannot_read_is_an_error(tmp_path):
    """Not a silent fall back to the starting cell."""
    with pytest.raises(ValueError, match="CELL_PARAMETERS"):
        _output(tmp_path, """\
CELL_PARAMETERS (nanometre)
   1.000000000   0.000000000   0.000000000
   0.000000000   1.000000000   0.000000000
   0.000000000   0.000000000   1.200000000

ATOMIC_POSITIONS (crystal)
Si            0.0000000000        0.0000000000        0.5000000000
""")


def test_a_fixed_cell_relax_still_uses_the_starting_axes(tmp_path):
    """No cell in the block is a fixed-cell run, and the start cell is right."""
    ref = _output(tmp_path, """\
ATOMIC_POSITIONS (crystal)
Si            0.0000000000        0.0000000000        0.5000000000
""")
    assert ref.final_cell is None
    assert np.allclose(ref.final_positions, [[0.0, 0.0, 5.0]], atol=1e-12)


def test_a_digit_in_a_species_label_is_not_a_coordinate(committed_benchmark):
    """``oscdft_pw`` relax: ``Ti1  1.89  1.89  4.755`` in angstrom."""
    ref = read_qe_output(committed_benchmark("oscdft_pw", "relax.in.args=2_scf.oscdft.in"))
    assert ref.final_positions.shape == (12, 3)
    assert np.allclose(ref.final_positions[1], np.array([1.89, 1.89, 4.755]) * ANGSTROM_TO_BOHR,
                       rtol=0, atol=1e-12)
    # a row with a plain label beside it reads as before
    assert np.allclose(ref.final_positions[2],
                       np.array([0.0, 1.89, 2.3343170435]) * ANGSTROM_TO_BOHR,
                       rtol=0, atol=1e-12)


def test_a_labelled_row_with_if_pos_flags(tmp_path):
    """``(A6, 3X, 3F20.10, 1X, 3I4)``: the flags are not coordinates either."""
    ref = _output(tmp_path, """\
ATOMIC_POSITIONS (bohr)
Fe1           0.1000000000        0.2000000000        0.3000000000   0   1   1
Fe2          -0.4000000000        0.5000000000        0.6000000000
""")
    assert np.allclose(ref.final_positions, [[0.1, 0.2, 0.3], [-0.4, 0.5, 0.6]], atol=0)
