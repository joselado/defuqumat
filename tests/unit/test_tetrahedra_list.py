"""The tetrahedra are folded onto the list they index (audit 2026-11-20c #1).

The corners of the tetrahedra are grid points folded with the run's group, and
the fold is an index into the list that group reduces the grid to. A k-set that
carries a grid but is another list gave corners on the wrong points with no
error: the whole 4x4x4 grid of fcc aluminium on a symmetric run had its 64-point
list read through a 10-point fold, and a list reduced with time reversal on a
``nosym`` run was read through a 64-point fold, past its end. ``tetra.f90``
stops on both ("cannot remap grid on k-point list"), and so does this now.
Setup only, no SCF.
"""

from pathlib import Path

import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.scf.tetrahedra import build_tetrahedra, tetrahedra_for
from defuqumat.system.builder import build_system
from defuqumat.system.kpoints import KPoints
from defuqumat.workflows.pdos import _projection_tetrahedra

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


def _system(case="al-tetrahedra", extra="", symmetric=False):
    """``case`` with ``extra`` in its ``&system``; ``symmetric`` drops its ``nosym``."""
    text = (CASES / f"{case}.in").read_text()
    if extra:
        text = text.replace("&system", f"&system\n{extra}")
    if symmetric:
        text = text.replace("nosym = .true.,", "").replace("nosym = .true.", "")
    return build_system(parse_pw_input(text))


def _both(system, kpoints):
    """The DOS's tetrahedra and the PDOS's, on ``kpoints`` with the run's triple."""
    group = system.symmetry_group(nosym=system.nosym)
    return (
        lambda: tetrahedra_for("tetrahedra", kpoints, group, system.cell,
                               system.grid_symmetry()),
        lambda: _projection_tetrahedra("tetrahedra", kpoints, group, system.cell,
                                       system.grid_symmetry()),
    )


def test_the_whole_grid_on_a_symmetric_run_is_refused():
    system = _system()
    assert system.kpoints.nk == 10 and system.kpoints.reduced
    whole = KPoints.automatic(system.kpoints.grid, system.kpoints.shift,
                              system.cell, precision=system.kpoints.precision)
    assert whole.nk == 64
    for build in _both(system, whole):
        with pytest.raises(ValueError, match="cannot remap grid"):
            build()


def test_a_time_reversal_wedge_on_a_nosym_run_is_refused():
    system = _system(extra="    nosym = .true.,")
    assert system.kpoints.nk == 64
    halved = KPoints.automatic(system.kpoints.grid, system.kpoints.shift,
                               system.cell, rotations=np.eye(3, dtype=int)[None],
                               time_reversal=True)
    assert halved.nk == 32
    for build in _both(system, halved):
        with pytest.raises(ValueError, match="cannot remap grid"):
            build()


def test_the_calculation_refuses_a_whole_grid_on_a_symmetric_run(pseudo_dir):
    """``Calculator.with_kpoints`` of the unreduced grid: the SCF's own table."""
    import dataclasses

    from defuqumat.pseudo import read_upf
    from defuqumat.scf.driver import Calculation
    from defuqumat.system.kpoints import for_spin

    system = _system()
    whole = KPoints.automatic(system.kpoints.grid, system.kpoints.shift,
                              system.cell, precision=system.kpoints.precision)
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    moved = dataclasses.replace(system, kpoints=for_spin(whole, system.nspin))
    with pytest.raises(ValueError, match="cannot remap grid"):
        Calculation(moved, pseudos).tetrahedron_table()


@pytest.mark.parametrize(("case", "extra", "symmetric"), [
    ("al-tetrahedra", "", False),
    ("al-tetrahedra", "    nosym = .true.,", False),
    ("alas-piezo", "    noinv = .true.,", False),
    ("co-tetragonal-anisotropy-soc", "", True),
])
def test_the_runs_own_k_set_folds_as_before(case, extra, symmetric):
    """The regime next to the refusal: the run's own list, with time reversal,
    without it, and a magnetic spinor whose wedge is its magnetic group's. The
    corners are the ones ``build_tetrahedra`` folds on its own, Bloechl's for
    the DOS and the linear ones for the PDOS, and every listed point is a
    corner of something."""
    system = _system(case, extra, symmetric)
    if symmetric:
        assert system.kpoints.reduced
    rotations, time_reversal, t_rev = system.grid_symmetry()
    if rotations is None:
        rotations, time_reversal, t_rev = np.eye(3, dtype=int)[None], False, None
    for kind, build in zip(("bloechl", "linear"), _both(system, system.kpoints)):
        reference = np.asarray(build_tetrahedra(
            kind, system.kpoints.grid, system.kpoints.shift or (0, 0, 0),
            rotations, np.asarray(system.cell.bg_2pi_alat),
            time_reversal=time_reversal, t_rev=t_rev).corners)
        corners = np.asarray(build().corners)
        np.testing.assert_array_equal(corners, reference)
        assert set(np.unique(corners)) == set(range(system.kpoints.nk))
