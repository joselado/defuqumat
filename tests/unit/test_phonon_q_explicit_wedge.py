"""An explicit K_POINTS wedge at equal weights is refused by the phonon at q (audit 2026-11-20c #27).

At ``q != 0`` the ``-k`` partner of a kept ``k`` gives ``conj(drho_{-q})`` and
not ``drho_q``, so a wedge summed at doubled weight is wrong. The guard tested
``use_symmetry`` and ``is_reduced``, and ``is_reduced`` reads a flag only
``KPoints.automatic`` sets, then the spread of the weights. The time-reversal
half of a shifted grid written out by hand has equal weights and no flag, and
under ``nosym`` and ``noinv`` the reader takes it as written, so it ran as the
whole zone. The class of 2026-11-20 #11, at a second site. Setup only.
"""

import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import Calculation
from defuqumat.system.builder import build_system
from defuqumat.system.kpoints import KPoints

pytestmark = pytest.mark.unit

_SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 10.0,
  nosym = .true., noinv = .true.
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
"""


def _calculation(points, pseudo_dir):
    """Silicon on ``points`` (crystal), written as an equal-weight K_POINTS list."""
    rows = "\n".join(" ".join(f"{x:.10f}" for x in p) + " 1.0" for p in points)
    system = build_system(parse_pw_input(
        _SILICON + f"K_POINTS crystal\n{len(points)}\n{rows}\n"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    return Calculation(system, pseudos)


def _cell():
    return build_system(parse_pw_input(
        _SILICON + "K_POINTS automatic\n 2 2 2 1 1 1\n")).cell


def test_the_time_reversal_half_written_out_is_refused(pseudo_dir):
    from defuqumat.response.phononq import require_a_two_sphere_regime

    cell = _cell()
    half = KPoints.automatic((2, 2, 2), (1, 1, 1), cell,
                             rotations=np.eye(3, dtype=int)[None], time_reversal=True)
    points = np.asarray(half.crystal(cell))
    assert len(points) == 4 and np.ptp(np.asarray(half.weights)) == 0.0
    calculation = _calculation(points, pseudo_dir)
    kpoints = calculation.system.kpoints
    assert kpoints.grid is None and kpoints.nk == 4 and not calculation.use_symmetry
    with pytest.raises(NotImplementedError, match="needs the whole k-grid"):
        require_a_two_sphere_regime(calculation, np.array([0.25, 0.0, 0.0]))


def test_the_whole_grid_written_out_still_runs(pseudo_dir):
    from defuqumat.response.phononq import require_a_two_sphere_regime

    cell = _cell()
    whole = KPoints.automatic((2, 2, 2), (1, 1, 1), cell)
    calculation = _calculation(np.asarray(whole.crystal(cell)), pseudo_dir)
    assert calculation.system.kpoints.nk == 8
    require_a_two_sphere_regime(calculation, np.array([0.25, 0.0, 0.0]))
