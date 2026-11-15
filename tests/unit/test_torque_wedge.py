"""A torque at the system's own orientation on a reduced k-set (audit 2026-11-15, ``magnetism-response`` F1).

The refusal of a wedge fired only when the texture turned, so the default
rotation and ``angle = 0`` differentiated the wedge sum of the moment's
magnetic group. Setup only: the check is on the k-set, before any solve.
"""

from pathlib import Path

import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.system.builder import build_system
from defuqumat.workflows.anisotropy import _require_a_whole_grid_for_a_torque

pytestmark = pytest.mark.unit

CASE = Path(__file__).resolve().parents[1] / "data" / "qe" / "co-tetragonal-anisotropy-soc.in"


def _system(nosym: bool):
    text = CASE.read_text()
    if not nosym:
        text = text.replace("nosym = .true.,", "").replace("nosym = .true.", "")
    return build_system(parse_pw_input(text))


def test_a_torque_on_the_magnetic_wedge_is_refused():
    reduced = _system(nosym=False)
    assert reduced.kpoints.reduced
    with pytest.raises(ValueError, match="wedge of the moment's magnetic group"):
        _require_a_whole_grid_for_a_torque(reduced, "a magnetic torque")
    _require_a_whole_grid_for_a_torque(_system(nosym=True), "a magnetic torque")
    # the identity-and-inversion wedge is complete for an axial vector
    _require_a_whole_grid_for_a_torque(reduced.with_turning_moments(), "a magnetic torque")
