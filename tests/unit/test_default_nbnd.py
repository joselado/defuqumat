"""``default_nbnd`` is ``setup.f90``'s rule, line for line (audit 2026-11-15, ``driver-a`` F1, N1).

The expected counts are worked by hand from ``PW/src/setup.f90:367-390`` and
``Modules/electrons_base.f90:set_nelup_neldw``: ``degspin = 2``, Fortran's
``NINT`` (half away from zero), the unconstrained split ``nelup = INT(nelec +
1)/2`` in every regime, the metallic ``MAX(NINT(1.2 ...), nbnd + 4)``, and the
spinor doubling last. The earlier form used Python's ``round`` and no split
outside ``nspin = 2``, and gave the counts in the comments.
"""

import pytest

from defuqumat.scf.driver import default_nbnd

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("nelec", "occupations", "noncolin", "expected"), [
    (8, "fixed", False, 4),
    (8, "smearing", False, 8),
    (5, "fixed", False, 3),       # was 2: round(2.5) = 2
    (5, "smearing", False, 7),    # was 6
    (9, "fixed", True, 10),       # was 8, one band short of nine electrons
    (9, "smearing", True, 18),    # was 16, the cobalt spinor of PLAN.md
    (13, "fixed", True, 14),      # was 12
    (1, "fixed", False, 1),
    (4.5, "smearing", False, 6),  # a fractional charge: nelup = neldw = 2.25, NINT 2, + 4
])
def test_the_default_band_count_is_setup_f90s(nelec, occupations, noncolin, expected):
    assert default_nbnd(nelec, occupations, noncolin=noncolin) == expected


def test_a_fixed_moment_decides_by_the_fuller_channel():
    """Eight electrons split 5/3 need five bands, and a smearing 20 per cent of the fuller one."""
    assert default_nbnd(8, "fixed", 5.0, 3.0) == 5
    assert default_nbnd(8, "smearing", 5.0, 3.0) == 9
