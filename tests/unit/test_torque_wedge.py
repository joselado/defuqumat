"""A torque on a reduced k-set (audit 2026-11-15, ``magnetism-response`` F1; audit 2026-11-20c #13).

The refusal of a wedge fired only when the texture turned, so the default
rotation and ``angle = 0`` differentiated the wedge sum of the moment's
magnetic group. That check is on the k-set, before any solve.

The wedge that ``with_turning_moments`` builds, the identity and the inversion
without time reversal, is the one a torque may be summed on, and until audit
2026-11-20c #13 a turned orientation refused it with "needs nosym" although the
wedge guard and the guide offered it. Two tests here diagonalise: they hold the
torque on that wedge (10 points) to the torque on the ``nosym`` grid (18
points), on tetragonal cobalt at ``ecutwfc = 12`` from the starting density,
which is enough because the statement is an identity of the k-sum at any fixed
inversion-symmetric potential. The other tests are setup only, with the
fixed-density solve replaced by a sentinel.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.system.builder import build_system
from defuqumat.system.kpoints import is_reduced
from defuqumat.workflows.anisotropy import _require_a_whole_grid_for_a_torque

pytestmark = pytest.mark.unit

CASE = Path(__file__).resolve().parents[1] / "data" / "qe" / "co-tetragonal-anisotropy-soc.in"
NICKEL = CASE.with_name("ni-tetragonal-relaxed-mae-paw.in")

#: An Euler triple off every symmetry element of the cell, so that the three
#: components of the torque are all there to compare.
EULER = (0.4, 0.7, 0.3)


class _Solved(Exception):
    """Raised where the fixed-density solve would start."""


def _system(nosym: bool, light: bool = False):
    text = CASE.read_text()
    if not nosym:
        text = text.replace("nosym = .true.,", "").replace("nosym = .true.", "")
    if light:
        heavy = "ecutwfc = 20.0, ecutrho = 160.0"
        assert heavy in text
        text = text.replace(heavy, "ecutwfc = 12.0, ecutrho = 96.0")
    return build_system(parse_pw_input(text))


def _pseudos(system, pseudo_dir):
    return tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)


@pytest.fixture
def solve_stops(monkeypatch):
    from defuqumat.workflows import anisotropy

    def sentinel(*args, **options):
        raise _Solved()

    monkeypatch.setattr(anisotropy, "fixed_density_states", sentinel)


@pytest.fixture(scope="module")
def cobalt(pseudo_dir):
    """The whole grid, the turning wedge, and one collinear density for both."""
    from defuqumat.scf.driver import Calculation

    whole = _system(nosym=True, light=True)
    wedge = _system(nosym=False, light=True).with_turning_moments()
    assert (whole.kpoints.nk, wedge.kpoints.nk) == (18, 10)
    pseudos = _pseudos(whole, pseudo_dir)
    # A collinear density along z, as the scalar-relativistic leg hands over.
    noncollinear = Calculation(whole, pseudos).starting_density()
    density = jnp.stack([(noncollinear[0] + noncollinear[3]) / 2.0,
                         (noncollinear[0] - noncollinear[3]) / 2.0])
    return whole, wedge, pseudos, density


def test_a_torque_on_the_magnetic_wedge_is_refused():
    reduced = _system(nosym=False)
    assert reduced.kpoints.reduced
    with pytest.raises(ValueError, match="wedge of the moment's magnetic group"):
        _require_a_whole_grid_for_a_torque(reduced, "a magnetic torque")
    _require_a_whole_grid_for_a_torque(_system(nosym=True), "a magnetic torque")
    # the identity-and-inversion wedge is complete for an axial vector
    _require_a_whole_grid_for_a_torque(reduced.with_turning_moments(), "a magnetic torque")


def test_a_turned_torque_reaches_the_solve_on_the_turning_wedge(solve_stops, pseudo_dir):
    """Refused with "needs nosym" until audit 2026-11-20c #13; the wedge does not move."""
    from defuqumat.workflows.anisotropy import (
        _with_quantization_axis,
        _with_rotation,
        rotation_from_euler,
        run_orientation_torque,
        run_torque,
    )

    reduced = _system(nosym=False)
    turning = reduced.with_turning_moments()
    pseudos = _pseudos(reduced, pseudo_dir)
    density = np.zeros((2, 4, 4, 4))
    rotation = rotation_from_euler(*EULER)
    with pytest.raises(_Solved):
        run_torque(turning, pseudos, density)
    with pytest.raises(_Solved):
        run_orientation_torque(turning, pseudos, density, rotation=rotation)
    # The turned system keeps the wedge it had: the same ten points, bitwise.
    for turned in (
        _with_rotation(turning, rotation, turning_wedge=True),
        _with_quantization_axis(turning, (1.0, 0.0, 0.0), turning_wedge=True),
    ):
        assert turned.kpoints.nk == 10 and is_reduced(turned.kpoints)
        assert np.array_equal(np.asarray(turned.kpoints.crystal(turned.cell)),
                              np.asarray(turning.kpoints.crystal(turning.cell)))
        assert np.array_equal(np.asarray(turned.kpoints.weights),
                              np.asarray(turning.kpoints.weights))
    # The wedge of the moment's magnetic group is refused as before.
    with pytest.raises(ValueError, match="needs nosym"):
        run_torque(reduced, pseudos, density)
    with pytest.raises(ValueError, match="needs nosym"):
        run_orientation_torque(reduced, pseudos, density, rotation=rotation)


def test_what_the_turning_wedge_does_not_carry_is_refused_before_a_solve(solve_stops, pseudo_dir):
    """The relaxation, the first-order term and a PAW dataset keep ``nosym``."""
    from defuqumat.workflows.anisotropy import (
        frozen_expectation,
        projector_channels,
        relax_orientation,
        rotation_from_euler,
        run_orientation_torque,
    )

    turning = _system(nosym=False).with_turning_moments()
    pseudos = _pseudos(turning, pseudo_dir)
    density = np.zeros((2, 4, 4, 4))
    density[0] = 1.0
    # ``relax_orientation`` turns one calculation between its steps, which
    # ``Calculation.with_texture`` refuses with symmetry: it used to pay for
    # the first step and stop at the second.
    with pytest.raises(ValueError, match="relax_orientation needs nosym"):
        relax_orientation(turning, pseudos, density)
    with pytest.raises(ValueError, match="needs nosym"):
        frozen_expectation(turning, pseudos, density, direction=(1.0, 0.0, 0.0))
    # ...and on the whole grid the relaxation still reaches its first solve.
    with pytest.raises(_Solved):
        relax_orientation(_system(nosym=True), pseudos, density)

    # A PAW dataset at a turned orientation: the wedge is measured on
    # ultrasoft cells only, so it keeps the refusal it had.
    text = NICKEL.read_text()
    assert "nosym = .true.," in text
    nickel = build_system(parse_pw_input(text.replace("nosym = .true.,", "")))
    nickel = nickel.with_turning_moments()
    assert is_reduced(nickel.kpoints)
    paw = _pseudos(nickel, pseudo_dir)
    assert paw[0].is_paw
    nh = len(projector_channels(paw[0]))
    with pytest.raises(ValueError, match="needs nosym"):
        run_orientation_torque(nickel, paw, density, rotation=rotation_from_euler(*EULER),
                               becsum=(np.zeros((2, 1, nh, nh)),))


def test_the_orientation_torque_on_the_turning_wedge_is_the_whole_grid_s(cobalt):
    """Three components at a general rotation, 10 points against 18.

    Measured here 7.4e-12 Ry/rad on a torque of 8.9e-5, and the band energy to
    8.9e-15 Ry. On other grids of the same cell the two differ by up to 5.7e-11
    (3x3x1), which the bound is three times; the wedge of the unturned moment's
    magnetic group, kept at this rotation, is 3.6e-5 to 1.9e-4 away, as far as
    the torque is large (audit 2026-11-20c #13).
    """
    from defuqumat.workflows.anisotropy import rotation_from_euler, run_orientation_torque

    whole, wedge, pseudos, density = cobalt
    rotation = rotation_from_euler(*EULER)
    reference = run_orientation_torque(whole, pseudos, density, rotation=rotation)
    reduced = run_orientation_torque(wedge, pseudos, density, rotation=rotation)
    assert np.max(np.abs(reference.torque)) > 1.0e-5, "a torque to compare, not a zero"
    assert np.max(np.abs(reduced.torque - reference.torque)) < 2.0e-10
    assert abs(reduced.band_energy - reference.band_energy) < 1.0e-12


def test_the_torque_at_an_angle_on_the_turning_wedge_is_the_whole_grid_s(cobalt):
    """``run_torque`` at its default 45 degrees, 10 points against 18.

    Measured here 4.3e-12 Ry/rad on a torque of 9.8e-5, the band energy to
    1.1e-14 Ry; up to 4.7e-11 on the other grids.
    """
    from defuqumat.workflows.anisotropy import run_torque

    whole, wedge, pseudos, density = cobalt
    reference = run_torque(whole, pseudos, density)
    reduced = run_torque(wedge, pseudos, density)
    assert abs(reference.torque) > 1.0e-5, "a torque to compare, not a zero"
    assert abs(reduced.torque - reference.torque) < 2.0e-10
    assert abs(reduced.band_energy - reference.band_energy) < 1.0e-12
