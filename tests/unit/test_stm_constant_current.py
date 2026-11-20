"""The constant-current scan: which way the tip is withdrawn, and how far.

A synthetic, periodic tunnelling density stands in for a slab: it decays as
``exp(-2 kappa d)`` with ``d`` the periodic distance from a sheet at crystal
coordinate 1/2 along the stacking axis, times a lateral corrugation. The
log-linear inversion is exact on an exponential, so the height at which the
scan reaches the set-point is known in closed form, and nothing needs an SCF.

* audit 2026-11-20c #10: the ``height`` shortcut with ``axis=1`` had its
  normal along ``-a2``, so the scan walked into the slab.
* audit 2026-11-20c #9: with no ``heights`` the scan ran over exactly one
  period, so its last plane was its first one again. No fixed fraction of the
  period is safe on a slab, so ``heights`` is now required, and a scan of one
  period is refused.
"""

import numpy as np
import pytest

from defuqumat.stm.plane import plot_plane
from defuqumat.system.cell import Cell
from defuqumat.workflows import stm, ultracell
from defuqumat.workflows.stm import _constant_current, _plane

pytestmark = pytest.mark.unit

KAPPA = 1.0


def _slab(axis, length):
    """``sample(points)`` of a sheet at 1/2 along ``axis``, corrugated along the next axis."""
    lateral = (axis + 1) % 3

    def sample(points):
        points = np.asarray(points, dtype=float)
        s, x = points[:, axis], points[:, lateral]
        d = np.abs((s % 1.0) - 0.5) * length          # periodic distance, bohr
        return (1.0 + 0.3 * np.cos(2.0 * np.pi * x)) * np.exp(-2.0 * KAPPA * d)

    return sample


def _exact(geometry, axis, start, length, current):
    """The height above the tip plane where the sheet's density is ``current``."""
    lateral = (axis + 1) % 3
    corrugation = 1.0 + 0.3 * np.cos(2.0 * np.pi * geometry.points[..., lateral])
    d0 = abs(start - 0.5) * length
    return (np.log(corrugation) - np.log(current)) / (2.0 * KAPPA) - d0


def _cartesian_normal(geometry, cell):
    return np.asarray(geometry.normal) @ np.asarray(cell.at)


RIGHT = np.diag([12.0, 14.0, 30.0])
LEFT = np.diag([12.0, 14.0, -30.0])                  # det < 0
TILTED = np.array([[12.0, 0.0, 0.0], [3.0, 14.0, 0.0], [1.0, -2.0, 30.0]])


@pytest.mark.parametrize("vectors", [RIGHT, LEFT, TILTED], ids=["right", "left", "tilted"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_the_shortcut_normal_points_along_its_axis(vectors, axis):
    cell = Cell.from_vectors(vectors)
    geometry = _plane(cell, 0.75, axis, None, (4, 4))
    normal = _cartesian_normal(geometry, cell)
    assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-12)
    assert normal @ vectors[axis] > 0.0
    # and it is still perpendicular to the plane
    for edge in (geometry.edge1 - geometry.origin, geometry.edge2 - geometry.origin):
        assert abs(normal @ (edge @ vectors)) < 1e-10


def test_on_a_diagonal_cell_axis_1_scans_along_plus_b():
    cell = Cell.from_vectors(RIGHT)
    geometry = _plane(cell, 0.75, 1, None, (4, 4))
    assert np.allclose(geometry.normal, [0.0, 1.0 / 14.0, 0.0], atol=1e-15)


def test_turning_the_normal_keeps_the_image_layout():
    """Only the normal turns: points, in-plane coordinates and edges are as before."""
    cell = Cell.from_vectors(TILTED)
    geometry = _plane(cell, 0.75, 1, None, (5, 3))
    plain = plot_plane(cell, (0, 0.75, 0), (1, 0.75, 0), (0, 0.75, 1), (5, 3))
    assert np.array_equal(geometry.points, plain.points)
    assert np.array_equal(geometry.coordinates, plain.coordinates)
    assert np.array_equal(geometry.edge1, plain.edge1)
    assert np.array_equal(geometry.edge2, plain.edge2)
    assert np.array_equal(geometry.normal, -plain.normal)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_scan_from_the_shortcut_finds_the_surface_below_the_tip(axis):
    """Tip plane at 3/4 above a sheet at 1/2: the scan goes up, into the vacuum."""
    lengths = (24.0, 28.0, 30.0)
    cell = Cell.from_vectors(np.diag(lengths))
    length, start = lengths[axis], 0.75
    current = float(np.exp(-2.0 * KAPPA * (abs(start - 0.5) * length + 2.0)))
    geometry = _plane(cell, start, axis, None, (6, 5))
    image = _constant_current(_slab(axis, length), geometry, cell, current, (0.0, 5.0), 60)
    expected = _exact(geometry, axis, start, length, current)
    assert np.isfinite(image.heights_bohr).all()
    assert np.allclose(image.heights_bohr, expected, rtol=0, atol=1e-9)


def test_a_scan_with_no_range_is_refused():
    """The old default, one period, gave ``nan`` at every pixel of a tip above the set-point."""
    cell = Cell.from_vectors(RIGHT)
    geometry = _plane(cell, 0.6, 2, None, (4, 4))
    with pytest.raises(ValueError, match="needs heights"):
        _constant_current(_slab(2, 30.0), geometry, cell, 1e-6, None, 60)


def test_a_scan_of_exactly_one_period_is_refused():
    """Its last plane is the periodic image of its first."""
    cell = Cell.from_vectors(RIGHT)
    geometry = _plane(cell, 0.6, 2, None, (4, 4))
    with pytest.raises(ValueError, match="periodic image"):
        _constant_current(_slab(2, 30.0), geometry, cell, 1e-6, (0.0, 30.0), 60)
    with pytest.raises(ValueError, match="periodic image"):
        _constant_current(_slab(2, 30.0), geometry, cell, 1e-6, (2.0, 32.0), 60)


def test_a_scan_that_ends_in_the_vacuum_still_runs():
    """The regime next to the guard: a range given, from 3 bohr out to the vacuum midpoint."""
    cell = Cell.from_vectors(RIGHT)
    start, length = 0.6, 30.0
    current = float(np.exp(-2.0 * KAPPA * (abs(start - 0.5) * length + 2.0)))
    geometry = _plane(cell, start, 2, None, (6, 5))
    image = _constant_current(_slab(2, length), geometry, cell, current, (0.0, 12.0), 120)
    assert np.allclose(image.heights_bohr, _exact(geometry, 2, start, length, current),
                       rtol=0, atol=1e-9)


class _Stop(Exception):
    pass


class _States:
    """Ultracell states that stop the test the moment they are read."""

    def __getattr__(self, name):
        raise _Stop


def test_the_entries_refuse_before_the_states_are_used(monkeypatch):
    """``run_stm`` before the band solve, ``run_ultracell_stm`` before the density."""
    def stop(*args, **kwargs):
        raise _Stop

    monkeypatch.setattr(stm, "refuse_an_ultracell_result", lambda *a, **k: None)
    monkeypatch.setattr(stm, "_refuse_what_has_no_fermi_level", stop)
    with pytest.raises(ValueError, match="needs heights"):
        stm.run_stm(None, None, None, height=0.6, mode="constant-current", current=1e-6)
    with pytest.raises(_Stop):           # with a range it goes on to the next check
        stm.run_stm(None, None, None, height=0.6, mode="constant-current", current=1e-6,
                    heights=(0.0, 5.0))

    monkeypatch.setattr(ultracell, "_states_of", lambda *a, **k: _States())
    monkeypatch.setattr(ultracell, "_refuse_what_has_no_tip_energy", lambda *a, **k: None)
    with pytest.raises(ValueError, match="needs heights"):
        ultracell.run_ultracell_stm(None, None, None, height=0.6, mode="constant-current",
                                    current=1e-6)
    with pytest.raises(_Stop):           # with a range it goes on to read the states
        ultracell.run_ultracell_stm(None, None, None, height=0.6, mode="constant-current",
                                    current=1e-6, heights=(0.0, 5.0))
