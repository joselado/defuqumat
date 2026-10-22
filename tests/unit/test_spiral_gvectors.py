"""The laboratory sphere a spin spiral keeps its transverse pair on (``OPEN.md`` Part XXI item 1).

:func:`~defuqumat.basis.gvectors.displaced_gvectors` cuts ``{G : |G + shift| <= G_cut}``
on the resident box, with the smooth sphere as its prefix. These are host-side
checks of the set itself; whether the spiral then equals its unfolded supercell
is ``tests/regression/test_spin_spirals_augmented.py``.
"""

from functools import lru_cache

import numpy as np
import pytest

from defuqumat.basis.builder import build_basis
from defuqumat.basis.gvectors import GVectors, displaced_gvectors
from defuqumat.io.pwin import parse_pw_input
from defuqumat.system import build_system
from defuqumat.system.spiral import spiral_cartesian
from tests.conftest import GENERATED


@lru_cache(maxsize=2)
def _cobalt(ecutrho: float):
    """The cobalt helix's cell, which has a separate smooth grid at 160 Ry."""
    text = (GENERATED / "co-helix4-spiral.in").read_text().replace(
        "ecutrho = 160.0", f"ecutrho = {ecutrho}")
    system = build_system(parse_pw_input(text))
    basis = build_basis(system)
    q = np.asarray(spiral_cartesian(system.spiral_q, system.cell)) * float(system.cell.tpiba)
    return system, basis, q


def test_at_no_shift_the_sets_are_the_resident_ones_index_for_index():
    """``shift = 0`` is the resident dense set and its smooth prefix, Miller index for Miller index.

    The ``q = 0`` spiral's energy test is a 1e-10 tolerance, which a
    reordering inside a shell would pass; this is the identity itself.
    """
    system, basis, _ = _cobalt(160.0)
    assert basis.doublegrid
    dense, smooth = displaced_gvectors(system.cell, basis.dense, np.zeros(3),
                                       smooth=basis.smooth)
    assert np.array_equal(np.asarray(dense.miller), np.asarray(basis.dense.miller))
    assert np.array_equal(np.asarray(smooth.miller), np.asarray(basis.smooth.miller))
    assert dense.grid == basis.dense.grid and smooth.grid == basis.smooth.grid


@pytest.mark.parametrize("ecutrho", [80.0, 160.0])
def test_the_displaced_set_is_the_laboratory_sphere(ecutrho):
    """Every G with ``|G - q|^2 <= ecutrho`` and no other, ``G = 0`` first, the smooth sphere its prefix.

    Counted against a brute-force enumeration over a box three times as wide,
    so a member the enumeration's range missed would show as a count short.
    The resident and the displaced counts differ (cobalt at 160 Ry: 4325
    against 4344), which is the shell the change is about.
    """
    system, basis, q = _cobalt(ecutrho)
    dense, smooth = displaced_gvectors(system.cell, basis.dense, -q, smooth=basis.smooth)
    bg = np.asarray(system.cell.bg)

    ranges = [np.arange(-n, n + 1) for n in basis.dense.grid]
    every = np.stack([a.ravel() for a in np.meshgrid(*ranges, indexing="ij")], axis=1)
    lab = np.sum((every @ bg - q) ** 2, axis=1)
    expected = {tuple(m) for m in every[lab <= basis.dense.ecut]}
    got = [tuple(int(v) for v in m) for m in np.asarray(dense.miller)]
    assert set(got) == expected and len(got) == len(expected)
    assert got[0] == (0, 0, 0)
    assert dense.ngm != basis.dense.ngm

    g2 = np.sum((np.asarray(dense.miller) @ bg - q) ** 2, axis=1)
    inner = g2 <= basis.smooth.ecut
    assert inner[:smooth.ngm].all() and not inner[smooth.ngm:].any()
    assert np.array_equal(np.asarray(smooth.miller), np.asarray(dense.miller)[:smooth.ngm])
    # Members past ``(n - 1) // 2`` are held at the wrapped index, and the
    # point of the check is that no two of them share one.
    assert len(np.unique(np.asarray(dense.fft_index))) == dense.ngm
    assert len(np.unique(np.asarray(smooth.fft_index))) == smooth.ngm


def test_a_box_that_would_fold_two_members_together_is_refused():
    """The guard fires on a box one plane narrower than the sphere spans.

    ``pw.x``'s FFT sizes leave room on every axis of every cell measured (the
    cobalt cell from 80 to 320 Ry at half a reciprocal vector along each axis,
    the oxygen chain at ``q3 = 1/4``, ``0.3`` and ``1/2``, silicon at ``0.3``
    and ``1/2``), so no real input reaches the refusal and the box is narrowed
    by hand: ``2 max|m_3|`` points along ``c`` for a sphere spanning
    ``2 max|m_3| + 1``. One plane wider, the same sphere is held.
    """
    system, basis, q = _cobalt(80.0)
    miller = np.asarray(basis.dense.miller)
    span = int(2 * np.abs(miller[:, 2]).max() + 1)
    held = GVectors(miller=basis.dense.miller, grid=basis.dense.grid[:2] + (span,),
                    ecut=basis.dense.ecut)
    narrow = GVectors(miller=basis.dense.miller, grid=basis.dense.grid[:2] + (span - 1,),
                      ecut=basis.dense.ecut)
    displaced_gvectors(system.cell, held, np.zeros(3))
    with pytest.raises(NotImplementedError, match="share a box index"):
        displaced_gvectors(system.cell, narrow, -q)


def test_g_space_mixing_on_a_magnetic_spiral_keeps_the_pair_on_its_own_sphere(
        pseudo_dir, monkeypatch):
    """``mixing_space = 'g'`` installs the spiral's layout, where it used to refuse (I74).

    Stopped as the layout is built, so nothing is diagonalised: the hydrogen
    chain is norm-conserving at dual 4 and its displaced sphere does not fit
    the box, so ``m'_+`` is kept on the whole box. That the run then reaches
    the real-space fixed point is the slow
    ``tests/unit/test_spiral_mixing_space.py::test_a_spiral_mixed_in_g_reaches_the_real_space_fixed_point``.
    """
    import defuqumat.scf.driver as driver
    from defuqumat.pseudo import read_upf
    from defuqumat.scf import run_scf

    class Reached(Exception):
        pass

    class Probe(driver.SpiralSphereLayout):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            raise Reached(self)

    monkeypatch.setattr(driver, "SpiralSphereLayout", Probe)
    system = build_system(parse_pw_input((GENERATED / "h-chain-spiral.in").read_text()))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    with pytest.raises(Reached) as caught:
        run_scf(system, pseudos, mixing_space="g", max_iterations=1)
    layout = caught.value.args[0]
    assert layout.npair == layout.points and layout.npair_shell == 0
    assert layout.stored_shape == (4, layout.points)


@lru_cache(maxsize=2)
def _oxygen(ecutrho: float):
    """The ultrasoft oxygen chain at ``q = b3/4``, its box and its displaced shift."""
    text = (GENERATED / "o-chain-spiral-us.in").read_text().replace(
        "ecutrho = 200.0", f"ecutrho = {ecutrho}")
    system = build_system(parse_pw_input(text))
    basis = build_basis(system)
    q = np.asarray(spiral_cartesian(system.spiral_q, system.cell)) * float(system.cell.tpiba)
    return system, basis, q


def test_a_real_input_that_folds_is_given_a_wider_box():
    """The oxygen chain at 240 Ry: the resident box folds, the widened one holds (I122).

    ``ecutrho = 240`` puts ``c``'s resident box at 25 points, exactly the
    resident span, and at ``q = b3/4`` the laboratory sphere spans 26, so the
    guard fires on the resident box. The widened box is 27 along ``c`` and
    untouched along ``a`` and ``b``, the dense sphere is the resident one Miller
    index for Miller index, and the smooth box (18 along ``c``) already held
    its sphere and is left as it was. 27 is a quarter of the four-cell
    supercell's own box at this cutoff (108), which is what lets the slow
    ``tests/regression/test_spin_spirals_augmented.py::test_a_widened_box_keeps_the_quarter_turn_identity``
    hold the spiral to its supercell.
    """
    from defuqumat.basis.builder import widen_for_displaced_sphere

    system, basis, q = _oxygen(240.0)
    assert basis.dense.grid == (60, 60, 25) and basis.smooth.grid == (40, 40, 18)
    with pytest.raises(NotImplementedError, match="share a box index"):
        displaced_gvectors(system.cell, basis.dense, -q, smooth=basis.smooth)

    widened = widen_for_displaced_sphere(system, basis, -q)
    assert widened.dense.grid == (60, 60, 27)
    assert widened.smooth.grid == basis.smooth.grid
    assert np.array_equal(np.asarray(widened.smooth.miller), np.asarray(basis.smooth.miller))
    assert np.array_equal(np.asarray(widened.dense.miller), np.asarray(basis.dense.miller))
    dense, smooth = displaced_gvectors(system.cell, widened.dense, -q, smooth=widened.smooth)
    assert len(np.unique(np.asarray(dense.fft_index))) == dense.ngm
    assert len(np.unique(np.asarray(smooth.fft_index))) == smooth.ngm


def test_a_box_that_already_holds_is_not_touched():
    """At 200 Ry the box holds, and the widening hands back the very same basis.

    So every spiral that ran before I122 runs on the box it had, which is what
    keeps the 200 Ry supercell identities and the cobalt helix where they were.
    """
    from defuqumat.basis.builder import widen_for_displaced_sphere

    system, basis, q = _oxygen(200.0)
    assert widen_for_displaced_sphere(system, basis, -q) is basis


@pytest.mark.slow
def test_a_spiral_calculation_runs_on_the_widened_box(pseudo_dir):
    """``Calculation`` widens the box before anything is sampled on it.

    Stopped as the displaced spheres are built, so nothing is diagonalised:
    the spiral's own sets are on (60, 60, 27) and the plane waves on the
    unchanged smooth box.
    """
    import defuqumat.scf.driver as driver
    from defuqumat.pseudo import read_upf

    system, _, _ = _oxygen(240.0)
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)

    class Reached(Exception):
        pass

    real = driver._spiral_gvectors

    def probe(basis, cell, qcart, needed):
        raise Reached(basis, real(basis, cell, qcart, needed))

    import unittest.mock as mock
    with mock.patch.object(driver, "_spiral_gvectors", probe):
        with pytest.raises(Reached) as caught:
            driver.Calculation(system, pseudos)
    basis, (dense, smooth) = caught.value.args
    assert basis.dense.grid == (60, 60, 27) and basis.smooth.grid == (40, 40, 18)
    assert dense.grid == (60, 60, 27) and smooth.grid == (40, 40, 18)
