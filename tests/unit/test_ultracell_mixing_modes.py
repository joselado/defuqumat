"""The ultracell's mixing modes: Herbst and Levitt's LDOS on the box, and what refuses.

Nothing here runs an SCF. The LDOS preconditioner on the box is the unit cell's
:func:`~defuqumat.scf.mixing.ldos_preconditioner` handed the whole box as its
G-set, and the identity that pins the hand-over is the one its docstring states:
a uniform LDOS ``D`` is Kerker with ``q_TF^2 = 8 pi D``, so on a uniform ``D``
:func:`~defuqumat.ultracell.mixing.box_ldos` must be
:func:`~defuqumat.ultracell.mixing.box_kerker` at that screening to round-off,
and a ``D`` that is not uniform must not be. The fixed-point check against
``'anderson'`` with Kerker is ``tests/regression/test_ultracell.py::
test_the_ldos_mode_reaches_the_kerker_fixed_point`` (slow).
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.units import E2, FPI
from defuqumat.ultracell.driver import run_ultracell
from defuqumat.ultracell.grid import Ultracell
from defuqumat.ultracell.mixing import box_kerker, box_ldos

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "tests" / "data" / "qe"


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    return Calculator.from_file(DATA / "si-ultracell.in", pseudo_dir=pseudo_dir)


def _box(silicon, shape=(2, 1, 1), cell_grid=(8, 8, 8)):
    return Ultracell.build(shape, cell_grid), silicon.system.cell


def _residual(grid, nspin, seed=0):
    """A random residual with its Nyquist planes removed.

    On an even box the plane ``n/2`` holds ``G`` and ``-G`` on one point, where
    ``|G + Q|^2`` is not symmetric, so the two operators take the real part of
    a different non-Hermitian factor there; with those planes in, uniform ``D``
    reads 1.4e-4 relative apart, and with them out 3e-16. Neither is the
    physics, and the fixed point does not depend on either.
    """
    rng = np.random.default_rng(seed)
    field = rng.standard_normal((nspin,) + tuple(grid))
    spectrum = np.fft.fftn(field, axes=(1, 2, 3))
    for axis, n in enumerate(grid):
        if n % 2 == 0:
            index = [slice(None)] * 4
            index[axis + 1] = n // 2
            spectrum[tuple(index)] = 0.0
    return np.real(np.fft.ifftn(spectrum, axes=(1, 2, 3)))


@pytest.mark.parametrize("nspin", [1, 2])
def test_a_uniform_ldos_is_kerker_at_eight_pi_d(silicon, nspin):
    ultracell, cell = _box(silicon)
    grid = ultracell.grid
    shape = (nspin,) + grid
    density = 0.03
    ldos = box_ldos(ultracell, cell, shape, beta=0.4)
    ldos.update_ldos(np.full(grid, density))
    kerker = box_kerker(ultracell, cell, 8.0, shape, beta=0.4,
                        screening=FPI * E2 * density)
    residual = _residual(grid, nspin).ravel()
    ours, theirs = np.asarray(ldos(residual)), np.asarray(kerker(residual))
    assert np.max(np.abs(ours - theirs)) < 1e-12 * np.max(np.abs(theirs))
    # Kerker's first guess is exact on a uniform D, so the solve stops at once.
    assert ldos.solves and all(k == 0 for k, _ in ldos.solves)


def test_a_modulated_ldos_is_not_kerker(silicon):
    """The identity above can fail: the hand-over is not Kerker under another name."""
    ultracell, cell = _box(silicon)
    grid = ultracell.grid
    x = np.arange(grid[0]) / grid[0]
    field = 0.03 * (1.0 + 0.9 * np.cos(2 * np.pi * x))[:, None, None] * np.ones(grid)
    ldos = box_ldos(ultracell, cell, (1,) + grid, beta=0.4)
    ldos.update_ldos(field)
    kerker = box_kerker(ultracell, cell, 8.0, (1,) + grid, beta=0.4,
                        screening=FPI * E2 * float(field.mean()))
    residual = _residual(grid, 1).ravel()
    ours, theirs = np.asarray(ldos(residual)), np.asarray(kerker(residual))
    assert np.max(np.abs(ours - theirs)) > 1e-3 * np.max(np.abs(theirs))
    assert ldos.solves[-1][0] > 0 and ldos.solves[-1][1] < 1e-3
    # the step keeps the electron count, as Kerker does
    assert abs(ours.sum()) < 1e-10 * np.abs(ours).sum()


def test_no_states_at_the_fermi_level_is_the_plain_step(silicon):
    ultracell, cell = _box(silicon)
    grid = ultracell.grid
    ldos = box_ldos(ultracell, cell, (1,) + grid, beta=0.4)
    ldos.update_ldos(np.zeros(grid))
    residual = _residual(grid, 1).ravel()
    assert np.allclose(np.asarray(ldos(residual)), 0.4 * residual, atol=0, rtol=0)


def test_robust_is_still_refused():
    with pytest.raises(NotImplementedError, match="mixing_mode = 'robust'"):
        run_ultracell(None, None, None, (2, 1, 1), mixing_mode="robust")


@pytest.mark.parametrize("kind", ["is_ultrasoft", "is_paw"])
def test_ldos_on_an_augmented_ultracell_is_refused(kind):
    pseudo = SimpleNamespace(is_ultrasoft=False, is_paw=False)
    setattr(pseudo, kind, True)
    with pytest.raises(NotImplementedError, match="ultrasoft or PAW"):
        run_ultracell(None, [pseudo], None, (2, 1, 1), mixing_mode="ldos")


def test_a_pointwise_mixer_under_kerker_is_still_refused():
    pseudo = SimpleNamespace(is_ultrasoft=False, is_paw=False)
    with pytest.raises(ValueError, match="does not take a preconditioner"):
        run_ultracell(None, [pseudo], None, (2, 1, 1), mixing_mode="adaptive")
