"""The structure factors refuse a spin spiral (audit 2026-11-20c #8).

A spiral's transverse magnetization is stored in the frame that turns with
``q``, and the laboratory one, which neutron diffraction sees, sits on the
satellites ``H -/+ q`` rather than on the reflections ``H`` the table holds.
On the hydrogen chain of the spirals guide the turning frame gives 0.54 Bohr
magnetons a cell at ``H = 0`` where the laboratory value is zero (measured in
the fix log). No SCF is run: the guard fires before the density is read, and
the noncollinear chain without a spiral still runs on a synthetic density.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.basis.builder import build_basis
from defuqumat.diffraction.structure_factor import structure_factors_of_field
from defuqumat.io.pwin import parse_pw_input
from defuqumat.system.builder import build_system
from defuqumat.workflows.sfac import run_structure_factors

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[2]
CHAIN = REPO / "docs" / "figures" / "spirals_eq" / "h-chain.in"
PSEUDO = REPO / "tests" / "data" / "pseudo"
SPIRAL_LINE = "spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = 0.25"


def _system(spiral=True):
    text = CHAIN.read_text()
    assert SPIRAL_LINE in text
    if not spiral:
        text = text.replace(SPIRAL_LINE, "")
    return build_system(parse_pw_input(text))


def _density(system, seed=0):
    grid = tuple(int(n) for n in build_basis(system).dense.grid)
    return np.random.default_rng(seed).normal(size=(4,) + grid)


def test_a_spiral_is_refused_by_name():
    system = _system()
    assert system.spiral
    result = SimpleNamespace(density=_density(system))
    with pytest.raises(NotImplementedError, match="spin spiral"):
        run_structure_factors(system, None, result, hmax=2.0)


class _Stop(Exception):
    pass


def test_the_calculator_refuses_before_the_scf(monkeypatch):
    calculator = Calculator.from_file(CHAIN, pseudo_dir=PSEUDO)

    def scf(*args, **kwargs):
        raise _Stop("the SCF was reached")

    monkeypatch.setattr(Calculator, "_ground_state", scf)
    with pytest.raises(NotImplementedError, match="spin spiral"):
        calculator.get_structure_factors(hmax=2.0)


def test_the_same_chain_without_a_spiral_still_runs():
    """Noncollinear and magnetic, the regime next to the guard."""
    system = _system(spiral=False)
    assert not system.spiral and system.noncolin
    density = _density(system, seed=1)
    factors = run_structure_factors(system, None, SimpleNamespace(density=density),
                                    hmax=2.0, transform=None, reduce=False)
    expected = structure_factors_of_field(density[1:], system.cell, factors.vectors.miller)
    assert factors.magnetization.shape == (len(factors.vectors.miller), 3)
    assert np.allclose(factors.magnetization, np.asarray(expected).T, rtol=0, atol=1e-12)
