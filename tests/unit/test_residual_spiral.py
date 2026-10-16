"""The residual (Newton-Krylov) solver on a spin spiral (``notes/INCOMPATIBILITIES.md`` I72).

``make_residual`` refused a spiral with one line and no reason. Read against the
mixing loop, nothing in :meth:`ScfResidual.step` is spiral-specific: the two
spheres at ``k +- q/2``, the rotated-frame density and the displaced
augmentation all live inside the ``Calculation`` methods the loop calls too, so
the residual's map is the loop's map and the check is that both land on the
same fixed point. That is the slow test below, on a norm-conserving, an
ultrasoft and a PAW chain; the fast tests pin the packing and the two sibling
refusals that must keep firing on a spiral (a field, and the Sternheimer
Jacobian).
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation, run_scf
from defuqumat.scf.potential import scf_accuracy
from defuqumat.scf.residual import _weights, make_residual
from defuqumat.system import build_system
from tests.conftest import GENERATED
from tests.tolerances import TOTAL_ENERGY_RY

SPIRAL = GENERATED / "h-chain-spiral.in"
SPIRAL_US = GENERATED / "o-chain-spiral-us.in"
SPIRAL_PAW = GENERATED / "o-chain-spiral-paw.in"

#: Both routes stop at this ``dr2``. The inputs' own 1e-11 is tighter than the
#: comparison needs, and the comparison is of two fixed points, not of a
#: reference.
CONV_THR = 1e-9

PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _system(text: str):
    system = build_system(parse_pw_input(text))
    return system, tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)


@pytest.mark.unit
def test_a_spiral_builds_a_residual_on_its_four_channels():
    """The packed state is the rotated-frame ``(n, m_x, m_y, m_z)`` on the dense grid."""
    system, pseudos = _system(SPIRAL.read_text())
    calculation = Calculation(system, pseudos)
    assert calculation.spiral and calculation.nspin_mag == 4
    residual = make_residual(calculation, 6, 1e-10)
    grid = tuple(calculation.basis.dense.grid)
    assert residual.shapes[0] == (4,) + grid
    assert residual.size == 4 * int(np.prod(grid))


@pytest.mark.unit
def test_a_spiral_under_a_field_is_still_refused():
    """The field guard sits before the spiral's old one and must keep firing on one."""
    text = SPIRAL.read_text().replace(
        "    nbnd = 6\n",
        "    nbnd = 6\n    constrained_magnetization = 'fsm'\n"
        "    fixed_magnetization(1) = 0.3\n",
    )
    system, pseudos = _system(text)
    calculation = Calculation(system, pseudos)
    assert calculation.spiral and calculation.magnetic_field is not None
    with pytest.raises(NotImplementedError, match="magnetic field"):
        make_residual(calculation, 6, 1e-10)


@pytest.mark.unit
def test_the_sternheimer_jacobian_is_still_refused_on_a_spiral():
    """``jvp='sternheimer'`` has no spiral response; it refuses before computing."""
    system, pseudos = _system(SPIRAL.read_text())
    calculation = Calculation(system, pseudos)
    residual = make_residual(calculation, 6, 1e-10)
    x = residual.pack(calculation.starting_density(), calculation.starting_becsum())
    with pytest.raises(NotImplementedError, match="sternheimer"):
        residual.sternheimer_jacobian(x, None)


@pytest.fixture
def _clear_compiled():
    yield
    import jax

    jax.clear_caches()


@lru_cache(maxsize=2)
def _both(path: Path):
    """The mixing loop and Newton-Krylov from the same start, at ``CONV_THR``."""
    system, pseudos = _system(path.read_text())
    mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                    conv_thr=CONV_THR, mixing_beta=0.3, max_iterations=200)
    newton = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos),
        conv_thr=CONV_THR, mixing_beta=0.3, max_iterations=200,
        scf_solver="newton-krylov",
        scf_solver_options={"forcing": 0.5, "warmup": 6},
    )
    return system, pseudos, mixed, newton


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.parametrize("path", [SPIRAL, SPIRAL_US, SPIRAL_PAW],
                         ids=["norm-conserving", "ultrasoft", "paw"])
def test_newton_krylov_finds_the_mixer_s_fixed_point_on_a_spiral(path, _clear_compiled):
    """The same state by two solvers: energy, density in ``dr2`` and the moment.

    Measured on D22 (``notes/incompat/I72.md``) by a harness doing this same
    computation.
    """
    system, pseudos, mixed, newton = _both(path)
    assert mixed.converged and newton.converged
    assert newton.solver is not None and newton.solver.steps > 0
    assert newton.total_energy == pytest.approx(mixed.total_energy, abs=TOTAL_ENERGY_RY)
    calculation = Calculation(system, pseudos)
    apart = float(scf_accuracy(np.asarray(newton.density) - np.asarray(mixed.density),
                               calculation.basis.dense, system.cell))
    assert apart < 10.0 * CONV_THR
    # The moment is bounded by ``dr2`` only weakly: its half is
    # ``V/2 w sum_G |dm_G|^2`` with ``w = e2 4 pi / (2 pi)^2``, and Cauchy-Schwarz
    # gives ``|d int|m|| <= int |dm| <= V sqrt(sum_G |dm_G|^2)``. At the
    # ``10 CONV_THR`` asserted above that is 4.8e-3 on the hydrogen chain, where
    # the two solvers were measured 4.5e-4 apart (``notes/incompat/I72.md``), so
    # a fixed tolerance tighter than the bound would test the stopping rule, not
    # the fixed point.
    volume = float(system.cell.volume)
    weight = 2.0 * 4.0 * np.pi / (2.0 * np.pi) ** 2
    bound = volume * np.sqrt(10.0 * CONV_THR / (0.5 * volume * weight))
    assert abs(newton.absolute_magnetization - mixed.absolute_magnetization) < bound
    # ``_weights`` is the residual's traceable copy of the occupations; on a
    # spiral the eigenvalues are per unshifted k and so are the weights.
    eigenvalues = np.asarray(mixed.eigenvalues_by_spin)
    expected, _ = calculation.occupations(eigenvalues)
    assert np.allclose(_weights(calculation, eigenvalues), np.asarray(expected), atol=1e-14)
