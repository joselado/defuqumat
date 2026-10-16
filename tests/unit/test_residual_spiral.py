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
def test_a_spiral_with_a_hubbard_u_is_refused():
    """A spinor ``ns`` is complex and the packed state keeps its real part only.

    Measured before the refusal (``notes/incompat/I72.md``): on this cell the
    residual's ``F`` at the mixing loop's fixed point is 2.3e-4 away from it.
    """
    text = (GENERATED / "nii-chain-spiral.in").read_text() + "HUBBARD {atomic}\nU Ni-3d 4.0\n"
    system, pseudos = _system(text)
    calculation = Calculation(system, pseudos)
    assert calculation.spiral and calculation.is_hubbard
    with pytest.raises(NotImplementedError, match="Hubbard U"):
        make_residual(calculation, 34, 1e-10)


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
    # ``run_scf`` hands the Newton state to the mixing loop, which iterates with
    # its own methods until it converges, so a final state compared with the
    # mixer's cannot tell a wrong residual map from a right one: a map with
    # ``m_x, m_y`` scaled by 0.97 passed every assertion below, with the loop
    # taking 10 iterations after the hand-off (``notes/incompat/I72.md``). What
    # tells them apart is that Newton converged on its own and the loop had
    # nothing left to do (1 iteration on all three cells, 10 for the wrong map),
    # and the residual's own ``F`` evaluated at the mixer's fixed point.
    assert newton.solver.converged
    assert newton.iterations == 1
    calculation = Calculation(system, pseudos)
    residual = make_residual(calculation, system.nbnd, 1e-13)
    r, _ = residual.residual(
        residual.pack(mixed.density, mixed.becsum, mixed.ns, None), None)
    size = int(np.prod(residual.shapes[0]))
    fixed_point = float(scf_accuracy(np.asarray(r)[:size].reshape(residual.shapes[0]),
                                     calculation.basis.dense, system.cell))
    assert fixed_point < CONV_THR
    assert newton.total_energy == pytest.approx(mixed.total_energy, abs=TOTAL_ENERGY_RY)
    # The distance between the two converged states is the stopping residual
    # magnified along a soft magnetic direction, not a property of the map:
    # with ``warmup = 0`` it read 1.3e-8 on the hydrogen chain, both solvers
    # converged. So it is bounded loosely here, and the map is tested above.
    apart = float(scf_accuracy(np.asarray(newton.density) - np.asarray(mixed.density),
                               calculation.basis.dense, system.cell))
    assert apart < 100.0 * CONV_THR
    # The moment is bounded by ``dr2`` only weakly: its half is
    # ``V/2 w sum_G |dm_G|^2`` with ``w = e2 4 pi / (2 pi)^2``, and Cauchy-Schwarz
    # gives ``|d int|m|| <= int |dm| <= V sqrt(sum_G |dm_G|^2)``. At
    # ``10 CONV_THR`` that is 4.8e-3 on the hydrogen chain, where
    # the two solvers were measured 4.5e-4 apart (``notes/incompat/I72.md``), so
    # a fixed tolerance tighter than the bound would test the stopping rule, not
    # the fixed point. The bound holds for the part of ``dm`` the resident
    # sphere carries; the transverse components it cuts are in
    # ``absolute_magnetization`` and not in ``dr2``, and the 10x margin measured
    # is what covers them.
    volume = float(system.cell.volume)
    weight = 2.0 * 4.0 * np.pi / (2.0 * np.pi) ** 2
    bound = volume * np.sqrt(10.0 * CONV_THR / (0.5 * volume * weight))
    assert abs(newton.absolute_magnetization - mixed.absolute_magnetization) < bound
    # ``_weights`` is the residual's traceable copy of the occupations; on a
    # spiral the eigenvalues are per unshifted k and so are the weights.
    eigenvalues = np.asarray(mixed.eigenvalues_by_spin)
    expected, _ = calculation.occupations(eigenvalues)
    assert np.allclose(_weights(calculation, eigenvalues), np.asarray(expected), atol=1e-14)
