"""I103: the Berry-phase finite field on an ultrasoft and a PAW dataset, against ``pw.x``.

Si8 (``si8-lelfield.in``'s cell and positions) on ``Si.pbe-n-rrkjus_psl.0.1``
and ``Si.pbe-n-kjpaw_psl.0.1``, two strings of five points, run as the
``pw_electric`` chain: the zero-field state with the operator on, then 0.001
Ry a.u. from it, then 0.002 from that. ``pw.x`` 7.5 is the reference **along
y**: along z its ``compute_qqc`` (``upflib/qvan2.f90``) forms
``|q|^2 = q(1)^2 + q(2)^2 + q(1)^2``, which vanishes for ``q = dk`` along z, so
it evaluates the cross augmentation at ``|q| = 0``. The z outputs are kept to
show that this is what ``pw.x`` does: with ``Q_ij(0)`` put in place of
``Q_ij(dk)`` this code reproduces them.

Measured on D22 at ``conv_thr = 1e-11`` (one sample each, the SCF is
deterministic): total within 5e-9 Ry of ``pw.x`` (its printed digits), ``el_pol``
within 1.7e-6 (US) and 6.6e-6 (PAW), against 4.7e-6 by which ``pw.x``'s own
``el_pol`` moves between its last two iterations; forces within 1e-7 Ry/bohr.
At ``conv_thr = 1e-8`` the same run is 2.3e-7 Ry and 4.6e-4 in ``el_pol``
away, so the agreement is converged in ``conv_thr``. With ``q_ij(+-dk)``
dropped (``cross_off``) ``el_pol`` is 1.1144, 0.134 from ``pw.x``.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defuqumat.forces import compute_forces
from defuqumat.io import read_qe_output
from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation, run_scf
from defuqumat.system import build_system

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: The file's printed digits (8 decimals); measured under 5e-9 Ry.
ENERGY = 3e-8
#: Measured 1.7e-6 (US) and 6.6e-6 (PAW); pw.x moves 4.7e-6 in its last iteration.
DIPOLE = 3e-5
#: Measured under 1e-7 Ry/bohr on both datasets (pw.x prints 8 decimals).
FORCE = 1e-6


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


def _dipoles(name):
    text = (CASES / f"reference.out.{name}").read_text()
    electronic = float(text.rsplit("Electronic Dipole per cell (Ry a.u.)", 1)[1].split()[0])
    ionic = float(text.rsplit("Ionic Dipole per cell (Ry a.u.)", 1)[1].split()[0])
    return electronic, ionic


def _run(case, efield, start=None, setup=None):
    text = (CASES / f"{case}.in").read_text().replace("efield=0.001", f"efield={efield}")
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    if setup is not None:
        setup(calculation.berry_field, calculation)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-11,
                     mixing_beta=0.5, max_iterations=100, starting_from=start)
    assert result.converged
    return calculation, result


@lru_cache(maxsize=2)
def _chain(case):
    _, zero = _run(case, 0.0)
    calculation, field = _run(case, 0.001, start=zero)
    return zero, calculation, field


@pytest.mark.parametrize("tag", ["us", "paw"])
def test_the_augmented_field_is_pw_x(tag):
    case = f"si8-{tag}-lelfield-y"
    zero, calculation, field = _chain(case)
    assert zero.total_energy == pytest.approx(
        read_qe_output(CASES / f"reference.out.{case}-zero").total_energy, abs=ENERGY)
    reference = read_qe_output(CASES / f"reference.out.{case}")
    electronic, ionic = _dipoles(case)
    el, ion, acc = field.berry_dipoles
    print(f"\n{tag}: dE = {field.total_energy - reference.total_energy:.2e}, "
          f"d el_pol = {el - electronic:.2e}")
    assert field.total_energy == pytest.approx(reference.total_energy, abs=ENERGY)
    assert acc == 0.0
    assert ion == pytest.approx(ionic, abs=1e-10)
    assert el == pytest.approx(electronic, abs=DIPOLE)
    forces = compute_forces(calculation, field, method="autodiff").forces
    np.testing.assert_allclose(forces, reference.forces, atol=FORCE)


def test_the_frozen_energy_is_the_total_energy():
    """The force functional with the augmented Berry enthalpy is the SCF total (1.3e-9 measured)."""
    from defuqumat.forces.energy import energy_at, state_from_result

    _, calculation, field = _chain("si8-us-lelfield-y")
    frozen = float(energy_at(calculation, state_from_result(field)))
    assert frozen == pytest.approx(field.total_energy, abs=1e-8)


def test_twice_the_field_is_pw_x():
    """0.002 from the 0.001 state: the response is right at second order too (dE 9e-9, d el 7.8e-6)."""
    _, _, field = _chain("si8-us-lelfield-y")
    _, double = _run("si8-us-lelfield-y", 0.002, start=field)
    reference = read_qe_output(CASES / "reference.out.si8-us-lelfield-y-2")
    electronic, _ = _dipoles("si8-us-lelfield-y-2")
    assert double.total_energy == pytest.approx(reference.total_energy, abs=ENERGY)
    assert double.berry_dipoles[0] == pytest.approx(electronic, abs=DIPOLE)


def test_the_check_tells_the_cross_term_from_its_absence():
    """Without ``q_ij(+-dk)`` ``el_pol`` is 1.1144 against ``pw.x``'s 0.98090 (0.134 off)."""
    zero = _chain("si8-us-lelfield-y")[0]

    def off(field, calculation):
        field.cross_off = True

    _, result = _run("si8-us-lelfield-y", 0.001, start=zero, setup=off)
    electronic, _ = _dipoles("si8-us-lelfield-y")
    assert abs(result.berry_dipoles[0] - electronic) > 0.1


def test_along_z_pw_x_evaluates_the_cross_term_at_q_zero():
    """``pw.x``'s z run is reproduced with ``Q_ij(0)`` for ``Q_ij(dk)`` (measured 5e-9 Ry and
    8e-6 in ``el_pol``), and the right term is 0.022 away from it in ``el_pol``."""
    from defuqumat.topology.augmentation import augmentation_species_at_q

    def at_zero(field, calculation):
        values = augmentation_species_at_q(calculation, np.zeros(3))
        field.cross = (values, values)

    _, zero = _run("si8-us-lelfield", 0.0)
    _, buggy = _run("si8-us-lelfield", 0.001, start=zero, setup=at_zero)
    _, right = _run("si8-us-lelfield", 0.001, start=zero)
    reference = read_qe_output(CASES / "reference.out.si8-us-lelfield")
    electronic, _ = _dipoles("si8-us-lelfield")
    assert buggy.total_energy == pytest.approx(reference.total_energy, abs=ENERGY)
    assert buggy.berry_dipoles[0] == pytest.approx(electronic, abs=DIPOLE)
    assert abs(right.berry_dipoles[0] - electronic) > 1e-2
