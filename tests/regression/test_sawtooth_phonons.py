"""I104: the dynamical matrix at ``Gamma`` under a saw-tooth field, against the forces.

``ph.x`` has no ``tefield`` branch at all (``PHonon/`` and ``LR_Modules/`` never
read it), so there is no reference output to compare with. The check is an
identity instead: a column of the force constants is minus the derivative of
the forces with respect to one coordinate, and the forces under the field are
validated against ``pw.x`` on these same inputs to 3e-7 Ry/bohr
(``tests/regression/test_sawtooth.py``). The central difference agrees with
the column to about 2e-4 Ry/bohr^2, one part in 10^4; below a step of about
0.0035 bohr the difference measures a deterministic roughness of the forces
(about 3.5e-7 Ry/bohr) that ``conv_thr`` does not move, rather than the force
constants.

The cell is QE's ultrasoft CO molecule (``efield-co.in`` and
``efield-co-dipole.in``) at ``eamp = 0.01`` rather than the files' 0.001, so
that the field's own effect on the matrix is larger than the tolerance and the
test can tell the field from no field. With ``dipfield`` the ions' dipole moves
the dipole correction's saw-tooth at a frozen density; the test also asserts
that the matrix misses the forces without that term, which is the term this
item added to ``dV_bare`` (``response/phonon.py:_core_charge_response``).

The numbers each tolerance is set from are in ``notes/incompat/I104.md``.
"""

from functools import lru_cache
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.forces import compute_forces
from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response import phonon as phonon_module
from defuqumat.response.phonon import dynamical_matrix
from defuqumat.scf import Calculation, run_scf
from defuqumat.system import build_system

pytestmark = [pytest.mark.regression, pytest.mark.slow,
              pytest.mark.filterwarnings("ignore:tefield"),
              pytest.mark.filterwarnings("ignore:.*symmetry")]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

IDENTITY = {"tr2": 1.0e-24, "threshold": 1.0e-12}
CONV_THR = 1.0e-13
EAMP = 0.01
#: The step of the central difference, bohr. It sits in the roughness: the
#: residual at it is 8.6e-6, 1.5e-6 (plain field) and 8.1e-5, 8.2e-5
#: (dipfield), while the neighbouring steps 0.0035 and 0.00125 give up to
#: 1.7e-4, and the Richardson pair at 0.01/0.005 leaves 1.9e-4 on O z
#: (``notes/incompat/I104.md``).
STEP = 0.0025
#: Set from the roughness and the Richardson offset (both up to 1.9e-4), not
#: from the residual at ``STEP``; still 12 times under the field's effect and
#: 300 times under the missing term.
COLUMN = 4.0e-4
#: The field moves the C z column by 4.9e-3 Ry/bohr^2 (eamp 0.01 against 0);
#: leaving out the ions' dipole under dipfield moves it by 0.12.
VISIBLE = 2.0e-3


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


def _system(case, eamp=EAMP):
    text = (CASES / f"{case}.in").read_text()
    assert "eamp=0.001" in text
    # The files give both masses as 0.0, which pw.x accepts and a dynamical
    # matrix cannot be mass-weighted with.
    text = text.replace("O  0.0", "O  15.999").replace("C  0.0", "C  12.011")
    return build_system(parse_pw_input(text.replace("eamp=0.001", f"eamp={eamp}")))


def _pseudos(system):
    return tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)


def _converged(system):
    pseudos = _pseudos(system)
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=CONV_THR,
                     max_iterations=200)
    assert result.converged
    return calculation, result


@lru_cache(maxsize=2)
def _matrix(case, eamp=EAMP, without_ion_dipole=False):
    calculation, result = _converged(_system(case, eamp))
    saved = phonon_module._core_charge_response
    if without_ion_dipole:
        phonon_module._core_charge_response = lambda *args, **kwargs: None
    try:
        phonons = dynamical_matrix(
            calculation, result.wavefunctions, result.eigenvalues, result.density,
            result.becsum, **IDENTITY,
        )
    finally:
        phonon_module._core_charge_response = saved
    assert phonons.converged
    return np.asarray(phonons.matrix)


def _without_net(column):
    """The column with its sum over atoms removed, as the forces have it.

    ``compute_forces`` subtracts the net force (QE's ``sumfor``), so a central
    difference of the forces obeys the translation sum rule exactly, while the
    Sternheimer column carries the residue the FFT grid leaves on a molecule
    in a box (5.2e-3 Ry/bohr^2 on CO's ``z`` column). The same projection is
    applied to the column before the two are compared.
    """
    return column - column.mean(axis=0)


def _difference(case, atom, cart, step=STEP):
    system = _system(case)
    positions = np.asarray(system.structure.positions)
    forces = []
    for sign in (+1, -1):
        moved = positions.copy()
        moved[atom, cart] += sign * step
        shifted = eqx.tree_at(lambda s: s.structure.positions, system,
                              jnp.asarray(moved))
        calculation, result = _converged(shifted)
        forces.append(np.asarray(compute_forces(calculation, result).forces))
    return -(forces[0] - forces[1]) / (2 * step)


@pytest.mark.parametrize("case", ["efield-co", "efield-co-dipole"])
@pytest.mark.parametrize("atom,cart", [(1, 2), (0, 2)])
def test_a_column_is_the_difference_of_the_forces(case, atom, cart):
    column = _without_net(_matrix(case)[atom, cart])
    np.testing.assert_allclose(column, _difference(case, atom, cart), atol=COLUMN)


def test_the_field_is_visible_in_the_matrix():
    """Without this the column test would pass with the field left out."""
    field = _without_net(_matrix("efield-co")[1, 2])
    none = _without_net(_matrix("efield-co", eamp=0.0)[1, 2])
    assert np.abs(field - none).max() > VISIBLE


def test_without_the_ions_dipole_the_column_misses_the_forces():
    """The term this item added: ``d ion_dip/dtau`` in ``dV_bare`` under dipfield."""
    without = _without_net(
        _matrix("efield-co-dipole", without_ion_dipole=True)[1, 2])
    assert np.abs(without - _difference("efield-co-dipole", 1, 2)).max() > VISIBLE
