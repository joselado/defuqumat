"""I104: which linear responses run under a saw-tooth field, and which still refuse.

The dynamical matrix at ``Gamma`` is the one response lifted under ``tefield``
(and ``dipfield``), on the regime it was measured in: a fixed-occupation
``nspin = 1`` insulator on the whole sphere, against a central difference of
the forces under the field (``tests/regression/test_sawtooth_phonons.py``,
``notes/incompat/I104.md``). Everything here is a guard that fires before any
ground state is built, so none needs an SCF: the other responses keep the
saw-tooth refusal, every response keeps the Berry-phase one, and the matrix
itself refuses the sub-regimes nobody ran with the field.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response.phonon import dynamical_matrix
from defuqumat.response.sternheimer import require_a_sternheimer_regime
from defuqumat.scf import Calculation
from defuqumat.system import build_system

pytestmark = [pytest.mark.unit, pytest.mark.filterwarnings("ignore:tefield"),
              pytest.mark.filterwarnings("ignore:.*symmetry")]

PACKAGE = Path(__file__).resolve().parents[2] / "defuqumat"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

_INPUT = """
 &control
    calculation='scf', tefield=.true., dipfield={dipfield}
 /
 &system
    ibrav=1, celldm(1)=10.0, nat=1, ntyp=1, ecutwfc=8,
    edir=3, eamp=0.002, emaxpos=0.4, eopreg=0.2, {extra}
 /
 &electrons
 /
ATOMIC_SPECIES
Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS bohr
Si 0.0 0.0 0.0
K_POINTS {kpoints}
"""


def _calculation(dipfield=".false.", extra="", kpoints="automatic\n2 2 2 0 0 0"):
    system = build_system(parse_pw_input(
        _INPUT.format(dipfield=dipfield, extra=extra, kpoints=kpoints)))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file)
                    for s in system.structure.species)
    return Calculation(system, pseudos)


def _phonons(calculation):
    """The guards run before any array is read, so placeholders do."""
    return dynamical_matrix(
        calculation, np.zeros((1, 1, 1, 1)), np.zeros((1, 1, 1)),
        np.zeros((1, 1, 1, 1)), (),
    )


@pytest.mark.parametrize("owner", [
    SimpleNamespace(sawtooth=object()),
    SimpleNamespace(efield=(2, 0.001, 0.5, 0.1, False)),
])
def test_every_other_response_refuses_the_sawtooth(owner):
    """The default every caller but the dynamical matrix takes."""
    with pytest.raises(NotImplementedError, match="saw-tooth"):
        require_a_sternheimer_regime(owner)


@pytest.mark.parametrize("sawtooth_ok", [False, True])
def test_the_berry_phase_field_is_refused_either_way(sawtooth_ok):
    owner = SimpleNamespace(berry_field=(2, 0.001))
    with pytest.raises(NotImplementedError, match="Berry phase"):
        require_a_sternheimer_regime(owner, sawtooth_ok=sawtooth_ok)


def test_only_the_dynamical_matrix_lifts_the_sawtooth():
    """A new caller passing ``sawtooth_ok`` has to be measured and added here."""
    callers = sorted(
        str(path.relative_to(PACKAGE)) for path in PACKAGE.rglob("*.py")
        if re.search(r"sawtooth_ok=True[,)]", path.read_text())
    )
    assert callers == ["response/phonon.py"]


def test_a_real_calculation_under_the_field_is_refused_by_the_default():
    with pytest.raises(NotImplementedError, match="saw-tooth"):
        require_a_sternheimer_regime(_calculation())


@pytest.mark.parametrize("extra,words", [
    ("occupations='smearing', degauss=0.02", "a metal"),
    ("nspin=2, starting_magnetization(1)=0.5, occupations='smearing', degauss=0.02",
     "spin-polarized"),
])
def test_the_dynamical_matrix_refuses_what_was_not_run_with_the_field(extra, words):
    with pytest.raises(NotImplementedError, match=words):
        _phonons(_calculation(extra=extra))


def test_the_dynamical_matrix_refuses_half_sphere_storage_under_the_field():
    calculation = _calculation(kpoints="gamma")
    assert calculation.gamma_only
    with pytest.raises(NotImplementedError, match="half-sphere"):
        _phonons(calculation)
