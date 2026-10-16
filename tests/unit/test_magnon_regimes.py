"""Which regimes the transverse spin susceptibility takes, each fired by name.

``notes/INCOMPATIBILITIES.md`` I58 to I62. No SCF here: every guard reads the
calculation alone, so each test builds one from the text of a committed input
with one line changed. The cell is ``h-fcc-magnon.in``, the one the magnon
identities are measured on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import Calculation
from defuqumat.system import build_system
from defuqumat.tddft.spinchi0 import require_a_transverse_regime

pytestmark = [pytest.mark.unit]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

HYDROGEN = (CASES / "h-fcc-magnon.in").read_text()
SYSTEM_LINE = "    nspin = 2\n"


def _calculation(text: str) -> Calculation:
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return Calculation(system, pseudos)


def _with(extra: str, card: str = "") -> str:
    assert SYSTEM_LINE in HYDROGEN
    return HYDROGEN.replace(SYSTEM_LINE, SYSTEM_LINE + extra) + card


# --- I58: an applied field ----------------------------------------------------


def test_a_uniform_field_is_accepted():
    """``B_field`` along the moment: the Larmor case, which the states carry."""
    calculation = _calculation(_with("    B_field(3) = 0.002\n"))
    assert calculation.magnetic_field is not None
    require_a_transverse_regime(calculation)


def test_a_per_atom_field_is_refused():
    """A ``LOCAL_MAGNETIC_FIELDS`` card is not a ``G = 0`` field.

    Larmor's theorem puts the uniform mode at one frequency only when the
    applied field is the same everywhere; a field confined to spheres is a
    perturbation with structure in ``G``, and no single frequency closes the
    identity.
    """
    calculation = _calculation(_with("", "LOCAL_MAGNETIC_FIELDS\n 0.0 0.0 0.002\n"))
    assert calculation.magnetic_field.atomic is not None
    with pytest.raises(NotImplementedError, match="per-atom field"):
        require_a_transverse_regime(calculation)


def test_a_constrained_moment_is_refused():
    """``fe-fsm.in``: a constrained moment is a different functional."""
    text = (CASES / "fe-fsm.in").read_text()
    calculation = _calculation(text)
    with pytest.raises(NotImplementedError, match="constrained moment"):
        require_a_transverse_regime(calculation)
