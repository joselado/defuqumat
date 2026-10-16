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


# --- I59 and I62: the noncollinear family -------------------------------------


def test_a_spiral_is_refused_for_its_own_reason():
    """``h-chain-spiral.in``: the spiral branch used to sit behind ``noncolin``.

    A spiral is noncollinear by construction, so the generic refusal fired
    first and the spiral's own reason (two spheres, the rotating frame) was
    never reached.
    """
    calculation = _calculation((CASES / "h-chain-spiral.in").read_text())
    with pytest.raises(NotImplementedError, match="spin spiral"):
        require_a_transverse_regime(calculation)


def test_nosource_is_refused_for_its_own_reason():
    """``ni-noncol-111-nosource.in``: the source-free branch, reached.

    ``nosource`` needs ``noncolin`` (``refuse_source_free``), so before this
    branch was moved ahead of the noncollinear one it could not fire at all.
    """
    calculation = _calculation((CASES / "ni-noncol-111-nosource.in").read_text())
    assert calculation.source_free
    with pytest.raises(NotImplementedError, match="under nosource"):
        require_a_transverse_regime(calculation)


def test_a_noncollinear_magnet_is_still_refused():
    """``h-chain-90deg.in``: a texture, which needs the 4x4 response."""
    calculation = _calculation((CASES / "h-chain-90deg.in").read_text())
    with pytest.raises(NotImplementedError, match="noncollinear ground state"):
        require_a_transverse_regime(calculation)
