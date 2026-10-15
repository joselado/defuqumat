"""The wedge refusals that stand after the wedge unit, each fired by name.

No SCF here: every guard reads the calculation alone. The transverse spin
susceptibility had no test firing its wedge refusal at all; it is the first one
below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from defuqumat.io.pwin import parse_pw_input, read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import Calculation
from defuqumat.system import build_system

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _calculation(text: str) -> Calculation:
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return Calculation(system, pseudos)


def test_a_magnon_on_a_symmetry_reduced_k_set_is_refused():
    """``ni-fcc-magnon.in`` with its ``nosym``/``noinv`` taken out.

    ``X_0(G, G')`` at ``q`` needs the states at ``k + q`` from the same set,
    which a wedge does not hold, and a magnetic group besides; the committed
    cell, which carries ``nosym``, passes the same guard.
    """
    from defuqumat.tddft.spinchi0 import require_a_transverse_regime

    text = (CASES / "ni-fcc-magnon.in").read_text()
    reduced = _calculation(text.replace("nosym = .true., noinv = .true.", ""))
    assert reduced.system.kpoints.nk < 64
    with pytest.raises(NotImplementedError, match="full k-grid"):
        require_a_transverse_regime(reduced)
    require_a_transverse_regime(_calculation(text))
