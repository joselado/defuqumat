"""I87: the elastic constants of an ultrasoft or PAW dataset, and what still refuses.

``require_a_measured_elastic_regime`` reads the calculation and nothing else, so
a real :class:`~defuqumat.scf.Calculation` on ``si-us-nosym.in`` (a basis and a
projector table, no SCF) is enough to see it fire.
"""

from pathlib import Path

import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation
from defuqumat.system import build_system

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _calculation(case):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return Calculation(system, pseudos)


@pytest.mark.parametrize("case", ["si-us-nosym", "si-paw-nosym"])
def test_the_augmented_elastic_constants_are_refused(case):
    from defuqumat.response.elastic import require_a_measured_elastic_regime

    with pytest.raises(NotImplementedError, match="ultrasoft or PAW"):
        require_a_measured_elastic_regime(_calculation(case))
