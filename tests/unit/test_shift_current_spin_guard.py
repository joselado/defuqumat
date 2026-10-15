"""Step 1 of I6: the ``nspin = 2`` refusal of ``shift_current`` fires (no SCF)."""

import re
from pathlib import Path

import jax.numpy as jnp
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo.upf import read_upf
from defuqumat.response.photocurrent import shift_current
from defuqumat.scf.driver import Calculation
from defuqumat.system.builder import build_system

pytestmark = pytest.mark.unit

DATA = Path(__file__).resolve().parents[1] / "data"


def test_the_nspin2_guard_fires(tmp_path):
    text = (DATA / "qe" / "alas-shg.in").read_text()
    text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", "ecutwfc = 6.0", text)
    text = re.sub(r"K_POINTS.*\n\s*[0-9 ]+\n", "K_POINTS (automatic)\n 1 1 1 0 0 0\n", text)
    text = text.replace("&system", "&system\n    nspin = 2, tot_magnetization = 0,", 1)
    path = tmp_path / "alas.in"
    path.write_text(text)
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(DATA / "pseudo" / s.pseudo_file) for s in system.structure.species)
    calc = Calculation(system, pseudos)
    nk = len(system.kpoints.weights)
    with pytest.raises(NotImplementedError, match="collinear spin-polarized"):
        shift_current(calc, jnp.zeros((2, nk, 4, 8), complex), jnp.zeros((2, nk, 4)), None)
