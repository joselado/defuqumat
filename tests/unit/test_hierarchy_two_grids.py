"""The self-consistent frequency-domain hierarchy on two FFT grids, and its siblings.

A norm-conserving run with ``ecutrho > 4 ecutwfc`` has its wavefunctions on the
smooth grid and its density and potential on the dense one. The guards here
fire before any solve, so none of them needs an SCF.
"""

from pathlib import Path

import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.realtime.hierarchy import hierarchy_linear_self_consistent
from defuqumat.scf import Calculation
from defuqumat.system import build_system

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


def _calculation(name, pseudo_dir):
    system = build_system(read_pw_input(CASES / name))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return Calculation(system, pseudos)


def _call(calculation, potential="hartree"):
    return hierarchy_linear_self_consistent(
        calculation, np.zeros((1, 1, 1)), np.zeros((1, 1)), np.zeros((1, 1)), None,
        omegas=(0.05,), eta=0.01, direction=(1.0, 0.0, 0.0), potential=potential)


def test_two_grids_pass_the_guard_that_refused_them(pseudo_dir):
    """The refusal of two grids is gone: the call reaches the arrays, which are placeholders."""
    calculation = _calculation("si2-nc-dual8.in", pseudo_dir)
    assert calculation.basis.smooth.grid != calculation.basis.dense.grid
    with pytest.raises(Exception) as caught:
        _call(calculation)
    assert "FFT grid" not in str(caught.value)


@pytest.mark.parametrize("potential", ["hartree", "hxc"])
def test_an_ultrasoft_dataset_on_two_grids_is_still_refused(pseudo_dir, potential):
    calculation = _calculation("si2-us.in", pseudo_dir)
    assert calculation.basis.smooth.grid != calculation.basis.dense.grid
    with pytest.raises(NotImplementedError, match="ultrasoft or PAW"):
        _call(calculation, potential)


def test_a_meta_gga_kernel_is_still_refused(pseudo_dir):
    with pytest.raises(NotImplementedError, match="meta-GGA"):
        _call(_calculation("si2-tb09.in", pseudo_dir), "hxc")
