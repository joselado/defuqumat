"""A PAW ground state written as an Elk state, on a template's meshes.

``tests/data/elk/si2_template`` is an Elk 11.0.2 run of Si2 under the
Perdew-Zunger LDA with small meshes, written after one loop; the defuqumat run
is the same crystal on ``Si.pz-n-kjpaw_psl.0.1`` with Elk's lattice vectors.
What is checked here owes nothing to Elk's convergence: the rebuilt density
holds the electrons, the frozen core and an ``l = 3`` channel a superposition
of atoms does not have, the potential on file is the transcribed ``potks`` of
the density on file, and the refusals fire. Elk continuing from such a file is
measured in ``PLAN.md`` P181 (Si2 LDA: 16 loops against 19; O2 LSDA and
noncollinear: 14 against 17).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.io.elk import ElkState
from defuqumat.io.elk_potential import (crystal_symmetrizer, kohn_sham_potential,
                                        read_elk_parameters)

TEMPLATE = Path(__file__).resolve().parents[1] / "data" / "elk" / "si2_template"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

SI2 = """&control
 calculation='scf'
/
&system
 ibrav=0, nat=2, ntyp=1, ecutwfc=15.0, ecutrho=120.0 {extra}
/
&electrons
 conv_thr=1d-8
/
CELL_PARAMETERS bohr
 0.0 5.10 5.10
 5.10 0.0 5.10
 5.10 5.10 0.0
ATOMIC_SPECIES
 Si 28.086 {pseudo}
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


def _calculator(pseudo="Si.pz-n-kjpaw_psl.0.1.UPF", extra=""):
    return Calculator.from_text(SI2.format(pseudo=pseudo, extra=extra), pseudo_dir=PSEUDO,
                                announce=False)


@lru_cache(maxsize=1)
def _written(directory):
    calc = _calculator()
    calc.write_elk_state(directory, TEMPLATE)
    import shutil
    shutil.copy(TEMPLATE / "GEOMETRY.OUT", directory)
    return ElkState.read(directory)


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    return _written(tmp_path_factory.mktemp("elk"))


def test_the_density_holds_every_electron(written):
    """28 electrons: 8 valence and the 20 of the two frozen cores."""
    state = written
    r = np.asarray(state.rsp[0])[:state.nrmt[0]]
    inside = [np.sqrt(4 * np.pi) * np.trapezoid(state.rhomt[0, :len(r), a] * r ** 2, r)
              for a in range(2)]
    np.testing.assert_allclose(inside[0], inside[1], rtol=1e-10)
    assert 12.0 < inside[0] < 12.6  # Elk's converged Si2 holds 12.30 in each sphere
    # The interstitial charge, on points outside both spheres.
    avec = np.asarray(state.geometry.avec)
    ng = np.array(state.ngridg)
    frac = np.indices(ng).reshape(3, -1, order="F").T / ng
    out = np.ones(len(frac), bool)
    for p in np.asarray(state.geometry.positions):
        d = frac - p
        d -= np.round(d)
        out &= np.linalg.norm(d @ avec, axis=1) > r[-1]
    omega = abs(np.linalg.det(avec))
    interstitial = state.rhoir.reshape(-1, order="F")[out].sum() * omega / len(frac)
    assert abs(sum(inside) + interstitial - 28.0) < 0.2  # grid quadrature of the interstitial


def test_the_bond_is_in_the_spheres(written):
    """The ``l = 3`` (xyz) channel at the tetrahedral site, absent from free atoms."""
    state = written
    lm = 3 * 4 - 2  # Elk's j = l(l+1) + m for l = 3, m = -2
    r = np.asarray(state.rsp[0])[:state.nrmt[0]]
    i = int(np.searchsorted(r, 1.8))
    assert abs(state.rhomt[lm, i, 0]) > 1e-2
    np.testing.assert_allclose(state.rhomt[lm, i, 1], -state.rhomt[lm, i, 0], rtol=1e-8)


def test_the_potential_on_file_is_potks_of_the_density_on_file(written):
    state = written
    params = read_elk_parameters(TEMPLATE)
    symmetry, nsym = crystal_symmetrizer(state, params.lmaxo)
    assert nsym == 48
    potential = kohn_sham_potential(state, params, symmetry)
    nr = state.nrmt[0]
    for key in ("vclmt", "vxcmt", "vsmt"):
        np.testing.assert_allclose(potential[key][:, :nr], getattr(state, key)[:, :nr] / 2.0,
                                   rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(potential["vsir"], state.vsir / 2.0, atol=1e-13)


@pytest.mark.parametrize("pseudo, extra, error, words", [
    ("Si.pz-vbc.UPF", "", NotImplementedError, "not PAW"),
    ("Si.pz-n-kjpaw_psl.0.1.UPF",
     ", nspin=2, starting_magnetization(1)=0.2, occupations='smearing', degauss=0.02",
     ValueError, "spin regime"),
])
def test_a_state_elk_cannot_take_is_refused(pseudo, extra, error, words, tmp_path):
    calc = _calculator(pseudo, extra)
    with pytest.raises(error, match=words):
        calc.write_elk_state(tmp_path, TEMPLATE, calc.get_scf(max_iterations=1))


def test_another_lattice_is_refused(tmp_path):
    text = SI2.format(pseudo="Si.pz-n-kjpaw_psl.0.1.UPF", extra="").replace(
        "CELL_PARAMETERS bohr\n 0.0 5.10 5.10\n 5.10 0.0 5.10\n 5.10 5.10 0.0",
        "CELL_PARAMETERS bohr\n -5.10 0.0 5.10\n 0.0 5.10 5.10\n -5.10 5.10 0.0")
    calc = Calculator.from_text(text, pseudo_dir=PSEUDO, announce=False)
    with pytest.raises(ValueError, match="lattice vectors"):
        calc.write_elk_state(tmp_path, TEMPLATE, calc.get_scf(max_iterations=1))
