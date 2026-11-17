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
    """28 electrons by Elk's own integrals (``chgmt`` and ``chgir``): 27.9963.

    8 valence electrons and the 20 of the two frozen cores; the 3.7e-3 missing
    are the core's tail past ``rmt``, which the interstitial does not carry,
    and the completeness of the partial waves.
    """
    charges = written.muffin_tin_charges()
    np.testing.assert_allclose(charges[0], charges[1], rtol=1e-10)
    assert abs(charges.sum() + written.interstitial_charge() - 28.0) < 1e-2


def test_the_rebuilt_density_is_elk_own_in_the_valence_shell(written):
    """Against Elk's converged Si2 on its full meshes, ``l = 0``, 0.6 bohr < r < rmt.

    ``converged.npz`` is that run's profile (397 points, ``xctype 2``, the
    radius of the template's spheres). The rebuilt state is 0.032 away (int
    |drho| r^2 dr); with the one-centre term left out it is 0.114, as far as
    Elk's own initial state (0.115), so the check fails without the PAW
    reconstruction it is about.
    """
    from scipy.interpolate import CubicSpline

    reference = np.load(TEMPLATE / "converged.npz")
    elk = CubicSpline(reference["r"], reference["f00"])

    def distance(state):
        r = np.asarray(state.rsp[0])[:state.nrmt[0]]
        sel = (r > 0.6) & (r <= reference["r"][-1])
        return np.sqrt(4 * np.pi) * np.trapezoid(
            np.abs(state.rhomt[0, :len(r), 0][sel] - elk(r[sel])) * r[sel] ** 2, r[sel])

    assert distance(written) < 0.06
    calc = _calculator()
    import shutil, tempfile
    from defuqumat.io.elk_allelectron import write_elk_state_from
    control = Path(tempfile.mkdtemp())
    write_elk_state_from(control, calc.get_scf(), calc.calculation, TEMPLATE, onecentre=False)
    shutil.copy(TEMPLATE / "GEOMETRY.OUT", control)
    assert distance(ElkState.read(control)) > 0.09


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


def test_a_spin_orbit_run_is_refused():
    """No spin-orbit cell has been run through Elk; refused before the result is read."""
    from types import SimpleNamespace

    calc = _calculator()
    system = SimpleNamespace(**{k: getattr(calc.calculation.system, k) for k in
                                ("cell", "structure", "noncolin", "nspin")}, lspinorb=True)
    calculation = SimpleNamespace(system=system, is_paw=True, pseudos=calc.calculation.pseudos,
                                  nspin_mag=1)
    from defuqumat.io.elk_allelectron import all_electron_state
    with pytest.raises(NotImplementedError, match="spin-orbit"):
        all_electron_state(None, calculation, TEMPLATE)
