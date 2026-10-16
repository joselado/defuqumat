"""What a partial dynamical matrix (``atoms=``) still refuses with symmetry on.

A subset that is a union of orbits runs with symmetry (``notes/incompat/I51.md``,
checked in ``tests/regression/test_phonons.py``); every case here is a guard
that fires before any ground state is built, so none needs an SCF:

* a subset that is **not** a union of orbits, where the group carries a mode
  inside it onto an atom outside it and the wedge sum has nothing to read there
  (diamond silicon's ``atoms=(0,)``, fluorite's ``atoms=(1,)``);
* a closed subset with symmetry on the route that walks the k axis a chunk at a
  time, which was not measured;
* and the acoustic sum rule on a subset, a sum over every atom.

The two host-side tables the symmetrisers read are checked here as well, on the
fluorite cell whose two As are one orbit exchanged by the inversion.
"""

from pathlib import Path

import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input, read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response.phonon import dynamical_matrix
from defuqumat.scf import Calculation
from defuqumat.system.builder import build_system
from defuqumat.system.symmetry import (
    atom_mapping, orbit_closure, subset_atom_mapping,
)

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: Si at the origin and As at +-(1/4, 1/4, 1/4): the two As are one orbit of O_h.
FLUORITE = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.4, nat = 3, ntyp = 2, ecutwfc = 10.0,
  occupations = 'smearing', smearing = 'mv', degauss = 0.05
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
 As 74.92  As.pz-bhs.UPF
ATOMIC_POSITIONS crystal
 Si  0.00  0.00  0.00
 As  0.25  0.25  0.25
 As -0.25 -0.25 -0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


def _calculation(system):
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file)
                    for s in system.structure.species)
    return Calculation(system, pseudos)


def _case(name: str):
    return _calculation(build_system(read_pw_input(CASES / f"{name}.in")))


def _fluorite():
    return _calculation(build_system(parse_pw_input(FLUORITE)))


def _call(calculation, **kwargs):
    """The guards run before any array is read, so placeholders do."""
    return dynamical_matrix(
        calculation, np.zeros((1, 1, 1, 1)), np.zeros((1, 1, 1)),
        np.zeros((1, 1, 1, 1)), (), **kwargs,
    )


def test_a_subset_of_a_symmetrised_run_is_refused():
    """Diamond silicon: half the operations exchange the sublattices, so
    ``atoms=(0,)`` is not closed and the refusal names ``(0, 1)``."""
    calculation = _case("si-epsilon")
    assert calculation.use_symmetry
    with pytest.raises(NotImplementedError, match=r"atoms=\(0, 1\).*nosym"):
        _call(calculation, atoms=(0,))


def test_half_an_orbit_is_refused_and_the_orbit_is_named():
    """Fluorite: one As without its inversion partner is not closed."""
    calculation = _fluorite()
    assert calculation.use_symmetry
    with pytest.raises(NotImplementedError, match=r"atoms=\(1, 2\)"):
        _call(calculation, atoms=(1,))


def test_a_closed_subset_on_the_streamed_route_is_refused(monkeypatch):
    """The closed-subset symmetrisation was measured on the whole-k route only.

    ``_streams`` is forced, because what decides it on a real run (a streamed
    store, a chunk smaller than the mesh, a card) is not this check's subject;
    AlAs's ``atoms=(0,)`` passes the orbit guard, so this refusal is the one
    that fires.
    """
    import defuqumat.response.phonon as phonon

    calculation = _case("alas-raman-wedge")
    assert calculation.use_symmetry
    monkeypatch.setattr(phonon, "_streams", lambda *args, **kwargs: True)
    with pytest.raises(NotImplementedError, match="whole-k route"):
        _call(calculation, atoms=(0,))


def test_the_acoustic_sum_rule_is_still_refused_on_a_closed_subset():
    """The sibling refusal, on a subset the symmetry guard now accepts."""
    calculation = _case("alas-raman-wedge")
    with pytest.raises(ValueError, match="acoustic sum rule"):
        _call(calculation, atoms=(0,), acoustic_sum_rule=True)


def test_the_subset_table_is_a_permutation_in_the_subset_indices():
    """``(1, 2)`` is closed and some operation swaps the two As; ``(1,)`` is not
    closed and has no table."""
    calculation = _fluorite()
    mapping = atom_mapping(calculation.system.cell, calculation.system.structure,
                           calculation.symmetries)
    assert orbit_closure(mapping, (1,)) == (1, 2)
    assert orbit_closure(mapping, (0,)) == (0,)
    table = subset_atom_mapping(mapping, (1, 2))
    assert table.shape == (len(calculation.symmetries.rotations), 2)
    assert all(sorted(row) == [0, 1] for row in table)
    assert any(list(row) == [1, 0] for row in table)
    with pytest.raises(ValueError, match="not closed"):
        subset_atom_mapping(mapping, (1,))
