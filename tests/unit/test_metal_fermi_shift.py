"""``sym_def``: a metal's Fermi-level shift per displacement, averaged over the group.

A displacement at ``q = 0`` moves a metal's Fermi level by ``def_{a,i} = d e_F /
d u_{a,i}`` (``LR_Modules/efermi_shift.f90``), and on a symmetry-reduced k-set
the raw value is a wedge sum. ``ph.x`` averages it over the group (``sym_def``)
before it enters the density as ``def ldos`` and the states as ``ef_shift_wfc``.
The phonon loop used the raw value (AUDIT-2026-11-20b #12), which is harmless
only where every atom is an inversion centre: on hcp aluminium (site group
-6m2) the raw shifts were 1e-2 Ry/bohr, their average zero, and the E mode read
150.96 cm^-1 on the wedge against 146.03 without symmetry.

What is here needs no ground state: the average itself, as the per-atom vector
the shift is (``symvector``), and the same average read off the displacement
symmetriser on a uniform field, which is what the loop's ``def ldos`` passed
through before. The whole phonon, wedge against ``nosym``, is the slow
``test_a_metal_off_inversion_centres_matches_nosym`` in
``tests/regression/test_phonons.py``.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation
from defuqumat.system import build_system

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

HCP_ALUMINIUM = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 4, celldm(1) = 5.35, celldm(3) = 1.633,
    nat = 2, ntyp = 1, ecutwfc = 8.0,
    occupations = 'smearing', smearing = 'marzari-vanderbilt', degauss = 0.05
    {extra}
 /
 &electrons
 /
ATOMIC_SPECIES
 Al 26.98 Al.pz-vbc.UPF
ATOMIC_POSITIONS (crystal)
 Al 0.333333333333 0.666666666667 0.25
 Al 0.666666666667 0.333333333333 0.75
K_POINTS automatic
 2 2 2 0 0 0
"""

#: The first pass's raw shifts on hcp aluminium at 12 Ry and 4x4x2, Ry/bohr,
#: in mode order (atom, direction).
RAW_HCP = np.array([[-0.009887, 0.010171, 0.000108],
                    [0.009887, -0.010171, 0.000108]])


def _calculation(text):
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return Calculation(system, pseudos)


def test_the_shift_of_hcp_aluminium_averages_to_zero():
    """Each atom's site group -6m2 allows no vector, so the average is zero.

    The raw wedge value is not, which is the whole of the finding. Without
    symmetry the average is the identity.
    """
    wedge = _calculation(HCP_ALUMINIUM.format(extra=""))
    assert wedge.use_symmetry
    averaged = wedge.symmetrize_atom_vector(RAW_HCP)
    assert np.abs(averaged).max() < 1.0e-15
    whole = _calculation(HCP_ALUMINIUM.format(extra="nosym = .true., noinv = .true."))
    assert not whole.use_symmetry
    np.testing.assert_array_equal(whole.symmetrize_atom_vector(RAW_HCP), RAW_HCP)


def test_the_average_is_the_displacement_symmetriser_on_a_uniform_field():
    """``symvector`` of the shifts is what ``symdvscf`` does to ``def`` times a constant.

    On AB bilayer graphene, whose sites have C3v symmetry, so that the ``z``
    component survives and the check is not of two zeros. A uniform field is
    invariant under every operation, so the displacement symmetriser acts on
    it through the mode labels alone, which is the average of ``def`` as a
    per-atom vector.
    """
    text = (CASES / "graphene-bilayer.in").read_text()
    for old, new in (("ecutwfc = 40", "ecutwfc = 10"), (" 12 12 1 0 0 0", " 2 2 1 0 0 0")):
        assert old in text
        text = text.replace(old, new)
    calculation = _calculation(text)
    assert calculation.use_symmetry
    raw = np.random.default_rng(12).normal(size=(4, 3))
    averaged = calculation.symmetrize_atom_vector(raw)

    grid = tuple(calculation.basis.dense.grid)
    fields = jnp.asarray(raw)[:, :, None, None, None, None] * jnp.ones((1,) + grid)
    through_fields = np.asarray(calculation.symmetrize_atom_displacement(fields))
    np.testing.assert_allclose(through_fields.mean(axis=(2, 3, 4, 5)), averaged,
                               atol=1.0e-12)
    # The in-plane components average away and z does not.
    assert np.abs(averaged[:, :2]).max() < 1.0e-12
    assert np.abs(averaged[:, 2]).min() > 1.0e-3


def test_the_atom_pairing_is_the_displacement_symmetriser_s_on_a_three_cycle():
    """Three hydrogens on a generic C3 orbit: every operation but the identity is a 3-cycle.

    On hcp aluminium, on fcc aluminium and on the graphene bilayer every atom
    permutation is an involution, so ``irt`` and its inverse give the same
    average and those checks cannot tell them apart (the trap the header of
    ``al4-metal.in`` records). Here they differ by 1.04 on a random vector, and
    the average must be the displacement symmetriser's, which is the one checked
    against ``ph.x`` on a cell with 3-cycles.
    """
    from defuqumat.system.symmetry import atom_mapping, symmetrize_vector

    point = np.array([0.30, 0.10])
    orbit = [point]
    for _ in range(2):
        x, y = orbit[-1]
        orbit.append(np.array([-y, x - y]))  # C3 about z in hexagonal axes
    rows = "\n".join(f" H {x % 1:.12f} {y % 1:.12f} 0.1" for x, y in orbit)
    text = f"""
 &control
 /
 &system
    ibrav = 4, celldm(1) = 6.0, celldm(3) = 1.5, nat = 3, ntyp = 1, ecutwfc = 8.0,
    occupations = 'smearing', smearing = 'gaussian', degauss = 0.02
 /
 &electrons
 /
ATOMIC_SPECIES
 H 1.008 H.pz-vbc.UPF
ATOMIC_POSITIONS (crystal)
{rows}
K_POINTS automatic
 2 2 2 0 0 0
"""
    calculation = _calculation(text)
    mapping = np.asarray(atom_mapping(calculation.system.cell, calculation.system.structure,
                                      calculation.symmetries))
    assert calculation.symmetries.nsym == 3
    # Not an involution: some operation applied twice is not the identity.
    assert any(not np.array_equal(row[row], np.arange(3)) for row in mapping)

    raw = np.random.default_rng(7).normal(size=(3, 3))
    averaged = calculation.symmetrize_atom_vector(raw)
    grid = tuple(calculation.basis.dense.grid)
    fields = jnp.asarray(raw)[:, :, None, None, None, None] * jnp.ones((1,) + grid)
    through_fields = np.asarray(
        calculation.symmetrize_atom_displacement(fields)).mean(axis=(2, 3, 4, 5))
    np.testing.assert_allclose(averaged, through_fields, atol=1.0e-12)
    # The other pairing is far off, so the check can fail.
    inverse = np.asarray(symmetrize_vector(raw, calculation.system.cell,
                                           calculation.symmetries,
                                           np.argsort(mapping, axis=1)))
    assert np.abs(inverse - through_fields).max() > 0.1
