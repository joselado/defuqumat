"""The Kramers average of a spin-orbit projection when the group is the identity (audit 2026-11-20c #22).

``sym_proj_so`` averages each ``|j m_j>`` column with its Kramers partner
``m_j -> -m_j`` on a nonmagnetic run, under ``lsym`` alone, so also when the
group is the identity. On a P1 crystal time reversal still halves the grid,
and the ``-k`` states the wedge leaves out are the Kramers partners of the
kept ones. ``build_projection_symmetry`` returned ``None`` at ``nsym = 1``, so
the wedge's ``m_j`` columns were the bare ``|proj0|^2`` of half the zone. Setup
only: random states on the calculation's own sphere, no SCF.
"""

import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import Calculation
from defuqumat.system.builder import build_system

pytestmark = pytest.mark.unit

# Three iodine atoms whose separations have no crystal component of 0 or 1/2
# on an orthorhombic lattice: nsym = 1 (``test_site_moments_time_reversal``).
_P1 = """
&control
  calculation = 'scf'
/
&system
  ibrav = 0, nat = 3, ntyp = 1, ecutwfc = 12.0,
  occupations = 'smearing', smearing = 'mv', degauss = 0.02,
  noncolin = .true., lspinorb = .true.
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 I 126.9 I.rel-pbe-nc-dojo.UPF
CELL_PARAMETERS bohr
 8.00 0.00 0.00
 0.00 8.40 0.00
 0.00 0.00 8.90
ATOMIC_POSITIONS crystal
 I 0.00 0.00 0.00
 I 0.45 0.16 0.33
 I 0.17 0.60 0.66
K_POINTS automatic
 2 2 2 1 1 1
"""


def _calculation(pseudo_dir, extra=""):
    system = build_system(parse_pw_input(_P1.format(extra=extra)))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    return Calculation(system, pseudos)


def _partners(channels):
    """Each column's ``m_j -> -m_j`` partner, read off the labels alone."""
    where = {(c.atom, c.wfc, c.l, c.j, c.m): c.index for c in channels}
    return np.array([where[(c.atom, c.wfc, c.l, c.j, int(round(2 * c.j)) - c.m)]
                     for c in channels])


def _random_states(calculation, nbnd=6, seed=3):
    nk = calculation.system.kpoints.nk
    npwx = calculation.basis.planewaves.npwx
    rng = np.random.default_rng(seed)
    shape = (1, nk, nbnd, 2 * npwx)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _projections(calculation, states, symmetrize=True):
    from defuqumat.projwfc.projections import atomic_projections

    return np.asarray(atomic_projections(calculation, states, symmetrize=symmetrize))


def test_the_wedge_of_a_p1_crystal_gets_the_kramers_average(pseudo_dir):
    from defuqumat.projwfc.projections import calculation_channels

    calculation = _calculation(pseudo_dir)
    kset = calculation.system.kpoints
    assert calculation.symmetries.nsym == 1 and not calculation.use_symmetry
    assert kset.reduced and kset.nk == 4  # time reversal alone halved the 8 points
    assert not calculation.system.domag

    states = _random_states(calculation)
    channels = calculation_channels(calculation)
    partner = _partners(channels)
    assert np.any(partner != np.arange(len(channels)))
    bare = _projections(calculation, states, symmetrize=False)
    averaged = _projections(calculation, states)
    # sym_proj_so's .not. domag pass at nsym = 1: 0.5 (|p_c|^2 + |p_partner(c)|^2)
    np.testing.assert_allclose(averaged, 0.5 * (bare + bare[:, :, partner]),
                               atol=1e-12 * bare.max())
    assert np.abs(bare - bare[:, :, partner]).max() > 1e-3 * bare.max()
    # each j shell's sum, and so every per-l charge and the total, is unchanged
    np.testing.assert_allclose(averaged.sum(axis=2), bare.sum(axis=2),
                               rtol=1e-12)


@pytest.mark.parametrize("extra", ["noinv = .true.", "nosym = .true."])
def test_a_whole_grid_keeps_the_bare_projection(extra, pseudo_dir):
    """The regime next to it: time reversal reduced nothing, so nothing is averaged."""
    calculation = _calculation(pseudo_dir, extra)
    assert calculation.system.kpoints.nk == 8
    states = _random_states(calculation)
    np.testing.assert_array_equal(_projections(calculation, states),
                                  _projections(calculation, states, symmetrize=False))


def test_the_partner_is_the_time_reversed_spin_angle_function():
    """``i sigma_y conj`` takes the ``m_j`` spin-angle function to the ``-m_j`` one.

    That is why the partner of a column is the column with the reversed
    ``m_j``: time reversal sends the orbital at ``k`` to the partner's at ``-k``,
    up to a phase, and ``|<T phi|T psi>|^2 = |<phi|psi>|^2``. Checked on the
    map the orbitals are built with, for every ``(l, j)`` a dataset carries.
    """
    from defuqumat.pseudo.atomic import _spin_angle_matrix

    for l in range(4):
        for j in (l - 0.5, l + 0.5):
            if j < 0:
                continue
            m = _spin_angle_matrix(l, j)  # (2j+1, 2, 2l+1)
            theta = np.stack([np.conj(m[:, 1]), -np.conj(m[:, 0])], axis=1)
            overlap = np.abs(np.einsum("asm,bsm->ab", np.conj(m), theta))
            np.testing.assert_allclose(overlap, np.eye(len(m))[::-1], atol=1e-12)
