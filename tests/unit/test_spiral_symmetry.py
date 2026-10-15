"""The spin space group of a flat spiral, without an SCF.

A spatial operation ``{S|f}`` with ``S q = q`` is a symmetry of a spiral
together with the laboratory spin rotation ``Rot_z(q . f)``, which the spiral's
own phase cancels on the rotated-frame fields the SCF holds: the operation
permutes ``n, m_x, m_y, m_z`` as four scalars
(:func:`~defuqumat.system.spiral.spiral_symmetries`). What is asserted here is
the group each cell gets, the k-set it gives, and that the seed is invariant
under the four-scalar rule and **not** under the two rules it replaces -- the
axial ``det(S) S`` of an ordinary magnetic run, and a leftover ``Rot_z(q . f)``
-- so that the residual is a discriminator rather than a null result. The
energies against the ``nosym`` runs are ``tests/regression/test_spiral_symmetry.py``.
"""

import dataclasses
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import Calculation, _symmetrize_noncollinear
from defuqumat.system.builder import build_system
from defuqumat.system.symmetry import cartesian_rotations, magnetization_signs

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
PSEUDO = ROOT / "data" / "pseudo"

#: fcc hydrogen with ``q = b3/4``, a ``[111]``-type direction in cartesian axes,
#: so the stabiliser in ``O_h`` is ``C3v``: 6 operations, ``f = 0``.
#: ``q = b3/8``, where the scan in that file's header converges in 10
#: iterations with ``|m| = 0.40``; at ``b3/4`` it does not converge, and at a
#: larger smearing the moment is gone (``|m| = 5e-5`` at ``degauss = 0.05``,
#: measured), which would make the comparison blind to the magnetization.
FCC = (ROOT / "data" / "qe" / "h-fcc-spiral-scan.in").read_text().replace(
    "nosym = .true., noinv = .true.", "noinv = .true."
).replace("spiral_q(3) = 0.0", "spiral_q(3) = 0.125")

#: A zigzag hydrogen chain along ``z``: two atoms per cell related by the
#: screw ``{C2z | c/2}`` and the glide ``{m_x | c/2}``, so two of the four
#: operations carry ``q . f = 2 pi (1/4)(1/2)``, the one arm where the
#: four-scalar rule and a leftover spin rotation read differently. The atoms
#: are 5 bohr apart along the chain, ``h-chain-spiral.in``'s spacing: at 4 bohr
#: the chain converges nonmagnetic (``|m| = 2e-5``, measured).
ZIGZAG = """\
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 6, celldm(1) = 12.0, celldm(3) = 0.83333333333333,
    nat = 2, ntyp = 1, ecutwfc = 25.0,
    occupations = 'smearing', smearing = 'gaussian', degauss = 0.10
    noncolin = .true.
    starting_magnetization(1) = 0.6
    angle1(1) = 90.0
    spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = 0.25
    nbnd = 8
 /
 &electrons
    mixing_beta = 0.3
    conv_thr = 1.0d-11
 /
ATOMIC_SPECIES
 H  1.008  H.pz-vbc.UPF
ATOMIC_POSITIONS (crystal)
 H  0.06 0.0 0.0
 H -0.06 0.0 0.5
K_POINTS {automatic}
 3 3 4 0 0 0
"""


#: The hydrogen chain every spiral test runs on, with symmetry and an in-plane
#: grid: ``C4v`` about the chain, ``f = 0``, the one cell known to be magnetic
#: at every ``q`` (``h-chain-spiral.in``).
CHAIN = (ROOT / "data" / "qe" / "h-chain-spiral.in").read_text().replace(
    "    nosym = .true.\n", ""
).replace(" 1 1 4 0 0 0", " 3 3 4 0 0 0")


def _nosym(text: str) -> str:
    if "noinv = .true." in text:
        return text.replace("noinv = .true.", "noinv = .true., nosym = .true.")
    return text.replace("    noncolin = .true.", "    noncolin = .true.\n    nosym = .true.")


def _calculation(text: str):
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, Calculation(system, pseudos)


@pytest.mark.parametrize("text, nsym, nk_sym, nk_full", [
    (FCC, 6, 20, 64),
    (ZIGZAG, 4, 16, 36),
    (CHAIN, 8, 12, 36),
], ids=["fcc", "zigzag", "chain"])
def test_the_spin_space_group_and_its_wedge(text, nsym, nk_sym, nk_full):
    """The stabiliser of ``q`` that keeps the seed, counted by hand, and its wedge.

    ``C3v`` for fcc with ``q`` along a body diagonal, and ``{E, C2z|c/2, m_x|c/2,
    m_y}`` for the zigzag chain. The k-set shrinks from the full grid to the
    wedge with no time reversal, which sends ``q`` to ``-q``.
    """
    system = build_system(parse_pw_input(text))
    full = build_system(parse_pw_input(_nosym(text)))
    group = system.symmetry_group()
    assert group.nsym == nsym
    assert not np.any(group.t_rev_array())
    assert system.kpoints.nk == nk_sym and system.kpoints.reduced
    assert full.kpoints.nk == nk_full
    assert np.sum(system.kpoints.weights) == pytest.approx(np.sum(full.kpoints.weights))
    # ``grid_symmetry`` is the triple an NSCF grid and the tetrahedra reduce
    # with, and it must be the same group with time reversal off.
    rotations, time_reversal, _ = system.grid_symmetry()
    assert len(rotations) == nsym and not time_reversal


def test_the_zigzag_group_carries_a_spiral_phase():
    """The screw and the glide have ``q . f`` of 45 degrees, so the check is not idle."""
    system = build_system(parse_pw_input(ZIGZAG))
    group = system.symmetry_group()
    phases = 2 * np.pi * group.translation_array() @ np.asarray(system.spiral_q)
    assert sorted(np.round(np.degrees(phases), 6)) == [0.0, 0.0, 45.0, 45.0]



def _relative(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("text", [FCC, ZIGZAG], ids=["fcc", "zigzag"])
def test_the_seed_is_invariant_under_four_scalars_and_not_otherwise(text):
    """Round-off under the identity, and order one under either wrong rule.

    The axial rule is ``det(S) S`` on the components, which an ordinary
    magnetic run uses; the leftover phase is ``Rot_z(q . f)``, which is what
    the laboratory spin rotation would leave behind if the spiral's phase did
    not cancel it. On fcc ``f = 0`` and the second rule is the first, so only
    the axial one is tested there.
    """
    system, calculation = _calculation(text)
    assert calculation.use_symmetry
    rho = calculation.starting_density()
    charge, moment = calculation.symmetry_residual(rho)
    assert charge < 1e-12 and moment < 1e-12

    dense = calculation.basis.dense
    maps = calculation._symmetry_maps

    def residual(rotations):
        out = _symmetrize_noncollinear(rho, dense.fft_index, dense.grid, maps,
                                       jnp.asarray(rotations))
        return _relative(out[1:4], rho[1:4])

    group = calculation.symmetries
    axial = (magnetization_signs(system.cell, group)[:, None, None]
             * cartesian_rotations(system.cell, group))
    assert residual(axial) > 0.1

    phases = 2 * np.pi * group.translation_array() @ np.asarray(system.spiral_q)
    if np.any(np.abs(phases) > 1e-8):
        c, s = np.cos(phases), np.sin(phases)
        leftover = np.zeros((len(phases), 3, 3))
        leftover[:, 0, 0] = c
        leftover[:, 0, 1] = -s
        leftover[:, 1, 0] = s
        leftover[:, 1, 1] = c
        leftover[:, 2, 2] = 1.0
        assert residual(leftover) > 0.1


def test_what_a_symmetrised_spiral_still_refuses():
    """Each regime the four-scalar rule was not written for, refused by name."""
    system, calculation = _calculation(ZIGZAG)

    # Another q has another stabiliser, and dE/dq on the wedge is not completed.
    with pytest.raises(NotImplementedError, match="nosym"):
        calculation.at_spiral_q((0.0, 0.0, 0.3))

    # A k-set reduced with another group: the zigzag chain's k-set handed to a
    # spiral along x, whose stabiliser is a different subgroup.
    moved = dataclasses.replace(system, spiral_q=(0.25, 0.0, 0.0))
    pseudos = calculation.pseudos
    with pytest.raises(NotImplementedError, match="nosym"):
        Calculation(moved, pseudos)

    # An explicit k-list cannot be checked to be the wedge.
    listed = ZIGZAG.replace(
        "K_POINTS {automatic}\n 3 3 4 0 0 0\n",
        "K_POINTS {crystal}\n 2\n 0.0 0.0 0.0 1.0\n 0.0 0.0 0.25 1.0\n",
    )
    with pytest.raises(NotImplementedError, match="nosym"):
        _calculation(listed)

    # A fully-relativistic dataset, whose only use is the coupling on top.
    with pytest.raises(ValueError, match="nosym"):
        build_system(parse_pw_input(
            ZIGZAG.replace("    noncolin = .true.",
                           "    noncolin = .true.\n    lspinorb = .true.\n"
                           "    soc_scale = 0.0")
        ))


def test_a_k_set_that_is_already_the_wedge_is_accepted():
    """The k-set check compares points, not provenance.

    A spiral ``System`` replaced to the same ``q`` carries the reader's own
    wedge, and is built rather than refused.
    """
    system, _ = _calculation(ZIGZAG)
    same = dataclasses.replace(system, spiral_q=tuple(system.spiral_q))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in same.structure.species)
    assert Calculation(same, pseudos).use_symmetry
