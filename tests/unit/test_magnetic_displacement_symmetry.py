"""The displacement response of a magnetic spinor, symmetrised on a wedge.

A perturbation that moves atom ``a`` along ``i`` induces a density with a
charge and, at ``nspin_mag = 4``, three magnetization components. The
direction is a polar index and the magnetization an axial one, so the group
average carries the plain rotation on the first and ``det(R) (-1)^t_rev`` on
the second, beside the atom permutation (I79). Cheap: no SCF, the
symmetriser reads only the calculation's group.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.system import build_system

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _calculation(case: str):
    from defuqumat.scf import Calculation

    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    return Calculation(system, pseudos)


def _random_fields(calculation, seed=0):
    """A displacement-response-shaped field with no symmetry of its own.

    Smooth on the grid (a few plane waves per component) so that the round trip
    through the G sphere is exact and the projector test below reads the
    average, not the cutoff.
    """
    nat = calculation.system.structure.nat
    grid = tuple(calculation.basis.dense.grid)
    rng = np.random.default_rng(seed)
    axes = [np.arange(n) / n for n in grid]
    r = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    fields = np.zeros((nat, 3, calculation.nspin_mag) + grid)
    for miller in [(0, 0, 1), (1, 0, 0), (0, 1, 2), (1, 1, 1)]:
        phase = 2.0 * np.pi * (r @ np.asarray(miller))
        a = rng.normal(size=fields.shape[:3])
        b = rng.normal(size=fields.shape[:3])
        fields += (a[..., None, None, None] * np.cos(phase)
                   + b[..., None, None, None] * np.sin(phase))
    return jnp.asarray(fields)


def test_the_textured_displacement_average_is_a_projector():
    """Applied twice it gives what it gave once, and it is not the identity.

    A wrong atom label (``irt`` where its inverse belongs) or a mismatched
    pair of rotations leaves an average over a set that is not a group action,
    and that is not idempotent: this is the reference-free half of the check,
    the wedge against the whole zone being the other
    (``tests/regression/test_spinor_phonons.py``). What it cannot see is the
    axial sign itself, since a consistent wrong sign is a projector onto a
    different representation.
    """
    calculation = _calculation("h4-cycloid-90")
    assert calculation.nspin_mag == 4 and calculation.use_symmetry
    assert calculation.symmetries.nsym > 1
    fields = _random_fields(calculation)
    once = calculation.symmetrize_atom_displacement(fields)
    twice = calculation.symmetrize_atom_displacement(once)
    scale = float(jnp.abs(once).max())
    assert float(jnp.abs(twice - once).max()) < 1e-12 * max(scale, 1.0)
    # The group does something to an arbitrary field, to the magnetization
    # channels as well as the charge, or the test above could not fail.
    assert float(jnp.abs(once - fields)[:, :, 0].max()) > 1e-2
    assert float(jnp.abs(once - fields)[:, :, 1:].max()) > 1e-2


def test_the_magnetization_is_rotated_as_an_axial_vector():
    """The three channels are not averaged as scalars.

    The scalar average of each magnetization channel is a projector too, so
    the two are told apart by value: on an arbitrary field they differ by an
    amount of the order of the field.
    """
    calculation = _calculation("h4-cycloid-90")
    fields = _random_fields(calculation, seed=1)
    axial = calculation.symmetrize_atom_displacement(fields)
    nat = calculation.system.structure.nat
    as_scalars = jnp.stack([
        calculation.symmetrize_atom_displacement(
            jnp.repeat(fields[:, :, c:c + 1], 4, axis=2))[:, :, 0]
        for c in range(4)
    ], axis=2)
    assert as_scalars.shape == (nat, 3, 4) + tuple(calculation.basis.dense.grid)
    # The charge channel is the same rule on both sides.
    np.testing.assert_allclose(axial[:, :, 0], as_scalars[:, :, 0], atol=1e-12)
    assert float(jnp.abs(axial[:, :, 1:] - as_scalars[:, :, 1:]).max()) > 1e-2


def test_the_walked_average_is_the_batched_one(monkeypatch):
    """The scan over operations, taken when the gather passes its budget."""
    calculation = _calculation("h4-cycloid-90")
    fields = _random_fields(calculation, seed=2)
    batched = calculation.symmetrize_atom_displacement(fields)
    monkeypatch.setenv("DEFUQUMAT_GATHER_BUDGET", "0")
    walked = calculation.symmetrize_atom_displacement(fields)
    np.testing.assert_allclose(walked, batched, atol=1e-12)
