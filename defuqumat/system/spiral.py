"""Spin spirals: the generalized Bloch theorem, and what it does to the k-points.

A flat spin spiral is a magnetization that turns by ``q . R`` from one cell to
the next,

    m(r + R) = Rot_z(q . R) m(r),

which is not periodic and would seem to need a supercell -- one commensurate
with ``q``, which for a general ``q`` does not exist. It does not need one.
Without spin-orbit coupling the Hamiltonian is invariant under a lattice
translation *combined* with the matching spin rotation, and the eigenstates can
be labelled by ``k`` again (Sandratskii, J. Phys. Condens. Matter 3, 8565
(1991)). Elk's manual (§5.146, ``vqlss``) writes the resulting spinor as

    Psi^q_k(r) = ( U_up(r) e^{i(k + q/2).r},  U_dn(r) e^{i(k - q/2).r} )

with ``U_up``, ``U_dn`` lattice periodic, and the magnetization it produces as

    m^q(r) = ( m_x(r) cos(q.r),  m_y(r) sin(q.r),  m_z(r) )

with ``m_x``, ``m_y``, ``m_z`` lattice periodic. **Everything the SCF touches is
one of those periodic functions**, so the density, the potential, the
exchange-correlation functional and the mixer are untouched by the spiral; what
changes is which plane waves each spinor component is built from.

**That is the whole implementation.** ``gengkqvec.f90`` in Elk puts the up
component at ``k + q/2`` and the down at ``k - q/2``, each with its own ``G+k``
set, and this module produces exactly that: one k-list of length ``2 nk``, the
up component's points first. Building both halves in *one* call to
``build_plane_wave_basis`` is not an optimisation -- it is what gives the two
components a common ``npwx``, so rule R7's padding, the ``vmap`` over k and the
stick layout all keep working unchanged.

**The shifted points are kept literal.** ``k + q/2`` is not wrapped back into
the first Brillouin zone: the wrapped point is the same physics through a
different G-index map, and the phase that relates them is exactly the trap P16
records for the zone edge (``u_{k+b}(G) = u_k(G+b)``). Nothing here needs the
wrapped form, so nothing wraps it.

**Symmetry is the spin space group, and it acts on four scalars.** Without
spin-orbit coupling a spatial operation does not turn the spin, so the spiral's
symmetries are pairs (a space-group operation ``{S|f}``, a spin rotation of its
own). Only the operations with ``S q = q`` survive (Elk checks this in
``findsymlat.f90``): for those, the spiral phase at the image point is the one
at the source plus ``q . f``, so the operation is a symmetry of the spiral
state together with the laboratory spin rotation ``Rot_z(q . f)``, and on the
*rotated-frame* fields the SCF holds -- ``n``, ``m_x``, ``m_y``, ``m_z``, all
lattice periodic -- the two cancel and the operation acts as on four scalars,
with no ``det(S) S`` turning the magnetization. :func:`spiral_symmetries` keeps
the operations that map every atom's starting moment (and applied field) onto
its image's *unrotated*, which is Elk's ``findsym.f90`` with the identity as the
global spin rotation, and the k-set is reduced with them and **without** time
reversal, which sends ``q`` to ``-q``. A spiral ``nosym`` run is the check.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from defuqumat.system.cell import Cell
from defuqumat.system.kpoints import KPoints
from defuqumat.system.symmetry import (
    _MAGNETIC_TOLERANCE,
    Symmetries,
    _axial_fields,
    atom_mapping,
    cartesian_rotations,
)

__all__ = ["spiral_kpoints", "spiral_kcart", "invariant_operations",
           "spiral_symmetries", "spiral_cartesian"]


def spiral_cartesian(q_crystal, cell: Cell) -> np.ndarray:
    """The spiral wavevector in cartesian units of ``2 pi / alat``.

    The input is in *lattice* coordinates, as Elk's ``vqlss`` is: a spiral at
    ``q = (0, 0, 1/2)`` doubles the magnetic period along the third lattice
    vector whatever the cell's shape, which is what makes that the useful way to
    say it.
    """
    return np.asarray(cell.k_to_cartesian(np.asarray(q_crystal, dtype=float)))


def spiral_kpoints(kpoints: KPoints, q_crystal, cell: Cell) -> KPoints:
    """The ``2 nk`` shifted points a spiral needs: ``k + q/2`` then ``k - q/2``.

    The weights are carried through unchanged in both halves. They are not used
    as integration weights on this list -- the *state* at ``k`` is one object
    with two components, and its weight is the original one -- but keeping them
    means the doubled list is a perfectly ordinary :class:`KPoints` that
    ``build_plane_wave_basis`` and the projectors accept without a special case.
    """
    half = 0.5 * spiral_cartesian(q_crystal, cell)
    coords = np.asarray(kpoints.coords)
    shifted = np.concatenate([coords + half, coords - half])
    weights = np.concatenate([np.asarray(kpoints.weights)] * 2)
    return KPoints.from_cartesian(shifted, weights, precision=kpoints.precision)


def spiral_kcart(kpoints: KPoints, q_crystal, cell: Cell) -> jnp.ndarray:
    """The same ``2 nk`` shifted points as :func:`spiral_kpoints`, in 1/bohr.

    The difference is that this one is **traceable in** ``q``: it does the
    arithmetic in JAX and never builds a :class:`KPoints`, whose constructor
    goes through ``np.asarray``. It is what makes ``|k+G|^2`` and ``vkb`` --
    and therefore the total energy -- differentiable functions of the spiral
    wavevector, which is the whole of ``dE/dq`` (:mod:`defuqumat.forces.spiral`).

    The *sphere* each point selects is not a function of ``q`` here: it is
    chosen once, host-side, and held fixed. That is exact rather than an
    approximation between the discrete values of ``q`` at which a plane wave
    crosses the cutoff -- see :mod:`defuqumat.forces.spiral` for what happens at
    them.
    """
    half = 0.5 * cell.k_to_cartesian(jnp.asarray(q_crystal))
    coords = jnp.asarray(kpoints.coords)
    return jnp.concatenate([coords + half, coords - half]) * cell.tpiba


def invariant_operations(symmetries: Symmetries, q_crystal) -> Symmetries:
    """The operations that leave the spiral wavevector alone, ``S^T q = q``.

    Elk's check, in ``findsymlat.f90``, and the *first* condition a spin space
    group imposes -- not the only one: :func:`spiral_symmetries` adds the
    moments. ``q`` is a reciprocal-space vector in crystal coordinates, so it
    transforms like a Miller index: ``q' = M q``.
    """
    q = np.asarray(q_crystal, dtype=float)
    kept, translations, t_rev = [], [], []
    for index, rotation in enumerate(symmetries.rotation_array()):
        if np.max(np.abs(rotation @ q - q)) > 1.0e-6:
            continue
        kept.append(symmetries.rotations[index])
        translations.append(symmetries.translations[index])
        t_rev.append(int(symmetries.t_rev_array()[index]))
    return Symmetries(
        rotations=tuple(kept),
        translations=tuple(translations),
        time_reversed=tuple(t_rev),
    )


def spiral_symmetries(cell: Cell, structure, symmetries: Symmetries, q_crystal,
                      fields=()) -> Symmetries:
    """The spin space group of a flat spiral, as the operations it keeps.

    ``symmetries`` is the crystal's group from positions alone
    (:func:`~defuqumat.system.symmetry.find_symmetries`), **not** the magnetic
    one: :func:`~defuqumat.system.symmetry.magnetic_symmetries` tests the
    moments as axial vectors turned by ``det(S) S`` and keeps operations with
    ``t_rev = 1``, and both are wrong on the rotated-frame magnetization, which
    an operation of the spin space group permutes without turning (see the
    module docstring).

    Two conditions, both exact rather than modulo a reciprocal vector:

    * ``S q = q``, tested on the *cartesian* rotation, so that whether the
      crystal-coordinate matrix acts on ``q`` or its transpose does is never a
      question. ``S q = q + G`` would put the up component's image on the
      sphere at ``S k + q/2 + G/2``, which is not one of the run's spheres.
    * every per-atom vector in ``fields`` (the starting moments, the
      ``LOCAL_MAGNETIC_FIELDS`` card, the uniform field tiled; each normalised
      as :func:`~defuqumat.system.symmetry.magnetic_symmetries` does) equals
      its image's **as it is**, all three components, which is
      :func:`~defuqumat.system.symmetry.collinear_symmetries`' permutation test
      on a vector. Elk's ``findsym.f90`` tries the global spin rotations in the
      lattice group with the identity first; this keeps the identity only, so
      the group can be a subgroup of Elk's and is never larger.

    No operation is time-reversed: ``t_rev`` is zero throughout, and the k-set
    is reduced with ``time_reversal = False``.
    """
    q = np.asarray(q_crystal, dtype=float)
    if symmetries.nsym <= 1:
        return symmetries
    qcart = np.asarray(spiral_cartesian(q, cell), dtype=float)
    rotations = cartesian_rotations(cell, symmetries)
    scale = max(float(np.linalg.norm(qcart)), 1.0)
    vectors = _axial_fields(fields, structure.nat)
    mapping = atom_mapping(cell, structure, symmetries) if vectors else None

    kept, translations = [], []
    for s, rotation in enumerate(rotations):
        if np.max(np.abs(rotation @ qcart - qcart)) > 1.0e-8 * scale:
            continue
        if any(np.max(np.abs(field[mapping[s]] - field)) >= _MAGNETIC_TOLERANCE
               for field in vectors):
            continue
        kept.append(symmetries.rotations[s])
        translations.append(symmetries.translations[s])
    return Symmetries(rotations=tuple(kept), translations=tuple(translations))
