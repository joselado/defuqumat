"""Completing a sum over a symmetry-reduced k-set for a quantity with free labels.

A Brillouin-zone sum over the irreducible wedge is exact for a scalar and for
nothing else: a quantity with a cartesian label (a conductivity, a shift
current, a second-harmonic tensor) or a reciprocal-lattice label (``chi_0(G,
G')``) comes out of the wedge as the sum over the wedge alone, and the
components the point group forbids are a residue of the reduction rather than
physics. The completion is an average over the group the wedge was cut with,

    X = (1/N) sum_g g.X_wedge,

which is exactly the sum over the whole grid whenever the wedge weights are the
star multiplicities of a set the group maps onto itself: each wedge point ``k``
of weight ``w_k`` and orbit ``O_k`` contributes ``w_k / |O_k|`` to every point
of its orbit, and when that distribution is uniform over the union of the
orbits, the average *is* the uniform sum over that union. :func:`wedge_group`
checks exactly that before it returns anything, so a k-set it accepts is one the
average completes and a k-set it cannot vouch for is refused by name.

**Time reversal is an element of the group, not a footnote to it.** QE reduces
a non-magnetic grid with ``k -> -k`` on top of the point group (``setup.f90``'s
``time_reversal = .NOT. noinv .AND. .NOT. magnetic_sym``), and an
anti-unitary operation does not act on a response the way a rotation does:
the per-k Kubo-Greenwood term at ``-k`` is the *transpose* of the one at ``k``
(``v_nm(-k) = -conj(v_nm(k))``), so the antisymmetric part of ``sigma_ab`` is
odd under it, while the shift-current integrand ``Im[r^b_mn r^{c;a}_nm]`` is
even. Each consumer therefore says what an anti-unitary image does to its own
object; this module only supplies the group and the average.

**What is still refused.** A k-set on a ``nosym`` run that is nevertheless
reduced (it was cut by a group the run does not carry), a set the group does not
tile uniformly (a shifted Monkhorst-Pack grid is not closed under the point
group, which is P24's rule), and a **magnetic** group -- one whose operations
carry ``t_rev``, or a noncollinear run with a moment -- for which no consumer
has a measured number yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from defuqumat.system.kpoints import is_reduced

__all__ = ["WedgeGroup", "wedge_group"]

#: Crystal coordinates are compared after rounding to this many decimals; a
#: k-point of a Monkhorst-Pack grid is a rational with a small denominator, so
#: anything finer than the grid is round-off.
_DECIMALS = 6


@dataclass(frozen=True)
class WedgeGroup:
    """The operations a k-set was reduced with, as they act on a response.

    Attributes:
        rotations: ``(nop, 3, 3)`` cartesian rotations, acting on column
            vectors (:func:`~defuqumat.system.symmetry.cartesian_rotations`).
        antiunitary: ``(nop,)``, whether each operation carries time reversal.
            With ``time_reversal`` the group is the point group twice over, the
            second copy anti-unitary.
        crystal: ``(nop, 3, 3)`` the same operations in crystal axes, as
            :meth:`~defuqumat.system.symmetry.Symmetries.rotation_array` gives
            them (integer matrices on the direct lattice).
        translations: ``(nop, 3)`` the fractional translations in crystal
            coordinates.
    """

    rotations: np.ndarray
    antiunitary: np.ndarray
    crystal: np.ndarray
    translations: np.ndarray

    @property
    def order(self) -> int:
        return int(len(self.rotations))

    @property
    def has_antiunitary(self) -> bool:
        return bool(np.any(self.antiunitary))

    def average(self, tensor, axes, antiunitary_image=None) -> np.ndarray:
        """``(1/N) sum_g R...R (A(X) if g is anti-unitary else X)``.

        Args:
            tensor: any array, real or complex, ``axes`` of which are cartesian
                (length 3); the others (a frequency axis) are carried.
            axes: the cartesian axes, each rotated by every operation.
            antiunitary_image: what an anti-unitary operation does to ``tensor``
                *before* its rotation -- a transpose of two labels for a
                conductivity, nothing for a quantity even under time reversal.
                ``None`` means the identity.
        """
        tensor = np.asarray(tensor)
        axes = [a % tensor.ndim for a in axes]
        image = tensor if antiunitary_image is None else np.asarray(
            antiunitary_image(tensor))
        total = np.zeros_like(tensor, dtype=np.result_type(tensor, image, float))
        for rotation, anti in zip(self.rotations, self.antiunitary):
            term = image if anti else tensor
            for axis in axes:
                term = np.moveaxis(
                    np.tensordot(rotation, term, axes=([1], [axis])), 0, axis)
            total = total + term
        return total / self.order


def _orbit_weights(system, kpoints, rotations, antiunitary) -> np.ndarray:
    """The weight each point of the union of the orbits receives, ``(npoints,)``."""
    cell = system.cell
    cart = np.asarray(kpoints.coords, dtype=float)
    weights = np.asarray(kpoints.weights, dtype=float)
    signs = np.where(antiunitary, -1.0, 1.0)
    # (nop, nk, 3): an anti-unitary operation sends k to -R k.
    images = signs[:, None, None] * np.einsum("sij,kj->ski", rotations, cart)
    crystal = np.asarray(cell.k_to_crystal(images.reshape(-1, 3)),
                         dtype=float).reshape(images.shape)
    crystal = np.round(crystal - np.floor(crystal + 0.5 * 10.0**-_DECIMALS),
                       _DECIMALS)
    crystal = np.where(crystal >= 1.0, crystal - 1.0, crystal)
    accumulated: dict = {}
    for k in range(cart.shape[0]):
        orbit = {tuple(p) for p in crystal[:, k]}
        share = weights[k] / len(orbit)
        for point in orbit:
            accumulated[point] = accumulated.get(point, 0.0) + share
    return np.asarray(list(accumulated.values()))


def wedge_group(system, kpoints=None, *, quantity: str,
                scalar: bool = False) -> WedgeGroup | None:
    """The group a reduced k-set was cut with, or ``None`` for a whole one.

    ``kpoints`` defaults to the system's own. Raises
    :class:`NotImplementedError`, naming ``quantity``, for every k-set the
    average of :class:`WedgeGroup` could not complete -- see the module
    docstring for the three cases.

    ``scalar`` is for a quantity every operation leaves alone (a band energy, a
    total energy), whose wedge sum needs no average at all, only the tiling
    check: a magnetic group is then accepted, its ``t_rev`` operations sending
    ``k`` to ``-R k`` as anti-unitary ones do.
    """
    from defuqumat.system.symmetry import cartesian_rotations

    kpoints = system.kpoints if kpoints is None else kpoints
    if not is_reduced(kpoints):
        return None
    if getattr(system, "nosym", False):
        raise NotImplementedError(
            f"{quantity} on a symmetry-reduced k-set of a nosym run is refused: "
            "the set was cut by a group this run does not carry, so there is "
            "nothing to complete the wedge sum with. Pass the whole grid"
        )
    symmetries = system.symmetry_group()
    t_rev = np.asarray(symmetries.t_rev_array(), dtype=int)
    magnetic = bool(np.any(t_rev != 0)) or (
        bool(getattr(system, "noncolin", False)) and bool(getattr(system, "domag", False))
    )
    if magnetic and not scalar:
        raise NotImplementedError(
            f"{quantity} on a symmetry-reduced k-set of a magnetic run is not "
            "implemented: the wedge of a magnetic group carries operations "
            "combined with time reversal, and no magnetic cell has been measured "
            "against its whole grid here (the non-magnetic wedge has). Pass the "
            "whole unshifted grid as kpoints="
        )
    _, time_reversal, _ = system.grid_symmetry()
    rotations = np.asarray(cartesian_rotations(system.cell, symmetries), dtype=float)
    crystal = np.asarray(symmetries.rotation_array(), dtype=float)
    translations = np.asarray(symmetries.translations, dtype=float).reshape(-1, 3)
    antiunitary = t_rev != 0
    if time_reversal:
        rotations = np.concatenate([rotations, rotations])
        crystal = np.concatenate([crystal, crystal])
        translations = np.concatenate([translations, translations])
        antiunitary = np.concatenate([antiunitary, ~antiunitary])

    shares = _orbit_weights(system, kpoints, rotations, antiunitary)
    if np.ptp(shares) > 1.0e-8 * np.max(np.abs(shares)):
        raise NotImplementedError(
            f"{quantity} on this symmetry-reduced k-set is refused: the "
            f"{int(len(rotations))} operations it would be completed with do not "
            "tile it uniformly (the weight each point of the unfolded set "
            f"receives spreads from {shares.min():.6g} to {shares.max():.6g}), "
            "so the set was not cut by this group from a grid the group maps "
            "onto itself -- which is what a *shifted* Monkhorst-Pack grid is. "
            "Use an unshifted grid, or pass the whole grid as kpoints="
        )
    grid = getattr(kpoints, "grid", None)
    if not scalar and grid is not None and len(shares) != int(np.prod(grid)):
        # Uniform over the union of the orbits is not enough: a shifted grid's
        # wedge unfolds uniformly onto its star closure, which is a different
        # and larger sample than the grid (256 points from 4 4 4 1 1 1 on fcc
        # silicon), so the completed sum would be over points the run did not
        # ask for. Measured on that cell: the completion agrees with the
        # explicit closure (chi_0 4.2e-6, STS 2.6e-8) and is 0.41 and 0.14 off
        # the 64-point shifted grid.
        raise NotImplementedError(
            f"{quantity} on this symmetry-reduced k-set is refused: its "
            f"{int(len(rotations))} operations unfold the wedge onto "
            f"{len(shares)} points and its grid {tuple(int(n) for n in grid)} has "
            f"{int(np.prod(grid))}, so the grid is not closed under the group -- "
            "which is what a *shifted* Monkhorst-Pack grid is (P24) -- and the "
            "completed sum would be over the star closure rather than the grid. "
            "Use an unshifted grid, or pass the whole grid as kpoints="
        )
    return WedgeGroup(rotations=rotations, antiunitary=antiunitary,
                      crystal=crystal, translations=translations)
