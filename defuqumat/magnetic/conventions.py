"""The conventions of a uniform orbital magnetic field, as code: one gauge, one sign, one k label.

Every route of the orbital-field work (Peierls on Wannier hoppings in
:mod:`defuqumat.magnetic.peierls`, the magnetic ultracell, the plane-wave magnetic Bloch
route) imports these and never redefines them, so that a comparison between routes
compares operators, not conventions. The derivations are in
``notes/ORBITAL-FIELD-NEXT.md`` section 12 and the design's section 0.

Units. Rydberg atomic units: ``hbar = 1``, ``m = 1/2``, ``e^2 = 2``, bohr, Ry. The field
enters only as the flux number ``n_phi = p/q`` per chemical cell (``p``, ``q`` coprime
Python ints, static) and as the vector potential in phase units, ``a(r) = (e/hbar c) A(r)``
in 1/bohr, so ``pi = -i grad + a`` and the kinetic operator is ``pi^2`` in Ry. Tesla
appears only in :mod:`defuqumat.units` (``flux_per_cell``, ``tesla_from_flux``,
``rational_flux``).

Gauge. Landau, ``a(r) = 2 pi n_phi (b1 . r) b2`` with ``b_i`` the *chemical* cell's scaled
reciprocal vectors (``a_i . b_j = delta_ij``). It is Lu et al.'s ``A_Lan`` (PRL 135, 196401
(2025), arXiv:2505.06585, eq. 4) written on the chemical cell. ``curl a = 2 pi n_phi a3 /
Omega``, so the field is along ``a3`` by construction, and ``B_ph = 2 pi n_phi / area`` in
1/bohr^2 with ``area = Omega / |a3|``. ``div a = 2 pi n_phi (b1 . b2)`` is not zero on a
hexagonal cell, so a kinetic term is never ``2 a.p + a^2``.

Sign. Electron: a hop from the ket at ``r_nu`` to the bra at ``r_mu`` carries
``exp(-i a(mid) . (r_mu - r_nu))``, the line integral along the straight bond, exact for a
linear ``a``; a counterclockwise loop seen from ``+a3`` of signed area ``S`` chemical cells
picks up ``exp(-i 2 pi n_phi S)``.

Magnetic translations. ``T(R) = exp(-i a(R) . r) t(R)``, ``T(R1) T(R2) = exp(i a(R2) . R1)
T(R1 + R2)``. The magnetic cell is ``q a1 x a2 x a3``; ``k = 2 pi sum_i kappa_i b_i^m`` with
``b^m = (b1/q, b2, b3)`` the magnetic cell's scaled reciprocal vectors. The Peierls Bloch
sum is the review's pair (i): ``exp(-i a(mid).(r_mu - r_nu)) exp(i k.P) exp(-i a(P).c_nu)``,
``c_nu`` the ket's position inside the magnetic cell (the column index; Wang et al.,
arXiv:2609.01053, eq. 9 prints the row and that is a typo). Wang and Vafek, PRB 108,
245109 (2023), arXiv:2303.16347, is the magnetic-ultracell construction these labels
serve.

GIPAW. A projector or a localized orbital at ``tau`` is dressed by ``exp(-i a(tau) . r - i
f(r - tau))``, ``f(d) = a(d) . d / 2``; the constant ``exp(-i a(tau) . tau)`` is dropped
(it cancels in ``|p~><p~|``), which leaves the phase ``-a(tau) . d - a(d) . d / 2`` of
:meth:`MagneticCell.gipaw_phase` on the minimum-image ``d``.

Differentiability. ``a`` carries a traced ``scale`` (default one) that multiplies it
everywhere, so ``d/dB`` at a fixed ``p/q`` survives a ``jax.grad`` in ``scale``; ``p``,
``q`` and the enlarged axis are static because they set array shapes.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from defuqumat.system.cell import Cell
from defuqumat.units import TPI, tesla_from_flux

__all__ = ["MagneticCell", "minimum_image", "signed_area", "refuse_regime"]


class MagneticCell(eqx.Module):
    """A chemical cell at flux ``p/q`` per cell, Landau gauge, magnetic cell ``q a1 x a2 x a3``.

    ``cell`` is the chemical :class:`~defuqumat.system.cell.Cell` (bohr); ``p`` and ``q``
    coprime positive ints; ``axis`` the enlarged lattice vector (0 only); ``scale`` a
    traced real factor on ``a`` (one by default) for a derivative in the field.
    """

    cell: Cell
    scale: jnp.ndarray
    p: int = eqx.field(static=True)
    q: int = eqx.field(static=True)
    axis: int = eqx.field(static=True, default=0)

    @classmethod
    def build(cls, cell, p: int, q: int, *, axis: int = 0, scale=1.0) -> "MagneticCell":
        """From a :class:`Cell` or three lattice vectors in bohr (rows), checking ``p/q``."""
        if not isinstance(cell, Cell):
            cell = Cell.from_vectors(np.asarray(cell, dtype=float))
        p, q = int(p), int(q)
        if q < 1 or p < 1:
            raise ValueError(f"the flux p/q needs positive p and q, got {p}/{q} "
                             "(a field along -a3 is the cell with a3 reversed)")
        if math.gcd(p, q) != 1:
            raise ValueError(f"p/q = {p}/{q} is not reduced; take {p // math.gcd(p, q)}/"
                             f"{q // math.gcd(p, q)}")
        if axis != 0:
            raise NotImplementedError("only the magnetic cell enlarged along a1 is implemented "
                                      f"(axis = 0), not axis = {axis}")
        return cls(cell=cell, scale=cell.precision.as_real(scale), p=p, q=q, axis=axis)

    # --- the cells -----------------------------------------------------------
    @property
    def lattice(self) -> jnp.ndarray:
        """``(3, 3)`` chemical lattice vectors, rows, bohr."""
        return self.cell.at

    @property
    def b_chem(self) -> jnp.ndarray:
        """``(3, 3)`` scaled reciprocal vectors of the chemical cell, rows, ``a_i . b_j = delta_ij``."""
        return jnp.linalg.inv(self.cell.at).T

    @property
    def magnetic_lattice(self) -> jnp.ndarray:
        """``(3, 3)`` rows ``q a1, a2, a3`` in bohr."""
        return self.cell.at.at[0].multiply(self.q)

    @property
    def magnetic_reciprocal(self) -> jnp.ndarray:
        """``(3, 3)`` rows ``b1/q, b2, b3``, the magnetic cell's scaled reciprocal vectors."""
        return self.b_chem.at[0].divide(self.q)

    @property
    def area(self) -> jnp.ndarray:
        """The chemical cell's area normal to ``a3``, ``Omega / |a3|``, bohr^2."""
        return self.cell.volume / jnp.linalg.norm(self.cell.at[2])

    @property
    def flux(self) -> float:
        """``n_phi = p/q``, the flux quanta through one chemical cell (without ``scale``)."""
        return self.p / self.q

    @property
    def b_ph(self) -> jnp.ndarray:
        """``|curl a| = 2 pi n_phi scale / area`` in 1/bohr^2; the free Landau spacing is ``2 B_ph`` Ry."""
        return TPI * self.flux * self.scale / self.area

    @property
    def field_tesla(self) -> float:
        """The field in tesla, for reporting (host floats, ``scale`` included)."""
        return float(tesla_from_flux(self.p, self.q, float(self.area)) * float(self.scale))

    # --- the gauge -----------------------------------------------------------
    def a(self, r) -> jnp.ndarray:
        """``a(r) = 2 pi n_phi scale (b1 . r) b2``, ``r`` cartesian bohr ``(..., 3)`` -> 1/bohr."""
        r = jnp.asarray(r, dtype=self.cell.at.dtype)
        b = self.b_chem
        return (TPI * self.flux * self.scale) * (r @ b[0])[..., None] * b[1]

    def kvector(self, kappa) -> jnp.ndarray:
        """``k = 2 pi sum_i kappa_i b_i^m`` in 1/bohr; ``kappa`` ``(..., 3)`` or ``(..., 2)``."""
        kappa = jnp.asarray(kappa, dtype=self.cell.at.dtype)
        if kappa.shape[-1] == 2:
            kappa = jnp.concatenate([kappa, jnp.zeros(kappa.shape[:-1] + (1,), kappa.dtype)], -1)
        return TPI * kappa @ self.magnetic_reciprocal

    def peierls_phase(self, r_from, r_to) -> jnp.ndarray:
        """``-a(mid) . (r_to - r_from)``: the phase of a hop from ``r_from`` (ket) to ``r_to`` (bra)."""
        r_from = jnp.asarray(r_from, dtype=self.cell.at.dtype)
        r_to = jnp.asarray(r_to, dtype=self.cell.at.dtype)
        mid = 0.5 * (r_from + r_to)
        return -jnp.sum(self.a(mid) * (r_to - r_from), axis=-1)

    def bloch_sum_phase(self, P, c_nu) -> jnp.ndarray:
        """``-a(P) . c_nu``: the magnetic Bloch-sum factor of a ket at ``c_nu + P``."""
        return -jnp.sum(self.a(P) * jnp.asarray(c_nu, dtype=self.cell.at.dtype), axis=-1)

    def gipaw_phase(self, d, tau) -> jnp.ndarray:
        """``-a(tau) . d - a(d) . d / 2`` at the minimum-image ``d = r - tau`` (bohr)."""
        d = jnp.asarray(d, dtype=self.cell.at.dtype)
        return -jnp.sum(self.a(tau) * d, axis=-1) - 0.5 * jnp.sum(self.a(d) * d, axis=-1)

    def translation_phase(self, R1, R2) -> jnp.ndarray:
        """``a(R2) . R1``: ``T(R1) T(R2) = exp(i a(R2) . R1) T(R1 + R2)``."""
        return jnp.sum(self.a(R2) * jnp.asarray(R1, dtype=self.cell.at.dtype), axis=-1)


def minimum_image(d, lattice) -> np.ndarray:
    """``d`` folded to the nearest image over the lattice ``(3, 3)`` (rows), host NumPy.

    The fold is by rounding the crystal coordinates and then a search over the 27
    neighbouring images, which is exact for any cell that is not extremely skewed.
    """
    lattice = np.asarray(lattice, dtype=float)
    d = np.asarray(d, dtype=float)
    frac = d @ np.linalg.inv(lattice)
    frac = frac - np.round(frac)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], float)
    cand = (frac[..., None, :] + shifts) @ lattice
    best = np.argmin(np.sum(cand ** 2, axis=-1), axis=-1)
    return np.take_along_axis(cand, best[..., None, None], axis=-2)[..., 0, :]


def signed_area(points, lattice) -> float:
    """The signed area of a closed polygon in units of the chemical cell's area.

    ``points`` ``(n, 3)`` cartesian, projected on the plane of ``a1, a2``; counterclockwise
    seen from ``+a3`` is positive (the shoelace formula in the crystal coordinates of
    ``a1, a2``, whose cell has area one by construction).
    """
    lattice = np.asarray(lattice, dtype=float)
    frac = np.asarray(points, dtype=float) @ np.linalg.inv(lattice)
    x, y = frac[:, 0], frac[:, 1]
    orientation = np.sign(np.dot(np.cross(lattice[0], lattice[1]), lattice[2]))
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y) * orientation)


def refuse_regime(system, pseudos, what: str) -> None:
    """Refuse, by name, every regime no orbital-field route implements.

    Spinors (``npol = 2``), ``nspin = 2``, ultrasoft and PAW datasets, spin spirals,
    DFT+U, meta-GGA and gamma-only storage. The field along ``a3`` is not a check: the
    gauge puts it there by construction.
    """
    reasons = []
    if getattr(system, "npol", 1) != 1:
        reasons.append("a spinor (npol = 2) calculation")
    if getattr(system, "nspin", 1) != 1:
        reasons.append(f"nspin = {system.nspin}")
    if getattr(system, "spiral_q", None) is not None:
        reasons.append("a spin spiral")
    if getattr(system, "hubbard", None) is not None:
        reasons.append("DFT+U")
    if getattr(getattr(system, "kpoints", None), "gamma_only", False):
        reasons.append("gamma-only storage (K_POINTS gamma)")
    for pseudo in pseudos or ():
        if pseudo.is_paw:
            reasons.append(f"the PAW dataset {pseudo.element.strip()}")
        elif pseudo.is_ultrasoft:
            reasons.append(f"the ultrasoft dataset {pseudo.element.strip()}")
    if pseudos:
        from defuqumat.xc.functional import resolve_functional

        functional = resolve_functional([p.functional for p in pseudos],
                                        getattr(system, "input_dft", None))
        if functional.is_meta:
            reasons.append(f"the meta-GGA {functional.name}")
    if reasons:
        raise NotImplementedError(
            f"{what} in a uniform orbital field is implemented for a collinear, "
            "unpolarized, norm-conserving calculation with no Hubbard U, no spiral, a "
            f"GGA or LDA functional and the full G sphere; refused: {', '.join(reasons)}")
