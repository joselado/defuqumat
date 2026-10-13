"""A homogeneous electric field across a slab, and the dipole correction.

``pw.x``'s ``tefield`` and ``dipfield`` (``PW/src/add_efield.f90``,
``compute_dip.f90``, ``extfield.f90``). A uniform field cannot be put into a
periodic cell as a linear potential, so it is made periodic as a **saw-tooth**:
the potential rises linearly over a fraction ``1 - eopreg`` of the cell along
the reciprocal vector ``b_edir`` and falls back over the remaining ``eopreg``,
a region that has to sit in the vacuum. ``dipfield`` adds the **dipole
correction** (Bengtsson, PRB 59, 12301 (1999); Meyer and Vanderbilt, PRB 63,
205426 (2001)): the field that the slab's own dipole sets up across the
vacuum through the periodic images is cancelled by a saw-tooth of the opposite
slope, whose amplitude is the dipole of the current density.

**The energy is the primitive here and the potential comes from `jax.grad`**,
as in :mod:`defuqumat.scf.fields`. QE has two code paths for this -- without
``dipfield`` the saw-tooth goes once into ``vltot`` and only the ions' part of
the energy is printed as ``etotefield``, with ``dipfield`` it goes into
``v_scf`` at every iteration and ``etotefield`` is the whole term -- and both
are the gradient of one function,

    E_f(rho, tau) = (Omega/4 pi) e2 [ -eamp D + [dipfield] D^2/2 ],
    D = ion_dip(tau) - el_dip(rho),
    el_dip = (4 pi/Omega) int rho(r) saw(x(r)) d dr,
    ion_dip = (4 pi/Omega) sum_a Z_a saw(x_a) d,

with ``x`` the crystal coordinate along ``edir``, ``d = 2 pi/|b_edir|`` the
spacing of the lattice planes normal to the field (QE's ``alat/bmod``) and the
electron density counted positive. Its derivative with respect to ``rho`` is
``e2 (eamp - D) saw(x) d``, which is ``add_efield``'s potential in both
branches, and its derivative with respect to ``tau`` is ``e2 (eamp - D) Z
b_edir/|b_edir|`` where the saw-tooth is rising, which is ``forcefield``. So
the force comes from :mod:`defuqumat.forces.energy` with nothing written by
hand, including the part an ultrasoft augmentation charge picks up as it moves
through the saw-tooth (``addusforce`` integrates ``vltot + v_scf``, and the
saw-tooth is in one or the other). Unlike :mod:`defuqumat.scf.fields`, this
energy **is** in the total, as QE's ``etotefield`` is.

``eamp`` is in Hartree atomic units, as ``INPUT_PW`` documents it, and ``e2 =
2`` turns it into the Rydberg potential the rest of the code carries. What
QE's output calls the dipole in "Ry au" is ``D Omega/(4 pi)``, which is what
:attr:`dipole` reports.

**What is not here.** The stress, which QE switches off with this field
(``input.f90``); ``gate``, the charged plate, which needs a charged cell; and
QE's mixing of ``el_dipole`` as an extra degree of freedom of the density
(``scf_mod.f90``), which changes the metric of the mixer and not its fixed
point, since ``el_dip`` is linear in ``rho``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.scf.potential import as_potential_components, total_charge

__all__ = ["SawtoothField", "saw", "build_sawtooth"]

E2 = 2.0
FPI = 4.0 * np.pi


def saw(emaxpos: float, eopreg: float, x):
    """``extfield.f90``'s ``saw``: the periodic saw-tooth of a crystal coordinate.

    Rises with slope one (in units of the cell length along ``edir``) except in
    ``[emaxpos, emaxpos + eopreg]``, where it falls back with slope
    ``-(1 - eopreg)/eopreg``. A ``jnp.where`` rather than a clamp, so that an
    ion sitting exactly on a kink gets one of the two slopes and not their mean.
    """
    z = x - emaxpos
    y = z - jnp.floor(z)
    rising = (-0.5 + (y - eopreg) / (1.0 - eopreg)) * (1.0 - eopreg)
    falling = (0.5 - y / eopreg) * (1.0 - eopreg)
    return jnp.where(y <= eopreg, falling, rising)


class SawtoothField(eqx.Module):
    """``tefield`` with its four parameters, and whether ``dipfield`` is on.

    ``edir`` is **0-based** here, where the input variable is 1-based.
    ``eamp`` is an array leaf so that the energy can be differentiated with
    respect to it (``dE/d(eamp)`` is the dipole, which is how it is tested).
    """

    eamp: jnp.ndarray
    edir: int = eqx.field(static=True)
    emaxpos: float = eqx.field(static=True)
    eopreg: float = eqx.field(static=True)
    dipfield: bool = eqx.field(static=True)

    def plane_spacing(self, cell):
        """``alat/bmod``: the distance between lattice planes normal to the field."""
        return 2.0 * np.pi / jnp.linalg.norm(cell.bg[self.edir])

    def grid_saw(self, shape):
        """``saw`` on the dense grid, ``x = i/n`` along ``edir``, broadcastable to ``shape``."""
        n = shape[self.edir]
        x = jnp.arange(n, dtype=self.eamp.dtype) / n
        values = saw(self.emaxpos, self.eopreg, x)
        reshape = [1, 1, 1]
        reshape[self.edir] = n
        return values.reshape(reshape)

    def dipoles(self, rho_total, positions, charges, cell):
        """``(D, el_dip, ion_dip)``, each in QE's units of ``4 pi/Omega`` times a dipole."""
        spacing = self.plane_spacing(cell)
        el_dip = FPI * jnp.mean(rho_total * self.grid_saw(rho_total.shape)) * spacing
        x = cell.to_crystal(positions)[:, self.edir]
        ion_dip = FPI / cell.volume * jnp.sum(
            jnp.asarray(charges, dtype=x.dtype) * saw(self.emaxpos, self.eopreg, x)
        ) * spacing
        return ion_dip - el_dip, el_dip, ion_dip

    def energy(self, rho_total, positions, charges, cell):
        """``E_f``: the field's energy with the electrons and the ions, in Ry."""
        dip, _, _ = self.dipoles(rho_total, positions, charges, cell)
        energy = -self.eamp * dip
        if self.dipfield:
            energy = energy + 0.5 * dip ** 2
        return cell.volume / FPI * E2 * energy

    def potential(self, rho_r, positions, charges, cell):
        """``(v, E_f, D)``: the potential laid out like ``rho_r``, its energy, the dipole.

        ``v`` is the functional derivative: the gradient of ``E_f`` with
        respect to the grid values of the total charge, divided by the volume
        element ``Omega/N``. It is spin-independent, so it goes to every
        channel of an ``(up, down)`` pair and to the first component only of
        an ``(n, m)`` stack (:func:`as_potential_components`).
        """
        rho_total = total_charge(rho_r)
        energy, gradient = jax.value_and_grad(self.energy)(
            rho_total, positions, charges, cell
        )
        v = gradient * (rho_total.size / cell.volume)
        dip, _, _ = self.dipoles(rho_total, positions, charges, cell)
        return as_potential_components(v, rho_r.shape[0]), energy, dip

    def dipole(self, dip, cell):
        """QE's printed "Dipole" in Ry atomic units: ``D Omega/(4 pi)``."""
        return dip * cell.volume / FPI


@jax.jit
def sawtooth_potential(field, rho_r, positions, charges, cell):
    """``add_efield`` for one density: ``(v, E_f, D)``."""
    return field.potential(rho_r, positions, charges, cell)


def build_sawtooth(settings, precision) -> SawtoothField | None:
    """The field a :class:`~defuqumat.system.builder.System` asks for, or ``None``.

    ``settings`` is :attr:`System.efield`: ``(edir, eamp, emaxpos, eopreg,
    dipfield)`` with ``edir`` 0-based.
    """
    if settings is None:
        return None
    edir, eamp, emaxpos, eopreg, dipfield = settings
    return SawtoothField(
        eamp=jnp.asarray(eamp, dtype=precision.real),
        edir=int(edir), emaxpos=float(emaxpos), eopreg=float(eopreg),
        dipfield=bool(dipfield),
    )


def refuse_sawtooth(owner, quantity: str, missing: str) -> None:
    """Stop when ``owner`` (a ``Calculation`` or a ``System``) carries ``tefield``.

    For the consumers that have not been measured with the field: each states
    ``missing``, what it would take, so that the refusal names a term rather
    than a feature.
    """
    asked = (getattr(owner, "sawtooth", None) is not None
             or getattr(owner, "efield", None) is not None)
    if asked:
        raise NotImplementedError(
            f"{quantity} with a saw-tooth electric field (tefield) is not "
            f"implemented: {missing}"
        )
