"""The Hubbard term at ``k + kappa(t)``, at a frozen occupation matrix.

``V_U = sum_I |wfcU_I> v_IJ <wfcU_J|`` (``vhpsi.f90``) is separable in the
way the nonlocal pseudopotential is, and its projectors are pseudo-atomic
orbitals at ``k + G`` (:mod:`defuqumat.hubbard.projectors`). So under the
velocity gauge they move with ``k + kappa`` exactly as the beta functions do:
the gauge factor ``exp(-i kappa.R)`` of an orbital centred on an atom cancels
between the bra and the ket of one site, the rest of ``exp(-i kappa.r)`` is
carried by evaluating the orbital at ``k + G + kappa``, and the current, which
is ``jax.grad`` in ``kappa`` of the band energy, then holds the Hubbard term's
share of ``[r, H]`` with nothing derived by hand. That share is what
:class:`~defuqumat.response.velocity.VelocityOperator` already differentiates
when it is handed ``ns``, through ``Calculation.at_kcart``.

**The occupation matrix is frozen**, at the one the caller hands in, the
ground state's: ``v_IJ`` is built from it once, as ``v_scf`` is frozen, which is
the independent-particle response. An ``ns(t)`` measured from the propagated
states would be the Hubbard analogue of the updated potentials and is refused
with them.

**The orbitals come from a table**, :func:`~defuqumat.realtime.radial.atomic_table`,
for the reasons the projectors do (``realtime/radial.py``): the transform costs
several Hamiltonian applications a rebuild, a pseudopotential cannot ride on a
pytree argument of the kept block, and the solid-harmonic form is
differentiable to every order at ``k + G + kappa = 0``, where the transform's
guarded harmonic loses the ``l = 1`` tangent.

**Which projector sets.** ``atomic`` (``wfcU = phi`` on a norm-conserving
dataset, where ``S`` is the identity) and ``norm-atomic`` (each column divided
by its own norm, an elementwise ``rsqrt`` of a smooth positive function).
``ortho-atomic`` is refused by the driver: the Löwdin ``O^{-1/2}`` goes through
an eigendecomposition whose second derivative, which the perturbative orders
take, carries ``1/(w_i - w_j)`` at the degeneracies symmetry puts in ``O``
(:func:`~defuqumat.hubbard.projectors._inverse_sqrt` repairs the first only).

**Memory.** Per k-chunk, the orbitals of the whole crystal at ``k + kappa``,
``16 nk_chunk npwx natomwfc`` bytes, of which ``nwfcU`` columns are kept, twice
a step (the midpoint and the current); ``natomwfc`` is a few per cent of the
bands a chunk carries.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.hubbard.operator import HubbardTerm
from defuqumat.pseudo.projectors import _apply_phases
from defuqumat.realtime.radial import atomic_layout, atomic_table

__all__ = ["HubbardMotion", "MOVING_PROJECTORS"]

#: The projector sets whose motion with ``kappa`` is implemented.
MOVING_PROJECTORS = ("atomic", "norm-atomic")


def _i_power(l_of, dtype):
    """``i^l`` for every column, built from real parts so the dtype is the policy's."""
    l_of = np.asarray(l_of) % 4
    real = np.select([l_of == 0, l_of == 2], [1.0, -1.0], 0.0)
    imag = np.select([l_of == 1, l_of == 3], [1.0, -1.0], 0.0)
    real_dtype = jnp.zeros((), dtype=dtype).real.dtype
    return jax.lax.complex(jnp.asarray(real, dtype=real_dtype), jnp.asarray(imag, dtype=real_dtype))


class HubbardMotion(eqx.Module):
    """What a chunk needs to build ``V_U(k + kappa)``: the table, the gathers, ``v`` per channel.

    Built once per run (:meth:`build`) and carried on every
    :class:`~defuqumat.realtime.propagate._Chunk` as an argument of the kept
    block, never closed over.
    """

    table: object
    phase: jnp.ndarray          # (ncols,) complex, i^l
    atom_of: jnp.ndarray        # (natomwfc,)
    column_of: jnp.ndarray      # (natomwfc,)
    columns: jnp.ndarray        # (nwfcU,), the Hubbard manifold among the natomwfc
    vns: jnp.ndarray            # (nchannel, nwfcU, nwfcU), real, frozen
    kind: str = eqx.field(static=True)

    @classmethod
    def build(cls, calculation, ns, s_max: float) -> "HubbardMotion":
        setup = calculation.hubbard
        if setup.projectors not in MOVING_PROJECTORS:
            raise NotImplementedError(
                f"real-time propagation with Hubbard_projectors = {setup.projectors!r} "
                "is not implemented: the Löwdin O^(-1/2) of 'ortho-atomic' goes through an "
                "eigendecomposition whose second derivative, which the perturbative "
                "orders and a kick's first order take, carries 1/(w_i - w_j) at the "
                "degeneracies symmetry puts in O; 'atomic' and 'norm-atomic' move")
        cell = calculation.system.cell
        types = calculation.system.structure.types
        table = atomic_table(calculation.pseudos, types, float(cell.volume), s_max)
        table = eqx.tree_at(lambda t: (t.coefficients, t.s_max), table,
                            (table.coefficients.astype(cell.precision.real),
                             table.s_max.astype(cell.precision.real)))
        _, _, _, l_of, atom_of, column_of = atomic_layout(calculation.pseudos, types)
        width = setup.npol
        columns = (np.concatenate([
            np.arange(offset, offset + width * ldim)
            for offset, ldim in zip(setup.atomwfc_offsets, setup.ldims)])
            if setup.nwfcU else np.zeros(0, dtype=int))
        _, _, terms = calculation.hubbard_terms(jnp.asarray(ns))
        vns = jnp.stack([term.vns for term in terms])
        return cls(table=table, phase=_i_power(l_of, cell.precision.complex),
                   atom_of=jnp.asarray(atom_of), column_of=jnp.asarray(column_of),
                   columns=jnp.asarray(columns), vns=vns, kind=str(setup.projectors))

    def orbitals(self, kg, positions, mask):
        """``(nk, npwx, natomwfc)``: every pseudo-atomic orbital at ``kg = k + G + kappa``."""
        columns = self.table.columns(kg).astype(self.phase.dtype) * self.phase
        return _apply_phases(columns, kg, positions, mask, self.atom_of, self.column_of)

    def projectors(self, kg, positions, mask):
        """``wfcU`` at ``kg``, ``(nk, npwx, nwfcU)``, on a norm-conserving dataset."""
        phi = self.orbitals(kg, positions, mask)
        if self.kind == "norm-atomic":
            norms = jnp.sum(jnp.real(jnp.conj(phi) * phi), axis=-2, keepdims=True)
            phi = phi * jax.lax.rsqrt(norms).astype(phi.dtype)
        return jnp.take(phi, self.columns, axis=-1)

    def term(self, kg, positions, mask, channel: int) -> HubbardTerm:
        """The channel's :class:`~defuqumat.hubbard.operator.HubbardTerm` at ``kg``."""
        return HubbardTerm(wfcU=self.projectors(kg, positions, mask), vns=self.vns[channel])
