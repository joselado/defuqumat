"""DFT+U on the ultracell: one occupation matrix per atom copy.

``PLAN.md`` P88. An ultracell holds ``N`` copies of every Hubbard atom, at
``tau + R_c`` with ``R_c`` the ``c``-th lattice vector of the ultracell's own
cell grid, and a modulation can occupy their manifolds differently, which is
the whole reason the refusal stood: the unit cell's ``ns`` is one matrix per
atom and the ultracell needs ``N``.

**The copies' orbitals are already in the frozen basis.** The supercell Bloch
sum of copy ``c``'s orbital at ``k0`` is a combination of the *unit cell's*
Bloch sums at the folded points,

    Phi^c_{k0} = N^{-1/2} sum_Q exp(-i (k0+Q) . R_c) phi_{k0+Q},

because summing ``exp(i Q . T)`` over the ``N`` folded ``Q`` keeps exactly the
unit-cell translations ``T`` that differ from ``R_c`` by an ultracell lattice
vector. So the projection of a frozen state onto copy ``c`` is the unit cell's
own projection ``<wfcU_{k0+Q}|psi_{k0+Q,n}>`` times a phase, and nothing is
built on the ultracell box at all: the Hubbard term is ``nwfcU`` columns per
basis function, exactly as it is in the unit cell. A Lowdin-orthogonalised
manifold carries over too, since the supercell's atomic overlap at ``k0`` is
block diagonal in ``Q`` in the same decomposition.

**Normalisation.** ``P^c`` below is ``N^{1/2} <Phi^c|Psi>``, so that at a
tiled state it is the unit cell's projection with a phase, and the occupation
matrix of every copy is then the unit cell's ``ns`` with the ultracell
occupations ``w_k0 f_j / N`` the loop already uses, the factor of ``N`` in
those occupations and the ``N^{1/2}`` here being the same number. The matrix
element carries the ``1/N`` back: at a tiled potential the phases sum to
``N delta_{QQ'}`` and the term is the unit cell's ``vhpsi`` at ``k0 + Q``.

What it costs: ``(N, N nbnd, nwfcU)`` complex per ``k0`` and channel, the
projections times the copies, which is ``N`` times the unit cell's ``proj`` for
the same states, and an ``(N nbnd)^2`` matrix product per copy per iteration.

Refused, each by name in the driver: an ultrasoft or PAW dataset (the
projection carries ``S``), a spinor (four spin blocks per copy), a
``TENSOR_MOMENTS`` constraint on ``ns``, and any projector but ``'atomic'``.
A Lowdin-orthogonalised manifold decomposes over ``Q`` the same way, since the
supercell's atomic overlap at ``k0`` is block diagonal in ``Q``, and it is not
measured. The intersite ``V`` is refused at the input for every run.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from defuqumat.hubbard.energy import hubbard_energy, hubbard_potential
from defuqumat.hubbard.operator import block_potential

__all__ = ["copy_projections", "copy_occupations", "copy_hubbard_matrix",
           "UltracellHubbardTerms", "ultracell_hubbard_terms"]


def copy_projections(wfcU, coefficients, kq_crystal, translations):
    """``X[s, k0, c, (Q, n), i]``: each frozen state's projection onto each copy.

    Args:
        wfcU: ``(nk0, N, npwx, nwfcU)`` the unit cell's Hubbard projectors on
            the folded k-set, ``ik = ik0 N + iq``.
        coefficients: ``(blocks, nk0, N, nbnd, npwx)`` the frozen states.
        kq_crystal: ``(nk0, N, 3)`` ``k0 + Q`` in the unit cell's crystal basis.
        translations: ``(N, 3)`` integer ``R_c`` in the unit cell's crystal
            basis, the ultracell's ``q_triples``.

    ``X = exp(2 pi i (k0 + Q) . R_c) <wfcU_{k0+Q}|psi_{k0+Q,n}>``.
    """
    proj = jnp.einsum("kqgi,skqng->skqni", jnp.conj(wfcU), coefficients)
    phase = np.exp(2j * np.pi * np.einsum(
        "kqd,cd->kcq", np.asarray(kq_crystal, dtype=float),
        np.asarray(translations, dtype=float)))
    blocks, nk0, cells, nbnd, nwfcU = proj.shape
    copies = jnp.asarray(phase, dtype=proj.dtype)[None, :, :, :, None, None] \
        * proj[:, :, None, :, :, :]
    return copies.reshape(blocks, nk0, cells, cells * nbnd, nwfcU)


def copy_occupations(projections, vectors, occupations, columns, mask, nspin):
    """``(N, nspin, nslot, ldmx, ldmx)``: ``new_ns`` for every copy.

    ``vectors[s][k0]`` is ``(N nbnd, nstate)`` and ``occupations`` is
    ``(blocks, nk0, nstate)``, the loop's own. The column map and the mask are
    the unit cell's (``Calculation._hubbard_columns``, ``_hubbard_mask``), and
    the factor of one half at ``nspin = 1`` is ``new_ns``'s, as in
    :func:`~defuqumat.hubbard.occupations.occupation_matrix`.
    """
    blocks, nk0, cells = projections.shape[:3]
    per_channel = []
    for spin in range(blocks):
        total = 0.0
        for ik in range(nk0):
            amplitudes = jnp.asarray(vectors[spin][ik])  # (N nbnd, nstate)
            state = jnp.einsum("aj,cai->cji", amplitudes, projections[spin, ik])
            block = state[..., columns[:, 0]]  # (N, nstate, nslot, ldmx)
            block = jnp.where(mask, block, 0.0)
            weight = jnp.asarray(occupations[spin][ik])
            total = total + jnp.einsum(
                "j,cjna,cjnb->cnab", weight, jnp.conj(block), block).real
        per_channel.append(total)
    ns = jnp.stack(per_channel, axis=1)
    return 0.5 * ns if int(nspin) == 1 else ns


def copy_hubbard_matrix(projections, delta_blocks):
    """``(1/N) sum_c conj(X_c) dV_c X_c^T`` for one ``k0`` and channel.

    Args:
        projections: ``(N, N nbnd, nwfcU)``, ``X`` at one ``(s, k0)``.
        delta_blocks: ``(N, nwfcU, nwfcU)`` each copy's block potential minus
            the reference's, which the frozen eigenvalues already carry.
    """
    cells = projections.shape[0]
    blocks = jnp.asarray(delta_blocks).astype(projections.dtype)
    return jnp.einsum("cai,cij,cbj->ab", jnp.conj(projections), blocks,
                      projections) / cells


class UltracellHubbardTerms:
    """One iteration's Hubbard energy, potential and matrix blocks per copy."""

    __slots__ = ("energy", "v_ns", "delta_blocks")

    def __init__(self, energy, v_ns, delta_blocks):
        self.energy = energy
        self.v_ns = v_ns
        self.delta_blocks = delta_blocks


def ultracell_hubbard_terms(ns_copies, coefficients, block_indices,
                            reference_blocks) -> UltracellHubbardTerms:
    """``eth`` per cell, ``v_ns`` per copy, and ``dV`` blocks per channel and copy.

    ``reference_blocks`` is ``(nspin, nwfcU, nwfcU)``, the block potential of
    the converged unit-cell ``ns``, subtracted because every frozen eigenvalue
    already carries it -- the same subtraction ``dV`` makes of ``v_scf``.
    """
    cells = int(ns_copies.shape[0])
    energy = 0.0
    potentials, deltas = [], []
    for c in range(cells):
        energy = energy + hubbard_energy(ns_copies[c], coefficients)
        v_ns = hubbard_potential(ns_copies[c], coefficients)
        potentials.append(v_ns)
        deltas.append(block_potential(v_ns, block_indices) - reference_blocks)
    return UltracellHubbardTerms(
        energy=float(energy) / cells,
        v_ns=jnp.stack(potentials),
        delta_blocks=jnp.stack(deltas, axis=1),  # (nspin, N, nwfcU, nwfcU)
    )
