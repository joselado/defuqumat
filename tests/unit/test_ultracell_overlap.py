"""The ultracell's ``S`` summed over its atom copies is the unit cell's, Q by Q.

``ultracell_overlap`` (I115) builds ``S|Psi>`` with the ``N`` copies of every
atom summed explicitly, so that the cross-``Q`` augmentation term vanishing is
computed rather than assumed. The copy sum ``sum_R e^{i(Q'-Q).R}`` is
``N delta_QQ'``, so the operator has to equal the unit cell's ``s_psi`` applied
on each ``Q``-slice at its own folded point. These tests hold the two against
each other on random projectors and a random Hermitian ``q_ij``, which needs no
SCF, and check that a copy set that is not the ultracell's translations
(one copy fewer) breaks the equality -- the test would otherwise not be able
to tell a copy sum from a block diagonal written by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

from defuqumat.ultracell.augmentation import ultracell_overlap
from defuqumat.ultracell.grid import Ultracell


class _Projectors:
    """Only what ``ultracell_overlap`` reads: one ``vkb`` per folded point."""

    def __init__(self, vkb):
        self.vkb = vkb

    def at_k(self, ik):
        return self.vkb[ik]


def _case(shape, npol, rng, nk0=2, npwx=7, nkb=3, rows=5):
    ultracell = Ultracell.build(shape, (8, 8, 8))
    cells = ultracell.cells
    k0 = rng.uniform(0, 1, size=(nk0, 3)) / np.asarray(shape)
    k_crystal = (k0[:, None, :] + ultracell.q_crystal[None]).reshape(-1, 3)
    vkb = (rng.normal(size=(nk0 * cells, npwx, nkb))
           + 1j * rng.normal(size=(nk0 * cells, npwx, nkb)))
    qq = rng.normal(size=(nkb, nkb)) + 1j * rng.normal(size=(nkb, nkb))
    qq = qq + qq.conj().T
    qq_so = rng.normal(size=(2, 2, nkb, nkb)) + 1j * rng.normal(size=(2, 2, nkb, nkb))
    # Hermitian as a (2 nkb)^2 matrix: [a,b,i,j] = conj([b,a,j,i]).
    qq_so = qq_so + np.conj(np.transpose(qq_so, (1, 0, 3, 2)))
    psi = (rng.normal(size=(rows, npol * cells * npwx))
           + 1j * rng.normal(size=(rows, npol * cells * npwx)))
    return ultracell, k_crystal, vkb, qq, qq_so, psi


def _block_diagonal(psi, ik0, vkb, qq, qq_so, cells, npol):
    """The unit cell's ``s_psi`` on each ``Q``-slice, as ``Calculation._overlap``."""
    rows = psi.shape[0]
    npwx = psi.shape[1] // (npol * cells)
    blocks = psi.reshape(rows, npol, cells, npwx).copy()
    out = blocks.copy()
    for iq in range(cells):
        v = vkb[ik0 * cells + iq]
        becp = np.einsum("gk,rag->rak", v.conj(), blocks[:, :, iq])
        if npol == 1:
            weighted = becp @ qq.T
        else:
            weighted = np.einsum("abij,rbj->rai", qq_so, becp)
        out[:, :, iq] += np.einsum("gk,rak->rag", v, weighted)
    return out.reshape(psi.shape)


@pytest.mark.parametrize("shape", [(2, 1, 1), (2, 3, 1)])
@pytest.mark.parametrize("npol", [1, 2])
def test_the_copy_sum_is_the_unit_cell_overlap_q_by_q(shape, npol):
    rng = np.random.default_rng(115)
    ultracell, k_crystal, vkb, qq, qq_so, psi = _case(shape, npol, rng)
    cells = ultracell.cells
    for ik0 in range(2):
        ours = np.asarray(ultracell_overlap(
            psi, ik0, projectors=_Projectors(vkb), k_crystal=k_crystal,
            ultracell=ultracell, qq=qq, qq_so=qq_so, npol=npol))
        reference = _block_diagonal(psi, ik0, vkb, qq, qq_so, cells, npol)
        assert np.abs(ours - reference).max() < 1e-12 * np.abs(reference).max()


def test_a_wrong_copy_set_leaves_a_cross_q_term():
    """The guard on the test: drop one copy and the equality has to break."""
    rng = np.random.default_rng(7)
    shape = (3, 1, 1)
    ultracell, k_crystal, vkb, qq, qq_so, psi = _case(shape, 1, rng)
    reference = _block_diagonal(psi, 0, vkb, qq, qq_so, ultracell.cells, 1)
    # Two copies with three Q-slices: ``sum_R e^{i(Q'-Q).R}`` is no longer a
    # delta, so the operator is not the block diagonal. The ultracell object
    # is read only for its translations (``shape``) and its ``N``.

    class _Mixed:
        shape = (2, 1, 1)
        cells = ultracell.cells

    ours = np.asarray(ultracell_overlap(
        psi, 0, projectors=_Projectors(vkb), k_crystal=k_crystal,
        ultracell=_Mixed, qq=qq, npol=1))
    assert np.abs(ours - reference).max() > 1e-2 * np.abs(reference).max()


def test_a_norm_conserving_dataset_is_the_identity():
    rng = np.random.default_rng(3)
    ultracell, k_crystal, vkb, _, _, psi = _case((2, 1, 1), 1, rng)
    ours = ultracell_overlap(psi, 1, projectors=_Projectors(vkb),
                             k_crystal=k_crystal, ultracell=ultracell, npol=1)
    assert np.array_equal(np.asarray(ours), psi)


def test_the_pad_is_zeroed():
    rng = np.random.default_rng(5)
    ultracell, k_crystal, vkb, qq, _, psi = _case((2, 1, 1), 1, rng)
    padding = np.ones((2, 7), dtype=bool)
    padding[:, -2:] = False
    ours = np.asarray(ultracell_overlap(
        psi, 0, projectors=_Projectors(vkb), k_crystal=k_crystal,
        ultracell=ultracell, qq=qq, npol=1, padding=padding)).reshape(5, 2, 7)
    assert np.all(ours[:, :, -2:] == 0.0)
    assert np.all(ours[:, :, :-2] != 0.0)


def test_the_volume_exit_refuses_spin_orbit_on_an_augmented_dataset():
    """I115 kept by name: the spin-orbit blocks of ``S`` are not measured.

    The guard sits before any projector is built, so stand-ins are enough;
    the same call without ``lspinorb`` must get past it (and fail later, on
    the missing states), which is what shows the guard is the one firing.
    """
    from types import SimpleNamespace

    from defuqumat.workflows.ultracell import _ultracell_overlap_operator

    pseudos = [SimpleNamespace(is_ultrasoft=True, is_paw=False)]
    with pytest.raises(NotImplementedError, match="spin-orbit coupling"):
        _ultracell_overlap_operator(SimpleNamespace(lspinorb=True), pseudos,
                                    None, "volume")
    with pytest.raises(AttributeError):
        _ultracell_overlap_operator(SimpleNamespace(lspinorb=False), pseudos,
                                    None, "volume")
    assert _ultracell_overlap_operator(
        SimpleNamespace(lspinorb=True), pseudos, None, "plane") is None
    assert _ultracell_overlap_operator(
        SimpleNamespace(lspinorb=True),
        [SimpleNamespace(is_ultrasoft=False, is_paw=False)], None,
        "volume") is None
