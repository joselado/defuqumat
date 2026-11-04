"""``M_mn(k, b)``, ``A_mn(k)`` and the eigenvalues: what ``pw2wannier90.x`` computes.

The overlaps are ``M_mn(k, b) = <u_mk|u_n,k+b>``, with ``k + b = k' + G0`` for
``k'`` on the mesh and ``G0 = nncell``. They are
:meth:`~defuqumat.topology.states.PlaneWaveStates.overlaps` on the
neighbour table of :func:`~defuqumat.wannier.kmesh.kmesh_get`, so the Miller
alignment across the wrap and the ultrasoft and PAW term ``q_ij(b)`` are the
ones the Berry phases and the quantum metric already use (``compute_mmn``
builds the same augmentation from ``qb``). The pairs are grouped by their
geometric ``b``, which ``overlaps`` asserts is common to a call.

The projections are ``A_mn(k) = <psi_mk|S|g_n>`` with ``g_n`` the trial
orbitals of :mod:`defuqumat.wannier.trial` (``compute_amn``, which applies
``s_psi`` to the guiding functions for an ultrasoft or PAW dataset), so
``S g = g + sum_ij |beta_i> q_ij <beta_j|g>``.

Spinors: a projection with ``spin = +1`` or ``-1`` puts ``g`` in the upper or
lower component along ``z`` (``compute_amn``'s ``spin_z_pos`` branch); another
quantisation axis is refused by name, since ``pw2wannier90``'s
``fac(ipol)`` form of it was not checked here.

**Memory.** Besides the states, ``A`` builds ``(nproj, npol * npwx)`` per
k-point and ``M`` holds ``num_bands^2 nntot nk`` complex numbers; for silicon on
4x4x4 with 12 bands that is 0.6 MB, and nothing of size ``nk npwx`` is created
here beyond what the states already hold.
"""

from __future__ import annotations

import numpy as np

from defuqumat.topology.states import _augmentation_term, _cached_augmentation, _project
from defuqumat.wannier.trial import guiding_functions, radial_transform

__all__ = ["overlap_matrices", "projection_matrices"]


def overlap_matrices(states, kmesh, num_bands: int, k_batch=None) -> np.ndarray:
    """``(num_bands, num_bands, nntot, nk)`` in Wannier90's layout."""
    nk = states.nk
    nntot = kmesh.nntot
    mmn = np.zeros((num_bands, num_bands, nntot, nk), dtype=complex)
    groups: dict = {}
    for k in range(nk):
        for nn in range(nntot):
            key = tuple(np.round(kmesh.bk[:, nn, k], 8))
            groups.setdefault(key, []).append((k, nn))
    for members in groups.values():
        pairs = [(k, int(kmesh.nnlist[k, nn]) - 1, tuple(int(g) for g in kmesh.nncell[:, k, nn]))
                 for k, nn in members]
        blocks = np.asarray(states.overlaps(pairs, k_batch=k_batch))
        for (k, nn), block in zip(members, blocks):
            mmn[:, :, nn, k] = block[:num_bands, :num_bands]
    return mmn


def projection_matrices(states, projections, num_bands: int, volume: float) -> np.ndarray:
    """``(num_bands, nproj, nk)``: ``<psi_mk|S|g_n>``.

    ``states`` must carry ``vkb`` (``keep_projectors=True``) when it carries
    ``becp``, i.e. on an ultrasoft or PAW dataset.
    """
    npol = states.npol
    npwx = states.npwx
    nk = states.nk
    for p in projections:
        if npol == 1 and p.spin is not None:
            raise ValueError("a spin projection (u)/(d) needs spinor states")
        if npol == 2:
            if p.spin is None:
                raise ValueError("a spinor calculation needs every projection's spin, (u) or (d)")
            if p.s_qaxis is not None and not np.allclose(p.s_qaxis, (0.0, 0.0, 1.0)):
                raise NotImplementedError(
                    "a spin quantisation axis other than z is not implemented for the "
                    "projections; pw2wannier90's spinor rotation was not checked here")
    factors = None
    if states.becp is not None:
        if states.vkb is None:
            raise ValueError("ultrasoft or PAW states need their projectors for S|g>")
        factors = _cached_augmentation(states._owner, np.zeros(3))
    amn = np.zeros((num_bands, len(projections), nk), dtype=complex)
    # The radial transforms depend on |k + G| alone, which repeats across the
    # mesh: one table over the distinct values (rounded to 1e-12 1/bohr) for
    # the whole mesh, rather than one per k-point.
    kgs = [states.kcart[k] + states.miller[k] @ states.bg for k in range(nk)]
    q_all = np.concatenate([np.linalg.norm(kg, axis=1) for kg in kgs])
    q_unique, inverse = np.unique(np.round(q_all, 12), return_inverse=True)
    keys = {(float(p.zona), int(p.radial)) for p in projections}
    tables = {key: radial_transform(q_unique, key[0], key[1], volume) for key in keys}
    offsets = np.cumsum([0] + [len(kg) for kg in kgs])
    for k in range(nk):
        kg = kgs[k]
        index = inverse[offsets[k]:offsets[k + 1]]
        radial = {key: table[:, index] for key, table in tables.items()}
        g = guiding_functions(projections, kg, states.at, volume, valid=states.valid[k],
                              radial=radial)
        if npol == 2:
            full = np.zeros((len(projections), 2 * npwx), dtype=complex)
            for n, p in enumerate(projections):
                component = 0 if p.spin == 1 else 1
                full[n, component * npwx:(component + 1) * npwx] = g[n]
            g = full
        psi = np.asarray(states.coefficients[k])[:num_bands]
        a = psi.conj() @ g.T
        if factors is not None:
            becp_g = _project(g.astype(psi.dtype), states.vkb[k], npol)
            a = a + np.asarray(_augmentation_term(states.becp[k][:num_bands], becp_g, factors))
        amn[:, :, k] = a
    return amn
