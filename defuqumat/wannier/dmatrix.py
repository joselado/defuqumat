"""The two representations site symmetry needs: ``D`` on the trial orbitals, ``d`` on the bands.

This is ``compute_dmn`` of ``pw2wannier90.f90`` (QE 7.5, ``PP/src``, lines
1986-2680), the ``.dmn`` file of Wannier90's ``site_symmetry`` mode, built from
the operations of the run's own :class:`~defuqumat.system.symmetry.Symmetries`
and from the states already solved on the whole mesh.

**The operation.** ``g`` is ``r -> R r + t`` with ``R`` the cartesian rotation
(:func:`~defuqumat.system.symmetry.cartesian_rotations`) and ``t`` the
fractional translation in bohr, the convention under which
:func:`~defuqumat.system.symmetry.atom_mapping` sends atom ``a`` to ``irt[a]``.
A function transforms as ``(g f)(r) = f(g^-1 r)``, so a Bloch state at ``k``
becomes one at ``R k``, which is the mesh point ``k' = R k - G0``
(``kptsym``, ``compute_dmn``'s ``iks2k``).

**The trial orbitals**, ``D(g, k)``. ``g t_n = sum_j t_j W_jn`` with ``t_j`` on
the image ``g(tau_n) = tau_j + T_n`` of the site of ``t_n``, and

    W_jn = int dOmega Y_j(R d) Y_n(d)

over the unit sphere (``compute_dmn``'s ``wws``, lines 2350-2364), each ``Y``
in its own local frame. The Bloch sum ``sum_L e^{ik.L} t(r - tau - L)`` then
picks up ``e^{-i (R k).T_n}``. ``compute_dmn`` also carries ``e^{-i t.G0}``
(its ``phs``, lines 2407-2410), a factor common to ``D`` and ``d`` at each
``(g, k)``, which ``U(gk) = d U(k) D^dagger`` does not see; it is kept, so
that the file written here is ``pw2wannier90.x``'s. A projection set that is
not a representation, a function whose image is not a combination of the
functions on the image site, is refused by ``compute_dmn``'s own test
(``err > 1e-3``, lines 2366-2376), and so is a site the operation does not map
onto a site of the set.

**The bands**, ``d_mn(g, k) = <psi_m,k'|S|g psi_nk>``. ``compute_dmn`` rotates
on the FFT grid; here the rotation is done on the sphere, where it is a
permutation with a phase: the plane wave ``k + G`` goes to ``R (k + G) =
k' + G'`` and takes ``e^{-i R(k+G).t}``. ``|k' + G'| = |k + G|``, so the map
is a bijection of the two spheres; it is asserted, and refused by name
otherwise. ``S`` commutes with ``g``, so for an ultrasoft or PAW dataset the
augmentation is ``sum q_ij(0) <psi_m,k'|beta_i><beta_j|g psi_nk>`` with the
projectors at ``k'`` acting on the rotated coefficients (``compute_dmn``'s
``calbec`` on ``evc_sk`` and ``qb`` at ``dxk = 0``, lines 2449-2475 and
2576-2637). The file stores ``CONJG(Mkb(n, m))`` (line 2645), which is ``d_mn``
in this notation.

Only the irreducible points of the mesh are needed, ``nkptirr x nsym``
matrices of ``num_bands^2``. The rotation is JAX on the states (it is a
gather and two contractions per ``(g, k)``, and runs where the states live);
the small dense ``D`` is host NumPy, as the engine is.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from defuqumat.wannier.sitesym import SymmetryData

__all__ = ["Operations", "operations_of", "kpoint_stars", "wannier_representation",
           "band_representation", "site_symmetry_data"]

#: ``compute_dmn``'s refusal of a function the operation does not keep in the set.
WWS_TOLERANCE = 1.0e-3
#: A Miller index computed through a rotation is an integer to this.
_MILLER_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class Operations:
    """``g: r -> R r + t``, cartesian, ``R`` ``(nsym, 3, 3)``, ``t`` ``(nsym, 3)`` in bohr."""

    rotations: np.ndarray
    translations: np.ndarray

    @property
    def nsym(self) -> int:
        return len(self.rotations)


def operations_of(cell, symmetries) -> Operations:
    """The operations of a :class:`~defuqumat.system.symmetry.Symmetries`, cartesian."""
    from defuqumat.system.symmetry import cartesian_rotations

    at = np.asarray(cell.at, dtype=float)
    rotations = cartesian_rotations(cell, symmetries)
    translations = symmetries.translation_array() @ at
    # The engine and the .dmn take the identity first (compute_dmn refuses
    # otherwise, line 2181); the run's list need not start with it.
    frac = symmetries.translation_array()
    identity = [s for s in range(len(rotations)) if np.allclose(rotations[s], np.eye(3))
                and np.allclose(frac[s] - np.rint(frac[s]), 0.0)]
    if len(identity) != 1:
        raise ValueError("the group of the run has no identity, or two")
    order = identity + [s for s in range(len(rotations)) if s != identity[0]]
    return Operations(rotations=rotations[order], translations=translations[order])


def _find(frac, points, what):
    d = np.asarray(frac)[None, :] - points
    d -= np.rint(d)
    hit = np.flatnonzero(np.abs(d).max(axis=1) < 1.0e-6)
    if len(hit) != 1:
        raise ValueError(f"{what} is not a point of the mesh: the mesh is not invariant "
                         "under the group of the run")
    return int(hit[0])


def kpoint_stars(points: np.ndarray, operations: Operations, at: np.ndarray, bg: np.ndarray):
    """``compute_dmn``'s ``iks2k``, ``ik2ir`` and ``ir2ik`` (lines 2186-2255), 0-based.

    ``points`` are the mesh in crystal coordinates; ``bg`` the reciprocal
    vectors as rows (with the ``2 pi``), ``at`` the direct ones. Returns
    ``(ik2ir, ir2ik, kptsym, g0)`` with ``kptsym[s, ir]`` the index of
    ``R k`` and ``g0[s, ir]`` the cartesian ``R k - k'``.
    """
    nk = len(points)
    kcart = points @ bg
    iks2k = np.empty((nk, operations.nsym), dtype=np.int64)
    for s in range(operations.nsym):
        images = (kcart @ operations.rotations[s].T) @ np.linalg.inv(bg)
        for ik in range(nk):
            iks2k[ik, s] = _find(images[ik], points, f"R k for k = {points[ik]}")
        if len(set(iks2k[:, s])) != nk:
            raise ValueError("an operation of the group does not permute the mesh")
    ik2ir = -np.ones(nk, dtype=np.int64)
    ir2ik = []
    for ik in range(nk):
        if ik2ir[ik] >= 0:
            continue
        ir2ik.append(ik)
        ik2ir[ik] = len(ir2ik) - 1
        for s in range(operations.nsym):
            if ik2ir[iks2k[ik, s]] < 0:
                ik2ir[iks2k[ik, s]] = len(ir2ik) - 1
    ir2ik = np.array(ir2ik, dtype=np.int64)
    kptsym = iks2k[ir2ik].T.copy()
    g0 = np.empty((operations.nsym, len(ir2ik), 3))
    for ir, ik in enumerate(ir2ik):
        for s in range(operations.nsym):
            g0[s, ir] = operations.rotations[s] @ kcart[ik] - kcart[kptsym[s, ir]]
    return ik2ir, ir2ik, kptsym, g0


def _sphere_quadrature():
    """Gauss-Legendre in ``cos theta`` times a uniform ``phi``: exact for the
    products of two functions up to ``l = 3`` (degree 6) with room to spare."""
    x, w = np.polynomial.legendre.leggauss(12)
    phi = 2.0 * np.pi * np.arange(24) / 24
    st = np.sqrt(1.0 - x * x)
    d = np.stack([np.outer(st, np.cos(phi)), np.outer(st, np.sin(phi)),
                  np.outer(x, np.ones_like(phi))], axis=-1).reshape(-1, 3)
    weights = np.outer(w, np.full(len(phi), 2.0 * np.pi / len(phi))).ravel()
    return d, weights


def _sites(projections):
    """Distinct projection centres (``compute_dmn``'s ``iw2ip``), crystal coordinates."""
    sites, of = [], []
    for p in projections:
        f = np.asarray(p.site_frac, dtype=float)
        for i, s in enumerate(sites):
            if np.allclose(s, f, atol=1.0e-6):
                of.append(i)
                break
        else:
            sites.append(f)
            of.append(len(sites) - 1)
    return np.array(sites), np.array(of)


def wannier_rotations(projections, operations: Operations, at: np.ndarray):
    """``(W, T)``: ``W[s]`` the ``(nw, nw)`` real matrix ``wws`` and ``T[s]`` the
    ``(nw, 3)`` cartesian lattice vector ``g(tau_n) - tau_j`` of each column.

    Refuses a set the group does not map onto itself.
    """
    from defuqumat.wannier.trial import angular_parts

    nw = len(projections)
    sites, of = _sites(projections)
    d, weights = _sphere_quadrature()
    frames = [(p.l, p.m, p.z_axis, p.x_axis) for p in projections]
    y0 = np.array([angular_parts(l, m, d, z, x).sum(axis=0) for (l, m, z, x) in frames])
    W = np.zeros((operations.nsym, nw, nw))
    T = np.zeros((operations.nsym, nw, 3))
    for s in range(operations.nsym):
        R, t = operations.rotations[s], operations.translations[s]
        yr = np.array([angular_parts(l, m, d @ R.T, z, x).sum(axis=0) for (l, m, z, x) in frames])
        image_site = np.empty(len(sites), dtype=np.int64)
        for i, site in enumerate(sites):
            image = (R @ (site @ at) + t) @ np.linalg.inv(at)
            image_site[i] = _find_site(image, sites, s, site)
            T_site = (image - sites[image_site[i]]) @ at
            for n in np.flatnonzero(of == i):
                T[s, n] = T_site
        for n in range(nw):
            rows = np.flatnonzero(of == image_site[of[n]])
            W[s, rows, n] = (yr[rows] * weights) @ y0[n]
        for n in range(nw):
            err = abs(0.5 * (np.sum(W[s, :, n] ** 2) + np.sum(W[s, n, :] ** 2)) - 1.0)
            if err > WWS_TOLERANCE:
                raise ValueError(
                    f"symmetry operation {s} does not transform trial orbital {n} "
                    f"({projections[n].l}, {projections[n].m}) into the orbitals on the image "
                    f"site (error {err:.2e} > {WWS_TOLERANCE}): the projections are not a "
                    "representation of the group, give whole shells or symmetric sets on "
                    "every site of the orbit")
        for n in range(nw):
            for j in np.flatnonzero(np.abs(W[s, :, n]) > 1.0e-8):
                a, b = projections[j], projections[n]
                if (a.radial, a.zona) != (b.radial, b.zona):
                    raise ValueError(
                        f"symmetry operation {s} mixes trial orbitals {n} and {j}, whose "
                        "radial parts differ (r= or zona=); the set is not a representation")
    return W, T


def _find_site(image, sites, s, site):
    d = image[None, :] - sites
    d -= np.rint(d)
    hit = np.flatnonzero(np.abs(d).max(axis=1) < 1.0e-5)
    if len(hit) != 1:
        raise ValueError(f"symmetry operation {s} maps the projection centre {site} onto "
                         f"{image}, which is not a centre of the set: the projections must "
                         "cover whole orbits of the group")
    return int(hit[0])


def wannier_representation(projections, operations: Operations, at, bg, points, stars):
    """``d_matrix_wann``, ``(nw, nw, nsym, nkptirr)``, ``compute_dmn``'s phases."""
    W, T = wannier_rotations(projections, operations, at)
    ik2ir, ir2ik, kptsym, g0 = stars
    kcart = points @ bg
    nw = len(projections)
    out = np.zeros((nw, nw, operations.nsym, len(ir2ik)), dtype=complex)
    for ir, ik in enumerate(ir2ik):
        for s in range(operations.nsym):
            rk = operations.rotations[s] @ kcart[ik]
            phase = np.exp(-1j * (T[s] @ rk)) * np.exp(-1j * (operations.translations[s] @ g0[s, ir]))
            out[:, :, s, ir] = W[s] * phase[None, :]
    return out


def _rotation_gather(states, ik: int, jk: int, R, t, g0):
    """Where each plane wave of ``jk`` (the target ``k' = R k - G0``) comes from
    at ``ik``, and the phase ``e^{-i R(k+G).t} e^{-i t.G0}``; asserts a bijection."""
    from defuqumat.topology.states import _pack

    bg = states.bg
    target = states.kcart[jk] + states.miller[jk] @ bg               # k' + G' = R (k + G)
    source = target @ R - states.kcart[ik]                          # R^T (k' + G') - k = G
    miller = source @ np.linalg.inv(bg)
    rounded = np.rint(miller)
    valid_t = np.asarray(states.valid[jk])
    if np.abs(miller - rounded)[valid_t].max(initial=0.0) > _MILLER_TOLERANCE:
        raise ValueError("a rotated plane wave is not on the reciprocal lattice: the "
                         "operation is not a symmetry of the lattice")
    keys = _pack(rounded.astype(np.int64))
    order = states.order[ik]
    sorted_keys = states.keys[ik][order]
    position = np.clip(np.searchsorted(sorted_keys, keys), 0, len(order) - 1)
    gather = order[position]
    found = (sorted_keys[position] == keys) & np.asarray(states.valid[ik])[gather]
    nvalid_i = int(np.asarray(states.valid[ik]).sum())
    if not np.all(found[valid_t]) or int(valid_t.sum()) != nvalid_i \
            or len(set(gather[valid_t].tolist())) != nvalid_i:
        raise ValueError(
            "the rotation of the plane-wave sphere is not a bijection between k and g k "
            f"({int((~found[valid_t]).sum())} of {int(valid_t.sum())} plane waves have no "
            "partner): a plane wave sits on the cutoff edge at round-off, move ecutwfc "
            "slightly")
    phase = np.exp(-1j * (target @ t)) * np.exp(-1j * (t @ g0))
    phase = np.where(valid_t, phase, 0.0)
    return gather, phase


def band_representation(states, num_bands: int, operations: Operations, stars) -> np.ndarray:
    """``d_matrix_band``, ``(num_bands, num_bands, nsym, nkptirr)``."""
    from defuqumat.topology.states import (_augmentation_term, _cached_augmentation,
                                           _permute, _project)

    if states.npol != 1:
        raise NotImplementedError("the band representation of spinors is not implemented")
    ik2ir, ir2ik, kptsym, g0 = stars
    dtype = states.coefficients.dtype
    factors = None
    if states.becp is not None:
        if states.vkb is None:
            raise ValueError("ultrasoft or PAW states need their projectors for d_matrix_band")
        factors = _cached_augmentation(states._owner, np.zeros(3))
    out = np.zeros((num_bands, num_bands, operations.nsym, len(ir2ik)), dtype=complex)
    for ir, ik in enumerate(ir2ik):
        source = states.coefficients[ik][:num_bands]
        for s in range(operations.nsym):
            jk = int(kptsym[s, ir])
            gather, phase = _rotation_gather(states, ik, jk, operations.rotations[s],
                                             operations.translations[s], g0[s, ir])
            rotated = _permute(source, jnp.asarray(gather), jnp.asarray(phase.astype(dtype)), 1)
            target = states.coefficients[jk][:num_bands]
            d = jnp.einsum("mg,ng->mn", target.conj(), rotated)
            if factors is not None:
                becp_rot = _project(rotated, states.vkb[jk], 1)
                d = d + _augmentation_term(states.becp[jk][:num_bands], becp_rot, factors)
            out[:, :, s, ir] = np.asarray(d)
    return out


def site_symmetry_data(states, projections, num_bands: int, operations: Operations,
                       points: np.ndarray, symmetrize_eps: float = 1.0e-3) -> SymmetryData:
    """The :class:`~defuqumat.wannier.sitesym.SymmetryData` of a run, in the mesh
    order of ``points``, ``d_matrix_band`` on the absolute band index."""
    at = np.asarray(states.at, dtype=float)
    bg = np.asarray(states.bg, dtype=float)
    stars = kpoint_stars(points, operations, at, bg)
    d_wann = wannier_representation(projections, operations, at, bg, points, stars)
    d_band = band_representation(states, num_bands, operations, stars)
    ik2ir, ir2ik, kptsym, _ = stars
    return SymmetryData(nsymmetry=operations.nsym, nkptirr=len(ir2ik), num_kpts=len(points),
                        ik2ir=ik2ir, ir2ik=ir2ik, kptsym=kptsym, d_matrix_wann=d_wann,
                        d_matrix_band=d_band, symmetrize_eps=float(symmetrize_eps))
