"""The GIPAW-dressed nonlocal term of the magnetic ultracell (route U, step 2).

Wang and Vafek, PRB 108, 245109 (2023), arXiv:2303.16347, build the magnetic Bloch problem
on zero-field hybrid Wannier states (:mod:`defuqumat.magnetic.ultracell_field`); their field
term is written for local potentials. A norm-conserving dataset adds ``V_NL = sum_tau
|beta_tau> D <beta_tau|``, which does **not** commute with the magnetic translations ``T(R)``
the whole basis rests on. The GIPAW dressing of Pickard and Mauri (PRB 63, 245101 (2001)),
used for the nonlocal term in a finite field by Lu et al. (PRL 135, 196401 (2025),
arXiv:2505.06585) and for Gaussians by Wang et al. (arXiv:2609.01053), restores it: each
projector is multiplied by

    p~_tau(r) = exp(-i a(tau) . d - i a(d) . d / 2) beta_tau(d),   d = r - tau,

(:meth:`~defuqumat.magnetic.conventions.MagneticCell.gipaw_phase`, the constant
``exp(-i a(tau) . tau)`` dropped since it cancels in ``|p~><p~|``), and then ``T(R) p~_tau =
exp(-i a(R) . (tau + R)) p~_{tau + R}`` exactly for a linear ``a``, so ``V_NL^B = sum_tau
|p~_tau> D <p~_tau|`` commutes with every ``T(R)``.

**What the ultracell solves with it.** The zero-field states are eigenstates of the full
zero-field Hamiltonian ``H_0 = T + V_loc + V_NL^0``, and ``H_0`` acts on the bra through
their eigenvalues (``h0 = "eigen"``). Since ``V_NL^0`` is already inside those eigenvalues,
the operator is

    H = H_0 + Delta H + (V_NL^B - V_NL^0),

with ``Delta H`` the kinetic field term of :mod:`~defuqumat.magnetic.ultracell_field` and
both nonlocal terms built from **the same sphere points**, the phase on and off, so that the
error of the grid projection cancels in the difference wherever the dressing is near one.
With ``h0 = "box"`` (``T + V_loc`` applied on the box) nothing nonlocal is inside ``H_0``
and the term added is ``V_NL^B`` itself. ``nonlocal = "bare"`` puts ``V_NL^0`` in place of
``V_NL^B`` (on the box, or left inside the eigenvalues), the control that carries the
undressed term as if it commuted with ``T(R)``.

**The grid projectors** (``# fold back into magnetic/gipaw.py``: the foundation did not
deliver them, so this route has its own, as the design asks). For every atom of the ``N1 x
1 x 1`` box (``N1 x nat`` of them, each atom's ``xi1 = j + x1`` folded to the sawtooth ``(-N1/2, N1/2]``
the field term uses, so that ``a(tau)`` is the same vector potential ``Delta H`` sees), the
box points within the cut radius of the atom are enumerated **in real space** and folded
into the box for their index. ``beta(d) = (r beta)(|d|) / |d| Y_lm(d)``: ``PP_BETA`` (``r
beta``) over ``r`` by a cubic spline on the radial mesh up to ``kkbeta``, the value at the
first mesh point below it, times :func:`~defuqumat.pseudo.harmonics.real_spherical_harmonics`;
no ``(-i)^l`` in real space (it is the transform's, :mod:`defuqumat.pseudo.projectors`). The
cut is the radius beyond which ``|beta|`` stays below ``cut`` of its largest value (the
design's 1e-6), and the norm it drops is recorded. Note on ``d``: it is the displacement from
the image of the atom the point belongs to, not the minimum image over the box. Graphene's
projector radius (2.42 bohr) exceeds half of ``a2`` (2.32 bohr), so a box point can belong
to two images along ``a2``; each contributes with its own ``d``, which is the image sum of a
function that is Bloch along ``a2``.

**The projection.** A box function is the periodic part ``u`` of ``psi = e^{iK.r} u /
sqrt(Omega_box)`` with ``K = 2 pi kappa b2`` the block's common wavevector and ``(1/npts)
sum |u|^2 = 1``, so

    <p~_tau | psi> = sqrt(Omega_box)/npts sum_{s in sphere} e^{-i phase_s} beta(d_s) e^{iK.d_s} u(r_s),

the constant ``e^{iK.tau}`` dropped (bra and ket of a block share ``K``, so it cancels in
``|p~><p~|``). The matrix element ``<chi| V T(Delta) |chi'> = sum_tau <h|p~_tau> D <p~_tau|
T(Delta) h'>`` is then two gathers over the sphere indices and two contractions; nothing is
scattered.

**What the real-space sum costs in accuracy.** It is exact up to the aliases of the
projector's transform at a full period of the box grid away from the wavefunction sphere;
for carbon's HGH ``s`` projector (``r_0 = 0.3`` bohr) on graphene's 24 x 24 x 128 grid at 50
Ry that is ``exp(-20)``, so what is left is the spline (measured on D22, see
``plans/orbital-field/u2.md``). Carbon HGH has one ``l = 0`` projector: the ``Y_lm`` ordering
and the ``l > 0`` channels are exercised only by the unit test on a dataset with ``p``
projectors, not by the graphene runs.

**Memory.** Per atom ``npt`` points (about 2300 at a 1.6 bohr cut on graphene's grid),
``(natom, npt)`` indices, phases and ``b2 . d``, ``(natom, nproj, npt)`` projector values;
the gathers of one block are ``2 nbnd natom npt`` complex words (6 MB at ``nbnd = 32``,
``N1 = 10``).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline

from defuqumat.pseudo.harmonics import real_spherical_harmonics
from defuqumat.pseudo.projectors import projector_channels
from defuqumat.units import TPI

__all__ = ["GridProjectors", "grid_projectors", "projections", "nonlocal_block", "sawtooth_cell"]


class GridProjectors(eqx.Module):
    """The projectors of every atom of the ``N1`` box on its own sphere of box points.

    ``index`` ``(natom, npt)`` flat box indices (padding points carry index 0 and value 0);
    ``beta`` ``(natom, nproj, npt)`` real, ``beta(d)`` times ``sqrt(Omega_box)/npts``;
    ``d_b2`` ``(natom, npt)`` ``b2 . d`` (for ``e^{iK.d}``, ``K = 2 pi kappa b2``);
    ``phase`` ``(natom, npt)`` the GIPAW phase (zero at zero flux); ``dij`` ``(natom,
    nproj, nproj)`` Ry, the coefficients the zero-field Hamiltonian used, zero-padded;
    ``tau`` ``(natom, 3)`` cartesian positions in the sawtooth frame (bohr); ``radius``
    and ``dropped`` per species: the cut radius (bohr) and the fraction of ``int |r beta|^2``
    beyond it.
    """

    index: jnp.ndarray
    beta: jnp.ndarray
    d_b2: jnp.ndarray
    phase: jnp.ndarray
    dij: jnp.ndarray
    tau: np.ndarray = eqx.field(static=False)
    radius: tuple = eqx.field(static=True)
    dropped: tuple = eqx.field(static=True)
    npt: int = eqx.field(static=True)

    def with_phase(self, phase) -> "GridProjectors":
        """The same projectors with another dressing phase (a control, or zero)."""
        return eqx.tree_at(lambda g: g.phase, self, jnp.asarray(phase, self.phase.dtype))


def sawtooth_cell(xi, n1: int):
    """``xi1`` (cells) folded to ``(-N1/2, N1/2]``, the frame of the field term."""
    xi = np.asarray(xi)
    return np.where(xi > n1 / 2.0, xi - n1, xi)


def _radial(pseudo, nb: int, cut: float):
    """``(spline of r beta / r, radius, dropped norm)`` for projector ``nb`` up to ``kkbeta``."""
    kk = int(pseudo.kkbeta)
    r = np.asarray(pseudo.r[:kk], dtype=float)
    rbeta = np.asarray(pseudo.projectors[nb].beta[:kk], dtype=float)
    f = rbeta / r
    spline = CubicSpline(r, f)
    big = np.max(np.abs(f))
    above = np.nonzero(np.abs(f) >= cut * big)[0] if cut > 0 else np.arange(len(r))
    radius = float(r[above[-1]]) if len(above) else float(r[-1])
    rab = np.asarray(pseudo.rab[:kk], dtype=float)
    total = float(np.sum(rbeta ** 2 * rab))
    outside = float(np.sum((rbeta ** 2 * rab)[r > radius]))
    return spline, float(r[0]), radius, outside / max(total, 1e-300)


def grid_projectors(cell, structure, pseudos, coefficients, atom_of_channel, cell_grid, n1: int,
                    magnetic=None, *, cut: float = 1.0e-6) -> GridProjectors:
    """The :class:`GridProjectors` of the ``N1 x 1 x 1`` box of ``cell`` (host setup).

    Args:
        cell: the chemical :class:`~defuqumat.system.cell.Cell` (bohr).
        structure: its :class:`~defuqumat.system.structure.Structure` (cartesian bohr).
        pseudos: the datasets, indexed by ``structure.types``.
        coefficients: ``(nkb, nkb)`` the zero-field Hamiltonian's ``D`` (Ry), channels in
            QE's order (atoms outermost, then :func:`projector_channels`).
        atom_of_channel: the atom of every channel.
        cell_grid: the chemical cell's FFT grid the box is built on.
        n1: the box length in cells along ``a1``.
        magnetic: the :class:`~defuqumat.magnetic.conventions.MagneticCell`, or ``None`` at
            zero flux (the phase is then zero).
        cut: the projector is kept where ``|r beta / r|`` is at least ``cut`` of its largest.
    """
    at = np.asarray(cell.at, dtype=float)
    b = np.linalg.inv(at).T
    grid = (int(n1) * int(cell_grid[0]), int(cell_grid[1]), int(cell_grid[2]))
    npts = int(np.prod(grid))
    omega_box = float(cell.volume) * int(n1)
    norm = np.sqrt(omega_box) / npts
    positions = np.asarray(structure.positions, dtype=float)
    frac = positions @ np.linalg.inv(at)
    coefficients = np.asarray(coefficients, dtype=float)
    atom_of_channel = np.asarray(atom_of_channel)

    per_species = {}
    for t, pseudo in enumerate(pseudos):
        if t not in structure.types:
            continue
        channels = projector_channels(pseudo)
        radial = [_radial(pseudo, nb, cut) for nb in range(len(pseudo.projectors))]
        radius = max([x[2] for x in radial], default=0.0)
        dropped = max([x[3] for x in radial], default=0.0)
        per_species[t] = (channels, radial, radius, dropped, max(pseudo.lmax, 0))

    rows = []
    for a, t in enumerate(structure.types):
        channels, radial, radius, _, lmax = per_species[t]
        nch = len(channels)
        mine = np.nonzero(atom_of_channel == a)[0]
        if len(mine) != nch:
            raise ValueError(f"atom {a} has {len(mine)} channels in D, its dataset {nch}")
        dblock = coefficients[np.ix_(mine, mine)]
        # integer grid offsets that can lie inside the sphere about this atom
        centre = frac[a] * np.asarray(cell_grid, dtype=float)
        reach = [int(np.ceil(radius * np.linalg.norm(b[i]) * cell_grid[i])) + 1 for i in range(3)]
        base = np.floor(centre).astype(np.int64)
        offs = np.stack(np.meshgrid(*[np.arange(-r_, r_ + 2) for r_ in reach], indexing="ij"),
                        -1).reshape(-1, 3)
        points = base[None, :] + offs                            # chemical-grid integer coordinates
        d = ((points / np.asarray(cell_grid, float)) - frac[a][None, :]) @ at
        keep = np.sum(d ** 2, axis=1) <= radius ** 2
        points, d = points[keep], d[keep]
        dist = np.linalg.norm(d, axis=1)
        ylm = np.asarray(real_spherical_harmonics(d, lmax)) if lmax > 0 else \
            np.full((len(d), 1), 1.0 / np.sqrt(4.0 * np.pi))
        values = np.zeros((nch, len(d)))
        for c, (nb, l, lm) in enumerate(channels):
            spline, r0, rad, _ = radial[nb]
            f = np.where(dist < r0, spline(r0), spline(np.clip(dist, r0, None)))
            f = np.where(dist <= rad, f, 0.0)
            values[c] = f * ylm[:, lm] * norm
        for j in range(int(n1)):
            # the atom's own xi1 = j + x1 folded as the grid's xi1 is, so a(tau) is the
            # vector potential Delta H sees at the atom
            js = j - int(n1) if j + frac[a][0] > n1 / 2.0 else j
            idx = points.copy()
            idx[:, 0] += j * int(cell_grid[0])
            idx = idx % np.asarray(grid, dtype=np.int64)
            flat = idx[:, 0] * (grid[1] * grid[2]) + idx[:, 1] * grid[2] + idx[:, 2]
            tau = positions[a] + js * at[0]
            rows.append((flat, values, d, tau, dblock))

    natom = len(rows)
    npt = max(len(r_[0]) for r_ in rows)
    nproj = max(r_[1].shape[0] for r_ in rows)
    index = np.zeros((natom, npt), np.int64)
    beta = np.zeros((natom, nproj, npt))
    dd = np.zeros((natom, npt, 3))
    dij = np.zeros((natom, nproj, nproj))
    tau = np.zeros((natom, 3))
    for i, (flat, values, d, t_, dblock) in enumerate(rows):
        n, m = len(flat), values.shape[0]
        index[i, :n] = flat
        beta[i, :m, :n] = values
        dd[i, :n] = d
        dij[i, :m, :m] = dblock
        tau[i] = t_
    real = cell.precision.real
    if magnetic is None:
        phase = jnp.zeros((natom, npt), dtype=real)
    else:
        phase = magnetic.gipaw_phase(jnp.asarray(dd, real), jnp.asarray(tau, real)[:, None, :])
        phase = jnp.where(jnp.asarray(np.any(beta != 0.0, axis=1)), phase, 0.0).astype(real)
    radii = tuple(float(per_species[t][2]) for t in sorted(per_species))
    dropped = tuple(float(per_species[t][3]) for t in sorted(per_species))
    return GridProjectors(index=jnp.asarray(index), beta=jnp.asarray(beta, real),
                          d_b2=jnp.asarray(dd @ b[1], real), phase=phase,
                          dij=jnp.asarray(dij, real), tau=tau, radius=radii, dropped=dropped,
                          npt=int(npt))


def _expi(x, dtype):
    return jax.lax.complex(jnp.cos(x), jnp.sin(x)).astype(dtype)


def projections(u, index, beta, d_b2, phase, kref):
    """``(P^0, P^B)``: ``<beta_tau|psi>`` and ``<p~_tau|psi>`` of box functions ``u``.

    ``u`` ``(nbnd, npts)`` periodic parts relative to ``K = 2 pi kref b2``; returns two
    ``(nbnd, natom, nproj)`` arrays (the module docstring's formula, with and without the
    dressing).
    """
    gathered = jnp.take(u, index, axis=1)                         # (nbnd, natom, npt)
    weights = beta * _expi(TPI * kref * d_b2, u.dtype)[:, None, :]   # (natom, nproj, npt)
    bare = jnp.einsum("anp,bap->ban", weights, gathered)
    dressed = jnp.einsum("anp,bap->ban", weights * _expi(-phase, u.dtype)[:, None, :], gathered)
    return bare, dressed


def nonlocal_block(u_bra, u_ket, index, beta, d_b2, phase, dij, kref, *, mode: str):
    """``<bra| V |ket>`` ``(nbnd, nbnd)`` for one block, ``mode`` one of

    ``"difference"`` (``V_NL^B - V_NL^0``, the eigenvalue ``H_0`` of a full zero-field basis),
    ``"gipaw"`` (``V_NL^B``) and ``"bare"`` (``V_NL^0``). ``u_ket`` is already rolled by
    ``Delta``; the atoms stay where they are in the bra's frame.
    """
    b0, bb = projections(u_bra, index, beta, d_b2, phase, kref)
    k0, kb = projections(u_ket, index, beta, d_b2, phase, kref)
    dij = dij.astype(u_bra.real.dtype)

    def v(left, right):
        return jnp.einsum("ban,anm,cam->bc", left.conj(), dij, right)

    if mode == "difference":
        return v(bb, kb) - v(b0, k0)
    if mode == "gipaw":
        return v(bb, kb)
    if mode == "bare":
        return v(b0, k0)
    raise ValueError(f"unknown nonlocal mode {mode!r}")
