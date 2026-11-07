"""The magnetic ultracell: a uniform field on the zero-field hybrid Wannier basis.

Wang and Vafek, PRB 108, 245109 (2023), arXiv:2303.16347, with the conventions of
:mod:`defuqumat.magnetic.conventions` (Landau gauge ``a(r) = 2 pi n_phi (b1 . r) b2``,
electron sign, magnetic cell ``q a1 x a2 x a3``, ``k = 2 pi sum kappa_i b_i^m``). The
plane-wave route it is checked against is Lu et al., PRL 135, 196401 (2025),
arXiv:2505.06585; the Wannier route of Wang et al., arXiv:2609.01053, is its weak-field
limit (:mod:`defuqumat.magnetic.peierls`).

**The basis.** Translation by ``a2`` commutes with ``H`` (``a(a2) = 0``), so the momentum
``kappa`` along the chemical ``b2`` is conserved; ``T(a1)`` maps ``kappa`` to ``kappa -
p/q``. For a magnetic ``kappa`` in ``[0, 1/q)`` the basis function of cell ``X`` is

    chi_{n,X}(kappa) = T(X a1) h_{n,0}(kappa + X p/q)
                     = exp(-i 2 pi (p/q) X (b2 . r)) h_{n,X}(kappa + X p/q),

with ``h`` the hybrid states of :mod:`defuqumat.magnetic.hybrid`, so every ``chi_{n,X}`` has
momentum ``kappa``. The magnetic Bloch sum over the magnetic lattice (period ``q`` along
``a1``) is ``W_{n,j}(kappa1) = N^(-1/2) sum_m exp(2 pi i kappa1 m) chi_{n, j + m q}``,
``q nbnd`` functions per magnetic ``k``.

**The matrix.** ``T(X a1)`` commutes with ``pi^2 + V_loc`` and ``T(-X a1) T(X' a1) =
T(Delta a1)`` (``a(a1) . a1 = 0``), so

    <chi_{n,X}| H |chi_{n',X'}> = <h_{n,0}(kappa_X)| H T(Delta a1) |h_{n',0}(kappa_X + Delta p/q)>,

``kappa_X = kappa + X p/q``, ``Delta = X' - X``, and ``H -> 1`` for the overlap. The blocks
``B_X(Delta)``, ``S_X(Delta)`` for ``X < q`` and ``|Delta| <= w`` are the whole matrix:
``H(kappa1)_{(j,n),(j',n')} = sum_m exp(2 pi i kappa1 m) B_j(j' + m q - j)`` over every
``m`` with ``|j' + m q - j| <= w`` (any ``q``, ``q <= 2 w`` included).

**The reduced points and the Miller relabelling.** The states are solved at the reduced
``kappa + m/q``, ``m = s mod q``, ``s = X p``, ``F_X = floor(s/q)``. A function at momentum
``kappa + s/q`` is the same function as at the reduced point, with periodic part
``c_{k + F b2}(G) = c_k(G + F b2)``. Written relative to the bra's reduced wavevector
``K_X = 2 pi (kappa + m_X/q) b2``, the ket ``T(Delta a1) h_{n',0}(kappa_{X+Delta})`` is the
reduced state of ``s' = (X + Delta) p`` with its Miller index ``G2`` raised by ``D = F_X -
F_{X+Delta}`` (the relabelling and the plane-wave factor ``exp(-i 2 pi (Delta p/q) b2 . r)``
together), and the roll by ``Delta`` cells is the phase ``exp(-2 pi i i Delta / N1)`` on its
``k1 = i/N1`` component (``b2 . a1 = 0``, so no other phase). At ``p = 1`` the shift is
nonzero on the wrap blocks (``X + Delta < 0`` or ``>= q``); at ``p = 3, q = 10`` inside.

**The box evaluation.** Every block lives on the ``N1 x 1 x 1`` box of
:class:`~defuqumat.ultracell.grid.Ultracell` on the zero-field Hamiltonian's grid. ``xi1 =
b1 . r`` in cells is the sawtooth in ``(-N1/2, N1/2]`` centred on the bra (cell 0). On a ket
``phi = e^{iK.r} u``:

    Delta H phi = c1 [ 2 xi1 (b2 . p phi) - i (b1 . b2) phi ] + c1^2 |b2|^2 xi1^2 phi,
    c1 = 2 pi (p/q) scale,   p phi = e^{iK.r} FFT^-1[(K + G) FFT u],

which is ``a.p + p.a + a^2 = 2 a.p - i div a + a^2`` with ``p`` on the smooth ket only and
``xi1`` pointwise (exact where both functions are negligible at the jump ``N1/2``; the
Hermiticity residual of the assembled ``H`` measures the jump). ``div a = c1 (b1 . b2)`` is
not zero on a hexagonal cell. ``H_0`` is taken in one of two ways (``h0``):

- ``"eigen"`` (the design's): on the bra through the eigenvalues, ``<H_0 h| = sum (1/N1) ...
  eps <psi|``, a reciprocal-space sum over the bra's sphere. It is ``P H_0 P`` with ``P`` the
  bra's sphere, so where the ket's plane waves leave that sphere (a ket at ``Delta != 0``
  sits ``Delta p/q`` of ``b2`` away) the two triangles of ``H`` are different truncations
  and the Hermiticity residual carries the difference.
- ``"box"``: ``T + V_loc`` of the box applied to the ket, ``|K + G|^2`` in reciprocal space
  and ``V`` tiled from the zero-field grid in real space. It is exactly Hermitian on the box
  and equal to ``"eigen"`` whenever the ket is inside the bra's sphere (``Delta = 0``, the
  free problem, ``p = 0``).

Inner products are box sums, ``<f|g> = (1/npts) sum_r conj(u_f) u_g`` with ``u = npts
ifftn(F)`` and ``F`` the coefficients on orthonormal box plane waves.

**Solve.** ``H c = eps S c`` per magnetic ``k`` by canonical orthogonalisation: ``S = U D
U^dagger``, the columns with ``d < drop max d`` dropped, the rank and the smallest kept
``d`` recorded; ``H`` is taken as its Hermitian part and the residual reported.

**Memory.** A block holds the bra's ``nbnd`` real-space boxes and four of the ket's, each
``npts = N1 n1 n2 n3`` complex: ``5 nbnd npts`` words, 1.9 GB at ``nbnd = 32``, ``N1 = 10``
on graphene's 24 x 24 x 128 grid. The states are ``N1 q nk nbnd npwx``.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.magnetic.conventions import MagneticCell
from defuqumat.magnetic.hybrid import HybridSet, half_width, outside_weight
from defuqumat.units import TPI
from defuqumat.ultracell.grid import Ultracell

__all__ = ["MagneticUltracell", "assemble", "solve", "hermiticity"]


class MagneticUltracell(eqx.Module):
    """The frozen magnetic ultracell of one flux: the hybrid sets and the box geometry.

    ``magnetic`` is the :class:`MagneticCell` (``None`` at zero flux, ``p = 0, q = 1``, the
    check that the machinery returns the zero-field bands). ``hybrids[k][m]`` is the
    :class:`HybridSet` of the ``k``-th magnetic ``kappa`` at the reduced ``kappa + m/q``.
    ``w`` is the half width of the band in ``Delta``; ``potential`` the tiled zero-field
    local potential on the box (``h0 = "box"`` only).
    """

    magnetic: MagneticCell | None
    hybrids: tuple
    g_b2: jnp.ndarray          # (npts,) b2 . G_J, 1/bohr
    gg: jnp.ndarray            # (npts,) |G_J|^2
    xi: jnp.ndarray            # (n1_box,) sawtooth xi1 in cells
    potential: jnp.ndarray | None
    b12: float = eqx.field(static=True)
    b22: float = eqx.field(static=True)
    p: int = eqx.field(static=True)
    q: int = eqx.field(static=True)
    n1: int = eqx.field(static=True)
    nbnd: int = eqx.field(static=True)
    w: int = eqx.field(static=True)
    kappas: tuple = eqx.field(static=True)
    grid: tuple = eqx.field(static=True)
    ultracell: Ultracell = eqx.field(static=True)
    h0: str = eqx.field(static=True, default="eigen")

    @classmethod
    def build(cls, cell, flux, hybrids, kappas, cell_grid, *, nbnd: int, n1: int,
              w: int | None = None, potential=None, h0: str = "eigen", scale=1.0):
        """From the chemical :class:`Cell`, ``flux = (p, q)`` and the hybrid sets.

        ``w = None`` sets it later from the measured spread (:meth:`with_width`).
        ``potential`` is the zero-field local potential on ``cell_grid`` (needed for
        ``h0 = "box"``).
        """
        p, q = int(flux[0]), int(flux[1])
        if h0 not in ("eigen", "box"):
            raise ValueError(f"h0 must be 'eigen' or 'box', got {h0!r}")
        if p == 0:
            if q != 1:
                raise ValueError("zero flux is p/q = 0/1")
            magnetic = None
        else:
            magnetic = MagneticCell.build(cell, p, q, scale=scale)
        ultracell = Ultracell.build((n1, 1, 1), cell_grid, cell.precision)
        grid = ultracell.grid
        b = np.linalg.inv(np.asarray(cell.at, dtype=float)).T        # scaled, a_i . b_j = delta
        freq = [np.fft.fftfreq(n, 1.0 / n) for n in grid]           # signed box Miller indices
        j1, j2, j3 = np.meshgrid(*freq, indexing="ij")
        gvec = TPI * (j1[..., None] / n1 * b[0] + j2[..., None] * b[1] + j3[..., None] * b[2])
        real = cell.precision.real
        xi = np.arange(grid[0]) / cell_grid[0]
        xi = np.where(xi > n1 / 2.0, xi - n1, xi)
        box_potential = None
        if h0 == "box":
            if potential is None:
                raise ValueError("h0 = 'box' needs the zero-field local potential")
            box_potential = jnp.asarray(ultracell.tile(np.asarray(potential)), dtype=real).reshape(-1)
        return cls(magnetic=magnetic, hybrids=tuple(tuple(h) for h in hybrids),
                   g_b2=jnp.asarray((gvec @ b[1]).reshape(-1), dtype=real),
                   gg=jnp.asarray(np.sum(gvec ** 2, axis=-1).reshape(-1), dtype=real),
                   xi=jnp.asarray(xi, dtype=real), potential=box_potential,
                   b12=float(b[0] @ b[1]), b22=float(b[1] @ b[1]), p=p, q=q, n1=int(n1),
                   nbnd=int(nbnd), w=-1 if w is None else int(w),
                   kappas=tuple(float(k) for k in kappas), grid=tuple(grid),
                   ultracell=ultracell, h0=h0)

    # --- the field strength ------------------------------------------------------
    @property
    def c1(self):
        """``2 pi (p/q) scale``, the slope of ``a`` along ``b2`` per cell (1/bohr)."""
        if self.magnetic is None:
            return jnp.zeros((), dtype=self.g_b2.dtype)
        return TPI * self.magnetic.flux * self.magnetic.scale

    def with_width(self, w: int) -> "MagneticUltracell":
        return eqx.tree_at(lambda m: m.w, self, int(w))

    # --- the bra side and the spread ---------------------------------------------
    def bra_index(self, hybrid: HybridSet) -> jnp.ndarray:
        return jnp.asarray(self.ultracell.box_index(hybrid.miller, hybrid.qindex))

    def profiles(self, k: int = 0) -> np.ndarray:
        """``(q, nbnd, n1_box)`` ``|h_{n,0}|^2`` summed over the box's other two axes."""
        out = []
        for hybrid in self.hybrids[k]:
            _, profile = _bra_fields(hybrid.values, self.bra_index(hybrid), self.grid)
            out.append(np.asarray(profile))
        return np.stack(out)

    def spread(self, tol: float = 1.0e-8):
        """``(w, weights, widths)``: the measured half width and the outside weights.

        ``weights`` ``(nk, q, nbnd, len(widths))`` for ``widths = 0 .. N1//2 - 1``; ``w``
        the smallest width with every weight below ``tol``, or ``None`` when none is.
        """
        widths = np.arange(self.n1 // 2)
        xi = np.asarray(self.xi)
        weights = np.stack([np.stack([outside_weight(p, xi, widths) for p in self.profiles(k)])
                            for k in range(len(self.kappas))])
        return half_width(weights.reshape(-1, len(widths)), widths, tol), weights, widths

    # --- the blocks -------------------------------------------------------------
    def blocks(self, k: int, extra: int = 1):
        """``(B, S, B_out, S_out)`` at the ``k``-th magnetic ``kappa``.

        ``B, S`` ``(q, 2w+1, nbnd, nbnd)`` for ``Delta = -w .. w``; ``B_out, S_out``
        ``(q, extra, nbnd, nbnd)`` at ``Delta = w + 1 .. w + extra``, the first blocks the
        truncation drops (measured only while the ket stays inside half the box).
        """
        if self.w < 0:
            raise ValueError("the half width is not set: call with_width(w) first")
        q, p, n1 = self.q, self.p, self.n1
        hybrids = self.hybrids[k]
        index_cache = {}

        def ket_index(m, shift):
            key = (m, shift)
            if key not in index_cache:
                h = hybrids[m]
                miller = h.miller + np.array([0, shift, 0])
                index_cache[key] = jnp.asarray(self.ultracell.box_index(miller, h.qindex))
            return index_cache[key]

        deltas = list(range(-self.w, self.w + 1))
        tail = [d for d in range(self.w + 1, self.w + 1 + extra) if d < n1 / 2.0]
        c1 = self.c1
        potential = self.potential if self.h0 == "box" else jnp.zeros((), self.g_b2.dtype)
        B, S, B_out, S_out = [], [], [], []
        for X in range(q):
            m_x, f_x = (X * p) % q, (X * p) // q
            bra = hybrids[m_x]
            bra_index = self.bra_index(bra)
            u_bra, _ = _bra_fields(bra.values, bra_index, self.grid)
            kref = self.kappas[k] + m_x / q
            rows_b, rows_s, out_b, out_s = [], [], [], []
            for delta in deltas + tail:
                s = (X + delta) * p
                m_k, f_k = s % q, s // q
                ket = hybrids[m_k]
                phase = jnp.asarray(np.exp(-2j * np.pi * ket.qindex * delta / n1),
                                    dtype=ket.values.dtype)
                h, o = _ket_block(u_bra, bra.values, bra.h0_values, bra_index,
                                  ket.values * phase[None, :], ket_index(m_k, f_x - f_k),
                                  self.g_b2, self.gg, self.xi, potential,
                                  jnp.asarray(kref, self.g_b2.dtype), c1, self.b12, self.b22,
                                  grid=self.grid, box_h0=self.h0 == "box")
                if delta in deltas:
                    rows_b.append(h)
                    rows_s.append(o)
                else:
                    out_b.append(h)
                    out_s.append(o)
            B.append(jnp.stack(rows_b))
            S.append(jnp.stack(rows_s))
            if out_b:
                B_out.append(jnp.stack(out_b))
                S_out.append(jnp.stack(out_s))
        B, S = jnp.stack(B), jnp.stack(S)
        if B_out:
            return B, S, jnp.stack(B_out), jnp.stack(S_out)
        return B, S, None, None


@partial(jax.jit, static_argnames=("grid",))
def _bra_fields(values, index, grid):
    """The bra's real-space periodic parts ``u = npts ifftn(F)`` and its profile along ``xi1``."""
    npts = grid[0] * grid[1] * grid[2]
    coeffs = jnp.zeros((values.shape[0], npts), values.dtype).at[:, index].add(values)
    u = jnp.fft.ifftn(coeffs.reshape((values.shape[0],) + grid), axes=(1, 2, 3)) * npts
    profile = jnp.sum(jnp.abs(u) ** 2, axis=(2, 3))
    return u.reshape(values.shape[0], npts), profile


@partial(jax.jit, static_argnames=("grid", "box_h0"))
def _ket_block(u_bra, bra_values, bra_h0, bra_index, ket_values, ket_index, g_b2, gg, xi,
               potential, kref, c1, b12, b22, *, grid, box_h0):
    """``(<bra| H T(Delta) |ket>, <bra| T(Delta) |ket>)`` for one block, ``(nbnd, nbnd)`` each."""
    nbnd = ket_values.shape[0]
    npts = grid[0] * grid[1] * grid[2]
    coeffs = jnp.zeros((nbnd, npts), ket_values.dtype).at[:, ket_index].add(ket_values)
    on_bra = coeffs[:, bra_index]                                    # (nbnd, Pbra)
    overlap = bra_values.conj() @ on_bra.T
    kb = TPI * kref                                                  # K = kb b2
    if box_h0:
        kinetic = gg + 2.0 * kb * g_b2 + kb * kb * b22               # |K + G|^2
        h0 = (bra_values.conj() * kinetic[bra_index]) @ on_bra.T
    else:
        h0 = bra_h0.conj() @ on_bra.T
    shape = (nbnd,) + grid
    u = (jnp.fft.ifftn(coeffs.reshape(shape), axes=(1, 2, 3)) * npts).reshape(nbnd, npts)
    momentum = (g_b2 + kb * b22).astype(coeffs.real.dtype)          # b2 . (K + G)
    up = (jnp.fft.ifftn((coeffs * momentum).reshape(shape), axes=(1, 2, 3)) * npts)
    up = up.reshape(nbnd, npts)
    x = jnp.repeat(xi, grid[1] * grid[2])                            # xi1 at every box point
    field = c1 * 2.0 * x * up + (c1 * c1 * b22) * (x * x) * u
    h_field = (u_bra.conj() @ field.T) / npts - 1j * c1 * b12 * overlap
    if box_h0:
        h0 = h0 + (u_bra.conj() @ (potential * u).T) / npts
    return h0 + h_field, overlap


def assemble(kappa1: float, B, S, w: int, q: int):
    """The dense ``(q nbnd, q nbnd)`` ``H`` and ``S`` at magnetic ``kappa1`` from the blocks.

    Row ``(j, n)`` is ``j nbnd + n``; ``H_{(j,n),(j',n')} = sum_m e^{2 pi i kappa1 m}
    B_j(j' + m q - j)`` over every ``m`` with ``|j' + m q - j| <= w``.
    """
    B, S = np.asarray(B), np.asarray(S)
    nbnd = B.shape[-1]
    h = np.zeros((q * nbnd, q * nbnd), dtype=complex)
    s = np.zeros_like(h)
    for j in range(q):
        for jp in range(q):
            lo = int(np.ceil((-w - jp + j) / q))
            hi = int(np.floor((w - jp + j) / q))
            for m in range(lo, hi + 1):
                d = jp + m * q - j
                phase = np.exp(2j * np.pi * kappa1 * m)
                h[j * nbnd:(j + 1) * nbnd, jp * nbnd:(jp + 1) * nbnd] += phase * B[j, d + w]
                s[j * nbnd:(j + 1) * nbnd, jp * nbnd:(jp + 1) * nbnd] += phase * S[j, d + w]
    return h, s


def hermiticity(h) -> float:
    """``max |H - H^dagger| / max |H|``."""
    h = np.asarray(h)
    return float(np.max(np.abs(h - h.conj().T)) / max(np.max(np.abs(h)), 1e-300))


def solve(h, s, drop: float = 1.0e-8):
    """``(eps, rank, d_min, d_all)``: the generalized problem by canonical orthogonalisation.

    ``d_min`` is the smallest kept eigenvalue of ``S`` relative to the largest, ``d_all``
    the smallest of all; both Hermitian parts are used.
    """
    h = 0.5 * (np.asarray(h) + np.asarray(h).conj().T)
    s = 0.5 * (np.asarray(s) + np.asarray(s).conj().T)
    d, u = np.linalg.eigh(s)
    top = d.max()
    keep = d > drop * top
    x = u[:, keep] / np.sqrt(d[keep])[None, :]
    eps = np.linalg.eigvalsh(x.conj().T @ h @ x)
    return eps, int(keep.sum()), float(d[keep].min() / top), float(d.min() / top)
