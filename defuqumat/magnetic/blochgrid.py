"""The plane-wave magnetic Bloch route: ``pi^2 + V_loc`` on the magnetic cell's FFT box.

Lu, Chen, Wang, Dai, Fang, Gong, Wu and Xiang, *General first-principles approach to
crystals in finite magnetic fields*, PRL 135, 196401 (2025), arXiv:2505.06585 (route P of
``notes/ORBITAL-FIELD-NEXT.md``; the design's section 3, step 1). Cai and Galli, PRL 92,
186402 (2004), is the forerunner without nonlocal pseudopotentials. The conventions are
:mod:`defuqumat.magnetic.conventions` and nothing here redefines them.

The state. ``u_k`` is a complex array on the magnetic cell ``q a1 x a2 x a3`` sampled on
the box ``(q n1, n2, n3)``, with ``(n1, n2, n3)`` the chemical cell's *wavefunction* box
(the smallest FFT size holding the ``ecutwfc`` sphere, :func:`wavefunction_box`), flattened
to ``ndim = q n1 n2 n3`` with every entry a basis function. It obeys the magnetic periodic
boundary condition of Lu eq. 4, ``u(xi1 + 1, xi2, xi3) = exp(-i 2 pi p xi2) u(xi1, xi2,
xi3)``, ``xi`` the magnetic cell's fractional coordinates, and is periodic along ``xi2`` and
``xi3``. The boundary condition is carried by the operator rather than by the vector: every
grid vector is admissible, and the twist enters only where a derivative along ``xi1`` is
taken.

The kinetic term. ``pi^2 = sum_ab g^{ab} pi_a pi_b`` with ``g^{ab} = b_a^m . b_b^m`` the
magnetic cell's scaled reciprocal metric (Rydberg units, so no 1/2), ``pi_a = -i d/dxi_a +
A_a + k_a``, ``k_a = a_a^m . k = 2 pi kappa_a``, and in the Landau gauge of the conventions
``A_1 = A_3 = 0``, ``A_2 = 2 pi p xi1`` (``xi1`` in ``[0, 1)``). It is applied as ``w_b =
pi_b u``, ``v_a = sum_b g^{ab} w_b``, ``T u = sum_a pi_a v_a``: six one-axis applications,
each consistent with the twist because ``pi_b u`` obeys it again. ``pi_2`` and ``pi_3`` are an
FFT along their axis, a multiplication by ``2 pi j`` (``j`` the integer frequency), the
inverse FFT, and the pointwise terms. ``pi_1`` is Lu eq. 5: ``h = exp(i 2 pi p xi1 xi2) u``
is periodic along ``xi1``, so ``-i d/dxi1 h`` is taken by FFT along ``xi1``, and ``pi_1 u =
exp(-i 2 pi p xi1 xi2) (-i d h/dxi1) + (-2 pi p xi2 + k_1) u``; the ``-2 pi p xi2`` is the
chain rule through ``h``, not an ``A_1``. Each ``pi_a`` is Hermitian on the grid (a unitary
conjugation of a real Fourier multiplier, plus a real diagonal), so ``T`` is.

The local potential. ``SCFResult.potential[0]`` on the dense grid of the chemical cell,
Fourier-filtered to the box's frequencies ``|m_i| <= n_i/2`` (:func:`filter_to_box`, the two
Nyquist halves folded onto the box's Nyquist slot, which keeps it real) and tiled ``q`` times
along ``xi1`` (P88's tile; exact since ``V`` is chemical-periodic). The bare nonlocal term is
**not** here: it does not commute with the magnetic translations, and the GIPAW-dressed one
is step 2 of the route.

The solver. :class:`MagneticGridHamiltonian` carries the duck-typed members
:func:`~defuqumat.solvers.davidson.davidson_eigensolver` reads, so the solve is
:func:`~defuqumat.solvers.davidson.davidson_eigensolver_all` unchanged. Two of them differ
from the sphere's: the preconditioner hook :meth:`~MagneticGridHamiltonian.precondition`
(``g_psi.f90``'s formula on the box's plane waves, through a periodic 3D FFT of the residual
that ignores the twist, the review's item 4) and the start block :func:`start_block`
(random plane-wave coefficients on ``|k + G|^2 <= ecutwfc`` damped by ``1/(1 + |k + G|^2)``
and transformed to the grid; the solver's own start damps pointwise and is wrong on a grid
state). The default ``'two-gauge'`` averages that rule over the ``u`` and the ``h``
representation, because no single gauge makes a state periodic along both in-plane axes;
``'kinetic'`` is the single-gauge rule and ``'identity'`` the comparison the hook is measured
against.

Memory per k-point in flight (:func:`estimate_bytes`): ``(2.18 nvecx + 4.2 nbnd) ndim``
complex numbers (Davidson's ``psi``, ``hpsi``, the subspace, the blocks), six box-sized FFT
buffers per band in flight, and the states kept for a Chern number. Nothing of size ``nk
ndim`` is stored on the operator: ``kinetic``, ``state_kinetic`` and ``state_mask`` are rows
computed for the k-point asked (:class:`KRows`).

Refused (by the workflow, :func:`~defuqumat.workflows.magnetic_bands.run_magnetic_bands`):
the regimes of :func:`~defuqumat.magnetic.conventions.refuse_regime`; a traced ``scale``
other than one (the field is quantised by the twist, so ``d/dB`` at a fixed box is not a
derivative of anything physical here); a memory estimate above :data:`MEMORY_LIMIT`.
"""

from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.batching import map_bands
from defuqumat.basis.fftgrid import fft_grid_dimensions, gcut_from_ecut
from defuqumat.config import DEFAULT_PRECISION, Precision
from defuqumat.magnetic.conventions import MagneticCell
from defuqumat.units import TPI

__all__ = [
    "MEMORY_LIMIT", "DENSE_LIMIT", "PRECONDITIONERS", "KRows", "MagneticGridHamiltonian",
    "wavefunction_box", "filter_to_box", "tile_potential", "estimate_bytes", "refuse_memory",
    "landau_problem", "start_block", "dense_matrix", "reference_matrix", "grid_positions",
    "chemical_copies", "sphere_local_levels", "magnetic_cell", "momentum_multipliers", "WINDOWS",
]

#: The largest estimated working set a solve is started with, bytes: D22's 31 GB less
#: what the system and the compiled programs hold (the design's 26 GB).
MEMORY_LIMIT = 26.0e9

#: The largest ``ndim`` the dense fallback diagonalises (the design's refuse_if).
DENSE_LIMIT = 20000

#: The preconditioners of :meth:`MagneticGridHamiltonian.precondition`.
PRECONDITIONERS = ("two-gauge", "kinetic", "identity")

#: The frequency windows of :func:`momentum_multipliers`.
WINDOWS = ("centred", "fixed")

#: An irrational offset of the window centre, so that no grid row sits on a rounding tie
#: and a row and its magnetic translate round the same way.
_TIE_OFFSET = 1.0 / (20.0 * np.pi)


def momentum_multipliers(box, p: int, kappa, window: str = "centred", real=np.float64):
    """``(M1, M2, M3)``: ``pi_a`` as a Fourier multiplier along its axis, row by row.

    For the FFT index ``f`` along an axis of ``N`` points any integer ``m = f mod N`` is the
    same grid function, and the choice of ``m`` is the frequency window. ``pi_2`` on ``u``
    is ``2 pi (m2 + kappa2 + p xi1)`` along ``xi2`` at each row ``xi1``, ``pi_1`` on ``h`` is
    ``2 pi (m1 + kappa1 - p xi2)`` along ``xi1`` at each row ``xi2``, ``pi_3`` is ``2 pi (m3 +
    kappa3)``. ``'fixed'`` takes ``m`` in ``[-N/2, N/2)`` everywhere (Lu's FFT as written,
    with ``A_2`` and ``-2 pi p xi2`` added pointwise). ``'centred'`` (the default) centres
    each row's window on zero mechanical momentum, ``m in c + [-N/2, N/2)`` with ``c`` the
    nearest integer to ``-(kappa2 + p xi1)`` (and to ``p xi2 - kappa1``, ``-kappa3``). The
    two differ only in which aliases the products take near the box's Nyquist frequency,
    and agree wherever the states are resolved; what the centred window buys is that the
    grid operator is exactly covariant under the magnetic translation ``T(a1)`` (a row
    and its translate get the same window, the twisted rows one shifted by ``p``, which is
    the twist) and exactly periodic in ``kappa`` (``u_{k+b} = exp(-i b.r) u_k`` on the grid),
    where the fixed window breaks both at the level of the states' weight at the Nyquist
    frequency (measured in ``plans/orbital-field/route-p.md``).
    """
    n1, n2, n3 = (int(n) for n in box)
    kappa = jnp.asarray(kappa, dtype=real)
    xi1 = jnp.arange(n1, dtype=real) / n1
    xi2 = jnp.arange(n2, dtype=real) / n2
    p = float(p)

    def assign(n, centre):
        f = jnp.arange(n, dtype=real)
        if window == "fixed":
            centre = jnp.zeros_like(centre)
        else:
            centre = jnp.floor(centre + 0.5 + _TIE_OFFSET)
        return centre + jnp.mod(f - centre + n // 2, n) - n // 2

    # pi_1 on h: the axis is xi1, one window per row xi2 -> (N1, N2) indexed [f1, j2]
    m1 = jax.vmap(lambda c: assign(n1, c), out_axes=1)(p * xi2 - kappa[0])
    mult1 = TPI * (m1 + kappa[0] - p * xi2[None, :])
    # pi_2 on u: the axis is xi2, one window per row xi1 -> (N1, N2) indexed [j1, f2]
    m2 = jax.vmap(lambda c: assign(n2, c), out_axes=0)(-(p * xi1) - kappa[1])
    mult2 = TPI * (m2 + kappa[1] + p * xi1[:, None])
    m3 = assign(n3, -kappa[2])
    mult3 = TPI * (m3 + kappa[2])
    return mult1, mult2, mult3


def wavefunction_box(cell, ecutwfc: float) -> tuple[int, int, int]:
    """The chemical cell's wavefunction box: QE's grid rule at ``ecutwfc`` instead of ``ecutrho``.

    ``2 nb_i + 1`` rounded up to a 2-3-5 size, ``nb_i`` the largest Miller index inside the
    sphere ``|G|^2 < ecutwfc`` (:func:`~defuqumat.basis.fftgrid.fft_grid_dimensions`).
    Graphene at 50 Ry, ``a = 2.46`` A, ``c = 15`` A: ``(12, 12, 64)``.
    """
    at_alat = np.asarray(cell.at_alat, dtype=float)
    bg = np.asarray(cell.bg_2pi_alat, dtype=float)
    return tuple(int(n) for n in fft_grid_dimensions(at_alat, bg,
                                                     gcut_from_ecut(float(ecutwfc), cell.alat)))


def _frequencies(n: int) -> np.ndarray:
    """The integer frequencies of an ``n``-point FFT, in its own order."""
    return np.rint(np.fft.fftfreq(n, 1.0 / n)).astype(np.int64)


def filter_to_box(field, box) -> jnp.ndarray:
    """A real periodic field on one grid, as its Fourier series cut to ``|m_i| <= box_i/2`` on another.

    The coefficients are those of ``field`` (the transform is normalised by the grid size, so
    the mean is kept exactly); a component with ``|m_i| > box_i / 2`` is dropped, and the two
    Nyquist halves ``m = +-box_i/2`` of an even box land on the one slot they share, which is
    ``2 Re V(n/2)`` and real. A box larger than the field's grid is a Fourier interpolation
    (the field's own Nyquist component is split evenly by taking the real part).
    Differentiable in ``field``: the index maps are static.
    """
    field = jnp.asarray(field)
    dense = tuple(int(n) for n in field.shape)
    box = tuple(int(n) for n in box)
    coefficients = jnp.fft.fftn(field) / float(np.prod(dense))
    keep, target = [], []
    for nd, nb in zip(dense, box):
        m = _frequencies(nd)
        inside = np.abs(m) <= nb // 2 if nb % 2 == 0 else np.abs(m) <= (nb - 1) // 2
        keep.append(np.flatnonzero(inside))
        target.append(np.mod(m[inside], nb))
    kept = coefficients[np.ix_(*keep)]
    out = jnp.zeros(box, dtype=coefficients.dtype).at[np.ix_(*target)].add(kept)
    return jnp.real(jnp.fft.ifftn(out) * float(np.prod(box))).astype(field.dtype)


def tile_potential(potential_cell, q: int) -> jnp.ndarray:
    """The chemical-cell field repeated ``q`` times along ``xi1`` (P88's tile)."""
    return jnp.tile(potential_cell, (int(q), 1, 1))


def grid_positions(box) -> np.ndarray:
    """``(ndim, 3)`` fractional coordinates of the box points in the magnetic cell, flattened in C order."""
    axes = [np.arange(n) / n for n in box]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def chemical_copies(field, q: int) -> np.ndarray:
    """A field on the magnetic box ``(q n1, n2, n3)`` as its ``q`` chemical-cell copies ``(q, n1, n2, n3)``."""
    field = np.asarray(field)
    n1 = field.shape[0] // int(q)
    return field.reshape((int(q), n1) + field.shape[1:])


class KRows:
    """An ``(nk, ndim)`` array the solver indexes by ``ik``, computed for the row asked.

    The members ``kinetic``, ``state_kinetic`` and ``state_mask`` of the protocol are read
    only as ``x[ik]`` inside the solve and as ``x.dtype`` outside it, so nothing of size
    ``nk ndim`` has to exist (``np.asarray`` builds the whole array for a reader that wants
    it).
    """

    def __init__(self, row, nk: int, ndim: int, dtype):
        self._row, self._nk, self._ndim, self.dtype = row, int(nk), int(ndim), np.dtype(dtype)

    @property
    def shape(self) -> tuple[int, int]:
        return (self._nk, self._ndim)

    def __getitem__(self, ik):
        return self._row(ik)

    def __array__(self, dtype=None, copy=None):
        out = np.stack([np.asarray(self._row(i)) for i in range(self._nk)])
        return out if dtype is None else out.astype(dtype)


class MagneticGridHamiltonian(eqx.Module):
    """``pi^2 + V_loc`` of the magnetic cell at flux ``p/q`` per chemical cell, on the box.

    Fields: ``potential_box`` ``(q n1, n2, n3)`` real (Ry), ``kpoints`` ``(nk, 3)`` crystal
    coordinates of the magnetic cell's reciprocal basis, ``metric`` ``(3, 3)`` ``g^{ab} =
    b_a^m . b_b^m`` (1/bohr^2); static ``p``, ``q``, ``grid`` (the chemical wavefunction box),
    ``ecutwfc`` (Ry, for the start block), ``preconditioner`` (``'two-gauge'``,
    ``'kinetic'`` or ``'identity'``), ``band_batch`` (:func:`~defuqumat.batching.map_bands`) and ``precision``
    (the band side's dtypes). ``p = 0`` is the zero-field operator on the same box.
    """

    potential_box: jnp.ndarray
    kpoints: jnp.ndarray
    metric: jnp.ndarray
    p: int = eqx.field(static=True)
    q: int = eqx.field(static=True)
    grid: tuple = eqx.field(static=True)
    ecutwfc: float = eqx.field(static=True)
    preconditioner: str = eqx.field(static=True, default="two-gauge")
    window: str = eqx.field(static=True, default="centred")
    band_batch: object = eqx.field(static=True, default="default")
    precision: Precision = eqx.field(static=True, default=DEFAULT_PRECISION)

    # --- construction ----------------------------------------------------------
    @classmethod
    def build(cls, cell, p: int, q: int, ecutwfc: float, kpoints, potential=None, *,
              grid=None, preconditioner: str = "two-gauge", window: str = "centred",
              band_batch="default",
              precision: Precision | None = None) -> "MagneticGridHamiltonian":
        """From the chemical :class:`~defuqumat.system.cell.Cell`, the flux and the dense potential.

        ``potential`` is the chemical cell's total local potential on any grid (Ry), or
        ``None`` for the free problem; ``kpoints`` ``(nk, 2 or 3)`` magnetic crystal
        coordinates.
        """
        if preconditioner not in PRECONDITIONERS:
            raise ValueError(f"preconditioner {preconditioner!r}: one of {PRECONDITIONERS}")
        if window not in WINDOWS:
            raise ValueError(f"window {window!r}: one of {WINDOWS}")
        mcell = magnetic_cell(cell, p, q)
        precision = cell.precision if precision is None else precision
        grid = wavefunction_box(cell, ecutwfc) if grid is None else tuple(int(n) for n in grid)
        box = (int(q) * grid[0], grid[1], grid[2])
        real = cell.precision.real
        if potential is None:
            potential_box = jnp.zeros(box, dtype=real)
        else:
            potential_box = tile_potential(filter_to_box(jnp.asarray(potential, dtype=real), grid), q)
        kpoints = np.asarray(kpoints, dtype=float)
        if kpoints.ndim == 1:
            kpoints = kpoints[None, :]
        if kpoints.shape[-1] == 2:
            kpoints = np.concatenate([kpoints, np.zeros((len(kpoints), 1))], axis=-1)
        reciprocal = mcell.magnetic_reciprocal
        metric = reciprocal @ reciprocal.T
        return cls(potential_box=potential_box.astype(precision.real),
                   kpoints=jnp.asarray(kpoints, dtype=precision.real),
                   metric=jnp.asarray(metric, dtype=precision.real), p=int(p), q=int(q),
                   grid=grid, ecutwfc=float(ecutwfc), preconditioner=preconditioner,
                   window=window,
                   band_batch=band_batch, precision=precision)

    # --- the protocol Davidson reads -------------------------------------------
    @property
    def box(self) -> tuple[int, int, int]:
        return (self.q * self.grid[0], self.grid[1], self.grid[2])

    @property
    def ndim(self) -> int:
        return int(np.prod(self.box))

    @property
    def npwx(self) -> int:
        return self.ndim

    @property
    def space(self) -> int:
        return self.ndim

    @property
    def npol(self) -> int:
        return 1

    @property
    def nk(self) -> int:
        return int(self.kpoints.shape[0])

    @property
    def dtype(self):
        return self.precision.complex

    @property
    def gamma_only(self) -> bool:
        return False

    @property
    def has_overlap(self) -> bool:
        return False

    @property
    def kinetic(self) -> KRows:
        """``(nk, ndim)`` ``|k + G|^2`` of the box's plane waves, in the box's FFT order (rows)."""
        return KRows(self.kinetic_row, self.nk, self.ndim, self.precision.real)

    @property
    def state_kinetic(self) -> KRows:
        return self.kinetic

    @property
    def state_mask(self) -> KRows:
        return KRows(lambda ik: jnp.ones((self.ndim,), dtype=bool), self.nk, self.ndim, bool)

    def kinetic_row(self, ik) -> jnp.ndarray:
        """``(ndim,)`` ``(2 pi)^2 (kappa + m)^T g (kappa + m)`` for the box frequencies ``m``."""
        real = self.precision.real
        kappa = self.kpoints[ik]
        q = [TPI * (jnp.asarray(_frequencies(n), dtype=real) + kappa[a])
             for a, n in enumerate(self.box)]
        shape = [(-1, 1, 1), (1, -1, 1), (1, 1, -1)]
        q = [v.reshape(s) for v, s in zip(q, shape)]
        total = sum(self.metric[a, b] * q[a] * q[b] for a in range(3) for b in range(3))
        return total.reshape(-1)

    def diagonal(self, ik) -> jnp.ndarray:
        """``|k + G|^2 + <V>``: the box plane waves' diagonal at zero field (``usnldiag``'s ``h_diag``)."""
        return self.kinetic_row(ik) + jnp.mean(self.potential_box)

    def overlap_diagonal(self, ik) -> jnp.ndarray:
        return jnp.ones((self.ndim,), dtype=self.precision.real)

    def s_projections(self, vectors, ik):
        """Zero-width ``(becp, q becp)``, as :meth:`Hamiltonian.s_projections` without an overlap."""
        becp = vectors @ jnp.zeros((self.ndim, 0), dtype=self.dtype)
        return becp, becp

    def s_correction(self, becq, ik):
        return jnp.zeros(becq.shape[:-1] + (self.ndim,), dtype=self.dtype)

    def apply_projected(self, psi, ik):
        return (self.apply(psi, ik), *self.s_projections(psi, ik))

    def apply(self, psi, ik) -> jnp.ndarray:
        """``(pi^2 + V) psi`` for ``psi`` ``(..., ndim)``, a band block at a time."""
        kappa = self.kpoints[ik]

        def block(states):
            u = states.reshape((-1,) + self.box)
            out = self._kinetic(u, kappa) + self.potential_box * u
            return out.reshape(states.shape)

        return map_bands(block, psi.astype(self.dtype), batch=self.band_batch)

    def precondition(self, residual, energies, ik, diagonal=None, overlap_diagonal=None):
        """``g_psi.f90`` on the box's plane waves, the twist ignored (the review's item 4).

        ``'kinetic'`` (the design's form): the residual's periodic 3D transform is divided by
        QE's ``(1 + x + sqrt(1 + (x - 1)^2)) / 2``, ``x = |k + G|^2 + <V> - e``, and
        transformed back. ``'two-gauge'`` (the default): the mean of that and the same rule
        applied to ``exp(i 2 pi p xi1 xi2) r`` (the ``h`` of Lu eq. 5) and turned back. The
        twisted bundle has no gauge in which a state is periodic along both ``xi1`` and
        ``xi2`` (its Chern number is ``p``), so a periodic transform of ``u`` sees a jump at
        ``xi1 = 0`` and one of ``h`` a jump at ``xi2 = 0``; the mean puts half the weight on
        each. Measured in ``plans/orbital-field/route-p.md``: on the free problem at ``q =
        10`` and ``david = 4`` the single gauge leaves 5 of 12 bands unsettled after 200 steps
        and the mean settles in 160. ``'identity'`` returns the residual, the comparison the
        hook is measured against.
        """
        if self.preconditioner == "identity":
            return residual
        if diagonal is None:
            diagonal = self.diagonal(ik)
        r = residual.reshape((-1,) + self.box)
        x = diagonal.reshape(self.box)[None] - energies.reshape(-1, 1, 1, 1)
        denominator = 0.5 * (1.0 + x + jnp.sqrt(1.0 + (x - 1.0) ** 2))

        def g_psi(field):
            return jnp.fft.ifftn(jnp.fft.fftn(field, axes=(1, 2, 3)) / denominator, axes=(1, 2, 3))

        out = g_psi(r)
        if self.preconditioner == "two-gauge" and self.p != 0:
            phase = self._gauge_phase()
            out = 0.5 * (out + jnp.conj(phase) * g_psi(phase * r))
        return out.reshape(residual.shape).astype(residual.dtype)

    def _gauge_phase(self):
        """``exp(i 2 pi p xi1 xi2)`` on the box, ``(1, N1, N2, 1)``: ``u -> h`` of Lu eq. 5."""
        real = self.precision.real
        n1, n2, _ = self.box
        xi1 = jnp.arange(n1, dtype=real) / n1
        xi2 = jnp.arange(n2, dtype=real) / n2
        return self._phase(TPI * float(self.p) * xi1[:, None] * xi2[None, :])[None, :, :, None]

    # --- the kinetic term -------------------------------------------------------
    def _phase(self, theta):
        """``exp(i theta)`` in the band precision, without a complex literal."""
        return jax.lax.complex(jnp.cos(theta), jnp.sin(theta)).astype(self.dtype)

    def multipliers(self, kappa):
        """The three mechanical-momentum multipliers ``(M1 (N1, N2), M2 (N1, N2), M3 (N3,))``.

        :func:`momentum_multipliers` at this operator's box, flux and window.
        """
        return momentum_multipliers(self.box, self.p, kappa, self.window, self.precision.real)

    def momentum(self, u, kappa, axis: int, multipliers=None) -> jnp.ndarray:
        """``pi_a u`` along ``axis`` (0, 1, 2) for ``u`` ``(b, N1, N2, N3)`` at magnetic crystal ``kappa``.

        ``pi_2`` and ``pi_3`` are Fourier multipliers on ``u`` along their axis, ``pi_1`` one on
        ``h = exp(i 2 pi p xi1 xi2) u`` along ``xi1`` (Lu eq. 5); the multipliers are
        :func:`momentum_multipliers`, so ``A_2`` and ``-2 pi p xi2`` are inside them.
        """
        m1, m2, m3 = self.multipliers(kappa) if multipliers is None else multipliers
        if axis == 0:
            phase = self._gauge_phase()
            dh = jnp.fft.ifft(m1[None, :, :, None] * jnp.fft.fft(phase * u, axis=1), axis=1)
            return jnp.conj(phase) * dh
        if axis == 1:
            return jnp.fft.ifft(m2[None, :, :, None] * jnp.fft.fft(u, axis=2), axis=2)
        return jnp.fft.ifft(m3[None, None, None, :] * jnp.fft.fft(u, axis=3), axis=3)

    def _kinetic(self, u, kappa):
        """``sum_ab g^{ab} pi_a pi_b u``: three ``pi_b``, the metric, three ``pi_a``."""
        multipliers = self.multipliers(kappa)
        w = [self.momentum(u, kappa, b, multipliers) for b in range(3)]
        total = None
        for a in range(3):
            v = sum(self.metric[a, b] * w[b] for b in range(3))
            term = self.momentum(v, kappa, a, multipliers)
            total = term if total is None else total + term
        return total

    def vector_potential_components(self) -> np.ndarray:
        """``(ndim, 3)`` ``A_a = a_a^m . a(r)`` the operator uses at the box points (for the convention test)."""
        xi = grid_positions(self.box)
        out = np.zeros_like(xi)
        out[:, 1] = TPI * self.p * xi[:, 0]
        return out


def magnetic_cell(cell, p: int, q: int) -> MagneticCell:
    """The conventions' :class:`MagneticCell`; ``p = 0`` (``q = 1``) is the zero-field cell."""
    if int(p) == 0:
        if int(q) != 1:
            raise ValueError(f"zero field is p = 0 with q = 1, not q = {q}")
        return MagneticCell(cell=cell, scale=cell.precision.as_real(1.0), p=0, q=1)
    return MagneticCell.build(cell, int(p), int(q))


def landau_problem(cell, p: int, q: int, grid, kpoints, *, ecutwfc: float = 50.0,
                   preconditioner: str = "two-gauge", precision=None) -> MagneticGridHamiltonian:
    """``pi^2`` alone on the magnetic box (all potentials off): the free Landau problem."""
    return MagneticGridHamiltonian.build(cell, p, q, ecutwfc, kpoints, None, grid=grid,
                                         preconditioner=preconditioner, precision=precision)


def estimate_bytes(q: int, nbnd: int, david: int = 2, *, grid=(12, 12, 64),
                   band_batch: int = 1, kept: int = 0, itemsize: int = 16) -> int:
    """The working set of one k-point's solve, bytes: the design's sizing.

    ``(2.18 nvecx + 4.2 nbnd) ndim`` complex numbers for Davidson (``nvecx = david nbnd``),
    six box-sized FFT buffers per band in flight, and ``kept`` stored states (``nk`` times
    the bands a Chern number needs) of ``ndim`` each.
    """
    ndim = int(q) * int(np.prod(grid))
    nvecx = int(david) * int(nbnd)
    davidson = (2.18 * nvecx + 4.2 * int(nbnd)) * ndim * itemsize
    fft = 6 * ndim * itemsize * max(1, int(band_batch))
    return int(davidson + fft + int(kept) * ndim * itemsize)


def refuse_memory(estimate: int, limit: float | None = None, what: str = "the solve") -> None:
    """Refuse by number a working set above ``limit`` (default :data:`MEMORY_LIMIT`)."""
    limit = MEMORY_LIMIT if limit is None else float(limit)
    if estimate > limit:
        raise MemoryError(
            f"{what} needs an estimated {estimate / 1e9:.2f} GB, above the limit of "
            f"{limit / 1e9:.2f} GB; lower q, nbnd or david, or raise memory_limit")


@partial(jax.jit, static_argnames=("nbnd",))
def start_block(hamiltonian: MagneticGridHamiltonian, ik, nbnd: int) -> jnp.ndarray:
    """``(nbnd, ndim)`` random box plane waves on ``|k + G|^2 <= ecutwfc``, damped by ``1/(1 + |k+G|^2)``.

    The twist is not imposed (every grid vector is admissible and the operator carries the
    boundary condition). The key is folded with ``ik`` so that the k-points start apart,
    and fixed, so a run is reproducible. Each row is normalised.
    """
    real = hamiltonian.precision.real
    kinetic = hamiltonian.kinetic_row(ik)
    key = jax.random.fold_in(jax.random.PRNGKey(0), ik)
    keys = jax.random.split(key, 2)
    shape = (nbnd, hamiltonian.ndim)
    draw = jax.lax.complex(jax.random.uniform(keys[0], shape, dtype=real) - 0.5,
                           jax.random.uniform(keys[1], shape, dtype=real) - 0.5)
    inside = kinetic <= hamiltonian.ecutwfc
    coefficients = jnp.where(inside[None], draw / (1.0 + kinetic)[None], 0.0)
    grid = jnp.fft.ifftn(coefficients.reshape((nbnd,) + hamiltonian.box), axes=(1, 2, 3))
    grid = grid.reshape(shape)
    norm = jnp.sqrt(jnp.sum(jnp.abs(grid) ** 2, axis=1, keepdims=True))
    return (grid / norm).astype(hamiltonian.dtype)


@jax.jit
def _apply_jit(hamiltonian, psi, ik):
    return hamiltonian.apply(psi, ik)


def dense_matrix(hamiltonian: MagneticGridHamiltonian, ik: int, chunk: int = 256) -> np.ndarray:
    """``(ndim, ndim)`` the operator at ``ik``, by application on unit vectors (host complex)."""
    n = hamiltonian.ndim
    columns = []
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        unit = jnp.zeros((chunk, n), dtype=hamiltonian.dtype)
        unit = unit.at[jnp.arange(stop - start), jnp.arange(start, stop)].set(1.0)
        columns.append(np.asarray(_apply_jit(hamiltonian, unit, ik))[: stop - start])
    return np.concatenate(columns, axis=0).T


def reference_matrix(hamiltonian: MagneticGridHamiltonian, ik: int) -> np.ndarray:
    """``(ndim, ndim)`` the same operator from explicit matrices, host NumPy, for a small box.

    Each ``pi_a`` is assembled row by row from the one-axis matrices ``F^-1 diag(M) F`` of
    :func:`momentum_multipliers`, placed by Kronecker products with the row projectors and
    identities, ``pi_1`` conjugated by ``diag(exp(i 2 pi p xi1 xi2))``; ``T = sum_ab g^{ab}
    pi_a pi_b``, ``H = T + diag(V)``. It shares the multipliers with
    :meth:`MagneticGridHamiltonian.apply` and none of its code path (no FFT of a state, no
    band batching), which is what the dense check compares.
    """
    n1, n2, n3 = box = hamiltonian.box
    m1, m2, m3 = (np.asarray(m, dtype=float) for m in hamiltonian.multipliers(hamiltonian.kpoints[ik]))
    g = np.asarray(hamiltonian.metric, dtype=float)
    xi = grid_positions(box)

    def one_axis(n, multiplier):
        f = np.fft.fft(np.eye(n), axis=0)
        return np.linalg.solve(f, np.diag(multiplier) @ f)

    def row(n, j):
        e = np.zeros((n, n))
        e[j, j] = 1.0
        return e

    eye1, eye2, eye3 = np.eye(n1), np.eye(n2), np.eye(n3)
    pi3 = np.kron(eye1, np.kron(eye2, one_axis(n3, m3)))
    pi2 = sum(np.kron(row(n1, j), np.kron(one_axis(n2, m2[j]), eye3)) for j in range(n1))
    phase = np.exp(1j * TPI * hamiltonian.p * xi[:, 0] * xi[:, 1])
    inner = sum(np.kron(one_axis(n1, m1[:, j]), np.kron(row(n2, j), eye3)) for j in range(n2))
    pi1 = np.conj(phase)[:, None] * inner * phase[None, :]
    pi = [pi1, pi2, pi3]
    kinetic = sum(g[a, b] * pi[a] @ pi[b] for a in range(3) for b in range(3))
    return kinetic + np.diag(np.asarray(hamiltonian.potential_box, dtype=float).reshape(-1))


def sphere_local_levels(calculation, potential, nbnd: int, *, ethr: float = 1e-12,
                        david: int = 4, max_iterations: int = 200):
    """The a = 0 reference: the sphere Davidson of ``-nabla^2 + V`` with the nonlocal term zeroed.

    ``calculation`` a :class:`~defuqumat.scf.driver.Calculation` at the k-points wanted (a
    norm-conserving scalar one), ``potential`` the total local potential on any grid; it is
    put on the calculation's dense grid by :func:`filter_to_box` (a Fourier interpolation when
    that grid is larger), then on the smooth one, and ``D_ij`` is replaced by zeros. Returns
    ``(eigenvalues (nk, nbnd), steps (nk,), unsettled (nk,))``.
    """
    from defuqumat.basis.interpolate import to_smooth
    from defuqumat.scf.driver import LocalTerms
    from defuqumat.solvers.davidson import davidson_eigensolver_all

    dense = calculation.basis.dense
    smooth = calculation.basis.smooth
    v = filter_to_box(jnp.asarray(potential, dtype=calculation.system.cell.precision.real),
                      tuple(dense.grid))
    v = to_smooth(v, dense, smooth).astype(calculation._band_real())
    dij = calculation.projectors.dij
    terms = LocalTerms(potentials=(v,), waves=(jnp.moveaxis(v, -1, -3),),
                       deeq=jnp.zeros((1,) + tuple(dij.shape), dtype=dij.dtype))
    hamiltonian = calculation.hamiltonian_from(terms)[0]
    values, _, steps, unsettled = davidson_eigensolver_all(
        hamiltonian, nbnd, ethr=ethr, david=david, max_iterations=max_iterations,
        return_steps=True)
    return np.asarray(values), np.asarray(steps), np.asarray(unsettled)
