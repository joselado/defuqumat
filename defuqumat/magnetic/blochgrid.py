"""The plane-wave magnetic Bloch route: ``pi^2 + V_loc + V_NL`` on the magnetic cell's FFT box.

Lu, Chen, Wang, Dai, Fang, Gong, Wu and Xiang, *General first-principles approach to
crystals in finite magnetic fields*, PRL 135, 196401 (2025), arXiv:2505.06585 (route P of
``notes/ORBITAL-FIELD-NEXT.md``; the design's section 3, steps 1 and 2). Cai and Galli, PRL 92,
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
conjugation of a real Fourier multiplier, plus a real diagonal), so ``T`` is. Each row's
frequencies are taken in a window centred on zero mechanical momentum
(:func:`momentum_multipliers`), which makes the grid operator exactly covariant under the
magnetic translation ``T(a1)`` and exactly periodic in ``kappa``; Lu's FFT as written
(``window = 'fixed'``) is neither, by the states' weight at the box's Nyquist frequency.

The local potential. ``SCFResult.potential[0]`` on the dense grid of the chemical cell, tiled
``q`` times along ``xi1`` (P88's tile; exact since ``V`` is chemical-periodic). By default
(``local = 'two-grid'``) ``V u`` is formed in QE's ``vloc_psi`` order: ``u`` interpolated onto
the dense grid one axis at a time in the gauge where it is periodic, multiplied, restricted
back (:meth:`MagneticGridHamiltonian.local_term`). The design's first form, ``V`` filtered to
the box's frequencies ``|m_i| <= n_i/2`` (:func:`filter_to_box`) and multiplied on the box
(``local = 'filtered'``), aliases the products, and at zero field puts the deep states of
local-only graphene 0.27 Ry below the sphere's at 50 Ry.

The nonlocal term (step 2, Lu eq. 6). The bare one does not commute with the magnetic
translations; each projector is dressed by the GIPAW phase of the conventions (Pickard and
Mauri, PRB 63, 245101 (2001)) and lives in real space on the potential's grid, as a sphere of
points per atom of the magnetic cell (:class:`GridProjectors`). It rides on the two-grid
interpolation: ``x = I u``, ``y = V x + c W^dagger D W x``, ``I^dagger y``, so the whole
operator is one Galerkin matrix and stays Hermitian. Against ``vkb`` at zero field the
projections agree to 1.1e-7 absolute, 4.2e-7 of the largest (graphene, HGH), and 2.7e-6
absolute, 6.4e-6 of the largest (silicon, ``vbc`` s and p) (``plans/orbital-field/route-p.md``). Norm-conserving only; ultrasoft and PAW are refused
by name.

``p < 0`` is the field along ``-a3`` (:func:`magnetic_cell`), which the magnetizability's
``E(-dB)`` needs.

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
buffers and three of the dense grid per band in flight, and the states kept for a Chern
number. Nothing of size ``nk
ndim`` is stored on the operator: ``kinetic``, ``state_kinetic`` and ``state_mask`` are rows
computed for the k-point asked (:class:`KRows`).

Refused (by the workflow, :func:`~defuqumat.workflows.magnetic_bands.run_magnetic_bands`):
the regimes of :func:`~defuqumat.magnetic.conventions.refuse_regime`; a memory estimate
above :data:`MEMORY_LIMIT`. The conventions' traced ``scale`` is not used here: the field
enters through the integer ``p`` of the twist, so ``B`` is quantised by the box and a ``d/dB``
at fixed ``p/q`` is not a derivative of this operator (Lu's magnetizability is a finite
difference over fluxes for the same reason). ``kappa`` is traced, so ``d/dk`` is available.
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
    "LOCAL_TERMS", "GridProjectors", "grid_projectors", "PROJECTOR_THRESHOLD", "sphere_levels",
    "sphere_state_on_box",
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

#: A projector's sphere ends where ``|r beta(r)|`` has fallen below this fraction of its
#: largest value beyond (the design's cut; the dropped norm is in :attr:`GridProjectors.dropped`).
PROJECTOR_THRESHOLD = 1e-6

#: The forms of the local term, :meth:`MagneticGridHamiltonian.local_term`.
LOCAL_TERMS = ("two-grid", "filtered")

#: An irrational offset of the window centre, so that no grid row sits on a rounding tie
#: and a row and its magnetic translate round the same way.
_TIE_OFFSET = 1.0 / (20.0 * np.pi)


def momentum_multipliers(box, p: int, kappa, window: str = "centred", real=None):
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
    real = DEFAULT_PRECISION.real if real is None else real
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
    potential_fine: jnp.ndarray | None = None
    projectors: "GridProjectors | None" = None
    p: int = eqx.field(static=True, default=1)
    q: int = eqx.field(static=True, default=1)
    grid: tuple = eqx.field(static=True, default=(1, 1, 1))
    ecutwfc: float = eqx.field(static=True, default=0.0)
    local: str = eqx.field(static=True, default="two-grid")
    preconditioner: str = eqx.field(static=True, default="two-gauge")
    window: str = eqx.field(static=True, default="centred")
    band_batch: object = eqx.field(static=True, default="default")
    precision: Precision = eqx.field(static=True, default=DEFAULT_PRECISION)

    # --- construction ----------------------------------------------------------
    @classmethod
    def build(cls, cell, p: int, q: int, ecutwfc: float, kpoints, potential=None, *,
              grid=None, preconditioner: str = "two-gauge", window: str = "centred",
              local: str = "two-grid", band_batch="default",
              precision: Precision | None = None, structure=None, pseudos=None,
              projector_threshold: float = PROJECTOR_THRESHOLD,
              dressed: bool = True) -> "MagneticGridHamiltonian":
        """From the chemical :class:`~defuqumat.system.cell.Cell`, the flux and the dense potential.

        ``potential`` is the chemical cell's total local potential on its dense grid (Ry),
        or ``None`` for the free problem; ``kpoints`` ``(nk, 2 or 3)`` magnetic crystal
        coordinates. ``local`` is ``'two-grid'`` (the default: ``V u`` on the potential's own
        grid tiled ``q`` times, QE's ``vloc_psi`` order, :meth:`local_term`) or ``'filtered'``
        (the design's first form: ``V`` Fourier-filtered to the box and multiplied there).
        ``structure`` and ``pseudos`` (norm-conserving) put the GIPAW-dressed nonlocal term
        in (:func:`grid_projectors`), on the potential's grid, which needs ``'two-grid'``.
        """
        if local not in LOCAL_TERMS:
            raise ValueError(f"local {local!r}: one of {LOCAL_TERMS}")
        if preconditioner not in PRECONDITIONERS:
            raise ValueError(f"preconditioner {preconditioner!r}: one of {PRECONDITIONERS}")
        if window not in WINDOWS:
            raise ValueError(f"window {window!r}: one of {WINDOWS}")
        mcell = magnetic_cell(cell, p, q)
        precision = cell.precision if precision is None else precision
        grid = wavefunction_box(cell, ecutwfc) if grid is None else tuple(int(n) for n in grid)
        box = (int(q) * grid[0], grid[1], grid[2])
        real = cell.precision.real
        potential_fine = None
        if potential is None:
            potential_box = jnp.zeros(box, dtype=real)
        else:
            potential = jnp.asarray(potential, dtype=real)
            potential_box = tile_potential(filter_to_box(potential, grid), q)
            if local == "two-grid":
                if any(nf < nb for nf, nb in zip(potential.shape, grid)):
                    raise ValueError(
                        f"the two-grid local term needs the potential's grid {potential.shape} "
                        f"at least the wavefunction box {grid} along every axis")
                potential_fine = tile_potential(potential, q).astype(precision.real)
        kpoints = np.asarray(kpoints, dtype=float)
        if kpoints.ndim == 1:
            kpoints = kpoints[None, :]
        if kpoints.shape[-1] == 2:
            kpoints = np.concatenate([kpoints, np.zeros((len(kpoints), 1))], axis=-1)
        reciprocal = mcell.magnetic_reciprocal
        metric = reciprocal @ reciprocal.T
        projectors = None
        if pseudos is not None:
            if potential_fine is None:
                raise ValueError("the nonlocal term lives on the potential's grid: it needs a "
                                 "potential and local='two-grid'")
            projectors = grid_projectors(mcell, structure, pseudos, potential_fine.shape,
                                         int(q) * int(np.prod(grid)),
                                         threshold=projector_threshold, dressed=dressed,
                                         precision=precision)
        return cls(potential_box=potential_box.astype(precision.real),
                   potential_fine=potential_fine, local=local, projectors=projectors,
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
        """``(ndim,)`` ``(2 pi)^2 (kappa + m)^T g (kappa + m)`` for the box frequencies ``m``.

        ``m`` is the plain FFT order (the fixed window) and ``A`` is left out: this feeds only
        the preconditioner and the start block, both approximations, never the operator.
        """
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
            out = self._kinetic(u, kappa) + self.local_term(u, kappa)
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

    # --- the local term ---------------------------------------------------------
    def local_term(self, u, kappa) -> jnp.ndarray:
        """``V u`` for ``u`` ``(b, N1, N2, N3)``.

        ``'filtered'``: the product on the box with ``V`` filtered to it. ``'two-grid'``: QE's
        order (``vloc_psi``): ``u`` is Fourier-interpolated onto the potential's grid, one axis
        at a time in the representation where it is periodic (``xi3`` and ``xi2`` on ``u``,
        each ``xi2`` row in its centred window; then ``xi1`` on ``h = exp(i 2 pi p xi1 xi2)
        u`` at every fine ``xi2`` row), multiplied by ``V`` there, and restricted back by the
        adjoint steps. It is ``I^dagger V I`` up to the grids' ratio, so Hermitian, the
        identity for a constant ``V``, and exact for the products a band-limited ``V`` makes
        with a band-limited state (the box's products alias on the box itself).
        """
        if self.potential_fine is None:
            return self.potential_box * u
        return self._two_grid(u, kappa)

    def _two_grid(self, u, kappa):
        x = self.interpolate(u, kappa)
        y = self.potential_fine[None] * x
        if self.projectors is not None:
            b = x.shape[0]
            y = y + self.projectors.apply(x.reshape(b, -1), kappa).reshape(x.shape)
        return self.restrict(y, kappa)

    def _fine_maps(self, kappa):
        """The index maps and gauge phases of the coarse-to-fine interpolation at ``kappa``."""
        real = self.precision.real
        n1, n2, n3 = self.box
        f1, f2, f3 = self.potential_fine.shape
        p = float(self.p)
        kappa = jnp.asarray(kappa, dtype=real)

        def window(n, centre):
            if self.window == "fixed":
                centre = jnp.zeros_like(centre)
            else:
                centre = jnp.floor(centre + 0.5 + _TIE_OFFSET)
            f = jnp.arange(n, dtype=real)
            return centre + jnp.mod(f - centre + n // 2, n) - n // 2

        # the fine-grid index each coarse frequency lands on, per axis and row
        i3 = jnp.mod(window(n3, -kappa[2]), f3).astype(jnp.int32)
        xi1c = jnp.arange(n1, dtype=real) / n1
        rows2 = jax.vmap(lambda c: window(n2, c))(-(p * xi1c) - kappa[1])      # (n1, n2)
        i2 = jnp.mod(rows2, f2).astype(jnp.int32)
        xi2f = jnp.arange(f2, dtype=real) / f2
        rows1 = jax.vmap(lambda c: window(n1, c), out_axes=1)(p * xi2f - kappa[0])  # (n1, f2)
        i1 = jnp.mod(rows1, f1).astype(jnp.int32)
        r1 = jnp.arange(n1)[:, None]
        c2 = jnp.arange(f2)[None, :]
        coarse_phase = self._phase(TPI * p * xi1c[:, None] * xi2f[None, :])[None, :, :, None]
        xi1f = jnp.arange(f1, dtype=real) / f1
        fine_phase = self._phase(TPI * p * xi1f[:, None] * xi2f[None, :])[None, :, :, None]
        return i1, i2, i3, r1, c2, coarse_phase, fine_phase

    def interpolate(self, u, kappa) -> jnp.ndarray:
        """``u`` ``(b, N1, N2, N3)`` on the box -> ``x`` ``(b, F1, F2, F3)`` on the potential's grid.

        One axis at a time in the gauge where ``u`` is periodic (``xi3``, ``xi2`` on ``u``,
        ``xi1`` on ``h``), zero-padded in each row's centred window. ``x`` is ``c u(r_l)``
        with ``c = N_box / N_fine`` the ratio of the two grids' sizes and ``u(r)`` the
        band-limited twisted function the box vector stands for; :meth:`restrict` is its
        adjoint, so ``restrict(V interpolate(u))`` is the Galerkin matrix of ``V``.
        """
        i1, i2, i3, r1, c2, coarse_phase, fine_phase = self._fine_maps(kappa)
        n1, n2, n3 = self.box
        f1, f2, f3 = self.potential_fine.shape
        b = u.shape[0]
        dtype = self.dtype
        x = jnp.fft.fft(u, axis=3)
        x = jnp.zeros((b, n1, n2, f3), dtype).at[..., i3].set(x)
        x = jnp.fft.ifft(x, axis=3)
        x = jnp.fft.fft(x, axis=2)
        x = jnp.zeros((b, n1, f2, f3), dtype).at[:, r1, i2, :].set(x)
        x = jnp.fft.ifft(x, axis=2)
        x = jnp.fft.fft(coarse_phase * x, axis=1)
        x = jnp.zeros((b, f1, f2, f3), dtype).at[:, i1, c2, :].set(x)
        return jnp.conj(fine_phase) * jnp.fft.ifft(x, axis=1)

    def restrict(self, y, kappa) -> jnp.ndarray:
        """The adjoint of :meth:`interpolate`: ``(b, F1, F2, F3)`` -> ``(b, N1, N2, N3)``."""
        i1, i2, i3, r1, c2, coarse_phase, fine_phase = self._fine_maps(kappa)
        y = jnp.fft.fft(fine_phase * y, axis=1)[:, i1, c2, :]
        y = jnp.conj(coarse_phase) * jnp.fft.ifft(y, axis=1)
        y = jnp.fft.fft(y, axis=2)[:, r1, i2, :]
        y = jnp.fft.ifft(y, axis=2)
        y = jnp.fft.fft(y, axis=3)[..., i3]
        return jnp.fft.ifft(y, axis=3)

    @property
    def grid_ratio(self) -> float:
        """``c = N_box / N_fine``: :meth:`interpolate` returns ``c u(r_l)``."""
        return float(self.ndim) / float(np.prod(self.potential_fine.shape))

    def with_potential(self, potential) -> "MagneticGridHamiltonian":
        """The same operator with the chemical cell's total local potential replaced (same grid).

        What an SCF iteration changes: ``potential`` ``(f1, F2, F3)`` on the chemical cell's
        dense grid, tiled ``q`` times; the box copy (the preconditioner's mean) is filtered.
        No shape changes, so nothing compiled is invalidated.
        """
        real = self.precision.real
        potential = jnp.asarray(potential, dtype=real)
        box = tile_potential(filter_to_box(potential, self.grid), self.q).astype(real)
        fine = tile_potential(potential, self.q).astype(real)
        if self.potential_fine is None or fine.shape != self.potential_fine.shape:
            raise ValueError(f"the potential's grid {potential.shape} is not this operator's "
                             f"{None if self.potential_fine is None else self.potential_fine.shape}")
        return eqx.tree_at(lambda h: (h.potential_box, h.potential_fine), self, (box, fine))

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
    if int(p) < 0:
        # The field along -a3: the conventions' cell at |p| with its traced scale at -1, so
        # a(r) is the conventions' at -|p|/q and the operator's twist is the integer p < 0.
        built = MagneticCell.build(cell, -int(p), int(q))
        return MagneticCell(cell=built.cell, scale=cell.precision.as_real(-1.0), p=built.p,
                            q=built.q, axis=built.axis)
    return MagneticCell.build(cell, int(p), int(q))


class GridProjectors(eqx.Module):
    """The GIPAW-dressed nonlocal projectors of the magnetic cell on the potential's grid.

    ``# fold back into magnetic/gipaw.py`` (the design's note: the foundation did not deliver
    the grid projectors, so route P builds its own here).

    For atom ``a`` at ``tau`` and each point of its sphere, ``index`` ``(natom, npts)`` is the
    flat index of the in-cell grid point ``r_l`` and ``displacement`` ``(natom, npts, 3)`` the
    actual ``d = r_l + P_l - tau`` (bohr), ``P_l`` the magnetic lattice vector that brings the
    sphere point back into the cell. Every image is enumerated, so a sphere wider than the
    cell is the sum over images that the plane-wave ``vkb`` is. ``values`` ``(natom, nproj,
    npts)`` is the k-independent half of the bra,

        sqrt(dV_box) beta_lm(d) exp(-i phi(d)) exp(-i a(P_l) . r_l),

    with ``beta_lm(d) = (r beta_l)(|d|)/|d| Y_lm(d)`` (``PP_BETA`` times
    :func:`~defuqumat.pseudo.harmonics.real_solid_harmonics`, the same ``Y_lm`` as ``vkb``, so
    the ``(-i)^l`` of ``vkb`` comes out of the plane-wave expansion rather than being put in),
    ``phi = -a(tau) . d - a(d) . d / 2`` the GIPAW phase of the conventions (Pickard and Mauri,
    PRB 63, 245101 (2001); Lu et al. eq. 6), and the last factor the magnetic Bloch condition
    ``u(r + P) = exp(-i a(P) . r) u(r)``. At ``kappa`` the bra is ``W = values exp(i k . d)``
    (the constant ``exp(i k . tau)`` is dropped: it cancels in ``|p><p|``), ``becp = W x`` on
    the interpolated state ``x`` of :meth:`MagneticGridHamiltonian.interpolate`, and the term
    added on the fine grid before the restriction is ``c W^dagger D becp``, ``c`` the grids'
    size ratio, which makes ``<u|V_NL|u> = becp^dagger D becp`` exactly. The quadrature is the
    fine grid's: the product ``beta x`` is resolved there up to the projector's own tail
    beyond twice the grid's Nyquist frequency.

    ``dij`` ``(natom, nproj, nproj)`` Ry, zero-padded where a species has fewer channels;
    ``reciprocal`` the magnetic cell's scaled reciprocal vectors (rows); ``ratio`` ``c``;
    ``dropped`` the largest fraction of ``int (r beta)^2 dr`` beyond a sphere's radius.
    """

    index: jnp.ndarray
    displacement: jnp.ndarray
    values: jnp.ndarray
    dij: jnp.ndarray
    reciprocal: jnp.ndarray
    ratio: float = eqx.field(static=True, default=1.0)
    dropped: float = eqx.field(static=True, default=0.0)
    radius: tuple = eqx.field(static=True, default=())

    @property
    def natom(self) -> int:
        return int(self.index.shape[0])

    @property
    def nproj(self) -> int:
        return int(self.values.shape[1])

    def bra(self, kappa) -> jnp.ndarray:
        """``W`` ``(natom, nproj, npts)`` at magnetic crystal ``kappa``."""
        real = self.displacement.dtype
        k = TPI * jnp.asarray(kappa, dtype=real) @ self.reciprocal
        theta = jnp.einsum("apc,c->ap", self.displacement, k)
        phase = jax.lax.complex(jnp.cos(theta), jnp.sin(theta)).astype(self.values.dtype)
        return self.values * phase[:, None, :]

    def project(self, x, kappa) -> jnp.ndarray:
        """``becp`` ``(b, natom, nproj)`` of fine-grid states ``x`` ``(b, Nfine)``."""
        return jnp.einsum("anp,bap->ban", self.bra(kappa), x[:, self.index])

    def apply(self, x, kappa) -> jnp.ndarray:
        """``c W^dagger D W x`` on the fine grid, ``(b, Nfine)``.

        The scatter is ``.at[].add`` (the CPU form; on a card ``jax.ops.segment_sum`` is the
        one to use, not built here).
        """
        w = self.bra(kappa)
        becp = jnp.einsum("anp,bap->ban", w, x[:, self.index])
        dbecp = jnp.einsum("aij,baj->bai", self.dij.astype(w.dtype), becp)
        values = self.ratio * jnp.einsum("anp,ban->bap", jnp.conj(w), dbecp)
        return jnp.zeros_like(x).at[:, self.index].add(values)


def _radial_tables(pseudo, threshold: float):
    """Per projector ``(l, spline of (r beta)/r^(l+1), radius)``, and the dropped norm fraction.

    The radius is the last mesh point where ``|r beta|`` is above ``threshold`` of its
    maximum, within the dataset's ``kkbeta`` (QE's cut, beyond which the tabulated projector
    is zero and which the plane-wave form factors also integrate to).
    """
    from scipy.interpolate import CubicSpline

    r = np.asarray(pseudo.r, dtype=float)
    out, dropped = [], 0.0
    for projector in pseudo.projectors:
        cut = int(projector.cutoff_index) if projector.cutoff_index else len(r)
        cut = min(cut, int(pseudo.kkbeta))
        rb = np.asarray(projector.beta, dtype=float)[:cut]
        rr = r[:cut]
        big = np.flatnonzero(np.abs(rb) >= threshold * np.max(np.abs(rb)))
        last = min(cut, (int(big[-1]) + 2) if big.size else cut)
        weights = np.gradient(rr)
        total = np.sum(rb ** 2 * weights)
        dropped = max(dropped, float(np.sum(rb[last:] ** 2 * weights[last:]) / total))
        l = int(projector.l)
        keep = rr[:last] > 0
        spline = CubicSpline(rr[:last][keep], rb[:last][keep] / rr[:last][keep] ** (l + 1))
        out.append((l, spline, float(rr[last - 1])))
    return out, dropped


def grid_projectors(mcell: MagneticCell, structure, pseudos, fine_shape, box_ndim: int, *,
                    threshold: float = PROJECTOR_THRESHOLD, dressed: bool = True,
                    precision: Precision | None = None) -> GridProjectors:
    """:class:`GridProjectors` for the ``q natoms`` atoms of the magnetic cell (host NumPy).

    ``structure`` the chemical cell's :class:`~defuqumat.system.structure.Structure`
    (cartesian bohr); ``pseudos`` its datasets, norm-conserving (an ultrasoft or PAW one is
    refused by name: its augmentation charge and overlap are not on the box); ``fine_shape``
    the magnetic fine grid ``(q f1, F2, F3)``; ``box_ndim`` the box size, for ``dV_box``.
    Atom ``j natoms + a`` of the magnetic cell is chemical atom ``a`` moved by ``j a1``.
    ``dressed=False`` drops the GIPAW phase (the bare projectors, which do not commute with
    the magnetic translations): only for the test that the covariance check can fail.
    """
    from defuqumat.pseudo.harmonics import real_solid_harmonics
    from defuqumat.pseudo.projectors import _expand_dij, projector_channels

    precision = mcell.cell.precision if precision is None else precision
    for pseudo in pseudos:
        if pseudo.is_paw or pseudo.is_ultrasoft:
            raise NotImplementedError(
                f"route P's nonlocal term is norm-conserving only: the dataset "
                f"{pseudo.element.strip()} is {'PAW' if pseudo.is_paw else 'ultrasoft'}, whose "
                "augmentation charge and overlap S are not on the magnetic box")
    lattice = np.asarray(mcell.magnetic_lattice, dtype=float)
    reciprocal = np.asarray(mcell.magnetic_reciprocal, dtype=float)
    a1 = np.asarray(mcell.lattice, dtype=float)[0]
    shape = np.asarray([int(n) for n in fine_shape])
    volume = abs(float(np.linalg.det(lattice)))
    sqrt_dv = np.sqrt(volume / float(box_ndim))
    tables, dropped, radii = {}, 0.0, []
    for t, pseudo in enumerate(pseudos):
        tables[t], fraction = _radial_tables(pseudo, threshold)
        dropped = max(dropped, fraction)
    channels = {t: projector_channels(p) for t, p in enumerate(pseudos)}
    nproj = max(len(c) for c in channels.values()) if channels else 0
    lmax = max((p.lmax for p in pseudos), default=0)
    positions = np.asarray(structure.positions, dtype=float)
    atoms = [(positions[a] + j * a1, structure.types[a])
             for j in range(mcell.q) for a in range(len(structure.types))]
    rows = []
    for tau, t in atoms:
        radius = max((rc for _, _, rc in tables[t]), default=0.0)
        radii.append(radius)
        frac = reciprocal @ tau
        width = radius * np.linalg.norm(reciprocal, axis=1)
        lo = np.floor((frac - width) * shape).astype(int) - 1
        hi = np.ceil((frac + width) * shape).astype(int) + 1
        axes = [np.arange(l0, h0 + 1) for l0, h0 in zip(lo, hi)]
        ints = np.stack([g.reshape(-1) for g in np.meshgrid(*axes, indexing="ij")], axis=-1)
        r = (ints / shape) @ lattice
        d = r - tau
        inside = np.sum(d * d, axis=-1) <= radius ** 2
        ints, d, r = ints[inside], d[inside], r[inside]
        wraps = np.floor_divide(ints, shape)
        flat = np.ravel_multi_index(tuple((ints - wraps * shape).T), tuple(shape))
        big_p = wraps @ lattice
        r_in = r - big_p
        solid = np.asarray(real_solid_harmonics(jnp.asarray(d), lmax))
        dist = np.sqrt(np.sum(d * d, axis=-1))
        beta = np.zeros((nproj, len(flat)))
        for i, (nb, _, lm) in enumerate(channels[t]):
            _, spline, rc = tables[t][nb]
            beta[i] = np.where(dist <= rc, spline(np.minimum(dist, rc)), 0.0) * solid[:, lm]
        phi = (np.asarray(mcell.gipaw_phase(jnp.asarray(d), jnp.asarray(tau))) if dressed
               else np.zeros(len(flat)))
        bloch = np.asarray(jnp.sum(mcell.a(jnp.asarray(big_p)) * jnp.asarray(r_in), axis=-1))
        theta = -(phi + bloch)
        phase = np.asarray(jax.lax.complex(jnp.asarray(np.cos(theta), dtype=precision.real),
                                           jnp.asarray(np.sin(theta), dtype=precision.real)))
        values = sqrt_dv * beta * phase[None, :]
        dij = np.zeros((nproj, nproj))
        block = _expand_dij(pseudos[t], channels[t])
        dij[: block.shape[0], : block.shape[1]] = block
        rows.append((flat, d, values, dij))
    npts = max(len(f) for f, _, _, _ in rows)
    natom = len(rows)
    index = np.zeros((natom, npts), dtype=np.int32)
    disp = np.zeros((natom, npts, 3))
    vals = np.zeros((natom, nproj, npts), dtype=precision.complex)
    dijs = np.zeros((natom, nproj, nproj))
    for a, (flat, d, values, dij) in enumerate(rows):
        n = len(flat)
        index[a, :n], disp[a, :n], vals[a, :, :n], dijs[a] = flat, d, values, dij
    return GridProjectors(
        index=jnp.asarray(index), displacement=jnp.asarray(disp, dtype=precision.real),
        values=jnp.asarray(vals, dtype=precision.complex),
        dij=jnp.asarray(dijs, dtype=precision.real),
        reciprocal=jnp.asarray(reciprocal, dtype=precision.real),
        ratio=float(box_ndim) / float(np.prod(shape)), dropped=float(dropped),
        radius=tuple(radii))


def landau_problem(cell, p: int, q: int, grid, kpoints, *, ecutwfc: float = 50.0,
                   preconditioner: str = "two-gauge", precision=None) -> MagneticGridHamiltonian:
    """``pi^2`` alone on the magnetic box (all potentials off): the free Landau problem."""
    return MagneticGridHamiltonian.build(cell, p, q, ecutwfc, kpoints, None, grid=grid,
                                         preconditioner=preconditioner, precision=precision)


def estimate_bytes(q: int, nbnd: int, david: int = 2, *, grid=(12, 12, 64),
                   band_batch: int = 1, kept: int = 0, fine_grid=None,
                   itemsize: int = 16) -> int:
    """The working set of one k-point's solve, bytes: the design's sizing.

    ``(2.18 nvecx + 4.2 nbnd) ndim`` complex numbers for Davidson (``nvecx = david nbnd``),
    six box-sized FFT buffers per band in flight, three of the potential's grid per band in
    flight for the two-grid local term (``fine_grid``, the chemical cell's dense grid), and
    ``kept`` stored states (``nk`` times the bands a Chern number needs) of ``ndim`` each.
    """
    ndim = int(q) * int(np.prod(grid))
    nvecx = int(david) * int(nbnd)
    davidson = (2.18 * nvecx + 4.2 * int(nbnd)) * ndim * itemsize
    batch = max(1, int(band_batch))
    fft = 6 * ndim * itemsize * batch
    if fine_grid is not None:
        fft += 3 * int(q) * int(np.prod(fine_grid)) * itemsize * batch
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
    values, _, steps, unsettled = sphere_levels(calculation, potential, nbnd, local_only=True,
                                                ethr=ethr, david=david,
                                                max_iterations=max_iterations)
    return values, steps, unsettled


def sphere_levels(calculation, potential, nbnd: int, *, local_only: bool = False,
                  ethr: float = 1e-12, david: int = 4, max_iterations: int = 200):
    """The ``a = 0`` reference on the sphere: ``-nabla^2 + V`` with or without ``V_NL``.

    As :func:`sphere_local_levels`, with the dataset's own ``D_ij`` (``vkb``, the plane-wave
    nonlocal term) unless ``local_only``. Returns ``(eigenvalues (nk, nbnd), vectors (nk,
    nbnd, npwx), steps (nk,), unsettled (nk,))``.
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
    deeq = jnp.zeros((1,) + tuple(dij.shape), dtype=dij.dtype) if local_only else dij[None]
    terms = LocalTerms(potentials=(v,), waves=(jnp.moveaxis(v, -1, -3),), deeq=deeq)
    hamiltonian = calculation.hamiltonian_from(terms)[0]
    values, vectors, steps, unsettled = davidson_eigensolver_all(
        hamiltonian, nbnd, ethr=ethr, david=david, max_iterations=max_iterations,
        return_steps=True)
    return np.asarray(values), np.asarray(vectors), np.asarray(steps), np.asarray(unsettled)


def sphere_state_on_box(calculation, vectors, ik: int, box) -> np.ndarray:
    """Sphere coefficients ``(nbnd, npwx)`` at k-point ``ik`` as box vectors ``(nbnd, ndim)``.

    ``u_j = sum_G c_G exp(i G . r_j) / sqrt(N_box)``: the same function on the box, so
    ``sum_j |u_j|^2 = sum_G |c_G|^2``. The box must hold the sphere (it does at ``q = 1``:
    the wavefunction box is built to); padded plane waves are masked out.
    """
    planewaves = calculation.basis.planewaves
    miller = np.asarray(calculation.basis.smooth.miller)[np.asarray(planewaves.indices[ik])]
    mask = np.asarray(planewaves.mask[ik])
    box = tuple(int(n) for n in box)
    coefficients = np.zeros((vectors.shape[0],) + box, dtype=np.asarray(vectors).dtype)
    slots = tuple(np.mod(miller[mask], box).T)
    for n in range(vectors.shape[0]):
        coefficients[n][slots] = np.asarray(vectors)[n][mask]
    grid = np.fft.ifftn(coefficients, axes=(1, 2, 3)) * np.sqrt(float(np.prod(box)))
    return grid.reshape(vectors.shape[0], -1)
