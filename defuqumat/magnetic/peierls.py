"""Peierls substitution on a set of hoppings: a magnetic torus and a ribbon from one builder.

The weak-field route of the orbital-field work (``notes/ORBITAL-FIELD-NEXT.md`` section 3
as corrected by its section 12). A tight-binding model ``H_mn(R)`` -- P162's maximally
localized Wannier functions, or a hand-made model -- is put in a uniform field by giving
every hop the line integral of the vector potential along the straight bond, in the
Landau gauge and with the sign of :mod:`defuqumat.magnetic.conventions`, and the result is
Bloch-summed over the magnetic lattice ``q a1 x a2 x a3`` with the magnetic Bloch-sum
factor of the review's pair (i):

    H_{mu nu}(k) = sum_P t_{mu nu} exp(-i a(mid).(r_mu - r_nu)) exp(i k.P) exp(-i a(P).c_nu),

``r_mu = j a1 + c_m`` the bra inside the magnetic cell, ``c_nu`` the ket inside it and
``r_nu = c_nu + P`` the ket's actual position. The factor's index is the column; Wang et
al. (arXiv:2609.01053, eq. 9) print the row, which is a typo (1.1 eV non-Hermitian at
``q = 10``), and dropping the factor leaves a matrix that is 4.7 eV non-Hermitian at
``q = 400`` for the nearest-neighbour honeycomb model (the review), which is why
Hermiticity at a ``k`` with both wraps active is the guard (:func:`hermiticity`). Lu et
al. (PRL 135, 196401 (2025), arXiv:2505.06585) is the exact reference this route is the
weak-field limit of; the magnetic translations follow Wang and Vafek (PRB 108, 245109
(2023), arXiv:2303.16347). Lado and Fernandez-Rossier (arXiv:1606.03038) is the ribbon
form, here the same bond list with the ``a1`` wraps dropped.

Each Wigner-Seitz image ``R + T`` of a Wannier hopping (Wannier90's ``use_ws_distance``,
:func:`defuqumat.wannier.interpolation.ws_distance_shifts`) is a bond of its own, with
weight ``H_mn(R) / (ndegen count)`` and its own phase (review item 9): the images are at
different distances, so they carry different fluxes.

**What is dropped.** The intra-cell orbital terms (the field-dressed onsite and hopping
blocks of a gauge-including basis) are the named missing term of this route; for
graphene's ``p_z`` they are ``mu_B B``, 0.58 meV at 10 T, 0.6 per cent of ``E_1`` (the
review). No Zeeman term and no spin: a level holds one electron per spin and the factor
two stays outside.

**Two paths, one set of phases.** :meth:`PeierlsTorus.matrix` is host NumPy and SciPy, a
CSR matrix, the reference and what the shift-invert solver factorises;
:meth:`PeierlsTorus.operator` is an :class:`PeierlsOperator` (``equinox.Module``) with the
same matrix stored as blocks ``(ncell, nshift, nw, nw)`` banded in the chemical-cell index
along ``a1``, applied by ``jnp.roll``, for :func:`defuqumat.kpm.moments.recursion` and a
later derivative. Both read the phases from one host function, so they agree to round-off.

Memory: the CSR matrix is ``q nb`` nonzeros (``nb`` bonds per chemical cell after the
cutoff), 16 B each plus indices; the blocks are ``q nshift nw^2`` complex. At 10 T on
graphene (``q = 7900``) the nearest-neighbour model is 47 400 nonzeros, and an 8-function
Wannier set with ``nb`` of a few thousand is of order 1e7, a few hundred MB.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from defuqumat.magnetic.conventions import MagneticCell, signed_area
from defuqumat.units import ANGSTROM_TO_BOHR, RY_TO_EV, TPI

__all__ = [
    "Bonds", "bonds_from_wannier", "honeycomb_nn_bonds", "PeierlsTorus", "PeierlsOperator",
    "levels_near", "cluster_levels", "block_weight", "hermiticity", "hermiticity_guard",
    "plaquette_check", "DENSE_LIMIT",
]

#: Below this dimension :func:`levels_near` diagonalises densely.
DENSE_LIMIT = 1600
#: The bonds of one chemical cell times the cells handled per chunk of the host build.
_CHUNK = 2_000_000


@dataclass(frozen=True)
class Bonds:
    """A flat list of hops ``<w_bra, 0| H |w_ket, R>``, Ry, with the orbital centres.

    ``bra`` and ``ket`` ``(nb,)`` orbital indices, ``cells`` ``(nb, 3)`` the integer lattice
    vector of the ket's cell (a Wigner-Seitz image already folded in), ``values`` ``(nb,)``
    complex Ry, ``centres`` ``(nw, 3)`` cartesian bohr, ``lattice`` ``(3, 3)`` bohr (rows).
    ``dropped`` is the largest ``|value|`` the magnitude cutoff left out (Ry) and
    ``dropped_norm`` the Frobenius norm of all of them per chemical cell.
    """

    bra: np.ndarray
    ket: np.ndarray
    cells: np.ndarray
    values: np.ndarray
    centres: np.ndarray
    lattice: np.ndarray
    dropped: float = 0.0
    dropped_norm: float = 0.0

    @property
    def nw(self) -> int:
        return int(len(self.centres))

    @property
    def nb(self) -> int:
        return int(len(self.values))

    def bra_positions(self) -> np.ndarray:
        return self.centres[self.bra]

    def ket_positions(self) -> np.ndarray:
        return self.centres[self.ket] + self.cells @ self.lattice

    def zero_field(self, kappa) -> np.ndarray:
        """``(nw, nw)`` ``H(k) = sum_b t_b exp(2 pi i kappa . R_b)`` of the chemical cell, crystal ``kappa``."""
        kappa = np.asarray(kappa, dtype=float)
        h = np.zeros((self.nw, self.nw), complex)
        np.add.at(h, (self.bra, self.ket), self.values * np.exp(TPI * 1j * (self.cells @ kappa)))
        return h

    @classmethod
    def from_blocks(cls, rvectors, blocks, centres, lattice, *, degeneracy=None,
                    ws_shifts=None, cutoff: float = 0.0) -> "Bonds":
        """From ``H_mn(R)`` blocks ``(nR, nw, nw)`` on integer ``rvectors`` ``(nR, 3)``.

        ``degeneracy`` divides each ``R`` (Wannier90's ``ndegen``); ``ws_shifts`` is the
        ``(shifted (nR, nw, nw, 8, 3), count (nR, nw, nw))`` pair, each image its own bond
        at ``1/(ndegen count)``. Hops with ``|value| <= cutoff`` (Ry) are dropped and the
        largest and the norm of what was dropped are kept.
        """
        rvectors = np.asarray(rvectors, dtype=np.int64)
        blocks = np.asarray(blocks)
        nr, nw, _ = blocks.shape
        degeneracy = np.ones(nr) if degeneracy is None else np.asarray(degeneracy, dtype=float)
        bra, ket, cells, values = [], [], [], []
        m_idx, n_idx = np.meshgrid(np.arange(nw), np.arange(nw), indexing="ij")
        for ir in range(nr):
            weight = blocks[ir] / degeneracy[ir]
            if ws_shifts is None:
                bra.append(m_idx.ravel())
                ket.append(n_idx.ravel())
                cells.append(np.broadcast_to(rvectors[ir], (nw * nw, 3)))
                values.append(weight.ravel())
                continue
            shifted, count = ws_shifts
            for m in range(nw):
                for n in range(nw):
                    c = int(count[ir, m, n])
                    for t in range(c):
                        bra.append([m])
                        ket.append([n])
                        cells.append(np.asarray(shifted[ir, m, n, t])[None, :])
                        values.append([weight[m, n] / c])
        bra = np.concatenate(bra).astype(np.int64)
        ket = np.concatenate(ket).astype(np.int64)
        cells = np.concatenate(cells).astype(np.int64)
        values = np.concatenate(values).astype(complex)
        keep = np.abs(values) > cutoff
        dropped = float(np.max(np.abs(values[~keep]), initial=0.0))
        dropped_norm = float(np.sqrt(np.sum(np.abs(values[~keep]) ** 2)))
        return cls(bra=bra[keep], ket=ket[keep], cells=cells[keep], values=values[keep],
                   centres=np.asarray(centres, dtype=float), lattice=np.asarray(lattice, dtype=float),
                   dropped=dropped, dropped_norm=dropped_norm)


def bonds_from_wannier(w, cutoff: float = 0.0) -> Bonds:
    """The bond list of a :class:`~defuqumat.workflows.wannier.WannierFunctions`, Ry and bohr.

    Every ``(R, m, n)`` and every one of its ``count`` Wigner-Seitz images ``R + T`` is a
    bond of weight ``H_mn(R) / (ndegen count)`` from the ket centre ``c_n + R + T`` to the
    bra centre ``c_m``; with ``w.ws_shifts`` absent, one bond per ``R`` at ``1/ndegen``.
    """
    if getattr(w, "spin", None) is not None:
        raise NotImplementedError("Landau levels of one spin channel of an nspin = 2 run "
                                  "are refused: no orbital-field route carries spin")
    return Bonds.from_blocks(w.rvectors, w.hoppings, w.centres, w.lattice,
                             degeneracy=w.degeneracy, ws_shifts=w.ws_shifts, cutoff=cutoff)


def honeycomb_nn_bonds(a: float = 2.46, t: float = -2.7, c: float = 15.0) -> Bonds:
    """The nearest-neighbour honeycomb model of the review, ``a`` and ``c`` in angstrom, ``t`` in eV.

    ``a1 = (a, 0, 0)``, ``a2 = (a/2, a sqrt(3)/2, 0)``, ``a3 = (0, 0, c)``; A at the origin
    and B at ``(a1 + a2)/3``; ``H_AB(0) = H_AB(-a1) = H_AB(-a2) = t`` and their conjugates
    (``plans/orbital-field-review/peierls_torus.py``). Values are stored in Ry.
    """
    lattice = np.array([[a, 0.0, 0.0], [a / 2, a * np.sqrt(3) / 2, 0.0], [0.0, 0.0, c]]) * ANGSTROM_TO_BOHR
    centres = np.array([np.zeros(3), (lattice[0] + lattice[1]) / 3])
    tr = t / RY_TO_EV
    bra, ket, cells = [], [], []
    for R in ((0, 0, 0), (-1, 0, 0), (0, -1, 0)):
        bra += [0, 1]
        ket += [1, 0]
        cells += [R, tuple(-x for x in R)]
    return Bonds(bra=np.array(bra), ket=np.array(ket), cells=np.array(cells, dtype=np.int64),
                 values=np.full(len(bra), tr, dtype=complex), centres=centres, lattice=lattice)


def _hop_data(cell: MagneticCell, bonds: Bonds, j: np.ndarray, kappa, ncell: int, torus: bool,
              hop_phase=None, bloch_factor: bool = True):
    """Rows, columns, values and ``R1`` of every bond leaving chemical cells ``j`` (host).

    ``ncell`` is ``q`` on the torus and the width on the ribbon. On the ribbon a bond whose
    ket cell falls outside ``[0, width)`` is dropped and the rest carry no wrap.
    """
    lattice = bonds.lattice
    nw = bonds.nw
    R = bonds.cells
    s = j[:, None] + R[None, :, 0]                          # (nj, nb) ket cell index along a1
    if torus:
        jp = np.mod(s, ncell)
        m1 = np.floor_divide(s, ncell)
        live = np.ones_like(s, dtype=bool)
    else:
        jp = s
        m1 = np.zeros_like(s)
        live = (s >= 0) & (s < ncell)
    a1 = lattice[0]
    r_mu = j[:, None, None] * a1 + bonds.centres[bonds.bra][None, :, :]
    c_nu = jp[..., None] * a1 + bonds.centres[bonds.ket][None, :, :]
    P = (m1[..., None] * (ncell * a1 if torus else 0.0)
         + R[None, :, 1, None] * lattice[1] + R[None, :, 2, None] * lattice[2])
    r_nu = c_nu + P
    hop = cell.peierls_phase if hop_phase is None else hop_phase
    phase = np.asarray(hop(r_nu, r_mu))
    kappa = np.asarray(kappa, dtype=float)
    kappa = np.concatenate([kappa, np.zeros(3 - len(kappa))]) if len(kappa) < 3 else kappa
    integer = np.stack([m1, np.broadcast_to(R[None, :, 1], m1.shape),
                        np.broadcast_to(R[None, :, 2], m1.shape)], axis=-1)
    if torus:
        phase = phase + TPI * (integer @ kappa)
        if bloch_factor:
            phase = phase + np.asarray(cell.bloch_sum_phase(P, c_nu))
    else:
        phase = phase + TPI * (integer[..., 1:] @ kappa[1:])
    rows = j[:, None] * nw + bonds.bra[None, :]
    cols = jp * nw + bonds.ket[None, :]
    values = bonds.values[None, :] * np.exp(1j * phase)
    return rows[live], cols[live], values[live], np.broadcast_to(R[None, :, 0], s.shape)[live], \
        np.broadcast_to(j[:, None], s.shape)[live]


class PeierlsOperator(eqx.Module):
    """``H v`` of a Peierls Hamiltonian at one ``k``, banded in the cell index along ``a1``.

    ``blocks[j, s]`` is the ``(nw, nw)`` block from cell ``j + shifts[s]`` (mod ``ncell``)
    to cell ``j``; on a ribbon the blocks that would wrap are zero, so the roll is still
    exact. ``apply`` takes ``(..., ncell nw)`` and returns the same shape.
    """

    blocks: jnp.ndarray
    shifts: tuple = eqx.field(static=True)
    ncell: int = eqx.field(static=True)
    nw: int = eqx.field(static=True)

    @property
    def dim(self) -> int:
        return self.ncell * self.nw

    def apply(self, v):
        return _apply(self.blocks, self.shifts, self.ncell, self.nw, v)

    def __call__(self, v):
        return self.apply(v)


def _apply_body(blocks, shifts, ncell, nw, v):
    lead = v.shape[:-1]
    u = v.reshape(lead + (ncell, nw))
    out = jnp.zeros_like(u)
    for index, shift in enumerate(shifts):
        rolled = jnp.roll(u, -shift, axis=-2)               # rolled[j] = u[j + shift]
        out = out + jnp.einsum("jmn,...jn->...jm", blocks[:, index], rolled)
    return out.reshape(lead + (ncell * nw,))


_apply = jax.jit(_apply_body, static_argnums=(1, 2, 3))


class PeierlsTorus:
    """A Peierls model on the magnetic torus ``q a1 x a2 x a3`` (or a ribbon ``width`` cells wide).

    ``cell`` carries ``p/q`` and the gauge; ``bonds`` the zero-field hoppings. ``kappa``
    is ``(kappa1, kappa2[, kappa3])`` in the magnetic cell's reciprocal basis; on a
    ribbon only ``kappa2`` (and ``kappa3``) mean anything.
    """

    def __init__(self, cell: MagneticCell, bonds: Bonds, *, width: int | None = None,
                 hop_phase=None, bloch_factor: bool = True):
        self.cell = cell
        self.bonds = bonds
        self.torus = width is None
        self.ncell = cell.q if width is None else int(width)
        self.hop_phase = hop_phase
        self.bloch_factor = bloch_factor
        if not np.allclose(np.asarray(cell.lattice), bonds.lattice, atol=1e-10):
            raise ValueError("the bonds were built on another lattice than the magnetic cell's")

    @property
    def nw(self) -> int:
        return self.bonds.nw

    @property
    def dim(self) -> int:
        return self.ncell * self.nw

    def ribbon(self, width: int) -> "PeierlsTorus":
        """The same bonds and flux on a ribbon of ``width`` chemical cells along ``a1``, open ends."""
        return PeierlsTorus(self.cell, self.bonds, width=width, hop_phase=self.hop_phase)

    def _chunks(self):
        step = max(1, _CHUNK // max(1, self.bonds.nb))
        for start in range(0, self.ncell, step):
            yield np.arange(start, min(self.ncell, start + step))

    def matrix(self, kappa=(0.0, 0.0), *, check: bool = True, tol: float = 1e-8) -> sp.csr_matrix:
        """The ``(ncell nw, ncell nw)`` Hamiltonian at ``kappa`` as CSR, Ry.

        With ``check`` the Hermiticity guard runs: ``max |H - H^dagger|`` above ``tol``
        times ``max |H|`` raises, which is what a wrong Bloch-sum factor gives.
        """
        rows, cols, vals = [], [], []
        for j in self._chunks():
            r, c, v, _, _ = _hop_data(self.cell, self.bonds, j, kappa, self.ncell, self.torus,
                                      self.hop_phase, self.bloch_factor)
            rows.append(r)
            cols.append(c)
            vals.append(v)
        h = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(self.dim, self.dim))
        h.sum_duplicates()
        if check:
            hermiticity_guard(h, tol)
        return h

    def operator(self, kappa=(0.0, 0.0), dtype=None) -> PeierlsOperator:
        """The same Hamiltonian as :class:`PeierlsOperator` blocks; ``dtype`` from the policy."""
        from defuqumat.config import DEFAULT_PRECISION

        dtype = DEFAULT_PRECISION.complex if dtype is None else dtype
        shifts = tuple(sorted(set(int(x) for x in self.bonds.cells[:, 0])))
        table = np.array(shifts, dtype=np.int64)
        blocks = np.zeros((self.ncell, len(shifts), self.nw, self.nw), complex)
        for j in self._chunks():
            r, c, v, r1, jj = _hop_data(self.cell, self.bonds, j, kappa, self.ncell, self.torus,
                                        self.hop_phase, self.bloch_factor)
            index = np.searchsorted(table, r1)
            np.add.at(blocks, (jj, index, r % self.nw, c % self.nw), v)
        return PeierlsOperator(blocks=jnp.asarray(blocks, dtype=dtype), shifts=shifts,
                               ncell=self.ncell, nw=self.nw)


def hermiticity(h) -> float:
    """``max |H - H^dagger|`` of a dense or sparse matrix."""
    if sp.issparse(h):
        d = (h - h.getH()).tocoo()
        return float(np.max(np.abs(d.data), initial=0.0))
    h = np.asarray(h)
    return float(np.max(np.abs(h - h.conj().T)))


def hermiticity_guard(h, tol: float = 1e-8) -> float:
    """Raise when ``max |H - H^dagger| > tol max |H|``; return the measured residual."""
    residual = hermiticity(h)
    scale = float(np.max(np.abs(h.data))) if sp.issparse(h) else float(np.max(np.abs(h)))
    if residual > tol * max(scale, 1e-300):
        raise ValueError(
            f"the Peierls Hamiltonian is not Hermitian: max |H - H^dagger| = "
            f"{residual * RY_TO_EV:.3e} eV against max |H| = {scale * RY_TO_EV:.3e} eV. A wrong or "
            "missing magnetic Bloch-sum factor exp(-i a(P).c_nu) does this")
    return residual


def levels_near(h, energy: float, n: int, *, vectors: bool = True, offset: float | None = None):
    """The ``n`` eigenvalues of ``h`` nearest ``energy`` (Ry), sorted, and their vectors.

    Dense ``eigh`` below :data:`DENSE_LIMIT`, otherwise SciPy's shift-invert ``eigsh``
    about ``energy + offset``: the offset (default 1e-7 Ry) keeps the factorisation
    regular when ``energy`` sits on a level, as zero modes do, and a few extra values are
    asked for so that the cut is made about ``energy`` rather than about the shift.
    """
    dim = h.shape[0]
    if dim <= DENSE_LIMIT:
        dense = h.toarray() if sp.issparse(h) else np.asarray(h)
        w, v = np.linalg.eigh(dense)
    else:
        import scipy.sparse.linalg as spla

        shift = energy + (1e-7 if offset is None else offset)
        k = min(dim - 2, n + 4)
        w, v = spla.eigsh(sp.csc_matrix(h), k=k, sigma=shift, which="LM")
    sel = np.argsort(np.abs(w - energy), kind="stable")[:n]
    sel = sel[np.argsort(w[sel], kind="stable")]
    return (w[sel], v[:, sel]) if vectors else w[sel]


def cluster_levels(values, tol: float):
    """``(levels, multiplicities)``: sorted values grouped where neighbours are within ``tol``."""
    values = np.sort(np.asarray(values, dtype=float))
    levels, counts = [], []
    for x in values:
        if counts and abs(x - levels[-1][-1]) <= tol:
            levels[-1].append(x)
            counts[-1] += 1
        else:
            levels.append([x])
            counts.append(1)
    return np.array([np.mean(group) for group in levels]), np.array(counts)


def block_weight(vectors, mask) -> float:
    """``tr P_mask`` over the span of ``vectors`` ``(dim, nblock)``: rule D4's invariant.

    The weight of one vector of a degenerate block on a set of orbitals is not
    invariant under the solver's rotation inside the block; the trace over the whole
    block is.
    """
    vectors = np.asarray(vectors)
    mask = np.asarray(mask, dtype=float)
    return float(np.einsum("ia,i,ia->", vectors.conj(), mask, vectors).real)


def plaquette_check(cell: MagneticCell, bonds: Bonds, loop, *, hop_phase=None,
                    tol: float = 1e-10) -> complex:
    """The product of the hop factors around ``loop``, checked against ``exp(-i 2 pi n_phi S)``.

    ``loop`` is a sequence of ``(orbital, R)`` vertices (``R`` an integer lattice vector),
    traversed in order and closed; every consecutive pair must be a bond of ``bonds``.
    ``S`` is the loop's signed area in chemical cells, counterclockwise seen from ``+a3``
    positive. Raises when the product is off by more than ``tol``; returns it.
    ``hop_phase`` replaces :meth:`MagneticCell.peierls_phase` (the test flips its sign to
    see the guard fire); the builder of :class:`PeierlsTorus` takes the same argument.
    """
    hop = cell.peierls_phase if hop_phase is None else hop_phase
    lattice = bonds.lattice
    vertices = [(int(o), np.asarray(R, dtype=np.int64)) for o, R in loop]
    positions = np.array([bonds.centres[o] + R @ lattice for o, R in vertices])
    known = {(int(b), int(k), tuple(int(x) for x in c))
             for b, k, c in zip(bonds.bra, bonds.ket, bonds.cells)}
    total = 0.0
    for i, (o_from, R_from) in enumerate(vertices):
        o_to, R_to = vertices[(i + 1) % len(vertices)]
        key = (o_to, o_from, tuple(int(x) for x in (R_from - R_to)))
        if key not in known:
            raise ValueError(f"the loop's hop {o_from}{tuple(R_from)} -> {o_to}{tuple(R_to)} is not a bond")
        total += float(np.asarray(hop(positions[i], positions[(i + 1) % len(vertices)])))
    product = complex(np.exp(1j * total))
    area = signed_area(positions, lattice)
    expected = complex(np.exp(-1j * TPI * cell.flux * float(cell.scale) * area))
    if abs(product - expected) > tol:
        raise ValueError(
            f"the plaquette product {product:.12f} around a loop of {area:+.6f} cells is not "
            f"exp(-i 2 pi n_phi S) = {expected:.12f}: the sign of the Peierls phase is not the "
            "electron's of magnetic/conventions.py")
    return product
