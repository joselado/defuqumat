"""The symmetrised Hamiltonian ``S^(-1/2) H S^(-1/2)`` of an ultrasoft or PAW run.

With an augmented dataset the states solve ``H x = e S x``, with
``S = 1 + B Q B^dagger`` at each k-point: ``B`` the ``(npw, nkb)`` projector
table and ``Q`` the ``qq`` matrix (``PW/src/s_psi.f90``). The kernel polynomial
method wants a Hermitian operator in the plain inner product whose spectrum is
the generalised one, and

    H^ = S^(-1/2) H S^(-1/2)

is that: it is similar to ``S^-1 H``, so it has the same eigenvalues, and its
eigenvectors are ``S^(1/2) x``. Then the doubling trick, the Lanczos interval,
the guard ``|mu_n| <= mu_0`` and the random-phase trace of
:mod:`defuqumat.kpm.moments` hold as they are.

**``S^(-1/2)`` is the identity plus a term in the span of ``B``.** With a thin
QR factorisation ``B = U R`` (``U`` orthonormal columns), ``S`` acts on the span
of ``U`` as ``M = 1 + R Q R^dagger``, Hermitian and positive definite because
``S`` is (``M = U^dagger S U``), and as the identity off it. So

    S^p = 1 + U [M^p - 1] U^dagger

for any power ``p``, with every small matrix ``nkb x nkb``. The QR form is the
well-conditioned one: the form through ``G = B^dagger B``,
``G^(-1/2) [(1 + G^(1/2) Q G^(1/2))^(-1/2) - 1] G^(-1/2)``, is the same matrix but
loses ``cond(G)`` in precision, and dropping the small eigenvalues of ``G`` is
not exact (a fable review measured ``S^(-1/2) S^(-1/2) S v - v`` at 1e-7 for
``cond(G) = 4e10`` with a cut at 1e-10, against 1.6e-15 for the QR form at
``cond(G) = 4e14``). A rank-deficient ``B`` is no problem: ``U`` is orthonormal
whatever ``R`` is.

**The fold (P174).** Write ``H = A + B D B^dagger``, with ``A = K + V_loc``
Hermitian and ``D`` the channel's ``deeq``, and ``W = M^(-1/2) - 1``. Since
``S^(-1/2) U = U M^(-1/2)``, the nonlocal term of ``H^`` is
``U D' U^dagger`` with ``D' = M^(-1/2) R D R^dagger M^(-1/2)``, and the local
term needs ``U^dagger A psi``, which is ``(A U)^dagger psi``. With
``c = U^dagger psi`` and ``a = (A U)^dagger psi``, exactly

    H^ psi = A psi + (A U)(W c) + U [F c + W a],    F = D' + W G W,  G = U^dagger A U.

That is four products of size ``npwx x nkb`` and one FFT pair per vector,
against six products and one FFT pair for the unfolded
``S^(-1/2) H S^(-1/2)``, whose two ``U^dagger psi`` products also take a
conjugated copy of ``U`` each. It is exact for ``H = A + B D B^dagger``, so it
is built only when the Hamiltonian has no other term (no DFT+U, no
``lelfield``) and stores the whole sphere (no ``gamma_only``); the unfolded
form, :meth:`SymmetrizedHamiltonian.apply_unfolded`, takes every other case.

**Cost and memory.** ``U`` is stored for every k-point, ``nk npwx nkb`` complex
numbers, one stored projector table, even when the calculation keeps a lazy set
(rebuilding it inside the recursion would cost a QR per application). It
depends on the projectors and ``qq`` alone, so the two channels of an
``nspin = 2`` run share one ``U`` and one ``M^(-1/2) - 1``. **The fold is a
deviation in memory**: one more ``(nk, npwx, nkb)`` table per channel, ``A U``,
and ``F`` (``nk nkb^2``), so ``(1 + nspin) nk npwx nkb`` complex numbers in all;
``fold=False`` gives the unfolded form and its single table back. Building
``A U`` costs ``nkb`` FFT pairs per k-point and channel, about one random
vector's worth of the recursion's transforms when ``N`` is near ``2 nkb``.
"""

from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp

from defuqumat.batching import map_k
from defuqumat.hamiltonian.operator import conjugated_contraction

#: Whether :func:`symmetrized` folds ``S^(-1/2)`` into the projector products
#: when the Hamiltonian allows it (``fold=None``).
FOLD_DEFAULT = True


def _overlap_factors(table, qq, exponent: float):
    """``(U, R, M^p - 1)`` of ``B = table = U R``, ``M = 1 + R Q R^dagger``."""
    basis, upper = jnp.linalg.qr(table, mode="reduced")
    inside = jnp.eye(upper.shape[0], dtype=upper.dtype) + upper @ qq.astype(upper.dtype) @ jnp.conj(upper).T
    inside = 0.5 * (inside + jnp.conj(inside).T)
    values, vectors = jnp.linalg.eigh(inside)
    shifted = (vectors * (values ** exponent - 1.0)[None, :]) @ jnp.conj(vectors).T
    return basis, upper, shifted


def overlap_power(table, qq, exponent: float):
    """``(U, M^p - 1)`` with ``S^p = 1 + U (M^p - 1) U^dagger`` for one k-point's ``B = table``."""
    basis, _, shifted = _overlap_factors(table, qq, exponent)
    return basis, shifted


def _masked_table(hamiltonian, ik):
    return jnp.where(hamiltonian.mask[ik][:, None], hamiltonian.projectors.at_k(ik), 0.0)


@functools.partial(jax.jit, static_argnames=("exponent", "k_batch"))
def overlap_powers(hamiltonian, exponent: float = -0.5, k_batch=None):
    """``(U, M^p - 1)`` at every k-point: ``(nk, npwx, nkb)`` and ``(nk, nkb, nkb)``."""
    qq = hamiltonian.projectors.qq

    def one_k(ik):
        return overlap_power(_masked_table(hamiltonian, ik), qq, exponent)

    return map_k(one_k, jnp.arange(hamiltonian.mask.shape[0]), batch=k_batch)


@functools.partial(jax.jit, static_argnames=("k_batch",))
def _overlap_setup(hamiltonian, k_batch=None):
    """``(U, R, M^(-1/2) - 1)`` at every k-point."""
    qq = hamiltonian.projectors.qq

    def one_k(ik):
        return _overlap_factors(_masked_table(hamiltonian, ik), qq, -0.5)

    return map_k(one_k, jnp.arange(hamiltonian.mask.shape[0]), batch=k_batch)


@functools.partial(jax.jit, static_argnames=("k_batch",))
def _fold(hamiltonian, basis, upper, shifted, k_batch=None):
    """``(A U, D' + W G W)`` at every k-point for one channel, the fold of the module docstring."""
    coefficients = hamiltonian.coefficients.astype(hamiltonian.dtype)

    def one_k(args):
        ik, u, r, w = args
        mask = hamiltonian.mask[ik]
        u = jnp.where(mask[:, None], u, 0.0)
        applied = hamiltonian.kinetic[ik][:, None] * u + hamiltonian._local(u.T, ik).T
        applied = jnp.where(mask[:, None], applied, 0.0)  # (npwx, nkb): A U
        gram = jnp.conj(u).T @ applied  # U^dagger A U
        gram = 0.5 * (gram + jnp.conj(gram).T)
        half = jnp.eye(w.shape[0], dtype=w.dtype) + w  # M^(-1/2)
        inner = r @ coefficients @ jnp.conj(r).T  # R D R^dagger
        folded = half @ inner @ half + w @ gram @ w
        return applied, 0.5 * (folded + jnp.conj(folded).T)

    nk = hamiltonian.mask.shape[0]
    return map_k(one_k, (jnp.arange(nk), basis, upper, shifted), batch=k_batch)


def apply_power(basis, shifted, psi, mask):
    """``S^p psi = psi + U (M^p - 1) U^dagger psi`` for ``psi`` of shape ``(..., npwx)``."""
    coefficients = jnp.einsum("gk,...g->...k", jnp.conj(basis), psi)
    out = psi + jnp.einsum("gk,...k->...g", basis, coefficients @ shifted.T)
    return jnp.where(mask, out, 0.0)


class SymmetrizedHamiltonian(eqx.Module):
    """``S^(-1/2) H S^(-1/2)`` of a collinear augmented Hamiltonian, as a Hamiltonian.

    It offers what the KPM engine reads from a Hamiltonian: :meth:`apply`,
    ``mask``, ``dtype``, ``has_overlap`` (``False``: the generalised problem is
    folded in) and ``gamma_only``. ``table`` and ``folded`` are the fold of the
    module docstring, or ``None`` for the unfolded form.
    """

    hamiltonian: object
    basis: jnp.ndarray  # (nk, npwx, nkb): U of B = U R
    shifted: jnp.ndarray  # (nk, nkb, nkb): M^(-1/2) - 1
    table: jnp.ndarray | None = None  # (nk, npwx, nkb): A U, this channel's
    folded: jnp.ndarray | None = None  # (nk, nkb, nkb): D' + W G W, this channel's

    @property
    def mask(self):
        return self.hamiltonian.mask

    @property
    def dtype(self):
        return self.hamiltonian.dtype

    @property
    def has_overlap(self) -> bool:
        return False

    @property
    def gamma_only(self) -> bool:
        return self.hamiltonian.gamma_only

    @property
    def kinetic(self):
        return self.hamiltonian.kinetic

    @property
    def is_folded(self) -> bool:
        return self.table is not None

    def half_inverse(self, psi, ik):
        """``S^(-1/2) psi`` at k-point ``ik``, for ``psi`` of shape ``(..., npwx)``."""
        return apply_power(self.basis[ik], self.shifted[ik], psi, self.mask[ik])

    def apply_unfolded(self, psi, ik):
        """``S^(-1/2) H S^(-1/2) psi``, ``S^(-1/2)`` on each side of ``H``'s own apply."""
        return self.half_inverse(self.hamiltonian.apply(self.half_inverse(psi, ik), ik), ik)

    def apply(self, psi, ik):
        """``S^(-1/2) H S^(-1/2) psi``, folded when :attr:`table` is set."""
        if self.table is None:
            return self.apply_unfolded(psi, ik)
        hamiltonian = self.hamiltonian
        mask = hamiltonian.mask[ik]
        psi = jnp.where(mask, psi, 0.0)
        basis, table, shifted = self.basis[ik], self.table[ik], self.shifted[ik]
        c = conjugated_contraction(basis, psi, "gk,...g->...k")  # U^dagger psi
        a = conjugated_contraction(table, psi, "gk,...g->...k")  # (A U)^dagger psi
        out = hamiltonian.kinetic[ik] * psi + hamiltonian._local(psi, ik)
        out = out + jnp.einsum("gk,...k->...g", table, c @ shifted.T)
        out = out + jnp.einsum("gk,...k->...g", basis, c @ self.folded[ik].T + a @ shifted.T)
        return jnp.where(mask, out, 0.0)


@functools.partial(jax.jit, static_argnames=("k_batch",))
def _raised(basis, shifted, vectors, mask, k_batch=None):
    """``S^(1/2) v`` at every k-point from ``U`` and ``W = M^(-1/2) - 1``: ``M^(1/2) - 1 = (1 + W)^-1 - 1``."""

    def one_k(args):
        u, w, v, m = args
        eye = jnp.eye(w.shape[0], dtype=w.dtype)
        return apply_power(u, jnp.linalg.inv(eye + w) - eye, v, m[None, :])

    return map_k(one_k, (basis, shifted, vectors, mask), batch=k_batch)


def raised_vectors(wrapped, vectors, k_batch=None):
    """``(nk, r, ndim)``: ``S^(1/2) v`` at every k-point, for start vectors of ``H^``.

    A projection ``<phi|S|x>`` of a generalised eigenvector ``x`` is
    ``<S^(1/2) phi|S^(1/2) x>``, the plain inner product with the eigenvector of
    ``H^``, so the projected density of states of ``(H, S)`` is the one of ``H^`` on
    ``S^(1/2) phi``. ``U`` and ``M^(-1/2) - 1`` are the wrapped operator's own,
    so no second QR is taken and no second ``U`` is held.
    """
    if not isinstance(wrapped, SymmetrizedHamiltonian):
        return vectors
    return _raised(wrapped.basis, wrapped.shifted, jnp.asarray(vectors).astype(wrapped.dtype),
                   wrapped.mask, k_batch=k_batch)


def _foldable(hamiltonian) -> bool:
    """Whether ``H = K + V_loc + B D B^dagger`` and nothing else, on the whole sphere."""
    return (getattr(hamiltonian, "hubbard", None) is None
            and getattr(hamiltonian, "berry", None) is None
            and not hamiltonian.gamma_only)


def symmetrized(hamiltonians, k_batch=None, fold=None) -> tuple:
    """Each augmented collinear Hamiltonian wrapped as :class:`SymmetrizedHamiltonian`.

    ``U`` and ``M^(-1/2) - 1`` are built once and shared by the channels that
    hold the same projectors and the same sphere (the two of an ``nspin = 2``
    run), since ``S`` does not depend on spin. ``fold`` (``None`` reads
    :data:`FOLD_DEFAULT`) builds the folded apply where the Hamiltonian allows it.
    """
    fold = FOLD_DEFAULT if fold is None else bool(fold)
    out = []
    shared = None  # (hamiltonian, U, R, M^(-1/2) - 1) of the first augmented channel
    for hamiltonian in hamiltonians:
        if not hamiltonian.has_overlap:
            out.append(hamiltonian)
            continue
        if hasattr(hamiltonian, "_as_state"):
            raise NotImplementedError(
                "the KPM trace of a spinor with an ultrasoft or PAW dataset is not "
                "implemented: qq_so couples the two spin blocks of S, and the symmetrised "
                "form S^(-1/2) H S^(-1/2) has only been built and checked for a collinear run")
        if (shared is not None and hamiltonian.projectors is shared[0].projectors
                and hamiltonian.mask is shared[0].mask):
            basis, upper, shifted = shared[1:]
        else:
            basis, upper, shifted = _overlap_setup(hamiltonian, k_batch=k_batch)
            if shared is None:
                shared = (hamiltonian, basis, upper, shifted)
        if fold and _foldable(hamiltonian):
            table, folded = _fold(hamiltonian, basis, upper, shifted, k_batch=k_batch)
            out.append(SymmetrizedHamiltonian(hamiltonian, basis, shifted, table, folded))
        else:
            out.append(SymmetrizedHamiltonian(hamiltonian, basis, shifted))
    return tuple(out)
