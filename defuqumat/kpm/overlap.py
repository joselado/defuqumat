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

**Cost and memory.** ``U`` is stored for every k-point, ``nk npwx nkb`` complex
numbers, one stored projector table, even when the calculation keeps a lazy set
(rebuilding it inside the recursion would cost a QR per application). Each
application of ``H^`` is ``H``'s own plus two products with ``U`` each way,
about three times the nonlocal term.
"""

from __future__ import annotations

import functools

import equinox as eqx
import jax
import jax.numpy as jnp

from defuqumat.batching import map_k


def overlap_power(table, qq, exponent: float):
    """``(U, M^p - 1)`` with ``S^p = 1 + U (M^p - 1) U^dagger`` for one k-point's ``B = table``."""
    basis, upper = jnp.linalg.qr(table, mode="reduced")
    inside = jnp.eye(upper.shape[0], dtype=upper.dtype) + upper @ qq.astype(upper.dtype) @ jnp.conj(upper).T
    inside = 0.5 * (inside + jnp.conj(inside).T)
    values, vectors = jnp.linalg.eigh(inside)
    shifted = (vectors * (values ** exponent - 1.0)[None, :]) @ jnp.conj(vectors).T
    return basis, shifted


@functools.partial(jax.jit, static_argnames=("exponent", "k_batch"))
def overlap_powers(hamiltonian, exponent: float = -0.5, k_batch=None):
    """``(U, M^p - 1)`` at every k-point: ``(nk, npwx, nkb)`` and ``(nk, nkb, nkb)``."""
    projectors = hamiltonian.projectors
    qq = projectors.qq

    def one_k(ik):
        table = jnp.where(hamiltonian.mask[ik][:, None], projectors.at_k(ik), 0.0)
        return overlap_power(table, qq, exponent)

    return map_k(one_k, jnp.arange(hamiltonian.mask.shape[0]), batch=k_batch)


def apply_power(basis, shifted, psi, mask):
    """``S^p psi = psi + U (M^p - 1) U^dagger psi`` for ``psi`` of shape ``(..., npwx)``."""
    coefficients = jnp.einsum("gk,...g->...k", jnp.conj(basis), psi)
    out = psi + jnp.einsum("gk,...k->...g", basis, coefficients @ shifted.T)
    return jnp.where(mask, out, 0.0)


class SymmetrizedHamiltonian(eqx.Module):
    """``S^(-1/2) H S^(-1/2)`` of a collinear augmented Hamiltonian, as a Hamiltonian.

    It offers what the KPM engine reads from a Hamiltonian: :meth:`apply`,
    ``mask``, ``dtype``, ``has_overlap`` (``False``: the generalised problem is
    folded in) and ``gamma_only``.
    """

    hamiltonian: object
    basis: jnp.ndarray  # (nk, npwx, nkb): U of B = U R
    shifted: jnp.ndarray  # (nk, nkb, nkb): M^(-1/2) - 1

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

    def half_inverse(self, psi, ik):
        """``S^(-1/2) psi`` at k-point ``ik``, for ``psi`` of shape ``(..., npwx)``."""
        return apply_power(self.basis[ik], self.shifted[ik], psi, self.mask[ik])

    def apply(self, psi, ik):
        """``S^(-1/2) H S^(-1/2) psi``."""
        return self.half_inverse(self.hamiltonian.apply(self.half_inverse(psi, ik), ik), ik)


def raised_vectors(wrapped, vectors, k_batch=None):
    """``(nk, r, ndim)``: ``S^(1/2) v`` at every k-point, for start vectors of ``H^``.

    A projection ``<phi|S|x>`` of a generalised eigenvector ``x`` is
    ``<S^(1/2) phi|S^(1/2) x>``, the plain inner product with the eigenvector of
    ``H^``, so the projected density of states of ``(H, S)`` is the one of ``H^`` on
    ``S^(1/2) phi``.
    """
    if not isinstance(wrapped, SymmetrizedHamiltonian):
        return vectors
    basis, shifted = overlap_powers(wrapped.hamiltonian, 0.5, k_batch=k_batch)
    return jax.vmap(apply_power)(basis, shifted, jnp.asarray(vectors).astype(wrapped.dtype),
                                 wrapped.mask[:, None, :])


def symmetrized(hamiltonians, k_batch=None) -> tuple:
    """Each augmented collinear Hamiltonian wrapped as :class:`SymmetrizedHamiltonian`."""
    out = []
    for hamiltonian in hamiltonians:
        if not hamiltonian.has_overlap:
            out.append(hamiltonian)
            continue
        if hasattr(hamiltonian, "_as_state"):
            raise NotImplementedError(
                "the KPM trace of a spinor with an ultrasoft or PAW dataset is not "
                "implemented: qq_so couples the two spin blocks of S, and the symmetrised "
                "form S^(-1/2) H S^(-1/2) has only been built and checked for a collinear run")
        out.append(SymmetrizedHamiltonian(hamiltonian, *overlap_powers(hamiltonian,
                                                                      k_batch=k_batch)))
    return tuple(out)
