"""Chebyshev moments ``mu_n = <v|T_n(H')|v>`` of a Kohn-Sham Hamiltonian.

``H' = (H - centre) / half`` has its spectrum inside ``[-1, 1]`` when the
centre and the half width bound it (:mod:`defuqumat.kpm.bounds`). The moments
come from the three-term recursion

    a_0 = v,   a_1 = H' v,   a_{n+1} = 2 H' a_n - a_{n-1},

on a block of start vectors at one k-point, with the doubling trick
``mu_{2n} = 2 <a_n|a_n> - mu_0`` and ``mu_{2n+1} = 2 <a_{n+1}|a_n> - mu_1``,
which follows from ``T_m T_n = (T_{m+n} + T_{|m-n|}) / 2`` and gives ``2 (M + 1)``
moments from ``M + 1`` applications of ``H``. The recursion is a ``lax.scan``
over ``n`` and the k-points go through :func:`~defuqumat.batching.map_k`, so it
is compiled once, differentiable, and runs unchanged on a card.

Three kinds of start vector:

* ``"stochastic"``: random phases ``exp(2 pi i u) / sqrt(n_live)`` on the
  ``n_live`` live plane waves of the k-point (``npw_k``, or ``2 npw_k`` for a
  spinor) and zero on the padding. ``E[v_i^* v_j] = delta_ij / n_live``, so
  ``n_live <v|T_n|v>`` is an unbiased estimate of ``Tr T_n(H')`` in the basis.
* ``"exact"``: every unit vector of the live sphere, so the sum over the block
  is the trace itself. ``ndim`` recursions, for tests on small cells.
* ``"given"``: a block the caller built (atomic orbitals, a point), for the
  deterministic traces of a later phase.

Ported from pyqula, ``kpmtk/kpmnumba.py`` (``kpm_moments_v``) and
``kpmtk/kpmjax.py``; the recursion is Weisse et al., Rev. Mod. Phys. 78, 275
(2006), arXiv:cond-mat/0504627, section II.
"""

from __future__ import annotations

import functools
import math

import jax
import jax.numpy as jnp
from jax import lax

from defuqumat.batching import map_k


def live_mask(hamiltonian) -> jnp.ndarray:
    """``(nk, ndim)``: which components of a state are live plane waves."""
    if hasattr(hamiltonian, "_as_state"):  # a spinor: both components share the sphere
        return hamiltonian.state_mask
    return hamiltonian.mask


def _real_dtype(dtype):
    return jnp.zeros((), dtype).real.dtype


def _dot(a, b):
    """``Re <a|b>`` over the last axis, one number per start vector."""
    return jnp.real(jnp.sum(jnp.conj(a) * b, axis=-1))


def recursion(apply, start, n_steps: int):
    """``(R, 2 (n_steps + 1))`` moments of the block ``start`` under ``apply = H'``."""
    first = apply(start)
    mu0 = _dot(start, start)
    mu1 = _dot(first, start)

    def step(carry, _):
        previous, current = carry
        following = 2.0 * apply(current) - previous
        return (current, following), (_dot(current, current), _dot(following, current))

    _, (even, odd) = lax.scan(step, (start, first), None, length=n_steps)
    even = jnp.concatenate([mu0[None], 2.0 * even - mu0[None]], axis=0)
    odd = jnp.concatenate([mu1[None], 2.0 * odd - mu1[None]], axis=0)
    moments = jnp.stack([even, odd], axis=1)  # (n_steps + 1, 2, R)
    return moments.reshape((2 * (n_steps + 1),) + mu0.shape).T


def random_phases(key, mask, n_random: int, dtype):
    """``(n_random, ndim)`` random phases on the live components, norm one each.

    ``dtype`` is the complex dtype of the states, so the phases follow the
    precision policy rather than a literal.
    """
    real = _real_dtype(dtype)
    angle = (2.0 * math.pi) * jax.random.uniform(key, (n_random, mask.shape[-1]), dtype=real)
    count = jnp.sum(mask).astype(real)
    phases = lax.complex(jnp.cos(angle), jnp.sin(angle)) / jnp.sqrt(count)
    return jnp.where(mask[None, :], phases, 0.0).astype(dtype)


def unit_vectors(mask, dtype):
    """``(ndim, ndim)``: the unit vector of every live component, zero rows elsewhere."""
    eye = jnp.eye(mask.shape[-1], dtype=dtype)
    return jnp.where(mask[:, None] & mask[None, :], eye, 0.0)


@functools.partial(jax.jit, static_argnames=("n_steps", "n_random", "kind", "k_batch"))
def chebyshev_moments(hamiltonian, centre, half, key, n_steps: int, n_random: int,
                      kind: str = "stochastic", k_batch=None, given=None):
    """``(nk, R, 2 (n_steps + 1))`` moments of every start vector at every k-point.

    A module-level ``jit`` that takes the Hamiltonian, the bounds and the key as
    arguments and closes over none of them, so a second call on another
    potential compiles nothing. ``given`` is ``(nk, R, ndim)`` for
    ``kind = "given"`` and ignored otherwise; the stochastic key is folded with
    the k index, so the start vectors do not depend on how the k axis is
    chunked.
    """
    mask = live_mask(hamiltonian)
    dtype = hamiltonian.dtype

    def one_k(args):
        ik, start = args
        live = mask[ik]

        def apply(v):
            out = (hamiltonian.apply(v, ik) - centre * v) / half
            return jnp.where(live, out, 0.0)

        if kind == "stochastic":
            start = random_phases(jax.random.fold_in(key, ik), live, n_random, dtype)
        elif kind == "exact":
            start = unit_vectors(live, dtype)
        return recursion(apply, start, n_steps)

    indices = jnp.arange(mask.shape[0])
    if kind == "given":
        return map_k(one_k, (indices, given), batch=k_batch)
    return map_k(lambda ik: one_k((ik, None)), indices, batch=k_batch)
