"""The interval a Chebyshev expansion of ``H(k)`` is taken on.

The expansion needs ``[centre - half, centre + half]`` to contain the whole
spectrum of ``H(k)`` in the plane-wave basis, at every k-point, and every per
cent of ``half`` it wastes is a per cent more moments for the same resolution.
In plane waves the top of the spectrum is set by the basis, roughly ``ecutwfc``
plus the largest value of the local potential, and the bottom by the lowest
band.

Both edges come from a short Lanczos run, ``steps`` steps from one random vector
at each k-point, whose extreme Ritz values approach the extreme eigenvalues from
inside. They are widened by the larger of ``margin`` of the width and the Ritz
residual ``beta_m |s_m|`` of each edge, as pyqula's BSE resolvent does it
(``bsetk/chebyshev.py``, 2 per cent). An estimate rather than a bound; the
a-posteriori guard (:mod:`defuqumat.kpm.guard`) is what makes it safe, since a
single eigenvalue outside ``[-1, 1]`` makes ``T_n`` grow as ``cosh(n acosh x)``.

Weyl's inequality, which ``realtime/propagate.spectral_bounds`` uses, is not
used for the edges: its lower edge (the local potential's minimum, about -14 Ry
on silicon) doubles ``half``, and that function's ``kappa_max`` widening is not
needed here.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from scipy.linalg import eigh_tridiagonal

from defuqumat.batching import map_k
from defuqumat.kpm.moments import live_mask, random_phases

#: Lanczos steps per k-point.
LANCZOS_STEPS = 40
#: The edges are widened by at least this fraction of the width, each side.
EDGE_MARGIN = 0.02


@functools.partial(jax.jit, static_argnames=("steps", "k_batch"))
def lanczos_coefficients(hamiltonian, key, steps: int, k_batch=None):
    """``(alpha, beta)``, each ``(nk, steps)``: the Lanczos tridiagonal at every k.

    Plain three-term Lanczos without reorthogonalisation: the copies of a
    converged extreme eigenvalue it produces sit on that eigenvalue and do not
    move the edges. ``beta[:, -1]`` is the norm of the residual vector the
    Ritz-value error bound needs.
    """
    mask = live_mask(hamiltonian)
    dtype = hamiltonian.dtype

    def one_k(ik):
        live = mask[ik]
        start = random_phases(jax.random.fold_in(key, ik), live, 1, dtype)[0]

        def apply(v):
            return jnp.where(live, hamiltonian.apply(v[None], ik)[0], 0.0)

        def step(carry, _):
            previous, current, beta_previous = carry
            w = apply(current) - beta_previous * previous
            alpha = jnp.real(jnp.vdot(current, w))
            w = w - alpha * current
            beta = jnp.sqrt(jnp.real(jnp.vdot(w, w)))
            following = w / jnp.where(beta > 0, beta, 1.0)
            return (current, following, beta), (alpha, beta)

        zero = jnp.zeros((), jnp.real(start).dtype)
        _, (alpha, beta) = lax.scan(step, (jnp.zeros_like(start), start, zero),
                                    None, length=steps)
        return alpha, beta

    return map_k(one_k, jnp.arange(mask.shape[0]), batch=k_batch)


def ritz_edges(alpha: np.ndarray, beta: np.ndarray) -> tuple[float, float, float, float]:
    """``(lowest, highest, error_low, error_high)`` of one tridiagonal, in its units.

    The error of a Ritz value ``theta_j`` is ``beta_m |s_mj|``, the last
    component of its Ritz vector times the last ``beta``: there is an
    eigenvalue of the operator within that distance of ``theta_j``.
    """
    values, vectors = eigh_tridiagonal(alpha, beta[:-1])
    errors = abs(beta[-1]) * np.abs(vectors[-1])
    return float(values[0]), float(values[-1]), float(errors[0]), float(errors[-1])


def spectral_interval(hamiltonians, seed: int = 0, steps: int = LANCZOS_STEPS,
                      margin: float = EDGE_MARGIN, k_batch=None) -> tuple[float, float, float]:
    """``(centre, half, lowest)`` in Ry: an interval holding every k-point's spectrum.

    Of every channel. ``lowest`` is the lowest Ritz value before the widening,
    the estimate of the bottom of the bands.
    """
    low, high = np.inf, -np.inf
    error_low, error_high = 0.0, 0.0
    key = jax.random.PRNGKey(seed)
    for channel, hamiltonian in enumerate(hamiltonians):
        alpha, beta = lanczos_coefficients(hamiltonian, jax.random.fold_in(key, channel),
                                           steps, k_batch=k_batch)
        for a, b in zip(np.asarray(alpha), np.asarray(beta)):
            lo, hi, elo, ehi = ritz_edges(a, b)
            if lo < low:
                low, error_low = lo, elo
            if hi > high:
                high, error_high = hi, ehi
    width = high - low
    lowest = low
    low -= max(margin * width, error_low)
    high += max(margin * width, error_high)
    return 0.5 * (high + low), 0.5 * (high - low), lowest
