"""A control variate for the stochastic KPM trace, from filtered atomic orbitals.

The plain estimate of ``Tr T_n(H')`` at one k-point is ``n_live <r|T_n(H')|r>``
for a random-phase ``r``. Its noise near the Fermi level comes from the states
there, and those live mostly in the atomic orbitals and their leak into the
interstitial. So we take the atomic orbitals ``phi``, pass them through a
Chebyshev low-pass filter of degree ``m``,

    Y = T_m((H - c_f) / e_f) phi,   [c_f - e_f, c_f + e_f] = [E_cut, E_top],

which damps every component above ``E_cut`` relative to those below it (the
polynomial grows as ``cosh(m acosh |x|)`` outside ``[-1, 1]``), orthonormalise
``Y`` and solve ``H`` in that subspace (Rayleigh-Ritz), giving pairs
``(theta_j, y_j)``. The estimator of each random sample is then

    n_live <r|T_n|r> - n_live sum_j |<y_j|r>|^2 T_n(theta_j') + sum_j T_n(theta_j'),

which has the same expectation, since ``E[n_live |<y|r>|^2] = <y|y> = 1`` for a
normalised ``y`` on the live sphere: it is unbiased **whatever** the quality of
the Ritz pairs. Its variance is that of ``<r|T_n(H) - sum_j T_n(theta_j)|y_j><y_j|
|r>``, which is small wherever the pairs are close to eigenpairs, so in the
valence band and a few conduction states. This is the form a subagent measured
on D22 (``PLAN.md``, end of P168): with ``m = 8`` and ``E_cut = E_F + 1`` Ry, 50 to
100 times less noise than the plain trace at the same number of applications of
``H`` on Si16 to Si128 at one random vector, and the Ritz eigenvalues good to
about 1e-5 Ry. Plain ``H phi`` Krylov vectors spend their rank on high kinetic
energy and are worse at three times the rank.

The cost is ``(m + 1) natw`` applications per k-point and channel, and a dense
SVD and eigensolve of rank ``natw``, ``O(npw natw^2)``, which is Davidson's
scaling: the variate pays when the error asked for is about a per cent. Memory:
the filtered block ``(natw, ndim)`` and its two recursion partners for one
k-point at a time, plus the orbitals of every k-point (``atomic_orbitals``).
The reference for the filter is Zhou, Saad, Tiago and Chelikowsky,
J. Comput. Phys. 219, 172 (2006); the control variate is the standard one of
Monte Carlo trace estimation (Hutchinson's estimator with a deflated part
estimated exactly, as in Meyer, Musco, Musco and Woodruff, arXiv:2010.09649).
"""

from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from defuqumat.kpm.moments import live_mask, random_phases

#: Degree of the Chebyshev low-pass filter.
FILTER_DEGREE = 8
#: The filter damps everything above ``E_F + FILTER_CUT`` (Ry).
FILTER_CUT = 1.0


def resolve_variate(control_variate, trace: str, calculation) -> bool:
    """Whether to use the variate: ``"auto"`` is on for random vectors when the datasets carry orbitals.

    ``"auto"`` (the default since 2026-11-07, a fable decision on the numbers of
    P169 and P170: the plain conductivity at one vector is 28 to 38 per cent off,
    the variate about one) is off for ``trace="exact"``, which has no noise, and
    off with a warning when no pseudopotential carries atomic orbitals
    (``PP_CHI``), since the plain trace is the same unbiased estimate with more
    noise. ``True`` keeps its errors in both cases, so a request is not dropped
    silently.
    """
    if control_variate != "auto":
        return bool(control_variate)
    if trace != "stochastic":
        return False
    from defuqumat.projwfc.projections import calculation_channels

    if not calculation_channels(calculation):
        warnings.warn(
            "control_variate='auto': no pseudopotential carries atomic orbitals (PP_CHI), "
            "so the random trace runs without the control variate and keeps its full noise",
            RuntimeWarning, stacklevel=3)
        return False
    return True


@functools.partial(jax.jit, static_argnames=("degree",))
def ritz_pairs(hamiltonian, ik, block, cut_centre, cut_half, degree: int):
    """``(theta, y)``: Ritz pairs of ``H`` in the low-pass-filtered span of ``block`` at ``ik``.

    ``block`` is ``(r, ndim)``. ``theta`` is ``(r,)`` in Ry and ``y`` ``(r, ndim)``
    orthonormal on the live sphere. Every direction of the filtered block is
    kept: an orthonormal set leaves the variate unbiased whatever it spans, so
    a weak direction costs efficiency, never correctness.
    """
    live = live_mask(hamiltonian)[ik]

    def apply(v):
        return jnp.where(live, hamiltonian.apply(v, ik), 0.0)

    def shifted(v):
        return (apply(v) - cut_centre * v) / cut_half

    first = jnp.where(live, block, 0.0)
    second = shifted(first)

    def step(_, carry):
        previous, current = carry
        return current, 2.0 * shifted(current) - previous

    _, filtered = lax.fori_loop(1, degree, step, (first, second))
    # orthonormalise the rows: filtered^T = U s V^dagger, keep U
    u, _, _ = jnp.linalg.svd(filtered.T, full_matrices=False)
    basis = u.T  # (r, ndim), orthonormal rows
    projected = jnp.conj(basis) @ apply(basis).T  # <u_i|H|u_j>
    projected = 0.5 * (projected + jnp.conj(projected.T))
    theta, z = jnp.linalg.eigh(projected)
    return theta, z.T @ basis


@functools.partial(jax.jit, static_argnames=("n_random",))
def sample_overlaps(hamiltonian, ik, key, ritz_vectors, n_random: int):
    """``(R, r)``: ``|<y_j|r_s>|^2`` for the random vectors :func:`chebyshev_moments` draws at ``ik``."""
    live = live_mask(hamiltonian)[ik]
    start = random_phases(jax.random.fold_in(key, ik), live, n_random, hamiltonian.dtype)
    return jnp.abs(start @ jnp.conj(ritz_vectors).T) ** 2


def control_corrections(hamiltonian, orbitals, key, centre, half, n_moments, n_random,
                        weights, cut, degree=FILTER_DEGREE):
    """``(R, N)``: the k-weighted correction to add to each random sample's moments.

    ``orbitals`` is ``(nk, r, ndim)``; ``key`` is the channel's key, the one
    :func:`~defuqumat.kpm.moments.chebyshev_moments` folds with ``ik``; ``cut`` is
    ``E_cut`` in Ry. Returns also the Ritz values, ``(nk, r)``, for the record.
    """
    top = centre + half
    cut_centre, cut_half = 0.5 * (top + cut), 0.5 * (top - cut)
    mask = np.asarray(live_mask(hamiltonian))
    n = np.arange(n_moments)
    total = np.zeros((n_random, n_moments))
    values = []
    for ik in range(mask.shape[0]):
        theta, vectors = ritz_pairs(hamiltonian, ik, orbitals[ik], cut_centre, cut_half, degree)
        overlaps = np.asarray(sample_overlaps(hamiltonian, ik, key, vectors, n_random))
        x = np.clip((np.asarray(theta) - centre) / half, -1.0, 1.0)
        chebyshev = np.cos(n[None, :] * np.arccos(x)[:, None])  # (r, N)
        n_live = float(mask[ik].sum())
        total += weights[ik] * (chebyshev.sum(axis=0)[None, :] - n_live * overlaps @ chebyshev)
        values.append(np.asarray(theta))
    return total, np.stack(values)
