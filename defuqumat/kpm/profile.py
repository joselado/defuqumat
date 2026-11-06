"""From Chebyshev moments to a function of energy.

With ``x = (E - centre) / half`` and ``c_n = g_n mu_n`` (``g_n`` the kernel),

    rho(x) = [c_0 + 2 sum_{n>=1} c_n T_n(x)] / (pi sqrt(1 - x^2)),

and, with ``theta = arccos x``, its integral from the bottom of the interval,

    N(x) = [c_0 (pi - theta) - 2 sum_{n>=1} c_n sin(n theta) / n] / pi,

which is the closed form pyqula's ``kpmtk/density.py`` uses. The series is
evaluated directly on the energy grid, ``T_n(x) = cos(n theta)``: at ``N nE`` of
about ``10^7`` there is no need for a discrete cosine transform. In ``jnp``, so a
DOS built from differentiated moments stays differentiable. Ported from pyqula,
``kpmtk/momenttoprofile.py``.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

#: Energies evaluated at once, so the ``(N, chunk)`` table stays small.
ENERGY_CHUNK = 1024


@jax.jit
def _profile(coefficients, x):
    n_moments = coefficients.shape[-1]
    inside = jnp.abs(x) < 1.0
    xc = jnp.where(inside, x, 0.0)
    theta = jnp.arccos(xc)
    n = jnp.arange(n_moments, dtype=theta.dtype)
    weights = jnp.where(n > 0, 2.0, 1.0)
    cosines = jnp.cos(n[:, None] * theta[None, :])
    density = (coefficients * weights) @ cosines / (math.pi * jnp.sqrt(1.0 - xc**2))
    density = jnp.where(inside, density, 0.0)
    sines = jnp.sin(n[1:, None] * theta[None, :]) / n[1:, None]
    count = (coefficients[..., :1] * (math.pi - theta)[None, :]
             - 2.0 * coefficients[..., 1:] @ sines) / math.pi
    count = count.reshape(density.shape)
    total = coefficients[..., :1]
    count = jnp.where(inside, count, jnp.where(x >= 1.0, total, 0.0))
    return density, count


def chebyshev_profile(moments, factors, centre: float, half: float, energies):
    """``(dos, integrated)`` on ``energies`` (Ry) from ``moments`` ``(..., N)``.

    ``dos`` is per Ry and ``integrated`` is a count, both with the moments'
    normalisation: moments of a trace give states.
    """
    coefficients = jnp.asarray(moments) * jnp.asarray(factors)
    energies = jnp.asarray(energies)
    x = (energies - centre) / half
    densities, counts = [], []
    for start in range(0, x.shape[0], ENERGY_CHUNK):
        d, c = _profile(coefficients, x[start:start + ENERGY_CHUNK])
        densities.append(d)
        counts.append(c)
    return (jnp.concatenate(densities, axis=-1) / half,
            jnp.concatenate(counts, axis=-1))
