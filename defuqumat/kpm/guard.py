"""The a-posteriori check that the interval contained the spectrum.

For a Hermitian ``H'`` with its spectrum in ``[-1, 1]``, ``|T_n(H')| <= 1``, so
``|mu_n| = |<v|T_n(H')|v>| <= <v|v> = mu_0`` for every start vector. An
eigenvalue at ``x = 1 + eps`` outside the interval makes ``T_n(x)`` grow as
``cosh(n acosh x)``, so a moment above the bound, or a NaN, means the interval
was too small and the series is meaningless; it is refused rather than used.
Ported from pyqula, ``kpmtk/scaleguard.py:check_scale``.
"""

from __future__ import annotations

import numpy as np

#: Relative slack of the bound, by precision.
TOLERANCE = {np.dtype(np.float64): 1e-6, np.dtype(np.float32): 1e-3}


class SpectralIntervalError(ValueError):
    """A Chebyshev moment exceeded ``mu_0``: the interval missed part of the spectrum."""


def check_moments(moments, centre: float, half: float) -> None:
    """Raise :class:`SpectralIntervalError` unless ``|mu_n| <= mu_0 (1 + tol)``.

    ``moments`` is ``(..., N)`` per start vector, with ``mu_0`` the squared norm.
    A check of values: it reads the moments on the host and is not part of any
    differentiated path.
    """
    moments = np.asarray(moments)
    tol = TOLERANCE.get(moments.dtype, 1e-3)
    norm = moments[..., :1]
    if not np.all(np.isfinite(moments)):
        raise SpectralIntervalError(
            f"a Chebyshev moment is not finite on [{centre - half:.4f}, "
            f"{centre + half:.4f}] Ry: the interval misses part of the spectrum"
        )
    excess = np.max(np.abs(moments) - norm * (1.0 + tol) - tol)
    if excess > 0.0:
        worst = float(np.max(np.abs(moments) / np.where(norm > 0, norm, 1.0)))
        raise SpectralIntervalError(
            f"a Chebyshev moment reached {worst:.3g} times mu_0 on the interval "
            f"[{centre - half:.4f}, {centre + half:.4f}] Ry, where it cannot exceed "
            "1 if the interval holds the whole spectrum: widen it (a larger "
            "margin or more Lanczos steps)"
        )
