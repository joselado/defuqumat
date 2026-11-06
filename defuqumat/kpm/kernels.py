"""Kernels of the kernel polynomial method, behind a name registry.

A Chebyshev series cut at ``N`` moments rings (the Gibbs oscillations); a kernel
``g_n`` multiplies the moments so that the truncated series of a delta function
is a positive, narrow peak instead. Each kernel carries the width of that peak,
so a broadening asked for in Ry becomes a number of moments.

* ``jackson`` (the default): the peak is close to a Gaussian, of standard
  deviation ``sigma(x) = (pi / N) sqrt(1 - x^2 + x^2 / N)`` in the rescaled
  energy ``x`` to leading order (Weisse, Wellein, Alvermann and Fehske,
  Rev. Mod. Phys. 78, 275 (2006), arXiv:cond-mat/0504627, eq. 72). It narrows
  towards the edges of the spectrum, which in plane waves is where the bands
  are, so the conversion is taken at a reference energy.
* ``lorentz``: ``g_n = sinh(lambda (1 - n/N)) / sinh(lambda)``, whose peak is a
  Lorentzian of half width ``lambda / N`` in ``theta = arccos x`` to leading order
  (same reference, eq. 79), which is ``1.19 lambda sqrt(1 - x^2) / N`` in ``x``
  (:data:`LORENTZ_HWHM`), the
  kernel a Green's function wants. ``lambda = 3`` is pyqula's default.
* ``fejer``: ``g_n = 1 - n/N``, the plain Cesaro mean. It carries no simple
  width, so it needs ``n_moments`` stated.

Ported from pyqula, ``kpmtk/kernels.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

#: ``lambda`` of the Lorentz kernel.
LORENTZ_LAMBDA = 3.0


@dataclass(frozen=True)
class Kernel:
    """A kernel: its factors, and the conversion from a width to a moment count.

    ``factors(N)`` is the ``(N,)`` array ``g_n``. ``moments_for(width, x)`` is
    the number of moments whose peak has half width at half maximum ``width``
    (in the rescaled energy) at ``x``, or ``None`` when the kernel has no width
    formula; ``width_at(N, x)`` is its inverse.
    """

    name: str
    factors: Callable[[int], np.ndarray]
    moments_for: Callable[[float, float], int] | None
    width_at: Callable[[int, np.ndarray], np.ndarray] | None


def _jackson(n_moments: int) -> np.ndarray:
    n = np.arange(n_moments)
    q = math.pi / (n_moments + 1)
    return ((n_moments - n + 1) * np.cos(q * n)
            + np.sin(q * n) / math.tan(q)) / (n_moments + 1)


#: Half width at half maximum of a Gaussian, in units of its standard deviation.
_HWHM = math.sqrt(2.0 * math.log(2.0))


def _jackson_width(n_moments, x):
    x = np.asarray(x)
    return _HWHM * (math.pi / n_moments) * np.sqrt(
        np.maximum(1.0 - x**2 + x**2 / n_moments, 0.0))


def _jackson_moments(width: float, x: float) -> int:
    # sigma(x) at large N is (pi / N) sqrt(1 - x^2); one fixed-point step adds
    # the x^2 / N term, which matters only within 1/N of an edge.
    n = _HWHM * math.pi * math.sqrt(max(1.0 - x * x, 0.0)) / width
    n = max(n, 2.0)
    n = _HWHM * math.pi * math.sqrt(max(1.0 - x * x + x * x / n, 0.0)) / width
    return int(math.ceil(n))


def _lorentz(n_moments: int) -> np.ndarray:
    n = np.arange(n_moments)
    return np.sinh(LORENTZ_LAMBDA * (1.0 - n / n_moments)) / math.sinh(LORENTZ_LAMBDA)


#: Half width at half maximum of the Lorentz kernel's peak, in units of
#: ``lambda sqrt(1 - x^2) / N``: the root of ``F(t) = F(0) / 2`` with
#: ``F(t) = int_0^1 sinh(lambda (1 - u)) cos(lambda u t) du`` at ``lambda = 3``,
#: the continuum form of ``sum_n g_n cos(n theta)``. The kernel is a Lorentzian
#: of width ``lambda / N`` in ``theta`` only to leading order; the second
#: exponential of the sinh widens it by 19 per cent, and ``dx = sin(theta)
#: dtheta`` narrows it towards the edges. Until P169 this was ``lambda / N``
#: flat, 37 per cent too wide at ``x = -0.85`` (measured).
LORENTZ_HWHM = 1.1919166


def _lorentz_width(n_moments, x):
    x = np.asarray(x)
    return LORENTZ_HWHM * (LORENTZ_LAMBDA / n_moments) * np.sqrt(np.maximum(1.0 - x**2, 0.0))


def _lorentz_moments(width: float, x: float) -> int:
    return int(math.ceil(LORENTZ_HWHM * LORENTZ_LAMBDA * math.sqrt(max(1.0 - x * x, 0.0))
                         / width))


def _fejer(n_moments: int) -> np.ndarray:
    return 1.0 - np.arange(n_moments) / n_moments


KERNELS: dict[str, Kernel] = {}


def register_kernel(kernel: Kernel) -> Kernel:
    """Add ``kernel`` to the registry under its name."""
    KERNELS[kernel.name] = kernel
    return kernel


register_kernel(Kernel("jackson", _jackson, _jackson_moments, _jackson_width))
register_kernel(Kernel("lorentz", _lorentz, _lorentz_moments, _lorentz_width))
register_kernel(Kernel("fejer", _fejer, None, None))


def get_kernel(name: str) -> Kernel:
    """The kernel registered as ``name``, case ignored."""
    try:
        return KERNELS[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown KPM kernel {name!r}; the registered ones are {sorted(KERNELS)}"
        ) from None


def moment_count(kernel: Kernel, broadening: float, half_width: float,
                 x_reference: float) -> int:
    """The even number of moments that gives ``broadening`` (Ry, HWHM) at ``x_reference``.

    Even because the doubling trick gives the moments two at a time.
    """
    if kernel.moments_for is None:
        raise ValueError(
            f"the {kernel.name!r} kernel has no width formula, so it cannot turn a "
            "broadening into a number of moments: pass n_moments instead"
        )
    n = kernel.moments_for(broadening / half_width, x_reference)
    return max(2, n + (n % 2))
