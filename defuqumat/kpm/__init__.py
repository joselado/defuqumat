"""The kernel polynomial method: spectral quantities from Chebyshev moments of ``H``.

``plans/kpm.md`` is the plan: what can win in a plane-wave basis, in which order,
and how each piece is checked. P167 is the engine and the density of states.
"""

from defuqumat.kpm.dos import KPMDensityOfStates, kpm_moments, run_kpm_dos
from defuqumat.kpm.guard import SpectralIntervalError
from defuqumat.kpm.kernels import KERNELS, get_kernel, register_kernel

__all__ = [
    "KERNELS",
    "KPMDensityOfStates",
    "SpectralIntervalError",
    "get_kernel",
    "kpm_moments",
    "register_kernel",
    "run_kpm_dos",
]
