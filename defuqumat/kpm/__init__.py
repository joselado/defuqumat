"""The kernel polynomial method: spectral quantities from Chebyshev moments of ``H``.

``plans/kpm.md`` is the plan: what can win in a plane-wave basis, in which order,
and how each piece is checked. P167 is the engine and the density of states, P168 the deterministic traces,
P169 the optical conductivity and P170 the control variate.
"""

from defuqumat.kpm.conductivity import KPMConductivity, run_kpm_conductivity
from defuqumat.kpm.dos import KPMDensityOfStates, kpm_moments, run_kpm_dos
from defuqumat.kpm.guard import SpectralIntervalError
from defuqumat.kpm.kernels import KERNELS, get_kernel, register_kernel

__all__ = [
    "KERNELS",
    "KPMConductivity",
    "KPMDensityOfStates",
    "SpectralIntervalError",
    "get_kernel",
    "kpm_moments",
    "register_kernel",
    "run_kpm_conductivity",
    "run_kpm_dos",
]
