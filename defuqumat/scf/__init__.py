"""The self-consistent field: potential, density, occupations, mixing, driver.

Two ways to reach the fixed point, behind ``run_scf``'s ``scf_solver`` (rule R4):
the mixing loop in :mod:`~defuqumat.scf.driver`, which is the default, and the
residual of :mod:`~defuqumat.scf.residual` solved by
:mod:`~defuqumat.scf.solvers`. See ``PLAN.md`` P22 for which to reach for.

:mod:`~defuqumat.scf.continuation` starts one run where another stopped, across a
change of spin regime -- an unpolarized density as the starting point of a
collinear run, a collinear one of a noncollinear run, and spin-orbit coupling
switched on and off without going back to the atoms (``PLAN.md`` P23).
"""

from defuqumat.scf.continuation import ContinuedState, continued_state
from defuqumat.scf.density import band_density, sum_band
from defuqumat.scf.driver import Calculation, SCFResult, default_nbnd, run_scf
from defuqumat.scf.ewald import ewald_energy
from defuqumat.scf.mixing import get_mixer
from defuqumat.scf.occupations import fermi_level, fixed_occupations, smeared_occupations
from defuqumat.scf.potential import Potential, v_of_rho
from defuqumat.scf.residual import ScfResidual, make_residual
from defuqumat.scf.solvers import SCF_SOLVERS, get_scf_solver, newton_krylov

__all__ = [
    "Calculation",
    "ContinuedState",
    "Potential",
    "SCF_SOLVERS",
    "ScfResidual",
    "SCFResult",
    "band_density",
    "continued_state",
    "default_nbnd",
    "ewald_energy",
    "fermi_level",
    "fixed_occupations",
    "get_mixer",
    "get_scf_solver",
    "make_residual",
    "newton_krylov",
    "run_scf",
    "smeared_occupations",
    "sum_band",
    "v_of_rho",
]
