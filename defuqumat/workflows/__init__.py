"""End-to-end calculations: SCF, bands, densities of state, projections, relaxation, topology, optical spectra."""

from defuqumat.workflows.bands import BandStructure, run_bands
from defuqumat.workflows.dos import DensityOfStates, compute_dos, energy_grid, run_dos
from defuqumat.workflows.nscf import (
    NSCFResult,
    denser_grid,
    fixed_density_bands,
    fixed_density_states,
    run_nscf,
)
from defuqumat.workflows.pdos import (
    LowdinCharges,
    ProjectedDOS,
    compute_pdos,
    project_states,
    run_pdos,
)
from defuqumat.workflows.relax import RelaxResult, run_relax
from defuqumat.workflows.conductivity import run_conductivity
from defuqumat.workflows.nesting import run_nesting
from defuqumat.workflows.orbital_magnetization import run_orbital_magnetization
from defuqumat.workflows.magnons import (MagnonDispersion, SpinSusceptibility,
                                       run_magnon_dispersion,
                                       run_spin_susceptibility)
from defuqumat.workflows.tddft import OpticalSpectrum, run_absorption
from defuqumat.workflows.polarization import run_polarization
from defuqumat.workflows.piezo_ladder import (PiezoelectricLadder,
                                            piezoelectric_kmesh_ladder)
from defuqumat.workflows.sfac import run_structure_factors
from defuqumat.workflows.spiral_soc import SpiralSpinOrbit, spiral_spin_orbit_energy
from defuqumat.workflows.stm import run_stm, run_sts
from defuqumat.workflows.realtime import (
    run_harmonic_orders, run_hhg, run_realtime, run_realtime_dielectric,
    run_third_harmonic, run_nonlinear_spectrum)
from defuqumat.workflows.topology import (
    DFTSource,
    run_berry_curvature,
    run_z2,
    run_z2_3d,
)

__all__ = [
    "BandStructure",
    "OpticalSpectrum",
    "DFTSource",
    "run_polarization",
    "PiezoelectricLadder",
    "piezoelectric_kmesh_ladder",
    "run_orbital_magnetization",
    "DensityOfStates",
    "NSCFResult",
    "RelaxResult",
    "LowdinCharges",
    "ProjectedDOS",
    "compute_dos",
    "compute_pdos",
    "denser_grid",
    "energy_grid",
    "run_absorption",
    "MagnonDispersion",
    "SpinSusceptibility",
    "run_magnon_dispersion",
    "run_spin_susceptibility",
    "run_conductivity",
    "run_nesting",
    "run_structure_factors",
    "SpiralSpinOrbit",
    "spiral_spin_orbit_energy",
    "run_stm",
    "run_realtime",
    "run_realtime_dielectric",
    "run_hhg",
    "run_harmonic_orders",
    "run_third_harmonic",
    "run_nonlinear_spectrum",
    "run_sts",
    "fixed_density_bands",
    "fixed_density_states",
    "project_states",
    "run_bands",
    "run_berry_curvature",
    "run_dos",
    "run_nscf",
    "run_pdos",
    "run_relax",
    "run_z2",
    "run_z2_3d",
]
