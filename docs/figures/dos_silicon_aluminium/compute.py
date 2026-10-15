"""The DOS of an insulator (silicon) and of a metal (aluminium)."""
import numpy as np
from defuqumat import Calculator
from defuqumat.units import RY_TO_EV
from common import PSEUDO, save

# silicon, with tetrahedra and with a Gaussian of 0.02 Ry
silicon = Calculator.from_file("si.in", pseudo_dir=PSEUDO)
tetra = silicon.get_dos(grid=(12, 12, 12), nbnd=8, scheme="tetrahedra")
gauss = silicon.get_dos(grid=(12, 12, 12), nbnd=8, scheme="gaussian", degauss=0.02)
levels = np.asarray(silicon.dos_states.eigenvalues)
vbm, cbm = levels[:, 3].max(), levels[:, 4].min()

# aluminium, with tetrahedra (the scheme of al.in)
aluminium = Calculator.from_file("al.in", pseudo_dir=PSEUDO)
al = aluminium.get_dos(grid=(16, 16, 16), nbnd=6)
bottom = np.asarray(aluminium.dos_states.eigenvalues).min()

# free electrons, D(E) = c sqrt(E - E0), fitted near the bottom of the band
near = (al.energies > bottom + 0.03) & (al.energies < bottom + 0.30)
slope, intercept = np.polyfit(al.energies[near], al.dos[near] ** 2, 1)
free = np.sqrt(np.maximum(slope * al.energies + intercept, 0.0))

save(si_e=(tetra.energies - vbm) * RY_TO_EV, si_tetra=tetra.dos_ev,
     si_gauss_e=(gauss.energies - vbm) * RY_TO_EV, si_gauss=gauss.dos_ev,
     si_gap=(cbm - vbm) * RY_TO_EV,
     al_e=(al.energies - al.fermi_energy) * RY_TO_EV, al_dos=al.dos_ev,
     al_free=free / RY_TO_EV, al_states_below_ef=al.states_below(al.fermi_energy))
