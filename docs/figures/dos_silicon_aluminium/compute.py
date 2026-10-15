"""Compute figure dos_silicon_aluminium: the DOS of an insulator and a metal.

Silicon with tetrahedra and with a Gaussian smearing; aluminium with
tetrahedra. Writes data.npz next to this file.
    bash docs/figures/run.sh dos_silicon_aluminium
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common

from defuqumat import Calculator
from defuqumat.units import RY_TO_EV

PSEUDO = os.path.join(HERE, "..", "..", "..", "tests", "data", "pseudo")
params = common.start(dict(silicon="si.in", aluminium="al.in",
                           grid_si=12, grid_al=16, degauss=0.02))

silicon = Calculator.from_file(os.path.join(HERE, "si.in"), pseudo_dir=PSEUDO)
n = params["grid_si"]
tetra = silicon.get_dos(grid=(n, n, n), nbnd=8, scheme="tetrahedra")
gauss = silicon.get_dos(grid=(n, n, n), nbnd=8, scheme="gaussian",
                        degauss=params["degauss"])
levels = np.asarray(silicon.dos_states.eigenvalues)
vbm, cbm = levels[:, 3].max(), levels[:, 4].min()

aluminium = Calculator.from_file(os.path.join(HERE, "al.in"), pseudo_dir=PSEUDO)
n = params["grid_al"]
al = aluminium.get_dos(grid=(n, n, n), nbnd=6)
bottom = np.asarray(aluminium.dos_states.eigenvalues).min()

# free electrons: D(E) = c sqrt(E - E0), fitted near the bottom of the band
window = (al.energies > bottom + 0.03) & (al.energies < bottom + 0.30)
slope, intercept = np.polyfit(al.energies[window], al.dos[window] ** 2, 1)
free = np.sqrt(np.maximum(slope * al.energies + intercept, 0.0))

common.save(__file__, params,
            si_e=(tetra.energies - vbm) * RY_TO_EV, si_tetra=tetra.dos_ev,
            si_gauss_e=(gauss.energies - vbm) * RY_TO_EV, si_gauss=gauss.dos_ev,
            si_gap=(cbm - vbm) * RY_TO_EV,
            al_e=(al.energies - al.fermi_energy) * RY_TO_EV, al_dos=al.dos_ev,
            al_free=free / RY_TO_EV,
            al_states_below_ef=al.states_below(al.fermi_energy))
