"""The second-harmonic spectrum of AlAs, with silicon as the control."""
import numpy as np
from defuqumat import Calculator
from defuqumat.system.kpoints import KPoints
from defuqumat.units import RY_TO_EV
from common import PSEUDO, save

# AlAs: the ground state first (with empty bands, for the gap), then chi^(2) on a 6x6x6 mesh
alas = Calculator.from_file("alas.in", pseudo_dir=PSEUDO)
levels = np.asarray(alas.get_scf(nbnd=12).eigenvalues)
gap = (levels[:, 4] - levels[:, 3]).min() * RY_TO_EV          # the smallest direct gap
mesh = KPoints.automatic((6, 6, 6), (0, 0, 0), alas.system.cell)
chi = alas.get_shg(kpoints=mesh, nbnd=23, window=0.6, nw=240, broadening=0.010)

# silicon: the same calculation, where inversion forbids every component
silicon = Calculator.from_file("si.in", pseudo_dir=PSEUDO)
si_mesh = KPoints.automatic((6, 6, 6), (0, 0, 0), silicon.system.cell)
si_chi = silicon.get_shg(kpoints=si_mesh, nbnd=14, window=0.6, nw=240, broadening=0.010)

allowed = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
forbidden = max(np.abs(chi.component(a, b, c)).max() for a in range(3) for b in range(3)
                for c in range(3) if (a, b, c) not in allowed)
save(ev=chi.frequencies_ev, xyz=chi.component(0, 1, 2), gap=gap,
     si_ev=si_chi.frequencies_ev, si_xyz=si_chi.component(0, 1, 2),
     si_peak=np.abs(si_chi.chi).max(), forbidden=forbidden,
     truncation=chi.truncation)
