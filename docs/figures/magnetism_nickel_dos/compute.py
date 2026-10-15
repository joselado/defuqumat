"""The bands and the DOS of fcc nickel, for each spin channel."""
import numpy as np
from defuqumat import Calculator
from defuqumat.system.kpoints import KPoints
from defuqumat.units import RY_TO_EV
from common import PSEUDO, save

calc = Calculator.from_file("ni.in", pseudo_dir=PSEUDO)
scf = calc.get_scf()
ef = scf.fermi_energy

# W - L - Gamma - X - W - K, in units of 2 pi / a
corners = [[0.5, 0.25, 0.75], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0],
           [0.0, 0.0, 1.0], [0.5, 0.25, 0.75], [0.375, 0.375, 0.75]]
counts = [24, 32, 32, 16, 12, 1]
path = KPoints.band_path(corners, counts, calc.system.cell, crystal=False)
bands = calc.get_bands(kpoints=path, nbnd=9)

dos = calc.get_dos(grid=(16, 16, 16), nbnd=9)
save(x=bands.path_length, bands=(np.asarray(bands.eigenvalues_by_spin) - ef) * RY_TO_EV,
     ticks=bands.path_length[np.cumsum([0] + counts[:-1])],
     energies=(dos.energies - dos.fermi_energy) * RY_TO_EV, dos=dos.dos / RY_TO_EV,
     moment=scf.magnetization)
