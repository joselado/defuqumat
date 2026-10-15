"""The band structure of silicon along L-Gamma-X-U|K-Gamma."""
import numpy as np
from defuqumat import Calculator
from defuqumat.system.kpoints import KPoints
from common import PSEUDO, save

calc = Calculator.from_file("si.in", pseudo_dir=PSEUDO)
scf = calc.get_scf()

# the corners of the path, in units of 2 pi / a
corners = {"L": [0.5, 0.5, 0.5], "G": [0.0, 0.0, 0.0], "X": [1.0, 0.0, 0.0],
           "U": [1.0, 0.25, 0.25], "K": [0.75, 0.75, 0.0]}
labels = ["L", "G", "X", "U", "K", "G"]
counts = [26, 30, 10, 0, 32, 1]           # 0: a jump from U to K, which are equivalent
path = KPoints.band_path([corners[p] for p in labels], counts, calc.system.cell,
                         crystal=False)
bands = calc.get_bands(kpoints=path, nbnd=8)

vbm = bands.eigenvalues_ev[:, :4].max()
corner = np.cumsum([0] + [max(c, 1) for c in counts[:-1]])   # index of each corner
save(x=bands.path_length, energies=bands.eigenvalues_ev - vbm,
     ticks=bands.path_length[corner], labels=np.array(labels),
     gap=bands.gap(8), total_energy=scf.total_energy)
