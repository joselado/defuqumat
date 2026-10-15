"""The bands of bismuthene without and with spin-orbit coupling."""
import numpy as np
from defuqumat import Calculator
from defuqumat.system.kpoints import KPoints
from defuqumat.units import RY_TO_EV
from common import PSEUDO, save

corners = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1 / 3, 1 / 3, 0.0], [0.0, 0.0, 0.0]]
counts = [30, 18, 36, 1]                      # Gamma - M - K - Gamma, crystal units
data = {}
for tag, nbnd, filled in (("nosoc", 20, 15), ("soc", 40, 30)):   # 30 electrons
    calc = Calculator.from_file(f"bi-{tag}.in", pseudo_dir=PSEUDO)
    scf = calc.get_scf()
    path = KPoints.band_path(corners, counts, calc.system.cell, crystal=True)
    bands = calc.get_bands(kpoints=path, nbnd=nbnd)
    e = bands.eigenvalues_ev - scf.fermi_energy * RY_TO_EV
    data[tag] = e
    data[tag + "_gap"] = (e[:, filled] - e[:, filled - 1]).min()   # smallest direct gap
    print(tag, "smallest direct gap %.4f eV" % data[tag + "_gap"], flush=True)

save(x=bands.path_length, ticks=bands.path_length[np.cumsum([0] + counts[:-1])], **data)
