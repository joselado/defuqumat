"""Compute figure bands_silicon: the band structure of silicon.

Writes data.npz next to this file.
    bash docs/figures/run.sh bands_silicon
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common

from defuqumat import Calculator
from defuqumat.system.kpoints import KPoints

PSEUDO = os.path.join(HERE, "..", "..", "..", "tests", "data", "pseudo")

# the path L - Gamma - X - U|K - Gamma, in units of 2 pi / a
corners = {"L": [0.5, 0.5, 0.5], "G": [0.0, 0.0, 0.0], "X": [1.0, 0.0, 0.0],
           "U": [1.0, 0.25, 0.25], "K": [0.75, 0.75, 0.0]}
labels = ["L", "G", "X", "U", "K", "G"]
counts = [26, 30, 10, 0, 32, 1]       # 0: jump from U to K, which are equivalent
params = common.start(dict(input="si.in", path="-".join(labels), counts=counts,
                           nbnd=8))

calc = Calculator.from_file(os.path.join(HERE, "si.in"), pseudo_dir=PSEUDO)
scf = calc.get_scf()

path = KPoints.band_path([corners[p] for p in labels], counts, calc.system.cell,
                         crystal=False)
bands = calc.get_bands(kpoints=path, nbnd=params["nbnd"])

vbm = bands.eigenvalues_ev[:, :4].max()
ticks = np.cumsum([0] + [max(c, 1) for c in counts[:-1]])
common.save(__file__, params,
            x=bands.path_length, energies=bands.eigenvalues_ev - vbm,
            ticks=bands.path_length[ticks], labels=np.array(labels),
            gap=bands.gap(8), total_energy=scf.total_energy)
