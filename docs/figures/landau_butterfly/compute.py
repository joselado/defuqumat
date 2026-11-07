"""The Hofstadter butterfly of the nearest-neighbour honeycomb model, t = -2.7 eV.

Every reduced flux p/q per cell with q <= 50, all levels on a 6 x 6 mesh of the
magnetic zone (the mesh that closes under p/q -> (q - p)/q). This is a tight-binding
model, not a DFT calculation: it is the same Peierls machinery the Wannier route uses.

The levels are rounded to 1 meV and repeated (flux, energy) pairs are dropped before
saving: the plot draws one pixel per point, so this changes nothing visible, and it takes
the file from 1.87 million points (29 MB) to about 0.16 million.
"""
import math

import numpy as np
from defuqumat.magnetic.conventions import MagneticCell
from defuqumat.magnetic.peierls import PeierlsTorus, honeycomb_nn_bonds
from defuqumat.units import RY_TO_EV
from common import save

bonds = honeycomb_nn_bonds()
mesh = [(i / 6, j / 6) for i in range(6) for j in range(6)]
flux, energy = [], []
for q in range(1, 51):
    for p in range(1, q + 1):
        if math.gcd(p, q) != 1 or (q == 1 and p != 1):
            continue
        torus = PeierlsTorus(MagneticCell.build(bonds.lattice, p, q), bonds)
        for kappa in mesh:
            w = np.linalg.eigvalsh(torus.matrix(kappa).toarray()) * RY_TO_EV
            flux.append(np.full(len(w), p / q))
            energy.append(w)
flux, energy = np.concatenate(flux), np.concatenate(energy)
pairs = np.unique(np.stack([np.round(flux * 1e9), np.round(energy * 1e3)], axis=1), axis=0)
save(flux=(pairs[:, 0] * 1e-9).astype(np.float32),
     energy=(pairs[:, 1] * 1e-3).astype(np.float32))
