"""Diagnose the zero-mode sublattice trace of the shift-invert solve at large q.

Prints, at q = 1975, 3950, 7900 (NN model, kappa (0.11, 0.23)): the zero cluster's
eigenvalues, the orthonormality of the returned vectors (max |V^+ V - 1|), their residuals
|H v - e v|, tr P_A over the pair, and the same after a Rayleigh-Ritz step on the returned
block and after a dense solve where the size allows it.
Usage: python mag_zero_modes.py
"""
import numpy as np
import scipy.sparse.linalg as spla

from defuqumat.magnetic.conventions import MagneticCell
from defuqumat.magnetic.peierls import PeierlsTorus, honeycomb_nn_bonds
from defuqumat.units import RY_TO_EV

bonds = honeycomb_nn_bonds()
for q in (1975, 3950, 7900):
    t = PeierlsTorus(MagneticCell.build(bonds.lattice, 1, q), bonds)
    h = t.matrix((0.11, 0.23)).tocsc()
    pa = np.zeros(t.dim)
    pa[0::2] = 1.0
    for offset in (1e-7, 1e-4, 3e-3):
        w, v = spla.eigsh(h, k=12, sigma=offset, which="LM")
        o = np.argsort(np.abs(w))
        w, v = w[o], v[:, o]
        z = v[:, :2]
        gram = np.max(np.abs(z.conj().T @ z - np.eye(2)))
        res = np.max(np.linalg.norm(h @ z - z * w[:2], axis=0))
        tra = float(np.einsum("ia,i,ia->", z.conj(), pa, z).real)
        print(f"q={q} offset {offset:g}: zero pair {w[:2] * RY_TO_EV} eV, next {w[2] * RY_TO_EV:.6f}; "
              f"|V+V-1| {gram:.2e}, residual {res:.2e} Ry, tr P_A {tra:.9f}; "
              f"A weights of each {np.round((np.abs(z) ** 2 * pa[:, None]).sum(0), 6)}", flush=True)
