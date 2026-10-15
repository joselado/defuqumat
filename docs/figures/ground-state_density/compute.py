"""Compute figure ground-state_density: the valence density of silicon.

The density is on the real-space grid of the run. We take its Fourier
coefficients and evaluate it on the (1-10) plane, which holds the bonds.
Writes data.npz next to this file.
    bash docs/figures/run.sh ground-state_density
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common

from defuqumat import Calculator

PSEUDO = os.path.join(HERE, "..", "..", "..", "tests", "data", "pseudo")
params = common.start(dict(input="si.in", plane="(1-10)", points=161))

calc = Calculator.from_file(os.path.join(HERE, "si.in"), pseudo_dir=PSEUDO)
scf = calc.get_scf()

rho = np.asarray(scf.density)[0]                  # electrons/bohr^3, (n1, n2, n3)
at = np.asarray(calc.system.cell.at)              # lattice vectors in bohr, rows
alat = calc.system.cell.alat
coef = np.fft.fftn(rho) / rho.size                # rho(r) = sum_G c_G exp(iG.r)
m = np.meshgrid(*[np.fft.fftfreq(n) * n for n in rho.shape], indexing="ij")
miller = np.stack([a.ravel() for a in m], axis=1)
g = miller @ (2 * np.pi * np.linalg.inv(at).T)    # cartesian G = m @ b, 1/bohr
coef = coef.ravel()

# the plane: [110] across, [001] up, one conventional cell
e1 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
e2 = np.array([0.0, 0.0, 1.0])
u = np.linspace(0.0, np.sqrt(2.0) * alat, params["points"])
v = np.linspace(0.0, alat, params["points"] * 10 // 14)
image = np.empty((len(u), len(v)))
for i, ui in enumerate(u):
    r = ui * e1[None] + v[:, None] * e2[None]
    image[i] = (np.exp(1j * r @ g.T) @ coef).real

# the atoms that lie on this plane
tau = np.asarray(calc.system.structure.positions)
atoms = []
for n1 in range(-3, 4):
    for n2 in range(-3, 4):
        for n3 in range(-3, 4):
            for t in tau:
                r = t + np.array([n1, n2, n3]) @ at
                a, b = r @ e1, r @ e2
                if (abs(r @ np.array([1.0, -1.0, 0.0])) < 1e-6
                        and -1e-6 <= a <= u[-1] + 1e-6 and -1e-6 <= b <= v[-1] + 1e-6):
                    atoms.append((a, b))

common.save(__file__, params, u=u, v=v, density=image, atoms=np.array(atoms),
            total_energy=scf.total_energy)
