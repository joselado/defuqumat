"""The valence density of silicon in the (1-10) plane, which holds the bonds."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

calc = Calculator.from_file("si.in", pseudo_dir=PSEUDO)
scf = calc.get_scf()
rho = np.asarray(scf.density)[0]             # electrons/bohr^3 on the FFT grid
at = np.asarray(calc.system.cell.at)         # lattice vectors in bohr, as rows
alat = calc.system.cell.alat

# the Fourier coefficients: rho(r) = sum_G c_G exp(i G.r)
c = np.fft.fftn(rho).ravel() / rho.size
m = np.meshgrid(*[np.fft.fftfreq(n) * n for n in rho.shape], indexing="ij")
g = np.stack([x.ravel() for x in m], axis=1) @ (2 * np.pi * np.linalg.inv(at).T)

# the plane: [110] across, [001] up, one conventional cell
e1 = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
e2 = np.array([0.0, 0.0, 1.0])
u = np.linspace(0.0, np.sqrt(2.0) * alat, 161)
v = np.linspace(0.0, alat, 115)
image = np.array([((np.exp(1j * (a * e1 + v[:, None] * e2) @ g.T)) @ c).real
                  for a in u])

# the atoms on this plane, from the atoms of nearby cells
cells = np.array([[i, j, k] for i in range(-3, 4) for j in range(-3, 4)
                  for k in range(-3, 4)]) @ at
r = (cells[:, None, :] + np.asarray(calc.system.structure.positions)).reshape(-1, 3)
on = ((abs(r @ [1.0, -1.0, 0.0]) < 1e-6) & (r @ e1 > -1e-6) & (r @ e1 < u[-1] + 1e-6)
      & (r @ e2 > -1e-6) & (r @ e2 < v[-1] + 1e-6))
save(u=u, v=v, density=image, atoms=np.stack([r[on] @ e1, r[on] @ e2], axis=1),
     total_energy=scf.total_energy)
