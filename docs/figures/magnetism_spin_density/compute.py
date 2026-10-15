"""The spin density of nickel on a face of the cubic cell, and the moment of each orbital."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

nickel = Calculator.from_file("ni.in", pseudo_dir=PSEUDO)
scf = nickel.get_scf()
n = np.asarray(scf.density)                  # (up, down) on the FFT grid, 1/bohr^3
m = n[0] - n[1]                              # the spin density, mu_B/bohr^3
omega = float(nickel.system.cell.volume)
print(f"moment: {m.mean() * omega:.4f} from the grid, {scf.magnetization:.4f} from the SCF")

# the Fourier coefficients: m(r) = sum_G c_G exp(i G.r), the small ones dropped
c = np.fft.fftn(m).ravel() / m.size
grid = np.meshgrid(*[np.fft.fftfreq(k) * k for k in m.shape], indexing="ij")
g = np.stack([x.ravel() for x in grid], axis=1) @ np.asarray(nickel.system.cell.bg)
keep = abs(c) > 1e-7 * abs(c).max()
c, g = c[keep], g[keep]

# the (001) face of the cubic cell, z = 0, with atoms at the corners and the centre
alat = nickel.system.cell.alat
s = np.linspace(0.0, alat, 151)
image = np.array([(np.exp(1j * (x * g[:, 0] + s[:, None] * g[:, 1])) @ c).real
                  for x in s])

# the Lowdin moment of each orbital, majority minus minority
pdos = nickel.get_pdos()
charges = pdos.charges.charges_by_spin[:, 0]  # (spin, l) on the one atom
print("Lowdin moments s, p, d:", charges[0] - charges[1], "spilling", pdos.charges.spilling)
save(s=s, spin_density=image, orbital_moment=charges[0] - charges[1],
     moment=scf.magnetization, grid_moment=m.mean() * omega)
