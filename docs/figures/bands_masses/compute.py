"""The two effective masses of the conduction valley of silicon, against its bands."""
import numpy as np
from defuqumat import Calculator
from defuqumat.system.kpoints import KPoints
from defuqumat.units import RY_TO_EV
from common import PSEUDO, save

calc = Calculator.from_file("si.in", pseudo_dir=PSEUDO)
cell = calc.system.cell
band = 4                                       # the lowest conduction band

# the minimum of the band on the line from Gamma to X, in units of 2 pi / a
line = KPoints.band_path([[0.7, 0.0, 0.0], [1.0, 0.0, 0.0]], [60, 1], cell, crystal=False)
e = np.asarray(calc.get_bands(kpoints=line, nbnd=8).eigenvalues)[:, band]
i = np.argmin(e)
x = np.linspace(0.7, 1.0, 61)
x0 = x[i] - 0.005 * (e[i + 1] - e[i - 1]) / (2 * (e[i + 1] - 2 * e[i] + e[i - 1]))
k0 = np.array([x0, 0.0, 0.0])

# the mass tensor at the minimum, in 1/m_e, cartesian
mass = calc.get_effective_mass(np.asarray(cell.k_to_crystal(k0)), nbnd=8)
inverse = mass.inverse_mass[band]
print(f"minimum at {x0:.4f} of Gamma-X, m_l = {1 / inverse[0, 0]:.3f}, "
      f"m_t = {1 / inverse[1, 1]:.3f} m_e")

# the bands through the minimum, along [100] (longitudinal) and [010] (transverse)
t = np.linspace(-0.08, 0.08, 41)               # 1/bohr
cuts = []
for direction in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]):
    ends = [k0 + t[0] / cell.tpiba * np.array(direction),
            k0 + t[-1] / cell.tpiba * np.array(direction)]
    cut = calc.get_bands(kpoints=KPoints.band_path(ends, [40, 1], cell, crystal=False),
                         nbnd=8)
    cuts.append(np.asarray(cut.eigenvalues)[:, band:] * RY_TO_EV)

e0 = mass.eigenvalues[band] * RY_TO_EV
save(t=t, longitudinal=cuts[0] - e0, transverse=cuts[1] - e0,
     inverse_mass=inverse, truncation=mass.truncation[band], x0=x0)
