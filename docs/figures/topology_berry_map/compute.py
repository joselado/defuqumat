"""The Berry curvature of AlAs and of silicon on the plane k3 = 0."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

data = {}
for name in ("alas", "si"):
    calc = Calculator.from_file(f"{name}.in", pseudo_dir=PSEUDO)
    omega = calc.get_berry_curvature(shape=(24, 24), nbnd=20, method="kubo")
    b = np.asarray(calc.system.cell.bg)             # b1, b2, b3 in 1/bohr
    area = np.linalg.norm(np.cross(b[0], b[1]))     # the area of the plane
    data[name] = omega.curvature / area             # Omega in bohr^2
    data[name + "_chern"] = calc.get_chern(shape=(24, 24))
    print(name, "max |Omega| %.4f bohr^2, C = %.2e"
          % (np.abs(data[name]).max(), data[name + "_chern"]), flush=True)

save(**data)
