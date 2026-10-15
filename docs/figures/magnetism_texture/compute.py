"""A 90-degree helix on a hydrogen chain: the moment on each site."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

calc = Calculator.from_file("h4-helix.in", pseudo_dir=PSEUDO)
scf = calc.get_scf()
z = np.asarray(calc.system.structure.positions)[:, 2]        # bohr

save(z=z, moments=np.array(scf.site_moments),
     cell_moment=np.array(scf.magnetization_vector), length=float(calc.system.cell.at[2, 2]))
