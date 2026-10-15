"""Silicon relaxing back to its lattice sites: energy and force at each step."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

calc = Calculator.from_file("si-displaced.in", pseudo_dir=PSEUDO)
relax = calc.get_relax()

alat = calc.system.cell.alat
steps = relax.steps
save(step=np.array([s.index for s in steps]),
     energy=np.array([s.total_energy for s in steps]),
     max_force=np.array([s.max_force for s in steps]),
     x=np.array([(s.positions[1, 0] - s.positions[0, 0]) / alat for s in steps]))
