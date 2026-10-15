"""Hydrogen started from the Elk density, and from atomic charges."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

calc = Calculator.from_file("h.in", pseudo_dir=PSEUDO)
seed = calc.get_elk_seed("elk")
seeded = calc.get_scf(starting_density=seed)
plain = Calculator.from_file("h.in", pseudo_dir=PSEUDO).get_scf()

# the grid points on the line from the atom along x, and along the diagonal
n = seed.shape[1]
i = np.arange(n)
save(x=i * 3.0 / n, d=i * 3.0 * np.sqrt(3.0) / n,
     seed_x=seed[0, i, 0, 0], scf_x=seeded.density[0, i, 0, 0],
     seed_d=seed[0, i, i, i], scf_d=seeded.density[0, i, i, i],
     energies=[seeded.total_energy, plain.total_energy],
     iterations=[seeded.iterations, plain.iterations])
