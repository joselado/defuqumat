"""The equation of state of silicon, and the pressure computed in two ways."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

calc = Calculator.from_file("si-eos.in", pseudo_dir=PSEUDO)
at = np.asarray(calc.system.cell.at)

volume, energy, pressure = [], [], []
for scale in np.linspace(0.96, 1.04, 9):          # the lattice constant, scaled
    strained = calc.with_cell(at * scale)
    volume.append(float(strained.system.cell.volume))
    energy.append(strained.get_scf().total_energy)
    pressure.append(strained.get_stress().pressure_kbar)
    print(f"V = {volume[-1]:7.2f} bohr^3   E = {energy[-1]:.8f} Ry   "
          f"P = {pressure[-1]:8.2f} kbar", flush=True)

save(volume=np.array(volume), energy=np.array(energy), pressure=np.array(pressure))
