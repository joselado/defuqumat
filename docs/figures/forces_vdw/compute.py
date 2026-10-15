"""The binding curve of bilayer graphene, with and without the D2 correction."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

calc = Calculator.from_file("bilayer.in", pseudo_dir=PSEUDO)
height = float(calc.system.cell.at[2, 2])          # bohr
positions = np.array(calc.system.structure.positions)

separations = np.array([5.4, 5.7, 6.0, 6.3, 6.6, 7.0, 7.5, 8.0, 9.0, 10.0])   # bohr
total, dispersion = [], []
for d in separations:
    positions[:2, 2] = (height - d) / 2             # the lower layer
    positions[2:, 2] = (height + d) / 2             # the upper layer
    scf = calc.with_positions(positions).get_scf()
    total.append(scf.total_energy)
    dispersion.append(scf.energy_terms["dispersion"])
    print(f"d = {d:5.2f} bohr   E = {total[-1]:.8f} Ry   E_disp = {dispersion[-1]:.8f} Ry",
          flush=True)

# D2 does not change the density, so PBE alone is the total minus the pair sum
save(separation=separations, total=np.array(total), dispersion=np.array(dispersion))
