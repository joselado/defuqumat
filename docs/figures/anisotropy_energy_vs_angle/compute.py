"""The energy of tetragonal cobalt against the direction of its moment."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

scalar = Calculator.from_file("co.in", pseudo_dir=PSEUDO)      # no spin-orbit coupling
spinor = Calculator.from_file("co-soc.in", pseudo_dir=PSEUDO)  # with it

angles = np.arange(0, 91, 10)                       # from the c axis towards a
directions = [(np.sin(np.radians(a)), 0.0, np.cos(np.radians(a))) for a in angles]
ry_to_mev = 13605.693
curves = {}
for name, scale in (("soc", 1.0), ("off", 0.0)):    # off: coupling scaled to zero
    mae = scalar.get_anisotropy(spinor, directions=directions, soc_scale=scale)
    curves[name] = (mae.free_energies - mae.free_energies[0]) * ry_to_mev
    curves[name + "_band"] = mae.energies_mev
torque = scalar.get_torque(spinor)                  # the slope at 45 degrees

save(angles=angles, K_torque=torque.anisotropy_constant_mev, **curves)
