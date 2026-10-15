"""Tunnelling spectra above the two kinds of surface atom of graphite."""
import numpy as np
from defuqumat import Calculator
from defuqumat.units import RY_TO_EV
from common import PSEUDO, save

calc = Calculator.from_file("graphite.in", pseudo_dir=PSEUDO)
fermi = calc.get_scf().fermi_energy

# the tip 3.5 bohr above the atom A (0, 0) and above the atom B (2/3, 1/3)
tips = [[0, 0, 0.85], [2 / 3, 1 / 3, 0.85]]
sts = calc.get_sts(energies=fermi + np.linspace(-0.3, 0.3, 121), tip=tips,
                   width=0.02, grid=(24, 24, 1), nbnd=16)

save(bias=sts.bias_axis * RY_TO_EV, didv=sts.values / RY_TO_EV,
     dos=sts.integral / RY_TO_EV)
