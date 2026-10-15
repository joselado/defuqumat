"""Compute figure dos_pdos: the projected DOS of silicon, by angular momentum.

Writes data.npz next to this file.
    bash docs/figures/run.sh dos_pdos
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common

from defuqumat import Calculator
from defuqumat.units import RY_TO_EV

PSEUDO = os.path.join(HERE, "..", "..", "..", "tests", "data", "pseudo")
params = common.start(dict(input="si-us.in", grid=12, scheme="tetrahedra"))

silicon = Calculator.from_file(os.path.join(HERE, "si-us.in"), pseudo_dir=PSEUDO)
n = params["grid"]
pdos = silicon.get_pdos(grid=(n, n, n), nbnd=8, scheme=params["scheme"])

zero = pdos.fermi_energy
common.save(__file__, params,
            energies=(pdos.energies - zero) * RY_TO_EV,
            total=pdos.total.dos_ev, s=pdos.select(l="s") / RY_TO_EV,
            p=pdos.select(l="p") / RY_TO_EV,
            charges=np.asarray(pdos.charges.total), spilling=pdos.charges.spilling)
