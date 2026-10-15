"""The projected DOS of silicon (ultrasoft), by angular momentum."""
import numpy as np
from defuqumat import Calculator
from defuqumat.units import RY_TO_EV
from common import PSEUDO, save

silicon = Calculator.from_file("si-us.in", pseudo_dir=PSEUDO)
pdos = silicon.get_pdos(grid=(12, 12, 12), nbnd=8, scheme="tetrahedra")

save(energies=(pdos.energies - pdos.fermi_energy) * RY_TO_EV,
     total=pdos.total.dos_ev,
     s=pdos.select(l="s") / RY_TO_EV, p=pdos.select(l="p") / RY_TO_EV,
     charges=np.asarray(pdos.charges.total), spilling=pdos.charges.spilling)
