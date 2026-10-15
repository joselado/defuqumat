"""The dipole of CO against the field, and where the charge goes."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

text = open("co.in").read()
fields = np.array([-0.004, -0.002, 0.0, 0.002, 0.004])       # Ha a.u.

dipole, force, density = [], [], []
for eamp in fields:
    calc = Calculator.from_text(text.replace("eamp=0.001", f"eamp={eamp}"),
                                pseudo_dir=PSEUDO)
    scf = calc.get_scf()
    dipole.append(scf.efield_dipole)                          # e bohr
    force.append(calc.get_forces().forces[0, 2])              # on O, Ry/bohr
    density.append(np.asarray(scf.density).sum(axis=0).mean(axis=(0, 1)))
    print(f"E = {eamp:+.3f}   p = {dipole[-1]:+.5f} e bohr   F(O) = {force[-1]:+.6f}",
          flush=True)

length = float(calc.system.cell.at[2, 2])
save(fields=fields, dipole=np.array(dipole), force=np.array(force),
     z=np.arange(len(density[0])) * length / len(density[0]), density=np.array(density))
