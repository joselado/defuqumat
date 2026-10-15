"""FeO with U = 4.3 eV and with U close to zero: the DOS and the occupations."""
import numpy as np
from defuqumat import Calculator
from defuqumat.units import RY_TO_EV
from common import PSEUDO, save

text = open("feo.in").read()
out = {}
for name, t in (("U", text), ("noU", text.replace("-3d 4.3", "-3d 1.d-8"))):
    calc = Calculator.from_text(t, pseudo_dir=PSEUDO)
    scf = calc.get_scf(max_iterations=200)
    dos = calc.get_dos(grid=(6, 6, 6), nbnd=24)
    ns = np.asarray(scf.ns)                       # (spin, Fe site, 5, 5)
    out[name + "_e"] = (dos.energies - dos.fermi_energy) * RY_TO_EV
    out[name + "_dos"] = dos.total_dos / RY_TO_EV
    out[name + "_occ"] = np.linalg.eigvalsh(ns).ravel()     # natural occupations
    out[name + "_energy"] = scf.total_energy
    print(name, f"E = {scf.total_energy:.9f} Ry in {scf.iterations} iterations", flush=True)

save(**out)
