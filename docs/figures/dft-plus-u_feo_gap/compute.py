"""The DOS of antiferromagnetic FeO with U = 4.3 eV and with U close to zero."""
import numpy as np
from defuqumat import Calculator
from defuqumat.units import RY_TO_EV
from common import PSEUDO, save

text = open("feo.in").read()
runs = {"U": text, "noU": text.replace("-3d 4.3", "-3d 1.d-8")}
out = {}
for name, t in runs.items():
    calc = Calculator.from_text(t, pseudo_dir=PSEUDO)
    scf = calc.get_scf(max_iterations=200)
    dos = calc.get_dos(grid=(6, 6, 6), nbnd=24)
    out[name + "_e"] = (dos.energies - dos.fermi_energy) * RY_TO_EV
    out[name + "_dos"] = dos.dos / RY_TO_EV           # (2, nE): up, down
    out[name + "_moments"] = np.array(scf.site_moments)
    out[name + "_energy"] = scf.total_energy
    print(name, f"E = {scf.total_energy:.9f} Ry in {scf.iterations} iterations", flush=True)

save(**out)
