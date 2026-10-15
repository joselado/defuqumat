"""The magnon of fcc nickel along Gamma-X, and the Stoner continuum."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

# a q-point must be a difference of two k-points, so the mesh sets the q-points
text = open("ni.in").read().replace(" 4 4 4 0 0 0", " 6 6 6 0 0 0")
nickel = Calculator.from_text(text, pseudo_dir=PSEUDO)
s = np.arange(4) / 6                                # (0, s, s): Gamma to X
qpoints = [(0.0, x, x) for x in s]
frequencies = np.linspace(0.0, 0.1, 101)            # Ry
disp = nickel.get_magnon_dispersion(qpoints, frequencies, nbnd=24, ecut_response=40.0,
                                    broadening=3e-3, goldstone_correction=True)
for q, w, lam in zip(qpoints, disp.energies_mev, disp.enhancements):
    print(f"q = {q}   magnon {w:.0f} meV   enhancement {lam:.4f}", flush=True)

alat = float(nickel.system.cell.alat)
save(q=s * 2 * np.pi / alat * 2 / 0.529177,        # |q| in 1/Angstrom, (0, s, s) = (0, 2s, 0) 2pi/a
     magnon=disp.energies_mev, enhancement=disp.enhancements,
     goldstone=disp.goldstone, moment=nickel.get_scf().magnetization,
     energies=frequencies * 13605.693,
     spectral=np.array([c.spectral_function for c in disp.susceptibilities]),
     stoner=np.array([c.kohn_sham_spectral_function for c in disp.susceptibilities]))
