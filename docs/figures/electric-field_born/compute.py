"""The polarization of AlAs against the displacement of each sublattice."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

alas = Calculator.from_file("alas.in", pseudo_dir=PSEUDO)
positions = np.asarray(alas.system.structure.positions)    # bohr
b = np.asarray(alas.system.cell.bg)                         # reciprocal vectors, 1/bohr
u = np.linspace(-0.08, 0.08, 5)                             # bohr, along x
strings = (4, 8)

# the dipole of the cell along x, Omega P_x, from Omega P . b = 2 pi phase
dipole = np.zeros((2, len(strings), len(u)))               # (atom, strings, u)
for atom in (0, 1):
    for i, du in enumerate(u):
        moved = positions.copy()
        moved[atom, 0] += du
        shifted = alas.with_positions(moved)
        for j, n in enumerate(strings):
            pol = shifted.get_polarization(transverse=(n, n))
            dipole[atom, j, i] = 2 * np.pi * pol.total_phase / b[pol.gdir, 0]
        print(f"atom {atom}  u = {du:+.2f} bohr  dipole {dipole[atom, :, i]} e bohr",
              flush=True)

slopes = np.array([[np.polyfit(u, dipole[a, j], 1)[0] for j in range(len(strings))]
                   for a in (0, 1)])
print("Born charges (Al, As) on 4x4 and 8x8 strings:", slopes)
save(u=u, dipole=dipole, slopes=slopes, strings=np.array(strings))
