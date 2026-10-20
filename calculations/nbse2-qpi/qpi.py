"""NbSe2 slabs: quasiparticle interference around one impurity, by the T-matrix.

The monolayer is calculations/nbse2-fermi-surface/nbse2.in (SG15 PBE, 60 Ry,
the structure of elkpy's slides) and the bilayer is nbse2-bilayer.in beside
this file, built from it. The bands are solved once on the whole N x N grid,
which is the impurity's supercell, and every QPI map below is post-processing
of those states: two impurity sites (the Nb and the Se of the top layer, the
highest of each), two broadenings, three energies around E_F. The tip plane is
3.5 A above the highest Se and the exit plane 3.5 A below the lowest Se, read
off the structure, as in elkpy's tunnelling slides.

Usage:  python3 qpi.py <input.in> <N> <nbnd> <out.npz>
"""
import sys
import time
from pathlib import Path

import numpy as np

from defuqumat import Calculator

HERE = Path(__file__).resolve().parent
PSEUDO = HERE.parents[1] / "tests" / "data" / "pseudo"
BOHR = 0.529177210903    # A
DISTANCE = 3.5           # A, tip and exit plane from the outer Se
POTENTIAL = 0.5          # Ry
SIGMA = 1.0              # bohr
BROADENINGS = (0.005, 0.01)       # Ry
OFFSETS = (-0.02, 0.0, 0.02)      # Ry from E_F
SHAPE = (4, 4)

source, n, nbnd, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
clock = time.time()
calculator = Calculator.from_file(source, pseudo_dir=PSEUDO)
fermi = calculator.get_scf().fermi_energy
print(f"SCF  E_F = {fermi:.6f} Ry  ({time.time() - clock:.0f} s)", flush=True)

cell = calculator.system.cell
height = float(cell.volume) / abs(np.linalg.det(np.asarray(cell.at)[:2, :2]))  # bohr
z = np.asarray(calculator.system.structure.positions_crystal(cell))[:, 2]
symbols = calculator.system.structure.symbols
se = [i for i, s in enumerate(symbols) if s == "Se"]
nb = [i for i, s in enumerate(symbols) if s == "Nb"]
TIP = z[se].max() + DISTANCE / BOHR / height
EXIT = z[se].min() - DISTANCE / BOHR / height
SITES = {"Nb": max(nb, key=lambda i: z[i]), "Se_top": max(se, key=lambda i: z[i])}
print(f"tip {TIP:.5f}, exit {EXIT:.5f}, sites {SITES}", flush=True)

clock = time.time()
states = calculator.get_qpi_bands(grid=(n, n, 1), nbnd=nbnd)
print(f"bands on the {n}x{n} grid, {nbnd} bands: {time.time() - clock:.0f} s",
      flush=True)
print(f"E_F of the grid {states.fermi_energy:.6f} Ry", flush=True)

energies = [states.fermi_energy + d for d in OFFSETS]
record = {"energies": np.array(energies), "offsets": np.array(OFFSETS),
          "broadenings": np.array(BROADENINGS), "fermi_scf": fermi,
          "fermi": states.fermi_energy, "tip": TIP, "exit": EXIT,
          "bg": np.asarray(calculator.system.cell.bg)[:2, :2],
          "eigenvalues": np.asarray(states.eigenvalues)}
for site, index in SITES.items():
    for eta in BROADENINGS:
        clock = time.time()
        qpi = calculator.get_qpi(POTENTIAL, TIP, impurity=index, sigma=SIGMA,
                                 energies=energies, readers=("ldos", "landauer"),
                                 exit_height=EXIT, shape=SHAPE, broadening=eta,
                                 states=states)
        key = f"{site}_eta{eta:g}"
        for reader in ("ldos", "landauer"):
            record[f"{key}_{reader}_q"] = qpi.transforms[reader]
            record[f"{key}_{reader}_map"] = qpi.maps[reader]
            record[f"{key}_{reader}_bg"] = qpi.backgrounds[reader]
        record[f"{key}_t"] = qpi.t_matrix
        record[f"{key}_wrap"] = qpi.wraparound
        record[f"{key}_contour"] = qpi.contour_points
        print(f"{key:16s} {time.time() - clock:6.0f} s  "
              f"|T| {np.abs(qpi.t_matrix).ravel().round(4)}  "
              f"wrap {qpi.wraparound.ravel().round(4)}  "
              f"contour {qpi.contour_points.ravel().round(0)}", flush=True)
        np.savez(out, **record)
print("done", flush=True)
