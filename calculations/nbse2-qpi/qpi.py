"""1H-NbSe2: quasiparticle interference around one impurity, by the T-matrix.

The ground state is calculations/nbse2-fermi-surface/nbse2.in (SG15 PBE, 60 Ry,
the structure of elkpy's slides). The bands are solved once on the whole
N x N grid, which is the impurity's supercell, and every QPI map below is
post-processing of those states: two impurity sites (the Nb atom and the
upper Se), two broadenings, three energies around E_F. The tip plane is
3.5 A above the upper Se and the exit plane 3.5 A below the lower Se, as in
elkpy's tunnelling slides.

Usage:  python3 qpi.py <input.in> <N> <nbnd> <out.npz>
"""
import sys
import time
from pathlib import Path

import numpy as np

from defuqumat import Calculator

HERE = Path(__file__).resolve().parent
PSEUDO = HERE.parents[1] / "tests" / "data" / "pseudo"
C_HEIGHT = 18.0          # A
SE_OFFSET = 1.68036      # A, the Se planes from the Nb plane
DISTANCE = 3.5           # A, tip and exit plane from the outer Se
TIP = 0.5 + (SE_OFFSET + DISTANCE) / C_HEIGHT
EXIT = 0.5 - (SE_OFFSET + DISTANCE) / C_HEIGHT
SITES = {"Nb": 0, "Se_top": 1}
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
