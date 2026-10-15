"""The total energy of diamond against the cutoff and against the k-mesh."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

text = open("diamond.in").read()       # 35 Ry and a 6x6x6 mesh


def energy(text):
    return Calculator.from_text(text, pseudo_dir=PSEUDO).get_scf().total_energy


cutoffs = [20, 25, 30, 35, 40, 50, 60]     # ecutwfc in Ry, with ecutrho = 8 ecutwfc
e_cutoff = [energy(text.replace("ecutwfc = 35.0, ecutrho = 280.0",
                                f"ecutwfc = {c:.1f}, ecutrho = {8 * c:.1f}"))
            for c in cutoffs]

meshes = [2, 4, 6, 8, 10]                  # n x n x n at 35 Ry
e_mesh = [energy(text.replace(" 6 6 6 0 0 0", f" {n} {n} {n} 0 0 0"))
          for n in meshes]

save(cutoffs=np.array(cutoffs, float), e_cutoff=np.array(e_cutoff),
     meshes=np.array(meshes), e_mesh=np.array(e_mesh))
