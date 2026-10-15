"""Compute figure own-crystal_convergence: the cutoff and k-mesh tests of diamond.

The total energy against ecutwfc (ecutrho = 8 ecutwfc) at the file's 6x6x6
mesh, and against the mesh n x n x n at the file's 35 Ry. Writes data.npz.
    bash docs/figures/run.sh own-crystal_convergence
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import common

from defuqumat import Calculator

PSEUDO = os.path.join(HERE, "..", "..", "..", "tests", "data", "pseudo")
params = common.start(dict(input="diamond.in", cutoffs=[20, 25, 30, 35, 40, 50, 60],
                           meshes=[2, 4, 6, 8, 10]))
text = open(os.path.join(HERE, "diamond.in")).read()


def energy(text):
    return Calculator.from_text(text, pseudo_dir=PSEUDO).get_scf().total_energy


cutoff = []
for ecut in params["cutoffs"]:
    new = text.replace("ecutwfc = 35.0, ecutrho = 280.0",
                       "ecutwfc = %.1f, ecutrho = %.1f" % (ecut, 8 * ecut))
    cutoff.append(energy(new))
    print("ecutwfc %4.0f Ry   E = %.8f Ry" % (ecut, cutoff[-1]), flush=True)

mesh = []
for n in params["meshes"]:
    mesh.append(energy(text.replace(" 6 6 6 0 0 0", " %d %d %d 0 0 0" % (n, n, n))))
    print("mesh %2d   E = %.8f Ry" % (n, mesh[-1]), flush=True)

common.save(__file__, params, cutoffs=np.array(params["cutoffs"], float),
            e_cutoff=np.array(cutoff), meshes=np.array(params["meshes"]),
            e_mesh=np.array(mesh))
