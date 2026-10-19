"""I88: pw.x's own energy curvature of al2-metal.in under strain, as an independent sign check.

pw.x rebuilds the plane-wave sphere at every strain, so this carries the Pulay
error the frozen-sphere comparison does not; it is a check of sign and size,
not of the last digits. Usage: pwfd.py <pw.x> <pseudo_dir> <workdir>
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

PW, PSEUDO, WORK = sys.argv[1], sys.argv[2], Path(sys.argv[3])
WORK.mkdir(parents=True, exist_ok=True)
CELL = np.array([[-0.50, 0.00, 0.50], [0.00, 0.50, 0.50], [-1.00, 1.00, 0.00]])
RY_BOHR3_TO_GPA = 14710.507848260711
TEMPLATE = """ &control
    calculation='scf', pseudo_dir='{pseudo}', outdir='./tmp-{tag}', prefix='al2{tag}'
 /
 &system
    ibrav=0, celldm(1)=7.50, nat=2, ntyp=1,
    ecutwfc=15.0, nosym=.true., noinv=.true.,
    occupations='smearing', smearing='marzari-vanderbilt', degauss=0.05
 /
 &electrons
    conv_thr = 1.0d-12
 /
CELL_PARAMETERS alat
{cell}
ATOMIC_SPECIES
 Al  26.98 Al.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Al  0.00  0.00  0.00
 Al  0.00  0.00  0.50
K_POINTS automatic
 4 4 2 0 0 0
"""


def energy(strain, tag):
    cell = CELL @ (np.eye(3) + strain).T
    text = TEMPLATE.format(pseudo=PSEUDO, tag=tag,
                           cell="\n".join("  " + "  ".join(f"{x:.12f}" for x in row) for row in cell))
    (WORK / f"{tag}.in").write_text(text)
    out = subprocess.run([PW, "-in", f"{tag}.in"], cwd=WORK, capture_output=True, text=True).stdout
    (WORK / f"{tag}.out").write_text(out)
    value = float(re.findall(r"!\s+total energy\s+=\s+(-?\d+\.\d+)", out)[-1])
    print(f"  {tag}: {value:.10f} Ry", flush=True)
    return value


volume = abs(np.linalg.det(CELL)) * 7.50 ** 3
e0 = energy(np.zeros((3, 3)), "e0")
print("E0", e0, "against reference.out.al2-metal:342 -8.33210380; volume", volume, flush=True)
for pair in ((0, 0), (0, 1)):
    tangent = np.zeros((3, 3))
    tangent[pair] += 0.5
    tangent[pair[::-1]] += 0.5
    for h in (0.005, 0.01):
        values = {m: energy(m * h * tangent, f"p{pair[0]}{pair[1]}_{m:+d}_{h}")
                  for m in (-2, -1, 1, 2)}
        values[0] = e0
        five = (-values[2] + 16 * values[1] - 30 * e0 + 16 * values[-1] - values[-2]) / (12 * h * h)
        print(f"pw.x {pair} h={h}: five-point C = {five / volume * RY_BOHR3_TO_GPA:.4f} GPa", flush=True)
