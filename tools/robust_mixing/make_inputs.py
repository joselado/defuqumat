"""Write the robust-mixing test bed's inputs that are not committed files.

python3 make_inputs.py <out_dir>

* ``al-v32``: ``benchmarks/al-slab.in``'s five Al(100) layers with 32 bohr of
  vacuum, positions in bohr (the ``tools/vacuum_sweep`` geometry).
* ``al8-rattled``: the conventional fcc Al cell doubled along z (eight atoms,
  7.5 x 7.5 x 15 bohr), every atom moved by a fixed pseudo-random amount of up
  to 0.15 bohr, Gaussian smearing of 0.01 Ry: Herbst and Levitt's elongated
  rattled aluminium (arXiv:2109.14018, Table I), run without Kerker.
* ``fe-unstable-k4``: ``benchmarks/fe-unstable.in`` on a 4x4x4 mesh, where
  ``PLAN.md`` P129 records the run landing on M = 0.
* ``ni-mae-gauss``: ``ni-tetragonal-relaxed-mae-paw.in`` with a Gaussian of the
  same width in place of ``mv``, the variational counterpart of the same cell.
* ``h4-pbe``: ``tests/data/qe/h4-noncolin-force.in`` under PBE, recorded as a limit
  cycle at 5e-6 (``notes/MAGNETISM-NEXT.md``).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)


def write(name, lines):
    (out / f"{name}.in").write_text("\n".join(lines) + "\n")


# Al(100), five layers, 15 bohr of metal, 32 bohr of vacuum.
a, c = 5.303301, 47.0
z = [(f - 0.5) * 31.0 for f in (0.258065, 0.379032, 0.5, 0.620968, 0.741935)]
xy = [(0.0, 0.0), (0.5, 0.5), (0.0, 0.0), (0.5, 0.5), (0.0, 0.0)]
write("al-v32", [
    "&control", "  calculation = 'scf'", "/",
    "&system", "  ibrav = 0", "  nat = 5", "  ntyp = 1", "  ecutwfc = 12.0",
    "  occupations = 'smearing'", "  smearing = 'gaussian'", "  degauss = 0.05", "/",
    "&electrons", "  conv_thr = 1.0d-8", "  electron_maxstep = 200", "/",
    "ATOMIC_SPECIES", "  Al 26.98 Al.pz-vbc.UPF",
    "CELL_PARAMETERS bohr", f"  {a:.10f} 0.0 0.0", f"  0.0 {a:.10f} 0.0", f"  0.0 0.0 {c:.10f}",
    "ATOMIC_POSITIONS bohr",
    *[f"  Al {f1 * a:.10f} {f2 * a:.10f} {0.5 * c + zz:.10f}" for (f1, f2), zz in zip(xy, z)],
    "K_POINTS automatic", "  2 2 1 0 0 0",
])

# Eight-atom elongated fcc Al, rattled.
alat = 7.5
basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
sites = np.concatenate([basis, basis + [0, 0, 1]]) * alat
rattle = np.random.default_rng(20261015).uniform(-0.15, 0.15, sites.shape)
write("al8-rattled", [
    "&control", "  calculation = 'scf'", "/",
    "&system", "  ibrav = 0", "  nat = 8", "  ntyp = 1", "  ecutwfc = 15.0", "  nosym = .true.",
    "  occupations = 'smearing'", "  smearing = 'gaussian'", "  degauss = 0.01", "/",
    "&electrons", "  conv_thr = 1.0d-8", "  electron_maxstep = 200", "/",
    "ATOMIC_SPECIES", "  Al 26.98 Al.pz-vbc.UPF",
    "CELL_PARAMETERS bohr", f"  {alat} 0.0 0.0", f"  0.0 {alat} 0.0", f"  0.0 0.0 {2 * alat}",
    "ATOMIC_POSITIONS bohr",
    *[f"  Al {x:.10f} {y:.10f} {zz:.10f}" for x, y, zz in sites + rattle],
    "K_POINTS automatic", "  4 4 2 0 0 0",
])

text = (ROOT / "benchmarks/fe-unstable.in").read_text()
assert " 2 2 2 0 0 0" in text
write("fe-unstable-k4", [text.replace(" 2 2 2 0 0 0", " 4 4 4 0 0 0").rstrip("\n")])

text = (ROOT / "tests/data/qe/h4-noncolin-force.in").read_text()
import re
head, sep, tail = re.split(r"(&system)", text, maxsplit=1, flags=re.IGNORECASE)
write("h4-pbe", [(head + sep + "\n    input_dft = 'pbe'," + tail).rstrip("\n")])

text = (ROOT / "tests/data/qe/ni-tetragonal-relaxed-mae-paw.in").read_text()
assert "smearing = 'mv'" in text
write("ni-mae-gauss", [text.replace("smearing = 'mv'", "smearing = 'gaussian'").rstrip("\n")])
