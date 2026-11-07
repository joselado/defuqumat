"""Timing pair (1), the reference side: pw.x on the 6x1x1 graphene supercell at zero field.

The same physics as route P's SCF at p/q = 1/6 with the field switched off: the magnetic cell
(6 a1 x a2 x a3), C.pbe-hgh at 50 Ry, mv smearing 0.01 Ry, the 2 x 12 x 1 mesh of route P's
magnetic zone. Run twice, with pw.x's symmetry and with nosym/noinv (route P has no point
group under the field). Usage: python mag_pwx_graphene6.py <pw.x> <pseudo_dir> <workdir>
"""
import subprocess
import sys
import time
from pathlib import Path

pwx, pseudo, work = sys.argv[1], sys.argv[2], Path(sys.argv[3])
a = 2.46
atoms = "\n".join(f" C {(i + x) / 6:.10f} {y:.10f} 0.0" for i in range(6)
                  for x, y in ((0.0, 0.0), (1 / 3, 2 / 3)))
for label, extra in (("sym", ""), ("nosym", "  nosym = .true., noinv = .true.\n")):
    d = work / f"pwx-{label}"
    (d / "tmp").mkdir(parents=True, exist_ok=True)
    text = f"""&control
  calculation = 'scf', prefix = 'g6', outdir = '{d / 'tmp'}', pseudo_dir = '{pseudo}'
/
&system
  ibrav = 0, nat = 12, ntyp = 1, ecutwfc = 50.0,
  occupations = 'smearing', smearing = 'mv', degauss = 0.01
{extra}/
&electrons
  conv_thr = 1.0d-8, mixing_beta = 0.7
/
ATOMIC_SPECIES
 C 12.011 C.pbe-hgh.UPF
CELL_PARAMETERS angstrom
 {6 * a:.10f} 0.0 0.0
 {-a / 2:.10f} {a * 3 ** 0.5 / 2:.10f} 0.0
 0.0 0.0 15.0
ATOMIC_POSITIONS crystal
{atoms}
K_POINTS automatic
 2 12 1 0 0 0
"""
    (d / "g6.in").write_text(text)
    t0 = time.perf_counter()
    with open(d / "g6.in") as fin, open(d / "pw.out", "w") as fout:
        status = subprocess.run([pwx], stdin=fin, stdout=fout, stderr=subprocess.STDOUT).returncode
    wall = time.perf_counter() - t0
    out = (d / "pw.out").read_text().splitlines()
    keep = [l for l in out if ("number of k points" in l or "convergence has been achieved" in l
                               or "PWSCF  " in l or l.startswith("!") or "FFT dimensions" in l
                               or "Kohn-Sham states" in l)]
    print(f"[{label}] exit {status}, wall {wall:.1f} s")
    for line in keep:
        print(f"[{label}]   {line.strip()}")
    sys.stdout.flush()
