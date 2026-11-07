"""Bit-identity probe of the Davidson precondition hook (foundation item 7).

Runs QE's test-suite pw_scf/scf.in (Si, 2 atoms, 12 Ry, 2 k) on the k-point path, the
same cell at K_POINTS gamma (the half-sphere, real-arithmetic branch of the solver) and
the same cell with noncolin (the spinor operator), and prints every total energy and a
digest of the eigenvalues with repr, so two runs (before and after the hook) can be
compared to the last digit. Usage, from the worktree root: python mag_si_energy.py
"""
import hashlib
import sys

import numpy as np

from defuqumat import Calculator

BASE = """ &control
    calculation = 'scf'
 /
 &system
    ibrav=2, celldm(1) =10.20,
    nat=2, ntyp=1,
    ecutwfc=12.0{extra}
 /
 &electrons
    conv_thr = 1.0d-10
 /
ATOMIC_SPECIES
 Si  28.086  Si.pz-vbc.UPF
ATOMIC_POSITIONS (alat)
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
{kpoints}
"""
KLIST = """K_POINTS
  2
   0.250000  0.250000  0.250000   1.00
   0.250000  0.250000  0.750000   3.00"""

CASES = {
    "kpoints": dict(extra="", kpoints=KLIST),
    "gamma": dict(extra=",\n    nosym=.true.", kpoints="K_POINTS gamma"),
    "noncolin": dict(extra=",\n    noncolin=.true.", kpoints=KLIST),
}

for name, fill in CASES.items():
    calc = Calculator.from_text(BASE.format(**fill), pseudo_dir="tests/data/pseudo")
    scf = calc.get_scf()
    eig = np.ascontiguousarray(np.asarray(scf.eigenvalues, dtype=np.float64))
    digest = hashlib.sha256(eig.tobytes()).hexdigest()[:16]
    print(f"{name}: converged {scf.converged} iterations {scf.iterations} "
          f"total_energy {scf.total_energy!r} eig_sha {digest}", flush=True)
sys.exit(0)
