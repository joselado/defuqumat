"""I118, second round: a local field, and the spectrum on a plane.

    PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 4-7 \
        python3 notes/incompat/I118-harness/harness_local.py

1. A ``LOCAL_MAGNETIC_FIELDS`` field on the one atom of ``h-sheet-field.in``
   (``System.atomic_b_field``, B_z = 0.01 Ry, the uniform field off): the stored
   states' image against a re-solve on the same k-set under the field, and with
   the field dropped from the Hamiltonian.
2. The tunnelling spectrum on a plane (the map at every energy), uniform field,
   at B = 0, 5e-4 and 1e-3 Ry: the exponents of the charge and the spin maps.
3. The two slow tests' re-solve numbers, through ``Calculator.get_stm``.
"""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as h  # noqa: E402

from defuqumat import Calculator  # noqa: E402
from defuqumat.workflows.stm import run_stm, run_sts  # noqa: E402

RESOLVE = dict(grid=(4, 4, 1), conv_thr=1e-11)
SHARED = dict(height=h.HEIGHT, shape=h.SHAPE)


def local(b):
    base = Calculator.from_file(h.CASE, pseudo_dir=h.PSEUDO)
    system = base.system.with_b_field((0.0, 0.0, 0.0))
    system = dataclasses.replace(system, atomic_b_field=((0.0, 0.0, b),))
    calc = Calculator(system, base.pseudos)
    return calc, calc.get_scf(conv_thr=1e-12)


calc, res = local(0.01)
field = res.magnetic_field
print(f"local: constraint {field.constraint!r}, E_F {res.fermi_energy:.10f}, "
      f"m {float(np.asarray(res.density)[0].mean() - np.asarray(res.density)[1].mean()) * float(calc.system.cell.volume):.5f}",
      flush=True)
q = run_stm(calc.system, calc.pseudos, res, calculation=calc.calculation, **SHARED)
s = run_stm(calc.system, calc.pseudos, res, calculation=calc.calculation,
            spin="up", **SHARED)
qr = run_stm(calc.system, calc.pseudos, res, calculation=calc.calculation,
             **SHARED, **RESOLVE)
sr = run_stm(calc.system, calc.pseudos, res, calculation=calc.calculation,
             spin="up", **SHARED, **RESOLVE)
print(f"local re-solve on: image charge {h.rel(qr.values, q.values):.3e}, spin "
      f"{h.rel(sr.values, s.values):.3e}; tip energy - E_F "
      f"{qr.energy - float(res.fermi_energy):+.3e}; field_scale {q.field_scale}",
      flush=True)
off = Calculator(dataclasses.replace(calc.system, atomic_b_field=()), calc.pseudos)
fieldless = SimpleNamespace(
    density=res.density, eigenvalues_by_spin=res.eigenvalues_by_spin,
    wavefunctions=res.wavefunctions, fermi_energy=res.fermi_energy,
    homo=res.homo, lumo=res.lumo, magnetic_field=None, field_scale=1.0)
qo = run_stm(off.system, off.pseudos, fieldless, **SHARED, **RESOLVE)
so = run_stm(off.system, off.pseudos, fieldless, spin="up", **SHARED, **RESOLVE)
print(f"local re-solve off: image charge {h.rel(qo.values, q.values):.3e}, spin "
      f"{h.rel(so.values, s.values):.3e}; tip energy - E_F "
      f"{qo.energy - float(res.fermi_energy):+.3e}", flush=True)

# 2. the spectrum on a plane
maps = {}
for b in (0.0, 5e-4, 1e-3):
    calc, res = h.calculator("collinear", b)
    if b == 0.0:
        energies = float(res.fermi_energy) + np.linspace(-0.1, 0.1, 5)
    charge = run_sts(calc.system, calc.pseudos, res, energies=energies,
                     height=h.HEIGHT, shape=(6, 6), calculation=calc.calculation)
    try:
        spin = run_sts(calc.system, calc.pseudos, res, energies=energies,
                       height=h.HEIGHT, shape=(6, 6), spin="up",
                       calculation=calc.calculation).values
    except NotImplementedError:
        spin = 0.5 * np.asarray(charge.values)
    maps[b] = (np.asarray(charge.values), np.asarray(spin))
    print(f"STS map B = {b}: shape {np.shape(charge.values)}, field_scale "
          f"{charge.field_scale}", flush=True)
d = {b: (h.rel(maps[b][0], maps[0.0][0]), h.rel(maps[b][1], maps[0.0][1]))
     for b in (5e-4, 1e-3)}
print(f"STS map distances: 5e-4 {d[5e-4][0]:.4e} / {d[5e-4][1]:.4e}, 1e-3 "
      f"{d[1e-3][0]:.4e} / {d[1e-3][1]:.4e}; exponents charge "
      f"{np.log2(d[1e-3][0] / d[5e-4][0]):.3f}, spin "
      f"{np.log2(d[1e-3][1] / d[5e-4][1]):.3f}", flush=True)

# 3. the slow test's numbers through the Calculator
from tests.regression import test_stm as t  # noqa: E402

c = t._in_a_field("collinear", 0.01)
scf_before = c.get_scf()
stored = c.get_stm(**SHARED)
stored_spin = c.get_stm(spin="up", **SHARED)
again = c.get_stm(**SHARED, **RESOLVE)
again_spin = c.get_stm(spin="up", **SHARED, **RESOLVE)
print(f"through Calculator: re-solve on charge "
      f"{t._relative(again.values, stored.values):.3e}, spin "
      f"{t._relative(again_spin.values, stored_spin.values):.3e}; tip energy - "
      f"E_F {again.energy - scf_before.fermi_energy:+.3e}; same SCF object "
      f"after the calls: {c.get_scf() is scf_before}", flush=True)
