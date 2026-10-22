"""I118: the STM image and the tunnelling spectrum of a sample in a field.

Run from the repository root:

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 taskset -c 4-7 \
        python3 notes/incompat/I118-harness/harness.py [collinear|noncollinear]

Cell: ``tests/data/qe/h-sheet-field.in`` (a paramagnetic hydrogen sheet), with
the uniform field along z set by ``System.with_b_field``. Three checks:

1. **The re-solve under the field reproduces the stored states' image** on the
   same k-set, and the same re-solve with the field dropped from the
   Hamiltonian does not (the "off" arm). The Fermi level of the re-solve
   against the SCF's.
2. **The weak-field limit**: the image and the spectrum at B against B = 0, at
   B = 0.0005 to 0.01 Ry, doubling. The charge channel of a paramagnet is even in B
   (differences fall as B^2) and the spin channel odd (as B).
3. **The floor**: the B = 0.01 image at two ``conv_thr``.
"""

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from defuqumat import Calculator
from defuqumat.workflows.stm import run_stm, run_sts

ROOT = Path(__file__).resolve().parents[3]
CASE = ROOT / "tests" / "data" / "qe" / "h-sheet-field.in"
PSEUDO = ROOT / "tests" / "data" / "pseudo"
HEIGHT = 0.72               # crystal, about 2.6 bohr above the sheet (c = 12)
SHAPE = (12, 12)
FIELDS = (0.0005, 0.001, 0.002, 0.004, 0.01)


def calculator(regime, b, conv_thr=1e-12):
    base = Calculator.from_file(CASE, pseudo_dir=PSEUDO)
    system = base.system
    if regime == "noncollinear":
        system = system.with_spin(4)
    system = system.with_b_field((0.0, 0.0, b))
    calc = Calculator(system, base.pseudos)
    result = calc.get_scf(conv_thr=conv_thr)
    return calc, result


def spin_of(regime):
    return "up" if regime == "collinear" else (0.0, 0.0, 1.0)


def images(calc, result, regime, **options):
    charge = run_stm(calc.system, calc.pseudos, result, height=HEIGHT,
                     shape=SHAPE, calculation=calc.calculation, **options)
    try:
        spin = run_stm(calc.system, calc.pseudos, result, height=HEIGHT,
                       shape=SHAPE, spin=spin_of(regime),
                       calculation=calc.calculation, **options)
    except NotImplementedError:
        # a spinor run at B = 0 carries no magnetization (nspin_mag = 1) and
        # refuses a spin tip; its spin image is half the charge exactly
        # (project_spin's convention for m = 0)
        spin = SimpleNamespace(values=0.5 * np.asarray(charge.values))
    return charge, spin


def spectrum(calc, result, regime, energies):
    tip = np.array([[0.0, 0.0, HEIGHT], [0.5, 0.5, HEIGHT]])
    charge = run_sts(calc.system, calc.pseudos, result, energies=energies,
                     tip=tip, calculation=calc.calculation)
    try:
        spin = np.asarray(run_sts(
            calc.system, calc.pseudos, result, energies=energies, tip=tip,
            spin=spin_of(regime), calculation=calc.calculation).values)
    except NotImplementedError:
        spin = 0.5 * np.asarray(charge.values)
    return np.asarray(charge.values), spin


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def main(regime):
    t0 = time.time()
    print(f"# regime {regime}", flush=True)
    calc0, res0 = calculator(regime, 0.0)
    q0, s0 = images(calc0, res0, regime)
    ef0 = float(res0.fermi_energy)
    energies = ef0 + np.linspace(-0.1, 0.1, 9)
    dq0, ds0 = spectrum(calc0, res0, regime, energies)
    print(f"B = 0: E_F {ef0:.10f}, image max {np.max(q0.values):.6e}, "
          f"field_scale {q0.field_scale}", flush=True)

    rows = []
    for b in FIELDS:
        calc, res = calculator(regime, b)
        q, s = images(calc, res, regime)
        dq, ds = spectrum(calc, res, regime, energies)
        rows.append((b, rel(q.values, q0.values), rel(s.values, s0.values),
                     rel(dq, dq0), rel(ds, ds0),
                     float(res.fermi_energy) - ef0))
        print(f"B = {b}: E_F - E_F(0) {rows[-1][5]:+.3e}; image charge "
              f"{rows[-1][1]:.4e}, spin {rows[-1][2]:.4e}; spectrum charge "
              f"{rows[-1][3]:.4e}, spin {rows[-1][4]:.4e}; field_scale "
              f"{q.field_scale}", flush=True)
        print(f"   charge/B^2 {rows[-1][1]/b**2:.4e}, spin/B {rows[-1][2]/b:.4e}", flush=True)
    for (b1, *r1), (b2, *r2) in zip(rows, rows[1:]):
        k = np.log(b2 / b1)
        print(f"exponents {b1}->{b2}: image charge {np.log(r2[0]/r1[0])/k:.3f}, "
              f"spin {np.log(r2[1]/r1[1])/k:.3f}; spectrum charge "
              f"{np.log(r2[2]/r1[2])/k:.3f}, spin {np.log(r2[3]/r1[3])/k:.3f}; "
              f"E_F {np.log(abs(r2[4]/r1[4]))/k:.3f}", flush=True)

    # 1. the re-solve under the field against the stored states, at B = 0.01
    b = FIELDS[-1]
    calc, res = calculator(regime, b)
    q, s = images(calc, res, regime)
    qr, sr = images(calc, res, regime, grid=(4, 4, 1), conv_thr=1e-11)
    print(f"re-solve on: image charge {rel(qr.values, q.values):.3e}, spin "
          f"{rel(sr.values, s.values):.3e}; tip energy - E_F "
          f"{qr.energy - float(res.fermi_energy):+.3e}", flush=True)
    # the off arm: the same density, the field dropped from the Hamiltonian
    # Collinear only: a spinor system without the field has nspin_mag = 1
    # (starting_magnetization = 0), so its re-solve cannot even hold the
    # field run's four-channel density, and the arm reads nothing.
    if regime == "collinear":
        off_system = calc.system.with_b_field((0.0, 0.0, 0.0))
        off_result = SimpleNamespace(**{
            name: getattr(res, name) for name in
            ("density", "eigenvalues_by_spin", "wavefunctions", "fermi_energy",
             "homo", "lumo")})
        off_result.magnetic_field = None
        off_result.field_scale = 1.0
        off = Calculator(off_system, calc.pseudos)
        qo = run_stm(off.system, off.pseudos, off_result, height=HEIGHT,
                     shape=SHAPE, grid=(4, 4, 1), conv_thr=1e-11)
        try:
            so = run_stm(off.system, off.pseudos, off_result, height=HEIGHT,
                         shape=SHAPE, grid=(4, 4, 1), conv_thr=1e-11,
                         spin=spin_of(regime))
            off_spin = f"{rel(so.values, s.values):.3e}"
        except NotImplementedError:
            # the spinor system without the field has nspin_mag = 1: its
            # re-solve carries no magnetization at all, so only the charge reads
            off_spin = "n/a (nspin_mag = 1 without the field)"
        print(f"re-solve off: image charge {rel(qo.values, q.values):.3e}, spin "
              f"{off_spin}; tip energy - E_F "
              f"{qo.energy - float(res.fermi_energy):+.3e}", flush=True)
    # the entry's alternative: the field states read at the field-free level
    qa = run_stm(calc.system, calc.pseudos, res, height=HEIGHT, shape=SHAPE,
                 energy=ef0, calculation=calc.calculation)
    print(f"tip at E_F(0) instead: image charge {rel(qa.values, q.values):.3e}",
          flush=True)

    # 3. the floor, at B = 0 and at B = 0.01
    for bf, ref_q, ref_s in ((0.0, q0, s0), (b, q, s)):
        for thr in (1e-10, 1e-11):
            calc, res = calculator(regime, bf, conv_thr=thr)
            qf, sf = images(calc, res, regime)
            print(f"floor (conv_thr {thr:.0e} vs 1e-12 at B = {bf}): image "
                  f"charge {rel(qf.values, ref_q.values):.3e}, spin "
                  f"{rel(sf.values, ref_s.values):.3e}", flush=True)
    print(f"# wall {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    for regime in (sys.argv[1:] or ["collinear", "noncollinear"]):
        main(regime)
