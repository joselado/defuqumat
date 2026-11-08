"""Route U step 3, the gate: the free Landau problem on the projected stripe trials, run through
``run_magnetic_ultracell`` (not through fable's probe).

Graphene cell with c = 5 A (the 15 A vacuum's ladder fills any window below the trial), 50 Ry,
flux 1/10, N = 64, kmesh (2, 2), window + box + symmetric. Usage:

    python mag_route_u3_gate.py hermite        sigma 0.5, J = 1, 2, 3, window (3, 4), N1 16
    python mag_route_u3_gate.py chirp          sigma = l_B |b1| = 1.356, chirped, J = 1, window (5, 6), N1 24
    python mag_route_u3_gate.py chirp-flip     the same, field reversed (scale = -1), window (3, 4), N1 16
    python mag_route_u3_gate.py hermite-flip   sigma 0.5, J = 1, field reversed, window (3, 4), N1 16

Writes $MAG_OUT/gate-<mode>.npz (default /l/ladovj1/review/mag/u3). Energies in units of B_ph.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

from defuqumat import Calculator
from defuqumat.magnetic.conventions import MagneticCell
from defuqumat.magnetic.hybrid import StripeTrials
from defuqumat.magnetic.landau import free_landau_levels
from defuqumat.units import RY_TO_EV
from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

OUT = Path(os.environ.get("MAG_OUT", "/l/ladovj1/review/mag/u3"))
mode = sys.argv[1]
np.set_printoptions(linewidth=200)

calc = Calculator.from_keywords(
    lattice="hexagonal", a=2.46, c=5.0, atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
    pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=50 * RY_TO_EV, kmesh=(12, 12, 1), smearing=0.01 * RY_TO_EV, smearing_type="mv")
mc = MagneticCell.build(calc.system.cell, 1, 10)
levels, mult, labels = free_landau_levels(mc, 12, g3max=2)
ladder = np.repeat(levels, mult)

configs = {
    "hermite": [dict(trials=StripeTrials(J=J, sigma=0.5), window=(3.0, 4.0), N1=16) for J in (1, 2, 3)],
    "chirp": [dict(trials=StripeTrials(J=1, sigma=1.356, chirp=True), window=(5.0, 6.0), N1=24)],
    "chirp-flip": [dict(trials=StripeTrials(J=1, sigma=1.356, chirp=True), window=(3.0, 4.0), N1=16,
                        scale=-1.0, strict=False)],
    "hermite-flip": [dict(trials=StripeTrials(J=1, sigma=0.5), window=(3.0, 4.0), N1=16, scale=-1.0,
                          strict=False)],
}[mode]
saved = {}
for kw in configs:
    t0 = time.perf_counter()
    r = run_magnetic_ultracell(calc.system, calc.pseudos, None, flux=(1, 10), free=True, N=64,
                               kmesh=(2, 2), **kw)
    J = r.nbnd
    b = r.b_ph if r.b_ph > 0 else abs(r.b_ph)
    n = min(r.levels.shape[1], len(ladder))
    above = r.levels[:, :n] - ladder[None, :n]
    err = r.levels[:, :6] / abs(r.b_ph) - (2 * np.arange(6) + 1)
    print(f"\n== {mode} J {J} {r.trials[0]} window {r.window} N1 {r.n1} scale {kw.get('scale', 1.0)}: "
          f"B_ph {r.b_ph!r} Ry, {r.field_tesla:.1f} T, w {r.w}", flush=True)
    print(f"1 - projectability: max {r.projectability_deficit:.3e}, per trial "
          f"{np.array2string(1 - r.projectability.min(axis=(0, 1, 2)), precision=3)}", flush=True)
    for ik, kap in enumerate(r.kappa):
        print(f"k ({kap[0]:.2f}, {kap[1]:.3f}) herm {r.hermiticity[ik]:.2e} rank {r.rank[ik]}/{10 * J} "
              f"d_min {r.d_min[ik]:.2e} d_all {r.d_all[ik]:.2e}  E/B_ph - (2n+1), n<6: "
              f"{np.array2string(err[ik], formatter={'float_kind': lambda x: f'{x:+.2e}'})}", flush=True)
    print(f"J {J}: max herm {r.hermiticity.max():.2e}; n = 0 error over k / B_ph min {err[:, 0].min():+.3e} "
          f"max {err[:, 0].max():+.3e}; lowest {n} levels minus the exact ladder (Ry): min {above.min():+.3e} "
          f"(every level above: {bool(above.min() > -1e-10)}); singular {r.singular:.3e}; timings "
          f"{ {k: round(v, 1) for k, v in r.timings.items()} }; wall {time.perf_counter() - t0:.1f} s",
          flush=True)
    for key, val in (("levels", r.levels), ("herm", r.hermiticity), ("rank", r.rank), ("d_min", r.d_min),
                     ("d_all", r.d_all), ("projectability", r.projectability), ("kappa", r.kappa)):
        saved[f"{key}_J{J}"] = val
    saved["b_ph"] = r.b_ph
np.savez(OUT / f"gate-{mode}.npz", ladder=ladder, **saved)
print(f"saved {OUT / f'gate-{mode}.npz'}")
