"""Route U step 1 probe: the magnetic ultracell on the shared graphene cell, free or local-only.

Usage:
    python mag_route_u.py --mode free|local --p 1 --q 10 --nbnd 2,4,8,16 --N1 10 \
        --kmesh 2,2 --h0 eigen|box [--out file.npz] [--w W] [--nlev 40]

Free: the potential is zero, so the levels are (2n+1) B_ph + (2 pi m / c)^2 with p states
per level per magnetic k; every computed level is printed against the sorted reference with
its multiplicity, and the error of the (n, m = 0) levels for n = 0, 1, 2. Local: the frozen
potential of plans/orbital-field/graphene-scf.npz with the nonlocal term off.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from defuqumat import Calculator
from defuqumat.magnetic.conventions import MagneticCell
from defuqumat.magnetic.landau import free_landau_levels
from defuqumat.units import RY_TO_EV
from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="free")
ap.add_argument("--p", type=int, default=1)
ap.add_argument("--q", type=int, default=10)
ap.add_argument("--nbnd", default="2,4,8,16")
ap.add_argument("--N1", type=int, default=10)
ap.add_argument("--kmesh", default="2,2")
ap.add_argument("--h0", default="eigen")
ap.add_argument("--w", type=int, default=None)
ap.add_argument("--nlev", type=int, default=40)
ap.add_argument("--ecut", type=float, default=50.0)
ap.add_argument("--conv", type=float, default=1e-10)
ap.add_argument("--scf", default="plans/orbital-field/graphene-scf.npz")
ap.add_argument("--out", default=None)
ap.add_argument("--window", default=None)
ap.add_argument("--N1list", default=None)
args = ap.parse_args()

calc = Calculator.from_keywords(
    lattice="hexagonal", a=2.46, c=15.0,
    atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
    pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=args.ecut * RY_TO_EV, kmesh=(12, 12, 1), smearing=0.01 * RY_TO_EV, smearing_type="mv")
free = args.mode == "free"
scf = None if free else args.scf
kmesh = tuple(int(x) for x in args.kmesh.split(","))
out = {}
np.set_printoptions(linewidth=150)
ref = None
if free and args.p > 0:
    mc = MagneticCell.build(calc.system.cell, args.p, args.q)
    levels, mult, labels = free_landau_levels(mc, nmax=12, g3max=4)
    ref = np.repeat(levels, mult)
    ref_labels = np.repeat(labels, mult, axis=0)
    print(f"B_ph {float(mc.b_ph)!r} Ry, field {mc.field_tesla:.1f} T", flush=True)
previous = None
for nbnd in [int(x) for x in args.nbnd.split(",")]:
    t0 = time.perf_counter()
    r = run_magnetic_ultracell(calc.system, calc.pseudos, scf, flux=(args.p, args.q), nbnd=nbnd,
                               N1=args.N1, kmesh=kmesh, local_only=True, free=free, w=args.w,
                               h0=args.h0, conv_thr=args.conv, strict=False,
                               window=None if args.window is None else
                               tuple(float(x) for x in args.window.split(",")))
    wall = time.perf_counter() - t0
    print(f"\n== nbnd {nbnd} flux {r.flux} N1 {r.n1} h0 {r.h0} window {r.window}: w {r.w} measured {r.w_measured} "
          f"outside(w) {r.outside:.3e} dropped |B(w+1)|/|B(0)| {r.dropped:.3e} singular {r.singular:.3e} "
          f"wall {wall:.1f} s", flush=True)
    print("outside weight by width:", np.array2string(r.outside_by_width, precision=3))
    print("timings:", json.dumps({k: round(v, 2) for k, v in r.timings.items()}), "davidson:", r.davidson)
    print("rank", r.rank.tolist(), "of", args.q * nbnd, " d_min", np.array2string(r.d_min, precision=3),
          " d_all", np.array2string(r.d_all, precision=3))
    print("hermiticity", " ".join(f"{x:.3e}" for x in r.hermiticity))
    print("centres (cells) min/max", float(r.centres.min()), float(r.centres.max()))
    nl = min(args.nlev, r.levels.shape[1])
    for ik, kap in enumerate(r.kappa):
        print(f"k {tuple(np.round(kap, 5))}: levels[:{nl}] Ry", np.array2string(r.levels[ik, :nl], precision=9))
    if ref is not None:
        n = min(nl, len(ref))
        diff = r.levels[:, :n] - ref[None, :n]
        print("level - reference (sorted, per k), Ry:")
        for ik in range(len(r.kappa)):
            print("  ", np.array2string(diff[ik], precision=2))
        print(f"min (level - reference) {diff.min():.3e}  max {diff.max():.3e}")
        for nn in range(3):
            pos = [i for i in range(n) if tuple(ref_labels[i]) == (nn, 0)]
            if pos:
                err = diff[:, pos]
                print(f"(n={nn}, m=0) at sorted positions {pos}: error per k {np.array2string(err.ravel(), precision=3)}")
        flat = r.levels.max(axis=0)[:n] - r.levels.min(axis=0)[:n]
        print(f"spread over k of the first {n} levels: max {flat.max():.3e}")
    if args.p == 0:
        # zero flux: the magnetic k (kappa1, kappa2) at q = 1 is the chemical k, and kappa1 = 0
        # is the mesh point i = 0, so the levels are the zero-field eigenvalues there
        for ik, kap in enumerate(r.kappa):
            jk = int(round(kap[1] * kmesh[1]))
            i = int(round(kap[0] * args.N1))
            if abs(kap[0] * args.N1 - i) < 1e-9:
                ez = r.eigenvalues[jk, 0, i % args.N1]
                print(f"zero flux k {tuple(kap)}: max |level - zero-field eigenvalue| "
                      f"{np.max(np.abs(r.levels[ik, :nbnd] - ez)):.3e}")
    if previous is not None:
        m = min(previous.shape[1], r.levels.shape[1], args.nlev)
        change = r.levels[:, :m] - previous[:, :m]
        print(f"change from the previous nbnd over the lowest {m}: max rise {change.max():.3e} "
              f"(positive = not monotone), largest fall {change.min():.3e}")
    previous = r.levels
    out[f"levels_nbnd{nbnd}"] = r.levels
    out[f"rank_nbnd{nbnd}"] = r.rank
    out[f"dmin_nbnd{nbnd}"] = r.d_min
    out[f"herm_nbnd{nbnd}"] = r.hermiticity
    out[f"w_nbnd{nbnd}"] = r.w
    out[f"outside_nbnd{nbnd}"] = r.outside_by_width
    out[f"timings_nbnd{nbnd}"] = json.dumps(r.timings)
    out["kappa"] = r.kappa
if args.out:
    np.savez(args.out, flux=np.array([args.p, args.q]), N1=args.N1, h0=args.h0, mode=args.mode, **out)
    print("saved", args.out)
