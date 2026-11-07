"""Route U step 2 probe: the magnetic ultracell with the GIPAW-dressed nonlocal term.

Usage:
    python mag_route_u2.py --basis full|local --nonlocal gipaw|bare|none --p 1 --q 10 \
        --nbnd 8,16 --N1 10 --kmesh 2,2 --h0 eigen|box [--out file.npz] [--w W]

On the shared graphene state (plans/orbital-field/graphene-scf.npz). Prints per nbnd the
Hermiticity per magnetic k, the projector check (grid V_NL^0 against vkb, Ry), the rank,
the levels about E_F, the sublattice weights of the two levels nearest E_F (trace over the
pair, no tolerance), the change from the previous nbnd, timings and the peak RSS. At p = 0
it also prints the levels against the zero-field eigenvalues (acceptance 1).
"""
import argparse
import json
import resource
import time

import numpy as np

from defuqumat import Calculator
from defuqumat.units import RY_TO_EV
from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

ap = argparse.ArgumentParser()
ap.add_argument("--basis", default="full")
ap.add_argument("--nonlocal", default="gipaw")
ap.add_argument("--p", type=int, default=1)
ap.add_argument("--q", type=int, default=10)
ap.add_argument("--nbnd", default="8")
ap.add_argument("--N1", type=int, default=10)
ap.add_argument("--kmesh", default="2,2")
ap.add_argument("--h0", default="eigen")
ap.add_argument("--w", type=int, default=None)
ap.add_argument("--nlev", type=int, default=None)
ap.add_argument("--ecut", type=float, default=50.0)
ap.add_argument("--conv", type=float, default=1e-10)
ap.add_argument("--scf", default="plans/orbital-field/graphene-scf.npz")
ap.add_argument("--out", default=None)
ap.add_argument("--scale", type=float, default=1.0)
args = ap.parse_args()

calc = Calculator.from_keywords(
    lattice="hexagonal", a=2.46, c=15.0,
    atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
    pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=args.ecut * RY_TO_EV, kmesh=(12, 12, 1), smearing=0.01 * RY_TO_EV, smearing_type="mv")
ef = float(np.load(args.scf)["fermi_energy"])
kmesh = tuple(int(x) for x in args.kmesh.split(","))
np.set_printoptions(linewidth=160)
out = {}
previous = None
print(f"E_F {ef!r} Ry; basis {args.basis}, nonlocal {args.nonlocal}, h0 {args.h0}, "
      f"flux {args.p}/{args.q}, N1 {args.N1}, kmesh {kmesh}", flush=True)
for nbnd in [int(x) for x in args.nbnd.split(",")]:
    t0 = time.perf_counter()
    r = run_magnetic_ultracell(calc.system, calc.pseudos, args.scf, flux=(args.p, args.q),
                               nbnd=nbnd, N1=args.N1, kmesh=kmesh, local_only=args.basis == "local",
                               nonlocal_term=args.nonlocal, w=args.w, h0=args.h0,
                               conv_thr=args.conv, strict=False, scale=args.scale,
                               sublattices=[[0], [1]])
    wall = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    print(f"\n== nbnd {nbnd}: mode {r.nonlocal_mode}, w {r.w} (measured {r.w_measured}), outside(w) "
          f"{r.outside:.3e}, dropped |B(w+1)|/|B(0)| {r.dropped:.3e}, singular {r.singular:.3e}, "
          f"wall {wall:.1f} s, peak RSS {rss:.0f} MB", flush=True)
    print("projector check |grid - vkb| (Ry):", None if r.nonlocal_error is None else
          f"{r.nonlocal_error:.3e}", " radius", r.projector_radius, " dropped norm", r.projector_dropped)
    print("timings:", json.dumps({k: round(v, 2) for k, v in r.timings.items()}), "davidson:", r.davidson)
    print("rank", r.rank.tolist(), "of", args.q * nbnd, " d_min", np.array2string(r.d_min, precision=3))
    print("hermiticity", " ".join(f"{x:.3e}" for x in r.hermiticity), flush=True)
    rel = r.levels - ef
    for ik, kap in enumerate(r.kappa):
        order = np.argsort(np.abs(rel[ik]))
        near = np.sort(rel[ik][order[:8]])
        pair = order[:2]
        wts = r.region_weights[ik][pair].sum(axis=0)
        print(f"k {tuple(np.round(kap, 5))}: 8 levels nearest E_F (eV) "
              f"{np.array2string(near * RY_TO_EV, precision=4)}; tr over the nearest pair of "
              f"(w_A, w_B) = {np.array2string(wts, precision=4)}")
    if args.p == 0:
        for ik, kap in enumerate(r.kappa):
            jk = int(round(kap[1] * kmesh[1]))
            i = int(round(kap[0] * args.N1))
            if abs(kap[0] * args.N1 - i) < 1e-9:
                ez = r.eigenvalues[jk, 0, i % args.N1]
                print(f"zero flux k {tuple(kap)}: max |level - zero-field eigenvalue| "
                      f"{np.max(np.abs(r.levels[ik, :nbnd] - ez)):.3e} Ry")
    if previous is not None:
        m = min(previous.shape[1], r.levels.shape[1])
        change = r.levels[:, :m] - previous[:, :m]
        print(f"change from the previous nbnd over the lowest {m}: max rise {change.max():.3e} Ry "
              f"(positive = not monotone), largest fall {change.min():.3e}")
    previous = r.levels
    for key, value in (("levels", r.levels), ("rank", r.rank), ("dmin", r.d_min),
                       ("herm", r.hermiticity), ("w", r.w), ("outside", r.outside_by_width),
                       ("timings", json.dumps(r.timings)), ("regions", r.region_weights),
                       ("nonlocal_error", -1.0 if r.nonlocal_error is None else r.nonlocal_error),
                       ("eigenvalues", r.eigenvalues)):
        out[f"{key}_nbnd{nbnd}"] = value
    out["kappa"] = r.kappa
if args.out:
    np.savez(args.out, flux=np.array([args.p, args.q]), N1=args.N1, h0=args.h0, basis=args.basis,
             nonlocal_term=args.nonlocal, fermi_energy=ef, **out)
    print("saved", args.out)
