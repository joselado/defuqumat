"""Timing pair (2): the Peierls torus at 10 T against the ribbon of 165 cells, same code, same B.

The 8-function Wannier set of the shared graphene state (graphene-wannier-auto.npz), hopping
cutoff 1e-6 Ry (foundation item 6), flux 1/7900 (9.989 T): the torus of 7900 cells (dim
63200) and the ribbon of 165 cells (dim 1320, five magnetic lengths), 30 levels nearest E_D
by shift-invert, each geometry run twice and the second call reported (the first is
reported too: there is no compilation on this path, so the two should agree).
Usage: python mag_time_peierls.py [dir with the npz]
"""
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from defuqumat.magnetic.landau import dirac_point
from defuqumat.units import RY_TO_EV
from defuqumat.wannier.interpolation import interpolate
from defuqumat.workflows.landau import run_landau_levels

src = Path(sys.argv[1] if len(sys.argv) > 1 else "plans/orbital-field")
z = np.load(src / "graphene-wannier-auto.npz")
shifted, count = z["ws_shifted"], z["ws_count"]
ws = None if shifted.size == 0 else (shifted, count)


def bands(kappa):
    return interpolate(np.atleast_2d(kappa), z["rvectors"], z["degeneracy"], z["hoppings"],
                       *(ws if ws is not None else (None, None)))[0]


w = SimpleNamespace(centres=z["centres"], rvectors=z["rvectors"], degeneracy=z["degeneracy"],
                    hoppings=z["hoppings"], lattice=z["lattice"], ws_shifts=ws, spin=None,
                    edge=float(z["fermi_energy"]), bands=bands)
e_d = dirac_point(w.lattice, bands, w.edge)["E_D"]
for geometry, width in (("torus", None), ("ribbon", 165)):
    for call in ("first", "second"):
        t0 = time.perf_counter()
        r = run_landau_levels(w, flux=(1, 7900), geometry=geometry, width=width, nlevels=30,
                              energy=e_d, cutoff=1e-6, kappa=((0.11, 0.23),))
        wall = time.perf_counter() - t0
        lev = np.sort(r.levels[0] - e_d) * RY_TO_EV * 1e3
        print(f"[{geometry} {call}] dim {r.dim}, {r.nbonds} bonds per cell, B {r.field_tesla:.3f} T, "
              f"wall {wall:.2f} s (bonds {r.timings['bonds_s']:.2f}, build {r.timings['build_s']:.2f}, "
              f"solve {r.timings['solve_s']:.2f}), peak RSS "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB; levels (meV from "
              f"E_D) {np.round(lev[12:18], 2).tolist()}", flush=True)
