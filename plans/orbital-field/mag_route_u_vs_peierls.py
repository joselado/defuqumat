"""Route U step 2, acceptance 4: the ultracell levels against Peierls on the Wannier set.

Usage: python mag_route_u_vs_peierls.py <route-u npz> [<route-u npz> ...]

For each route-U file (written by mag_route_u2.py; its flux and magnetic k-mesh are read
from it) the foundation's Peierls model on both saved Wannier sets
(graphene-wannier-auto.npz and -pz.npz) is solved at the same p/q and the same magnetic
k, with E_D and v_F from the zero-field interpolated bands (dirac_point). Per k the two
levels nearest E_D are the zero pair, the two above it n = +1 and the two below n = -1
(two valleys, p = 1); the sets are pooled over the mesh and sorted, never compared at a
named k. Printed: the largest |U - P| over n = +1 and n = -1 divided by E_1 = <P(n=+1)> -
E_D, per nbnd of the route-U file, with the Peierls levels beside them.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from defuqumat.units import RY_TO_EV
from defuqumat.wannier.interpolation import interpolate
from defuqumat.workflows.landau import run_landau_levels

here = Path(__file__).resolve().parent


def wannier(mode):
    z = np.load(here / f"graphene-wannier-{mode}.npz")
    shifted, count = z["ws_shifted"], z["ws_count"]
    ws = None if shifted.size == 0 else (shifted, count)

    def bands(kappa):
        return interpolate(np.atleast_2d(kappa), z["rvectors"], z["degeneracy"], z["hoppings"],
                           *(ws if ws is not None else (None, None)))[0]

    return SimpleNamespace(centres=z["centres"], rvectors=z["rvectors"], degeneracy=z["degeneracy"],
                           hoppings=z["hoppings"], lattice=z["lattice"], ws_shifts=ws, spin=None,
                           edge=float(z["fermi_energy"]), bands=bands)


def ladder(levels, e_d):
    """``(zero pair, n = +1, n = -1)`` pooled over k, each sorted (Ry)."""
    zero, plus, minus = [], [], []
    for row in levels:
        row = np.sort(row)
        pair = np.sort(np.argsort(np.abs(row - e_d))[:2])
        zero += list(row[pair])
        plus += list(row[pair[1] + 1:pair[1] + 3])
        minus += list(row[pair[0] - 2:pair[0]])
    return np.sort(zero), np.sort(plus), np.sort(minus)


sets = {mode: wannier(mode) for mode in ("auto", "pz")}
for path in sys.argv[1:]:
    u = np.load(path)
    p, q = (int(x) for x in u["flux"])
    kappa = u["kappa"]
    print(f"\n== {path}: flux {p}/{q}, {len(kappa)} magnetic k", flush=True)
    peierls = {}
    for mode, w in sets.items():
        out = run_landau_levels(w, flux=(p, q), kappa=kappa, nlevels=16, dirac=True)
        e_d = out.dirac["E_D"]
        z, plus, minus = ladder(out.levels, e_d)
        e1 = float(np.mean(plus) - e_d)
        peierls[mode] = (e_d, z, plus, minus, e1)
        print(f"Peierls ({mode}): E_D {e_d!r} Ry, E_1 {e1 * RY_TO_EV:.5f} eV (Dirac "
              f"{out.dirac['E1_dirac_ev']:.5f}), n=+1 {np.array2string((plus - e_d) * RY_TO_EV, precision=5)} eV, "
              f"n=-1 {np.array2string((minus - e_d) * RY_TO_EV, precision=5)} eV, zero "
              f"{np.array2string((z - e_d) * RY_TO_EV, precision=5)} eV", flush=True)
    for key in sorted(k for k in u.files if k.startswith("levels_nbnd")):
        nbnd = key[len("levels_nbnd"):]
        for mode, (e_d, z, plus, minus, e1) in peierls.items():
            uz, up, um = ladder(u[key], e_d)
            dev = max(np.max(np.abs(up - plus)), np.max(np.abs(um - minus))) / e1
            print(f"nbnd {nbnd} vs {mode}: max |U - P| / E_1 over n = +-1 = {dev:.4f}; U n=+1 "
                  f"{np.array2string((up - e_d) * RY_TO_EV, precision=5)} eV, n=-1 "
                  f"{np.array2string((um - e_d) * RY_TO_EV, precision=5)} eV, zero "
                  f"{np.array2string((uz - e_d) * RY_TO_EV, precision=5)} eV", flush=True)
