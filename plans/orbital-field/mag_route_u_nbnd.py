"""Acceptance 4 of route U from the saved level sets: the lowest 4q levels against nbnd.

Usage: python mag_route_u_nbnd.py route-u-q1-local.npz [route-u-q2-local.npz ...]
For each file: q, and for each step of nbnd the largest rise (positive = not monotone)
and the largest change of the lowest 4q levels per magnetic k.
"""
import sys

import numpy as np

for path in sys.argv[1:]:
    d = np.load(path)
    q = int(d["flux"][1])
    nbnds = sorted(int(k.split("nbnd")[1]) for k in d.files if k.startswith("levels_nbnd"))
    n = 4 * q
    print(f"{path}: q = {q}, nbnd {nbnds}, lowest {n} levels over {d['kappa'].shape[0]} k")
    for a, b in zip(nbnds, nbnds[1:]):
        la, lb = d[f"levels_nbnd{a}"], d[f"levels_nbnd{b}"]
        m = min(n, la.shape[1], lb.shape[1])
        change = lb[:, :m] - la[:, :m]
        print(f"  {a} -> {b}: lowest {m}: largest rise {change.max():.3e} Ry, largest fall "
              f"{change.min():.3e} Ry, max |change| {np.abs(change).max():.3e} Ry")
