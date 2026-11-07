"""Route U step 2, acceptance 3: the 2q levels about E_D against nbnd, from saved level sets.

Usage: python mag_route_u2_nbnd.py <npz> [<npz> ...]   (NumPy only, no JAX)

Graphene has 4 occupied bands per chemical cell, so 4q levels of the magnetic cell lie
below E_D; the 2q levels about E_D are the indices 3q .. 5q - 1 counted from the bottom of
the spectrum at each magnetic k, which is the count Rayleigh-Ritz bounds (a nested basis
gives levels that fall as it grows, level by level). Printed per nbnd step: the largest rise
(positive = not monotone), the largest fall, and the largest change, in Ry; the rank, d_min
and Hermiticity of each nbnd.
"""
import sys

import numpy as np

for path in sys.argv[1:]:
    z = np.load(path)
    p, q = (int(x) for x in z["flux"])
    nb = sorted(int(k[len("levels_nbnd"):]) for k in z.files if k.startswith("levels_nbnd"))
    lo, hi = 3 * q, 5 * q
    print(f"\n== {path}: flux {p}/{q}, levels {lo} .. {hi - 1} from the bottom, E_F "
          f"{float(z['fermi_energy'])!r} Ry")
    for n in nb:
        print(f"nbnd {n}: rank {z[f'rank_nbnd{n}'].tolist()}, d_min "
              f"{np.array2string(z[f'dmin_nbnd{n}'], precision=3)}, hermiticity max "
              f"{np.max(z[f'herm_nbnd{n}']):.3e}, w {int(z[f'w_nbnd{n}'])}")
    for a, b in zip(nb, nb[1:]):
        la, lb = z[f"levels_nbnd{a}"][:, lo:hi], z[f"levels_nbnd{b}"][:, lo:hi]
        change = lb - la
        print(f"nbnd {a} -> {b}: max rise {change.max():+.4e} Ry, largest fall {change.min():+.4e} Ry, "
              f"max |change| {np.max(np.abs(change)):.4e} Ry")
