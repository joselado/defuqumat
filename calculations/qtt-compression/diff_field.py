"""The change a defect makes to a field, compressed on its own.

    python3 diff_field.py <results dir> <defect cell> <reference cell> <out.json> [--dR 1]

f_defect = f_ref + df, with f_ref and f_defect on the same G set (same cell, same
cutoff). df is compressed (TT-SVD, grouped order, tolerance bisected) and counted in
plane waves (largest first) so that the error is tol * max|f_defect|, the target the
whole field had. The cost of the decomposition is then the matched plane waves of
f_ref plus the train (or the plane waves) of df.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qtt_cell import TOLS, from_bits, synth, to_bits, tt_svd  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("defect")
    ap.add_argument("ref")
    ap.add_argument("out")
    ap.add_argument("--dR", type=int, default=1)
    args = ap.parse_args()
    A = np.load(os.path.join(args.results, f"{args.defect}-fields.npz"))
    B = np.load(os.path.join(args.results, f"{args.ref}-fields.npz"))
    assert np.array_equal(A["mill_d"], B["mill_d"]), "not the same G set"
    m = A["mill_d"]
    Rmin = [int(np.ceil(np.log2(2 * h + 1))) for h in np.abs(m).max(axis=0)]
    Rs = [r + args.dR for r in Rmin]
    Ns = [2 ** r for r in Rs]
    rows = []
    for key, name in [("rho_g", "density"), ("v_g", "potential")]:
        cd, cr = A[key], B[key]
        dc = cd - cr
        fd = synth(cd, m, Ns).real
        df = synth(dc, m, Ns).real
        scale = np.abs(fd).max()
        t = to_bits(df, Rs, (0, 1, 2))
        order = np.argsort(-np.abs(dc))
        for tol in TOLS:
            target = tol * scale
            if np.abs(df).max() <= target:
                rows.append(dict(field=name, tol=tol, train_params=0, maxrank=0, pw=0))
                continue
            lo, hi = -14.0, 0.0
            for _ in range(25):
                mid = 0.5 * (lo + hi)
                _, _, full = tt_svd(t, 10 ** mid, rebuild=True)
                err = np.abs(from_bits(full, Rs, (0, 1, 2)) - df).max()
                lo, hi = (mid, hi) if err <= target else (lo, mid)
            ranks, p, _ = tt_svd(t, 10 ** lo)
            a, b = 0, len(dc)
            while b - a > 1:
                c = (a + b) // 2
                g = synth(dc[order[:c]], m[order[:c]], Ns).real
                a, b = (a, c) if np.abs(g - df).max() <= target else (c, b)
            rows.append(dict(field=name, tol=tol, Rs=Rs, maxrank=max(ranks), train_params=p, pw=b,
                             max_df_over_max_f=float(np.abs(df).max() / scale)))
            print(rows[-1], flush=True)
    json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
