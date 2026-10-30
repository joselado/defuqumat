"""Exact quantics ranks (TT-SVD) of silicon fields under different bit orderings,
and a greedy search over orderings. Sites are single bits (axis, level),
level 0 = coarsest."""
import os, sys, json, itertools
import numpy as np
sys.path.insert(0, sys.path[0])
from qtt_probe import fields, ttsvd_ranks

def reorder(arr, order):
    N = arr.shape[0]; R = int(round(np.log2(N)))
    t = arr.reshape([2] * (3 * R))
    return t.transpose([a * R + l for a, l in order])

def named_orders(R):
    il = [(a, l) for l in range(R) for a in range(3)]
    return {
        "interleaved xyz, coarse->fine": il,
        "interleaved xyz, fine->coarse": [(a, l) for l in reversed(range(R)) for a in range(3)],
        "grouped x|y|z, coarse->fine": [(a, l) for a in range(3) for l in range(R)],
        "grouped x|y|z, x fine->coarse": [(0, l) for l in reversed(range(R))] + [(a, l) for a in (1, 2) for l in range(R)],
        "interleaved, z,y,x zig-zag": [(a, l) for l in range(R) for a in ((0, 1, 2) if l % 2 == 0 else (2, 1, 0))],
    }

def cost(arr, order, tol):
    return ttsvd_ranks(reorder(arr, order), tol)

def greedy(arr, order, tol, sweeps=6):
    """Adjacent and long swaps, accept when the parameter count drops."""
    best = list(order); bc = cost(arr, best, tol)[1]
    n = len(best)
    for _ in range(sweeps):
        improved = False
        for i, j in itertools.combinations(range(n), 2):
            trial = list(best); trial[i], trial[j] = trial[j], trial[i]
            c = cost(arr, trial, tol)[1]
            if c < bc:
                best, bc, improved = trial, c, True
        if not improved:
            break
    return best, bc

if __name__ == "__main__":
    Rs = [int(x) for x in sys.argv[1].split(",")]
    tol = float(sys.argv[2])
    do_greedy = len(sys.argv) > 3 and sys.argv[3] == "greedy"
    F, ngm, npw = fields(Rs)
    print("ngm", ngm, "npw", npw)
    out = []
    for R, d in F.items():
        for name in (["density", "u_k band1"] if do_greedy else ["density", "potential", "u_k band1", "u_k band4", "random same sphere"]):
            arr = d[name]
            for oname, order in named_orders(R).items():
                rk, p = cost(arr, order, tol)
                out.append(dict(R=R, field=name, order=oname, ranks=rk, params=p))
                print(f"R={R} {name:20s} {oname:32s} maxrank={max(rk):4d} params={p:8d}  ranks={rk}", flush=True)
            if do_greedy:
                starts = {k: greedy(arr, named_orders(R)[k], tol) for k in ["interleaved xyz, coarse->fine", "grouped x|y|z, coarse->fine"]}
                best, bc = min(starts.values(), key=lambda b: b[1])
                rk, p = cost(arr, best, tol)
                out.append(dict(R=R, field=name, order="greedy", sites=best, ranks=rk, params=p))
                print(f"R={R} {name:20s} {'GREEDY':32s} maxrank={max(rk):4d} params={p:8d}  order={best}", flush=True)
    json.dump(out, open(f"{os.path.dirname(os.path.abspath(__file__))}/results/orderings_R{'-'.join(map(str,Rs))}_tol{tol:.0e}{'_greedy' if do_greedy else ''}.json", "w"), indent=1)
