import sys, time, json, os
import numpy as np
sys.path.insert(0, sys.path[0])
from qtt_probe import fields, ttsvd_ranks, tci
from orderings import reorder, named_orders
Rs = [int(x) for x in sys.argv[1].split(",")]
F, ngm, npw = fields(Rs)
rows = []
for R, d in F.items():
    for name in ["density", "u_k band1", "random same sphere"]:
        t = reorder(d[name], named_orders(R)["grouped x|y|z, coarse->fine"])
        for tol in [1e-2, 1e-4, 1e-6]:
            rk, p = ttsvd_ranks(t, tol)
            t0 = time.time(); trk, tp, ns, err = tci(t, tol)
            row = dict(input=os.environ.get("QTT_INPUT", "benchmarks/si-1k.in"), ngm=ngm, npw=npw, R=R, field=name, tol=tol,
                       svd_rank=max(rk), svd_params=p, tci_rank=trk, tci_params=tp, samples=ns, N3=t.size, maxerr=err, seconds=time.time()-t0)
            rows.append(row)
            print(f"R={R} {name:20s} tol={tol:.0e} svd r={max(rk):3d} p={p:7d} | tci r={trk:3d} p={tp:7d} samples={ns:8d} ({ns/t.size:.4f} of N^3) maxerr={err:.1e} {row['seconds']:.0f}s", flush=True)
json.dump(rows, open(sys.argv[2], "w"), indent=1)
