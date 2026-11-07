import numpy as np, time, sys
from variants import build, levels
from peierls_torus import dirac_levels
for q, p in [(7900, 1), (3950, 1), (1975, 1), (400, 2)]:
    B, vF, En = dirac_levels(q, p, 11)
    t0 = time.time()
    H = build(q, p, (0.11, 0.23), "peierls")
    herm = abs(H - H.getH()).max()
    w, v = levels(H, 2 * p * 12 + 2)
    pos = np.sort(w[w > 1e-6])
    lev = []
    for x in pos:
        if not lev or abs(x - lev[-1][0]) > 1e-7: lev.append([x, 1])
        else: lev[-1][1] += 1
    ratios = [lev[i][0] / En[11 + i + 1] for i in range(min(10, len(lev)))]
    print(f"q={q} p={p} B={B:.3f} T dim={2*q} herm={herm:.0e} zero-mult={int((abs(w)<1e-6).sum())} "
          f"mult(n>0)={[m for x, m in lev[:3]]} time {time.time()-t0:.1f}s")
    print("   E_1 (eV) =", round(lev[0][0], 5), " Dirac E_1 =", round(En[12], 5))
    print("   E_n/Dirac n=1..10:", np.round(ratios, 5))
