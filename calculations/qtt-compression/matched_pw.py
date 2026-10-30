"""Plane waves needed at matched accuracy, and the max error of the TT-SVD train.

For each tolerance: keep the largest |c_G| first and find the smallest number kept
whose field has max|error| <= tol * max|f| on the 2^R grid (bisection on the count).
The TT-SVD train of the grouped order is also rebuilt to measure its max error.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtt_probe as qp
from orderings import reorder, named_orders

def tt_svd_maxerr(t, tol):
    dims = t.shape; n = len(dims)
    eps = tol * np.linalg.norm(t) / np.sqrt(n - 1)
    cores = []; C = t.reshape(dims[0], -1); r = 1
    for i in range(n - 1):
        C = C.reshape(r * dims[i], -1)
        U, s, Vh = np.linalg.svd(C, full_matrices=False)
        tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]
        k = max(1, int(np.sum(tail > eps)))
        cores.append(U[:, :k].reshape(r, dims[i], k)); C = s[:k, None] * Vh[:k]; r = k
    cores.append(C.reshape(r, dims[-1], 1))
    full = cores[0]
    for c in cores[1:]:
        full = np.tensordot(full, c, axes=([-1], [0]))
    return np.max(np.abs(full.reshape(t.shape) - t)) / np.max(np.abs(t))

if __name__ == "__main__":
    R = int(sys.argv[1]); N = 2 ** R
    from defuqumat import Calculator
    from defuqumat.basis.fft import r_to_g
    calc = Calculator.from_file(os.environ.get("QTT_INPUT", "benchmarks/si-1k.in"), pseudo_dir="tests/data/pseudo")
    res = calc.get_scf(); basis = calc.calculation.basis; dense = basis.dense
    mill_d = np.asarray(dense.miller).astype(int)
    rho_g = np.asarray(r_to_g(np.asarray(res.density[0]), dense.fft_index))
    pw = basis.planewaves; mask = np.asarray(pw.mask)[0]
    mill_k = np.asarray(pw.miller(dense))[0].astype(int)[mask]
    psi_g = np.asarray(res.wavefunctions)[0, 0, 0][mask]

    def synth(coef, mill):
        box = np.zeros((N, N, N), complex)
        np.add.at(box, (mill[:, 0] % N, mill[:, 1] % N, mill[:, 2] % N), coef)
        return np.fft.ifftn(box) * N ** 3

    rows = []
    for name, coef, mill, real in [("density", rho_g, mill_d, True), ("u_k band1", psi_g, mill_k, False)]:
        f = synth(coef, mill); f = f.real if real else f
        fmax = np.abs(f).max(); order = np.argsort(-np.abs(coef))
        def err(n):
            keep = order[:n]; g = synth(coef[keep], mill[keep]); g = g.real if real else g
            return np.abs(g - f).max() / fmax
        t = reorder(f, named_orders(R)["grouped x|y|z, coarse->fine"])
        for tol in [1e-2, 1e-4, 1e-6]:
            lo, hi = 0, len(coef)
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if err(mid) <= tol: hi = mid
                else: lo = mid
            rk, p = qp.ttsvd_ranks(t, tol)
            row = dict(input=os.environ.get("QTT_INPUT", "benchmarks/si-1k.in"), field=name, R=R, tol=tol,
                       n_full=len(coef), n_matched=hi, tt_params=p, tt_svd_maxerr=tt_svd_maxerr(t, tol))
            rows.append(row); print(row, flush=True)
    json.dump(rows, open(sys.argv[2], "w"), indent=1)
