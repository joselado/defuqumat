"""QTT of the plane-wave coefficients c(G) on the Miller-index box.

c(h) is placed at index (h + N/2) on a 2^R box per axis (R the smallest that holds
the sphere), the box is unfolded into quantics bits as in real space, and the train
is truncated (TT-SVD, Frobenius) or learned (TCI, max error relative to max|c|).
The error that matters is the field's, so each truncated train is mapped back to
real space and its max error relative to max|f| is measured on a 2^7 grid.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qtt_probe as qp
from orderings import reorder, named_orders
from matched_pw import tt_svd_maxerr  # noqa: F401  (rebuilds cores; reused below)

def tt_svd_full(t, tol):
    dims = t.shape; n = len(dims)
    eps = tol * np.linalg.norm(t) / np.sqrt(n - 1)
    cores = []; C = t.reshape(dims[0], -1); r = 1; p = 0
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
    return full.reshape(t.shape), [c.shape[-1] for c in cores[:-1]], sum(c.size for c in cores)

def unreorder(t, order, R):
    inv = np.argsort([a * R + l for a, l in order])
    return t.transpose(inv).reshape(2 ** R, 2 ** R, 2 ** R)

def synth(coef, mill, N):
    box = np.zeros((N, N, N), complex)
    np.add.at(box, (mill[:, 0] % N, mill[:, 1] % N, mill[:, 2] % N), coef)
    return np.fft.ifftn(box) * N ** 3

if __name__ == "__main__":
    from defuqumat import Calculator
    from defuqumat.basis.fft import r_to_g
    inp = os.environ.get("QTT_INPUT", "benchmarks/si-1k.in")
    calc = Calculator.from_file(inp, pseudo_dir="tests/data/pseudo")
    res = calc.get_scf(); basis = calc.calculation.basis; dense = basis.dense
    mill_d = np.asarray(dense.miller).astype(int)
    rho_g = np.asarray(r_to_g(np.asarray(res.density[0]), dense.fft_index))
    pw = basis.planewaves; mask = np.asarray(pw.mask)[0]
    mill_k = np.asarray(pw.miller(dense))[0].astype(int)[mask]
    psi_g = np.asarray(res.wavefunctions)[0, 0, 0][mask]
    rows = []
    NR = 2 ** 7
    for name, coef, mill, real in [("density", rho_g, mill_d, True), ("u_k band1", psi_g, mill_k, False)]:
        hmax = np.abs(mill).max(); R = int(np.ceil(np.log2(2 * hmax + 1))); N = 2 ** R
        box = np.zeros((N, N, N), complex)
        box[mill[:, 0] + N // 2, mill[:, 1] + N // 2, mill[:, 2] + N // 2] = coef
        f = synth(coef, mill, NR); f = f.real if real else f; fmax = np.abs(f).max()
        nz = int(np.sum(np.abs(coef) > 1e-12 * np.abs(coef).max()))
        print(f"{inp} {name}: box {N}^3 (R={R}), {len(coef)} coefficients, {nz} nonzero", flush=True)
        for oname in ["interleaved xyz, coarse->fine", "grouped x|y|z, coarse->fine"]:
            order = named_orders(R)[oname]
            t = reorder(box, order)
            for tol in [1e-2, 1e-4, 1e-6]:
                # bisection on the TT tolerance so that the FIELD error meets tol
                lo, hi = -12.0, 0.0
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    full, rk, p = tt_svd_full(t, 10 ** mid)
                    cb = unreorder(full, order, R)
                    g = synth(cb[mill[:, 0] + N // 2, mill[:, 1] + N // 2, mill[:, 2] + N // 2], mill, NR)
                    e = np.abs((g.real if real else g) - f).max() / fmax
                    if e <= tol: lo = mid
                    else: hi = mid
                full, rk, p = tt_svd_full(t, 10 ** lo)
                # TCI on the same tensor, at the TT tolerance found above (max-norm on c)
                t0 = time.time()
                trk, tp, ns, cerr, tfull = qp.tci(t.astype(complex), 10 ** lo, return_full=True)
                cb = unreorder(tfull, order, R)
                g = synth(cb[mill[:, 0] + N // 2, mill[:, 1] + N // 2, mill[:, 2] + N // 2], mill, NR)
                tci_field_err = np.abs((g.real if real else g) - f).max() / fmax
                row = dict(input=inp, field=name, order=oname, R=R, tol_field=tol, tt_tol=10 ** lo,
                           n_full=len(coef), n_nonzero=nz, svd_rank=max(rk), svd_params_complex=p,
                           tci_rank=trk, tci_params_complex=tp, tci_samples=ns, tci_field_err=tci_field_err, box=N ** 3)
                rows.append(row)
                print(f"  {oname:30s} field tol={tol:.0e} (tt tol {10**lo:.1e}): svd r={max(rk):3d} p={p:6d} | "
                      f"tci r={trk:3d} p={tp:6d} samples={ns:6d} of {N**3} field err={tci_field_err:.1e}", flush=True)
    json.dump(rows, open(sys.argv[1], "w"), indent=1)
