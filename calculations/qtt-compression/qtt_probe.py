"""Quantics compressibility of DFT fields from defuqumat, with qutecipy's TCI.

Each field is put on a 2^R x 2^R x 2^R grid in crystal coordinates by exact
Fourier synthesis (zero-padded box), then
  * TT-SVD gives the exact quantics ranks at a relative Frobenius tolerance,
  * qutecipy.crossinterpolate2 learns it from samples; we record rank,
    number of distinct samples, and the measured max error on the full array.
Controls: white noise, and random coefficients on the same G sphere.
"""
import sys, time, json
import numpy as np
import os
sys.path.insert(0, os.environ.get("QUTECIPY_PATH", os.path.expanduser("~/Documents/programs/claude/qutecipy")))
from qutecipy import crossinterpolate2

# ----------------------------------------------------------------- unfolding
def to_quantics(arr, scheme):
    """(N,N,N) -> tensor with one leg per site. interleaved: 3R legs of 2,
    order x1 y1 z1 x2 ... (coarse first). fused: R legs of 8."""
    N = arr.shape[0]; R = int(round(np.log2(N)))
    t = arr.reshape([2] * (3 * R))           # x bits, y bits, z bits (coarse first)
    perm = [d * R + r for r in range(R) for d in range(3)]
    t = t.transpose(perm)
    if scheme == "fused":
        t = t.reshape([8] * R)
    return t

def ttsvd_ranks(t, tol):
    """Exact TT-SVD ranks, global relative Frobenius error <= tol."""
    dims = t.shape; n = len(dims)
    eps = tol * np.linalg.norm(t) / np.sqrt(max(n - 1, 1))
    ranks, nparam = [], 0
    C = t.reshape(dims[0], -1); r = 1
    for i in range(n - 1):
        C = C.reshape(r * dims[i], -1)
        U, s, Vh = np.linalg.svd(C, full_matrices=False)
        tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]   # tail[k] = norm of s[k:]
        keep = max(1, int(np.sum(tail > eps)))
        nparam += r * dims[i] * keep
        C = s[:keep, None] * Vh[:keep]; r = keep
        ranks.append(keep)
    nparam += r * dims[-1]
    return ranks, nparam

def tci(t, tol, seed=0, return_full=False):
    dims = list(t.shape)
    calls = {}
    dtype = np.complex128 if np.iscomplexobj(t) else np.float64
    def f(v):
        v = tuple(int(i) for i in v)
        if v not in calls: calls[v] = t[v]
        return calls[v]
    rng = np.random.default_rng(seed)
    start = list(np.unravel_index(np.argmax(np.abs(t)), t.shape))
    tt, ranks, errs = crossinterpolate2(dtype, f, dims, [start], tolerance=tol,
                                        maxiter=40, normalizeerror=True)
    # dense reconstruction to measure the true error
    from qutecipy.tensortrain.core import tensortrain
    cores = tensortrain(tt).sitetensors()
    full = cores[0]
    for c in cores[1:]:
        full = np.tensordot(full, c, axes=([-1], [0]))
    full = full.reshape(t.shape)
    err = np.max(np.abs(full - t)) / np.max(np.abs(t))
    nparam = sum(c.size for c in cores)
    if return_full:
        return max(tt.linkdims()), nparam, len(calls), err, full
    return max(tt.linkdims()), nparam, len(calls), err

# ----------------------------------------------------------------- the cell
def fields(R_list):
    from defuqumat import Calculator
    calc = Calculator.from_file(__import__("os").environ.get("QTT_INPUT", "benchmarks/si-1k.in"), pseudo_dir="tests/data/pseudo")
    res = calc.get_scf()
    basis = calc.calculation.basis
    dense = basis.dense
    mill_d = np.asarray(dense.miller).astype(int)
    from defuqumat.basis.fft import r_to_g
    rho_g = np.asarray(r_to_g(np.asarray(res.density[0]), dense.fft_index))
    v_g = np.asarray(r_to_g(np.asarray(res.potential[0]), dense.fft_index))
    pw = basis.planewaves
    mill_k = np.asarray(pw.miller(dense))[0].astype(int)
    mask = np.asarray(pw.mask)[0]
    psi = np.asarray(res.wavefunctions)[0, 0]          # (nbnd, npwx)
    print("ngm", len(mill_d), "npw", int(mask.sum()), "grid", dense.grid,
          "nbnd", psi.shape[0], "E", res.total_energy)
    def synth(coef, mill, N):
        if np.abs(mill).max() * 2 + 1 > N:
            return None
        box = np.zeros((N, N, N), complex)
        np.add.at(box, (mill[:, 0] % N, mill[:, 1] % N, mill[:, 2] % N), coef)
        return np.fft.ifftn(box) * N ** 3
    rng = np.random.default_rng(1)
    rnd_g = rng.normal(size=len(mill_d)) + 1j * rng.normal(size=len(mill_d))
    # hermitian: make field real by symmetrising afterwards (take real part)
    out = {}
    for R in R_list:
        N = 2 ** R
        d = {}
        a = synth(rho_g, mill_d, N)
        if a is None: continue
        d["density"] = a.real
        d["potential"] = synth(v_g, mill_d, N).real
        c = np.where(mask, psi[0], 0); d["u_k band1"] = synth(c[mask], mill_k[mask], N)
        c = np.where(mask, psi[3], 0); d["u_k band4"] = synth(c[mask], mill_k[mask], N)
        d["random same sphere"] = synth(rnd_g, mill_d, N).real
        d["white noise"] = rng.normal(size=(N, N, N))
        out[R] = d
    return out, len(mill_d), int(mask.sum())

if __name__ == "__main__":
    Rs = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [4, 5, 6]
    tols = [1e-2, 1e-4, 1e-6]
    F, ngm, npw = fields(Rs)
    rows = []
    for R, d in F.items():
        for name, arr in d.items():
            for scheme in ["interleaved", "fused"]:
                t = to_quantics(arr, scheme)
                for tol in tols:
                    rk, npar = ttsvd_ranks(t, tol)
                    t0 = time.time()
                    try:
                        trk, tpar, ncall, err = tci(t, tol)
                    except Exception as e:
                        trk = tpar = ncall = -1; err = float("nan"); print("TCI fail", e)
                    row = dict(R=R, field=name, scheme=scheme, tol=tol,
                               svd_rank=max(rk), svd_params=npar,
                               tci_rank=trk, tci_params=tpar, samples=ncall,
                               N3=arr.size, tci_maxerr=err, t=time.time() - t0)
                    rows.append(row)
                    print(f"R={R} {name:20s} {scheme:11s} tol={tol:.0e} "
                          f"svd r={max(rk):4d} p={npar:8d} | tci r={trk:4d} p={tpar:8d} "
                          f"samples={ncall:8d} ({ncall/arr.size:.3f} of N^3) "
                          f"maxerr={err:.1e} {time.time()-t0:.1f}s", flush=True)
    json.dump(dict(ngm=ngm, npw=npw, rows=rows), open(sys.argv[2] if len(sys.argv) > 2 else "/dev/null", "w"), indent=1)
