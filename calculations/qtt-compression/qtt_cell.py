"""Quantics compressibility of the fields of one cell, in real space and in G space.

    python3 qtt_cell.py <cell> <outdir> [--dR 1,2,3] [--tci-dR 3]

One SCF per cell (cached in <outdir>/<cell>-fields.npz), then for each field
(density, local potential, u_k of band 1 and of the top occupied band, and a
random field on the density's G sphere as the control):

* real space: the field synthesised exactly on 2^Ra x 2^Rb x 2^Rc points, with
  R per axis the smallest that holds the sphere plus dR. Reference trains by
  TT-SVD (Frobenius tolerance) for the grouped orders (all six axis orders) and
  interleaved; TCI on the best grouped order, error measured on the full array.
* matched plane-wave count: largest |c_G| kept first, fewest whose field has max
  error <= tol * max|f| (on the dR=1 grid).
* G space: c(h) on the smallest 2^R box per axis, TT-SVD with the tolerance
  bisected until the field error meets tol, and TCI from (a) the largest
  coefficient only and (b) the 32 largest as initial pivots; field error of each.

Parameter counts are real numbers for a real field, complex for a state or for a
G-space box. Everything is written to <outdir>/<cell>.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get(
    "QUTECIPY_PATH", os.path.expanduser("~/Documents/programs/claude/qutecipy")))
from qutecipy import crossinterpolate2  # noqa: E402

TOLS = (1e-2, 1e-4, 1e-6)
BOHR = 0.529177210903
RY_EV = 13.605693122994


# ---------------------------------------------------------------- the cells
def _si_cubic(n: int):
    """n x n x n conventional cubic cells of silicon, crystal coordinates."""
    basis = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],
                      [.25, .25, .25], [.25, .75, .75], [.75, .25, .75], [.75, .75, .25]])
    pos = [(b + np.array(c)) / n for c in itertools.product(range(n), repeat=3) for b in basis]
    return np.array(pos)


def _si_keywords(pos, pseudo_dir, smearing=0.1):
    from defuqumat import Calculator
    return Calculator.from_keywords(
        lattice="sc", a=10.20 * BOHR * 2, atoms=[("Si", tuple(p)) for p in pos],
        pseudos={"Si": "Si.pz-vbc.UPF"}, pseudo_dir=pseudo_dir,
        ecut=12.0 * RY_EV, kmesh=(1, 1, 1), smearing=smearing)


def make_calculator(cell: str, repo: str):
    from defuqumat import Calculator
    pseudo = os.path.join(repo, "tests/data/pseudo")
    if cell in ("si2", "si8", "al-slab"):
        name = {"si2": "si-1k", "si8": "si8-1k", "al-slab": "al-slab"}[cell]
        return Calculator.from_file(os.path.join(repo, "benchmarks", name + ".in"), pseudo_dir=pseudo)
    pos = _si_cubic(2)                      # 64 atoms, a = 2 x 10.20 bohr
    if cell == "si64":
        pass
    elif cell == "si63-vac":
        pos = pos[1:]
    elif cell == "si64-disp":
        rng = np.random.default_rng(0)       # 0.05 A rms per component
        pos = pos + rng.normal(scale=0.05 / (10.20 * BOHR * 2), size=pos.shape)
    else:
        raise ValueError(cell)
    return _si_keywords(pos, pseudo)


def load_fields(cell: str, repo: str, outdir: str):
    path = os.path.join(outdir, f"{cell}-fields.npz")
    if os.path.exists(path):
        return dict(np.load(path))
    from defuqumat.basis.fft import r_to_g
    t0 = time.time()
    calc = make_calculator(cell, repo)
    res = calc.get_scf()
    basis = calc.calculation.basis
    dense = basis.dense
    pw = basis.planewaves
    mask = np.asarray(pw.mask)[0]
    occ = np.asarray(res.occupations)[0]
    top = int(np.nonzero(occ > 0.5 * occ.max())[0].max())
    psi = np.asarray(res.wavefunctions)[0, 0]
    out = dict(
        rho_g=np.asarray(r_to_g(np.asarray(res.density[0]), dense.fft_index)),
        v_g=np.asarray(r_to_g(np.asarray(res.potential[0]), dense.fft_index)),
        mill_d=np.asarray(dense.miller).astype(int),
        mill_k=np.asarray(pw.miller(dense))[0].astype(int)[mask],
        psi_1=psi[0][mask], psi_top=psi[top][mask], top=top,
        grid=np.array(dense.grid), energy=res.total_energy,
        converged=res.converged, scf_seconds=time.time() - t0)
    np.savez(path, **out)
    return out


# ---------------------------------------------------------------- tensors
def synth(coef, mill, Ns):
    box = np.zeros(Ns, complex)
    np.add.at(box, tuple(mill[:, a] % Ns[a] for a in range(3)), coef)
    return np.fft.ifftn(box) * np.prod(Ns)


def sites(Rs, order):
    """order: 'interleaved' or an axis permutation such as (2, 0, 1), grouped."""
    if order == "interleaved":
        return [(a, l) for l in range(max(Rs)) for a in range(3) if l < Rs[a]]
    return [(a, l) for a in order for l in range(Rs[a])]


def to_bits(arr, Rs, order):
    t = arr.reshape([2] * sum(Rs))
    offs = np.cumsum([0, Rs[0], Rs[1]])
    return t.transpose([offs[a] + l for a, l in sites(Rs, order)])


def from_bits(t, Rs, order):
    offs = np.cumsum([0, Rs[0], Rs[1]])
    inv = np.argsort([offs[a] + l for a, l in sites(Rs, order)])
    return t.transpose(inv).reshape([2 ** r for r in Rs])


def tt_svd(t, tol, rebuild=False):
    dims = t.shape
    n = len(dims)
    eps = tol * np.linalg.norm(t) / np.sqrt(n - 1)
    cores, ranks, C, r = [], [], t.reshape(dims[0], -1), 1
    for i in range(n - 1):
        C = C.reshape(r * dims[i], -1)
        U, s, Vh = np.linalg.svd(C, full_matrices=False)
        tail = np.sqrt(np.cumsum(s[::-1] ** 2))[::-1]
        k = max(1, int(np.sum(tail > eps)))
        cores.append(U[:, :k].reshape(r, dims[i], k))
        C = s[:k, None] * Vh[:k]
        r = k
        ranks.append(k)
    cores.append(C.reshape(r, dims[-1], 1))
    params = sum(c.size for c in cores)
    full = contract(cores).reshape(dims) if rebuild else None
    return ranks, params, full


def contract(cores):
    full = cores[0]
    for c in cores[1:]:
        full = np.tensordot(full, c, axes=([-1], [0]))
    return full


def tci(t, tol, pivots=None):
    from qutecipy.tensortrain.core import tensortrain
    calls = {}
    dtype = np.complex128 if np.iscomplexobj(t) else np.float64

    def f(v):
        v = tuple(int(i) for i in v)
        if v not in calls:
            calls[v] = t[v]
        return calls[v]

    if pivots is None:
        pivots = [list(np.unravel_index(np.argmax(np.abs(t)), t.shape))]
    tt, _, _ = crossinterpolate2(dtype, f, list(t.shape), pivots, tolerance=tol,
                                 maxiter=40, normalizeerror=True)
    cores = tensortrain(tt).sitetensors()
    full = contract(cores).reshape(t.shape)
    return dict(rank=int(max(tt.linkdims())), params=int(sum(c.size for c in cores)),
                samples=len(calls)), full


# ---------------------------------------------------------------- the analysis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("outdir")
    ap.add_argument("--repo", default=os.getcwd())
    ap.add_argument("--dR", default="1,2,3")
    ap.add_argument("--tci-dR", type=int, default=3)
    ap.add_argument("--skip", default="", help="comma list of: orders,tci,matched,gspace")
    args = ap.parse_args()
    dRs = [int(x) for x in args.dR.split(",")]
    skip = set(args.skip.split(",")) - {""}
    os.makedirs(args.outdir, exist_ok=True)
    jpath = os.path.join(args.outdir, f"{args.cell}.json")
    report = json.load(open(jpath)) if os.path.exists(jpath) else {}

    def save():
        json.dump(report, open(jpath, "w"), indent=1, default=float)

    def log(*a):
        print(time.strftime("%H:%M:%S"), *a, flush=True)

    F = load_fields(args.cell, args.repo, args.outdir)
    mill_d, mill_k = F["mill_d"], F["mill_k"]
    rng = np.random.default_rng(1)
    rnd = rng.normal(size=len(mill_d)) + 1j * rng.normal(size=len(mill_d))
    fields = {  # name: (coefficients, miller, real)
        "density": (F["rho_g"], mill_d, True),
        "potential": (F["v_g"], mill_d, True),
        "u_k band 1": (F["psi_1"], mill_k, False),
        f"u_k band {int(F['top']) + 1} (top occupied)": (F["psi_top"], mill_k, False),
        "random same sphere": (rnd, mill_d, True),
    }
    hmax = np.abs(mill_d).max(axis=0)
    Rmin = [int(np.ceil(np.log2(2 * h + 1))) for h in hmax]
    report["info"] = dict(cell=args.cell, ngm=len(mill_d), npw=len(mill_k), fft_grid=F["grid"].tolist(),
                          hmax=hmax.tolist(), Rmin=Rmin, energy=float(F["energy"]),
                          converged=bool(F["converged"]), top_band=int(F["top"]) + 1,
                          nonzero_density=int(np.sum(np.abs(F["rho_g"]) > 1e-12 * np.abs(F["rho_g"]).max())),
                          nonzero_state=int(np.sum(np.abs(F["psi_1"]) > 1e-12 * np.abs(F["psi_1"]).max())))
    log("info", report["info"])
    save()
    perms = list(itertools.permutations(range(3)))

    def real_field(name, Ns):
        c, m, real = fields[name]
        f = synth(c, m, Ns)
        return f.real if real else f

    # --- real space: orders, then rank against resolution
    if "orders" not in skip:
        rows = report.setdefault("realspace", [])
        done = {(r["dR"], r["field"], r["order"], r["tol"]) for r in rows}
        for dR in dRs:
            Rs = [r + dR for r in Rmin]
            Ns = [2 ** r for r in Rs]
            for name in fields:
                f = real_field(name, Ns)
                orders = ["interleaved"] + perms if dR == max(dRs) else [(0, 1, 2)]
                best = None
                for order in orders:
                    tols = TOLS if order == (0, 1, 2) else (1e-4,)
                    for tol in tols:
                        key = (dR, name, str(order), tol)
                        if key in done:
                            continue
                        t0 = time.time()
                        ranks, p, _ = tt_svd(to_bits(f, Rs, order), tol)
                        rows.append(dict(dR=dR, Rs=Rs, field=name, order=str(order), tol=tol,
                                         maxrank=max(ranks), ranks=ranks, params=p, points=int(np.prod(Ns)),
                                         seconds=time.time() - t0))
                        log("svd", dR, Rs, name, order, tol, max(ranks), p)
                        save()
        # best grouped order of the density at the finest dR, used by TCI below
    if "tci" not in skip:
        dR = args.tci_dR
        Rs = [r + dR for r in Rmin]
        Ns = [2 ** r for r in Rs]
        rows = report.setdefault("tci_realspace", [])
        done = {(r["field"], r["tol"]) for r in rows}
        rs = [r for r in report.get("realspace", []) if r["dR"] == dR and r["tol"] == 1e-4
              and r["order"] != "interleaved"]
        for name in fields:
            if name == "random same sphere":
                continue
            cand = [r for r in rs if r["field"] == name]
            order = eval(min(cand, key=lambda r: r["params"])["order"]) if cand else (0, 1, 2)
            f = real_field(name, Ns)
            t = to_bits(f, Rs, order)
            for tol in TOLS[:2]:
                if (name, tol) in done:
                    continue
                t0 = time.time()
                info, full = tci(t, tol)
                info.update(field=name, tol=tol, Rs=Rs, order=str(order), points=int(np.prod(Ns)),
                            maxerr=float(np.abs(full - t).max() / np.abs(t).max()), seconds=time.time() - t0)
                rows.append(info)
                log("tci", info)
                save()

    # --- matched plane-wave counts, on the dR=1 grid
    if "matched" not in skip:
        Ns = [2 ** (r + 1) for r in Rmin]
        rows = report.setdefault("matched", [])
        done = {(r["field"], r["tol"]) for r in rows}
        for name, (c, m, real) in fields.items():
            f = real_field(name, Ns)
            fmax = np.abs(f).max()
            order = np.argsort(-np.abs(c))

            def err(n):
                g = synth(c[order[:n]], m[order[:n]], Ns)
                return np.abs((g.real if real else g) - f).max() / fmax

            for tol in TOLS:
                if (name, tol) in done:
                    continue
                lo, hi = 0, len(c)
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    lo, hi = (lo, mid) if err(mid) <= tol else (mid, hi)
                rows.append(dict(field=name, tol=tol, n_full=len(c), n_matched=hi))
                log("matched", name, tol, hi, len(c))
                save()

    # --- G space
    if "gspace" not in skip:
        Nf = [2 ** (r + 1) for r in Rmin]
        rows = report.setdefault("gspace", [])
        done = {(r["field"], r["order"], r["tol"], r["method"]) for r in rows}
        for name, (c, m, real) in fields.items():
            if name == "random same sphere":
                continue
            hm = np.abs(m).max(axis=0)
            Rg = [int(np.ceil(np.log2(2 * h + 1))) for h in hm]
            Ng = [2 ** r for r in Rg]
            box = np.zeros(Ng, complex)
            idx = tuple(m[:, a] + Ng[a] // 2 for a in range(3))
            box[idx] = c
            f = real_field(name, Nf)
            fmax = np.abs(f).max()

            def field_err(cb):
                g = synth(cb[idx], m, Nf)
                return float(np.abs((g.real if real else g) - f).max() / fmax)

            for order in ["interleaved", (0, 1, 2)]:
                t = to_bits(box, Rg, order)
                for tol in TOLS:
                    if (name, str(order), tol, "svd") not in done:
                        lo, hi = -12.0, 0.0
                        for _ in range(25):
                            mid = 0.5 * (lo + hi)
                            _, _, full = tt_svd(t, 10 ** mid, rebuild=True)
                            lo, hi = (mid, hi) if field_err(from_bits(full, Rg, order)) <= tol else (lo, mid)
                        ranks, p, full = tt_svd(t, 10 ** lo, rebuild=True)
                        rows.append(dict(field=name, order=str(order), tol=tol, method="svd", Rg=Rg,
                                         box=int(np.prod(Ng)), tt_tol=10 ** lo, maxrank=max(ranks), params=p,
                                         field_err=field_err(from_bits(full, Rg, order))))
                        log("gspace svd", rows[-1])
                        save()
                    if order != (0, 1, 2) or tol == 1e-6:
                        continue
                    flat = np.argsort(-np.abs(t).ravel())
                    for method, piv in [("tci, 1 pivot", None),
                                        ("tci, 32 pivots", [list(np.unravel_index(i, t.shape)) for i in flat[:32]])]:
                        if (name, str(order), tol, method) in done:
                            continue
                        t0 = time.time()
                        info, full = tci(t.astype(complex), tol, piv)
                        info.update(field=name, order=str(order), tol=tol, method=method, Rg=Rg,
                                    box=int(np.prod(Ng)), field_err=field_err(from_bits(full, Rg, order)),
                                    seconds=time.time() - t0)
                        rows.append(info)
                        log("gspace", info)
                        save()
    log("done")


if __name__ == "__main__":
    main()
