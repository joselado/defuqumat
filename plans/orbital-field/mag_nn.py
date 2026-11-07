"""Foundation acceptance items 1 to 5 on the nearest-neighbour honeycomb model, as numbers.

t = -2.7 eV, a = 2.46 A, B at (a1 + a2)/3 (plans/orbital-field-review/peierls_torus.py).
Usage: python mag_nn.py <outdir> [items], items a comma list of 1,2,3,4,5 (default all).
Writes nn-butterfly.npz (and nn-butterfly.png when matplotlib is there) for item 5.
"""
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from defuqumat.magnetic.conventions import MagneticCell
from defuqumat.magnetic.peierls import (
    PeierlsTorus, cluster_levels, hermiticity, honeycomb_nn_bonds, levels_near, plaquette_check,
)
from defuqumat.units import RY_TO_EV, TPI
from defuqumat.workflows.landau import run_landau_levels

out = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
items = set((sys.argv[2] if len(sys.argv) > 2 else "1,2,3,4,5").split(","))
BONDS = honeycomb_nn_bonds()
KAPPAS = ((0.0, 0.0), (0.37, 0.0), (0.0, 0.29), (0.21, 0.63))
record = {}


def torus(p, q, **kw):
    return PeierlsTorus(MagneticCell.build(BONDS.lattice, p, q), BONDS, **kw)


def model():
    def bands(kappa):
        return np.array([np.linalg.eigvalsh(BONDS.zero_field(k)) for k in np.atleast_2d(kappa)])
    return SimpleNamespace(bands=bands, edge=0.0, spin=None)


if "1" in items:
    t = torus(1, 400)
    herm, spectra = [], []
    for kappa in KAPPAS:
        h = t.matrix(kappa)
        herm.append(hermiticity(h) * RY_TO_EV)
        spectra.append(levels_near(h, 0.0, 22, vectors=False))
    spread = max(np.max(np.abs(s - spectra[0])) for s in spectra[1:]) * RY_TO_EV
    broken = torus(1, 400, bloch_factor=False)
    hb = hermiticity(broken.matrix((0.37, 0.29), check=False)) * RY_TO_EV
    try:
        broken.matrix((0.37, 0.29))
        raised = False
    except ValueError:
        raised = True
    cell = MagneticCell.build(BONDS.lattice, 1, 400)
    lattice = BONDS.lattice
    sites = [(0, (1, 0, 0)), (0, (0, 1, 0)), (0, (1, 1, 0)), (1, (0, 0, 0)), (1, (1, 0, 0)), (1, (0, 1, 0))]
    centre = 2 * (lattice[0] + lattice[1]) / 3
    pos = [BONDS.centres[o] + np.array(R) @ lattice - centre for o, R in sites]
    loop = [sites[i] for i in np.argsort([math.atan2(x[1], x[0]) for x in pos])]
    prod = plaquette_check(cell, BONDS, loop)
    try:
        plaquette_check(cell, BONDS, loop, hop_phase=lambda a, b: -cell.peierls_phase(a, b))
        flipped = False
    except ValueError:
        flipped = True
    record["1"] = {"hermiticity_ev": herm, "spread_22_ev": spread, "no_bloch_factor_hermiticity_ev": hb,
                   "guard_raised": raised, "plaquette_dev": abs(prod - np.exp(-1j * TPI / 400)),
                   "flipped_sign_raised": flipped,
                   "levels_ev": (spectra[0] * RY_TO_EV).tolist()}
    print("item 1:", json.dumps(record["1"]), flush=True)

if "2" in items:
    res = {}
    for p, q in ((2, 401), (2, 399), (3, 400)):
        t = torus(p, q)
        per = []
        for kappa in KAPPAS:
            w = levels_near(t.matrix(kappa), 0.0, 4 * p * 6, vectors=False)
            levels, counts = cluster_levels(w, 1e-6 / RY_TO_EV)
            zero = int(np.argmin(np.abs(levels)))
            per.append({"zero": int(counts[zero]), "positive": counts[zero + 1:-1].tolist(),
                        "levels_ev": np.round(levels[zero:] * RY_TO_EV, 6).tolist()[:6]})
        res[f"{p}/{q}"] = per
    record["2"] = res
    print("item 2:", json.dumps(res), flush=True)

if "3" in items:
    res = {}
    for q in (7900, 3950, 1975):
        t0 = time.perf_counter()
        r = run_landau_levels(model(), bonds=BONDS, flux=(1, q), kappa=((0.11, 0.23),), nlevels=48, dirac=True)
        wall = time.perf_counter() - t0
        d = r.dirac
        e = d["per_k"][0]
        res[q] = {"B": r.field_tesla, "dim": r.dim, "E1_ev": e["energies_ev"].get(1), "E1_dirac_ev": d["E1_dirac_ev"],
                  "Em1_ev": e["energies_ev"].get(-1), "v_F": d["v_F"],
                  "ratios": e["ratios"], "fit_electron": e.get("fit_electron"), "fit_hole": e.get("fit_hole"),
                  "zero_multiplicity": e["zero_multiplicity"], "zero_site_traces": e["zero_site_traces"],
                  "multiplicities": e["multiplicities"], "hermiticity_rel": float(r.relative[0]),
                  "timings": r.timings, "wall_s": wall}
        print(f"item 3 q={q}:", json.dumps(res[q]), flush=True)
    record["3"] = res

if "4" in items:
    q, kappa2 = 400, 0.29
    t = torus(1, q)
    wt = levels_near(t.matrix((0.0, kappa2)), 0.0, 40, vectors=False)
    pos = np.sort(wt[wt > 1e-8])[:12] * RY_TO_EV
    ribbon = t.ribbon(q)
    wr, vr = np.linalg.eigh(ribbon.matrix((0.0, kappa2)).toarray())
    nw = BONDS.nw
    edge = np.zeros(ribbon.dim, bool)
    edge[:20 * nw] = True
    edge[-20 * nw:] = True
    edge_weight = np.sum(np.abs(vr[edge]) ** 2, axis=0)
    bulk = wr[edge_weight < 1e-6] * RY_TO_EV
    diffs = [float(np.min(np.abs(bulk - e))) for e in pos]
    record["4"] = {"torus_first_12_positive_ev": pos.tolist(), "nearest_bulk_ribbon_diff_ev": diffs,
                   "n_bulk_ribbon": int(len(bulk))}
    print("item 4:", json.dumps(record["4"]), flush=True)

if "5" in items:
    # The two identities. E -> -E holds at every k (the sublattice operator anticommutes
    # with every hop, phases included). p/q -> (q - p)/q is n_phi -> -n_phi up to a whole
    # flux quantum per hexagon: a gauge transformation that is not periodic over the
    # magnetic cell, composed with time reversal, so it maps k to another k and the two
    # spectra are equal as sets over the zone, not at one k. The mesh that closes under
    # that map is measured here: N x N magnetic kappa, N in (1, 2, 3, 6, 12).
    def spectra_on(N, qmax):
        mesh = [(i / N, j / N) for i in range(N) for j in range(N)]
        out = {}
        for q in range(1, qmax + 1):
            for p in range(1, q + 1):
                if math.gcd(p, q) != 1 or (q == 1 and p != 1):
                    continue
                t = torus(p, q)
                out[(p, q)] = [np.linalg.eigvalsh(t.matrix(k).toarray()) for k in mesh]
        return out

    def residuals(spectra):
        chiral = per_k = union = 0.0
        for (p, q), ws in spectra.items():
            for w in ws:
                chiral = max(chiral, float(np.max(np.abs(w + w[::-1]))))
            if q > 1:
                other = spectra[(q - p, q)]
                per_k = max(per_k, max(float(np.max(np.abs(a - b))) for a, b in zip(ws, other)))
                union = max(union, float(np.max(np.abs(np.sort(np.concatenate(ws))
                                                       - np.sort(np.concatenate(other))))))
        return chiral * RY_TO_EV, per_k * RY_TO_EV, union * RY_TO_EV

    meshes = {}
    for N in (1, 2, 3, 6, 12):
        meshes[N] = residuals(spectra_on(N, 12))
        print(f"item 5 mesh {N}x{N}, q <= 12: chiral {meshes[N][0]:.2e} eV, p/q vs (q-p)/q per kappa "
              f"{meshes[N][1]:.2e} eV, over the mesh {meshes[N][2]:.2e} eV", flush=True)
    t0 = time.perf_counter()
    N = 6
    spectra = spectra_on(N, 50)
    chiral, per_k, union = residuals(spectra)
    wall = time.perf_counter() - t0
    fl = np.concatenate([np.full(len(w), p / q) for (p, q), ws in spectra.items() for w in ws])
    en = np.concatenate([w for ws in spectra.values() for w in ws]) * RY_TO_EV
    np.savez(out / "nn-butterfly.npz", flux=fl, energy_ev=en, mesh=N)
    record["5"] = {"meshes_q12": {str(k): v for k, v in meshes.items()}, "mesh": N, "chiral_ev": chiral,
                   "p_vs_q_minus_p_per_kappa_ev": per_k, "p_vs_q_minus_p_mesh_union_ev": union,
                   "n_flux": len(spectra), "wall_s": wall}
    print("item 5:", json.dumps(record["5"]), flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(fl, en, ",", color="k")
        ax.set_xlabel("flux per cell p/q")
        ax.set_ylabel("E (eV)")
        ax.set_title("honeycomb, t = -2.7 eV, q <= 50")
        fig.tight_layout()
        fig.savefig(out / "nn-butterfly.png", dpi=200)
        print("figure saved", flush=True)
    except ImportError as exc:
        print("no figure:", exc, flush=True)

with open(out / f"nn-record-{'-'.join(sorted(items))}.json", "w") as f:
    json.dump(record, f, indent=1, default=str)
