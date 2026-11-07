"""The crosscheck of route P, route U and Peierls on the shared graphene state.

Run from the worktree root on D22 (every subcommand imports defuqumat or reads its npz):

    python mag_cross.py exact             route P alone: the full mesh against q copies of the
                                          representatives, from every route-p-q*.npz
    python mag_cross.py floor local|full  the zero-field box (route P at p = 0) against the sphere
                                          (route U at p = 0) on the chemical k that fold onto the
                                          magnetic representatives of q = 1, 2, 3, 4, 6, and both
                                          Dirac energies at K; writes floor-<kind>.npz
    python mag_cross.py analyse           items 1, 3 and 4 from the npz files and the Peierls
                                          model; reads floor-*.npz when present

Everything compared is a sorted level set pooled over the four magnetic representatives
kappa1 in {0, 1/2}, kappa2 in {0, 1/(2q)} (route P's full mesh is q copies of it, which
``exact`` checks), a DOS, or an energy; never eps(k) at a named k (Lu et al., PRL 135,
196401 (2025), Fig. 3, the magnetic band shift). The ultracell is Wang and Vafek, PRB 108,
245109 (2023); the Peierls model the foundation's, on the two P162 Wannier sets.
Output directory: $MAG_OUT, default /l/ladovj1/review/mag/cross. Energies in Ry unless said.
"""
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DATA = Path("plans/orbital-field")
SCF = str(DATA / "graphene-scf.npz")
OUT = Path(os.environ.get("MAG_OUT", "/l/ladovj1/review/mag/cross"))
QS = (1, 2, 3, 4, 6)
RY_EV = 13.605693122990
TRAPZ = getattr(np, "trapezoid", None) or np.trapz


def reps(kpoints, q):
    """Mask of route P's representatives kappa2 in {0, 1/(2q)} on the (2, 2q) mesh."""
    return np.isin(np.round(np.asarray(kpoints)[:, 1] * 2 * q).astype(int), [0, 1])


def pooled(levels, lo, hi):
    """Per-k levels lo .. hi - 1 from the bottom, pooled over k and sorted."""
    return np.sort(np.sort(np.asarray(levels), axis=1)[:, lo:hi].ravel())


# ----------------------------------------------------------------------------- exact

def exact():
    worst = 0.0
    for path in sorted(DATA.glob("route-p-q*.npz")):
        z = np.load(path)
        q = int(z["flux"][1])
        e, kp = z["eigenvalues"], z["kpoints"]
        rep = reps(kp, q)
        copies = np.sort(np.tile(e[rep].ravel(), q))
        d = float(np.max(np.abs(z["sorted_levels"] - copies)))
        extra = ""
        if "representatives_sorted" in z.files:
            extra = (f", stored representatives against the mesh's own "
                     f"{np.max(np.abs(z['representatives_sorted'] - np.sort(e[rep].ravel()))):.1e}")
        worst = max(worst, d)
        print(f"{path.name}: q {q}, nk {len(kp)}, {int(rep.sum())} representatives, nbnd {e.shape[1]}: "
              f"full mesh against q copies {d:.3e} Ry{extra}")
    print(f"worst over the files: {worst:.3e} Ry (design 1e-9)")


# ----------------------------------------------------------------------------- floor

def chemical_points():
    """The chemical k ((kappa1 + j)/q, kappa2) of every representative of every q, and K."""
    pts = set()
    for q in QS:
        for i in range(2 * q):
            for j in (0, 1):
                pts.add((round(i / (2 * q), 12), round(j / (2 * q), 12)))
    pts = sorted(pts)
    corners = [(1 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1 / 3)]
    return np.array([[a, b, 0.0] for a, b in pts] + [[a, b, 0.0] for a, b in corners])


def graphene():
    from defuqumat import Calculator
    from defuqumat.units import RY_TO_EV

    return Calculator.from_keywords(
        lattice="hexagonal", a=2.46, c=15.0, atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
        pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
        ecut=50.0 * RY_TO_EV, kmesh=(1, 1, 1), smearing=0.01 * RY_TO_EV, smearing_type="mv")


def dirac_pair(levels, ef):
    """The pair at K nearest E_F: (mean, split, index)."""
    lv = np.sort(levels)
    i = int(np.argmin([abs(0.5 * (lv[n] + lv[n + 1]) - ef) + abs(lv[n + 1] - lv[n])
                       for n in range(len(lv) - 1)]))
    return 0.5 * (lv[i] + lv[i + 1]), lv[i + 1] - lv[i], i


def floor(kind, nbnd=10):
    import resource

    from defuqumat.magnetic.blochgrid import sphere_levels
    from defuqumat.system.kpoints import KPoints
    from defuqumat.workflows.magnetic_bands import local_potential_of, run_magnetic_bands

    local = kind == "local"
    calc = graphene()
    pts = chemical_points()
    pot = local_potential_of(SCF)
    ef = float(np.load(SCF)["fermi_energy"])
    t0 = time.perf_counter()
    box = run_magnetic_bands(calc.system, calc.pseudos, SCF, flux=(0, 1), kpoints=pts.tolist(),
                             nbnd=nbnd, local_only=local)
    t_box = time.perf_counter() - t0
    print(f"[{kind}] box at p = 0: {len(pts)} k, ndim {box.ndim}, steps {box.steps.min()} to "
          f"{box.steps.max()}, unsettled {int(np.sum(box.unsettled))}, {t_box:.1f} s", flush=True)
    t0 = time.perf_counter()
    sub = calc.with_kpoints(KPoints.from_crystal(pts, np.full(len(pts), 1.0 / len(pts)), calc.system.cell))
    es, _, st, un = sphere_levels(sub.calculation, pot, nbnd, local_only=local)
    t_sph = time.perf_counter() - t0
    print(f"[{kind}] sphere: steps {st.min()} to {st.max()}, unsettled {int(np.sum(un))}, {t_sph:.1f} s",
          flush=True)
    eb = np.asarray(box.eigenvalues)
    d = es - eb
    print(f"[{kind}] sphere - box over all {len(pts)} k and {nbnd} bands: min {d.min():.3e}, max {d.max():.3e} Ry")
    for b in range(nbnd):
        print(f"[{kind}]   band {b}: sphere - box from {d[:, b].min():+.3e} to {d[:, b].max():+.3e} Ry")
    for ik in range(len(pts) - 3, len(pts)):
        mb, sb, ib = dirac_pair(eb[ik], ef)
        ms, ss, is_ = dirac_pair(es[ik], ef)
        print(f"[{kind}] corner {pts[ik, :2].round(6).tolist()}: box pair {ib},{ib + 1} mean {mb:.10f} split "
              f"{sb:.3e}; sphere pair {is_},{is_ + 1} mean {ms:.10f} split {ss:.3e}; box - sphere {mb - ms:+.3e} Ry")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"floor-{kind}.npz", kpoints=pts, box=eb, sphere=es, fermi_energy=ef,
             box_steps=box.steps, time_box=t_box, time_sphere=t_sph)
    print(f"[{kind}] saved {OUT / f'floor-{kind}.npz'}; peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")


def folded(floor_npz, which, q, lo, hi):
    """Zero-field levels lo..hi-1 per magnetic representative, from the chemical points folded onto it."""
    pts = floor_npz["kpoints"][:, :2]
    lev = floor_npz[which]
    key = {(round(a, 9), round(b, 9)): i for i, (a, b) in enumerate(pts)}
    rows, guard = [], np.inf
    for kap1 in (0.0, 0.5):
        for kap2 in (0.0, 1 / (2 * q)):
            idx = [key[(round(((kap1 + j) / q) % 1.0, 9), round(kap2, 9))] for j in range(q)]
            union = np.sort(lev[idx].ravel())
            guard = min(guard, float(np.min(lev[idx][:, -1])))
            rows.append(union[lo:hi])
            if hi < len(union) and union[hi - 1] >= guard:
                raise ValueError(f"q {q}: the folded window reaches the top computed band")
    return np.sort(np.concatenate(rows))


# ----------------------------------------------------------------------------- Peierls

def wannier(mode):
    from defuqumat.wannier.interpolation import interpolate

    z = np.load(DATA / f"graphene-wannier-{mode}.npz")
    shifted, count = z["ws_shifted"], z["ws_count"]
    ws = None if shifted.size == 0 else (shifted, count)

    def bands(kappa):
        return interpolate(np.atleast_2d(kappa), z["rvectors"], z["degeneracy"], z["hoppings"],
                           *(ws if ws is not None else (None, None)))[0]

    return SimpleNamespace(centres=z["centres"], rvectors=z["rvectors"], degeneracy=z["degeneracy"],
                           hoppings=z["hoppings"], lattice=z["lattice"], ws_shifts=ws, spin=None,
                           edge=float(z["fermi_energy"]), bands=bands, nw=int(z["num_wann"]))


def peierls_levels(w, bonds, q, kappa):
    from defuqumat.magnetic.conventions import MagneticCell
    from defuqumat.magnetic.peierls import PeierlsTorus

    torus = PeierlsTorus(MagneticCell.build(w.lattice, 1, q), bonds)
    out, herm = [], 0.0
    for kap in kappa:
        h = torus.matrix(np.asarray(kap, float)).toarray()
        herm = max(herm, float(np.max(np.abs(h - h.conj().T)) / np.max(np.abs(h))))
        out.append(np.linalg.eigvalsh(0.5 * (h + h.conj().T)))
    return np.array(out), herm


def dos(levels, grid, sigma):
    """States per Ry per magnetic k (Gaussian), averaged over the rows of ``levels``."""
    x = (grid[:, None] - np.asarray(levels).ravel()[None, :]) / sigma
    return np.exp(-0.5 * x ** 2).sum(1) / (sigma * np.sqrt(2 * np.pi) * len(levels))


def analyse():
    from defuqumat.magnetic.landau import dirac_levels, dirac_point
    from defuqumat.magnetic.peierls import bonds_from_wannier
    from defuqumat.units import tesla_from_flux

    floors = {k: (np.load(OUT / f"floor-{k}.npz") if (OUT / f"floor-{k}.npz").exists() else None)
              for k in ("local", "full")}
    kappa_rep = None
    saved = {}

    # ---- item 1: local-only, route U against route P, lowest 4q per k
    print("== item 1: pi^2 + V_loc, route U (sphere-based ultracell) minus route P (box), lowest 4q per k, "
          "pooled over the 4 representatives and sorted")
    for q in QS:
        P = np.load(DATA / f"route-p-q{q}-local.npz")
        U = np.load(DATA / f"route-u-q{q}-local.npz")
        rep = reps(P["kpoints"], q)
        kappa_rep = P["kpoints"][rep][:, :2]
        if not np.allclose(np.sort(kappa_rep, axis=0), np.sort(U["kappa"], axis=0)):
            raise ValueError(f"q {q}: route U's kappa {U['kappa'].tolist()} are not route P's representatives")
        p_set = pooled(P["eigenvalues"][rep], 0, 4 * q)
        fl = None
        if floors["local"] is not None:
            fl = folded(floors["local"], "sphere", q, 0, 4 * q) - folded(floors["local"], "box", q, 0, 4 * q)
        for nb in (8, 16, 32):
            u_set = pooled(U[f"levels_nbnd{nb}"], 0, 4 * q)
            d = u_set - p_set
            quarters = [np.max(np.abs(d[g * 4 * q:(g + 1) * 4 * q])) for g in range(4)]
            line = (f"q {q} nbnd {nb:2d}: U - P min {d.min():+.3e} max {d.max():+.3e} Ry, below -1e-8: "
                    f"{int(np.sum(d < -1e-8))} of {d.size}; max |U - P| by quarter of the set (deepest first) "
                    + " ".join(f"{x:.3e}" for x in quarters)
                    + f"; route U Hermiticity max {np.max(U[f'herm_nbnd{nb}']):.2e}")
            if fl is not None:
                ex = d - fl
                line += (f"; zero-field floor (sphere - box) {fl.min():+.3e} to {fl.max():+.3e}, "
                         f"U - P - floor {ex.min():+.3e} to {ex.max():+.3e}, mean {ex.mean():+.3e}")
            print(line, flush=True)
            saved[f"item1_q{q}_nbnd{nb}"] = d

    # ---- item 3: full H against Peierls, and route U against route P
    sets = {mode: wannier(mode) for mode in ("auto", "pz")}
    bonds = {mode: bonds_from_wannier(w) for mode, w in sets.items()}
    point = {mode: dirac_point(w.lattice, w.bands, w.edge) for mode, w in sets.items()}
    area = abs(np.linalg.det(sets["pz"].lattice)) / np.linalg.norm(sets["pz"].lattice[2])
    for mode, pt in point.items():
        print(f"Peierls {mode}: E_D {pt['E_D']!r} Ry, v_F {pt['v_F']:.4e} m/s (holes {pt['v_F_hole']:.4e}), "
              f"K {pt['K'].tolist()}")
    ed_box = ed_sphere = None
    if floors["full"] is not None:
        f = floors["full"]
        ef = float(f["fermi_energy"])
        k_corner = [i for i, kp in enumerate(f["kpoints"]) if np.allclose(kp[:2], point["pz"]["K"][:2])]
        ed_box = dirac_pair(f["box"][k_corner[0]], ef)[0]
        ed_sphere = dirac_pair(f["sphere"][k_corner[0]], ef)[0]
        print(f"route P E_D on the box at K {ed_box!r} Ry; on the sphere {ed_sphere!r} Ry; "
              f"box - sphere {ed_box - ed_sphere:+.3e} Ry")
    if ed_box is None:
        raise SystemExit("run 'floor full' first: route P's E_D is taken on its own box")
    vf = point["pz"]["v_F"]
    # the zero-field reference of the same statistic: route P's box at p = 0 and each Wannier set's
    # interpolated bands on the chemical k that fold onto the representatives, each about its E_D
    zero = {mode: {"kpoints": floors["full"]["kpoints"], "box": floors["full"]["box"],
                   "sphere": floors["full"]["sphere"], "wan": np.asarray(w.bands(floors["full"]["kpoints"]))}
            for mode, w in sets.items()}
    for mode in sets:
        kp = zero[mode]["kpoints"]
        sph = zero[mode]["sphere"]
        wan = zero[mode]["wan"]
        near = [np.min(np.abs(sph[i][:, None] - wan[i][None, :]), axis=0) for i in range(len(kp))]
        within = np.concatenate([n[np.abs(wan[i] - point[mode]["E_D"]) < 2.0 / RY_EV] for i, n in enumerate(near)])
        print(f"Wannier {mode} interpolation against the sphere on the {len(kp)} folded k: nearest-band "
              f"|W - sphere| within E_D +- 2 eV max {within.max() * RY_EV * 1e3:.2f} meV")
    print("\n== item 3: full H, route P against Peierls; the 2q levels about E_D per k (route P and the auto "
          "set 3q..5q-1, the pz set all 2q), each referred to its own E_D, pooled and sorted; N levels on "
          "each side of the gap at E_D per k")
    rows = []
    for q in (2, 3, 4, 6):
        P = np.load(DATA / f"route-p-q{q}-full.npz")
        rep = reps(P["kpoints"], q)
        kap = P["kpoints"][rep][:, :2]
        field = tesla_from_flux(1, q, area)
        e1 = float(dirac_levels(field, vf, 1)[2]) / RY_EV
        pe = np.sort(P["eigenvalues"][rep], axis=1)
        for mode, w in sets.items():
            lev, herm = peierls_levels(w, bonds[mode], q, kap)
            off = 3 * q if w.nw == 8 else 0
            gap_w, gap_p = off + q, 4 * q
            e1_w = float(np.mean(lev[:, gap_w + 1]) - point[mode]["E_D"])  # first level above the zero pair
            for N in sorted({1, 2, q}):
                if N > q:
                    continue
                pw = np.sort((lev[:, gap_w - N:gap_w + N] - point[mode]["E_D"]).ravel())
                pp = np.sort((pe[:, gap_p - N:gap_p + N] - ed_box).ravel())
                d = pp - pw
                d0 = ((folded(zero[mode], "box", q, gap_p - N, gap_p + N) - ed_box)
                      - (folded(zero[mode], "wan", q, gap_w - N, gap_w + N) - point[mode]["E_D"]))
                ex = d - d0
                raw = np.sort((pe[:, gap_p - N:gap_p + N]).ravel()) - np.sort((lev[:, gap_w - N:gap_w + N]).ravel())
                rows.append((q, field, mode, N, np.max(np.abs(d)), np.mean(d), e1, e1_w, np.max(np.abs(raw)),
                             np.max(np.abs(ex)), np.mean(ex), np.max(np.abs(d0))))
                print(f"q {q} ({field:.0f} T) {mode:4s} N {N}: max |P - W| {np.max(np.abs(d)) * RY_EV * 1e3:8.2f} meV, "
                      f"mean P - W {np.mean(d) * RY_EV * 1e3:+8.2f} meV, max/E_1 {np.max(np.abs(d)) / e1:.4f} "
                      f"(E_1 Dirac {e1 * RY_EV:.3f} eV, Peierls first level above the zero pair "
                      f"{e1_w * RY_EV:.3f} eV); without the E_D referencing max |P - W| "
                      f"{np.max(np.abs(raw)) * RY_EV * 1e3:.2f} meV; the same statistic at B = 0 on the folded k "
                      f"max {np.max(np.abs(d0)) * RY_EV * 1e3:.2f} meV mean {np.mean(d0) * RY_EV * 1e3:+.2f} meV; "
                      f"(P - W) - (P - W)_0 max {np.max(np.abs(ex)) * RY_EV * 1e3:.2f} meV mean "
                      f"{np.mean(ex) * RY_EV * 1e3:+.2f} meV; Peierls Hermiticity {herm:.1e}; "
                      f"P levels {np.array2string(pp * RY_EV, precision=3, max_line_width=400)} eV; "
                      f"W {np.array2string(pw * RY_EV, precision=3, max_line_width=400)} eV", flush=True)
                saved[f"item3_q{q}_{mode}_N{N}"] = d
    print("\nfits over q = 2, 3, 4, 6 (B in tesla), s = a + c B and s = c' B, extrapolated to 10 T, and s/E_1 "
          "against B and against sqrt(B):")
    e1_10 = float(dirac_levels(10.0, vf, 1)[2])                         # eV
    for mode in ("auto", "pz"):
        for N in (1, 2):
            for stat, col in (("max |P - W|", 4), ("mean P - W", 5), ("max |(P - W) - (P - W)_0|", 9),
                              ("mean (P - W) - (P - W)_0", 10)):
                sel = [r for r in rows if r[2] == mode and r[3] == N]
                if len(sel) < 3:
                    continue
                B = np.array([r[1] for r in sel])
                s = np.array([r[col] for r in sel]) * RY_EV            # eV
                ratio = s / (np.array([r[6] for r in sel]) * RY_EV)
                c1, a1 = np.polyfit(B, s, 1)
                c0 = float(B @ s / (B @ B))
                res1 = np.max(np.abs(a1 + c1 * B - s))
                res0 = np.max(np.abs(c0 * B - s))
                rb = np.polyfit(B, ratio, 1)
                rs = np.polyfit(np.sqrt(B), ratio, 1)
                print(f"{mode:4s} N {N} {stat}: s = {a1 * 1e3:+.3f} meV + {c1 * 1e6:+.4f} ueV/T B (max residual "
                      f"{res1 * 1e3:.3f} meV); through 0: {c0 * 1e6:+.4f} ueV/T (residual {res0 * 1e3:.3f} meV); "
                      f"at 10 T the slope gives {c1 * 10 * 1e3:+.4f} meV = {c1 * 10 / e1_10:+.2e} of E_1(10 T) "
                      f"= {e1_10 * 1e3:.2f} meV, through 0 {c0 * 10 / e1_10:+.2e}; s/E_1 vs B residual "
                      f"{np.max(np.abs(np.polyval(rb, B) - ratio)):.2e}, vs sqrt(B) residual "
                      f"{np.max(np.abs(np.polyval(rs, np.sqrt(B)) - ratio)):.2e}; s/E_1 = "
                      f"{np.array2string(ratio, precision=4)} at B = {np.array2string(B, precision=0)}", flush=True)

    print("\n== item 3b: full H, route U (nbnd 32) minus route P, pooled over the 4 representatives and sorted")
    for q in (2, 3, 4, 6):
        P = np.load(DATA / f"route-p-q{q}-full.npz")
        U = np.load(DATA / f"route-u-q{q}-full.npz")
        rep = reps(P["kpoints"], q)
        for nb in (8, 16, 32):
            for name, lo, hi in (("lowest 4q", 0, 4 * q), ("2q about E_D", 3 * q, 5 * q)):
                d = pooled(U[f"levels_nbnd{nb}"], lo, hi) - pooled(P["eigenvalues"][rep], lo, hi)
                line = (f"q {q} nbnd {nb:2d} {name:12s}: U - P min {d.min():+.3e} max {d.max():+.3e} mean "
                        f"{d.mean():+.3e} Ry, below -1e-8: {int(np.sum(d < -1e-8))} of {d.size}")
                if floors["full"] is not None:
                    fl = folded(floors["full"], "sphere", q, lo, hi) - folded(floors["full"], "box", q, lo, hi)
                    ex = d - fl
                    line += (f"; floor {fl.min():+.3e} to {fl.max():+.3e}; U - P - floor {ex.min():+.3e} to "
                             f"{ex.max():+.3e}")
                print(line + f"; route U Hermiticity max {np.max(U[f'herm_nbnd{nb}']):.2e}", flush=True)
                saved[f"item3b_q{q}_nbnd{nb}_{lo}_{hi}"] = d

    # ---- item 4: DOS at q = 6
    print("\n== item 4: DOS at q = 6, Gaussian 20 meV, each about its own E_D, 4 representatives (route P's full "
          "mesh is 6 copies), states per eV per magnetic k")
    q = 6
    P = np.load(DATA / "route-p-q6-full.npz")
    rep = reps(P["kpoints"], q)
    kap = P["kpoints"][rep][:, :2]
    pe = P["eigenvalues"][rep] - ed_box
    sigma = 0.020 / RY_EV
    for half in (1.0, 3.0):
        grid = np.linspace(-half, half, int(2000 * half) + 1) / RY_EV
        if np.min(pe[:, -1]) < grid[-1] + 5 * sigma:
            print(f"window +-{half} eV: route P's top band reaches into it; skipped")
            continue
        dp = dos(pe, grid, sigma)
        for mode, w in sets.items():
            lev, _ = peierls_levels(w, bonds[mode], q, kap)
            lw = lev - point[mode]["E_D"]
            dw = dos(lw, grid, sigma)
            rel = np.sqrt(TRAPZ((dp - dw) ** 2, grid) / TRAPZ(dp ** 2, grid))
            n_p = np.sum(np.abs(pe) < half / RY_EV) / len(pe)
            n_w = np.sum(np.abs(lw) < half / RY_EV) / len(lw)
            print(f"window E_D +- {half} eV, {mode}: relative L2 |D_P - D_W| / |D_P| = {rel:.4f}; states per k in the "
                  f"window P {n_p:.2f}, W {n_w:.2f}; P levels {np.array2string(np.sort(pe[np.abs(pe) < half / RY_EV]) * RY_EV, precision=4, max_line_width=300)} eV; "
                  f"W {np.array2string(np.sort(lw[np.abs(lw) < half / RY_EV]) * RY_EV, precision=4, max_line_width=300)} eV",
                  flush=True)
            saved[f"item4_{mode}_{half}"] = np.stack([grid * RY_EV, dp / RY_EV, dw / RY_EV])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "cross.npz", **saved)
    print(f"saved {OUT / 'cross.npz'}")


if __name__ == "__main__":
    t0 = time.perf_counter()
    cmd = sys.argv[1]
    if cmd == "exact":
        exact()
    elif cmd == "floor":
        floor(sys.argv[2])
    elif cmd == "analyse":
        analyse()
    else:
        raise SystemExit(__doc__)
    print(f"wall {time.perf_counter() - t0:.1f} s")
