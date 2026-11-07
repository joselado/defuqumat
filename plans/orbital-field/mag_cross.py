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
    python mag_cross.py reps q            route P's full H at 1/q on the four representatives only
                                          (q = 8, 10, beyond the committed meshes); route-p-q<q>-reps.npz
    python mag_cross.py diag              route P's zero pair against Peierls and route U, a kappa scan
    MAG_ECUT=70 python mag_cross.py floor full | reps q   the same at another cutoff, files tagged -70
    python mag_cross.py cutoff 70         route P's zero pair at 50 and 70 Ry against Peierls

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
FLOOR_QS = (1, 2, 3, 4, 6, 8, 10)
FULL_QS = (2, 3, 4, 6, 8, 10)


def route_p_full(q):
    """Route P's full-H set at 1/q: the committed (2, 2q) mesh, or the four representatives of
    ``representatives_run`` (plans/orbital-field/ first, then $MAG_OUT); None when neither exists."""
    for path in (DATA / f"route-p-q{q}-full.npz", DATA / f"route-p-q{q}-reps.npz", OUT / f"route-p-q{q}-reps.npz"):
        if path.exists():
            return np.load(path)
    return None
RY_EV = 13.605693122990
# the cutoff of the box (Ry); 50 is the design's and every committed set's. Another value tags the
# files written by floor and reps, so the 50 Ry ones are never overwritten.
ECUT = float(os.environ.get("MAG_ECUT", "50"))
TAG = "" if ECUT == 50.0 else f"-{ECUT:g}"
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
        if path.name.endswith("-reps.npz"):
            continue                                    # the four representatives only, no mesh
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
    for q in FLOOR_QS:
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
        ecut=ECUT * RY_TO_EV, kmesh=(1, 1, 1), smearing=0.01 * RY_TO_EV, smearing_type="mv")


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
    np.savez(OUT / f"floor-{kind}{TAG}.npz", kpoints=pts, box=eb, sphere=es, fermi_energy=ef,
             box_steps=box.steps, time_box=t_box, time_sphere=t_sph)
    print(f"[{kind}] ecut {ECUT} Ry, saved {OUT / f'floor-{kind}{TAG}.npz'}; peak RSS "
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


def folded_rows(floor_npz, which, q):
    """Per magnetic representative, every zero-field level of the chemical points folded onto it."""
    pts = floor_npz["kpoints"][:, :2]
    lev = np.asarray(floor_npz[which])
    key = {(round(a, 9), round(b, 9)): i for i, (a, b) in enumerate(pts)}
    rows = []
    for kap1 in (0.0, 0.5):
        for kap2 in (0.0, 1 / (2 * q)):
            idx = [key[(round(((kap1 + j) / q) % 1.0, 9), round(kap2, 9))] for j in range(q)]
            rows.append(np.sort(lev[idx].ravel()))
    return rows


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

    def floor_file(kind):
        for path in (OUT / f"floor-{kind}.npz", DATA / f"cross-floor-{kind}.npz"):
            if path.exists():
                return np.load(path)
        return None

    floors = {k: floor_file(k) for k in ("local", "full")}
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
        kp, sph, wan = zero[mode]["kpoints"], zero[mode]["sphere"], zero[mode]["wan"]
        for half in (1.0, 2.5):
            near = np.concatenate([np.min(np.abs(sph[i][:, None] - wan[i][None, :]), axis=0)
                                   [np.abs(wan[i] - point[mode]["E_D"]) < half / RY_EV] for i in range(len(kp))])
            print(f"Wannier {mode} interpolation against the sphere on the {len(kp)} folded k, the {near.size} "
                  f"levels within E_D +- {half} eV: nearest-band |W - sphere| max {near.max() * RY_EV * 1e3:.3f} meV")
    print("\n== item 3: full H, route P against Peierls, each set of levels about its own E_D (route P's on its "
          "box), pooled over the 4 representatives and sorted. 'zero pair' = the two levels at the gap per k "
          "(index 4q-1, 4q for P and the auto set, q-1, q for pz); 'within H eV' = every level with "
          "|E - E_D| < H, compared only when the count per k agrees. (P - W)_0 is the same statistic at B = 0 "
          "on the chemical k that fold onto the representatives (box at p = 0, interpolated Wannier bands).")
    rows = []

    def windowed(rows_, ed, half):
        return [np.sort(r[np.abs(r - ed) < half / RY_EV]) - ed for r in rows_]

    def against(a_rows, b_rows):
        ca, cb = [len(r) for r in a_rows], [len(r) for r in b_rows]
        if ca != cb:
            return None, ca, cb
        return np.sort(np.concatenate(a_rows)) - np.sort(np.concatenate(b_rows)), ca, cb

    for q in FULL_QS:
        P = route_p_full(q)
        if P is None:
            print(f"q {q}: no route P set, skipped")
            continue
        rep = reps(P["kpoints"], q)
        kap = P["kpoints"][rep][:, :2]
        field = tesla_from_flux(1, q, area)
        e1 = float(dirac_levels(field, vf, 1)[2]) / RY_EV
        pe = np.sort(P["eigenvalues"][rep], axis=1)
        p0 = folded_rows(zero["pz"], "box", q)
        for mode, w in sets.items():
            ed_w = point[mode]["E_D"]
            lev, herm = peierls_levels(w, bonds[mode], q, kap)
            w0 = folded_rows(zero[mode], "wan", q)
            gap_w = (3 * q if w.nw == 8 else 0) + q
            gap_p = 4 * q
            cases = [("zero pair", [r[gap_p - 1:gap_p + 1] - ed_box for r in pe],
                      [r[gap_w - 1:gap_w + 1] - ed_w for r in lev],
                      [np.sort(r)[gap_p - 1:gap_p + 1] - ed_box for r in p0],
                      [np.sort(r)[gap_w - 1:gap_w + 1] - ed_w for r in w0])]
            for half in (1.0, 2.5):
                cases.append((f"within {half} eV", windowed(pe, ed_box, half), windowed(lev, ed_w, half),
                              windowed(p0, ed_box, half), windowed(w0, ed_w, half)))
            for name, pr, wr, pr0, wr0 in cases:
                d, cp, cw = against(pr, wr)
                d0, cp0, cw0 = against(pr0, wr0)
                if d is None:
                    print(f"q {q} ({field:.0f} T) {mode:4s} {name:15s}: counts per k differ, P {cp} W {cw}; "
                          f"not compared", flush=True)
                    continue
                line = (f"q {q} ({field:.0f} T) {mode:4s} {name:15s}: {d.size} levels, max |P - W| "
                        f"{np.max(np.abs(d)) * RY_EV * 1e3:8.2f} meV, mean P - W {np.mean(d) * RY_EV * 1e3:+8.2f} meV, "
                        f"max/E_1 {np.max(np.abs(d)) / e1:.4f} (E_1 Dirac {e1 * RY_EV:.3f} eV)")
                ex = None
                if d0 is not None and d0.size == d.size:
                    ex = d - d0
                    line += (f"; at B = 0 max {np.max(np.abs(d0)) * RY_EV * 1e3:.2f} meV mean "
                             f"{np.mean(d0) * RY_EV * 1e3:+.2f} meV; (P - W) - (P - W)_0 max "
                             f"{np.max(np.abs(ex)) * RY_EV * 1e3:.2f} meV mean {np.mean(ex) * RY_EV * 1e3:+.2f} meV")
                else:
                    line += f"; at B = 0 the counts differ (P {cp0}, W {cw0}) or the size does"
                print(line + f"; Peierls Hermiticity {herm:.1e}", flush=True)
                print(f"      P {np.array2string(np.sort(np.concatenate(pr)) * RY_EV, precision=4, max_line_width=1000)} eV",
                      flush=True)
                print(f"      W {np.array2string(np.sort(np.concatenate(wr)) * RY_EV, precision=4, max_line_width=1000)} eV",
                      flush=True)
                rows.append((q, field, mode, name, np.max(np.abs(d)), np.mean(d), e1,
                             np.nan if ex is None else np.max(np.abs(ex)), np.nan if ex is None else np.mean(ex)))
                saved[f"item3_q{q}_{mode}_{name.replace(' ', '_')}"] = d
    print("\nfits over the q present (B in tesla): s = a + c B and s = c' B, the slope extrapolated to 10 T "
          "against E_1(10 T), and s/E_1 against B and against sqrt(B):")
    e1_10 = float(dirac_levels(10.0, vf, 1)[2])                         # eV
    for mode in ("auto", "pz"):
        for name in ("zero pair", "within 1.0 eV", "within 2.5 eV"):
            for stat, col in (("max |P - W|", 4), ("mean P - W", 5), ("max |(P - W) - (P - W)_0|", 7),
                              ("mean (P - W) - (P - W)_0", 8)):
                sel = [r for r in rows if r[2] == mode and r[3] == name and np.isfinite(r[col])]
                if len(sel) < 3:
                    print(f"{mode:4s} {name} {stat}: {len(sel)} points, no fit")
                    continue
                B = np.array([r[1] for r in sel])
                sv = np.array([r[col] for r in sel]) * RY_EV            # eV
                ratio = sv / (np.array([r[6] for r in sel]) * RY_EV)
                c1, a1 = np.polyfit(B, sv, 1)
                c0 = float(B @ sv / (B @ B))
                res1 = np.max(np.abs(a1 + c1 * B - sv))
                res0 = np.max(np.abs(c0 * B - sv))
                rb = np.polyfit(B, ratio, 1)
                rs = np.polyfit(np.sqrt(B), ratio, 1)
                print(f"{mode:4s} {name} {stat}: s = {a1 * 1e3:+.3f} meV + {c1 * 1e6:+.4f} ueV/T B (max residual "
                      f"{res1 * 1e3:.3f} meV); through 0: {c0 * 1e6:+.4f} ueV/T (residual {res0 * 1e3:.3f} meV); "
                      f"10 T times the slope {c1 * 10 * 1e3:+.5f} meV = {c1 * 10 / e1_10:+.2e} of E_1(10 T) "
                      f"= {e1_10 * 1e3:.2f} meV, through 0 {c0 * 10 / e1_10:+.2e}; s/E_1 linear in B residual "
                      f"{np.max(np.abs(np.polyval(rb, B) - ratio)):.2e}, in sqrt(B) "
                      f"{np.max(np.abs(np.polyval(rs, np.sqrt(B)) - ratio)):.2e}; "
                      f"s (meV) {np.array2string(sv * 1e3, precision=2)}, s/E_1 {np.array2string(ratio, precision=4)} "
                      f"at B {np.array2string(B, precision=0)}", flush=True)

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

    # ---- the pi states see no nonlocal term: C.pbe-hgh carries one s projector, even under the
    # mirror z -> -z, and the pi states are odd, so route P local-only and full must agree on them
    # The local-only operator has two more bands per cell below the pi pair (index 6q-1, 6q at the gap),
    # so its 4q + 8 levels hold the zero pair only for q <= 3; each full level within 1 eV of E_D is
    # matched to the nearest local level at the same k of the same mesh (one grid, two operators).
    print("\n== route P local-only against full on the levels within 1 eV of the box E_D (mirror-odd pi "
          "states; the s projector cannot reach them): nearest local level at the same k")
    for q in QS:
        L = np.load(DATA / f"route-p-q{q}-local.npz")
        F = np.load(DATA / f"route-p-q{q}-full.npz")
        near = []
        for lr, fr in zip(L["eigenvalues"], F["eigenvalues"]):
            for e in fr[np.abs(fr - ed_box) < 1.0 / RY_EV]:
                near.append(np.min(np.abs(lr - e)))
        near = np.array(near)
        print(f"q {q}: {near.size} full levels within 1 eV, nearest local level max {near.max() if near.size else 0:.3e} Ry, "
              f"min {near.min() if near.size else 0:.3e} Ry; local's top computed level minus E_D "
              f"{(np.min(L['eigenvalues'][:, -1]) - ed_box) * RY_EV:+.3f} eV", flush=True)

    # The diamagnetic inequality: for a local scalar potential the ground state of (p + A)^2 + V is
    # never below that of p^2 + V (Simon), so route P local-only cannot put a level below the
    # zero-field minimum of the same box; a uniform spurious downward shift would show here.
    fl_loc, fl_full = floors["local"], floors["full"]
    print("\n== the diamagnetic inequality, lowest level of route P against the zero-field minimum of the same box")
    # the deep s band's mass at Gamma on the box, from Gamma and (0, 1/12): its lowest Landau level
    # sits at E_0 + (m/m*) B_ph, which is the shift a uniform offset of route P would spoil
    lat = np.asarray(np.load(SCF)["lattice"])
    bvec = 2 * np.pi * np.linalg.inv(lat).T
    dk2 = float(np.sum((bvec[1] / 12) ** 2))
    kp_f = fl_full["kpoints"][:, :2]
    i_g = int(np.argmin(np.sum(kp_f ** 2, axis=1)))
    i_1 = int(np.argmin(np.sum((kp_f - np.array([0.0, 1 / 12])) ** 2, axis=1)))
    ratio_full = float((fl_full["box"][i_1, 0] - fl_full["box"][i_g, 0]) / dk2)
    ratio_loc = float((fl_loc["box"][i_1, 0] - fl_loc["box"][i_g, 0]) / dk2)
    print(f"deep band at Gamma on the box: m/m* = {ratio_full:.4f} (full), {ratio_loc:.4f} (local-only), from "
          f"Gamma and (0, 1/12)")
    for q in QS:
        L = np.load(DATA / f"route-p-q{q}-local.npz")
        F = np.load(DATA / f"route-p-q{q}-full.npz")
        e0l, e0f = float(np.min(fl_loc["box"][:, 0])), float(np.min(fl_full["box"][:, 0]))
        print(f"q {q}: local-only lowest {np.min(L['eigenvalues']):.6f} Ry, zero-field box minimum {e0l:.6f} Ry, "
              f"difference {(np.min(L['eigenvalues']) - e0l) * RY_EV * 1e3:+.2f} meV (B_ph {0.0335723149 * 10 / q * RY_EV * 1e3:.0f} meV); "
              f"full lowest minus its zero-field minimum {(np.min(F['eigenvalues']) - e0f) * RY_EV * 1e3:+.2f} meV "
              f"against (m/m*) B_ph = {ratio_full * 0.0335723149 * 10 / q * RY_EV * 1e3:.2f} meV (local-only "
              f"{ratio_loc * 0.0335723149 * 10 / q * RY_EV * 1e3:.2f} meV)", flush=True)

    # ---- item 4: DOS at q = 6
    print("\n== item 4: DOS at q = 6, Gaussian 20 meV, each about its own E_D, 4 representatives (route P's full "
          "mesh is 6 copies), states per eV per magnetic k")
    q = 6
    P = np.load(DATA / "route-p-q6-full.npz")
    rep = reps(P["kpoints"], q)
    kap = P["kpoints"][rep][:, :2]
    pe = P["eigenvalues"][rep] - ed_box
    sigma = 0.020 / RY_EV
    for half in (1.0, 2.5):
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


def representatives_run(q):
    """Route P's full H at 1/q on the four representatives only (q copies give the closed (2, 2q) mesh)."""
    import resource

    from defuqumat.workflows.magnetic_bands import run_magnetic_bands

    calc = graphene()
    kpts = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 1 / (2 * q), 0.0], [0.5, 1 / (2 * q), 0.0]]
    t0 = time.perf_counter()
    b = run_magnetic_bands(calc.system, calc.pseudos, SCF, flux=(1, q), kpoints=kpts, nbnd=4 * q + 8,
                           local_only=False)
    wall = time.perf_counter() - t0
    print(f"q {q}: {b.field_tesla:.1f} T, ndim {b.ndim}, nbnd {4 * q + 8}, steps {b.steps.tolist()}, unsettled "
          f"{b.unsettled.tolist()}, solver {sorted(set(b.solver))}, memory estimate {b.memory_estimate / 1e9:.3f} GB, "
          f"{wall:.0f} s, peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"route-p-q{q}-reps{TAG}.npz", ecut=ECUT, eigenvalues=b.eigenvalues, kpoints=b.kpoints, steps=b.steps,
             flux=np.array([1, q]), box=np.array(b.box), field_tesla=b.field_tesla, wall_s=wall,
             description="route P full H (GIPAW projectors), graphene 50 Ry, the four representatives only; Ry")
    print(f"ecut {ECUT} Ry, saved {OUT / f'route-p-q{q}-reps{TAG}.npz'}")


def cutoff(other):
    """Route P's zero pair against Peierls at 50 Ry and at ``other`` Ry, each about its own box E_D.

    A field-induced grid effect of the box moves with the cutoff; the Peierls error does not.
    """
    from defuqumat.magnetic.landau import dirac_point
    from defuqumat.magnetic.peierls import bonds_from_wannier

    w = wannier("pz")
    bonds = bonds_from_wannier(w)
    pt = dirac_point(w.lattice, w.bands, w.edge)
    tag = f"-{float(other):g}"
    for label, fname in (("50 Ry", "floor-full.npz"), (f"{other} Ry", f"floor-full{tag}.npz")):
        f = np.load(OUT / fname)
        ik = [i for i, kp in enumerate(f["kpoints"]) if np.allclose(kp[:2], pt["K"][:2])][0]
        print(f"{label}: box E_D at K {dirac_pair(f['box'][ik], float(f['fermi_energy']))[0]:.10f} Ry, sphere "
              f"{dirac_pair(f['sphere'][ik], float(f['fermi_energy']))[0]:.10f} Ry")
    for q in (6, 10):
        rows = {}
        for label, fl, rp in (("50 Ry", "floor-full.npz", route_p_full(q)),
                              (f"{other} Ry", f"floor-full{tag}.npz",
                               np.load(OUT / f"route-p-q{q}-reps{tag}.npz") if (OUT / f"route-p-q{q}-reps{tag}.npz").exists() else None)):
            if rp is None:
                print(f"q {q} {label}: no route P set")
                continue
            f = np.load(OUT / fl)
            ik = [i for i, kp in enumerate(f["kpoints"]) if np.allclose(kp[:2], pt["K"][:2])][0]
            ed = dirac_pair(f["box"][ik], float(f["fermi_energy"]))[0]
            rep = reps(rp["kpoints"], q)
            kap = rp["kpoints"][rep][:, :2]
            pz = np.sort((np.sort(rp["eigenvalues"][rep], axis=1)[:, 4 * q - 1:4 * q + 1] - ed).ravel())
            lev, _ = peierls_levels(w, bonds, q, kap)
            wz = np.sort((lev[:, q - 1:q + 1] - pt["E_D"]).ravel())
            rows[label] = pz
            print(f"q {q} {label}: zero pair about the box E_D {np.array2string(pz * RY_EV, precision=4)} eV; Peierls "
                  f"{np.array2string(wz * RY_EV, precision=4)} eV; max |P - W| {np.max(np.abs(pz - wz)) * RY_EV * 1e3:.1f} "
                  f"meV, mean P - W {np.mean(pz - wz) * RY_EV * 1e3:+.1f} meV", flush=True)
        if len(rows) == 2:
            a, b = rows.values()
            print(f"q {q}: route P's zero pair, {other} Ry minus 50 Ry: max {np.max(np.abs(b - a)) * RY_EV * 1e3:.1f} meV, "
                  f"mean {np.mean(b - a) * RY_EV * 1e3:+.1f} meV", flush=True)


def diag():
    """Is route P's zero pair against Peierls a labelling offset? Route U beside both, and a kappa scan."""
    from defuqumat.magnetic.landau import dirac_point
    from defuqumat.magnetic.peierls import bonds_from_wannier

    f = np.load(OUT / "floor-full.npz")
    ef = float(f["fermi_energy"])
    w = wannier("pz")
    bonds = bonds_from_wannier(w)
    pt = dirac_point(w.lattice, w.bands, w.edge)
    ik = [i for i, kp in enumerate(f["kpoints"]) if np.allclose(kp[:2], pt["K"][:2])][0]
    ed_box, ed_sph = dirac_pair(f["box"][ik], ef)[0], dirac_pair(f["sphere"][ik], ef)[0]
    for q in (2, 3, 4, 6):
        P = np.load(DATA / f"route-p-q{q}-full.npz")
        U = np.load(DATA / f"route-u-q{q}-full.npz")
        rep = reps(P["kpoints"], q)
        kap = P["kpoints"][rep][:, :2]
        pz = np.sort((np.sort(P["eigenvalues"][rep], axis=1)[:, 4 * q - 1:4 * q + 1] - ed_box).ravel())
        lev, _ = peierls_levels(w, bonds, q, kap)
        wz = np.sort((lev[:, q - 1:q + 1] - pt["E_D"]).ravel())
        print(f"q {q}: zero pair (eV), P about the box E_D {np.array2string(pz * RY_EV, precision=4)}; "
              f"W about its E_D {np.array2string(wz * RY_EV, precision=4)}", flush=True)
        for nb in (8, 16, 32):
            uz = np.sort((np.sort(U[f"levels_nbnd{nb}"], axis=1)[:, 4 * q - 1:4 * q + 1] - ed_sph).ravel())
            print(f"      U nbnd {nb:2d} about the sphere E_D {np.array2string(uz * RY_EV, precision=4)}; "
                  f"mean U - P {np.mean(uz - pz) * RY_EV * 1e3:+.1f} meV, mean U - W {np.mean(uz - wz) * RY_EV * 1e3:+.1f} meV",
                  flush=True)
        # every kappa offset of the four representatives: does a shifted Peierls set match route P?
        best = (np.inf, None)
        for d1 in np.linspace(0.0, 0.5, 11):
            for d2 in np.linspace(0.0, 1.0 / (2 * q), 11):
                lv, _ = peierls_levels(w, bonds, q, kap + np.array([d1, d2]))
                ws = np.sort((lv[:, q - 1:q + 1] - pt["E_D"]).ravel())
                m = float(np.max(np.abs(pz - ws)))
                if m < best[0]:
                    best = (m, (d1, d2), float(np.mean(pz - ws)))
        print(f"      kappa scan of Peierls (11 x 11 offsets): smallest max |P - W| {best[0] * RY_EV * 1e3:.1f} meV at "
              f"offset {np.round(best[1], 4).tolist()}, mean there {best[2] * RY_EV * 1e3:+.1f} meV", flush=True)
        dense = np.array([(i / 10, j / (20 * q)) for i in range(10) for j in range(20 * q)])
        lv, _ = peierls_levels(w, bonds, q, dense)
        lo, hi = lv[:, q - 1] - pt["E_D"], lv[:, q] - pt["E_D"]
        print(f"      Peierls zero pair over a 10 x {20 * q} magnetic mesh: lower member {lo.min() * RY_EV:+.4f} to "
              f"{lo.max() * RY_EV:+.4f} eV, upper {hi.min() * RY_EV:+.4f} to {hi.max() * RY_EV:+.4f} eV, mean of the "
              f"pair {0.5 * np.mean(lo + hi) * RY_EV * 1e3:+.1f} meV; route P's 4-point mean {np.mean(pz) * RY_EV * 1e3:+.1f} meV, "
              f"Peierls' 4-point mean {np.mean(wz) * RY_EV * 1e3:+.1f} meV", flush=True)


if __name__ == "__main__":
    t0 = time.perf_counter()
    cmd = sys.argv[1]
    if cmd == "exact":
        exact()
    elif cmd == "floor":
        floor(sys.argv[2])
    elif cmd == "analyse":
        analyse()
    elif cmd == "diag":
        diag()
    elif cmd == "cutoff":
        cutoff(sys.argv[2])
    elif cmd == "reps":
        representatives_run(int(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
    print(f"wall {time.perf_counter() - t0:.1f} s")
