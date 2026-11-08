"""Route U step 3, the acceptance proper: graphene on the projected atomic trials against route P.

The trials are one nested list, p_z of both atoms (J = 2), then the other pseudo-atomic orbitals of
C.pbe-hgh.UPF's PP_CHI (2s, 2p: J = 8), then hydrogenic d at zona Z on both atoms (J = 18). One run
at J = 18 with keep_blocks gives the blocks of every subset, since the trials are independent
functions and the blocks of J = 2 and 8 are the leading sub-blocks of those of J = 18. N = 32,
window (3, 4), N1 16, h0 box, form symmetric, the shared zero-field potential graphene-scf.npz.

    python mag_route_u3_graphene.py zero <q> full|local [Z]   p = 0 at the chemical k that fold onto
                                                              the representatives of 1/q
    python mag_route_u3_graphene.py field <q> full|local [Z]  flux 1/q, the four representatives
    python mag_route_u3_graphene.py analyse [Z]               the comparison, from the npz files
    python mag_route_u3_graphene.py span full|local [Z]       p = 0 on the N1 mesh against the bands
                                                              (MAG_WINDOW=none MAG_N1=15: no window)

Output and input directory $MAG_OUT (default /l/ladovj1/review/mag/u3). Energies in Ry unless said.
The comparison is fable's (2026-11-08, Q4 and Q5 of the step's decisions): per representative,
route P's levels within E_D +- 2.5 eV (E_D on route P's box), indexed relative to the zero pair
(P and the 8- and 18-function sets at 4q - 1, 4q; the p_z set at q - 1, q), compared at equal
relative index in absolute energy, so that the variational order U >= P can be read. The pass
quantity is (U - U0) - (P - B0), with U0 the projected basis at p = 0 folded onto the
representative, B0 and S0 route P's box and the sphere at p = 0 (cross-floor-*.npz); (U - P),
the span error U0 - S0 and the floor S0 - B0 are printed beside it.
"""
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path("plans/orbital-field")
OUT = Path(os.environ.get("MAG_OUT", "/l/ladovj1/review/mag/u3"))
RY_EV = 13.605693122990
ED_BOX = -0.130751159          # route P's E_D on its own box at K (cross.md, item 3)
ED_SPHERE = -0.119587541       # the sphere's E_D at K (cross.md, item 3)
JS = (2, 8, 18)
# the band window; 32 is the design's, another value (MAG_N=48) tags the files, as a reported
# number and not a nested knob (fable's decision, "if 5 mRy is not met at J 18")
NBANDS = int(os.environ.get("MAG_N", "32"))
NTAG = "" if NBANDS == 32 else f"-N{NBANDS}"
# the window and N1, a diagnostic of the span error (MAG_WINDOW=none MAG_N1=15: the bare projected
# trials, odd N1 and every Delta of the box, so at p = 0 on the N1 mesh the levels are the
# Rayleigh-Ritz values of the projected trials at that k alone)
_WIN = os.environ.get("MAG_WINDOW", "3,4")
WINDOW = None if _WIN == "none" else tuple(float(x) for x in _WIN.split(","))
N1 = int(os.environ.get("MAG_N1", "16"))
if (WINDOW, N1) != ((3.0, 4.0), 16):
    NTAG += f"-w{_WIN.replace(',', '_')}-n{N1}"


def graphene():
    from defuqumat import Calculator
    from defuqumat.units import RY_TO_EV

    return Calculator.from_keywords(
        lattice="hexagonal", a=2.46, c=15.0, atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
        pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
        ecut=50 * RY_TO_EV, kmesh=(1, 1, 1), smearing=0.01 * RY_TO_EV, smearing_type="mv")


def nested_trials(calc, zona):
    from defuqumat.wannier.pao import pao_projections
    from defuqumat.wannier.projections import parse_projections

    system = calc.system
    crystal = np.asarray(system.structure.positions_crystal(system.cell), dtype=float)
    paos = pao_projections(calc.pseudos, system.structure, crystal)
    pz = [p for p in paos if (p.l, p.m) == (1, 1)]
    rest = [p for p in paos if (p.l, p.m) != (1, 1)]
    d = parse_projections([f"C:d:zona={zona}"], list(system.structure.symbols), crystal.T,
                          np.asarray(system.cell.bg, dtype=float))
    trials = pz + rest + d
    assert len(pz) == 2 and len(trials) == 18, (len(pz), len(rest), len(d))
    return trials


def run(kind, q, op, zona):
    from defuqumat.magnetic.ultracell_field import assemble, hermiticity, solve
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = graphene()
    trials = nested_trials(calc, zona)
    if kind == "zero":
        flux, kmesh, kappa2 = (0, 1), (2 * q, 1), [0.0, 1.0 / (2 * q)]
    else:
        flux, kmesh, kappa2 = (1, q), (2, 2), None
    t0 = time.perf_counter()
    r = run_magnetic_ultracell(calc.system, calc.pseudos, str(DATA / "graphene-scf.npz"), flux=flux,
                               N=NBANDS, trials=trials, kmesh=kmesh, kappa2=kappa2, window=WINDOW,
                               N1=N1, w=None if WINDOW is not None else (N1 - 1) // 2,
                               local_only=(op == "local"), keep_blocks=True, strict=False,
                               projectability_tol=None)
    wall = time.perf_counter() - t0
    qq = flux[1]
    kap2 = sorted({float(k[1]) for k in r.kappa})
    kap1 = [i / kmesh[0] for i in range(kmesh[0])]
    print(f"== {kind} q {q} {op} zona {zona}: flux {flux}, {r.field_tesla:.1f} T, w {r.w}, window {r.window}, "
          f"N1 {r.n1}, N {r.N}, nonlocal {r.nonlocal_term}/{r.nonlocal_mode}, nonlocal_error "
          f"{r.nonlocal_error}, wall {wall:.1f} s, timings { {k: round(v, 1) for k, v in r.timings.items()} }, "
          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)
    saved = {"kappa1": np.array(kap1), "kappa2": np.array(kap2), "trials": np.array(r.trials),
             "eigenvalues": r.eigenvalues, "projectability": r.projectability, "wall_s": wall,
             "flux": np.array(flux), "w": r.w}
    for J in JS:
        lost = 1.0 - r.projectability[..., :J].min(axis=(0, 1, 2))
        levels, herm, rank, dmin = [], [], [], []
        for jk in range(len(kap2)):
            B, S = r.blocks[jk]
            B, S = np.asarray(B)[..., :J, :J], np.asarray(S)[..., :J, :J]
            for k1 in kap1:
                h, s = assemble(k1, B, S, r.w, qq)
                herm.append(hermiticity(h))
                eps, rk, dm, da = solve(h, s, 1e-8)
                levels.append(eps[:qq * J] if len(eps) >= qq * J else np.pad(eps, (0, qq * J - len(eps)),
                                                                             constant_values=np.nan))
                rank.append(rk)
                dmin.append(dm)
        levels = np.array(levels).reshape(len(kap2), len(kap1), -1)
        print(f"J {J:2d}: 1 - projectability max {lost.max():.3e} (per trial "
              f"{np.array2string(lost, precision=2, max_line_width=300)}), Hermiticity max {max(herm):.2e}, rank "
              f"{min(rank)} to {max(rank)} of {qq * J}, d_min {min(dmin):.2e}", flush=True)
        saved[f"levels_J{J}"] = levels
        saved[f"herm_J{J}"] = np.array(herm)
        saved[f"rank_J{J}"] = np.array(rank)
        saved[f"dmin_J{J}"] = np.array(dmin)
        saved[f"lost_J{J}"] = lost
    path = OUT / f"u3-{kind}-q{q}-{op}-z{zona:g}{NTAG}.npz"
    np.savez(path, **saved)
    print(f"saved {path}", flush=True)


def span(op, zona):
    """p = 0 on the N1 mesh itself (kappa1 = i/N1, kappa2 in {0, 1/12}): every level of J = 2, 8, 18
    against the zero-field bands of the same NSCF at the same k, so the span error is read with no
    folding and no floor file. With MAG_WINDOW=none and an odd MAG_N1 it is the Rayleigh-Ritz value
    of the projected trials at that k alone; with the window, what the window cuts as well."""
    from defuqumat.magnetic.ultracell_field import assemble, solve
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = graphene()
    trials = nested_trials(calc, zona)
    r = run_magnetic_ultracell(calc.system, calc.pseudos, str(DATA / "graphene-scf.npz"), flux=(0, 1),
                               N=NBANDS, trials=trials, kmesh=(N1, 1), kappa2=[0.0, 1.0 / 12],
                               window=WINDOW, N1=N1, w=None if WINDOW is not None else (N1 - 1) // 2,
                               local_only=(op == "local"), keep_blocks=True, strict=False,
                               projectability_tol=None)
    print(f"== span {op} zona {zona}: window {r.window}, N1 {r.n1}, w {r.w}, N {r.N}", flush=True)
    for J in JS:
        worst_low, worst_near, above = 0.0, 0.0, True
        for jk in range(2):
            B, S = (np.asarray(x)[..., :J, :J] for x in r.blocks[jk])
            for i in range(N1):
                eps, _, _, _ = solve(*assemble(i / N1, B, S, r.w, 1), 1e-8)
                bands = r.eigenvalues[jk, 0, i]
                n = min(len(eps), 4)
                worst_low = max(worst_low, float(np.max(np.abs(eps[:n] - bands[:n]))) if J >= 8 else 0.0)
                above &= bool(np.all(eps - bands[:len(eps)] > -1e-9))
                for level in eps[np.abs(eps - ED_SPHERE) < 2.5 / RY_EV]:
                    worst_near = max(worst_near, float(np.min(np.abs(bands - level))))
        print(f"J {J:2d}: levels within E_D +- 2.5 eV, nearest band max {worst_near * RY_EV * 1e3:.2f} meV; "
              + (f"lowest 4 against the lowest 4 bands max {worst_low * RY_EV * 1e3:.2f} meV; " if J >= 8 else "")
              + f"every level above its band: {above}", flush=True)


def folded_union(levels_by_k, kpoints, kappa1, kappa2, q):
    """Sorted union of the zero-field levels at the chemical k ((kappa1 + j)/q, kappa2), j < q."""
    key = {(round(a, 9), round(b, 9)): i for i, (a, b) in enumerate(np.asarray(kpoints)[:, :2])}
    idx = [key[(round(((kappa1 + j) / q) % 1.0, 9), round(kappa2, 9))] for j in range(q)]
    rows = np.asarray(levels_by_k)[idx]
    return np.sort(rows.ravel()), float(np.min(rows[:, -1]))


def analyse(zona):
    for op, pfile in (("full", None), ("local", None)):
        floor = np.load(DATA / f"cross-floor-{op}.npz")
        for q in ((6, 10) if op == "full" else (6,)):
            fz = OUT / f"u3-zero-q{q}-{op}-z{zona:g}{NTAG}.npz"
            ff = OUT / f"u3-field-q{q}-{op}-z{zona:g}{NTAG}.npz"
            if not (fz.exists() and ff.exists()):
                print(f"{op} q {q}: missing {fz.name} or {ff.name}")
                continue
            Z, F = np.load(fz), np.load(ff)
            P = np.load(DATA / (f"route-p-q{q}-full.npz" if (op == "full" and q == 6) else
                                f"route-p-q{q}-reps.npz" if op == "full" else f"route-p-q{q}-local.npz"))
            pk = np.asarray(P["kpoints"])
            zk = np.array([[k1, k2] for k2 in Z["kappa2"] for k1 in Z["kappa1"]])
            print(f"\n== {op} H, q {q}, N {NBANDS}: route P {P['eigenvalues'].shape[1]} levels per k; E_D (box) {ED_BOX} Ry; "
                  f"U field wall {float(F['wall_s']):.0f} s, U zero wall {float(Z['wall_s']):.0f} s")
            for J in JS:
                zl = Z[f"levels_J{J}"].reshape(-1, J)                    # q = 1: J levels per chemical k
                rows = {k: [] for k in ("UP", "line1", "floor", "pass", "U", "P")}
                counts, guard_hit, below = [], 0, 0
                for a, k2 in enumerate(F["kappa2"]):
                    for b, k1 in enumerate(F["kappa1"]):
                        U = np.sort(F[f"levels_J{J}"][a, b])
                        sel = np.where((np.abs(pk[:, 0] - k1) < 1e-9) & (np.abs(pk[:, 1] - k2) < 1e-9))[0]
                        Pk = np.sort(P["eigenvalues"][sel[0]])
                        U0, _ = folded_union(zl, zk, k1, k2, q)
                        B0, b0top = folded_union(floor["box"], floor["kpoints"], k1, k2, q)
                        S0, s0top = folded_union(floor["sphere"], floor["kpoints"], k1, k2, q)
                        if op == "full":
                            zp_p, zp_u = 4 * q - 1, (q - 1 if J == 2 else 4 * q - 1)
                            idx = np.where(np.abs(Pk - ED_BOX) < 2.5 / RY_EV)[0]
                            counts.append((int(np.sum(np.abs(U - ED_BOX) < 2.5 / RY_EV)), len(idx)))
                            inside = U[np.abs(U - ED_BOX) < 2.5 / RY_EV]
                            print(f"   J {J:2d} rep ({k1:.2f}, {k2:.4f}): U levels below E_D {int(np.sum(U < ED_BOX))} "
                                  f"(zero pair expects {zp_u + 1}), P {int(np.sum(Pk < ED_BOX))} (expects {zp_p + 1}); "
                                  f"U in the window {np.array2string((inside - ED_BOX) * RY_EV, precision=3)} eV, P "
                                  f"{np.array2string((Pk[idx] - ED_BOX) * RY_EV, precision=3)} eV", flush=True)
                        else:
                            if J == 2:
                                break
                            zp_p = zp_u = 0
                            idx = np.arange(4 * q)
                        for i in idx:
                            r = i - zp_p
                            iu, i0 = zp_u + r, zp_p + r
                            if not (0 <= iu < len(U)) or i0 >= len(B0) or B0[i0] >= b0top or S0[i0] >= s0top:
                                guard_hit += 1
                                continue
                            up = U[iu] - Pk[i]
                            l1 = U0[iu] - S0[i0]
                            fl = S0[i0] - B0[i0]
                            for key, v in (("UP", up), ("line1", l1), ("floor", fl), ("pass", up - l1 - fl),
                                           ("U", U[iu]), ("P", Pk[i])):
                                rows[key].append(v)
                            below += int(up < -1e-8)
                if not rows["UP"]:
                    print(f"J {J:2d}: nothing compared")
                    continue
                v = {k: np.array(x) for k, x in rows.items()}
                print(f"J {J:2d}: {v['UP'].size} levels; U - P {v['UP'].min():+.3e} to {v['UP'].max():+.3e} "
                      f"(U < P - 1e-8: {below}); span error U0 - S0 {v['line1'].min():+.3e} to {v['line1'].max():+.3e}; "
                      f"floor S0 - B0 {v['floor'].min():+.3e} to {v['floor'].max():+.3e}; (U - U0) - (P - B0) "
                      f"{v['pass'].min():+.3e} to {v['pass'].max():+.3e}, max abs {np.max(np.abs(v['pass'])):.3e} Ry "
                      f"({np.max(np.abs(v['pass'])) * RY_EV * 1e3:.1f} meV), mean {v['pass'].mean():+.3e}; "
                      f"(U - P) - (U0 - S0) max abs {np.max(np.abs(v['UP'] - v['line1'])):.3e}; skipped {guard_hit}; "
                      f"Hermiticity max {F[f'herm_J{J}'].max():.1e}, 1 - p max {F[f'lost_J{J}'].max():.1e}"
                      + (f"; window counts (U, P) per rep {counts}" if counts else ""), flush=True)
                print(f"      U - E_D (eV) {np.array2string((v['U'] - ED_BOX) * RY_EV, precision=4, max_line_width=400)}")
                print(f"      P - E_D (eV) {np.array2string((v['P'] - ED_BOX) * RY_EV, precision=4, max_line_width=400)}")
                # the zero-field span error as cross.md measured the Wannier sets: every U0 level within
                # E_D +- 2.5 eV (the sphere's E_D) at each folded chemical k, against the nearest sphere band
                fkey = {(round(x, 9), round(y, 9)): i for i, (x, y) in enumerate(floor["kpoints"][:, :2])}
                near = []
                for row, (x, y) in zip(zl, zk):
                    sph = floor["sphere"][fkey[(round(x % 1.0, 9), round(y, 9))]]
                    for level in row[np.abs(row - ED_SPHERE) < 2.5 / RY_EV]:
                        near.append(np.min(np.abs(sph - level)))
                near = np.array(near)
                if near.size:
                    print(f"      zero field, U0 within E_D +- 2.5 eV on the {len(zk)} folded k: {near.size} levels, "
                          f"nearest sphere band max {near.max() * RY_EV * 1e3:.2f} meV, mean "
                          f"{near.mean() * RY_EV * 1e3:.2f} meV", flush=True)
            if op == "full":
                # mirror z -> -z: s, px, py are even, p_z and the pi levels odd, so the odd levels of J = 8
                # are those of J = 2 (the same numbers check the slicing of the J = 18 blocks)
                worst = 0.0
                for a in range(len(F["kappa2"])):
                    for b in range(len(F["kappa1"])):
                        u8 = np.sort(F["levels_J8"][a, b])
                        worst = max(worst, max(float(np.min(np.abs(u8 - x))) for x in F["levels_J2"][a, b]))
                print(f"   mirror: every J = 2 level is a J = 8 level to {worst:.2e} Ry", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd in ("zero", "field"):
        run(cmd, int(sys.argv[2]), sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 2.0)
    elif cmd == "span":
        span(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 2.0)
    elif cmd == "analyse":
        analyse(float(sys.argv[2]) if len(sys.argv) > 2 else 2.0)
    else:
        raise SystemExit(__doc__)
