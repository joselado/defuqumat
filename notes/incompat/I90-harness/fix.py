"""I90 fix round: the wedge completion on a polar norm-conserving crystal.

python3 fix.py OUTDIR, from the worktree root (through slot.sh on D22).

AlAs (alas-raman.in, 10 Ry) with As at a(1/4,1/4,0.29), mm2. For each mesh
4, 6, 8: the default 'autodiff' route on the symmetry-reduced k-set read from
the card (nosym and noinv removed) with the completion on, the same with it
switched off (full_zone=False, the falsifier), 'zstar_eu' on the same wedge,
and 'autodiff' on the closed grid (nosym). Then zincblende (z = 0.25) on its
wedge, where the cheaper non-polar branch is taken and must be unchanged.
One child process per run, so each sits in its own JAX state.
"""
from __future__ import annotations

import functools
import json
import multiprocessing
import queue as _queue
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
PSEUDO = ROOT / "tests" / "data" / "pseudo"
BASE = ROOT / "tests" / "data" / "qe" / "alas-raman.in"
CONV_THR = 1.0e-10


def _text(z, mesh, wedge):
    t = BASE.read_text()
    for old, new in ((" 4 4 4 0 0 0", f" {mesh} {mesh} {mesh} 0 0 0"),
                     (" As 0.25 0.25 0.25", f" As 0.25 0.25 {z:.4f}")):
        assert old in t, old
        t = t.replace(old, new)
    if wedge:
        for old in ("    nosym = .true.\n", "    noinv = .true.\n"):
            assert old in t
            t = t.replace(old, "")
    return t


def _resp(p, q):
    from defuqumat.calculator import Calculator
    import defuqumat.response.piezo as piezo
    if not p["full_zone"]:
        piezo.clamped_ion_piezoelectric = functools.partial(
            piezo.clamped_ion_piezoelectric, full_zone=False)
    tag = f"z{p['z']:.4f}-m{p['mesh']}-{'w' if p['wedge'] else 'n'}"
    path = Path(p["outdir"]) / f"{tag}.in"
    path.write_text(_text(p["z"], p["mesh"], p["wedge"]))
    c = Calculator.from_file(path, pseudo_dir=str(PSEUDO), announce=False)
    t = time.time()
    c.get_scf(conv_thr=CONV_THR)
    tensor = c.get_piezoelectric_tensor(method=p["method"], kmesh_warning=False)
    kp = c.calculation.system.kpoints
    q.put({"e": np.asarray(tensor.e).tolist(), "converged": bool(tensor.converged),
           "nk": int(kp.nk), "reduced": bool(getattr(kp, "reduced", False)),
           "nsym": int(c.calculation.symmetries.nsym), "seconds": time.time() - t})


def _guard(p, q):
    try:
        _resp(p, q)
    except BaseException as exc:  # noqa: BLE001
        traceback.print_exc()
        q.put({"error": repr(exc)})


def spawn(**p):
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    ch = ctx.Process(target=_guard, args=(p, q))
    ch.start()
    while True:
        try:
            r = q.get(timeout=30)
            break
        except _queue.Empty:
            if not ch.is_alive():
                return {"error": f"child died, exit {ch.exitcode}"}
    ch.join()
    return r


def comps(e):
    e = np.asarray(e)
    return {"e31": e[2, 0, 0], "e32": e[2, 1, 1], "e33": e[2, 2, 2],
            "e15": e[0, 0, 2], "e14": e[0, 1, 2]}


def main():
    from defuqumat.units import E_BOHR2_TO_C_M2 as C
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rec = {"runs": []}

    def resp(z, mesh, wedge, method, full_zone=True):
        r = spawn(outdir=str(out), z=z, mesh=mesh, wedge=wedge, method=method,
                  full_zone=full_zone)
        r.update(z=z, mesh=mesh, wedge=wedge, method=method, full_zone=full_zone)
        if "e" in r:
            r["C_m2"] = {k: round(float(v) * C, 7) for k, v in comps(r["e"]).items()}
        rec["runs"].append(r)
        print("resp", {k: r.get(k) for k in ("z", "mesh", "wedge", "method",
                                             "full_zone", "nk", "reduced", "nsym",
                                             "seconds", "C_m2", "error")}, flush=True)
        (out / "result-fix.json").write_text(json.dumps(rec, indent=1, default=float))

    for mesh in (4, 6, 8):
        resp(0.29, mesh, True, "autodiff", True)
        resp(0.29, mesh, True, "autodiff", False)
        resp(0.29, mesh, True, "zstar_eu")
        resp(0.29, mesh, False, "autodiff", True)
    resp(0.25, 4, True, "autodiff", True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
