"""I73 feedback round: Newton with the driven field in its state, against the mixing loop.

argv: any of fsm, afsm, afsmdir, optionally prefixed 'mutant-' (the field row of
F replaced by the identity, so Newton holds the starting field and solves the
unconstrained problem; the mixing loop after the hand-over then has to drive
the field itself, which is what the iteration count after the hand-over sees)
or 'free-' (the cell with no constraint, the answer a solver that lost the
constraint would land on). A trailing ':<conv_thr>' sets the threshold
(default 1e-11).

* fsm      tests/data/qe/fe-fsm.in, collinear, one k-point, target 2.0 mu_B
* afsm     tests/data/qe/fe2-canted-nosoc.in under 'atomic fsm', targets 1.868
           along x and y (test_holding_a_texture._iron), Elk's adaptive mixer
* afsmdir  the same cell under 'atomic fsm direction'
"""
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from defuqumat import Calculator  # noqa: E402
from defuqumat.scf import fields as fields_module  # noqa: E402

PSEUDO = "tests/data/pseudo"
IRON = "tests/data/qe/fe2-canted-nosoc.in"


def text_of(case, free=False):
    if case == "fsm":
        text = Path("tests/data/qe/fe-fsm.in").read_text()
        if free:
            text = (text.replace("    constrained_magnetization = 'fsm'\n", "")
                    .replace("    lambda = 0.02\n", "")
                    .replace("    fixed_magnetization(3) = 2.0\n", ""))
        return text
    scheme = {"afsm": "atomic fsm", "afsmdir": "atomic fsm direction"}[case]
    text = Path(IRON).read_text()
    if free:
        return text
    text = text.replace(
        "    nosym = .true.\n",
        f"    nosym = .true.\n    constrained_magnetization = '{scheme}'\n"
        "    lambda = 0.02\n")
    rows = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) * 1.868
    return text + "STARTING_MOMENTS\n" + "".join(
        f"  {r[0]:.6f}  {r[1]:.6f}  {r[2]:.6f}\n" for r in rows)


def mixing_options(case):
    if case == "fsm":
        return {"mixing_beta": 0.3}
    return {"mixing_mode": "adaptive", "mixing_beta": 0.05}


def moment(r):
    return np.asarray(r.magnetization if r.magnetization is not None
                      else r.magnetization_vector, dtype=float)


def show(tag, r, t0):
    s = r.solver
    field = r.magnetic_field
    driven = None if field is None or field.driven_leaf is None else np.asarray(field.driven())
    sites = getattr(r, "site_moments", None)
    angle = ""
    if sites is not None and np.ndim(sites) == 2 and np.shape(sites)[0] == 2:
        sites = np.asarray(sites)
        hats = sites / np.linalg.norm(sites, axis=1)[:, None]
        angle = f" between={np.degrees(np.arccos(np.clip(hats[0] @ hats[1], -1, 1))):.6f}"
    print(f"{tag}: conv={r.converged} acc={r.accuracy:.3e} E={r.total_energy:.12f} "
          f"m={moment(r).tolist()} e_field={r.field_energy} loop_iterations={r.iterations}"
          f"{angle} time={time.perf_counter() - t0:.1f}s", flush=True)
    if sites is not None:
        print(f"   site_moments={np.asarray(sites).tolist()}", flush=True)
    if driven is not None:
        print(f"   driven_field={driven.tolist()}", flush=True)
    if s is not None:
        print(f"   newton conv={s.converged} acc={s.accuracy:.3e} outer={s.iterations} "
              f"steps={s.steps} jvps={s.jvps}", flush=True)


real_elk = fields_module.MagneticField.elk_driven

for arg in sys.argv[1:]:
    arg, _, thr = arg.partition(":")
    conv_thr = float(thr) if thr else 1e-11
    mutant = arg.startswith("mutant-")
    free = arg.startswith("free-")
    case = arg.removeprefix("mutant-").removeprefix("free-")
    if mutant:
        fields_module.MagneticField.elk_driven = lambda self, rho, cell: self.driven()
    else:
        fields_module.MagneticField.elk_driven = real_elk
    options = dict(mixing_options(case), conv_thr=conv_thr, max_iterations=400,
                   verbose=False)
    text = text_of(case, free=free)
    if not mutant:
        t0 = time.perf_counter()
        mixed = Calculator.from_text(text, PSEUDO, announce=False).get_scf(**options)
        show(f"{arg} mixing conv_thr={conv_thr}", mixed, t0)
    if free:
        continue
    t0 = time.perf_counter()
    newton = Calculator.from_text(text, PSEUDO, announce=False).get_scf(
        scf_solver="newton-krylov",
        scf_solver_options={"forcing": 0.5, "warmup": 6, "max_iterations": 40},
        **options)
    show(f"{arg} newton conv_thr={conv_thr}", newton, t0)
    if not mutant:
        print(f"   dE = {newton.total_energy - mixed.total_energy:.3e} Ry  "
              f"d e_field = {newton.field_energy - mixed.field_energy:.3e}  "
              f"dm = {(moment(newton) - moment(mixed)).tolist()}",
              flush=True)
        a = np.asarray(newton.magnetic_field.driven())
        b = np.asarray(mixed.magnetic_field.driven())
        print(f"   d field max = {np.max(np.abs(a - b)):.3e} Ry", flush=True)
        if getattr(newton, "site_moments", None) is not None:
            print(f"   d site moments max = "
                  f"{np.max(np.abs(np.asarray(newton.site_moments) - np.asarray(mixed.site_moments))):.3e}",
                  flush=True)
print("# done", flush=True)
