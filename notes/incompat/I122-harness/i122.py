"""I122: the ultrasoft oxygen chain's quarter turn against its four-cell supercell at 240 Ry.

At ``ecutrho = 240`` the chain's resident dense box is (60, 60, 25) and the
spiral's laboratory sphere ``|G - q| < G_rho`` at ``q = b3/4`` spans 26 Miller
indices along ``c``, which used to be refused. The widened box is (60, 60, 27),
and the four-cell supercell's own box at the same cutoff is (60, 60, 108) =
4 x 27, so the identity of ``test_a_quarter_turn_is_the_ninety_degree_supercell``
holds on it exactly, to the SCF's convergence. Run at the ``conv_thr`` given.

    python3 i122.py out.json 1e-11
"""

import json
import sys
import time

import numpy as np

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import run_scf
from defuqumat.system import build_system
from pathlib import Path

from tests.conftest import GENERATED

PSEUDO_DIR = Path(__file__).resolve().parents[3] / "tests" / "data" / "pseudo"

out, conv_thr = sys.argv[1], float(sys.argv[2])
ECUTRHO = 240.0


def text(name, q3=None):
    t = (GENERATED / name).read_text().replace("ecutrho = 200.0", f"ecutrho = {ECUTRHO}")
    assert f"ecutrho = {ECUTRHO}" in t
    if q3 is not None:
        t = t.replace("spiral_q(3) = 0.25", f"spiral_q(3) = {q3}")
    return t


def run(t):
    system = build_system(parse_pw_input(t))
    pseudos = tuple(read_upf(PSEUDO_DIR / s.pseudo_file) for s in system.structure.species)
    t0 = time.time()
    res = run_scf(system, pseudos, conv_thr=conv_thr, mixing_beta=0.3, max_iterations=300)
    return dict(grid=list(res.density.shape[1:]),
                converged=bool(res.converged), iterations=int(res.iterations),
                total=float(res.total_energy),
                electronic=float(res.total_energy - res.energy_terms["ewald"]),
                seconds=time.time() - t0)


record = dict(conv_thr=conv_thr, ecutrho=ECUTRHO)
record["spiral"] = run(text("o-chain-spiral-us.in"))
print(record, flush=True)
record["supercell"] = run(text("o-chain-90deg-us.in"))
record["difference_ry"] = record["spiral"]["electronic"] - record["supercell"]["electronic"] / 4.0
print(record, flush=True)
with open(out, "w") as f:
    json.dump(record, f, indent=1)
