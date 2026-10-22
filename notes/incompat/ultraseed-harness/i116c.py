"""I116, third round: the lifted driver route itself, seed_from_field=True.

The same two-cell hydrogen under B cos(pi x) as i116b.py, through
``run_ultracell(seed_from_field=True)`` rather than a hand seed, at B = 0.002
and 0.02, nbnd 16 and 24, against the supercell numbers of job 21118949
(aligned: -0.9368278750536042 and -0.9438090463689511 Ry per cell). Then the
refusal on a paramagnet: the nspin = 2 silicon cell, which converges to no
moment, must raise "seed cannot be taken".

Usage: i116c.py <out.json>
"""
import json, sys, time, warnings
from pathlib import Path
import tempfile

import numpy as np
import jax

from defuqumat import Calculator
from tests.regression.test_ultracell import _hydrogen, MAGNETIC
from defuqumat.ultracell import run_ultracell

out = sys.argv[1]
PSEUDO = str(Path("tests/data/pseudo").resolve())
SUPERCELL = {0.002: -0.9368278750536042, 0.02: -0.9438090463689511}
tmp = Path(tempfile.mkdtemp())
t0 = time.time()
shape, kgrid = (2, 1, 1), (2, 2, 2)
calc = _hydrogen(tmp, PSEUDO, (4, 2, 2))
scf = calc.get_scf(conv_thr=1e-12, nbnd=8)
records = dict(runs=[])
for b, exact in SUPERCELL.items():
    for nbnd in (16, 24):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_ultracell(calc.system, calc.pseudos, scf, shape, kgrid, nbnd=nbnd,
                              magnetic_field=lambda x, b=b: b * np.cos(np.pi * x[..., 0]),
                              conv_thr=1e-10, states_conv_thr=1e-9, seed_from_field=True,
                              max_iterations=200)
        total = float(r.total_energy) + float(r.field_energy)
        rec = dict(B=b, nbnd=nbnd, converged=bool(r.converged), iterations=int(r.iterations),
                   seed_origin=r.seed_origin, minimised=total, above=total - exact,
                   cell_moments=[float(v) for v in np.asarray(r.cell_moments())])
        records["runs"].append(rec)
        print(json.dumps(rec), flush=True)
        jax.clear_caches()

path = tmp / "si_mag.in"
path.write_text(MAGNETIC.replace("2 2 2 0 0 0", "2 1 1 0 0 0"))
si = Calculator.from_file(path, pseudo_dir=PSEUDO)
si_scf = si.get_scf(conv_thr=1e-9, nbnd=8)
try:
    run_ultracell(si.system, si.pseudos, si_scf, (2, 1, 1), (1, 1, 1), nbnd=8,
                  seed_from_field=True,
                  magnetic_field=lambda x: 0.01 * np.cos(np.pi * x[..., 0]))
    records["paramagnet"] = "NOT REFUSED"
except ValueError as error:
    records["paramagnet"] = str(error)
records["paramagnet_moment"] = float(si_scf.magnetization)
print(json.dumps({k: records[k] for k in ("paramagnet", "paramagnet_moment")}), flush=True)
records["seconds"] = time.time() - t0
with open(out, "w") as handle:
    json.dump(records, handle, indent=1)
print("# done", records["seconds"])
