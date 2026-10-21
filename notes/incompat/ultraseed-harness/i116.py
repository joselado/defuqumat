"""I116: the collinear field seed against the tiled start, on two-cell hydrogen.

The unit cell is ``tests/regression/test_ultracell.py``'s ``HYDROGEN`` (a = 5.5,
nspin = 2, moment 0.62 on the folded (4, 2, 2) grid), the ultracell (2, 1, 1)
at kgrid (2, 2, 2), nbnd = 16, under B cos(pi x). Three starts per field and
per conv_thr: tiled (seed_from_field=False), the field's seed
(seed_from_field=True, sign(B) sign(M)), and the wrong-sign domain start
(seed_magnetization = -sign(B)), the control a sign error would be.

The driver still refuses seed_from_field=True at nspin_mag = 2 (the item is
partial), so the field's seed is handed over as seed_magnetization, evaluated by
``seed.collinear_field_seed`` itself on the same box coordinates the field is,
which is exactly what the lifted branch did (commit 8c7bf8e5).

Usage: i116.py <out.json> [amplitudes, comma-separated] [conv_thrs] [nbnd]
"""
import json, sys, time, warnings
from pathlib import Path
import tempfile

import numpy as np
import jax

from tests.regression.test_ultracell import _hydrogen, _fourier
from defuqumat.ultracell import run_ultracell
from defuqumat.ultracell.seed import collinear_field_seed

out = sys.argv[1]
amplitudes = [float(a) for a in (sys.argv[2] if len(sys.argv) > 2 else "0.002,0.02").split(",")]
thresholds = [float(a) for a in (sys.argv[3] if len(sys.argv) > 3 else "1e-8,1e-10").split(",")]
NBND = int(sys.argv[4]) if len(sys.argv) > 4 else 16

tmp = Path(tempfile.mkdtemp())
shape, kgrid = (2, 1, 1), (2, 2, 2)
folded = tuple(n * m for n, m in zip(shape, kgrid))
calc = _hydrogen(tmp, "tests/data/pseudo", folded)
t0 = time.time()
scf = calc.get_scf(conv_thr=1e-12, nbnd=8)
volume = float(calc.system.cell.volume)
records = dict(reference=dict(converged=bool(scf.converged),
                              magnetization=float(scf.magnetization),
                              total_energy=float(scf.total_energy),
                              seconds=time.time() - t0), runs=[])
print(json.dumps(records["reference"]), flush=True)


def summarise(result, tag, amplitude, thr):
    rho = np.asarray(result.density)
    box = tuple(result.ultracell.grid)
    m = rho[0] - rho[1]
    mq = float(np.abs(_fourier(m, box, np.array([[1, 0, 0]])))[0]) * volume
    m0 = float(np.real(_fourier(m, box, np.array([[0, 0, 0]])))[0]) * volume
    return dict(tag=tag, B=amplitude, conv_thr=thr, converged=bool(result.converged),
                iterations=int(result.iterations),
                first_dr2=float(result.history[0][0]),
                seed_origin=result.seed_origin,
                energy=float(result.total_energy) / 2,
                field_energy=float(result.field_energy or 0.0) / 2,
                minimised=(float(result.total_energy) + float(result.field_energy or 0.0)) / 2,
                cell_moments=[float(v) for v in np.asarray(result.cell_moments())],
                m_Q=mq, m_0=m0, seconds=float(result.seconds)), rho


for amplitude in amplitudes:
    field = lambda x, a=amplitude: a * np.cos(np.pi * x[..., 0])
    anti = lambda x: -np.sign(np.cos(np.pi * x[..., 0]))
    sign_m = float(np.sign(scf.magnetization))

    def from_field(x, f=field):
        b = f(x)
        seed, reason = collinear_field_seed(b, np.full(b.shape, sign_m), 1.0)
        assert reason is None, reason
        return seed
    for thr in thresholds:
        densities = {}
        for tag, options in (("tiled", dict(seed_from_field=False)),
                             ("field", dict(seed_magnetization=from_field)),
                             ("anti", dict(seed_magnetization=anti))):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_ultracell(
                    calc.system, calc.pseudos, scf, shape, kgrid, nbnd=NBND,
                    magnetic_field=field, conv_thr=thr, states_conv_thr=1e-9,
                    max_iterations=400, **options)
            rec, rho = summarise(result, tag, amplitude, thr)
            densities[tag] = rho
            if tag != "tiled":
                ref = densities["tiled"]
                rec["drho_vs_tiled"] = float(np.abs(rho - ref).max() / np.abs(ref).max())
            records["runs"].append(rec)
            print(json.dumps(rec), flush=True)
            jax.clear_caches()
        with open(out, "w") as handle:
            json.dump(records, handle, indent=1)
records["seconds"] = time.time() - t0
with open(out, "w") as handle:
    json.dump(records, handle, indent=1)
print("# done", records["seconds"])
