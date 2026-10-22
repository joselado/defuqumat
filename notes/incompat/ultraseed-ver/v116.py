"""Verifier, I116: the driver's seed_from_field=True at nspin=2 against the explicit
two-atom supercell under the same field, at nbnd values and a k-mesh the implementer
did not use, the supercell itself at two nbnd, and the tiled start re-measured.

Usage: v116.py <out.json> <kgrid a,b,c> <ladder> <B list> <supercell nbnd list>
"""
import json, sys, time, warnings
from pathlib import Path
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
from defuqumat import Calculator
import defuqumat.scf.driver as scf_driver
from tests.regression.test_ultracell import _hydrogen, ANTIFERROMAGNET
from defuqumat.ultracell import run_ultracell

out = sys.argv[1]
kgrid = tuple(int(v) for v in sys.argv[2].split(","))
LADDER = [int(v) for v in sys.argv[3].split(",")]
amplitudes = [float(v) for v in sys.argv[4].split(",")]
SUPER = [int(v) for v in sys.argv[5].split(",")]
PSEUDO = str(Path("tests/data/pseudo").resolve())

FIELD = {"b": None}
_original = scf_driver.as_potential_components


def with_field(scalar, nspin):
    base = _original(scalar, nspin)
    b = FIELD["b"]
    if b is None or nspin != 2 or jnp.ndim(scalar) != 3:
        return base
    nx = scalar.shape[0]
    x = 2.0 * np.arange(nx) / nx
    field = jnp.asarray(b * np.cos(np.pi * x)[:, None, None] * np.ones(scalar.shape),
                        dtype=base.dtype)
    return base + jnp.stack([-field, field])


scf_driver.as_potential_components = with_field

tmp = Path(tempfile.mkdtemp())
t0 = time.time()
shape = (2, 1, 1)
folded = tuple(n * m for n, m in zip(shape, kgrid))
calc = _hydrogen(tmp, PSEUDO, folded)
scf = calc.get_scf(conv_thr=1e-12, nbnd=8)
rec = dict(kgrid=kgrid, magnetization=float(scf.magnetization), converged=bool(scf.converged),
           grid=list(np.asarray(scf.density).shape[1:]))
records = dict(reference=rec, ultracell=[], supercell=[])
print(json.dumps(rec), flush=True)


def save():
    records["seconds"] = time.time() - t0
    json.dump(records, open(out, "w"), indent=1)


for b in amplitudes:
    field = lambda x, b=b: b * np.cos(np.pi * x[..., 0])
    for tag, (m1, m2) in (("aligned", (0.8, -0.8)), ("ferro", (0.8, 0.8))):
        for snb in SUPER:
            path = tmp / f"pair_{tag}_{b:g}_{snb}.in"
            text = ANTIFERROMAGNET.format(m2=m2, k0=kgrid[0], k1=kgrid[1], k2=kgrid[2])
            text = text.replace("starting_magnetization(1)=0.8",
                                f"starting_magnetization(1)={m1}")
            path.write_text(text)
            pair = Calculator.from_file(path, pseudo_dir=PSEUDO)
            FIELD["b"] = b
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ref = pair.get_scf(conv_thr=1e-11, nbnd=snb, max_iterations=300)
            finally:
                FIELD["b"] = None
            rho = np.asarray(ref.density)
            r = dict(tag=tag, B=b, nbnd=snb, converged=bool(ref.converged),
                     minimised=float(ref.total_energy) / 2, grid=list(rho.shape[1:]))
            records["supercell"].append(r)
            print(json.dumps(r), flush=True)
            save()
            jax.clear_caches()
    aligned = min(r["minimised"] for r in records["supercell"]
                  if r["B"] == b and r["tag"] == "aligned")
    ferro = min(r["minimised"] for r in records["supercell"]
                if r["B"] == b and r["tag"] == "ferro")
    starts = [("tiled", 16, dict())] + [("field", n, dict(seed_from_field=True))
                                        for n in LADDER]
    for tag, nbnd, options in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_ultracell(calc.system, calc.pseudos, scf, shape, kgrid, nbnd=nbnd,
                              magnetic_field=field, conv_thr=1e-10, states_conv_thr=1e-9,
                              max_iterations=400, david=None if nbnd < 40 else 2,
                              **options)
        total = float(r.total_energy) + float(r.field_energy)
        rr = dict(tag=tag, B=b, nbnd=nbnd, converged=bool(r.converged),
                  iterations=int(r.iterations), seed_origin=r.seed_origin,
                  minimised=total, above_aligned=total - aligned,
                  above_ferro=total - ferro, box=list(np.asarray(r.density).shape[1:]))
        records["ultracell"].append(rr)
        print(json.dumps(rr), flush=True)
        save()
        jax.clear_caches()
save()
print("# done", records["seconds"])
