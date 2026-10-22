"""I116, second round: which state the collinear field seed reaches, against the
explicit two-atom supercell under the same field.

Unit cell ``HYDROGEN`` (one atom, a = 5.5, nspin = 2, moment 0.62 on (4, 2, 2)),
ultracell (2, 1, 1) at kgrid (2, 2, 2), field B cos(pi x) (unit-cell crystal x),
conv_thr 1e-10. Four ultracell starts: tiled, the field's seed sign(B) sign(M)
(handed over through ``seed.collinear_field_seed``, since the driver refuses
seed_from_field=True at nspin_mag = 2), the wrong-sign domain -sign(B), and the
smooth cos(pi x) seed, the one ``test_a_collinear_seed_reaches_the_state_the_tiled_loop_cannot``
converges to the zero-field antiferromagnet with. nbnd 16 for all, 24 and 40
for the field's seed.

The supercell is the same two atoms as two species (``ANTIFERROMAGNET``),
started aligned with the field (+0.8, -0.8), against it (-0.8, +0.8) and
ferromagnetic (+0.8, +0.8), and the field reaches it through the local
potential: ``scf.driver.as_potential_components`` is wrapped so that at
nspin = 2 the up channel gets -B and the down +B on the dense grid
(``add_bfield.f90:237-238``'s sign, the ultracell's ``_as_field``). Since
``deband`` removes only ``v_scf``, the reported total then contains -int B m,
so it is compared with the ultracell's ``total_energy + field_energy``.

**Energies are per unit cell**: ``UltracellResult.total_energy`` already is
(the first round's ``i116.py`` halved it again, so its energies and gaps are
half the true ones), and the supercell's total is divided by two.

Usage: i116b.py <out.json> [amplitudes] [nbnd list for the field seed]
"""
import json, sys, time, warnings
from pathlib import Path
import tempfile

import numpy as np
import jax
import jax.numpy as jnp

from defuqumat import Calculator
import defuqumat.scf.driver as scf_driver
from tests.regression.test_ultracell import _hydrogen, _fourier, ANTIFERROMAGNET
from defuqumat.ultracell import run_ultracell
from defuqumat.ultracell.seed import collinear_field_seed

out = sys.argv[1]
amplitudes = [float(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "0.002,0.02").split(",")]
LADDER = [int(v) for v in (sys.argv[3] if len(sys.argv) > 3 else "16,24,40").split(",")]
A = 5.5
LENGTHS = np.array([2 * A, A, A])
SITES = [np.array([0.0, 0.0, 0.0]), np.array([A, 0.0, 0.0])]
PSEUDO = str(Path("tests/data/pseudo").resolve())


def sphere(m, radius=1.5):
    grid = m.shape
    axes = [np.arange(g) / g * L for g, L in zip(grid, LENGTHS)]
    r = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    dv = np.prod(LENGTHS) / np.prod(grid)
    res = []
    for site in SITES:
        d = r - site
        d -= np.round(d / LENGTHS) * LENGTHS
        res.append(float(m[(d ** 2).sum(axis=-1) < radius ** 2].sum() * dv))
    return res


FIELD = {"b": None}
_original = scf_driver.as_potential_components


def with_field(scalar, nspin):
    base = _original(scalar, nspin)
    b = FIELD["b"]
    if b is None or nspin != 2 or jnp.ndim(scalar) != 3:
        return base
    nx = scalar.shape[0]
    x = 2.0 * np.arange(nx) / nx
    field = jnp.asarray(b * np.cos(np.pi * x)[:, None, None]
                        * np.ones(scalar.shape), dtype=base.dtype)
    return base + jnp.stack([-field, field])


scf_driver.as_potential_components = with_field

tmp = Path(tempfile.mkdtemp())
t0 = time.time()
shape, kgrid = (2, 1, 1), (2, 2, 2)
folded = tuple(n * m for n, m in zip(shape, kgrid))
calc = _hydrogen(tmp, PSEUDO, folded)
scf = calc.get_scf(conv_thr=1e-12, nbnd=8)
volume = float(calc.system.cell.volume)
sign_m = float(np.sign(scf.magnetization))
records = dict(reference=dict(magnetization=float(scf.magnetization),
                              total_energy=float(scf.total_energy)),
               ultracell=[], supercell=[])
print(json.dumps(records["reference"]), flush=True)


def save():
    records["seconds"] = time.time() - t0
    with open(out, "w") as handle:
        json.dump(records, handle, indent=1)


for b in amplitudes:
    field = lambda x, b=b: b * np.cos(np.pi * x[..., 0])

    def from_field(x, f=field):
        values = f(x)
        seed, reason = collinear_field_seed(values, np.full(values.shape, sign_m), 1.0)
        assert reason is None, reason
        return seed

    starts = [("tiled", 16, dict(seed_from_field=False)),
              ("anti", 16, dict(seed_magnetization=lambda x: -np.sign(np.cos(np.pi * x[..., 0])))),
              ("cos", 16, dict(seed_magnetization=lambda x: np.cos(np.pi * x[..., 0])))]
    starts += [("field", nbnd, dict(seed_magnetization=from_field)) for nbnd in LADDER]
    for tag, nbnd, options in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = run_ultracell(calc.system, calc.pseudos, scf, shape, kgrid, nbnd=nbnd,
                              magnetic_field=field, conv_thr=1e-10, states_conv_thr=1e-9,
                              max_iterations=400, david=None if nbnd < 40 else 2, **options)
        rho = np.asarray(r.density)
        m = rho[0] - rho[1]
        rec = dict(tag=tag, B=b, nbnd=nbnd, converged=bool(r.converged),
                   iterations=int(r.iterations), energy=float(r.total_energy),
                   field_energy=float(r.field_energy),
                   minimised=float(r.total_energy) + float(r.field_energy),
                   sites=sphere(m), cell_moments=[float(v) for v in np.asarray(r.cell_moments())],
                   m_Q=float(np.abs(_fourier(m, tuple(r.ultracell.grid), np.array([[1, 0, 0]])))[0]) * volume)
        records["ultracell"].append(rec)
        print(json.dumps(rec), flush=True)
        save()
        jax.clear_caches()

    for tag, (m1, m2) in (("aligned", (0.8, -0.8)), ("against", (-0.8, 0.8)),
                          ("ferro", (0.8, 0.8))):
        path = tmp / f"pair_{tag}_{b:g}.in"
        text = ANTIFERROMAGNET.format(m2=m2, k0=kgrid[0], k1=kgrid[1], k2=kgrid[2])
        text = text.replace("starting_magnetization(1)=0.8", f"starting_magnetization(1)={m1}")
        path.write_text(text)
        pair = Calculator.from_file(path, pseudo_dir=PSEUDO)
        FIELD["b"] = b
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ref = pair.get_scf(conv_thr=1e-11, nbnd=16, max_iterations=300)
        finally:
            FIELD["b"] = None
        rho = np.asarray(ref.density)
        m = rho[0] - rho[1]
        nx = m.shape[0]
        bfield = b * np.cos(np.pi * 2.0 * np.arange(nx) / nx)[:, None, None]
        field_energy = -float((bfield * m).sum()) * float(pair.system.cell.volume) / m.size
        rec = dict(tag=tag, B=b, converged=bool(ref.converged), iterations=int(ref.iterations),
                   minimised=float(ref.total_energy) / 2,
                   field_energy=field_energy / 2,
                   energy=(float(ref.total_energy) - field_energy) / 2,
                   sites=sphere(m), grid=list(m.shape))
        records["supercell"].append(rec)
        print(json.dumps(rec), flush=True)
        save()
        jax.clear_caches()

save()
print("# done", records["seconds"])
