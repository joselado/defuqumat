"""I108 and I109 on a converged cell: nosource under 'grad_m' and under TB09.

The four-site hydrogen chain of ``tests/data/qe/h4-noncolin-force.in`` (each
moment 90 degrees from the one before), with the functional, ``nosource`` and
``noncollinear_gga`` changed, and ``conv_thr`` loosened to 1e-8. For each run:
whether it converged and in how many steps, the moments, ``int |div B_xc|^2`` on
the dense sphere at the converged density with the projection on and off, the
residue on the whole box, and (PBE only; the torque refuses a meta-GGA) the
exchange-correlation torque. Run on D22 through slot.sh from the repository
root; prints one block per case.
"""

import sys
import time
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from defuqumat import Calculator
from defuqumat.basis.gradients import divergence
from defuqumat.scf.potential import v_of_rho

ROOT = Path.cwd()
CASES = ROOT / "tests" / "data" / "qe"
PSEUDO = ROOT / "tests" / "data" / "pseudo"
BASE = (CASES / "h4-noncolin-force.in").read_text()


def text(functional, nosource, choice, ecutrho=None):
    t = BASE.replace("    tprnfor = .true.\n", "").replace("    tstress = .true.\n", "")
    t = t.replace("conv_thr = 1.0d-11", "conv_thr = 1.0d-8\n    electron_maxstep = 300")
    extra = f"\n    input_dft = '{functional}'"
    if nosource:
        extra += "\n    nosource = .true."
    if choice is not None:
        extra += f"\n    noncollinear_gga = '{choice}'"
    if ecutrho is not None:
        extra += f"\n    ecutrho = {ecutrho}"
    return t.replace("noncolin = .true.", "noncolin = .true." + extra)


def fields(calc, scf, source_free):
    c = calc.calculation
    density = jnp.asarray(scf.density)
    tau = None if scf.tau is None else jnp.asarray(scf.tau)
    p = v_of_rho(density, c.basis.dense, calc.system.cell, c.rho_core,
                 c.functional, c.rho_core_g, c.quantization_axis, tau,
                 source_free, c.noncollinear_gga)
    return np.asarray(jnp.real(p.v_scf[1:]))


def sphere(field, calc):
    c = calc.calculation
    d = divergence(jnp.asarray(field), c.basis.dense, calc.system.cell)
    return float(calc.system.cell.volume / d.size * jnp.sum(d ** 2))


def box(field, calc):
    cell = calc.system.cell
    n = field.shape[1:]
    f = np.fft.fftn(field, axes=(1, 2, 3))
    miller = np.stack(np.meshgrid(*[np.fft.fftfreq(k) * k for k in n], indexing="ij"), -1)
    g = miller @ np.asarray(cell.bg)
    return float(np.sum(np.abs(np.einsum("...i,i...->...", g, f)) ** 2))


CASELIST = {
    "pbe-gradm-free": ("PBE", True, "grad_m", None),
    "pbe-kubler-free": ("PBE", True, "kubler", None),
    "pbe-gradm-plain": ("PBE", False, "grad_m", None),
    "pbe-gradm-free-200": ("PBE", True, "grad_m", 200.0),
    "tb09-kubler-free": ("TB09", True, None, None),
    "tb09-gradm-free": ("TB09", True, "grad_m", None),
    "tb09-kubler-plain": ("TB09", False, "kubler", None),
}

names = sys.argv[1:] or list(CASELIST)
for name in names:
    functional, nosource, choice, ecutrho = CASELIST[name]
    start = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc = Calculator.from_text(text(functional, nosource, choice, ecutrho),
                                    PSEUDO, announce=False)
        scf = calc.get_scf()
    c = calc.calculation
    print(f"== {name}: functional {c.functional.name} source_free {c.source_free} "
          f"noncollinear_gga {c.noncollinear_gga} grid {c.basis.dense.grid}", flush=True)
    print(f"   converged {scf.converged} iterations {scf.iterations} "
          f"E {scf.total_energy:.10f} Ry  {time.time() - start:.0f} s")
    print(f"   magnetization_vector {np.round(np.asarray(scf.magnetization_vector), 6).tolist()}")
    moments = np.asarray(scf.site_moments)
    for i, m in enumerate(moments):
        print(f"   site {i} moment {np.round(m, 5).tolist()} |m| {np.linalg.norm(m):.5f}")
    on, off = fields(calc, scf, True), fields(calc, scf, False)
    print(f"   sphere int|div B|^2: projected {sphere(on, calc):.3e} unprojected {sphere(off, calc):.3e}")
    print(f"   box residue projected/unprojected {box(on, calc) / box(off, calc):.3e}")
    if not c.functional.is_meta:
        from defuqumat.scf.spin_torque import torque_of_result

        t = torque_of_result(c, scf)
        print(f"   torque sites {np.round(np.asarray(t.sites), 8).tolist()} "
              f"parallel_fraction {float(t.parallel_fraction):.6f} "
              f"total {np.linalg.norm(np.asarray(t.total)):.2e}")
    if nosource:
        for what, call in (("forces", calc.get_forces),):
            try:
                call()
                print(f"   {what}: NOT REFUSED")
            except NotImplementedError as error:
                print(f"   {what}: refused ({str(error)[:60]}...)")
    sys.stdout.flush()
