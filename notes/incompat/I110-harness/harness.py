"""I110: the meta-GGA exchange torque against m x B_xc read off the SCF's own field.

The cell is ``h-chain-90deg.in`` (four H moments 90 degrees apart, no field, no
constraint, nosym) under ``input_dft = 'tb09'``; ``noncollinear_gga`` is the
default, which for a noncollinear meta-GGA is ``'grad_m'``. For each
``conv_thr``:

* ``rebuilt``: ``torque_of_result(calc, scf)``, ``v_of_rho`` at the result's
  ``(density, tau)``, which are the last iteration's output.
* ``scf``: ``exchange_torque(scf.density, scf.potential)``. Run 2 showed that
  ``scf.potential`` is rebuilt at convergence on the *output* ``(rho, tau)``
  (``driver.py``, the ``vnew`` block), so this is the same field and the gap is
  round-off at every ``conv_thr``: an identity of the plumbing.
* ``input``: ``exchange_torque(scf.density, scf.potential -
  scf.potential_change)``, ``m_out x B_xc[rho_in, tau_in]``, the field the last
  iteration actually applied. This gap is the self-consistency residual and
  falls with ``conv_thr``.

The term switched off: the same rebuild with ``tau``'s vector part zeroed, and
with ``tau`` replaced by Thomas-Fermi of the charge.
Run with ``python3 harness.py`` from the repository root.
"""

import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from defuqumat import Calculator
from defuqumat.scf.spin_torque import exchange_torque, torque_of_result
from defuqumat.xc.mgga import thomas_fermi_tau

ROOT = Path.cwd()
CASE = ROOT / "tests" / "data" / "qe" / "h-chain-90deg.in"
PSEUDO = ROOT / "tests" / "data" / "pseudo"
ARGS = [a for a in sys.argv[1:] if not a.startswith("a=")]
SIDE = [float(a[2:]) for a in sys.argv[1:] if a.startswith("a=")]
THRESHOLDS = [float(x) for x in ARGS] or [1.0e-6, 1.0e-8, 1.0e-10]


def text():
    """The chain under TB09; ``a=<bohr>`` on the command line shrinks the box
    side across the chain, keeping the chain itself (20 bohr, four atoms)."""
    body = CASE.read_text().replace(
        "    nosym = .true.", "    nosym = .true.\n    input_dft = 'tb09'", 1)
    if SIDE:
        body = body.replace(
            "celldm(1) = 12.0, celldm(3) = 1.66666666666667",
            f"celldm(1) = {SIDE[0]}, celldm(3) = {20.0 / SIDE[0]:.14f}", 1)
    return body.replace("conv_thr = 1.0d-11", "conv_thr = 1.0d-6", 1)


def norm(x):
    return float(np.linalg.norm(np.asarray(x)))


for thr in THRESHOLDS:
    start = time.time()
    calc = Calculator.from_text(text(), pseudo_dir=PSEUDO, conv_thr=thr,
                                max_iterations=300)
    calculation = calc.calculation
    scf = calc.get_scf()
    cell = calculation.system.cell
    regions = calculation.local_regions()
    print(f"\n=== conv_thr = {thr:.0e}: converged {scf.converged} in "
          f"{scf.iterations} iterations, {time.time() - start:.0f} s, "
          f"noncollinear_gga = {calculation.noncollinear_gga}, "
          f"meta_c = {scf.meta_c}", flush=True)
    print("site moments", np.round(np.asarray(scf.site_moments), 4).tolist())
    print("tau shape", np.asarray(scf.tau).shape)

    rebuilt = torque_of_result(calculation, scf)
    own = exchange_torque(scf.density, scf.potential, cell, regions)

    from defuqumat.scf.potential import v_of_rho
    field_rebuilt = v_of_rho(
        jnp.asarray(scf.density), calculation.basis.dense, cell,
        calculation.rho_core, calculation.functional, calculation.rho_core_g,
        calculation.quantization_axis, jnp.asarray(scf.tau),
        calculation.source_free, calculation.noncollinear_gga,
        calculation._spiral_q_argument(),
    ).v_scf[1:]
    db = float(jnp.max(jnp.abs(field_rebuilt - jnp.asarray(scf.potential)[1:])))

    tau = jnp.asarray(scf.tau)
    no_vector = torque_of_result(calculation, scf, density=scf.density,
                                 tau=tau.at[1:].set(0.0))
    tf = jnp.zeros_like(tau).at[0].set(
        thomas_fermi_tau(jnp.real(jnp.asarray(scf.density)[0]), 1))
    thomas_fermi = torque_of_result(calculation, scf, density=scf.density, tau=tf)

    v_in = jnp.asarray(scf.potential) - jnp.asarray(scf.potential_change)
    applied = exchange_torque(scf.density, v_in, cell, regions)
    s_in = np.asarray(applied.sites)
    print(f"max |B_out - B_in| = "
          f"{float(jnp.max(jnp.abs(jnp.asarray(scf.potential_change)[1:]))):.3e} Ry")
    s_new = np.asarray(rebuilt.sites)
    s_own = np.asarray(own.sites)
    print(f"parallel_fraction rebuilt {float(rebuilt.parallel_fraction):.8f} "
          f"scf {float(own.parallel_fraction):.8f}")
    print(f"max |B_rebuilt - B_scf| = {db:.3e} Ry")
    print("sites rebuilt (Ry)", s_new.tolist())
    print("sites scf     (Ry)", s_own.tolist())
    print(f"|sites| max  = {np.abs(s_new).max():.6e}")
    print(f"sites gap    = {np.abs(s_new - s_own).max():.3e}  "
          f"(relative {np.abs(s_new - s_own).max() / np.abs(s_new).max():.3e})")
    print(f"total rebuilt {np.asarray(rebuilt.total).tolist()}  |.| = {norm(rebuilt.total):.3e}")
    print(f"total scf     {np.asarray(own.total).tolist()}  |.| = {norm(own.total):.3e}")
    print(f"input-field sites gap = {np.abs(s_new - s_in).max():.3e}  "
          f"(relative {np.abs(s_new - s_in).max() / np.abs(s_new).max():.3e}); "
          f"total gap {norm(np.asarray(rebuilt.total) - np.asarray(applied.total)):.3e}")
    print(f"tau vector off: sites gap to input = "
          f"{np.abs(np.asarray(no_vector.sites) - s_in).max():.3e}")
    print(f"tau vector off: sites gap to scf = "
          f"{np.abs(np.asarray(no_vector.sites) - s_own).max():.3e}")
    print(f"Thomas-Fermi tau: sites gap to scf = "
          f"{np.abs(np.asarray(thomas_fermi.sites) - s_own).max():.3e}", flush=True)
