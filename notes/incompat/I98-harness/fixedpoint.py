"""I98: the computation of the slow test_the_augmented_update_is_a_fixed_point_and_conserves_its_energy, printed.

    python3 fixedpoint.py <worktree> <tmpdir>
"""
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1])
tmp = Path(sys.argv[2])
tmp.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "tests" / "regression"))

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import test_realtime_augmented_update as t  # noqa: E402
from defuqumat.io.pwin import read_pw_input  # noqa: E402
from defuqumat.pseudo import read_upf  # noqa: E402
from defuqumat.realtime.propagate import propagate  # noqa: E402
from defuqumat.realtime.pulse import Kick  # noqa: E402
from defuqumat.scf import Calculation  # noqa: E402
from defuqumat.system import build_system  # noqa: E402
from defuqumat.system.kpoints import KPoints  # noqa: E402

pseudo_dir = root / "tests/data/pseudo"
for case, extra in [("si-epsilon-us", ""), ("si-epsilon-paw", "input_dft = 'sla+pw',")]:
    path = t._input(tmp, case, 8.0, 8.0, extra)
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    system = eqx.tree_at(lambda s: s.kpoints, system, KPoints(coords=t.KSET[0], weights=t.KSET[1]))
    calc = Calculation(system, pseudos)
    v_scf = calc.potential(calc.starting_density()).v_scf
    ddd = None
    psi, w = t._dense_states(calc, v_scf, None)
    if calc.is_paw:
        _, ddd = calc.onecenter(calc.becsum(jnp.asarray(psi)[None], jnp.asarray(w)[None]))
        psi, w = t._dense_states(calc, v_scf, ddd)

    def run(strength):
        return propagate(calc, jnp.asarray(psi), w, v_scf,
                         Kick(strength=strength, direction=(1.0, 0.0, 0.0)), dt=0.05,
                         start=0.0, duration=2.0, potential="hxc", block_steps=20,
                         k_batch=None, ddd_paw=ddd)

    still = run(0.0)
    print(f"# {case} no field: |v - v_scf| {still.extras['potential_change']:.3e}, "
          f"|dJ| {np.abs(still.current - still.current[0]).max():.3e}", flush=True)
    kicked = run(0.01)
    print(f"# {case} kick 0.01: dv {kicked.extras['potential_change']:.3e}, energy ptp "
          f"{np.ptp(kicked.energy):.3e}, norm {kicked.norm_drift:.3e}", flush=True)
    print("    energy:", " ".join(f"{e - kicked.energy[0]:+.2e}" for e in kicked.energy), flush=True)
    held = calc.becsum(jnp.asarray(psi)[None], jnp.asarray(w)[None])
    original = Calculation.augmented
    Calculation.augmented = lambda self, rho, becsum_: original(self, rho, held)
    try:
        off = run(0.01)
    finally:
        Calculation.augmented = original
    print(f"# {case} kick 0.01, augmentation held at the start's becsum: dv "
          f"{off.extras['potential_change']:.3e}, energy ptp {np.ptp(off.energy):.3e}", flush=True)
    jax.clear_caches()
