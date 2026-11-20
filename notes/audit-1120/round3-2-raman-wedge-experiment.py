"""#2 experiment: the Raman tensor of the PAW wedge with a full-zone becsum tangent in D."""
import sys, time
from pathlib import Path
import jax, jax.numpy as jnp, numpy as np
import defuqumat
print("package:", defuqumat.__file__, flush=True)
import importlib
nl = importlib.import_module("defuqumat.response.nonlinear")
es = importlib.import_module("defuqumat.response.electrostriction")
from defuqumat.response.phonon import symmetrize_becsum_modes
from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.system import build_system
from defuqumat.scf import Calculation, run_scf

orig = nl.susceptibility_displacement_derivative


def wrapped(calculation, solver, rho, b, u, positions, dpsi, drho,
            geometry_tangent=True, verbose=False, ort=None, stored=None):
    psi, w = solver.psi, solver.weights
    positions = jnp.asarray(positions)
    nat = positions.shape[0]
    per_mode = np.empty((nat, 3), dtype=object)
    for a in range(nat):
        for c in range(3):
            t = jnp.zeros_like(positions).at[a, c].set(1.0)
            mode_psi = es._project_conduction(psi, jnp.asarray(dpsi[a, c])[None],
                                              solver.hamiltonians, calculation.k_batch)[0]
            if ort is not None:
                mode_psi = mode_psi + jnp.asarray(ort[a, c])
            per_mode[a, c] = jax.jvp(
                lambda x, s: es.mixed_becsum(calculation.at_positions(x), s, w),
                (positions, psi), (t, mode_psi))[1]
    full = symmetrize_becsum_modes(calculation, per_mode)
    sample = full[0, 0]
    stacks = tuple(None if sample[sp] is None else jnp.stack([
        jnp.stack([full[a, c][sp] for c in range(3)]) for a in range(nat)])
        for sp in range(len(sample)))
    if calculation._becsum_symmetry is not None:
        es._INJECT = (positions, stacks)
    try:
        return orig(calculation, solver, rho, b, u, positions, dpsi, drho,
                    geometry_tangent=geometry_tangent, verbose=verbose, ort=ort,
                    stored=stored)
    finally:
        es._INJECT = None


nl.susceptibility_displacement_derivative = wrapped
ROOT = Path(defuqumat.__file__).parents[1]
CASES, PSEUDO = ROOT / "tests/data/qe", ROOT / "tests/data/pseudo"
case = sys.argv[1]
system = build_system(read_pw_input(CASES / f"{case}.in"))
pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
calc = Calculation(system, pseudos, memory_mode="speed")
res = run_scf(system, pseudos, calculation=calc, conv_thr=1e-12, max_iterations=100,
              mixing_fit="flat")
t = time.time()
out = nl.raman_tensors(calc, res, tr2=1e-24, threshold=1e-12)
tensor = np.asarray(out.raman)
print(case, f"raman {time.time()-t:.1f} s; d(eps_yz)/d(tau_0x) =", tensor[0, 0, 1, 2],
      "residue", out.translational_residue, flush=True)
np.savez(f"exp-{case}-speed-raman.npz", tensor=tensor)
