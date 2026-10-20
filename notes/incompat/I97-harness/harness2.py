"""I97 check v2 (ecut_response sweep, Sternheimer tr2): static limit of the adiabatic PBE kernel against PBE Sternheimer eps_inf."""
import sys, time, warnings, json
sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as np
from pathlib import Path
from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation, run_scf
from defuqumat.system import build_system
from defuqumat.tddft import alda_matrix, independent_response, solve_dyson
from defuqumat.workflows.nscf import fixed_density_states
from defuqumat.response.efield import dielectric_tensor
from defuqumat.scf.potential import exchange_correlation

T0 = time.time()


def log(*a):
    print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


CASES = Path("tests/data/qe")
PSEUDO = Path("tests/data/pseudo")
FUNC = sys.argv[1] if len(sys.argv) > 1 else "pbe"
parsed = read_pw_input(CASES / "si-epsilon-unshifted-nosym.in")
parsed.namelists["system"]["input_dft"] = FUNC
warnings.filterwarnings("ignore", message="input_dft asks for")
system = build_system(parsed)
pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
calc0 = Calculation(system, pseudos)
log("functional", calc0.functional.name)
scf = run_scf(system, pseudos, calculation=calc0, conv_thr=1e-12, max_iterations=80)
density = jnp.asarray(scf.density)
log("scf done")


def isotropic(m):
    return float(np.real(np.diag(np.asarray(m))).mean())


out = {}
calc_st, _, ev, wf = fixed_density_states(system, pseudos, scf.density, nbnd=8, conv_thr=1e-10)
nocc = int(round(calc_st.nelec / 2))
for scr, tr2 in (("full", None), ("hartree", None), ("full", 1e-16), ("full", 1e-18)):
    key = f"stern_{scr}" + ("" if tr2 is None else f"_{tr2:g}")
    try:
        r = dielectric_tensor(calc_st, wf[:, :, :nocc], ev[:, :, :nocc], density,
                              screening=scr, born_charges=False, tr2=tr2)
        out[key] = isotropic(r.epsilon)
        log("sternheimer", key, out[key])
    except Exception as exc:
        log("FAILED", key, repr(exc))


def local_matrix(calculation, sphere):
    """The truncation the old refusal named: the local part of the functional only."""
    _, kr = jax.jvp(lambda r: exchange_correlation(r, calculation.system.cell,
                    calculation.rho_core, calculation.functional)[0],
                    (density,), (jnp.ones_like(density),))
    grid = np.asarray(calculation.basis.dense.grid)
    kg = np.fft.fftn(np.asarray(kr[0])) / float(np.prod(grid))
    body = np.asarray(sphere.miller)
    idx = (body[:, None, :] - body[None, :, :]) % grid
    c = np.asarray(sphere.sqrt_coulomb)
    return jnp.asarray(kg[idx[..., 0], idx[..., 1], idx[..., 2]] / np.outer(c, c))


POINTS = [(60, 8.0), (60, 12.0), (60, 16.0), (60, 20.0), (90, 8.0), (90, 12.0), (90, 16.0), (90, 20.0)]
states = {}
for nbnd, ecut in POINTS:
    try:
        if nbnd not in states:
            states[nbnd] = fixed_density_states(system, pseudos, scf.density, nbnd=nbnd, conv_thr=1e-10)
            log("states", nbnd)
        calculation, _, eigenvalues, wavefunctions = states[nbnd]
        potential = calculation.potential(density)
        chi = independent_response(calculation, wavefunctions, eigenvalues, potential.v_scf,
                                   np.array([0.0]), ecut_response=ecut, broadening=0.0)
        t = time.time()
        gga = alda_matrix(calculation, density, chi.sphere)
        tg = time.time() - t
        row = {"nbody": int(chi.sphere.nbody), "t_kernel": tg}
        row["alda"] = isotropic(solve_dyson(chi, "alda", {"alda_matrix": gga}).epsilon[0])
        row["local"] = isotropic(solve_dyson(
            chi, "alda", {"alda_matrix": local_matrix(calculation, chi.sphere)}).epsilon[0])
        row["rpa"] = isotropic(solve_dyson(chi, "rpa", {}).epsilon[0])
        out[f"{nbnd}_{ecut}"] = row
        log(nbnd, ecut, row,
            "d_full", row["alda"] - out["stern_full"],
            "d_local", row["local"] - out["stern_full"],
            "d_rpa_vs_hartree", row["rpa"] - out["stern_hartree"])
    except Exception as exc:
        log('FAILED', nbnd, ecut, repr(exc))
    jax.clear_caches()
print("RESULT", json.dumps(out), flush=True)
