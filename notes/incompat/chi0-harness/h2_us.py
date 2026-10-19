"""I94 harness: RPA eps_M(0) of ultrasoft silicon against the Sternheimer solve.

P94's comparison (si-us-nosym.in, run_absorption's static residual, RPA against
screening = 'hartree'), now with the ultrasoft head as the generalised velocity
(P99) and the body's Q_ij(G) from augmentation_at_q, swept over nbnd and over
ecut_response, beside the norm-conserving control and two partial controls.

    python h2_us.py <part>     part in: us, nc, controls
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
import jax
import numpy as np

import defuqumat.tddft.chi0 as chi0_module
from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response.velocity import VelocityOperator
from defuqumat.scf import run_scf
from defuqumat.system import build_system
from defuqumat.workflows.tddft import run_absorption

CASES = Path("tests/data/qe")
PSEUDO = Path("tests/data/pseudo")
t0 = time.time()


def stamp(msg):
    print(f"[{time.time() - t0:8.1f} s] {msg}", flush=True)


def converged(text):
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    result = run_scf(system, pseudos, conv_thr=1e-12, max_iterations=200)
    return system, pseudos, result


def residual(system, pseudos, result, nbnd, ecut):
    spectrum = run_absorption(system, pseudos, result.density, np.array([0.1]),
                              kernel="rpa", nbnd=nbnd, broadening=0.01,
                              ecut_response=ecut, static_residual=True,
                              conv_thr=1e-11)
    jax.clear_caches()
    return spectrum.static_rpa, spectrum.static_residual


US = (CASES / "si-us-nosym.in").read_text()
NC = US.replace("Si.pz-n-rrkjus_psl.0.1.UPF", "Si.pz-vbc.UPF").replace(", ecutrho = 96.0", "")
part = sys.argv[1]

if part == "us":
    system, pseudos, result = converged(US)
    stamp(f"US SCF: {result.total_energy:.10f} Ry, npwx {result.calculation.basis.planewaves.npwx if hasattr(result, 'calculation') else '?'}")
    for ecut in (8.0, 16.0, 24.0):
        for nbnd in (30, 60, 80):
            if ecut == 24.0 and nbnd == 80:
                continue
            here, res = residual(system, pseudos, result, nbnd, ecut)
            stamp(f"US ecut_response={ecut} nbnd={nbnd}: eps_M(0) {here:.4f} residual {res:+.4f}")

if part == "us2":
    system, pseudos, result = converged(US)
    for ecut, nbnd in ((8.0, 45), (8.0, 70), (16.0, 30), (16.0, 60)):
        try:
            here, res = residual(system, pseudos, result, nbnd, ecut)
            stamp(f"US ecut_response={ecut} nbnd={nbnd}: eps_M(0) {here:.4f} residual {res:+.4f}")
        except Exception as exc:  # the referee at a large nbnd, seen at 80
            stamp(f"US ecut_response={ecut} nbnd={nbnd}: {type(exc).__name__} {str(exc)[:120]}")
            jax.clear_caches()

if part == "nc":
    if "ecutwfc = 12.0, ecutrho" in NC:
        raise SystemExit("NC text still has ecutrho")
    system, pseudos, result = converged(NC)
    for ecut in (8.0, 16.0):
        for nbnd in (30, 60, 90):
            here, res = residual(system, pseudos, result, nbnd, ecut)
            stamp(f"NC ecut_response={ecut} nbnd={nbnd}: eps_M(0) {here:.4f} residual {res:+.4f}")

if part == "controls":
    system, pseudos, result = converged(US)
    # (a) no augmentation in the body, generalised head
    original = chi0_module.augmentation_factors
    chi0_module.augmentation_factors = lambda *a, **k: None
    try:
        for nbnd in (30, 60):
            here, res = residual(system, pseudos, result, nbnd, 8.0)
            stamp(f"control body-without-Q nbnd={nbnd}: eps_M(0) {here:.4f} residual {res:+.4f}")
    finally:
        chi0_module.augmentation_factors = original
    # (b) the plain dH/dk head (P94's), with the body's Q
    tangent, connections = VelocityOperator.tangent_elements, VelocityOperator.connections
    VelocityOperator.tangent_elements = (
        lambda self, psi, energies, sequential=False: self.matrix_elements(psi, sequential=sequential))
    VelocityOperator.connections = lambda self, psi, sequential=False: None
    try:
        for nbnd in (30, 60):
            here, res = residual(system, pseudos, result, nbnd, 8.0)
            stamp(f"control plain-dH/dk-head nbnd={nbnd}: eps_M(0) {here:.4f} residual {res:+.4f}")
    finally:
        VelocityOperator.tangent_elements, VelocityOperator.connections = tangent, connections
stamp("done")
