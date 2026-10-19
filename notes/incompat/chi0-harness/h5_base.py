"""The fixed-occupation spectrum before and after: run on incompat-base3 and on the branch."""
import sys

sys.path.insert(0, ".")
from pathlib import Path

import numpy as np

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import run_scf
from defuqumat.system import build_system
from defuqumat.workflows.tddft import run_absorption

np.set_printoptions(precision=10, linewidth=200)
text = Path("tests/data/qe/si-epsilon-unshifted-nosym.in").read_text()
system = build_system(parse_pw_input(text))
pseudos = tuple(read_upf(Path("tests/data/pseudo") / s.pseudo_file) for s in system.structure.species)
result = run_scf(system, pseudos, conv_thr=1e-12, max_iterations=200)
w = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
spectrum = run_absorption(system, pseudos, result.density, w, kernel="rpa", nbnd=16,
                          broadening=0.01, ecut_response=2.0, static_residual=False,
                          conv_thr=1e-11)
print("eps_M:", np.trace(np.asarray(spectrum.epsilon), axis1=1, axis2=2) / 3.0)
spectrum = run_absorption(system, pseudos, result.density, w, kernel="bootstrap", nbnd=16,
                          broadening=0.01, ecut_response=2.0, static_residual=True,
                          conv_thr=1e-11, scissor=0.02)
print("bootstrap+scissor eps_M:", np.trace(np.asarray(spectrum.epsilon), axis1=1, axis2=2) / 3.0)
print("static residual:", spectrum.static_residual)
