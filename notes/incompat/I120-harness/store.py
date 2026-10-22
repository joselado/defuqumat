"""I120: which wavefunction store a default run on this platform gets.

    python3 store.py

Prints the resolved store and whether a converged SCF's wavefunctions are a
host (streamed) store, for a two-k-point cell at the default memory mode, and
then whether the Sternheimer Jacobian refuses that state.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jvp import ROOT, edited  # noqa: E402

from defuqumat import batching  # noqa: E402
from defuqumat.io.pwin import parse_pw_input  # noqa: E402
from defuqumat.pseudo import read_upf  # noqa: E402
from defuqumat.scf.driver import Calculation, run_scf  # noqa: E402
from defuqumat.scf.residual import make_residual  # noqa: E402
from defuqumat.scf.streaming import is_host_store  # noqa: E402
from defuqumat.system import build_system  # noqa: E402

print("platform default store:", batching.resolve_wfc_store("default", "default"))
print("memory preset, k_batch 1 of 8:", batching.resolve_scf_wfc_store(None, "memory", 1, 8))
text = edited("si2-tb09", "2 2 2 1 1 1", 15)
system = build_system(parse_pw_input(text))
pseudos = tuple(read_upf(ROOT / "tests/data/pseudo" / s.pseudo_file)
                for s in system.structure.species)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

calc = Calculation(system, pseudos)
r = run_scf(system, pseudos, calculation=calc, conv_thr=1e-11, max_iterations=100,
            wfc_store="stream")
print("stream: k_batch", calc.k_batch, "host store:", is_host_store(r.wavefunctions))
residual = make_residual(calc, 10, float(r.ethr))
x = np.asarray(residual.pack(r.density, tuple(r.becsum or ()), None, r.tau))
v = np.random.default_rng(7).normal(size=x.shape) * x
v /= np.linalg.norm(v)
host = residual.sternheimer_jacobian(x, r.wavefunctions)(v)
device = residual.sternheimer_jacobian(x, jnp.asarray(r.wavefunctions))(v)
print("RESULT streamed against resident action: |dJv|/|Jv + v| =",
      float(np.linalg.norm(host - device) / np.linalg.norm(device + v)))
