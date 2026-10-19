"""I90 fix round: the walked route against the whole route on the polar wedge.

python3 walked.py, from the worktree root. The same computation as
tests/regression/test_streamed_piezo.py::test_the_walked_tensor_is_the_whole_k_tensor
at ("alas-raman-polar-wedge", 3), which is slow and is listed for the user.
"""
import warnings

import jax.numpy as jnp
import numpy as np

from defuqumat.calculator import Calculator
from defuqumat.response.electrostriction import refined_states
from defuqumat.response.piezo import _piezoelectric_from_states
from defuqumat.units import E_BOHR2_TO_C_M2 as C

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    calculator = Calculator.from_file(
        "tests/data/qe/alas-raman-polar-wedge.in", pseudo_dir="tests/data/pseudo",
        k_batch=3, conv_thr=1e-12, announce=False, origin_tangent=False)
    calculation = calculator.calculation
    result = calculator.get_scf()
    eigenvalues, psi = refined_states(calculation, result)
print("nk", calculation.system.kpoints.nk, flush=True)


def tensor(states):
    e, field = _piezoelectric_from_states(
        calculation, states, eigenvalues, jnp.asarray(result.density),
        result.becsum, threshold=1.0e-12)
    return np.asarray(e), "field" in field.internals


whole, w_walked = tensor(psi)
walked, k_walked = tensor(np.asarray(psi))
print("routes walked?", w_walked, k_walked)
print("max |walked - whole| e/bohr^2", np.abs(walked - whole).max())
for name, idx in (("e31", (2, 0, 0)), ("e33", (2, 2, 2)), ("e15", (0, 0, 2)),
                  ("e14", (0, 1, 2))):
    print(name, "whole", whole[idx] * C, "walked", walked[idx] * C)
print("done", flush=True)
