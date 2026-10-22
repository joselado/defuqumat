"""The tail of test_what_a_spiral_refuses, run as a script (the test file is slow)."""
import os, sys
sys.path.insert(0, os.getcwd())
from pathlib import Path
import jax, jax.numpy as jnp
from defuqumat.io.pwin import read_pw_input
from defuqumat.system import build_system
from defuqumat.scf.driver import Calculation
from defuqumat.pseudo.augmentation import TabulatedAugmentation
from tests.regression.test_spin_spirals import _pseudos

suite = Path("/u/40/ladovj1/data/Documents/programs/claude/defuqumat/quantum_espresso/qe-7.5-ReleasePack/qe-7.5/test-suite")
pseudo_dir = Path("tests/data/pseudo")
ultrasoft = read_pw_input(suite / "pw_noncolin" / "noncolin.in")
marker = ultrasoft.namelists["system"]
marker["nosym"] = ".true."
marker["spiral_q"] = {(3,): 0.5}
system = build_system(ultrasoft)
os.environ["DEFUQUMAT_AUG_MAX_BYTES"] = "1"
tabulated = Calculation(system, _pseudos(system, pseudo_dir))
assert isinstance(tabulated.cross_augmentation, TabulatedAugmentation)
moved = jax.jit(lambda q: tabulated.at_spiral_q(
    q, rebuild_basis=False).cross_augmentation.gcart)(jnp.asarray([0.0, 0.0, 0.4]))
assert bool(jnp.all(jnp.isfinite(moved)))
assert isinstance(tabulated.at_spiral_q((0.0, 0.0, 0.4), rebuild_basis=False).cross_augmentation,
                  TabulatedAugmentation)
print("ok", moved.shape)
