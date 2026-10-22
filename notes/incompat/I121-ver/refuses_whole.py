"""The whole body of test_what_a_spiral_refuses, called as a function (the file is slow)."""
import os, sys, time
sys.path.insert(0, os.getcwd())
from pathlib import Path
t0 = time.time()
from tests.regression.test_spin_spirals import test_what_a_spiral_refuses
suite = Path("/u/40/ladovj1/data/Documents/programs/claude/defuqumat/quantum_espresso/qe-7.5-ReleasePack/qe-7.5/test-suite")
assert suite.is_dir()
test_what_a_spiral_refuses(Path("tests/data/pseudo"), suite)
print(f"test_what_a_spiral_refuses body passed, {time.time() - t0:.0f} s")
