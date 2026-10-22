"""The two slow tests' bodies, called directly (no pytest, so no slow run).

    PYTHONPATH=. OMP_NUM_THREADS=1 taskset -c 4-7 \
        python3 notes/incompat/I118-harness/run_test_bodies.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.regression import test_stm as t  # noqa: E402

for name, call in [
    ("weak field, collinear",
     lambda: t.test_the_image_in_a_weak_field_tends_to_the_field_free_one("collinear")),
    ("weak field, noncollinear",
     lambda: t.test_the_image_in_a_weak_field_tends_to_the_field_free_one("noncollinear")),
    ("the run's own states", t.test_the_states_in_a_field_are_the_run_s_own),
]:
    start = time.time()
    call()
    print(f"{name}: passed in {time.time() - start:.1f} s", flush=True)
