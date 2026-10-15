"""One SCF of the robust-mixing test bed.

python3 -u run_one.py <cell> <mode> <out.json> [generated_dir] [key=value ...]

Writes the outcome, the iteration and diagonalization counts, the energy and
moments, and the per-iteration accuracy and energy, so that a run is judged
by the state it reached and not only by its count.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cells import CELLS, MAX_ITERATIONS  # noqa: E402

from defuqumat import Calculator  # noqa: E402
import defuqumat.scf.driver as driver  # noqa: E402

# The control arm: the robust mode with its inner mixer pinned (``anderson``),
# to separate what the supervisor does from the preconditioner it would choose.
driver.ROBUST_INNER = os.environ.get("ROBUST_INNER") or None

cell, mode, out = sys.argv[1:4]
generated = Path(sys.argv[4]) if len(sys.argv) > 4 and "=" not in sys.argv[4] else ROOT / "tools/robust_mixing/inputs"
extra = {}
for item in sys.argv[4:]:
    if "=" in item:
        key, value = item.split("=", 1)
        try:
            extra[key] = float(value) if "." in value or "e" in value else int(value)
        except ValueError:
            extra[key] = value
path = CELLS[cell][0]
path = ROOT / path if "/" in path else generated / f"{path}.in"

calc = Calculator.from_file(path, pseudo_dir=ROOT / "tests/data/pseudo", announce=False)
options = {"mixing_mode": mode, "max_iterations": MAX_ITERATIONS, **extra}
record = {"cell": cell, "input": str(path), "mode": mode, "options": options}
start = time.perf_counter()
try:
    result = calc.get_scf(**options)
    history = result.history
    record.update(
        converged=bool(result.converged), iterations=int(result.iterations),
        diagonalizations=int(sum(h.get("trials", 1) for h in history)),
        energy=float(result.total_energy),
        accuracies=[float(h["accuracy"]) for h in history],
        energies=[float(h["total_energy"]) for h in history],
        merits=[h.get("merit") for h in history],
        steps=[h.get("supervisor") for h in history if h.get("supervisor")],
    )
    if result.magnetization is not None:
        record["magnetization"] = [float(x) for x in
                                   (result.magnetization if hasattr(result.magnetization, "__len__")
                                    else [result.magnetization])]
        record["absolute_magnetization"] = float(result.absolute_magnetization)
except Exception as error:  # a diverged run is a result, not a crash of the bed
    record.update(converged=False, error=f"{type(error).__name__}: {error}")
record["seconds"] = time.perf_counter() - start


def clean(x):
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if isinstance(x, list):
        return [clean(v) for v in x]
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    return x


Path(out).write_text(json.dumps(clean(record)))
print(cell, mode, record.get("iterations"), record.get("diagonalizations"), record.get("converged"),
      record.get("energy"), record.get("absolute_magnetization"), f"{record['seconds']:.1f}s",
      record.get("error", ""), flush=True)
