"""Tabulate a directory of run_one.py JSON files: python3 aggregate.py <dir>.

One row a (cell, beta), one column pair a mode: the diagonalisation count, or
'x' when it did not converge, and the energy the run reached, so that two
modes landing on different states are visible rather than counted alike.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

rows = defaultdict(dict)
for path in sorted(Path(sys.argv[1]).glob("*.json")):
    r = json.loads(path.read_text())
    beta = r.get("options", {}).get("mixing_beta")
    rows[(r["cell"], beta)][r["mode"]] = r
modes = sorted({m for v in rows.values() for m in v})
print(f"{'cell':18s} {'beta':>5s} " + " ".join(f"{m:>26s}" for m in modes))
for (cell, beta), by_mode in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1] or 0)):
    cols = []
    for m in modes:
        r = by_mode.get(m)
        if r is None:
            cols.append(f"{'':>26s}")
            continue
        count = r.get("diagonalizations", r.get("iterations"))
        mark = f"{count}" if r.get("converged") else f"x{count if count else ''}"
        energy = r.get("energy")
        mag = r.get("absolute_magnetization")
        text = f"{mark:>4s} {energy:.8f}" if isinstance(energy, float) else f"{mark:>4s} {r.get('error', '')[:16]}"
        if isinstance(mag, float):
            text += f" |m|{mag:.2f}"
        cols.append(f"{text:>26s}")
    print(f"{cell:18s} {beta!s:>5s} " + " ".join(cols))
