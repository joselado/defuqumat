"""Helpers shared by every docs/figures/<id>/compute.py.

run.sh starts compute.py inside its own folder, with docs/figures on the
path, so a compute.py reads

    from defuqumat import Calculator
    from common import PSEUDO, save

    calc = Calculator.from_file("si.in", pseudo_dir=PSEUDO)
    ...
    save(energy=e, dos=d)

save() writes data.npz in the current folder. Next to the arrays it stores
one JSON string under the key "params": the git commit, the host, the
defuqumat path and the wall time. load(path) reads both back.
"""
import json
import os
import socket
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PSEUDO = os.path.normpath(os.path.join(HERE, "..", "..", "tests", "data", "pseudo"))
_T0 = time.time()


def git_commit():
    """The commit of this checkout, or the one written in docs/figures/COMMIT.

    A copy that is not a git repository (a run on another machine) takes
    the commit from docs/figures/COMMIT, written by whoever copied it.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                             capture_output=True, text=True, check=True)
        commit = out.stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "defuqumat"],
                               cwd=HERE, capture_output=True, text=True).stdout
        return commit + ("-dirty" if dirty.strip() else "")
    except Exception:
        try:
            return open(os.path.join(HERE, "COMMIT")).read().strip()
        except OSError:
            return "unknown"


def save(**arrays):
    """Write data.npz in the current folder, with the run's metadata."""
    import defuqumat
    meta = {"git_commit": git_commit(), "host": socket.gethostname(),
            "defuqumat_path": os.path.dirname(defuqumat.__file__),
            "wall_time_s": round(time.time() - _T0, 1)}
    path = os.path.abspath("data.npz")
    np.savez(path, params=json.dumps(meta), **arrays)
    print("wrote", path, "wall time %.1f s" % meta["wall_time_s"])
    return path


def load(path):
    """Return (metadata dict, npz arrays) of a data.npz."""
    d = np.load(path)
    return json.loads(str(d["params"])), d
