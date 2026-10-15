"""Helpers shared by every docs/figures/<id>/compute.py.

    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import common

    params = common.start(dict(input="si.in", grid=12))   # prints them
    ...                                                    # compute
    common.save(__file__, params, energy=e, dos=d)          # data.npz

save() writes data.npz next to compute.py, with the parameters as one JSON
string under the key "params", completed with the git commit, the host,
the defuqumat path and the wall time. Read it back with common.load(path).
"""
import json
import os
import socket
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
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


def start(params):
    """Print the parameters before the calculation starts; return them."""
    global _T0
    _T0 = time.time()
    print("parameters:")
    for k, v in params.items():
        print("  %s = %r" % (k, v))
    print(flush=True)
    return params


def save(script, params, **arrays):
    """Write data.npz next to `script`, with params stored as JSON."""
    import defuqumat
    p = dict(params)
    p["git_commit"] = git_commit()
    p["host"] = socket.gethostname()
    p["defuqumat_path"] = os.path.dirname(defuqumat.__file__)
    p["wall_time_s"] = round(time.time() - _T0, 1)
    path = os.path.join(os.path.dirname(os.path.abspath(script)), "data.npz")
    np.savez(path, params=json.dumps(p), **arrays)
    print("wrote", path, "wall time %.1f s" % p["wall_time_s"])
    return path


def load(path):
    """Return (params dict, npz arrays) of a data.npz."""
    d = np.load(path)
    return json.loads(str(d["params"])), d
