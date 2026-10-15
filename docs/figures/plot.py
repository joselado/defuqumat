"""Run docs/figures/<id>/plot.py inside its folder, with docs/figures on the path.

    python3 docs/figures/plot.py <id> [<id> ...] [--theme light|dark]
    python3 docs/figures/plot.py all

Each plot.py reads data.npz from its own folder and writes
docs/guide/img/<id>.png and <id>-dark.png. Nothing here imports defuqumat.
"""
import argparse
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import style  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("ids", nargs="+")
p.add_argument("--theme", choices=["light", "dark"], default=None)
args = p.parse_args()
if args.theme:
    style.THEMES[:] = [args.theme]
ids = args.ids
if ids == ["all"]:
    ids = sorted(d for d in os.listdir(HERE)
                 if os.path.isfile(os.path.join(HERE, d, "plot.py"))
                 and os.path.isfile(os.path.join(HERE, d, "data.npz")))
for figid in ids:
    folder = os.path.join(HERE, figid)
    os.chdir(folder)
    runpy.run_path(os.path.join(folder, "plot.py"), run_name="__main__")
