import sys, json
from pathlib import Path
sys.argv += []
exec(open("pwx_reads_ours.py").read().split("def main():")[0])
for name in sys.argv[1:]:
    src = Path("inputs") / f"{name}.in"
    work = Path("runs/fix") / name
    r = run_pwx(set_control(src.read_text(), (work / "out").resolve(), "pwscf"), work)
    print(name, json.dumps({k: r[k] for k in ("iterations", "first_accuracy", "energy", "wall")}), flush=True)
