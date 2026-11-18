"""defuqumat writes a save directory, pw.x restarts from it with startingpot='file'.

usage: python3 pwx_reads_ours.py <pw.in> <workdir> [--control updown]
"""
import os, re, sys, shutil, subprocess, time, json
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import numpy as np

import os as _os
PWX = _os.environ.get("PW_X", str(REPO / "quantum_espresso/qe-7.5-ReleasePack/qe-7.5/bin/pw.x"))
PSEUDO = REPO / "tests/data/pseudo"

def set_control(text, outdir, prefix, extra_electrons=""):
    text = re.sub(r"pseudo_dir\s*=\s*'[^']*'", f"pseudo_dir='{PSEUDO}'", text)
    text = re.sub(r"outdir\s*=\s*'[^']*'", f"outdir='{outdir}'", text)
    if "prefix" in text:
        text = re.sub(r"prefix\s*=\s*'[^']*'", f"prefix='{prefix}'", text)
    else:
        text = re.sub(r"&control", f"&control\n    prefix='{prefix}',", text, flags=re.I)
    if "pseudo_dir" not in text:
        text = re.sub(r"&control", f"&control\n    pseudo_dir='{PSEUDO}',", text, flags=re.I)
    if "outdir" not in text:
        text = re.sub(r"&control", f"&control\n    outdir='{outdir}',", text, flags=re.I)
    if extra_electrons:
        text = re.sub(r"&electrons", "&electrons\n    " + extra_electrons, text, flags=re.I)
    return text

def run_pwx(text, cwd):
    cwd.mkdir(parents=True, exist_ok=True)
    (cwd / "pw.in").write_text(text)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    t = time.perf_counter()
    pin = ["taskset", "-c", "0"] if os.environ.get("TASKSET", "1") == "1" else []
    out = subprocess.run(pin + [str(PWX), "-in", "pw.in"], cwd=cwd,
                         capture_output=True, text=True, env=env).stdout
    (cwd / "pw.out").write_text(out)
    acc = [float(x) for x in re.findall(r"estimated scf accuracy\s+<\s+([0-9.Ee+-]+)", out)]
    it = re.search(r"convergence has been achieved in\s+(\d+) iterations", out)
    e = re.search(r"!\s+total energy\s+=\s+([-0-9.]+)", out)
    return dict(iterations=int(it.group(1)) if it else None,
                first_accuracy=acc[0] if acc else None,
                energy=float(e.group(1)) if e else None,
                renormalised=("renormalised" in out),
                read_from_file=("initial density is read from file" in out),
                infomsg=[l.strip() for l in out.splitlines() if "Message from routine" in l or "zero-padding" in l or "Conversion" in l],
                wall=time.perf_counter() - t)

def main():
    src = Path(sys.argv[1]); work = Path(sys.argv[2]).resolve()
    control = "--control" in sys.argv
    if work.exists(): shutil.rmtree(work)
    text = src.read_text()
    fresh = run_pwx(set_control(text, work / "fresh_out", "pwscf"), work / "fresh")

    from defuqumat import Calculator
    from defuqumat.io import qe_density as qd
    ours_text = src.read_text()
    for pair in [a.split("=", 1)[1] for a in sys.argv if a.startswith("--ours=")]:
        old, new = pair.split("->")
        ours_text = ours_text.replace(old, new)
    calc = Calculator.from_text(ours_text, pseudo_dir=PSEUDO, announce=False)
    res = calc.get_scf()
    save = work / "ours_out" / "pwscf.save"
    if control:
        orig = qd._to_qe_layout
        qd._to_qe_layout = lambda rho: rho   # writes (up, down): the wrong layout
    qd.write_qe_save(save, res, calc.calculation, gamma_only=(False if "--full" in sys.argv else None))
    if control:
        qd._to_qe_layout = orig
    # coefficients against pw.x's own file
    g1, b1, m1, r1 = qd.read_rhog(work / "fresh_out" / "pwscf.save" / "charge-density.dat")
    g2, b2, m2, r2 = qd.read_rhog(save / "charge-density.dat")
    same_order = np.array_equal(m1, m2)
    coef = float(np.abs(r1 - r2).max()) if r1.shape == r2.shape else None
    extra = "startingpot='file'"
    restart = run_pwx(set_control(text, work / "ours_out", "pwscf", extra), work / "restart")
    report = dict(input=src.name, control=control, ngm=(len(m1), len(m2)), same_order=same_order,
                  coefficient_diff=coef, coefficient_peak=float(np.abs(r1).max()),
                  defuqumat=dict(iterations=res.iterations, energy=float(res.total_energy)),
                  pwx_fresh=fresh, pwx_restart_from_ours=restart)
    print(json.dumps(report, indent=1))

main()
