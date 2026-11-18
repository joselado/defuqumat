"""P180: does pw.x use the ekin-density defuqumat writes? A swap test on TB09.

pw.x's `tb09` is not defuqumat's TB09 (PLAN.md P30: libxc is called with a zero
Laplacian and c = 1), so the two codes share no fixed point and an iteration
count from defuqumat's whole state cannot test the kinetic energy density.
Instead pw.x restarts from its *own* converged save directory with the
ekin-density file replaced: by its own coefficients written back through
defuqumat's `write_rhog`, by defuqumat's TB09 tau, and by defuqumat's tau in
the wrong layout (the charge's (total, magnetization) at nspin = 2, twice the
value at nspin = 1). The first scf accuracy says whether pw.x reads the file,
and in which layout.

usage: python3 tb09_swap.py <work dir of pwx_reads_ours.py> <pw.x>
"""
import json, re, shutil, subprocess, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np
from defuqumat import Calculator
from defuqumat.io import qe_density as qd

rt = Path(sys.argv[1]).resolve()
pwx = sys.argv[2]
own = rt / "fresh_out" / "pwscf.save"
restart_input = (rt / "restart" / "pw.in").read_text()   # startingpot='file'
calc = Calculator.from_text((rt / "fresh" / "pw.in").read_text(),
                            pseudo_dir=REPO / "tests/data/pseudo", announce=False)
res = calc.get_scf()
ours = rt / "swap" / "ours.save"
if ours.exists():
    shutil.rmtree(ours)
qd.write_qe_save(ours, res, calc.calculation)
gamma, b, miller, tau = qd.read_rhog(ours / "ekin-density.dat")
gamma_own, b_own, miller_own, tau_own = qd.read_rhog(own / "ekin-density.dat")
assert gamma == gamma_own and np.array_equal(miller, miller_own)
variants = {"own": tau_own, "ours": tau}
if tau.shape[0] == 2:
    variants["ours_as_total_m"] = np.stack([tau[0] + tau[1], tau[0] - tau[1]])
else:
    variants["ours_x2"] = 2.0 * tau
report = dict(input=rt.name, components=int(tau.shape[0]),
              tau_diff=float(np.abs(tau - tau_own).max()),
              tau_peak=float(np.abs(tau_own).max()),
              tau_g0=[float(t[0].real) for t in tau], tau_g0_own=[float(t[0].real) for t in tau_own])
for name, t in variants.items():
    d = rt / "swap" / name
    if d.exists():
        shutil.rmtree(d)
    save = d / "out" / "pwscf.save"
    shutil.copytree(own, save)
    qd.write_rhog(save / "ekin-density.dat", b_own, miller_own, t, gamma_own)
    (d / "pw.in").write_text(re.sub(r"outdir\s*=\s*'[^']*'", f"outdir='{d / 'out'}'",
                                    restart_input))
    out = subprocess.run([pwx, "-in", "pw.in"], cwd=d, capture_output=True, text=True).stdout
    (d / "pw.out").write_text(out)
    acc = [float(x) for x in re.findall(r"estimated scf accuracy\s+<\s+([0-9.Ee+-]+)", out)]
    it = re.search(r"convergence has been achieved in\s+(\d+) iterations", out)
    report[name] = dict(first_accuracy=acc[0] if acc else None,
                        iterations=int(it.group(1)) if it else None,
                        read_kinetic="Reading meta-gga kinetic term" in out,
                        read_density="The initial density is read from file" in out)
print(json.dumps(report, indent=1))
