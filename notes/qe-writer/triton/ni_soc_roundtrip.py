"""Ni with spin-orbit coupling and a moment, PAW: the pw.x save directory both ways (P180).

usage: python3 ni_soc_roundtrip.py <ecutwfc> <ecutrho> <workdir> <pw.x>
pw.x from scratch, defuqumat seeded from its save directory (get_qe_seed), the
loaded becsum against the seeded run's, then defuqumat's state written
(write_qe_save) and pw.x restarted from it with startingpot = 'file'.
"""
import json, os, re, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import numpy as np

ecutwfc, ecutrho, work, pwx = float(sys.argv[1]), float(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
PSEUDO = REPO / "tests/data/pseudo"
TEXT = """ &control
    calculation='scf', pseudo_dir='{pseudo}', outdir='{outdir}', prefix='pwscf'
 /
 &system
    ibrav=2, celldm(1)=6.65, nat=1, ntyp=1, ecutwfc={ecutwfc}, ecutrho={ecutrho},
    occupations='smearing', smearing='mv', degauss=0.02,
    noncolin=.true., lspinorb=.true., starting_magnetization(1)=0.5
 /
 &electrons
    mixing_beta=0.3, conv_thr=1.0d-10, electron_maxstep=200 {extra}
 /
ATOMIC_SPECIES
 Ni 58.6934 Ni.rel-pbe-spn-kjpaw_psl.1.0.0.UPF
ATOMIC_POSITIONS crystal
 Ni 0.0 0.0 0.0
K_POINTS automatic
 4 4 4 0 0 0
"""


def pw(outdir, extra, tag):
    text = TEXT.format(pseudo=PSEUDO, outdir=outdir, ecutwfc=ecutwfc, ecutrho=ecutrho, extra=extra)
    d = work / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "pw.in").write_text(text)
    t = time.perf_counter()
    out = subprocess.run([pwx, "-in", "pw.in"], cwd=d, capture_output=True, text=True).stdout
    (d / "pw.out").write_text(out)
    acc = re.findall(r"estimated scf accuracy\s+<\s+([0-9.Ee+-]+)", out)
    it = re.search(r"convergence has been achieved in\s+(\d+) iterations", out)
    e = re.search(r"!\s+total energy\s+=\s+([-0-9.]+)", out)
    m = re.findall(r"total magnetization\s+=\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)", out)
    return dict(iterations=int(it.group(1)) if it else None, first_accuracy=float(acc[0]) if acc else None,
                energy=float(e.group(1)) if e else None, magnetization=m[-1] if m else None,
                renormalised="renormalised" in out, read_from_file="read from file" in out,
                wall=round(time.perf_counter() - t, 1)), text


report = dict(ecutwfc=ecutwfc, ecutrho=ecutrho)
report["pwx_fresh"], text = pw(work / "fresh_out", "", "fresh")
from defuqumat import Calculator
from defuqumat.io.qe_density import QESave
calc = Calculator.from_text(text, pseudo_dir=PSEUDO, announce=False)
save = QESave.read(work / "fresh_out" / "pwscf.save")
report["file"] = dict(rho_components=int(save.nspin_mag), paw_numbers=int(save.becsum.size))
t = time.perf_counter()
seed = calc.get_qe_seed(save.directory)
seeded = calc.get_scf(**seed)
loaded = np.asarray(next(b for b in seed["starting_becsum"] if b is not None))
own = np.asarray(next(b for b in seeded.becsum if b is not None))
report["defuqumat_seeded"] = dict(iterations=seeded.iterations, energy=float(seeded.total_energy),
                                  magnetization=seeded.magnetization_vector,
                                  becsum_shape=list(loaded.shape),
                                  becsum_diff=[float(np.abs(loaded[c] - own[c]).max()) for c in range(loaded.shape[0])],
                                  becsum_peak=[float(np.abs(own[c]).max()) for c in range(own.shape[0])],
                                  wall=round(time.perf_counter() - t, 1))
calc.write_qe_save(work / "ours_out" / "pwscf.save", seeded)
report["pwx_restart_from_ours"], _ = pw(work / "ours_out", ", startingpot='file'", "restart")
print(json.dumps(report, indent=1, default=str))
