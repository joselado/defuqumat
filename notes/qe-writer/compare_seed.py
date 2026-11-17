"""pw.x save -> defuqumat seed: each loaded array against defuqumat's own converged one."""
import sys, re, json, numpy as np
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from defuqumat import Calculator
from defuqumat.io.qe_density import QESave
name = sys.argv[1]
save = QESave.read(Path(f"runs/fix/{name}/out/pwscf.save"))
pwout = Path(f"runs/fix/{name}/pw.out").read_text()
e_qe = float(re.search(r"!\s+total energy\s+=\s+([-0-9.]+)", pwout).group(1))
calc = Calculator.from_text(Path(f"inputs/{name}.in").read_text(), pseudo_dir=REPO / "tests/data/pseudo", announce=False)
own = calc.get_scf()
seed = calc.get_qe_seed(save.directory)
rep = dict(name=name, own_iterations=own.iterations, own_energy=float(own.total_energy), pwx_energy=e_qe)
rho = np.asarray(seed["starting_density"]); ref = np.asarray(own.density)
rep["density"] = (float(np.abs(rho - ref).max()), float(np.abs(ref).max()))
if "starting_becsum" in seed:
    out = []
    for a, b in zip(seed["starting_becsum"], own.becsum):
        if a is None: continue
        a, b = np.asarray(a), np.asarray(b)
        off = ~np.eye(a.shape[-1], dtype=bool)
        out.append(dict(shape=a.shape, diff=float(np.abs(a - b).max()), peak=float(np.abs(b).max()),
                        peak_offdiag=float(np.abs(b[..., off]).max()),
                        per_component=[float(np.abs(a[c] - b[c]).max()) for c in range(a.shape[0])],
                        mag_peak=[float(np.abs(b[c]).max()) for c in range(a.shape[0])]))
    rep["becsum"] = out
if "starting_ns" in seed:
    a, b = np.asarray(seed["starting_ns"]), np.asarray(own.ns)
    rep["ns"] = dict(shape=a.shape, diff=float(np.abs(a - b).max()), peak=float(np.abs(b).max()),
                     imag_peak=float(np.abs(np.imag(b)).max()),
                     diff_if_conjugated=float(np.abs(np.conj(a) - b).max()),
                     diff_if_transposed=float(np.abs(np.swapaxes(a, -1, -2) - b).max()))
seeded = calc.get_scf(**seed)
rep["seeded"] = dict(iterations=seeded.iterations, energy=float(seeded.total_energy),
                     first_accuracy=(seeded.history[0].get("accuracy") if seeded.history and isinstance(seeded.history[0], dict) else None))
print(json.dumps(rep, indent=1, default=str))
