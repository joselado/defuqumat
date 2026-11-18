"""The vacuum of graphane under the local mBJ: the potential along z, and the gap against the size.

usage: python3 vacuum_check.py <dft> <height bohr> <conv_thr> [gate]

Saves z, the planar averages of the density, of the total local potential and
of c(r), the SCF eigenvalues and the band-route eigenvalues at the SCF's own
k-points, to vac-<dft>-<height>[-gate<gate>].npz. ``gate`` overrides the local
mBJ's potential gate (mgga.LMBJ_RHO_THRESHOLD) for the measurement only.
"""
import sys, json, warnings
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
warnings.filterwarnings("ignore")
import numpy as np
dft, height, conv = sys.argv[1], float(sys.argv[2]), sys.argv[3]
gate = sys.argv[4] if len(sys.argv) > 4 else None
if gate is not None:
    import defuqumat.xc.functional as F
    F.LMBJ_RHO_THRESHOLD = float(gate)
from defuqumat import Calculator
from defuqumat.units import RY_TO_EV
ANG = 1.0 / 0.529177210903
a = 2.54 * ANG
zc, zh = 0.23 * ANG, 1.33 * ANG
dft_line = "" if dft == "pbe" else f", input_dft='{dft}'"
text = f"""&control
 calculation='scf'
/
&system
 ibrav=4, celldm(1)={a:.6f}, celldm(3)={height / a:.6f}, nat=4, ntyp=2,
 ecutwfc=60.0, nbnd=8{dft_line}
/
&electrons
 conv_thr={conv}, mixing_beta=0.4, electron_maxstep=400
/
ATOMIC_SPECIES
 C 12.011 C.pbe-hgh.UPF
 H 1.008 H.pbe-hgh.UPF
ATOMIC_POSITIONS bohr
 C 0.0 {a / np.sqrt(3):.6f} {zc:.6f}
 C {a / 2:.6f} {a / (2 * np.sqrt(3)):.6f} {-zc:.6f}
 H 0.0 {a / np.sqrt(3):.6f} {zh:.6f}
 H {a / 2:.6f} {a / (2 * np.sqrt(3)):.6f} {-zh:.6f}
K_POINTS automatic
 6 6 1 0 0 0
"""
calc = Calculator.from_text(text, pseudo_dir=REPO / "tests/data/pseudo", announce=False)
res = calc.get_scf()
bands = calc.get_bands(kpoints=calc.system.kpoints)
rho = np.asarray(res.density)[0]
v = np.asarray(res.potential)[0]
n3 = rho.shape[2]
z = np.arange(n3) * height / n3
field = res.meta_c_field
tag = f"vac-{dft}-{int(height)}" + ("" if gate is None else f"-gate{gate}")
np.savez(Path(sys.argv[0]).resolve().parent / f"{tag}.npz" if False else f"{tag}.npz",
         z=z, rho=rho.mean(axis=(0, 1)), rho_min=rho.min(axis=(0, 1)), v=v.mean(axis=(0, 1)),
         c=None if field is None else np.asarray(field).mean(axis=(0, 1)),
         eig=np.asarray(res.eigenvalues) * RY_TO_EV, bands=np.asarray(bands.eigenvalues) * RY_TO_EV)
eb = np.asarray(bands.eigenvalues) * RY_TO_EV
print(json.dumps(dict(tag=tag, iterations=res.iterations, converged=bool(res.converged),
                      gap=float(eb[:, 5].min() - eb[:, 4].max()),
                      rho_min_planar=float(rho.mean(axis=(0, 1)).min()))), flush=True)
