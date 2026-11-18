"""Graphane (CH, chair) at two vacuum sizes, global TB09 against the local mBJ."""
import sys, json, warnings
sys.path.insert(0, "/u/40/ladovj1/data/Documents/programs/claude/defuqumat")
warnings.filterwarnings("ignore")
import numpy as np
from defuqumat import Calculator
from defuqumat.units import RY_TO_EV
ANG = 1.0 / 0.529177210903
dft, height = sys.argv[1], float(sys.argv[2])     # height in bohr
a = 2.54 * ANG
zc, zh = 0.23 * ANG, (0.23 + 1.10) * ANG
text = f"""&control
 calculation='scf'
/
&system
 ibrav=4, celldm(1)={a:.6f}, celldm(3)={height / a:.6f}, nat=4, ntyp=2,
 ecutwfc=60.0, nbnd=8{'' if dft == 'pbe' else f", input_dft='{dft}'"}
/
&electrons
 conv_thr=1.0d-9, mixing_beta=0.4
/
ATOMIC_SPECIES
 C 12.011 C.pbe-hgh.UPF
 H 1.008 H.pbe-hgh.UPF
ATOMIC_POSITIONS bohr
 C {0.0:.6f} {a / np.sqrt(3):.6f} {zc:.6f}
 C {a / 2:.6f} {a / (2 * np.sqrt(3)):.6f} {-zc:.6f}
 H {0.0:.6f} {a / np.sqrt(3):.6f} {zh:.6f}
 H {a / 2:.6f} {a / (2 * np.sqrt(3)):.6f} {-zh:.6f}
K_POINTS automatic
 6 6 1 0 0 0
"""
calc = Calculator.from_text(text, pseudo_dir="/u/40/ladovj1/data/Documents/programs/claude/defuqumat/tests/data/pseudo", announce=False)
res = calc.get_scf()
e = np.asarray(res.eigenvalues) * RY_TO_EV
rho = np.asarray(res.density)[0]
out = dict(dft=dft, height=height, iterations=res.iterations, converged=bool(res.converged),
           c=None if res.meta_c is None else float(res.meta_c), mesh_gap_eV=float(e[:, 5].min() - e[:, 4].max()),
           below_threshold=float((rho <= 1e-6).mean()))
if res.meta_c_field is not None:
    profile = np.asarray(res.meta_c_field).mean(axis=(0, 1))
    out.update(c_middle=float(profile[0]), c_vacuum=float(profile[len(profile) // 2]),
               c_max=float(profile.max()))
    np.save(f"cz-{dft}-{int(height)}.npy", profile)
np.save(f'eig-{dft}-{int(height)}-new.npy', e)
print(json.dumps(out), flush=True)
