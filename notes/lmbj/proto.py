"""NumPy prototype of the local mBJ c(r) on a converged density (full FFT box)."""
import sys, warnings, json
import numpy as np
from scipy.special import erf
sys.path.insert(0, "/u/40/ladovj1/data/Documents/programs/claude/defuqumat")
warnings.filterwarnings("ignore")
ALPHA, BETA, SIGMA, RS = 0.488, 0.5, 3.78, 5.0
RHO_TH = 3.0 / (4.0 * np.pi * RS**3)

def lmbj_c(rho, A, sigma=SIGMA, rho_th=RHO_TH):
    n = rho.shape
    B = 2 * np.pi * np.linalg.inv(A).T          # rows b_i, 1/bohr
    m = np.meshgrid(*[np.fft.fftfreq(k, 1.0 / k) for k in n], indexing="ij")
    G = sum(m[i][..., None] * B[i] for i in range(3))   # (n1,n2,n3,3)
    rg = np.fft.fftn(rho)
    grad = np.real(np.fft.ifftn(1j * G * rg[..., None], axes=(0, 1, 2)))
    mag = np.linalg.norm(grad, axis=-1)
    x = np.clip(rho, 0, None) / rho_th
    erfx_over_x = np.where(x > 1e-8, erf(x) / np.where(x > 1e-8, x, 1.0), 2 / np.sqrt(np.pi))
    g = (1 - ALPHA) / BETA * (1 - erf(x)) + mag / rho_th * erfx_over_x
    gbar = np.real(np.fft.ifftn(np.fft.fftn(g) * np.exp(-0.5 * sigma**2 * np.sum(G**2, -1))))
    return ALPHA + BETA * gbar, g, mag, grad

def run(text):
    from defuqumat import Calculator
    calc = Calculator.from_text(text, pseudo_dir="/u/40/ladovj1/data/Documents/programs/claude/defuqumat/tests/data/pseudo", announce=False)
    res = calc.get_scf()
    cell = calc.system.cell
    A = np.asarray(cell.at)
    return np.asarray(res.density)[0], A, float(cell.volume)

cases = {
 "Si (pz-vbc)": """&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=10.20, nat=2, ntyp=1, ecutwfc=30.0
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 4 4 4 0 0 0
""",
 "diamond (pbe-hgh)": """&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=6.74, nat=2, ntyp=1, ecutwfc=60.0
/
&electrons
/
ATOMIC_SPECIES
 C 12.011 C.pbe-hgh.UPF
ATOMIC_POSITIONS alat
 C 0.00 0.00 0.00
 C 0.25 0.25 0.25
K_POINTS automatic
 4 4 4 0 0 0
"""}
if __name__ == "__main__":
    for name, text in cases.items():
        rho, A, omega = run(text)
        c, g, mag, grad = lmbj_c(rho, A)
        active = rho > 1e-6
        gbar_tb09 = float((mag[active] / rho[active]).mean())
        print(json.dumps(dict(case=name, electrons=round(float(rho.mean() * omega), 6),
            rho_min=float(rho.min()), below_rho_th=float((rho < RHO_TH).mean()),
            tb09_c=round(-0.012 + 1.023 * np.sqrt(gbar_tb09), 4),
            linear_c_no_blend=round(ALPHA + BETA * gbar_tb09, 4),
            lmbj_c_mean=round(float(c.mean()), 4), lmbj_c_min=round(float(c.min()), 4), lmbj_c_max=round(float(c.max()), 4))))
