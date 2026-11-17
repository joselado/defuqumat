"""Gate for the Elk writer: is the PAW all-electron density closer to Elk's
converged density than Elk's own initial state?"""
import sys, json
from pathlib import Path
import numpy as np
from scipy.special import spherical_jn
from scipy.interpolate import CubicSpline
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from defuqumat import Calculator
from defuqumat.io.elk import ElkState
from defuqumat.io.elk_density import real_spherical_harmonics as rlm_elk
from defuqumat.pseudo.harmonics import real_spherical_harmonics as ylm_qe

ELK = Path(sys.argv[1])        # directory holding conv/ and init/
ecut = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
conv = ElkState.read(ELK / "conv"); init = ElkState.read(ELK / "init")
lmmaxo = conv.lmmaxo; lmaxo = int(round(np.sqrt(lmmaxo))) - 1

text = f"""&control
 calculation='scf'
/
&system
 ibrav=0, nat=2, ntyp=1, ecutwfc={ecut}, ecutrho={8*ecut}
/
&electrons
 conv_thr=1d-11
/
CELL_PARAMETERS bohr
 0.0 5.10 5.10
 5.10 0.0 5.10
 5.10 5.10 0.0
ATOMIC_SPECIES
 Si 28.086 Si.pz-n-kjpaw_psl.0.1.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 4 4 4 0 0 0
"""
calc = Calculator.from_text(text, pseudo_dir=REPO / "tests/data/pseudo", announce=False)
res = calc.get_scf()
cal = calc.calculation
cell = cal.system.cell
at = np.asarray(cell.at); omega = abs(np.linalg.det(at))
assert np.allclose(at, conv.geometry.avec, atol=1e-8), (at, conv.geometry.avec)
rho = np.asarray(res.density)[0]
grid = np.array(rho.shape)
coef = np.fft.fftn(rho) / rho.size                      # rho(r) = sum c_G e^{iGr}
idx = np.indices(grid).reshape(3, -1).T
mill = np.where(idx > grid // 2, idx - grid, idx)
b = 2 * np.pi * np.linalg.inv(at).T                      # rows b_i, 1/bohr
G = mill @ b; c = coef.reshape(-1)
keep = np.abs(c) > 1e-14; G, c, mill = G[keep], c[keep], mill[keep]
Gn = np.linalg.norm(G, axis=1)
print("defuqumat", res.iterations, res.total_energy, "ngm used", len(c), "charge", (coef[0,0,0]*omega).real)

pseudo = cal.pseudos[0]; paw = pseudo.paw
r_paw = np.asarray(pseudo.r)
ae = np.asarray(paw.ae_wfc); ps = np.asarray(paw.ps_wfc)
iraug = min(paw.cutoff_index or len(r_paw), len(r_paw))
qfuncl = np.asarray(pseudo.augmentation.qfuncl)
pspec = cal.paw.species[0]
C = np.asarray(pspec.coefficients); chan = np.asarray(pspec.channel_of); nlm = C.shape[0]
lmax_rho = int(round(np.sqrt(nlm))) - 1

# QE real harmonics -> Elk real harmonics, per l, by quadrature.
xg, wg = np.polynomial.legendre.leggauss(24)
ph = np.arange(48) * 2 * np.pi / 48
TH, PH = np.meshgrid(np.arccos(xg), ph, indexing="ij")
dirs = np.stack([np.sin(TH)*np.cos(PH), np.sin(TH)*np.sin(PH), np.cos(TH)], -1).reshape(-1, 3)
wq = (np.repeat(wg[:, None], 48, 1) * 2 * np.pi / 48).reshape(-1)
Yq = np.asarray(ylm_qe(dirs, lmax_rho))                 # (nq, nlm) QE
Re = rlm_elk(max(lmaxo, lmax_rho), dirs)                # (nq, lmmax) Elk
T = np.einsum("q,qa,qb->ab", wq, Re[:, :nlm], Yq)        # Elk_lm <- QE_lm'
print("QE->Elk transform orthogonal:", np.abs(T @ T.T - np.eye(nlm)).max())

def onecentre(becsum, truncated):
    # r^2 rho_lm for AE and PS (with compensation), on the PAW mesh
    pf = np.einsum("nr,mr->nmr", ae, ae); pt = np.einsum("nr,mr->nmr", ps, ps)
    if truncated:
        pf[:, :, iraug:] = 0; pt[:, :, iraug:] = 0
    pairs = np.einsum("ij,lij,ib,jc->bcl", becsum, C, chan, chan)
    d = np.einsum("bcl,bcr->lr", pairs, pf - pt)
    for lm in range(nlm):
        l = int(np.sqrt(lm))
        if l < qfuncl.shape[2]:
            d[lm] -= np.einsum("bc,bcr->r", pairs[:, :, lm], qfuncl[:, :, l, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        d = np.where(r_paw > 0, d / r_paw**2, 0.0)
    return T @ d                                         # Elk basis, (nlm, mesh)

def interp(r_from, f, r_to):
    out = np.zeros(f.shape[:-1] + (len(r_to),))
    last = np.max(np.nonzero(np.any(np.abs(f) > 0, axis=tuple(range(f.ndim - 1))))[0]) + 1
    rr = r_from[:last]; inside = r_to <= rr[-1]
    sp = CubicSpline(rr, f[..., :last], axis=-1)
    x = np.clip(r_to[inside], rr[0], None)
    out[..., inside] = sp(x)
    return out

becsum = np.asarray(res.becsum[0])[0]                    # (nat, nh, nh)
geo = conv.geometry
tau = geo.positions @ geo.avec                           # cartesian, bohr
rsp = conv.rsp[0]; nrmt = conv.nrmt[0]; rmt = conv.rmt[0]
core = np.asarray(paw.ae_rho_core)

def mt_density(ia, variant):
    """Elk f_lm(r) (lmmaxo, nrmt) of our all-electron density at atom ia."""
    phase = np.exp(1j * G @ tau[ia]) * c
    f = np.zeros((lmmaxo, nrmt))
    RG = rlm_elk(lmaxo, np.where(Gn[:, None] > 0, G, [0, 0, 1.0]))
    shells, inv = np.unique(np.round(Gn, 10), return_inverse=True)
    for l in range(lmaxo + 1):
        jl = spherical_jn(l, np.outer(shells, rsp))      # (nshell, nr)
        for m in range(-l, l + 1):
            lm = l * (l + 1) + m
            amp = np.zeros(len(shells), complex)
            np.add.at(amp, inv, phase * RG[:, lm])
            f[lm] = (4 * np.pi * (1j ** l) * amp @ jl).real
    if variant in ("trunc", "full"):
        d = onecentre(becsum[ia], truncated=(variant == "trunc"))
        f[:nlm] += interp(r_paw, d, rsp)[: min(nlm, lmmaxo)]
    if variant in ("trunc", "full", "core"):
        f[0] += interp(r_paw, core[None], rsp)[0] * np.sqrt(4 * np.pi)
    return f

def l1_mt(fa, fb, rmax_lo=0.0, rmax_hi=None):
    """int |rho_a - rho_b| over the shell rmax_lo < r < rmax_hi, by quadrature."""
    Rq = rlm_elk(lmaxo, dirs)                            # (nq, lmmaxo)
    da = Rq @ fa; db = Rq @ fb                           # (nq, nr)
    w_r = np.gradient(rsp) * rsp**2
    sel = (rsp >= rmax_lo) & (rsp < (rmax_hi or rmt + 1))
    return float(np.einsum("q,qr,r->", wq, np.abs(da - db)[:, sel], w_r[sel]))

report = {}
for name, (fa_fn) in [("elk-init", None), ("pseudo", "pseudo"), ("ae-trunc", "trunc"), ("ae-full", "full")]:
    rows = []
    for ia in range(2):
        ref = conv.rhomt[:, :nrmt, ia]
        f = init.rhomt[:, :nrmt, ia] if fa_fn is None else mt_density(ia, fa_fn)
        q = float(np.sqrt(4*np.pi) * np.trapezoid(f[0] * rsp**2, rsp))
        rows.append(dict(charge=q, charge_ref=float(np.sqrt(4*np.pi)*np.trapezoid(ref[0]*rsp**2, rsp)),
                         L1_core=l1_mt(f, ref, 0, 0.6), L1_valence=l1_mt(f, ref, 0.6, rmt + 1),
                         l3_xyz=[float(f[l*(l+1)-2, np.searchsorted(rsp, x)]) for l, x in [(3, 1.0), (3, 1.8)]],
                         l3_ref=[float(ref[l*(l+1)-2, np.searchsorted(rsp, x)]) for l, x in [(3, 1.0), (3, 1.8)]],
                         l4=[float(f[20, np.searchsorted(rsp, 1.5)]), float(ref[20, np.searchsorted(rsp, 1.5)])]))
    report[name] = rows
    print(name, json.dumps(rows[0]))

# Interstitial: points of Elk's grid outside both spheres
ng = np.array(conv.ngridg)
fr = np.indices(ng).reshape(3, -1).T / ng
pts = fr @ geo.avec
out = np.ones(len(pts), bool)
for ia in range(2):
    d = fr - geo.positions[ia]; d -= np.round(d); out &= np.linalg.norm(d @ geo.avec, axis=1) > rmt
ours_ir = (np.exp(1j * pts @ G.T) @ c).real
ref_ir = conv.rhoir.reshape(-1, order="F"); ini_ir = init.rhoir.reshape(-1, order="F")
print("interstitial points outside spheres:", out.sum(), "of", len(pts))
for name, v in [("elk-init", ini_ir), ("ours", ours_ir)]:
    dv = (v - ref_ir)[out]
    print(f"interstitial {name}: max|d| {np.abs(dv).max():.3e}  rms {np.sqrt(np.mean(dv**2)):.3e}  L1*vol {np.abs(dv).mean()*omega*out.mean():.3e}  (ref peak {np.abs(ref_ir[out]).max():.3f})")
