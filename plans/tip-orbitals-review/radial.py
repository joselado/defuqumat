"""Q12: the three candidate radial integrals K_b, on an exactly solvable well, on Slater
shapes and on the pseudo-atomic orbitals on disk. kappa = 0.57/bohr unless said."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import spherical_in, spherical_kn, spherical_jn

def il(l, x, d=False): return spherical_in(l, x, derivative=d)
def kl(l, x, d=False): return spherical_kn(l, x, derivative=d) * 2 / np.pi   # k0 = exp(-x)/x

# ---- 1. exactly solvable: spherical square well, bound state at -kappa^2 (vacuum at 0)
print("square well: (c) = int R U i_l r^2 against (b) = -A/kappa, and (a) = int R i_l r^2 to a cutoff")
kap = 0.57
for l in (0, 1):
    aw, U0 = 2.5, 1.2
    qin = np.sqrt(U0 - kap ** 2)
    # tune the radius so that the bound state sits at exactly -kappa^2
    f = lambda aw: qin * spherical_jn(l, qin * aw, derivative=True) * kl(l, kap * aw) \
        - kap * kl(l, kap * aw, d=True) * spherical_jn(l, qin * aw)
    grid = np.linspace(0.5, 6, 400); vals = [f(x) for x in grid]
    i = next(i for i in range(len(grid) - 1) if vals[i] * vals[i + 1] < 0)
    aw = brentq(f, grid[i], grid[i + 1])
    A = spherical_jn(l, qin * aw) / kl(l, kap * aw)
    c = quad(lambda r: spherical_jn(l, qin * r) * (-U0) * il(l, kap * r) * r * r, 0, aw)[0]
    a_cut = [quad(lambda r: spherical_jn(l, qin * r) * il(l, kap * r) * r * r, 0, aw)[0]
             + quad(lambda r: A * kl(l, kap * r) * il(l, kap * r) * r * r, aw, R)[0] for R in (5, 10, 20)]
    print(f" l={l}: well radius {aw:.3f}: (c) {c:+.6f}, -A/kappa {-A/kap:+.6f};"
          f" (a) cut at 5, 10, 20 bohr: {a_cut[0]:.3f}, {a_cut[1]:.3f}, {a_cut[2]:.3f} (does not converge)")

# ---- 2. Slater shapes
def slater(n, zeta):
    N = (2 * zeta) ** (n + 0.5) / np.sqrt(float(np.prod(np.arange(1, 2 * n + 1))))
    R = lambda r: N * r ** (n - 1) * np.exp(-zeta * r)
    dR = lambda r: N * ((n - 1) * r ** (n - 2) - zeta * r ** (n - 1)) * np.exp(-zeta * r)
    return R, dR

def wron(l, R, dR, r, kap):          # r^2 W[i_l, R] = int_0^r i_l (lap_l - kappa^2) R
    return r * r * (il(l, kap * r) * dR(r) - kap * il(l, kap * r, d=True) * R(r))

def report(tag, n, l, zeta, kap=0.57):
    R, dR = slater(n, zeta)
    rr = np.linspace(1e-6, min(600.0, 80.0 / (zeta - kap)), 400001)
    f = R(rr) * il(l, kap * rr) * rr * rr
    cum = np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(rr))])
    Ka = cum[-1]; half = np.interp(0.5 * Ka, cum, rr)
    rm = (n + l) / (zeta - kap)
    Kb = -wron(l, R, dR, rm, kap)                        # = A/kappa where value and slope match k_l
    rs = np.linspace(0.2, 40, 4000); w = -wron(l, R, dR, rs, kap)
    rstar, Kmax = rs[np.argmax(w)], w.max()
    Kc_inf = -wron(l, R, dR, 60.0 / (zeta - kap), kap)
    print(f" {tag:22s} n={n} l={l} zeta={zeta:.4f}: (a) {Ka:.4e} [half of it beyond {half:.2f} bohr]"
          f"  (b) A/kappa at r_m={rm:.2f}: {Kb:.4e}; W at r_m*0.8, *1.2: {-wron(l,R,dR,0.8*rm,kap):.3e}, {-wron(l,R,dR,1.2*rm,kap):.3e};"
          f" max of W {Kmax:.4e} at {rstar:.2f}; (c) over all space {Kc_inf:.1e};  (a)/(b) = {Ka/Kb:.2f} bohr^2")
    return Ka, Kb

print("\nSlater orbitals at kappa = 0.57 (O exponents: Clementi and Raimondi 1963 as I recall them, not opened;"
      " the 6s exponent is scanned, no source at hand)")
a2p, b2p = report("O 2p", 2, 1, 2.2266)
a2s, b2s = report("O 2s", 2, 0, 2.2458)
print(f"   O: K_2p/K_2s by (a) {a2p/a2s:.3f}, by (b) {b2p/b2s:.3f}")
for z in (1.2, 1.4, 1.6, 1.8, 2.0):
    a6, b6 = report("6s (W-like), scanned", 6, 0, z)
    print(f"   weight of this 6s against O 2p: by (a) {a6/a2p:.2f}, by (b) {b6/b2p:.2f}, (a)/(b) mismatch {a6/a2p/(b6/b2p):.2f}")
print(" (a) as zeta -> kappa, 1s, closed form 4 zeta^2.5/(zeta^2-kappa^2)^2:", [f"{4*z**2.5/(z*z-0.57**2)**2:.2e}" for z in (2.0, 1.0, 0.7, 0.6, 0.58)])

# ---- 3. the pseudo-atomic orbitals on disk (outside r_c they are the atom's own tails)
from defuqumat.pseudo.upf import read_upf
import warnings; warnings.simplefilter("ignore")
print("\npseudo-atomic orbitals (PP_PSWFC): A(r_m) = R(r_m)/k_l(kappa r_m) by value, and the Wronskian form")
for path, lab, eps in (("O.pbe-n-rrkjus_psl.1.0.0.UPF", "2P", -0.663752), ("O.pbe-n-rrkjus_psl.1.0.0.UPF", "2S", -1.761148),
                       ("Pt.pbe-n-kjpaw_psl.0.1.UPF", "6S", -0.415929), ("Bi.pbe-dn-rrkjus_psl.1.0.0.UPF", "6P", -0.335940),
                       ("Bi.pbe-dn-rrkjus_psl.1.0.0.UPF", "6S", -1.064016)):
    p = read_upf("tests/data/pseudo/" + path)
    orb = next(o for o in p.orbitals if o.label.upper() == lab)
    r = np.asarray(p.r); chi = np.asarray(orb.chi); l = int(orb.l)
    chi = chi * np.sign(chi[np.argmax(np.abs(chi))])
    last = np.max(np.flatnonzero(np.abs(chi) > 1e-7 * np.abs(chi).max()))
    keep = (r > 1e-6) & (np.arange(len(r)) <= last)
    r, chi = r[keep], chi[keep]
    R = chi / r
    dR = np.gradient(R, r)
    kat = np.sqrt(-eps)
    out = []
    for kap in (0.57, kat):
        w = -(r * r * (il(l, kap * r) * dR - kap * il(l, kap * r, d=True) * R))
        vals = [np.interp(x, r, w) for x in (2.0, 3.0, 4.0, 6.0, 8.0)]
        im = np.argmax(w[r < 12])
        out.append(f"kappa={kap:.3f}: W at 2,3,4,6,8 bohr = " + ", ".join(f"{v:.4f}" for v in vals) + f"; max {w[im]:.4f} at {r[im]:.2f}")
    # overlap (a) at kappa = 0.57, cumulative
    integrand = R * il(l, 0.57 * r) * r * r
    cum = np.concatenate([[0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(r))])
    tot = cum[-1]
    halfr = np.interp(0.5 * tot, cum, r)
    # best Slater fit to chi (n from the label), by maximal overlap, and what it does to the tail
    n = int(lab[0]) if lab[0] in "12" else int(lab[0])
    def ov(z):
        s = slater(n, z)[0](r) * r
        return -np.trapezoid(s * chi, r) / np.sqrt(np.trapezoid(chi * chi, r))
    from scipy.optimize import minimize_scalar
    zfit = minimize_scalar(ov, bounds=(0.3, 6), method="bounded").x
    Rs, dRs = slater(n, zfit)
    rm = (n + l) / (zfit - 0.57)
    Kb_slater = -wron(l, Rs, dRs, rm, 0.57)
    report(f"   fit of {lab}", n, l, zfit)
    print(f" {path.split('.')[0]} {lab}: eps={eps:.3f} Ry, atomic kappa {kat:.3f}; r max of chi {r[np.argmax(np.abs(chi))]:.2f}\n    "
          + "\n    ".join(out)
          + f"\n    overlap (a) at 0.57 to r={r[-1]:.0f}: {tot:.3f}, half of it beyond {halfr:.1f} bohr"
          + f"\n    Slater fit (n={n}): zeta={zfit:.3f}, overlap {-ov(zfit):.4f}; its (b) at r_m={rm:.2f}: {Kb_slater:.4f}")
