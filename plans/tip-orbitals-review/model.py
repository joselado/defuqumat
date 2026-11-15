"""A honeycomb sheet of Gaussian wells, diagonalised exactly in plane waves.
Vacuum level is 0, so kappa^2 = -E exactly. Used for Q2, Q3/V7, Q4 (TR), Q6, Q11."""
import types
import numpy as np
from defuqumat.basis.sample import sample_wavefunctions
from defuqumat.transport.substrate import exit_overlap
from defuqumat.transport.green import transmission

a, c = 4.65, 24.0
at = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, c]])
inv = np.linalg.inv(at)
bg = 2 * np.pi * inv.T
vol = abs(np.linalg.det(at))
cell = types.SimpleNamespace(at=at, volume=vol)
V0, sig = 3.0, 1.0
tau = np.array([[1 / 3, 2 / 3, 0.5], [2 / 3, 1 / 3, 0.5]])     # A, B; hollow at (0,0)
ECUT = 30.0

def sphere(k):
    n = [int(np.ceil(np.sqrt(ECUT) * np.linalg.norm(at[i]) / (2 * np.pi))) + 1 for i in range(3)]
    h = np.array(np.meshgrid(*[np.arange(-m, m + 1) for m in n], indexing="ij")).reshape(3, -1).T
    q = (h + k) @ bg
    keep = (q ** 2).sum(1) <= ECUT
    return h[keep]

def solve(k, nb=6):
    k = np.asarray(k, float)
    h = sphere(k)
    q = (h + k) @ bg
    dh = h[:, None, :] - h[None, :, :]
    dG = dh @ bg
    sf = np.exp(-2j * np.pi * (dh @ tau.T)).sum(-1)
    H = -V0 * (2 * np.pi) ** 1.5 * sig ** 3 / vol * np.exp(-0.5 * (dG ** 2).sum(-1) * sig ** 2) * sf
    H = H + np.diag((q ** 2).sum(1))
    e, v = np.linalg.eigh(H)
    return e[:nb], v[:, :nb].T.copy(), h, q

def amp(coef, h, k, rcart, F=None):
    co = coef if F is None else coef * F[None]
    return sample_wavefunctions(co, h, np.asarray(k, float), np.atleast_2d(rcart) @ inv, vol)

def plane(height, n=9):
    s = np.array([[i / n, j / n, 0.5 + height / c] for i in range(n) for j in range(n)])
    return s @ at

def Vr(rcart):
    out = np.zeros(len(rcart))
    for t in tau:
        for i in range(-3, 4):
            for j in range(-3, 4):
                R = (t + [i, j, 0]) @ at
                out += -V0 * np.exp(-((rcart - R) ** 2).sum(1) / (2 * sig ** 2))
    return out

print(f"npw at Gamma: {sphere(np.zeros(3)).shape[0]}")
# ---------------------------------------------------------------- Q2, Q11, Q6 at Gamma and a generic k
for k in ([0, 0, 0], [0.21, 0.13, 0]):
    e, v, h, q = solve(k)
    n0 = 0
    kap = np.sqrt(-e[n0])
    print(f"\n=== k={k}: E = {np.round(e[:4], 4)} Ry, band 0 kappa = {kap:.4f}/bohr")
    for hgt in (3.0, 4.0, 5.0):
        r = plane(hgt)
        s = amp(v[n0:n0 + 1], h, k, r)[0]
        pz = amp(v[n0:n0 + 1], h, k, r, 1j * q[:, 2] / kap)[0]
        lap = amp(v[n0:n0 + 1], h, k, r, -(q ** 2).sum(1))[0]
        res = np.abs(lap - kap ** 2 * s) / np.abs(kap ** 2 * s)
        resV = np.abs(Vr(r)) / kap ** 2
        ratio = np.abs(s - pz) ** 2 / np.abs(s + pz) ** 2
        print(f" h={hgt}: Re(pz/s) in [{(pz/s).real.min():+.3f},{(pz/s).real.max():+.3f}]"
              f"  T(s-pz)/T(s+pz) min over pixels {ratio.min():.1f}"
              f"  helmholtz residue max {res.max():.2e} (|V(r)|/kappa^2 max {resV.max():.2e},"
              f" difference {np.abs(res-resV).max():.1e})")
        for w in (0.5, 1.0, 2.0):
            g = amp(v[n0:n0 + 1], h, k, r, np.exp(-0.5 * (q ** 2).sum(1) * w ** 2))[0]
            pred = np.exp(0.5 * kap ** 2 * w ** 2)
            dev = np.abs(g / s / pred - 1)
            cor = np.corrcoef(np.abs(g) ** 2, np.abs(s) ** 2)[0, 1]
            print(f"    gaussian s, width {w}: amp/(point*exp(k^2w^2/2)) - 1: max {dev.max():.2e},"
                  f" map corr {cor:.6f}, K0={pred:.3f}")
        for zeta in (2.0, 1.0, 0.8):
            g = amp(v[n0:n0 + 1], h, k, r, 8 * zeta ** 4 / (zeta ** 2 + (q ** 2).sum(1)) ** 2)[0]
            pred = 8 * zeta ** 4 / (zeta ** 2 - kap ** 2) ** 2
            dev = np.abs(g / s / pred - 1)
            print(f"    slater 1s, zeta {zeta}: amp/(point*K0) - 1: max {dev.max():.2e}"
                  f"  (exp(-(zeta-kappa)h) = {np.exp(-(zeta-kap)*hgt):.1e})")
    # Q6: eigensolver-like noise, uniform on the sphere, norm 1e-6
    rng = np.random.default_rng(1)
    d = rng.normal(size=v[n0].shape) + 1j * rng.normal(size=v[n0].shape)
    d *= 1e-6 / np.linalg.norm(d)
    for hgt in (5.0, 8.0):
        r = plane(hgt)
        s = amp(v[n0:n0 + 1], h, k, r)[0]
        out = []
        for lab, F in (("s", np.ones(len(q))), ("px", 1j * q[:, 0] / kap), ("pz", 1j * q[:, 2] / kap),
                       ("dz2", -3 * q[:, 2] ** 2 / kap ** 2 - 1)):
            noise = amp(d[None], h, k, r, F)[0]
            out.append(f"{lab} {np.sqrt((np.abs(noise)**2).mean())/np.sqrt((np.abs(s)**2).mean()):.1e}")
        print(f" noise 1e-6 on c, h={hgt}: rms noise amplitude / rms |psi|: " + ", ".join(out)
              + f"   (max|k+G|/kappa = {np.sqrt(ECUT)/kap:.1f})")

# ---------------------------------------------------------------- V7: the Dirac pair at K
K = np.array([1 / 3, 1 / 3, 0.0])
e, v, h, q = solve(K)
print(f"\n=== K: E = {np.round(e[:4], 6)}  (pair split {e[1]-e[0]:.1e})")
kap = np.sqrt(-e[0])
pair = v[:2]
# rotate the pair by a random unitary, as a degenerate solver may (rule D4)
rng = np.random.default_rng(3)
u, _ = np.linalg.qr(rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)))
pair_rot = u @ pair
S = exit_overlap(pair, h, 0.5 - 4.0 / c, 2, cell)[None]
print(f" S on the pair (exit plane 4 bohr below): diag {np.real(np.diag(S[0]))}, offdiag {abs(S[0][0,1]):.1e}")
fx, fy, fz = (1j * q[:, i] / kap for i in range(3))
tips = {"s": np.ones(len(q), complex), "pz": fz, "px": fx, "py": fy,
        "px+i py (C*)": fx + np.conj(1j) * fy, "px-i py (C*)": fx + np.conj(-1j) * fy}
sites = {"hollow": [0, 0], "A": [1 / 3, 2 / 3], "B": [2 / 3, 1 / 3]}
for name, st in sites.items():
    r0 = np.array([[st[0], st[1], 0.5 + 3.0 / c]]) @ at
    line = []
    for lab, F in tips.items():
        t = []
        for P in (pair, pair_rot):
            Sp = exit_overlap(P, h, 0.5 - 4.0 / c, 2, cell)[None]
            A = amp(P, h, K, r0, F)[None]                       # (1, 2, 1)
            t.append(transmission(A, Sp, np.ones(1), np.ones((1, 2)))[0])
        line.append(f"{lab}: {t[0]:.3e} (rotated pair {t[1]:.3e})")
    # winding, no derivative: m-resolved weight on a circle of radius 0.2 bohr, summed over the pair
    phi = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    circ = r0 + 0.2 * np.c_[np.cos(phi), np.sin(phi), 0 * phi]
    psi = amp(pair, h, K, circ)
    W = {m: (np.abs((psi * np.exp(-1j * m * phi)).mean(1)) ** 2).sum() for m in (-1, 0, 1)}
    one = {m: np.abs((psi[0] * np.exp(-1j * m * phi)).mean()) ** 2 for m in (-1, 1)}
    psir = amp(pair_rot, h, K, circ)
    oner = {m: np.abs((psir[0] * np.exp(-1j * m * phi)).mean()) ** 2 for m in (-1, 1)}
    print(f" site {name}:\n   " + "\n   ".join(line))
    print(f"   winding weight on the circle, pair sum: m=+1 {W[1]:.3e}, m=-1 {W[-1]:.3e}, m=0 {W[0]:.3e}")
    print(f"   band 0 alone: m=+1 {one[1]:.3e}, m=-1 {one[-1]:.3e}; after the rotation: {oner[1]:.3e}, {oner[-1]:.3e}")

# ---------------------------------------------------------------- time reversal: T_{-k}(C) against T_k(C*)
k = np.array([0.21, 0.13, 0.0])
ep, vp, hp, qp = solve(k, nb=3)
em, vm, hm, qm = solve(-k, nb=3)
kap = 0.6
pts = np.array([[0.11, 0.37, 0.5 + 3.0 / c], [0.52, 0.23, 0.5 + 3.5 / c], [0.8, 0.71, 0.5 + 3.0 / c]]) @ at
def T(v_, h_, q_, kk, C):
    F = sum(coef.conjugate() * {"s": np.ones(len(q_), complex), "px": 1j * q_[:, 0] / kap,
                                "py": 1j * q_[:, 1] / kap, "pz": 1j * q_[:, 2] / kap}[b]
            for b, coef in C.items())
    Sk = exit_overlap(v_, h_, 0.5 - 4.0 / c, 2, cell)[None]
    return transmission(amp(v_, h_, kk, pts, F)[None], Sk, np.ones(1), np.ones((1, 3)))
print("\n=== time reversal, three bands, exit-plane S")
for C in ({"px": 1, "py": 1j}, {"s": 1, "px": 0.5j}, {"s": 1, "pz": -0.5}):
    Cc = {b: np.conj(x) for b, x in C.items()}
    tm, tp, tpc = T(vm, hm, qm, -k, C), T(vp, hp, qp, k, C), T(vp, hp, qp, k, Cc)
    print(f" C={C}: |T_-k(C) - T_k(C*)|/T = {np.abs(tm-tpc).max()/tm.max():.1e},"
          f"  |T_-k(C) - T_k(C)|/T = {np.abs(tm-tp).max()/tm.max():.1e}")
