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
import sys
V0, sig = float(sys.argv[1]), 1.0
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


for k in ([0, 0, 0], [1/3, 1/3, 0]):
    e, v, h, q = solve(k)
    kap = np.sqrt(-e[0])
    nb = 2 if k[0] else 1
    print(f"V0={V0} k={np.round(k,3)}: E={np.round(e[:3],4)}, kappa={kap:.3f}")
    for hgt in (3.0, 4.0, 5.0, 7.0):
        r = plane(hgt, 12)
        s_ = amp(v[:nb], h, k, r)
        pm = (np.abs(s_) ** 2).sum(0)
        for w in (0.5, 1.0, 2.0):
            g = amp(v[:nb], h, k, r, np.exp(-0.5 * (q ** 2).sum(1) * w ** 2))
            gm = (np.abs(g) ** 2).sum(0)
            ratio = gm / pm / np.exp(kap ** 2 * w ** 2)
            good = pm > 1e-3 * pm.max()
            print(f"  h={hgt} w={w}: map ratio / K0^2: mean {ratio[good].mean():.4f}, spread over pixels (max/min) {ratio[good].max()/ratio[good].min():.4f}, corr {np.corrcoef(gm, pm)[0,1]:.5f}")
