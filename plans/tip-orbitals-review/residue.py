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
import sys
ECUT = float(sys.argv[1])

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


e, v, h, q = solve([0, 0, 0])
kap = np.sqrt(-e[0])
out = []
for hgt in (3.0, 4.0, 5.0):
    r = plane(hgt)
    s_ = amp(v[:1], h, [0, 0, 0], r)[0]
    lap = amp(v[:1], h, [0, 0, 0], r, -(q ** 2).sum(1))[0]
    out.append(f"h={hgt}: {(np.abs(lap - kap**2 * s_) / np.abs(kap**2 * s_)).max():.2e} (|V|/k^2 {np.abs(Vr(r)).max()/kap**2:.1e})")
print(f"ECUT={ECUT} npw={len(h)} E0={e[0]:.6f} residue max: " + "; ".join(out))
