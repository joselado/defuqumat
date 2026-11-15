"""Q1: multiplier i(k+G)_cart/kappa before the fold, against central differences."""
import numpy as np
from defuqumat.basis.sample import sample_wavefunctions
from defuqumat.workflows.transport import _tip_sphere, _flat_axis
from defuqumat.transport.substrate import plane_groups

rng = np.random.default_rng(0)

def sphere(at, k, ecut):
    bg = 2 * np.pi * np.linalg.inv(at).T
    n = [int(np.ceil(np.sqrt(ecut) * np.linalg.norm(at[i]) / (2 * np.pi))) + 1 for i in range(3)]
    h = np.array(np.meshgrid(*[np.arange(-m, m + 1) for m in n], indexing="ij")).reshape(3, -1).T
    q = (h + k) @ bg
    keep = (q ** 2).sum(1) <= ecut
    return h[keep], bg

def run(name, at, k, npol=2, pad=7):
    at = np.asarray(at, float)
    vol = abs(np.linalg.det(at))
    miller, bg = sphere(at, k, 12.0)
    npw = miller.shape[0]
    # padding: entries pointing at G = 0, masked
    miller_p = np.vstack([miller, np.zeros((pad, 3), int)])
    mask = np.r_[np.ones(npw, bool), np.zeros(pad, bool)]
    npwx = npw + pad
    nb = 3
    c = rng.normal(size=(nb, npol * npwx)) + 1j * rng.normal(size=(nb, npol * npwx))
    # decay at large G so the function is smooth-ish but not trivially so
    q = (miller_p + k) @ bg
    kappa = 0.57
    inv = np.linalg.inv(at)
    # flat tip plane: crystal height 0.6 along axis 2, 7 points
    s = np.c_[rng.uniform(size=(7, 2)), np.full(7, 0.6)]
    r = s @ at
    def sample(coeff, rcart, folded):
        pts = rcart @ inv
        if folded:
            flat = _flat_axis(pts, miller_p[None])
            assert flat == 2, flat
            groups = plane_groups(miller_p, 2)
            co, mi, kk, mk = _tip_sphere(coeff, miller_p, mask, k, npol, flat, pts, 2, groups)
        else:
            co, mi, kk, mk = coeff.reshape((-1, npol, npwx)), miller_p, k, mask
        return sample_wavefunctions(co, mi, kk, pts, vol, mask=mk)
    print(f"--- {name}: npw={npw} npol={npol} k={k}  a3={at[2]}")
    for folded in (False, True):
        for a, lab in enumerate("xyz"):
            F = np.tile(1j * q[:, a] / kappa, npol)
            exact = sample(c * F[None], r, folded)
            errs = []
            hs = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]
            for h in hs:
                e = np.zeros(3); e[a] = h
                # note: a z step leaves the plane, so each side folds on its own plane
                fd = (sample(c, r + e, folded) - sample(c, r - e, folded)) / (2 * h * kappa)
                errs.append(np.abs(fd - exact).max() / np.abs(exact).max())
            slope = np.polyfit(np.log(hs[:4]), np.log(errs[:4]), 1)[0]
            print(f" folded={folded!s:5} d/d{lab}: rel err at h=1e-3: {errs[-1]:.2e}  slope {slope:.2f}")
        # second derivative dz2 (Chen form) and dxy
        Fzz = np.tile(-(q[:, 2] ** 2) / kappa ** 2, npol)
        ex = sample(c * Fzz[None], r, folded)
        h = 1e-3; e = np.array([0, 0, h])
        fd = (sample(c, r + e, folded) - 2 * sample(c, r, folded) + sample(c, r - e, folded)) / (h * kappa) ** 2
        print(f" folded={folded!s:5} d2/dz2: rel err h=1e-3: {np.abs(fd-ex).max()/np.abs(ex).max():.2e}")
    # the two paths against each other
    F = np.tile(1j * q[:, 0] / kappa, npol)
    d = np.abs(sample(c * F[None], r, True) - sample(c * F[None], r, False)).max()
    print(f" folded vs unfolded (px): {d:.2e}")
    # what the wrong order gives: fold first, then multiply by the in-plane (k+G)_x only
    pts = r @ inv
    groups = plane_groups(miller_p, 2)
    co, mi, kk, mk = _tip_sphere(c, miller_p, mask, k, npol, 2, pts, 2, groups)
    qin = (mi + kk) @ bg            # h3 = 0, k3 = 0 after the fold
    wrong = sample_wavefunctions(co * (1j * qin[:, 0] / kappa)[None, None], mi, kk, pts, vol)
    right = sample(c * F[None], r, True)
    print(f" multiply AFTER the fold, px: rel error {np.abs(wrong-right).max()/np.abs(right).max():.2e}")

a = 4.65
hexagonal = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 15.0]]
sheared = [[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [1.3, 0.7, 15.0]]
run("hexagonal", hexagonal, np.array([1 / 3, 1 / 3, 0.0]))
run("hexagonal, k3 != 0", hexagonal, np.array([0.21, 0.13, 0.25]), npol=1)
run("sheared a3", sheared, np.array([0.21, 0.13, 0.0]))
run("sheared a3, k3 != 0", sheared, np.array([0.21, 0.13, 0.25]), npol=1)
