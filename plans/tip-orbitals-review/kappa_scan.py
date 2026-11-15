"""Bardeen weight of a pseudo-atomic orbital: (c) = int i_l(kr) U_t R r^2 dr with
U_t R = (eps - T_l) R, against the overlap (a) = int i_l(kr) R r^2 dr, scanned in kappa
through the orbital's own decay kappa_a = sqrt(-eps). Ry units, T = -nabla^2."""
import warnings
import numpy as np
from scipy.special import spherical_in
from defuqumat.pseudo.upf import read_upf
warnings.simplefilter("ignore")

cases = (("O.pbe-n-rrkjus_psl.1.0.0.UPF", "2P", -0.663752),
         ("Pt.pbe-n-kjpaw_psl.0.1.UPF", "6S", -0.415929))
for path, lab, eps in cases:
    p = read_upf("tests/data/pseudo/" + path)
    orb = next(o for o in p.orbitals if o.label.upper() == lab)
    r = np.asarray(p.r, float); chi = np.asarray(orb.chi, float); l = int(orb.l)
    chi = chi * np.sign(chi[np.argmax(np.abs(chi))])
    keep = r > 1e-4
    r, chi = r[keep], chi[keep]
    # chi = r R; T_l R = (-chi'' + l(l+1) chi / r^2) / r
    d1 = np.gradient(chi, r); d2 = np.gradient(d1, r)
    source = (eps * chi + d2 - l * (l + 1) * chi / r**2) / r      # U_t R
    ka = np.sqrt(-eps)
    # where the source has died: the radius holding 99.9 per cent of |source| r^2
    s_abs = np.cumsum(np.abs(source) * r * r * np.gradient(r))
    r999 = r[np.searchsorted(s_abs, 0.999 * s_abs[-1])]
    print(f"{path.split('.')[0]} {lab}: l={l} eps={eps} kappa_atom={ka:.3f} "
          f"grid end {r[-1]:.1f} bohr, 99.9% of |U_t R| inside {r999:.2f} bohr")
    trap = lambda f, x: float(np.sum(0.5 * (f[1:] + f[:-1]) * np.diff(x)))
    for cut in (8.0, 12.0, 16.0):
        m = r <= cut
        row = []
        for kap in (0.40, 0.57, 0.70, ka, 0.90, 1.00):
            il = spherical_in(l, kap * r[m])
            a = trap(il * chi[m] * r[m], r[m])               # int i_l R r^2
            c = -trap(il * source[m] * r[m] * r[m], r[m])    # sign as the review's (c)
            row.append(f"k={kap:.3f}: a={a:9.3f} c={c:8.4f} (k2-ka2)a={(kap**2-ka**2)*a:8.4f}")
        print(f"  cut {cut:4.1f}:  " + "\n              ".join(row))
