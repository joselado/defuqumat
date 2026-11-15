
import numpy as np, warnings
warnings.simplefilter("ignore")
from scipy.special import spherical_in
from defuqumat.pseudo.upf import read_upf
kap = 0.57
for path, lab, eps in (("O.pbe-n-rrkjus_psl.1.0.0.UPF", "2P", -0.663752), ("Pt.pbe-n-kjpaw_psl.0.1.UPF", "6S", -0.415929)):
    p = read_upf("tests/data/pseudo/" + path)
    orb = next(o for o in p.orbitals if o.label.upper() == lab)
    r = np.asarray(p.r); chi = np.asarray(orb.chi); l = int(orb.l)
    chi = chi * np.sign(chi[np.argmax(np.abs(chi))])
    last = np.max(np.flatnonzero(np.abs(chi) > 1e-7 * np.abs(chi).max()))
    keep = (r > 1e-6) & (np.arange(len(r)) <= last)
    r, chi = r[keep], chi[keep]; R = chi / r; dR = np.gradient(R, r)
    il = spherical_in(l, kap * r); dil = spherical_in(l, kap * r, derivative=True)
    w = -(r * r * (il * dR - kap * dil * R))             # -int_0^r i_l (lap - kappa^2) R
    f = R * il * r * r
    cum = np.concatenate([[0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(r))])
    c_plan = w + (-eps - kap ** 2) * cum                  # -int_0^r i_l (lap + eps) R = -int i_l U_t R
    at = lambda x, arr: np.interp(x, r, arr)
    print(path.split(".")[0], lab, "plan (c), -int_0^r i_l(kappa r)(eps - T)R r^2, at r = 3, 4, 6, 8, end:",
          ", ".join(f"{at(x, c_plan):.3f}" for x in (3, 4, 6, 8, r[-1])), f"(end r = {r[-1]:.0f});  W there:",
          ", ".join(f"{at(x, w):.3f}" for x in (3, 4, 6, 8, r[-1])))
