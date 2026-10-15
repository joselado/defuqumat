"""The phonons of aluminium and their electron-phonon linewidths, along L-Gamma-X."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

calc = Calculator.from_file("al.in", pseudo_dir=PSEUDO)

# q in units of 2 pi / a. A metal at q = 0 gives an error, so Gamma is left out.
to_l = [(t, t, t) for t in (0.5, 0.375, 0.25, 0.125)]
to_x = [(t, 0, 0) for t in (0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)]
results = [calc.get_electron_phonon(q=q, q_cartesian=True) for q in to_l + to_x]

save(x=np.array([-np.linalg.norm(q) for q in to_l] + [q[0] for q in to_x]),
     frequencies=np.array([r.frequencies for r in results]),
     gamma_ghz=np.array([r.gamma_ghz for r in results]),    # (q, sigma, mode)
     lambdas=np.array([r.lambdas for r in results]),
     sigmas=results[0].sigmas, dos=results[0].dos)
