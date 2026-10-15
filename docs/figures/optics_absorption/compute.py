"""The absorption spectrum of silicon with three kernels: RPA, ALDA and the bootstrap."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

calc = Calculator.from_file("si.in", pseudo_dir=PSEUDO)
omega = np.arange(0.0, 0.60, 0.004)      # Ry

out = {}
for kernel in ("rpa", "alda", "bootstrap"):
    s = calc.get_absorption(omega, kernel=kernel, nbnd=60, ecut_response=8.0,
                            broadening=0.012)
    out[kernel] = s.absorption
    out[kernel + "_alpha"] = s.alpha
    out[kernel + "_passes"] = s.iterations
    out[kernel + "_residual"] = s.static_residual
    if kernel == "rpa":
        eps = s.epsilon_no_local_fields
        out["rpa_no_local_fields"] = np.mean([eps[:, i, i].imag for i in range(3)], axis=0)

save(energy=s.frequencies_ev, **out)
