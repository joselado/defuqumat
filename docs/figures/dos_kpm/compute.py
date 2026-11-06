"""The DOS of silicon by the kernel polynomial method, near the gap and over the whole basis."""
import numpy as np
from defuqumat import Calculator
from defuqumat.units import RY_TO_EV
from common import PSEUDO, save

silicon = Calculator.from_keywords(
    lattice="fcc", a=5.43, atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
    pseudos={"Si": "Si.pz-vbc.UPF"}, pseudo_dir=PSEUDO, ecut=163.3, kmesh=(4, 4, 4),
    announce=False)
scf = silicon.get_scf()
kpm = silicon.get_kpm_dos(broadening=0.01, n_random=16, emax=scf.homo + 0.6)

# the same window from 12 bands, with a Gaussian of the same half width
bands = silicon.get_dos(nbnd=12, scheme="gaussian", degauss=kpm.broadening / np.sqrt(np.log(2)),
                        emin=kpm.energies[0], emax=kpm.energies[-1], delta_e=0.002)

# the same moments over the whole basis, cut to 200 of them
edges = (kpm.centre - kpm.half_width, kpm.centre + kpm.half_width)
whole = kpm.reconstructed(energies=np.linspace(*edges, 3000), n_moments=200)
v0 = float(np.mean(scf.potential))
e = whole.energies
inside = (e > v0) & (e < v0 + silicon.system.ecutwfc)
free = np.where(inside, float(silicon.system.cell.volume)
                * np.sqrt(np.clip(e - v0, 0.0, None)) / (2.0 * np.pi**2), 0.0)

gap = scf.homo + 0.02
save(e=(kpm.energies - scf.homo) * RY_TO_EV, kpm=kpm.dos / RY_TO_EV,
     kpm_error=kpm.dos_error / RY_TO_EV,
     bands_e=(bands.energies - scf.homo) * RY_TO_EV, bands=bands.dos / RY_TO_EV,
     whole_e=e * RY_TO_EV, whole=whole.dos / RY_TO_EV, free=free / RY_TO_EV,
     n_moments=kpm.n_moments, count=kpm.count_at(gap), count_error=kpm.count_error_at(gap))
print(f"{kpm.n_moments} moments, {kpm.count_at(gap):.3f} +- {kpm.count_error_at(gap):.3f} "
      "states below the gap")
