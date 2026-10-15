"""The optical conductivity and the Kerr angle of fcc nickel, on three k-meshes."""
from defuqumat import Calculator
from defuqumat.system.kpoints import KPoints
from common import PSEUDO, save

calc = Calculator.from_file("ni-soc.in", pseudo_dir=PSEUDO)
scf = calc.get_scf()
cell = calc.system.cell

out = {"moment": scf.magnetization_vector[2]}
for n in (4, 6, 8):
    mesh = KPoints.automatic((n, n, n), (0, 0, 0), cell, precision=cell.precision)
    sigma = calc.get_optical_conductivity(kpoints=mesh, nbnd=36, window=0.6,
                                          nw=200, broadening=0.01,
                                          fermi_energy=scf.fermi_energy)
    s = sigma.sigma_s_per_cm                     # (frequency, 3, 3), S/cm
    out[f"sxx{n}"] = s[:, 0, 0]
    out[f"sxy{n}"] = 0.5 * (s[:, 0, 1] - s[:, 1, 0])
    out[f"kerr{n}"] = sigma.kerr                 # degrees
    out[f"hall{n}"] = sigma.hall_conductivity[0, 1]
    out[f"fsum{n}"] = sigma.fsum
    out[f"plasma{n}"] = sigma.plasma_ev[0, 0]   # hbar omega_p, eV

save(energy=sigma.frequencies_ev, **out)
