"""The spectral function of a perfect and of a vacancy Si8 supercell unfolded to the fcc path L-G-X."""
import numpy as np
from defuqumat import Calculator
from defuqumat.units import RY_TO_EV
from common import PSEUDO, save

si = dict(pseudos={"Si": "Si.pz-vbc.UPF"}, pseudo_dir=PSEUDO, ecut=163.3, announce=False)
primitive = Calculator.from_keywords(
    lattice="fcc", a=5.43, atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
    kmesh=(4, 4, 4), **si)
path = primitive.kpath("L-G-X", 40)
bands = primitive.get_bands(path, nbnd=8)
kpoints = np.asarray(path.crystal(primitive.system.cell))
lattice = np.asarray(primitive.system.cell.at)
sites = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0),
         (.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)]
out = {}
for name, atoms in (("perfect", sites), ("vacancy", sites[1:])):
    cell = Calculator.from_keywords(lattice="cubic", a=5.43, atoms=[("Si", p) for p in atoms],
                                    kmesh=(2, 2, 2), smearing=0.1, **si)
    scf = cell.get_scf()
    a = cell.get_kpm_unfolded(lattice, kpoints, trace="stochastic", n_random=8, broadening=0.01,
                              emin=scf.fermi_energy - 1.0, emax=scf.fermi_energy + 0.4)
    out[f"{name}_e"] = (a.energies - scf.fermi_energy) * RY_TO_EV
    out[name] = a.spectral / RY_TO_EV
    out[f"{name}_n"] = a.n_moments
top = np.asarray(bands.eigenvalues)[:, 3].max()
save(bands=(np.asarray(bands.eigenvalues) - top) * RY_TO_EV, **out)
print({k: v for k, v in out.items() if k.endswith("_n")})
