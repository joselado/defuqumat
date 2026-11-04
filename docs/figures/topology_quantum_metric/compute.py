"""The quantum metric of diamond, silicon and AlAs along L-G-X, and their localization tensors."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

CRYSTALS = {
    "C": dict(a=3.567, atoms=[("C", (0, 0, 0)), ("C", (0.25, 0.25, 0.25))],
              pseudos={"C": "C.pz-rrkjus.UPF"}, ecut=367.4, ecutrho=2939.0),
    "Si": dict(a=5.43, atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
               pseudos={"Si": "Si.pz-vbc.UPF"}, ecut=163.3),
    "AlAs": dict(a=5.66, atoms=[("Al", (0, 0, 0)), ("As", (0.25, 0.25, 0.25))],
                 pseudos={"Al": "Al.pz-vbc.UPF", "As": "As.pz-bhs.UPF"}, ecut=217.7),
}

data = {}
for name, keywords in CRYSTALS.items():
    calc = Calculator.from_keywords(lattice="fcc", kmesh=(6, 6, 6), pseudo_dir=PSEUDO,
                                    **keywords)
    path = calc.kpath("L-G-X", 81)
    along = calc.get_quantum_metric(kpoints=path, nbnd=30)
    data[name + "_trace"] = np.trace(along.metric, axis1=1, axis2=2)
    coords = np.asarray(path.coords)                 # 2 pi / alat
    data[name + "_x"] = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))])
    mesh = Calculator.from_keywords(lattice="fcc", kmesh=(12, 12, 12), pseudo_dir=PSEUDO,
                                    **keywords).get_quantum_metric(nbnd=30)
    data[name + "_x2"] = np.trace(mesh.localization_tensor) / 3.0
    data[name + "_truncation"] = mesh.truncation
    bands = calc.get_bands(path, nbnd=8)
    e = np.asarray(bands.eigenvalues)
    data[name + "_direct_gap"] = float(np.min(e[:, 4] - e[:, 3])) * 13.605693123
    print(name, "<x^2>_c %.4f bohr^2, truncation %.1e, max Tr g on the path %.3f bohr^2, direct gap %.3f eV"
          % (data[name + "_x2"], mesh.truncation, data[name + "_trace"].max(),
             data[name + "_direct_gap"]), flush=True)

save(**data)
