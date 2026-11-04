"""Silicon: eight sp3 Wannier functions from twelve bands, the interpolated
bands against the bands computed directly, on a 4x4x4 and a 6x6x6 mesh."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

RY = 13.605693123
calc = Calculator.from_keywords(
    lattice="fcc", a=5.3976, atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
    pseudos={"Si": "Si.pz-vbc.UPF"}, pseudo_dir=PSEUDO, ecut=272.1, kmesh=(8, 8, 8),
)
scf = calc.get_scf()
path = calc.kpath("L-G-X-W-K-G", 161)
crystal = np.asarray(calc.system.cell.k_to_crystal(path.coords))
coords = np.asarray(path.coords)
data = {"x": np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))])}
# The corners are where the direction of the path changes.
step = np.diff(coords, axis=0)
step /= np.linalg.norm(step, axis=1)[:, None]
turn = np.nonzero(np.einsum("ij,ij->i", step[1:], step[:-1]) < 0.999)[0] + 1
data["ticks"] = data["x"][np.concatenate([[0], turn, [len(coords) - 1]])]
data["labels"] = np.array(["L", "G", "X", "W", "K", "G"])
data["direct"] = np.asarray(calc.get_bands(path, nbnd=12).eigenvalues) * RY
for n in (4, 6):
    w = calc.get_wannier(8, ["Si:sp3"], num_bands=12, divisions=(n, n, n),
                         dis_win=(None, 17.0), dis_froz=(None, 6.4),
                         engine_options=dict(num_iter=200, conv_window=3))
    data[f"wannier{n}"] = w.bands_ev(crystal)
    valence = np.max(np.abs(data[f"wannier{n}"][:, :4] - data["direct"][:, :4]))
    data[f"error{n}"] = valence
    data[f"spread{n}"] = w.spread_terms_angstrom2()["omega_total"]
    print(f"{n}x{n}x{n}: Omega {data[f'spread{n}']:.4f} A^2, valence bands within {valence:.3f} eV",
          flush=True)
data["frozen"] = 6.4
save(**data)
