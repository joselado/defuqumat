"""Foundation item 6: Peierls on P162's graphene Wannier set at 10 T (q = 7900), against Dirac.

Reads graphene-wannier-<mode>.npz and graphene-direct-bands.npz written by
mag_graphene_scf.py, so no SCF is repeated. First the hopping cutoff: the zero-field bands
of the bond list at each cutoff against the uncut list on the off-mesh 6x6 points, within
E_F +- 2 eV, and the largest cutoff whose error is under 0.1 meV is taken. Then
run_landau_levels at flux 1/7900, kappa (0.11, 0.23), 60 levels, dirac=True.
Usage: python mag_wannier_landau.py <dir with the npz> <mode> [q]
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from defuqumat.magnetic.peierls import Bonds
from defuqumat.units import RY_TO_EV
from defuqumat.wannier.interpolation import interpolate
from defuqumat.workflows.landau import run_landau_levels

src = Path(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else "pz"
q = int(sys.argv[3]) if len(sys.argv) > 3 else 7900
z = np.load(src / f"graphene-wannier-{mode}.npz")
shifted, count = z["ws_shifted"], z["ws_count"]
ws = None if shifted.size == 0 else (shifted, count)


def bands(kappa):
    return interpolate(np.atleast_2d(kappa), z["rvectors"], z["degeneracy"], z["hoppings"],
                       *(ws if ws is not None else (None, None)))[0]


w = SimpleNamespace(centres=z["centres"], rvectors=z["rvectors"], degeneracy=z["degeneracy"],
                    hoppings=z["hoppings"], lattice=z["lattice"], ws_shifts=ws, spin=None,
                    edge=float(z["fermi_energy"]), bands=bands)
ef = float(z["fermi_energy"])
direct = np.load(src / "graphene-direct-bands.npz")
pts = direct["off-mesh_kpoints"]

full = Bonds.from_blocks(w.rvectors, w.hoppings, w.centres, w.lattice, degeneracy=w.degeneracy, ws_shifts=ws)
ref = np.array([np.linalg.eigvalsh(full.zero_field(k)) for k in pts])
window = np.abs(ref - ef) * RY_TO_EV <= 2.0
print(f"uncut: {full.nb} bonds per cell; interpolation identity "
      f"{np.max(np.abs(ref - bands(pts))) * RY_TO_EV:.2e} eV", flush=True)
chosen = 0.0
study = []
for cutoff in (1e-7, 1e-6, 1e-5, 3e-5, 1e-4):
    cut = Bonds.from_blocks(w.rvectors, w.hoppings, w.centres, w.lattice, degeneracy=w.degeneracy,
                            ws_shifts=ws, cutoff=cutoff)
    e = np.array([np.linalg.eigvalsh(cut.zero_field(k)) for k in pts])
    err = float(np.max(np.abs(e - ref)[window])) * RY_TO_EV
    study.append({"cutoff_ry": cutoff, "nb": cut.nb, "dropped_max_ry": cut.dropped,
                  "dropped_norm_ry": cut.dropped_norm, "band_error_ev": err})
    print("cutoff", json.dumps(study[-1]), flush=True)
    if err < 1e-4:
        chosen = cutoff

t0 = time.perf_counter()
out = run_landau_levels(w, flux=(1, q), kappa=((0.11, 0.23),), nlevels=60, dirac=True, cutoff=chosen)
wall = time.perf_counter() - t0
d = out.dirac
e = d["per_k"][0]
record = {"mode": mode, "q": q, "B": out.field_tesla, "dim": out.dim, "nbonds": out.nbonds,
          "cutoff_ry": chosen, "cutoff_study": study, "E_D_ry": d["E_D"], "E_D_minus_EF_ev": (d["E_D"] - ef) * RY_TO_EV,
          "v_F": d["v_F"], "v_F_hole": d["v_F_hole"], "E1_dirac_ev": d["E1_dirac_ev"],
          "energies_ev": e["energies_ev"], "ratios": e["ratios"], "fit_electron": e.get("fit_electron"),
          "fit_hole": e.get("fit_hole"), "zero_multiplicity": e["zero_multiplicity"],
          "zero_level_ev": e["zero_level_ev"], "zero_site_traces": e["zero_site_traces"],
          "sites": out.sites.tolist(), "multiplicities": e["multiplicities"], "splittings_ev": e["splittings_ev"],
          "hermiticity_ry": out.hermiticity.tolist(), "hermiticity_rel": out.relative.tolist(),
          "plaquette": None if out.plaquette is None else [out.plaquette.real, out.plaquette.imag],
          "timings": out.timings, "wall_s": wall}
print("item 6:", json.dumps(record, default=float), flush=True)
with open(src / f"wannier-landau-{mode}-{q}.json", "w") as f:
    json.dump(record, f, indent=1, default=float)
