"""The shared zero-field graphene state of the orbital-field work, and P162's Wannier set on it.

Foundation item 6 and the state every route reads. Writes, into the directory given as
the first argument (default plans/orbital-field):

- graphene-scf.npz: density and potential (nspin, n1, n2, n3), fermi_energy (Ry),
  eigenvalues (nk, nbnd) Ry, the SCF k-points (crystal), total_energy, the lattice (bohr).
- graphene-wannier.npz: every array of the WannierFunctions the Peierls model needs
  (centres, rvectors, degeneracy, hoppings, lattice, ws shifts, edge, labels, spreads),
  so the Landau probes can rebuild it without a second SCF.

and prints the band distance of the interpolated bands against direct DFT bands on two
6x6 meshes: on the 12x12 Wannier mesh (exact by construction, a sanity check) and shifted
off it by (1/24, 1/24) (the real test), within E_F + 2 eV. Usage:
    python mag_graphene_scf.py [outdir] [auto|pz]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from defuqumat import Calculator
from defuqumat.system.kpoints import KPoints
from defuqumat.units import RY_TO_EV
from defuqumat.wannier.interpolation import band_distance

out = Path(sys.argv[1] if len(sys.argv) > 1 else "plans/orbital-field")
mode = sys.argv[2] if len(sys.argv) > 2 else "auto"
out.mkdir(parents=True, exist_ok=True)

calc = Calculator.from_keywords(
    lattice="hexagonal", a=2.46, c=15.0,
    atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
    pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=50 * RY_TO_EV, kmesh=(12, 12, 1), smearing=0.01 * RY_TO_EV, smearing_type="mv")
t0 = time.perf_counter()
scf = calc.get_scf()
t_scf = time.perf_counter() - t0
cell = calc.system.cell
lattice = np.asarray(cell.at)
kcrys = np.asarray(calc.system.kpoints.crystal(cell))
print(f"scf: converged {scf.converged} iterations {scf.iterations} total_energy {scf.total_energy!r} "
      f"fermi {scf.fermi_energy!r} Ry, density {np.asarray(scf.density).shape}, {t_scf:.1f} s", flush=True)
np.savez(out / "graphene-scf.npz",
         density=np.asarray(scf.density), potential=np.asarray(scf.potential),
         fermi_energy=np.float64(scf.fermi_energy), eigenvalues=np.asarray(scf.eigenvalues),
         kpoints_crystal=kcrys, total_energy=np.float64(scf.total_energy), lattice=lattice,
         description=np.array("graphene a=2.46 A c=15 A C.pbe-hgh 50 Ry 12x12x1 mv 0.01 Ry; Ry, bohr"))

t0 = time.perf_counter()
if mode == "auto":
    w = calc.get_wannier(num_wann=8, projections="auto", divisions=(12, 12, 1))
else:
    ef = scf.fermi_energy * RY_TO_EV
    w = calc.get_wannier(num_wann=2, projections=["C:pz"], num_bands=8, divisions=(12, 12, 1),
                         dis_froz=(None, ef + 2.0))
t_w = time.perf_counter() - t0
print(f"wannier[{mode}]: num_wann {w.num_wann} num_bands {w.num_bands} converged {w.converged} "
      f"iterations {w.iterations} omega_total {w.omega_total:.6f} bohr^2 nsym {w.nsym} "
      f"edge {w.edge!r} labels {w.projection_labels} {t_w:.1f} s", flush=True)
print("centres (bohr):", np.round(w.centres, 5).tolist())
print("spreads (A^2):", np.round(w.spreads_angstrom2, 5).tolist())
shifted, count = (None, None) if w.ws_shifts is None else w.ws_shifts
np.savez(out / f"graphene-wannier-{mode}.npz",
         num_wann=w.num_wann, num_bands=w.num_bands, centres=w.centres, spreads=w.spreads,
         rvectors=w.rvectors, degeneracy=w.degeneracy, hoppings=w.hoppings, lattice=w.lattice,
         ws_shifted=shifted if shifted is not None else np.zeros(0),
         ws_count=count if count is not None else np.zeros(0),
         edge=np.float64(w.edge if w.edge is not None else np.nan),
         fermi_energy=np.float64(scf.fermi_energy),
         labels=np.array(w.projection_labels), divisions=np.array(w.divisions))

ef_ev = scf.fermi_energy * RY_TO_EV
for label, offset in (("on-mesh", 0.0), ("off-mesh", 1.0 / 24.0)):
    g = (np.arange(6) / 6.0) + offset
    pts = np.array([[a, b, 0.0] for a in g for b in g])
    kp = KPoints.from_crystal(pts, np.ones(len(pts)), cell)
    t0 = time.perf_counter()
    bands = calc.get_bands(kp, nbnd=16)
    direct = np.asarray(bands.eigenvalues) * RY_TO_EV
    interp = w.bands_ev(pts)
    # band n against band n, over the bands below E_F + 2 eV + a margin, lowest num_wann
    nb = w.num_wann
    eta, eta_max = band_distance(direct[:, :nb], interp[:, :nb], ef_ev, nu=2.0, sigma=0.1)
    within = direct[:, :nb] <= ef_ev + 2.0
    dmax = np.max(np.abs(direct[:, :nb] - interp[:, :nb])[within])
    print(f"band distance [{label}]: eta_2 {eta * 1e3:.3f} meV eta_2^max {eta_max * 1e3:.3f} meV "
          f"max |dE| below E_F+2 {dmax * 1e3:.3f} meV ({time.perf_counter() - t0:.1f} s)", flush=True)
print(json.dumps({"scf_s": t_scf, "wannier_s": t_w}))
