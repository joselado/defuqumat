"""The snippets of docs/guide/magnetic-field.md, run as they are written there (from the repository root).

The diamond snippet of the page is plans/orbital-field/mag_chi_facade.py (a big job).
"""
import numpy as np
from defuqumat import Calculator
from defuqumat.units import RY_TO_EV

# --- graphene at 10 T, Peierls on the p_z Wannier model
graphene = Calculator.from_keywords(
    lattice="hexagonal", a=2.46, c=15.0, atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
    pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=50 * RY_TO_EV, kmesh=(12, 12, 1), smearing=0.01 * RY_TO_EV, smearing_type="mv")
ef = graphene.get_scf().fermi_energy * RY_TO_EV
pz = graphene.get_wannier(2, ["C:pz"], num_bands=16, dis_froz=(ef - 3, ef + 3), symmetry=False,
                          engine_options=dict(num_iter=1000, dis_num_iter=1000))
ll = graphene.get_landau_levels(B=10.0, wannier=pz, nlevels=44)
row = ll.dirac["per_k"][0]
print(f"flux {ll.flux}, B = {ll.field_tesla:.3f} T, dimension {ll.dim}")
print(f"v_F = {ll.dirac['v_F']:.4e} m/s, E_1 Dirac = {ll.dirac['E1_dirac_ev'] * 1e3:.2f} meV")
for n in (1, 2, 3, 10, -1, -2, -3, -10):
    print(f"n = {n:3d}: E_n - E_D = {row['energies_ev'][n] * 1e3:8.2f} meV, ratio {row['ratios'][n]:.5f}")
print(f"zero level {row['zero_level_ev'] * 1e6:.1f} ueV, {row['zero_multiplicity']} states, "
      f"site traces {np.round(row['zero_site_traces'], 6)}")
print(f"Hermiticity {ll.hermiticity.max():.1e} Ry, plaquette {ll.plaquette}")

# --- a ribbon of 330 cells at the same field, the orbit placed at its middle
k2 = ll.dirac["K"][1] + 165 / ll.flux[1]
ribbon = graphene.get_landau_levels(B=10.0, wannier=pz, nlevels=20, geometry="ribbon", width=330,
                                    kappa=((0.0, k2),))
print(f"ribbon of {ribbon.dim // 2} cells at kappa2 = {k2:.5f}: levels (meV from E_D)",
      np.round(np.sort(ribbon.levels[0] - ll.dirac["E_D"]) * RY_TO_EV * 1e3, 2)[7:13])

# --- the free Landau problem on the plane-wave magnetic cell at p/q = 1/10
free = graphene.get_magnetic_bands(flux=(1, 10), kmesh=(1, 1), nbnd=6, free=True, local_only=True)
print(f"B = {free.field_tesla:.1f} T, B_ph = {free.b_ph:.8f} Ry, ndim {free.ndim}")
print("levels / B_ph:", np.round(free.eigenvalues[0] / free.b_ph, 8))
