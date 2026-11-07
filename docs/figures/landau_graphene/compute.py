"""Graphene at 10 T: the Landau levels of the p_z Wannier model against the Dirac formula."""
import numpy as np
from defuqumat import Calculator
from defuqumat.magnetic.landau import dirac_levels
from defuqumat.units import RY_TO_EV
from common import PSEUDO, save

graphene = Calculator.from_keywords(
    lattice="hexagonal", a=2.46, c=15.0, atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
    pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir=PSEUDO, ecut=50 * RY_TO_EV, kmesh=(12, 12, 1),
    smearing=0.01 * RY_TO_EV, smearing_type="mv", announce=False)
scf = graphene.get_scf()
ef = scf.fermi_energy * RY_TO_EV
pz = graphene.get_wannier(2, ["C:pz"], num_bands=16, divisions=(12, 12, 1),
                          dis_froz=(ef - 3.0, ef + 3.0), symmetry=False,
                          engine_options=dict(num_iter=1000, dis_num_iter=1000))
ll = graphene.get_landau_levels(B=10.0, wannier=pz, nlevels=44, kappa=((0.11, 0.23),))
d = ll.dirac
row = d["per_k"][0]
n = np.array(sorted(row["energies_ev"]))
save(n=n, energy=np.array([row["energies_ev"][k] for k in n]),
     levels=(ll.levels[0] - d["E_D"]) * RY_TO_EV,
     dirac=dirac_levels(ll.field_tesla, d["v_F"], 10), field=ll.field_tesla,
     flux=np.array(ll.flux), v_F=d["v_F"], zero_level=row["zero_level_ev"],
     zero_multiplicity=row["zero_multiplicity"], zero_traces=np.array(row["zero_site_traces"]))
print(f"flux {ll.flux}, B = {ll.field_tesla:.3f} T, v_F = {d['v_F']:.4e} m/s, "
      f"E_1 = {row['energies_ev'][1] * 1e3:.2f} meV against Dirac {d['E1_dirac_ev'] * 1e3:.2f} meV, "
      f"zero level {row['zero_level_ev'] * 1e6:.1f} ueV x {row['zero_multiplicity']}")
