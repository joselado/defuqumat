"""The orbital of the tip: graphene seen by an s, a pz and a pxy tip (PLAN.md P179)."""
import numpy as np
from defuqumat import Calculator
from defuqumat.units import BOHR_TO_ANGSTROM, RY_TO_EV
from common import PSEUDO, save

# one sheet at 0.5 of a 20 bohr cell: the substrate plane 2.4 bohr below it,
# the tip 4 bohr above it, the states continued from 3 bohr above it
c = 20.0
graphene = Calculator.from_keywords(
    lattice="hexagonal", a=4.6511 * BOHR_TO_ANGSTROM, c=c * BOHR_TO_ANGSTROM,
    atoms=[("C", (0, 0, 0.5)), ("C", (1 / 3, 2 / 3, 0.5))],
    pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir=PSEUDO,
    ecut=40 * RY_TO_EV, kmesh=(12, 12, 1), smearing=0.02 * RY_TO_EV, nbands=12,
    qe={"electrons": {"conv_thr": 1e-10, "mixing_beta": 0.5}})
graphene.get_scf()
states = graphene.get_tunnelling_states(grid=(6, 6, 1), nbnd=20, conv_thr=1e-10)
maps, notes = {}, {}
for tip in ("s", "pz", "pxy"):
    flow = graphene.get_vertical_transport(
        exit_height=0.38, height=0.5 + 4.0 / c, shape=(30, 30), broadening=0.02,
        states=states, tip_orbital=tip, match_height=0.5 + 3.0 / c, incoherent=False)
    maps[tip] = flow.image
    notes[tip] = flow.notes["tip"]
xy = flow.coordinates
# the ratio hollow / atom against the height, raw tail and continued
heights = np.array([3.5, 4.0, 5.0, 6.0, 8.0, 10.0])
raw, continued = [], []
for h in heights:
    z = 0.5 + h / c
    sites = np.array([[0, 0, z], [2 / 3, 1 / 3, z]])
    one = {}
    for match in (None, 0.5 + 3.0 / c):
        flow = graphene.get_vertical_transport(
            exit_height=0.38, tip=sites, broadening=0.02, states=states,
            tip_orbital="pxy", match_height=match, incoherent=False)
        one[match] = np.asarray(flow.values)
    raw.append(one[None][1] / one[None][0])
    continued.append(one[0.5 + 3.0 / c][1] / one[0.5 + 3.0 / c][0])
save(xy=xy, s=maps["s"], pz=maps["pz"], pxy=maps["pxy"], heights=heights,
     raw=np.array(raw), continued=np.array(continued),
     kappa=notes["pz"]["kappa"][0], vacuum=notes["pz"]["vacuum_condition"][0],
     basis=notes["pz"]["basis_residue"][0])
for h, r, k in zip(heights, raw, continued):
    print(f"{h:4.1f} bohr: pxy hollow/A raw {r:.4f}, continued {k:.4f}")
