"""The magnetizability of diamond through Calculator.get_magnetizability, end to end.

Lu's 3x1x1 conventional cell (a = 3.56 A, 24 atoms), C.pbe-hgh at 60 Ry, kmesh (2, 6, 6),
nbnd 52 (48 occupied), the three SCFs at -1, 0, +1 flux quantum in one call. route-p.md item 4
ran the three as separate probe processes (diamond-p0/p1/m1.log) and took chi from their
energies; this is the wrapper run once. Usage: python mag_chi_facade.py
"""
import resource
import time

from defuqumat import Calculator
from defuqumat.units import RY_TO_EV

a = 3.56
base = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
sites = base + [(x + .25, y + .25, z + .25) for x, y, z in base]
diamond = Calculator.from_keywords(
    lattice=[[3 * a, 0, 0], [0, a, 0], [0, 0, a]],
    atoms=[("C", ((x + j) / 3, y, z)) for j in range(3) for x, y, z in sites],
    pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=60 * RY_TO_EV, kmesh=(2, 6, 6))
t0 = time.perf_counter()
diamond.get_scf(conv_thr=1e-11)
t_scf = time.perf_counter() - t0
chi = diamond.get_magnetizability(kmesh=(2, 6, 6), nbnd=52, conv_thr=1e-8)
wall = time.perf_counter() - t0
print(f"zero-field SCF {t_scf:.0f} s; field step {chi.field_tesla:.1f} T")
print(f"energies (Ry, the 24-atom cell): {chi.energies}")
print(f"E(+) - E(-) = {chi.asymmetry:.3e} Ry; iterations "
      f"{ {p: r.iterations for p, r in chi.results.items()} }")
print(f"chi = {chi.chi_per_unit:.3f} x 1e-6 cm^3/mol per C2, {chi.chi_per_atom:.3f} per C atom "
      f"(Lu, LDA: -11.16 per C2); {wall:.0f} s in all, peak RSS "
      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
