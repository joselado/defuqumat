"""The nesting function of a half-filled hydrogen chain and of fcc aluminium."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

chain = Calculator.from_file("h-chain.in", pseudo_dir=PSEUDO)
nest = chain.get_nesting(grid=(1, 1, 60))
q3, n3 = nest.along(2)
where, peak = nest.peak()

aluminium = Calculator.from_file("al.in", pseudo_dir=PSEUDO)
metal = aluminium.get_nesting(grid=(12, 12, 12))
cell = aluminium.system.cell
q = metal.qpoints - np.rint(metal.qpoints)          # into the first zone
length = np.linalg.norm(cell.k_to_cartesian(q), axis=1) * cell.tpiba
al_where, al_peak = metal.peak()

save(q3=q3, chain=n3 / nest.nesting[0], chain_peak=where, chain_ratio=peak / nest.nesting[0],
     chain_dos=nest.fermi_dos, q=length, metal=metal.nesting / metal.nesting[0],
     metal_peak=al_where, metal_ratio=al_peak / metal.nesting[0], metal_dos=metal.fermi_dos)
