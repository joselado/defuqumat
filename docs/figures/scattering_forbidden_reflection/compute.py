"""The X-ray structure factors of silicon, and X-ray against neutron on an antiferromagnet."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

silicon = Calculator.from_file("si.in", pseudo_dir=PSEUDO)
si = silicon.get_structure_factors(hmax=6.0)

# two hydrogen atoms with opposite moments: keep the primitive (0 0 l) along the chain
chain = Calculator.from_file("h-afm.in", pseudo_dir=PSEUDO)
afm = chain.get_structure_factors(hmax=2.0, transform=None, reduce=False)
miller = np.asarray(afm.vectors.miller)
rows = [r for r in range(len(afm)) if miller[r, 0] == 0 and miller[r, 1] == 0 and miller[r, 2] >= 0]
rows = sorted(rows, key=lambda r: miller[r, 2])

save(hkl=np.asarray(si.vectors.indices), length=np.asarray(si.vectors.length),
     F=np.abs(si.charge), multiplicity=np.asarray(si.vectors.multiplicity),
     l=miller[rows, 2], xray=np.abs(afm.charge[rows]),
     neutron=np.abs(afm.magnetization[rows, 0]))
