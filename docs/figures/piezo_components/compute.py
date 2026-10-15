"""The piezoelectric tensor of AlAs and silicon, and e_14 of AlAs on a ladder of k-meshes."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

alas = Calculator.from_file("alas.in", pseudo_dir=PSEUDO)
silicon = Calculator.from_file("si.in", pseudo_dir=PSEUDO)
alas_e = alas.get_piezoelectric_tensor(kmesh_warning=False).voigt     # (3, 6), C/m^2
silicon_e = silicon.get_piezoelectric_tensor(kmesh_warning=False).voigt

ladder = alas.get_piezoelectric_kmesh_ladder(meshes=(4, 6, 8, 10, 12, 14), method="zstar_eu",
                                             verbose=True)
save(alas=alas_e, silicon=silicon_e, meshes=np.array(ladder.meshes),
     nk=np.array(ladder.nk), e14=np.array(ladder.e14), steps=np.array(ladder.steps),
     seconds=np.array(ladder.seconds))
