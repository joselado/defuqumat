"""Vertical transport through graphene and bilayer graphene, against the STM image."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

# one sheet at 0.5 of c: the substrate plane below it, the tip plane above it
graphene = Calculator.from_file("graphene.in", pseudo_dir=PSEUDO)
one = graphene.get_vertical_transport(exit_height=0.38, height=0.62, shape=(30, 30),
                                      broadening=0.02, grid=(6, 6, 1), nbnd=20)
one_stm = graphene.get_stm(height=0.62, shape=(30, 30), width=0.02,
                           grid=(6, 6, 1), nbnd=20)

# two sheets at 0.325 and 0.675 of c
bilayer = Calculator.from_file("graphite.in", pseudo_dir=PSEUDO)
two = bilayer.get_vertical_transport(exit_height=0.14, height=0.86, shape=(30, 30),
                                     broadening=0.02, grid=(6, 6, 1), nbnd=24)
two_stm = bilayer.get_stm(height=0.86, shape=(30, 30), width=0.02,
                          grid=(6, 6, 1), nbnd=24)

save(xy=two.coordinates, transport=two.image, stm=two_stm.values,
     incoherent=two.incoherent,
     corr_one=np.corrcoef(one.image.ravel(), one_stm.values.ravel())[0, 1],
     corr_two=np.corrcoef(two.image.ravel(), two_stm.values.ravel())[0, 1])
