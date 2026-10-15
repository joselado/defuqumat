"""A constant-height STM image of graphite, and its decay into the vacuum."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

calc = Calculator.from_file("graphite.in", pseudo_dir=PSEUDO)

# the tip plane at 0.85 of c, which is 3.5 bohr above the top layer (0.675)
image = calc.get_stm(height=0.85, shape=(60, 60))

# a vertical cut from the top layer (0.675) to the middle of the vacuum (1.0),
# through the atom A (0, 0), the hexagon centre (1/3, 2/3) and the atom B (2/3, 1/3);
# the plane is given by three corners in crystal coordinates
cut = calc.get_stm(plane=((0, 0, 0.675), (1, -1, 0.675), (0, 0, 1.0)), shape=(3, 130))
c = float(np.asarray(calc.system.cell.at)[2, 2])

save(xy=image.coordinates, image=image.values, integral=image.integral,
     z=np.arange(130) / 130 * 0.325 * c, decay=cut.values)
