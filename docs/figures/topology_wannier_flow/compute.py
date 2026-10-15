"""The flow of the Wannier charge centres of a Bi(111) bilayer."""
from defuqumat import Calculator
from common import PSEUDO, save

calc = Calculator.from_file("bi-bilayer.in", pseudo_dir=PSEUDO)
flow = calc.get_z2(nloop=16, npump=9)            # Wilson loops, the default
print("z2 =", flow.z2, " gap_step = %.3f" % flow.gap_step, flush=True)

save(pump=flow.pump, centers=flow.centers_fractional,
     gap=flow.gap_center, z2=flow.z2, gap_step=flow.gap_step)
