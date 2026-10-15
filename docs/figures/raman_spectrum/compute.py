"""The Raman and infrared modes of silicon and AlAs at Gamma, and the LO mode of AlAs."""
from defuqumat import Calculator
from common import PSEUDO, save

si = Calculator.from_file("si.in", pseudo_dir=PSEUDO).get_vibrational_spectrum()

alas = Calculator.from_file("alas.in", pseudo_dir=PSEUDO)
to = alas.get_vibrational_spectrum(neutralize=True)
lo = alas.get_vibrational_spectrum(loto_direction=(1, 0, 0), neutralize=True)

save(si_freq=si.frequencies, si_raman=si.raman_activity, si_ir=si.infrared,
     alas_freq=to.frequencies, alas_raman=to.raman_activity, alas_ir=to.infrared,
     lo_freq=lo.frequencies, eps_inf=to.epsilon[0, 0], eps_0=to.static_permittivity[0, 0])
