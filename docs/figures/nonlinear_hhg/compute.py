"""High harmonics of silicon and AlAs under the same 800 nm pulse."""
from defuqumat import Calculator
from defuqumat.realtime import Sin2
from common import PSEUDO, save

# 3e12 W/cm^2, 1.55 eV photons, eight cycles, polarized along [111]
pulse = Sin2.from_intensity(3e12, 1.55, 8, polarization=(1, 1, 1))

out = {}
for name in ("si", "alas"):
    calc = Calculator.from_file(name + ".in", pseudo_dir=PSEUDO)
    hhg = calc.get_hhg(pulse, dt=0.2, nbnd=8)
    out[name + "_orders"] = hhg.orders
    out[name + "_total"] = hhg.total / hhg.harmonic(1)       # over the fundamental
    out[name + "_peaks"] = hhg.harmonics(30) / hhg.harmonic(1)
    out[name + "_cutoff"] = hhg.cutoff(floor=1e-6)
    out[name + "_excited"] = hhg.realtime.excited
save(**out)
