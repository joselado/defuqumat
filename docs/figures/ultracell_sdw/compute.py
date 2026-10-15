"""A spin density wave over eight cells of silicon, and what a tip sees above it."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

calc = Calculator.from_file("si-mag.in", pseudo_dir=PSEUDO)
field = lambda x: 0.02 * np.cos(2 * np.pi * x[..., 0] / 8)    # Ry, x in unit cells
wave = calc.get_ultracell(supercell=(8, 1, 1), kgrid=(1, 2, 2), nbnd=32,
                          magnetic_field=field)

# STM images on a plane across the eight cells, for three tips
plane = dict(height=0.35, axis=2, shape=(96, 12))
images = {name: np.asarray(calc.get_ultracell_stm(spin=spin, **plane).values)
          for name, spin in (("charge", None), ("up", "up"), ("down", "down"))}

cells = np.arange(8) + 0.5
m = np.asarray(wave.magnetization)
save(cells=cells, moments=wave.cell_moments(), field=field(cells[:, None]),
     x=np.arange(m.shape[0]) * 8 / m.shape[0], profile=m.mean(axis=(1, 2)),
     iterations=wave.iterations, **images)
