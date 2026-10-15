"""The phonon branches of silicon along Gamma-X-K-Gamma-L, one q at a time."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

calc = Calculator.from_file("si.in", pseudo_dir=PSEUDO)

# q in units of 2 pi / a. X = (1, 1, 0) is the same point as X = (1, 0, 0),
# so the second segment goes from X to K along (1, 1, 0) and back to Gamma.
segments = [
    [(0, 0, 0), (0.25, 0, 0), (0.5, 0, 0), (0.75, 0, 0), (1, 0, 0)],
    [(1, 1, 0), (0.75, 0.75, 0), (0.5, 0.5, 0), (0.25, 0.25, 0), (0, 0, 0)],
    [(0, 0, 0), (0.125, 0.125, 0.125), (0.25, 0.25, 0.25), (0.375, 0.375, 0.375),
     (0.5, 0.5, 0.5)],
]
done = {}
x, frequencies, start = [], [], 0.0
for points in segments:
    for q in points:
        if q not in done:
            done[q] = calc.get_phonons_at_q(q=q, q_cartesian=True).frequencies
        x.append(start + np.linalg.norm(np.subtract(q, points[0])))
        frequencies.append(done[q])
    start = x[-1]

save(x=np.array(x), frequencies=np.array(frequencies),
     ticks=np.array([0.0, x[4], x[6], x[9], x[-1]]))
