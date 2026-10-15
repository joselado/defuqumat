"""E(q) of the hydrogen chain, from the energies and from the slope dE/dq."""
import numpy as np
from defuqumat import Calculator
from common import PSEUDO, save

chain = Calculator.from_file("h-chain.in", pseudo_dir=PSEUDO)
q = np.linspace(0.0, 0.5, 21)
scan = chain.get_spiral_scan([[0.0, 0.0, x] for x in q], gradients=True,
                             max_iterations=200)

save(q=q, direct=scan.relative, integrated=scan.integrated,
     moment=np.linalg.norm(scan.moments, axis=1), converged=np.asarray(scan.converged))
