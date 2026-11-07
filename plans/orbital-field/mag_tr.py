"""Which term breaks time reversal on the small box: free, local-only (two forms), full."""
import numpy as np

from defuqumat import Calculator
from defuqumat.magnetic.blochgrid import MagneticGridHamiltonian, dense_matrix
from defuqumat.units import RY_TO_EV
from defuqumat.workflows.magnetic_bands import local_potential_of

calc = Calculator.from_keywords(
    lattice="hexagonal", a=2.46, c=15.0, atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
    pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo", ecut=50 * RY_TO_EV, kmesh=(1, 1, 1))
pot = local_potential_of("plans/orbital-field/graphene-scf.npz")
cell = calc.system.cell
for grid in ((5, 5, 9), (4, 4, 8)):
    for kappa in ((1.0 / 48.0, 0.25, 0.0), (0.31, 0.23, 0.0), (0.37, 0.29, 0.0), (0.0, 0.0, 0.0)):
        for label, options in (("free", dict(potential=None, local="filtered")),
                               ("filtered", dict(potential=pot, local="filtered")),
                               ("two-grid", dict(potential=pot, local="two-grid")),
                               ("full", dict(potential=pot, local="two-grid", structure=calc.system.structure,
                                             pseudos=calc.pseudos))):
            v = options.pop("potential")
            spectra = []
            for p, k in ((1, kappa), (-1, tuple(-x for x in kappa))):
                h = MagneticGridHamiltonian.build(cell, p, 2, 50.0, [k], v, grid=grid, **options)
                m = dense_matrix(h, 0, chunk=32)
                spectra.append(np.linalg.eigvalsh(m))
                if p == 1:
                    m_plus = m
                else:
                    m_minus = m
            d = np.abs(spectra[0] - spectra[1])
            conj = np.max(np.abs(m_plus - m_minus.conj())) / np.max(np.abs(m_plus))
            print(f"grid {grid} kappa {kappa} {label:9s}: spectra +B,k vs -B,-k max {d.max():.3e} "
                  f"(lowest 4 {d[:4].max():.3e}); |H(+B,k) - conj H(-B,-k)| / max {conj:.3e}", flush=True)
