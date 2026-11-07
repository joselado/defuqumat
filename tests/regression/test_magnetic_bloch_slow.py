"""Route P step 1 at the acceptance sizes (slow; not a push gate).

The numbers of ``plans/orbital-field/route-p.md`` as tests, on the graphene cell of the shared
zero-field state (a = 2.46 A, c = 15 A, 50 Ry, box 12 x 12 x 64 per cell):

- the free Landau problem at q = 10: (2n + 1) B_ph + (2 pi m/c)^2 to 1e-8 Ry, flat over a 4 x 4
  magnetic mesh to 1e-10 Ry, Chern number 1 for each of the three lowest m = 0 levels
  (about 8 minutes on one D22 core);
- p = 2, q = 10: two states per level;
- the strong translational symmetry at p/q = 1/3, kmesh (2, 6): spectra at kappa2 and kappa2 -
  1/3 equal to 1e-10 Ry (about 2.5 minutes);
- the full mesh against q copies of the representatives at q = 3, kmesh (2, 6), to 1e-9 Ry.
"""

import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.magnetic.conventions import MagneticCell
from defuqumat.magnetic.landau import free_landau_levels
from defuqumat.units import RY_TO_EV
from defuqumat.workflows.magnetic_bands import run_magnetic_bands

pytestmark = [pytest.mark.regression, pytest.mark.slow]

SCF = "plans/orbital-field/graphene-scf.npz"


@pytest.fixture(scope="module")
def graphene():
    return Calculator.from_keywords(
        lattice="hexagonal", a=2.46, c=15.0,
        atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
        pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
        ecut=50 * RY_TO_EV, kmesh=(1, 1, 1))


@pytest.fixture(scope="module")
def scf_path():
    from pathlib import Path

    return Path(__file__).resolve().parents[2] / SCF


@pytest.mark.parametrize("p", [1, 2])
def test_the_free_landau_problem_at_q10(graphene, p):
    system = graphene.system
    bands = run_magnetic_bands(system, graphene.pseudos, None, flux=(p, 10), kmesh=(4, 4),
                               nbnd=12, free=True, chern=(p == 1))
    levels, _, labels = free_landau_levels(MagneticCell.build(system.cell, p, 10), nmax=12,
                                           g3max=3)
    reference = np.repeat(levels, p)[:12]
    assert abs(bands.b_ph - 0.03357231492638168 * p) < 1e-12
    assert np.max(np.abs(bands.eigenvalues - reference[None])) < 1e-8
    assert np.max(bands.eigenvalues.max(0) - bands.eigenvalues.min(0)) < 1e-10
    assert set(bands.solver) == {"davidson"}
    if p == 1:
        label = np.repeat(labels, p, axis=0)[:12]
        singles = [(first, c) for first, last, c, _ in bands.chern
                   if first == last and label[first][1] == 0]
        assert [first for first, _ in singles][:3] == [0, 3, 6]
        assert all(abs(c - 1.0) < 1e-9 for _, c in singles[:3])
    else:
        e = bands.eigenvalues[:, :12].reshape(len(bands.kpoints), -1, 2)
        assert np.max(np.abs(e[..., 1] - e[..., 0])) < 1e-9


def test_the_strong_translational_symmetry_at_one_third(graphene, scf_path):
    q = 3
    bands = run_magnetic_bands(graphene.system, graphene.pseudos, scf_path, flux=(1, q),
                               kmesh=(2, 6), nbnd=4 * q + 8)
    e = bands.eigenvalues.reshape(2, 6, -1)
    worst = max(np.abs(e[i, j] - e[i, (j - 2) % 6]).max() for i in range(2) for j in range(6))
    assert worst < 1e-10
    full = np.sort(bands.eigenvalues.reshape(-1))
    representatives = np.sort(np.repeat(e[:, :2, :].reshape(-1), q))
    assert np.max(np.abs(full - representatives)) < 1e-9
