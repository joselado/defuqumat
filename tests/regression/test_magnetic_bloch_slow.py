"""Route P, steps 1 and 2, at the acceptance sizes (slow; not a push gate).

The numbers of ``plans/orbital-field/route-p.md`` as tests, on the graphene cell of the shared
zero-field state (a = 2.46 A, c = 15 A, 50 Ry, box 12 x 12 x 64 per cell):

- the free Landau problem at q = 10: (2n + 1) B_ph + (2 pi m/c)^2 to 1e-8 Ry, flat over a 4 x 4
  magnetic mesh to 1e-10 Ry, Chern number 1 for each of the three lowest m = 0 levels
  (about 8 minutes on one D22 core);
- p = 2, q = 11: two states per level (2/10 is not a reduced flux);
- the strong translational symmetry at p/q = 1/3, kmesh (2, 6): spectra at kappa2 and kappa2 -
  1/3 equal to 1e-10 Ry (about 2.5 minutes);
- the full mesh against q copies of the representatives at q = 3, kmesh (2, 6), to 1e-9 Ry;
- step 2, the full Hamiltonian with the GIPAW-dressed projectors: the Chern numbers at
  n_Phi = 1 on a 12 x 12 mesh (1, 0, and 3 for the third and fourth together, Lu; about 6
  minutes), the band shift of a rigid translation by a1/2 (the sorted set over 6 x 6 to
  1e-8 Ry, about 5 minutes), the SCF by the fold at 1/3 against the closed mesh (about 14
  minutes).
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


@pytest.mark.parametrize("p,q", [(1, 10), (2, 11)])
def test_the_free_landau_problem_at_q10(graphene, p, q):
    # 2/10 is not a reduced flux (the conventions refuse it), so two states per level is 2/11
    system = graphene.system
    # 2/11 took 265 to 361 Davidson steps without the buffer bands (2/5 with them: 101 to 118);
    # 2/11 with the buffer was not measured, so the pairs keep the larger budget here
    bands = run_magnetic_bands(system, graphene.pseudos, None, flux=(p, q), kmesh=(4, 4),
                               nbnd=12, free=True, chern=(p == 1),
                               max_iterations=200 if p == 1 else 400)
    levels, _, labels = free_landau_levels(MagneticCell.build(system.cell, p, q), nmax=12,
                                           g3max=3)
    reference = np.repeat(levels, p)[:12]
    assert abs(bands.b_ph - 0.03357231492638168 * p * 10 / q) < 1e-12
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


def _graphene_smeared():
    return Calculator.from_keywords(
        lattice="hexagonal", a=2.46, c=15.0,
        atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
        pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
        ecut=50 * RY_TO_EV, kmesh=(12, 12, 1), smearing=0.01 * RY_TO_EV, smearing_type="mv")


def test_the_chern_numbers_at_one_flux_quantum(graphene, scf_path):
    # measured on D22 (route-p.md, item 2): C = 1, 0, -3, +6 for bands 0 to 3, the pair 3
    from defuqumat.workflows.magnetic_bands import _chern_numbers

    bands = run_magnetic_bands(graphene.system, graphene.pseudos, scf_path, flux=(1, 1),
                               kmesh=(12, 12), nbnd=8, local_only=False, keep_states=True)
    out = {(a, b): c for a, b, c, _ in _chern_numbers(bands.states, (12, 12), bands.box,
                                                       [(0, 0), (1, 1), (2, 3)])}
    assert abs(out[(0, 0)] - 1.0) < 1e-9
    assert abs(out[(1, 1)]) < 1e-9
    assert abs(out[(2, 3)] - 3.0) < 1e-9


def test_the_band_shift_of_a_rigid_translation(scf_path):
    from defuqumat.magnetic.fold import magnetic_occupations
    from defuqumat.workflows.magnetic_bands import local_potential_of

    calc = _graphene_smeared()
    a1 = np.asarray(calc.system.cell.at)[0]
    moved = calc.with_positions(np.asarray(calc.system.structure.positions) + 0.5 * a1)
    potential = local_potential_of(scf_path)
    rolled = np.roll(potential, potential.shape[0] // 2, axis=0)
    runs = []
    for c, v in ((calc, potential), (moved, rolled)):
        b = run_magnetic_bands(c.system, c.pseudos, {"potential": v[None]}, flux=(1, 1),
                               kmesh=(6, 6), nbnd=12, local_only=False)
        wg, _, _ = magnetic_occupations(b.eigenvalues, np.full(36, 2.0 / 36), 8.0, calc.system)
        runs.append((b, float(np.sum(wg * b.eigenvalues))))
    (b0, e0), (b1, e1) = runs
    assert np.max(np.abs(b0.eigenvalues - b1.eigenvalues)) > 0.1
    assert np.max(np.abs(b0.sorted_levels - b1.sorted_levels)) < 1e-8
    assert abs(e0 - e1) < 1e-8


def test_the_fold_at_one_third(scf_path):
    from defuqumat.magnetic.fold import fold_check, run_magnetic_scf

    calc = _graphene_smeared()
    result = run_magnetic_scf(calc.system, calc.pseudos, scf_path, flux=(1, 3), kmesh=(2, 6),
                              nbnd=20, conv_thr=1e-8, verbose=False)
    assert result.converged and result.iterations <= 12
    density, levels, _ = fold_check(result, calc.system, calc.pseudos)
    assert density < 1e-10 and levels < 1e-9
