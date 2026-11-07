"""Route P step 1: ``pi^2 + V_loc`` on the magnetic box (Lu et al., PRL 135, 196401 (2025)).

What is checked, on boxes of at most 256 points so every test is seconds:

- the gauge the operator uses is the conventions' ``a(r)``, and its twist is ``exp(-i a(P).r)``
  for the enlarged vector ``P = q a1``;
- each ``pi_a`` against the analytic derivative of a smooth function obeying the twist;
- ``apply`` against an independent Kronecker-product matrix to round-off, and Hermitian;
- the free Landau ladder ``(2n + 1) B_ph + (2 pi m/c)^2`` from the dense matrix, ``p`` states
  per level, and the magnetic translation ``kappa2 -> kappa2 - p/q`` exact on the grid;
- the filter of the potential keeps it real and keeps its mean;
- Davidson through the hook converges to the dense answer, the hook is live (the identity
  preconditioner takes a different number of steps), the Chern number of the lowest Landau
  level, the dense fallback and its refusal above the limit;
- the refusals: the nonlocal term, a field not along a3, a memory estimate, the regimes.
"""

import math
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.magnetic import blochgrid
from defuqumat.magnetic.blochgrid import (
    MagneticGridHamiltonian, dense_matrix, estimate_bytes, filter_to_box, grid_positions,
    landau_problem, reference_matrix,
)
from defuqumat.magnetic.conventions import MagneticCell
from defuqumat.magnetic.landau import free_landau_levels
from defuqumat.units import RY_TO_EV, TPI
from defuqumat.workflows import magnetic_bands
from defuqumat.workflows.magnetic_bands import run_magnetic_bands

pytestmark = pytest.mark.unit

SMALL = (4, 4, 8)
SCF = "plans/orbital-field/graphene-scf.npz"


@pytest.fixture(scope="module")
def graphene():
    return Calculator.from_keywords(
        lattice="hexagonal", a=2.46, c=15.0,
        atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
        pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
        ecut=50 * RY_TO_EV, kmesh=(1, 1, 1))


@pytest.fixture(scope="module")
def potential():
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / SCF
    with np.load(path) as data:
        return np.asarray(data["potential"])[0]


def test_the_gauge_and_the_twist_are_the_conventions(graphene):
    cell = graphene.system.cell
    h = landau_problem(cell, 2, 3, SMALL, [[0.0, 0.0, 0.0]])
    mcell = MagneticCell.build(cell, 2, 3)
    xi = grid_positions(h.box)
    r = xi @ np.asarray(mcell.magnetic_lattice)
    covariant = np.asarray(mcell.a(r)) @ np.asarray(mcell.magnetic_lattice).T
    assert np.max(np.abs(covariant - h.vector_potential_components())) < 1e-12
    P = np.asarray(mcell.magnetic_lattice)[0]
    twist = np.asarray(jnp.sum(mcell.a(P) * r, axis=-1))
    assert np.max(np.abs(twist - TPI * 2 * xi[:, 1])) < 1e-12
    assert abs(float(mcell.b_ph) - TPI * 2 / 3 / float(mcell.area)) < 1e-15


def _twisted_function(box, p, sigma=0.08, shift=0.3):
    """``u = sum_m exp(2 pi i m xi2) G(xi1 + m/p)``: periodic in xi2, twisted in xi1, and its derivatives."""
    xi = grid_positions(box).reshape(box + (3,))
    x1, x2 = xi[..., 0], xi[..., 1]
    u = np.zeros(box, complex)
    d1 = np.zeros(box, complex)
    d2 = np.zeros(box, complex)
    for m in range(-12 * p, 12 * p + 1):
        y = x1 + m / p - shift
        g = np.exp(-y ** 2 / (2 * sigma ** 2))
        wave = np.exp(1j * TPI * m * x2)
        u += wave * g
        d1 += wave * g * (-y / sigma ** 2)
        d2 += 1j * TPI * m * wave * g
    return u, d1, d2


@pytest.mark.parametrize("p", [1, 2])
def test_each_momentum_against_the_analytic_derivative_of_a_twisted_function(graphene, p):
    cell = graphene.system.cell
    box_cell = (64, 16, 2)
    h = landau_problem(cell, p, 1, box_cell, [[0.0, 0.0, 0.0]])
    u, d1, d2 = _twisted_function(h.box, p)
    # the twist itself: u(xi1 + 1) = exp(-i 2 pi p xi2) u(xi1)
    xi = grid_positions(h.box).reshape(h.box + (3,))
    shifted, _, _ = _twisted_function(h.box, p, shift=0.3 - 1.0)
    assert np.max(np.abs(shifted - np.exp(-1j * TPI * p * xi[..., 1]) * u)) < 1e-12
    k = jnp.zeros(3)
    pi1 = np.asarray(h.momentum(jnp.asarray(u)[None], k, 0))[0]
    pi2 = np.asarray(h.momentum(jnp.asarray(u)[None], k, 1))[0]
    scale = np.max(np.abs(d1))
    assert np.max(np.abs(pi1 - (-1j) * d1)) / scale < 1e-10
    assert np.max(np.abs(pi2 - ((-1j) * d2 + TPI * p * xi[..., 0] * u))) / scale < 1e-10


@pytest.mark.parametrize("kappa", [(0.0, 0.0, 0.0), (0.37, 0.29, 0.0)])
def test_apply_is_the_kronecker_reference_and_hermitian(graphene, potential, kappa):
    h = MagneticGridHamiltonian.build(graphene.system.cell, 1, 2, 50.0, [kappa], potential,
                                      grid=SMALL)
    built = dense_matrix(h, 0)
    reference = reference_matrix(h, 0)
    assert np.max(np.abs(built - reference)) / np.max(np.abs(reference)) < 1e-13
    assert np.max(np.abs(built - built.conj().T)) / np.max(np.abs(built)) < 1e-13
    rng = np.random.default_rng(3)
    v = rng.normal(size=(2, h.ndim)) + 1j * rng.normal(size=(2, h.ndim))
    hv = np.asarray(h.apply(jnp.asarray(v), 0))
    lhs, rhs = np.vdot(v[0], hv[1]), np.vdot(hv[0], v[1])
    assert abs(lhs - rhs) / (np.linalg.norm(v[0]) * np.linalg.norm(hv[1])) < 1e-12


@pytest.mark.parametrize("p,q", [(1, 2), (2, 3)])
def test_the_free_landau_ladder_from_the_dense_matrix(graphene, p, q):
    cell = graphene.system.cell
    h = landau_problem(cell, p, q, SMALL, [[0.0, 0.0, 0.0], [0.4, 0.15, 0.0]])
    levels, _, _ = free_landau_levels(MagneticCell.build(cell, p, q), nmax=6, g3max=2)
    reference = np.repeat(levels, p)[:4 * p]
    for ik in range(2):
        values = np.linalg.eigvalsh(dense_matrix(h, ik))[:4 * p]
        # a 4 x 4 x 8 box has about two points per magnetic length: its error is a
        # discretisation, measured on D22 (plans/orbital-field/route-p.md), not round-off
        assert np.max(np.abs(values - reference)) < 2e-5 * reference[0] / 0.168
        # p states per level: the members of a level agree to round-off
        assert np.max(np.abs(values.reshape(-1, p) - values.reshape(-1, p)[:, :1])) < 1e-10


@pytest.mark.parametrize("p,q", [(1, 3), (2, 3)])
def test_the_magnetic_translation_moves_kappa2_by_p_over_q_exactly(graphene, potential, p, q):
    kappa2 = 0.21
    shifted = (kappa2 - p / q) % 1.0
    h = MagneticGridHamiltonian.build(graphene.system.cell, p, q, 50.0,
                                      [[0.13, kappa2, 0.0], [0.13, shifted, 0.0]], potential,
                                      grid=(4, 4, 4))
    a = np.linalg.eigvalsh(dense_matrix(h, 0))
    b = np.linalg.eigvalsh(dense_matrix(h, 1))
    assert np.max(np.abs(a - b)) < 1e-10
    # and a shift that is not a magnetic translation does move the spectrum
    other = MagneticGridHamiltonian.build(graphene.system.cell, p, q, 50.0,
                                          [[0.13, (kappa2 - 0.5 / q) % 1.0, 0.0]], potential,
                                          grid=(4, 4, 4))
    assert np.max(np.abs(a - np.linalg.eigvalsh(dense_matrix(other, 0)))) > 1e-4


def test_the_filter_keeps_the_potential_real_and_its_mean(potential):
    box = filter_to_box(jnp.asarray(potential), (12, 12, 64))
    assert box.shape == (12, 12, 64) and not jnp.iscomplexobj(box)
    assert abs(float(jnp.mean(box)) - float(np.mean(potential))) < 1e-12
    same = filter_to_box(jnp.asarray(potential), potential.shape)
    assert np.max(np.abs(np.asarray(same) - potential)) < 1e-12
    # a Fourier interpolation up and back is the identity on the band-limited field
    up = filter_to_box(box, (30, 30, 160))
    assert np.max(np.abs(np.asarray(filter_to_box(up, (12, 12, 64))) - np.asarray(box))) < 1e-12


def test_estimate_bytes_is_the_designs_sizing():
    # notes/ORBITAL-FIELD-NEXT.md section 12 item 5: q = 30, nbnd = 1.2 x 4q, david = 2, 5.5 GB
    q, nbnd = 30, 144
    ndim = q * 12 * 12 * 64
    davidson = estimate_bytes(q, nbnd, 2) - 6 * ndim * 16
    assert abs(davidson / 1e9 - 5.45) < 0.01
    assert estimate_bytes(q, nbnd, 2, kept=10) - estimate_bytes(q, nbnd, 2) == 10 * ndim * 16


@pytest.fixture(scope="module")
def landau_runs(graphene):
    system, pseudos = graphene.system, graphene.pseudos
    out = {}
    for precond in ("two-gauge", "identity"):
        out[precond] = run_magnetic_bands(system, pseudos, None, flux=(1, 2), kmesh=(3, 3),
                                          nbnd=4, free=True, chern=(precond == "two-gauge"),
                                          preconditioner=precond, grid=SMALL, david=4,
                                          ethr=1e-11, max_iterations=400)
    return out


def test_davidson_through_the_hook_is_the_dense_answer_and_the_hook_is_live(graphene, landau_runs):
    run = landau_runs["two-gauge"]
    h = landau_problem(graphene.system.cell, 1, 2, SMALL, run.kpoints)
    for ik in range(len(run.kpoints)):
        dense = np.linalg.eigvalsh(dense_matrix(h, ik))[:4]
        assert np.max(np.abs(run.eigenvalues[ik] - dense)) < 1e-9
    assert set(run.solver) == {"davidson"}
    other = landau_runs["identity"]
    assert np.max(np.abs(other.eigenvalues - run.eigenvalues)) < 1e-9
    assert not np.array_equal(other.steps, run.steps)


def test_the_lowest_landau_level_carries_one_chern_number(landau_runs):
    groups = landau_runs["two-gauge"].chern
    first, last, chern, max_flux = groups[0]
    assert (first, last) == (0, 0)
    assert abs(abs(chern) - 1.0) < 1e-9
    assert max_flux < math.pi / 2


def test_an_unconverged_solve_falls_back_to_dense_and_is_refused_above_the_limit(graphene, monkeypatch):
    system, pseudos = graphene.system, graphene.pseudos
    args = dict(flux=(1, 2), kmesh=(1, 1), nbnd=3, free=True, grid=SMALL, david=2, max_iterations=1)
    run = run_magnetic_bands(system, pseudos, None, **args)
    assert run.solver == ("dense",)
    h = landau_problem(system.cell, 1, 2, SMALL, run.kpoints)
    assert np.max(np.abs(run.eigenvalues[0] - np.linalg.eigvalsh(dense_matrix(h, 0))[:3])) < 1e-12
    monkeypatch.setattr(magnetic_bands, "DENSE_LIMIT", 100)
    with pytest.raises(RuntimeError, match="unsettled"):
        run_magnetic_bands(system, pseudos, None, **args)


def test_the_refusals(graphene):
    system, pseudos = graphene.system, graphene.pseudos
    common = dict(flux=(1, 2), kmesh=(1, 1), nbnd=2, free=True, grid=SMALL)
    with pytest.raises(NotImplementedError, match="GIPAW"):
        run_magnetic_bands(system, pseudos, None, local_only=False, **common)
    with pytest.raises(NotImplementedError, match="along a3"):
        run_magnetic_bands(system, pseudos, None, field_axis=0, **common)
    with pytest.raises(MemoryError, match="GB"):
        run_magnetic_bands(system, pseudos, None, memory_limit=1e3, **common)
    with pytest.raises(ValueError, match="not reduced"):
        run_magnetic_bands(system, pseudos, None, **{**common, "flux": (2, 4)})
    with pytest.raises(ValueError, match="closed plane mesh"):
        run_magnetic_bands(system, pseudos, None, chern=True, kpoints=[[0, 0, 0]],
                           **{k: v for k, v in common.items() if k != "kmesh"})
    spinor = SimpleNamespace(npol=2, nspin=1, cell=system.cell, ecutwfc=system.ecutwfc)
    with pytest.raises(NotImplementedError, match="spinor"):
        run_magnetic_bands(spinor, (), None, **common)
    real = pseudos[0]
    soft = SimpleNamespace(is_paw=False, is_ultrasoft=True, element="C ",
                           functional=real.functional)
    with pytest.raises(NotImplementedError, match="ultrasoft"):
        run_magnetic_bands(system, (soft,), None, **common)
    with pytest.raises(ValueError, match="zero field"):
        blochgrid.magnetic_cell(system.cell, 0, 2)
