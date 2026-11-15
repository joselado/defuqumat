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

import jax
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
from defuqumat.magnetic.fold import run_magnetic_scf
from defuqumat.pseudo.upf import read_upf
from defuqumat.workflows.magnetic_bands import run_magnetic_bands

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear():
    # the file compiles programs for six or more box shapes; drop them after each test
    yield
    jax.clear_caches()

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
                                      grid=SMALL, local="filtered")
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
        # p states per level, split by the coarse box (1.4e-5 Ry at p/q = 2/3 measured at the fixed window; under 1e-4 of the level asserted)
        assert np.max(np.abs(values.reshape(-1, p) - values.reshape(-1, p)[:, :1])) < 1e-4 * reference[0]


@pytest.mark.parametrize("p,q", [(1, 3), (2, 3)])
def test_the_magnetic_translation_moves_kappa2_by_p_over_q_exactly(graphene, potential, p, q):
    kappa2 = 0.21
    shifted = (kappa2 - p / q) % 1.0
    points = [[0.13, kappa2, 0.0], [0.13, shifted, 0.0], [0.13 + 1.0, kappa2 - 1.0, 0.0]]

    def spectra(window, kpoints=points, local="filtered"):
        h = MagneticGridHamiltonian.build(graphene.system.cell, p, q, 50.0, kpoints, potential,
                                          grid=(4, 4, 4), window=window, local=local)
        return [np.linalg.eigvalsh(dense_matrix(h, ik)) for ik in range(len(kpoints))]

    a, b, c = spectra("centred")
    assert np.max(np.abs(a - b)) < 1e-10 * np.max(np.abs(a))
    # and the period of the zone: kappa and kappa + (1, -1, 0) are one point
    assert np.max(np.abs(a - c)) < 1e-10 * np.max(np.abs(a))
    # a shift that is not a magnetic translation does move the spectrum
    (other,) = spectra("centred", [[0.13, (kappa2 - 0.5 / q) % 1.0, 0.0]])
    assert np.max(np.abs(a - other)) > 1e-4
    # the two-grid local term keeps both
    ga, gb, gc = spectra("centred", local="two-grid")
    assert np.max(np.abs(ga - gb)) < 1e-10 * np.max(np.abs(ga))
    assert np.max(np.abs(ga - gc)) < 1e-10 * np.max(np.abs(ga))
    # the fixed window (Lu's FFT as written) breaks both on this coarse box
    fa, fb, fc = spectra("fixed")
    assert np.max(np.abs(fa - fb)) > 1e-3 and np.max(np.abs(fa - fc)) > 1e-3


def test_the_two_grid_local_term_is_the_plane_wave_galerkin_matrix_at_zero_field(graphene, potential):
    # p = 0: the box is a plane-wave basis, and V u on the dense grid restricted back is
    # <m'|V|m> = V(m' - m) exactly, the coefficient of the dense potential
    h = MagneticGridHamiltonian.build(graphene.system.cell, 0, 1, 50.0, [[0.0, 0.0, 0.0]], potential,
                                      grid=SMALL, local="two-grid")
    n = h.ndim
    basis = np.fft.ifftn(np.eye(n).reshape((n,) + h.box), axes=(1, 2, 3)).reshape(n, n) * np.sqrt(n)
    applied = np.asarray(h.local_term(jnp.asarray(basis.reshape((n,) + h.box)), jnp.zeros(3))).reshape(n, n)
    matrix = basis.conj() @ applied.T                                   # [m', m]
    coefficients = np.fft.fftn(potential) / potential.size
    m = np.stack(np.meshgrid(*[np.rint(np.fft.fftfreq(k, 1.0 / k)).astype(int) for k in h.box],
                             indexing="ij"), axis=-1).reshape(-1, 3)
    d = m[:, None, :] - m[None, :, :]
    expected = coefficients[d[..., 0] % potential.shape[0], d[..., 1] % potential.shape[1],
                            d[..., 2] % potential.shape[2]]
    assert np.max(np.abs(matrix - expected)) < 1e-12 * np.max(np.abs(expected))


def test_the_two_grid_local_term_is_hermitian_and_one_for_a_constant(graphene, potential):
    h = MagneticGridHamiltonian.build(graphene.system.cell, 2, 3, 50.0, [[0.37, 0.29, 0.0]], potential,
                                      grid=(4, 4, 4), local="two-grid")
    built = dense_matrix(h, 0)
    assert np.max(np.abs(built - built.conj().T)) / np.max(np.abs(built)) < 1e-13
    flat = MagneticGridHamiltonian.build(graphene.system.cell, 2, 3, 50.0, [[0.37, 0.29, 0.0]],
                                         np.full(potential.shape, -0.7), grid=(4, 4, 4))
    rng = np.random.default_rng(5)
    u = rng.normal(size=(1,) + flat.box) + 1j * rng.normal(size=(1,) + flat.box)
    out = np.asarray(flat.local_term(jnp.asarray(u), flat.kpoints[0]))
    assert np.max(np.abs(out + 0.7 * u)) < 1e-12


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
    # the sign is pinned: +1 at p = +1 with the repository's FHS orientation, so a flipped
    # field or a flipped orientation fails here (the magnitude alone would pass for -1)
    assert abs(chern - 1.0) < 1e-9
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
    with pytest.raises(ValueError, match="GIPAW"):
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
    ultrasoft = read_upf("tests/data/pseudo/C.pz-rrkjus.UPF")
    with pytest.raises(NotImplementedError, match="norm-conserving only"):
        blochgrid.grid_projectors(blochgrid.magnetic_cell(system.cell, 1, 1), system.structure,
                                  (ultrasoft,), (24, 24, 128), 1000)
    with pytest.raises(ValueError, match="starts from a density"):
        run_magnetic_scf(system, pseudos, None, flux=(1, 1), kmesh=(1, 1), nbnd=4)


# --- step 2: the GIPAW-dressed nonlocal term, the fold, the energy ------------------------

ODD = (5, 5, 9)


def _spectra(h, chunk=32):
    return [np.linalg.eigvalsh(dense_matrix(h, ik, chunk=chunk)) for ik in range(h.nk)]


def _full(graphene, potential, p, q, kpoints, grid=ODD, **options):
    return MagneticGridHamiltonian.build(graphene.system.cell, p, q, 50.0, kpoints, potential,
                                         grid=grid, structure=graphene.system.structure,
                                         pseudos=graphene.pseudos, **options)


@pytest.fixture(scope="module")
def silicon():
    return Calculator.from_keywords(
        lattice="fcc", a=10.2 * 0.529177210903,
        atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
        pseudos={"Si": "Si.pz-vbc.UPF"}, pseudo_dir="tests/data/pseudo",
        ecut=12 * RY_TO_EV, kmesh=(1, 1, 1))


def test_the_grid_projectors_are_vkb_on_silicon(silicon):
    # s and p projectors: the (-i)^l of vkb has to come out of the real-space form. Measured
    # on D22 on the four lowest eigenstates: becp within 2.7e-6 of vkb's (6.4e-6 of the largest), <V_NL>
    # within 6.6e-6 Ry.
    from defuqumat.magnetic.blochgrid import sphere_state_on_box
    from defuqumat.system.kpoints import KPoints

    cell = silicon.system.cell
    kcrys = np.array([0.5, 0.0, 0.25])
    sub = silicon.with_kpoints(KPoints.from_crystal(kcrys[None], np.ones(1), cell))
    calculation = sub.calculation
    vltot = np.asarray(calculation.vltot)
    h = MagneticGridHamiltonian.build(cell, 0, 1, sub.system.ecutwfc, [kcrys], vltot,
                                      structure=sub.system.structure, pseudos=sub.pseudos)
    planewaves = calculation.basis.planewaves
    miller = np.asarray(calculation.basis.smooth.miller)[np.asarray(planewaves.indices[0])]
    g2 = np.sum((miller + kcrys) ** 2, axis=1)
    mask = np.asarray(planewaves.mask[0], float)
    rng = np.random.default_rng(11)
    c = (rng.normal(size=(3, mask.size)) + 1j * rng.normal(size=(3, mask.size))) * mask / (1.0 + g2)
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    u = sphere_state_on_box(calculation, c, 0, h.box)
    vkb = np.asarray(calculation.projectors.project(jnp.asarray(c), 0))
    tau = np.asarray(sub.system.structure.positions)
    k = TPI * kcrys @ np.asarray(jnp.linalg.inv(cell.at).T)

    def grid_becp(kappa):
        x = h.interpolate(jnp.asarray(u).reshape((-1,) + h.box), jnp.asarray(kappa))
        b = np.asarray(h.projectors.project(x.reshape(x.shape[0], -1), jnp.asarray(kappa)))
        return (b * np.exp(1j * (tau @ k))[None, :, None]).reshape(len(c), -1)

    scale = np.max(np.abs(vkb))
    assert h.projectors.nproj == 4 and h.projectors.dropped < 1e-10
    assert np.max(np.abs(grid_becp(kcrys) - vkb)) < 1e-4 * scale
    # the comparison can fail: the bra at another k is far off
    assert np.max(np.abs(grid_becp(kcrys + np.array([0.1, 0.0, 0.0])) - vkb)) > 1e-2 * scale


def test_the_full_operator_is_hermitian_and_covariant_and_the_bare_one_is_not(graphene, potential):
    p, q, kappa2 = 1, 2, 0.21
    points = [[0.13, kappa2, 0.0], [0.13, (kappa2 - p / q) % 1.0, 0.0]]
    h = _full(graphene, potential, p, q, points)
    rng = np.random.default_rng(3)
    v = rng.normal(size=(2, h.ndim)) + 1j * rng.normal(size=(2, h.ndim))
    hv = np.asarray(h.apply(jnp.asarray(v), 0))
    lhs, rhs = np.vdot(v[0], hv[1]), np.vdot(hv[0], v[1])
    assert abs(lhs - rhs) / (np.linalg.norm(v[0]) * np.linalg.norm(hv[1])) < 1e-12
    a, b = _spectra(h)
    assert np.max(np.abs(a - b)) < 1e-10 * np.max(np.abs(a))
    # the projectors without the GIPAW phase break the magnetic translation (the guard fires)
    bare_a, bare_b = _spectra(_full(graphene, potential, p, q, points, dressed=False))
    assert np.max(np.abs(bare_a - bare_b)) > 1e-6
    # and the nonlocal term is there: it moves the spectrum of the local-only operator
    local = MagneticGridHamiltonian.build(graphene.system.cell, p, q, 50.0, points, potential, grid=ODD)
    assert np.max(np.abs(_spectra(local)[0] - a)) > 1e-2


def test_the_field_along_minus_a3_is_the_time_reversed_operator(graphene, potential):
    # H(-B, -k) = conj H(B, k) holds on the grid only where every row's frequency window is
    # mirror-symmetric: an odd magnetic box (q n1, n2, n3 all odd; an even axis has its
    # half-open Nyquist slot) and no window centre -(p xi1 + kappa2), p xi2 - kappa1 within
    # the tie offset of a half-integer. On the 10 x 5 x 9 box of q = 2 the matrices differ by
    # 0.09 of their largest entry at this kappa and on 4 x 4 x 8 by 0.34 (M/tr2.log); here q = 1.
    kappa = (1.0 / 48.0, 0.25, 0.0)
    plus = _full(graphene, potential, 1, 1, [kappa])
    minus = _full(graphene, potential, -1, 1, [[-x for x in kappa]])
    hp, hm = dense_matrix(plus, 0, chunk=32), dense_matrix(minus, 0, chunk=32)
    assert np.max(np.abs(hp - hm.conj())) < 1e-12 * np.max(np.abs(hp))
    a, b = np.linalg.eigvalsh(hp), np.linalg.eigvalsh(hm)
    assert np.max(np.abs(a - b)) < 1e-10 * np.max(np.abs(a))
    # while the opposite field at the same k is another operator
    (c,) = _spectra(_full(graphene, potential, -1, 1, [kappa]))
    assert np.max(np.abs(a - c)) > 1e-6
    cell = blochgrid.magnetic_cell(graphene.system.cell, -1, 2)
    assert float(cell.scale) == -1.0 and float(cell.b_ph) < 0


def test_a_rigid_shift_of_the_crystal_moves_the_levels_and_keeps_their_set(graphene, potential):
    # Lu's magnetic band shift (Fig. 3e,f): at n_Phi = 1 moving every atom by a1/2 moves k by
    # half of b2, so eps(k) changes and the set over a mesh closed under that shift does not
    import equinox as eqx

    structure = graphene.system.structure
    a1 = np.asarray(graphene.system.cell.at)[0]
    moved = eqx.tree_at(lambda s: s.positions, structure, structure.positions + 0.5 * jnp.asarray(a1))
    rolled = np.roll(potential, potential.shape[0] // 2, axis=0)
    mesh = [[i / 2, j / 2, 0.0] for i in range(2) for j in range(2)]
    cell = graphene.system.cell
    h0 = MagneticGridHamiltonian.build(cell, 1, 1, 50.0, mesh, potential, grid=(4, 5, 9),
                                       structure=structure, pseudos=graphene.pseudos)
    h1 = MagneticGridHamiltonian.build(cell, 1, 1, 50.0, mesh, rolled, grid=(4, 5, 9),
                                       structure=moved, pseudos=graphene.pseudos)
    s0, s1 = _spectra(h0), _spectra(h1)
    assert max(np.max(np.abs(x - y)) for x, y in zip(s0, s1)) > 1e-3
    assert np.max(np.abs(np.sort(np.concatenate(s0)) - np.sort(np.concatenate(s1)))) < 1e-10
    # kappa2 -> kappa2 - 1/2 pairs them point by point
    assert np.max(np.abs(s0[0] - s1[1])) < 1e-10 * np.max(np.abs(s0[0]))


def test_the_fold_and_the_density_of_a_state(graphene, potential):
    from defuqumat.magnetic.fold import band_density, fold_to_chemical, representatives

    kpts, weights = representatives((2, 6), 3)
    assert kpts.shape == (4, 3) and abs(weights.sum() - 2.0) < 1e-15
    assert np.allclose(sorted(set(kpts[:, 1])), [0.0, 1 / 6])
    with pytest.raises(ValueError, match="multiple of q"):
        representatives((2, 5), 3)
    blocks = np.arange(12).reshape(6, 2, 1).astype(float)
    assert np.allclose(np.asarray(fold_to_chemical(blocks, 3)), blocks.reshape(3, 2, 2, 1).mean(0))
    h = _full(graphene, potential, 1, 2, [[0.37, 0.29, 0.0]])
    rng = np.random.default_rng(9)
    v = rng.normal(size=(3, h.ndim)) + 1j * rng.normal(size=(3, h.ndim))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    rho = band_density(h, jnp.asarray(v), jnp.asarray([2.0, 1.0, 0.5]), 0)
    volume = float(graphene.system.cell.volume) * 2
    assert abs(float(jnp.sum(rho)) * volume / rho.size - 3.5) < 1e-12


def test_the_occupations_and_the_molar_magnetizability():
    from defuqumat.magnetic.fold import AVOGADRO, RY_ERG, magnetic_occupations, molar_magnetizability

    fixed = SimpleNamespace(occupations="fixed")
    e = np.array([[0.0, 1.0, 2.0], [0.1, 1.1, 2.1]])
    w = np.array([1.0, 1.0])
    wg, ef, ts = magnetic_occupations(e, w, 4.0, fixed)
    assert np.allclose(wg, [[1, 1, 0], [1, 1, 0]]) and 1.1 < ef < 2.0 and ts == 0.0
    with pytest.raises(ValueError, match="crossing"):
        magnetic_occupations(np.array([[0.0, 1.0, 2.0], [0.1, 2.5, 3.0]]), w, 4.0, fixed)
    smeared = SimpleNamespace(occupations="smearing", degauss=0.01, smearing="mv")
    with pytest.raises(ValueError, match="raise nbnd"):
        magnetic_occupations(e, w, 5.9, smeared)
    with pytest.raises(NotImplementedError, match="tetrahedra"):
        magnetic_occupations(e, w, 4.0, SimpleNamespace(occupations="tetrahedra"))
    # E = E0 - chi B^2 / 2 per formula unit gives chi back
    chi, b = -11.16e-6, 10877.0
    de = -0.5 * chi * (b * 1e4) ** 2 / AVOGADRO / RY_ERG * 12
    assert abs(molar_magnetizability(de, 0.0, de, b, units=12) - chi * 1e6) < 1e-9


def test_item1_the_full_operator_at_zero_field_against_the_sphere(graphene, potential):
    # a = 0, q = 1, k = 0 on the shared potential, 50 Ry. On the sphere's own states the box
    # operator's Rayleigh quotients differ from the sphere eigenvalues by the grid projectors'
    # error alone (the kinetic and the two-grid local term are exact there): 5.1e-7 Ry on D22.
    # The box's own levels are lower, by its larger basis: 2.0e-2 Ry at most at 50 Ry and
    # 7.2e-3 at 70 Ry (plans/orbital-field/route-p.md, item 1).
    from defuqumat.magnetic.blochgrid import sphere_levels, sphere_state_on_box

    calculation = graphene.calculation
    es, vs, _, unsettled = sphere_levels(calculation, potential, 8)
    assert int(unsettled[0]) == 0
    h = MagneticGridHamiltonian.build(graphene.system.cell, 0, 1, 50.0, [[0.0, 0.0, 0.0]],
                                      potential, structure=graphene.system.structure,
                                      pseudos=graphene.pseudos)
    u = sphere_state_on_box(calculation, vs[0], 0, h.box)
    hu = np.asarray(h.apply(jnp.asarray(u), 0))
    rayleigh = np.real(np.einsum("bi,bi->b", u.conj(), hu))
    assert np.max(np.abs(rayleigh - es[0])) < 2e-6
    bands = run_magnetic_bands(graphene.system, graphene.pseudos, {"potential": potential[None]},
                               flux=(0, 1), kpoints=[[0, 0, 0]], nbnd=8, local_only=False)
    difference = bands.eigenvalues[0] - es[0]
    assert np.all(difference < 1e-8) and np.max(np.abs(difference)) < 2.5e-2


def test_the_magnetizability_refuses_an_unconverged_field_scf(monkeypatch):
    """An SCF stopped at its budget has a non-variational total; chi divides its error by dB^2 (audit ``magnetic`` F4)."""
    from types import SimpleNamespace

    import defuqumat.magnetic.fold as fold

    calls = []

    def stand_in(system, pseudos, scf0, *, flux, **options):
        calls.append(flux[0])
        return SimpleNamespace(total_energy=-1.0 + 0.01 * flux[0] ** 2,
                               converged=flux[0] != 1)

    monkeypatch.setattr(fold, "run_magnetic_scf", stand_in)
    with pytest.raises(RuntimeError, match=r"did not converge at flux \[1\]"):
        fold.run_magnetizability(None, None, None, kmesh=(1, 1, 1), nbnd=4)


def test_the_time_reversal_control_can_be_skipped(monkeypatch):
    """``time_reversal_check=False`` runs ``0`` and ``+1`` only and takes ``E(-) = E(+)`` (audit ``opt-repeat`` F3)."""
    from types import SimpleNamespace

    import defuqumat.magnetic.fold as fold

    calls = []

    def stand_in(system, pseudos, scf0, *, flux, **options):
        calls.append(flux[0])
        return SimpleNamespace(total_energy=-1.0 + 0.01 * flux[0] ** 2, converged=True)

    monkeypatch.setattr(fold, "run_magnetic_scf", stand_in)
    monkeypatch.setattr(fold, "magnetic_cell", lambda cell, p, q: SimpleNamespace(area=10.0))
    system = SimpleNamespace(cell=None, structure=SimpleNamespace(types=(0, 0)))
    both = fold.run_magnetizability(system, None, None, kmesh=(1, 1, 1), nbnd=4)
    assert calls == [0, 1, -1] and both.asymmetry == 0.0
    calls.clear()
    one = fold.run_magnetizability(system, None, None, kmesh=(1, 1, 1), nbnd=4,
                                   time_reversal_check=False)
    assert calls == [0, 1] and one.asymmetry is None
    assert one.chi_per_unit == both.chi_per_unit


# 8.9 s on this workstation (2026-11-15), above the gate's five seconds for an identity
@pytest.mark.slow
def test_the_fold_scf_keeps_time_reversal_on_an_odd_box():
    """The route P SCF itself, at +1 and -1 flux quantum per cell (audit 2026-11-15, P177 hole).

    No fast test ran the loop (diagonalise, fold, mix, energy). ``H(-B, -k) = conj H(B, k)``
    holds on the grid when every row's frequency window is mirror-symmetric, which needs an
    odd box (P177). Silicon at 8 Ry, one k-point, from a uniform density, on the odd 9 x 9 x 9
    box: E(+) - E(-) = -1.3e-11 Ry (2026-11-15). On the default even 8 x 8 x 8 box at 12 Ry it
    is 2.1e-5, and 6.1e-6 on 10 x 10 x 10, with both runs at conv_thr = 1e-10: the Nyquist
    slot of an even box, not the SCF floor. A sign or gauge defect is first order in B,
    about 0.6 Ry here. The energy is pinned as well, so a change in the loop shows.
    """
    calc = Calculator.from_keywords(
        lattice="fcc", a=10.2 * 0.529177210903,
        atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
        pseudos={"Si": "Si.pz-vbc.UPF"}, pseudo_dir="tests/data/pseudo",
        ecut=8 * RY_TO_EV, kmesh=(1, 1, 1), qe={"system": {"nosym": True, "noinv": True}})
    grid = tuple(calc.calculation.basis.dense.grid)
    start = {"density": np.full((1,) + grid, calc.calculation.nelec / float(calc.system.cell.volume))}
    energies = {}
    for p in (1, -1):
        r = run_magnetic_scf(calc.system, calc.pseudos, start, flux=(p, 1), kmesh=(1, 1, 1),
                             nbnd=6, conv_thr=1e-9, verbose=False, grid=(9, 9, 9))
        assert r.converged and r.iterations <= 15
        energies[p] = r.total_energy
    assert abs(energies[1] - energies[-1]) < 1e-9
    assert energies[1] == pytest.approx(-15.1857113446, abs=1e-8)
