"""Route U step 1, the magnetic ultracell (Wang and Vafek, PRB 108, 245109 (2023)).

The fast checks: the parallel-transport gauge on a model whose centres are known, the
zero-flux limit (the magnetic machinery must give back the zero-field bands to round-off),
the free Landau problem on a small cell, the Hermiticity and rank guards, and the
refusals. The acceptance numbers on the graphene cell are probes
(``plans/orbital-field/mag_route_u.py``) and ``tests/regression/test_magnetic_ultracell_slow.py``.
"""

import jax
import numpy as np
import pytest

from defuqumat.magnetic.hybrid import outside_weight, half_width, transport_gauge
from defuqumat.magnetic.ultracell_field import assemble, hermiticity, hermiticity_guard, solve


@pytest.fixture(autouse=True)
def _clear():
    yield
    jax.clear_caches()


def _orbital_links(taus, n1, seed=0):
    """Links of a model whose Bloch states are point orbitals at ``taus`` (cells along a1).

    The periodic part of orbital ``s`` at ``k1 = i/n1`` is ``e^{-2 pi i k1 tau_s}`` on its
    own site, so ``<u_i|u_{i+1}> = diag(e^{-2 pi i tau/n1})``, and the closing link (``u_N(G)
    = u_0(G + b1)``) is the same. Each k-point gets a random unitary mixing.
    """
    rng = np.random.default_rng(seed)
    nb = len(taus)
    mix = []
    for _ in range(n1):
        a = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
        q, _ = np.linalg.qr(a)
        mix.append(q)
    base = np.diag(np.exp(-2j * np.pi * np.asarray(taus) / n1))
    links = [mix[i].conj().T @ base @ mix[(i + 1) % n1] for i in range(n1)]
    return np.stack(links), mix


def test_the_transport_gauge_finds_the_centres_and_is_periodic():
    taus = [0.1, -0.3, 0.45]
    n1 = 7
    links, mix = _orbital_links(taus, n1)
    gauge = transport_gauge(links)
    assert np.allclose(np.sort(gauge.centres), np.sort(taus), atol=1e-12)
    # every link in the new gauge, the closing one included, is diag(e^{i phi/N1}) times a
    # Hermitian positive matrix, and for this model the positive part is the identity
    r = gauge.rotation
    for i in range(n1):
        new = r[i].conj().T @ links[i] @ r[(i + 1) % n1]
        assert np.allclose(new, np.diag(np.exp(1j * gauge.phases / n1)), atol=1e-12)
    assert gauge.singular == pytest.approx(1.0, abs=1e-12)


def test_the_other_sign_of_the_phase_is_not_periodic():
    """The design's exp(-i phi i/N1) leaves a jump of 2 phi on the closing link."""
    taus = [0.2]
    n1 = 6
    links, _ = _orbital_links(taus, n1, seed=3)
    gauge = transport_gauge(links)
    wrong = np.stack([gauge.rotation[i] * np.exp(-2j * gauge.phases * i / n1)
                      for i in range(n1)])
    closing = (wrong[-1].conj().T @ links[-1] @ wrong[0])[0, 0]
    inner = (wrong[0].conj().T @ links[0] @ wrong[1])[0, 0]
    jump = np.angle(closing / inner)
    assert abs(abs(jump) - abs(2 * gauge.phases[0])) < 1e-10
    assert abs(jump) > 1.0


def test_the_outside_weight_and_the_half_width():
    xi = np.linspace(-4.5, 4.5, 91)
    profile = np.exp(-xi ** 2 / 0.05)[None, :]
    widths = np.arange(4)
    weights = outside_weight(profile, xi, widths)
    assert weights[0, 0] > 1e-6
    assert half_width(weights, widths, 1e-8) == 1
    assert half_width(weights, widths, 1e-300) is None


def test_assemble_is_the_bloch_sum_of_a_banded_chain():
    """One band, q = 3, nearest neighbour t: H(kappa1) is the 3-site ring with phase on the wrap."""
    q, w, t = 3, 1, -0.7
    B = np.zeros((q, 2 * w + 1, 1, 1), complex)
    S = np.zeros_like(B)
    B[:, 0], B[:, 2] = t, t
    S[:, 1] = 1.0
    for kappa1 in (0.0, 0.25, 0.5):
        h, s = assemble(kappa1, B, S, w, q)
        assert hermiticity(h) < 1e-15
        eps, rank, d_min, d_all = solve(h, s)
        k = 2 * np.pi * (kappa1 + np.arange(q)) / q
        assert rank == q and d_min == pytest.approx(1.0)
        assert np.allclose(np.sort(eps), np.sort(2 * t * np.cos(k)), atol=1e-14)


def test_assemble_folds_a_wide_band_onto_a_small_magnetic_cell():
    """q = 1 with w = 2: every Delta lands on the same block, each with its own phase."""
    B = np.zeros((1, 5, 1, 1), complex)
    S = np.zeros_like(B)
    B[0, :, 0, 0] = [0.1, -0.5, 1.0, -0.5, 0.1]
    S[0, 2] = 1.0
    h, s = assemble(0.3, B, S, 2, 1)
    k = 2 * np.pi * 0.3
    assert h[0, 0] == pytest.approx(1.0 - 1.0 * np.cos(k) + 0.2 * np.cos(2 * k))


def test_the_hermiticity_guard_fires():
    h = np.array([[1.0, 0.5], [0.5 + 1e-6, 2.0]])
    assert hermiticity_guard(np.array([[1.0, 0.5], [0.5, 2.0]])) == 0.0
    with pytest.raises(ValueError, match="not Hermitian"):
        hermiticity_guard(h, tol=1e-8)


def test_the_rank_is_recorded_when_a_direction_is_dropped():
    s = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]])
    h = np.diag([1.0, 2.0])
    eps, rank, d_min, d_all = solve(h, s, drop=1e-8)
    assert rank == 1 and d_all < 1e-8 and d_min == pytest.approx(1.0)


# --- the workflow on a small graphene cell ---------------------------------------------

def _small_graphene():
    from defuqumat import Calculator
    from defuqumat.units import RY_TO_EV

    return Calculator.from_keywords(
        lattice="hexagonal", a=2.46, c=5.0,
        atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
        pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
        ecut=20 * RY_TO_EV, kmesh=(4, 4, 1), smearing=0.01 * RY_TO_EV, smearing_type="mv")


def _uniform_density(calc):
    from defuqumat.scf.driver import Calculation

    calculation = Calculation(calc.system, calc.pseudos)
    grid = calculation.basis.dense.grid
    return {"density": np.full((1,) + tuple(grid), float(calculation.nelec)
                               / float(calc.system.cell.volume))}


@pytest.mark.parametrize("h0", ["eigen", "box"])
def test_zero_flux_gives_back_the_zero_field_bands(h0):
    """p = 0: the hybrid states, the box, the rolls and the Bloch sum return eps(k) exactly.

    With an odd N1 and w = (N1 - 1)/2 every Delta of the box is counted once, and the
    magnetic kappa1 = 0 is the mesh point i = 0, so the levels are the zero-field ones.
    """
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _small_graphene()
    r = run_magnetic_ultracell(calc.system, calc.pseudos, _uniform_density(calc), flux=(0, 1),
                               nbnd=4, N1=5, w=2, kmesh=(1, 2), h0=h0)
    for ik, (kappa1, kappa2) in enumerate(r.kappa):
        jk = int(round(kappa2 * 2))
        assert np.max(np.abs(r.levels[ik, :4] - r.eigenvalues[jk, 0, 0])) < 1e-10
    assert np.all(r.hermiticity < 1e-12)
    assert np.all(r.rank == 4)


def test_the_bare_hybrid_basis_of_the_free_problem_is_refused_by_the_hermiticity_guard():
    """Free electrons, q = 2: the lowest bands are entangled, the hybrid states have sinc
    tails, and the assembled H is not Hermitian; strict mode raises, and the reversed field
    reads a larger residual, so the guard is not reading round-off."""
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _small_graphene()
    with pytest.raises(ValueError, match="not Hermitian"):
        run_magnetic_ultracell(calc.system, calc.pseudos, None, flux=(1, 2), nbnd=2, N1=6,
                               kmesh=(1, 1), free=True)
    same = run_magnetic_ultracell(calc.system, calc.pseudos, None, flux=(1, 2), nbnd=2, N1=6,
                                  kmesh=(1, 1), free=True, strict=False)
    reversed_ = run_magnetic_ultracell(calc.system, calc.pseudos, None, flux=(1, 2), nbnd=2,
                                       N1=6, kmesh=(1, 1), free=True, strict=False, scale=-1.0)
    assert np.all(same.hermiticity > 1e-8)
    assert np.all(reversed_.hermiticity > 2.0 * same.hermiticity)


def test_the_window_makes_the_free_problem_hermitian_and_variational():
    """The same free problem with a smooth window (2.0, 2.5) cells on N1 = 10 and the
    symmetric a.p + p.a: H is Hermitian to round-off and every level lies above the exact
    (2n + 1) B_ph + (2 pi m/c)^2 (Rayleigh-Ritz of the box operator)."""
    from defuqumat.magnetic.conventions import MagneticCell
    from defuqumat.magnetic.landau import free_landau_levels
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _small_graphene()
    r = run_magnetic_ultracell(calc.system, calc.pseudos, None, flux=(1, 2), nbnd=2, N1=10,
                               kmesh=(1, 1), free=True, h0="box", window=(2.0, 2.5),
                               form="symmetric")
    assert r.w == 4
    assert np.all(r.hermiticity < 1e-12)
    levels, mult, _ = free_landau_levels(MagneticCell.build(calc.system.cell, 1, 2), 6, g3max=2)
    ref = np.repeat(levels, mult)
    n = min(len(ref), r.levels.shape[1])
    assert np.min(r.levels[:, :n] - ref[None, :n]) > -1e-8


def test_the_nonlocal_term_and_the_regimes_are_refused_by_name():
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _small_graphene()
    with pytest.raises(NotImplementedError, match="nonlocal term"):
        run_magnetic_ultracell(calc.system, calc.pseudos, None, flux=(1, 2), nbnd=2,
                               local_only=False)

    class Ultrasoft:
        is_paw = False
        is_ultrasoft = True
        element = "C"
        functional = calc.pseudos[0].functional

    with pytest.raises(NotImplementedError, match="ultrasoft dataset C"):
        run_magnetic_ultracell(calc.system, (Ultrasoft(),), None, flux=(1, 2), nbnd=2, free=True)


@pytest.mark.parametrize("form", ["ket", "symmetric"])
def test_the_field_kernels_give_the_landau_ladder_on_an_analytic_state(form):
    """The block kernels on exp(-i f) exp(-B x^2/2) and x times it, x = b1hat . r, f the gauge
    of the part of a along b1: E = B_ph and 3 B_ph to round-off; the reversed field does not.
    The operator check that the basis-limited runs cannot give (plans/orbital-field/
    mag_route_u_landau_check.py, review/mag/u1/landau-check-n18.log)."""
    import jax.numpy as jnp

    from defuqumat.magnetic.ultracell_field import MagneticUltracell, _bra_side, _ket_block
    from defuqumat.system.cell import Cell
    from defuqumat.units import BOHR_RADIUS_SI

    a = 2.46e-10 / BOHR_RADIUS_SI
    lattice = np.array([[a, 0, 0], [-a / 2, a * np.sqrt(3) / 2, 0], [0, 0, 15e-10 / BOHR_RADIUS_SI]])
    cell = Cell.from_vectors(lattice)
    q, n1, grid1 = 10, 18, (48, 4, 4)
    for scale, expect in ((1.0, (1.0, 3.0)), (-1.0, None)):
        ultra = MagneticUltracell.build(cell, (1, q), [[]], [0.0], grid1, nbnd=1, n1=n1, w=0,
                                        potential=np.zeros(grid1), h0="box", scale=scale)
        b = np.linalg.inv(lattice).T
        b_ph = 2 * np.pi / q / float(ultra.magnetic.area)
        xi = np.asarray(ultra.xi)
        x = xi / np.linalg.norm(b[0])
        g0 = np.exp(-1j * (2 * np.pi / q) * (b[0] @ b[1]) * x ** 2 / 2) * np.exp(-b_ph * x ** 2 / 2)
        grid = ultra.grid
        u = np.broadcast_to(np.stack([g0, x * g0])[:, :, None, None], (2,) + grid).astype(complex)
        npts = int(np.prod(grid))
        values = jnp.asarray(np.fft.fftn(u, axes=(1, 2, 3)).reshape(2, npts) / npts)
        index = jnp.arange(npts)
        window = jnp.ones_like(ultra.xi)
        zero = jnp.asarray(0.0)
        u_bra, h_bra, p_bra = _bra_side(values, values, index, window, ultra.gg, ultra.g_b2,
                                        ultra.potential, zero, ultra.b22, grid=grid,
                                        box_h0=True, windowed=False)
        h, s = _ket_block(u_bra, h_bra, p_bra, values, index, jnp.asarray(0), window,
                          ultra.g_b2, ultra.xi, zero, ultra.c1, ultra.b12, ultra.b22,
                          grid=grid, windowed=False, symmetric=form == "symmetric")
        e = np.real(np.diag(np.asarray(h)) / np.diag(np.asarray(s))) / b_ph
        if expect is None:
            assert np.all(np.abs(e - (1.0, 3.0)) > 0.5)
        else:
            assert np.allclose(e, expect, rtol=0, atol=1e-12)
            assert abs(np.asarray(h)[0, 1]) < 1e-12 * abs(np.asarray(h)[0, 0])
