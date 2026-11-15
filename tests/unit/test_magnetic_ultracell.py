"""Route U step 1, the magnetic ultracell (Wang and Vafek, PRB 108, 245109 (2023)).

The fast checks: the parallel-transport gauge on a model whose centres are known, the
zero-flux limit (the magnetic machinery must give back the zero-field bands to round-off),
the free Landau problem on a small cell, the Hermiticity and rank guards, and the
refusals. The acceptance numbers on the graphene cell are probes
(``plans/orbital-field/mag_route_u.py``) and ``tests/regression/test_magnetic_ultracell_slow.py``.
"""

import functools

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
    with pytest.raises(ValueError, match="free Landau problem has no nonlocal term"):
        run_magnetic_ultracell(calc.system, calc.pseudos, None, flux=(1, 2), nbnd=2,
                               free=True, nonlocal_term="gipaw")
    with pytest.raises(ValueError, match="carries V_NL\\^0 in its eigenvalues"):
        run_magnetic_ultracell(calc.system, calc.pseudos, _uniform_density(calc), flux=(1, 2),
                               nbnd=2, local_only=False, nonlocal_term="none")

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
        h, s, _ = _ket_block(u_bra, h_bra, p_bra, values, index, jnp.asarray(0), window,
                          ultra.g_b2, ultra.xi, zero, ultra.c1, ultra.b12, ultra.b22,
                          grid=grid, windowed=False, symmetric=form == "symmetric")
        e = np.real(np.diag(np.asarray(h)) / np.diag(np.asarray(s))) / b_ph
        if expect is None:
            assert np.all(np.abs(e - (1.0, 3.0)) > 0.5)
        else:
            assert np.allclose(e, expect, rtol=0, atol=1e-12)
            assert abs(np.asarray(h)[0, 1]) < 1e-12 * abs(np.asarray(h)[0, 0])


@pytest.mark.parametrize("symmetric", [False, True])
def test_the_field_block_keeps_a_single_precision_dtype(symmetric):
    """The block kernel takes its complex dtype from its inputs: complex64 in, complex64 out.

    The ``-i div a`` term of the ``ket`` form was written with a complex literal, which
    CLAUDE.md forbids in a kernel (the dtype comes from the policy); it is now built with
    ``lax.complex`` on the traced ``c1``. Both forms are checked, on random inputs.
    """
    import jax.numpy as jnp

    from defuqumat.magnetic.ultracell_field import _ket_block

    grid, nbnd = (4, 2, 2), 2
    npts = int(np.prod(grid))
    rng = np.random.default_rng(3)

    def c64(*shape):
        return jnp.asarray(rng.normal(size=shape) + 1j * rng.normal(size=shape), dtype=jnp.complex64)

    def f32(*shape):
        return jnp.asarray(rng.normal(size=shape), dtype=jnp.float32)

    u_bra, h_bra, p_bra, values = c64(nbnd, npts), c64(nbnd, npts), c64(nbnd, npts), c64(nbnd, npts)
    h, s, _ = _ket_block(u_bra, h_bra, p_bra, values, jnp.arange(npts), jnp.asarray(1),
                         f32(grid[0]), f32(npts), f32(grid[0]), jnp.asarray(0.1, jnp.float32),
                         jnp.asarray(0.3, jnp.float32), np.float32(-0.2), np.float32(0.7),
                         grid=grid, windowed=False, symmetric=symmetric)
    assert h.dtype == jnp.complex64 and s.dtype == jnp.complex64


# --- step 2: the GIPAW-dressed nonlocal term --------------------------------------------

@pytest.mark.parametrize("h0", ["eigen", "box"])
def test_zero_flux_with_the_nonlocal_term_gives_back_the_full_bands(h0):
    """p = 0, the full zero-field Hamiltonian (acceptance 1 of step 2 on a small cell).

    ``eigen``: H = H_0 + (V_NL^B - V_NL^0) with a zero phase, so the levels are the
    eigenvalues to round-off whatever the grid projectors are. ``box``: T + V_loc on the box
    plus V_NL^0 from the grid projectors, so the levels are the eigenvalues only if the
    grid projectors are the plane-wave ones (the normalisation, the e^{iK.r}, D); the
    projector check ``nonlocal_error`` says the same block by block.
    """
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _small_graphene()
    r = run_magnetic_ultracell(calc.system, calc.pseudos, _uniform_density(calc), flux=(0, 1),
                               nbnd=4, N1=5, w=2, kmesh=(1, 2), h0=h0, local_only=False)
    assert r.nonlocal_mode == ("difference" if h0 == "eigen" else "gipaw")
    worst = max(np.max(np.abs(r.levels[ik, :4] - r.eigenvalues[int(round(k2 * 2)), 0, 0]))
                for ik, (_, k2) in enumerate(r.kappa))
    print(f"\nh0 {h0}: levels - eigenvalues {worst:.3e} Ry, grid - vkb block "
          f"{r.nonlocal_error:.3e} Ry, radius {r.projector_radius}, dropped {r.projector_dropped}")
    # measured on D22 (review/mag/u2/t-unit1.log): 6.1e-6 Ry block, 1.5e-5 Ry box levels,
    # 2.7e-15 Ry eigen levels; the block error is the projector cut at 1e-6 of beta's peak
    assert r.nonlocal_error < 1e-5
    assert worst < (1e-10 if h0 == "eigen" else 3e-5)
    assert np.all(r.hermiticity < 1e-12)


def test_the_grid_projection_error_is_the_cut_of_the_projector():
    """The same zero-flux box run with the projector kept to 1e-12 of its peak (2.1 bohr
    against 1.6): the error of the grid projection falls, so what the default leaves is the
    cut (an amplitude error of the square root of the dropped norm), not a normalisation."""
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _small_graphene()
    runs = [run_magnetic_ultracell(calc.system, calc.pseudos, _uniform_density(calc),
                                   flux=(0, 1), nbnd=4, N1=3, w=1, kmesh=(1, 1), h0="box",
                                   local_only=False, projector_cut=cut)
            for cut in (1e-6, 1e-12)]
    print(f"\ncut 1e-6: {runs[0].nonlocal_error:.3e} Ry at {runs[0].projector_radius}; cut 1e-12: "
          f"{runs[1].nonlocal_error:.3e} Ry at {runs[1].projector_radius}")
    assert runs[1].projector_radius[0] > runs[0].projector_radius[0]
    assert runs[1].nonlocal_error < runs[0].nonlocal_error / 3


def test_the_grid_projectors_with_p_channels_match_vkb():
    """Silicon, Si.pz-vbc (an s and a p projector), 12 Ry, zero flux: the box projection of
    every hybrid block against the plane-wave projectors. This is where the real harmonics,
    their ordering and two channels per atom are exercised; carbon's HGH dataset has one s."""
    from defuqumat import Calculator
    from defuqumat.units import RY_TO_EV
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = Calculator.from_keywords(
        lattice="fcc", a=5.43, atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
        pseudos={"Si": "Si.pz-vbc.UPF"}, pseudo_dir="tests/data/pseudo", ecut=12 * RY_TO_EV,
        kmesh=(2, 2, 2))
    r = run_magnetic_ultracell(calc.system, calc.pseudos, _uniform_density(calc), flux=(0, 1),
                               nbnd=4, N1=3, w=1, kmesh=(1, 1), h0="box", local_only=False)
    worst = float(np.max(np.abs(r.levels[0, :4] - r.eigenvalues[0, 0, 0])))
    print(f"\nsilicon: grid - vkb block {r.nonlocal_error:.3e} Ry, box levels - eigenvalues "
          f"{worst:.3e} Ry")
    assert r.nonlocal_error < 1e-3
    assert worst < 1e-3


def _toy_projectors(n1, p, q, cell_grid=(12, 12, 24)):
    """The grid projectors of the small graphene cell on an ``n1`` box at flux p/q."""
    from defuqumat.magnetic.conventions import MagneticCell
    from defuqumat.magnetic.ultracell_nonlocal import grid_projectors
    from defuqumat.pseudo.projectors import projector_channels

    calc = _small_graphene()
    structure = calc.system.structure
    blocks, atom_of = [], []
    for a, t in enumerate(structure.types):
        pseudo = calc.pseudos[t]
        for nb, _, _ in projector_channels(pseudo):
            blocks.append(float(pseudo.dij[nb, nb]))
            atom_of.append(a)
    magnetic = MagneticCell.build(calc.system.cell, p, q)
    gp = grid_projectors(calc.system.cell, structure, calc.pseudos, np.diag(blocks), atom_of,
                         cell_grid, n1, magnetic)
    return calc, magnetic, gp


def test_the_dressed_nonlocal_term_commutes_with_the_magnetic_translation():
    """<T f| V_NL^B |T g> = <f| V_NL^B |g> for T = T(a1), on the grid, to round-off.

    T(a1) e^{iK.r} u = e^{i(K - c1 b2).r} u(r - a1): the box function rolled by one cell at
    kappa - p/q. f and g are localized at the box centre (nothing reaches the sawtooth's
    jump). The bare V_NL^0 does not commute, and neither does the dressing with its sign
    flipped: the two controls that make the pass mean something.
    """
    import jax.numpy as jnp

    from defuqumat.magnetic.ultracell_nonlocal import projections

    n1, p, q, cell_grid = 8, 1, 3, (12, 12, 24)
    calc, magnetic, gp = _toy_projectors(n1, p, q, cell_grid)
    grid = (n1 * cell_grid[0],) + cell_grid[1:]
    xi = np.arange(grid[0]) / cell_grid[0]
    xi = np.where(xi > n1 / 2, xi - n1, xi)
    rng = np.random.default_rng(1)
    y = np.arange(grid[1]) / grid[1]
    z = np.arange(grid[2]) / grid[2]
    funcs = []
    for s in range(2):
        along = np.exp(-(xi - 0.3 * s) ** 2 / (2 * 0.4 ** 2))
        c = rng.normal(size=4) + 1j * rng.normal(size=4)
        side = (c[0] + c[1] * np.exp(2j * np.pi * y))[:, None] * \
            (c[2] + c[3] * np.cos(2 * np.pi * z))[None, :]
        funcs.append((along[:, None, None] * side[None]).reshape(-1))
    u = jnp.asarray(np.stack(funcs))
    u_t = jnp.asarray(np.roll(np.stack(funcs).reshape((2,) + grid), cell_grid[0], axis=1)
                      .reshape(2, -1))
    kappa = 0.21

    def matrix(gp_, uu, kref, dressed=True):
        bare, dress = projections(uu, gp_.index, gp_.beta, gp_.d_b2, gp_.phase, kref)
        pr = dress if dressed else bare
        return np.asarray(jnp.einsum("ban,anm,cam->bc", pr.conj(), gp_.dij, pr))

    m = matrix(gp, u, kappa)
    m_t = matrix(gp, u_t, kappa - p / q)
    scale = np.max(np.abs(m))
    print(f"\ndressed: |M_T - M| / |M| = {np.max(np.abs(m_t - m)) / scale:.3e}")
    assert scale > 1e-3
    assert np.max(np.abs(m_t - m)) < 1e-12 * scale
    bare = np.max(np.abs(matrix(gp, u_t, kappa - p / q, False) - matrix(gp, u, kappa, False)))
    flipped = gp.with_phase(-gp.phase)
    wrong = np.max(np.abs(matrix(flipped, u_t, kappa - p / q) - matrix(flipped, u, kappa)))
    print(f"bare {bare / scale:.3e}, flipped sign {wrong / scale:.3e}")
    assert bare > 1e-2 * scale
    assert wrong > 1e-2 * scale


def test_the_grid_projectors_are_the_atoms_of_the_box_in_the_sawtooth_frame():
    """N1 x nat atoms, each once, with xi1 in (-N1/2, N1/2], the same sphere per species."""
    n1 = 6
    calc, magnetic, gp = _toy_projectors(n1, 1, 4)
    at = np.asarray(calc.system.cell.at)
    xi = gp.tau @ np.linalg.inv(at)[:, 0]
    assert gp.index.shape[0] == 2 * n1
    assert np.all(xi > -n1 / 2 - 1e-12) and np.all(xi <= n1 / 2 + 1e-12)
    assert len(np.unique(np.round(xi, 9))) == 2 * n1
    count = np.sum(np.asarray(gp.beta)[:, 0] != 0, axis=1)
    assert len(set(count.tolist())) <= 2 and gp.radius[0] > 1.0
    assert gp.dropped[0] < 1e-9
    assert np.max(np.abs(np.asarray(gp.phase))) > 0.1


def test_the_sublattice_masks_partition_the_box():
    from defuqumat.ultracell.grid import Ultracell
    from defuqumat.workflows.landau_ultracell import sublattice_masks

    calc = _small_graphene()
    box = Ultracell.build((3, 1, 1), (12, 12, 24), calc.system.cell.precision)
    masks = sublattice_masks(calc.system.cell, calc.system.structure, box, [[0], [1]])
    assert masks.shape == (2, 36 * 12 * 24)
    assert np.allclose(masks.sum(axis=0), 1.0)
    # inversion about the bond centre maps this grid onto itself and A onto B
    assert abs(masks[0].mean() - 0.5) < 1e-12


# --- step 3: the projected trial basis --------------------------------------------------

def test_projected_hybrid_is_the_overlap_and_its_projectability():
    """A = <psi|g> over the lowest N bands; p = 1 for a trial inside the window, the weight
    left inside for one that is not; and the bands above N do not enter."""
    from defuqumat.magnetic.hybrid import projected_hybrid

    rng = np.random.default_rng(5)
    n1, nbnd, npw = 3, 5, 12
    psi = np.stack([np.linalg.qr(rng.normal(size=(npw, nbnd)) + 1j * rng.normal(size=(npw, nbnd)))[0].T
                    for _ in range(n1)])                                   # orthonormal rows
    inside = np.einsum("km,kmg->kg", rng.normal(size=(n1, 3)), psi[:, :3])
    half = psi[:, 0] + psi[:, 4]                                            # half of it above N
    trials = np.stack([inside, half], axis=1)
    gauge = projected_hybrid(psi, trials, 3)
    assert gauge.rotation.shape == (n1, 3, 2)
    assert np.allclose(gauge.rotation, np.einsum("kmg,kjg->kmj", psi[:, :3].conj(), trials))
    assert np.allclose(gauge.projectability[:, 0], 1.0, atol=1e-14)
    assert np.allclose(gauge.projectability[:, 1], 0.5, atol=1e-14)
    assert np.all(gauge.centres == 0.0) and 0.0 < gauge.singular <= 1.0 + 1e-14
    with pytest.raises(ValueError, match="fewer than the window"):
        projected_hybrid(psi, trials, 6)


def test_the_stripe_trial_is_periodic_in_kappa():
    """g at (k1, k2 + 1) on the sphere relabelled by G2 -> G2 - 1 is g at (k1, k2): the trial is a
    function of k + G alone, which the reuse of the reduced states needs. The G2 = 0 stripe of
    the probe's first try is not."""
    from defuqumat.magnetic.hybrid import StripeTrials, stripe_trials

    g1, g2 = np.meshgrid(np.arange(-3, 4), np.arange(-2, 3), indexing="ij")
    miller = np.stack([g1.ravel(), g2.ravel(), np.zeros(g1.size, int)], -1)
    valid = np.ones(len(miller), bool)
    spec = StripeTrials(J=3, sigma=0.5)
    for k2 in (0.2, 0.75):
        a = stripe_trials(spec, [[0.3, k2, 0.0]], miller[None], valid[None])[0]
        b = stripe_trials(spec, [[0.3, k2 + 1.0, 0.0]], (miller - [0, 1, 0])[None], valid[None])[0]
        assert np.allclose(a, b, atol=0.0) and np.max(np.abs(a)) > 0.1
        assert np.count_nonzero(np.abs(a[0]) > 0) == 7                     # one stripe, 7 G1
    flat = np.where(miller[:, 1] == 0, 1.0, 0.0)
    a = stripe_trials(spec, [[0.3, 0.75, 0.0]], miller[None], valid[None])[0, 0]
    assert np.all(np.abs(a[flat > 0]) == 0.0)                              # k2 0.75 keeps G2 = -1
    with pytest.raises(ValueError, match="J = 1 only"):
        StripeTrials(J=2, chirp=True)


@functools.lru_cache(maxsize=2)
def _free_projected(J, scale=1.0, N=48, strict=True):
    from defuqumat.magnetic.hybrid import StripeTrials
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _small_graphene()
    return run_magnetic_ultracell(calc.system, calc.pseudos, None, flux=(1, 2), free=True,
                                  trials=StripeTrials(J=J, sigma=0.5), N=N, kmesh=(1, 1),
                                  scale=scale, strict=strict)


def test_the_projected_levels_fall_monotonically_in_J_and_stay_above_the_ladder():
    """Free electrons at 1/2, the stripe Hermite-Gaussians, J = 1 then 2 at N = 48: nested spans,
    so every level at J = 2 is at or below the same level at J = 1, every level is above the
    exact (2n + 1) B_ph + (2 pi m/c)^2 (the window, the box and the symmetric form make it an
    exact Rayleigh-Ritz problem), and H is Hermitian to round-off."""
    from defuqumat.magnetic.conventions import MagneticCell
    from defuqumat.magnetic.landau import free_landau_levels

    r1, r2 = _free_projected(1), _free_projected(2)
    assert (r1.nbnd, r2.nbnd, r1.N, r1.n1, r1.window, r1.h0, r1.form) == \
        (1, 2, 48, 16, (3.0, 4.0), "box", "symmetric")
    assert r1.eigenvalues.shape[-1] == 48 and r2.projectability.shape[-1] == 2
    print(f"\nJ 1: 1 - p {r1.projectability_deficit:.2e}, herm {r1.hermiticity.max():.1e}, "
          f"levels/B {r1.levels[0, :4] / r1.b_ph}; J 2: 1 - p {r2.projectability_deficit:.2e}, "
          f"herm {r2.hermiticity.max():.1e}, levels/B {r2.levels[0, :4] / r2.b_ph}")
    assert np.all(r1.hermiticity < 1e-10) and np.all(r2.hermiticity < 1e-10)
    n = r1.levels.shape[1]
    assert np.all(r2.levels[:, :n] <= r1.levels + 1e-10)
    assert np.any(r2.levels[:, 0] < r1.levels[:, 0] - 1e-6)                # J = 2 does improve
    levels, mult, _ = free_landau_levels(MagneticCell.build(_small_graphene().system.cell, 1, 2), 6,
                                         g3max=2)
    ref = np.repeat(levels, mult)
    for r in (r1, r2):
        m = min(len(ref), r.levels.shape[1])
        assert np.min(r.levels[:, :m] - ref[None, :m]) > -1e-8
    # And from above, on the n = 0 level, which is the one this basis converges: it
    # reads 0.130 and 0.058 B_ph above the ladder at J = 1 and 2 (2026-11-15). A
    # positive term added to the projected H keeps every assertion above, so this is
    # the one that can see it (audit 2026-11-15, ``tests`` hole on P176 step 3).
    errors = [float((r.levels[0, 0] - ref[0]) / r.b_ph) for r in (r1, r2)]
    assert errors[0] < 0.2 and errors[1] < 0.09, errors


def test_the_projectability_guard_fires_on_a_trial_outside_the_window():
    """N = 4 free bands do not hold a sigma = 0.5 cell Gaussian: refused by name, and the same
    run with the guard off records the deficit it refused on."""
    from defuqumat.magnetic.hybrid import StripeTrials
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _small_graphene()
    kw = dict(flux=(1, 2), free=True, trials=StripeTrials(J=1), N=4, kmesh=(1, 1))
    with pytest.raises(ValueError, match="not inside the 4-band window"):
        run_magnetic_ultracell(calc.system, calc.pseudos, None, **kw)
    r = run_magnetic_ultracell(calc.system, calc.pseudos, None, projectability_tol=None,
                               strict=False, **kw)
    assert r.projectability_deficit > 1e-2
    assert _free_projected(1).projectability_deficit < 1e-4


def test_the_reversed_field_fires_the_hermiticity_guard_on_the_projected_basis():
    """scale = -1 puts the field against the momentum shifts the basis was built with: the
    strict run raises, and the recorded residual is orders above the forward run's."""
    with pytest.raises(ValueError, match="not Hermitian"):
        _free_projected(1, scale=-1.0)
    flipped = _free_projected(1, scale=-1.0, strict=False)
    print(f"\nreversed field: herm {flipped.hermiticity.max():.2e}, forward "
          f"{_free_projected(1).hermiticity.max():.2e}")
    assert np.all(flipped.hermiticity > 1e-3)
    assert np.all(_free_projected(1).hermiticity < 1e-10)


@pytest.mark.parametrize("h0", ["eigen", "box"])
def test_zero_flux_with_J_equal_to_N_gives_back_the_bands(h0):
    """p = 0, local only, the 8 pseudo-atomic orbitals of the HGH dataset at N = 8 on the line
    kappa2 = 1/2, where their projections span the 8 bands at every k: the levels are the bands to
    round-off (no window, odd N1, every Delta of the box). At kappa2 = 0 they do not: at Gamma one
    of the 8 bands is orthogonal to every s and p orbital (d_all 1e-16, rank 7, review/mag/u3/zj.log),
    which is the span, not the plumbing. The p_z pair alone (J = 2 < N) lies above the bands."""
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _small_graphene()
    density = _uniform_density(calc)
    kw = dict(flux=(0, 1), N=8, N1=5, w=2, kmesh=(1, 1), kappa2=[0.5], h0=h0, window=None,
              projectability_tol=None)
    r = run_magnetic_ultracell(calc.system, calc.pseudos, density, trials="pao", **kw)
    worst = float(np.max(np.abs(r.levels[0, :8] - r.eigenvalues[0, 0, 0])))
    print(f"\nh0 {h0}: J = N = 8, levels - bands {worst:.2e} Ry, rank {r.rank}, d_min "
          f"{r.d_min.min():.2e}, 1 - p {r.projectability_deficit:.2e}")
    assert r.nbnd == 8 and np.all(r.rank == 8)
    assert worst < 1e-10
    pz = [t for t in __import__("defuqumat.wannier.pao", fromlist=["pao_projections"]).pao_projections(
        calc.pseudos, calc.system.structure,
        np.asarray(calc.system.structure.positions_crystal(calc.system.cell))) if (t.l, t.m) == (1, 1)]
    r2 = run_magnetic_ultracell(calc.system, calc.pseudos, density, trials=pz, **kw)
    assert np.all(r2.levels[0, :2] >= r2.eigenvalues[0, 0, 0, :2] - 1e-10)


def test_the_projected_run_refuses_its_wrong_knobs():
    from defuqumat.magnetic.hybrid import StripeTrials
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _small_graphene()
    with pytest.raises(ValueError, match="needs N"):
        run_magnetic_ultracell(calc.system, calc.pseudos, None, flux=(1, 2), free=True,
                               trials=StripeTrials(J=1))
    with pytest.raises(ValueError, match="nbnd is not"):
        run_magnetic_ultracell(calc.system, calc.pseudos, None, flux=(1, 2), free=True,
                               trials=StripeTrials(J=1), N=8, nbnd=8)
    with pytest.raises(ValueError, match="give trials= with it"):
        run_magnetic_ultracell(calc.system, calc.pseudos, None, flux=(1, 2), free=True, N=8)
    with pytest.raises(ValueError, match="needs N1 >= 15"):
        run_magnetic_ultracell(calc.system, calc.pseudos, None, flux=(1, 2), free=True,
                               trials=StripeTrials(J=1), N=8, N1=10)
