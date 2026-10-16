"""The LDOS preconditioner, ``mixing_mode = 'ldos'`` (Herbst and Levitt).

``eps~ = 1 - chi0~ v`` with ``chi0~ dV = -D dV + D <D, dV>/<D, 1>`` and ``D`` the local
density of states at the Fermi level. What is pinned here, on the five-layer aluminium
slab of ``benchmarks/al-slab.in`` (a metal beside vacuum):

- the operator, against ``eps~`` written again in numpy with a ``D`` that varies in
  space and a field for which the rank-one term is not zero, to round-off;
- the electron count, which the rank-one term is responsible for;
- the solve, at a tolerance tight enough to see an error in it;
- the two limits: a uniform ``D`` is Kerker with ``q_TF^2 = 8 pi D``, and no states at
  the Fermi level is the plain step exactly;
- the clamp to ``D >= 0``, and that only the charge is screened;
- the LDOS itself: it comes out of the density's own pass with the density unchanged,
  it integrates to ``dN/de_F``, it is in the metal and not in the vacuum, and the
  streamed store gives the same one;
- the run-level promises: on an insulator the mode is plain Anderson to the bit, the
  screened modes warn there, and every place that builds a mixer without an LDOS to
  hand it refuses the mode by name.

The measurement that motivates the mode is ``notes/VACUUM-MIXING-NEXT.md``.
"""

from functools import lru_cache
from pathlib import Path
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import defuqumat.scf.mixing as mixing
from tests.tolerances import TOTAL_ENERGY_RY
from defuqumat.scf.mixing import (
    kerker_preconditioner,
    ldos_dielectric,
    ldos_preconditioner,
)

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"
SLAB = BENCHMARKS / "al-slab.in"


@pytest.fixture(autouse=True)
def _clear_caches():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _pieces(path):
    """The dense G-vectors and the cell of an input, built without an SCF."""
    from defuqumat.basis.builder import build_basis
    from defuqumat.io.pwin import read_pw_input
    from defuqumat.system import build_system

    system = build_system(read_pw_input(path))
    return build_basis(system).dense, system.cell


def cell_of(dense):
    """The slab's cell, which every field here lives in."""
    return _pieces(SLAB)[1]


def _slab_ldos(dense, cell, rng):
    """A ``D`` like a metal slab's: large in the middle 15 bohr, zero in the vacuum,
    rippled in the plane so that nothing about it is uniform."""
    grid = dense.grid
    c = float(np.linalg.norm(np.asarray(cell.at)[2]))
    z = (np.arange(grid[2]) + 0.5) / grid[2] * c
    profile = 0.5 * (np.tanh((z - (c / 2 - 7.5)) / 0.6) - np.tanh((z - (c / 2 + 7.5)) / 0.6))
    ripple = 1.0 + 0.3 * _band_limited(dense, rng, scale=0.5).reshape(grid)
    return (0.04 * np.clip(ripple, 0.1, None) * profile[None, None, :]).reshape(-1)


def _numpy_vx(x, dense, cell):
    """The Hartree potential of ``x``, ``8 pi/G^2`` on the dense sphere, in numpy."""
    grid = tuple(dense.grid)
    g2 = np.asarray(dense.kinetic(cell))
    index = np.asarray(dense.fft_index)
    kernel = np.zeros_like(g2)
    kernel[g2 > 1e-12] = 8.0 * np.pi / g2[g2 > 1e-12]
    box = np.fft.fftn(x.reshape(grid)).reshape(-1)
    out = np.zeros_like(box)
    out[index] = box[index] * kernel
    return np.real(np.fft.ifftn(out.reshape(grid))).reshape(-1)


def _numpy_eps(x, d, dense, cell):
    """``eps~ x = x + D v x - D <D, v x>/<D, 1>``, written here and not imported."""
    vx = _numpy_vx(x, dense, cell)
    return x + d * vx - d * np.sum(d * vx) / np.sum(d)


def _band_limited(dense, rng, mean=0.0, scale=None):
    """A real random field on the dense sphere, with a given mean; with ``scale``,
    smooth on that length in 1/bohr (a Gaussian filter in G) and of unit spread."""
    grid = tuple(dense.grid)
    index = np.asarray(dense.fft_index)
    box = np.zeros(int(np.prod(grid)), dtype=complex)
    coefficients = np.fft.fftn(rng.standard_normal(grid)).reshape(-1)[index]
    if scale is not None:
        coefficients = coefficients * np.exp(-np.asarray(dense.kinetic(cell_of(dense)))
                                             / (2 * scale ** 2))
    box[index] = coefficients
    field = np.real(np.fft.ifftn(box.reshape(grid))).reshape(-1)
    field = field - field.mean()
    if scale is not None:
        field = field / np.std(field)
    return field + mean


def _project(field, dense):
    """``field`` with every component outside the dense sphere removed."""
    grid = tuple(dense.grid)
    index = np.asarray(dense.fft_index)
    box = np.fft.fftn(field.reshape(grid)).reshape(-1)
    kept = np.zeros_like(box)
    kept[index] = box[index]
    return np.real(np.fft.ifftn(kept.reshape(grid))).reshape(-1)


def test_the_operator_is_eps_written_independently():
    dense, cell = _pieces(SLAB)
    rng = np.random.default_rng(1)
    d = _slab_ldos(dense, cell, rng)
    x = _band_limited(dense, rng)
    # The rank-one term must fire, or a wrong one would pass: an odd field on a
    # symmetric slab has <D, v x> = 0. This one is random, and the term is a
    # sizeable part of the answer.
    vx = _numpy_vx(x, dense, cell)
    rank_one = d * np.sum(d * vx) / np.sum(d)
    assert np.linalg.norm(rank_one) > 1e-2 * np.linalg.norm(d * vx)
    eps = np.asarray(ldos_dielectric(dense, cell)(jnp.asarray(x), jnp.asarray(d)))
    expected = _numpy_eps(x, d, dense, cell)
    np.testing.assert_allclose(eps, expected, rtol=0, atol=1e-13 * np.max(np.abs(expected)))


def test_eps_keeps_the_electron_count():
    """``integral eps~ f = integral f``: the rank-one term's whole job."""
    dense, cell = _pieces(SLAB)
    rng = np.random.default_rng(2)
    d = _slab_ldos(dense, cell, rng)
    f = _band_limited(dense, rng, mean=0.3)
    eps = np.asarray(ldos_dielectric(dense, cell)(jnp.asarray(f), jnp.asarray(d)))
    assert abs(eps.sum() - f.sum()) < 1e-13 * np.abs(eps).sum()
    # And the check can fail: without the rank-one term the count moves.
    without = f + d * _numpy_vx(f, dense, cell)
    assert abs(without.sum() - f.sum()) > 1e-6 * np.abs(without).sum()


def test_the_solve_solves_eps_at_a_tight_tolerance(monkeypatch):
    monkeypatch.setattr(mixing, "LDOS_TOL", 1e-10)
    dense, cell = _pieces(SLAB)
    rng = np.random.default_rng(3)
    d = _slab_ldos(dense, cell, rng)
    residual = _band_limited(dense, rng)
    shape = (1,) + tuple(dense.grid)
    precondition = ldos_preconditioner(dense, cell, shape, beta=0.7)
    precondition.update_ldos(d)
    x = np.asarray(precondition(residual)) / 0.7
    # The solve runs on the dense sphere's coefficients, so it is eps~ projected
    # onto the sphere that it inverts: D v x has components outside it, which a
    # band-limited residual cannot ask for and the step does not carry.
    error = np.linalg.norm(_project(_numpy_eps(x, d, dense, cell), dense) - residual)
    assert error < 1e-8 * np.linalg.norm(residual)
    iterations, relative = precondition.solves[-1]
    assert relative < 1e-10 and iterations > 1


def test_a_uniform_ldos_is_kerker_at_8_pi_d():
    """The Kerker limit. With a uniform ``D`` the first guess is the exact inverse,
    so this pins the guess and the preconditioner, not the conjugate gradients."""
    dense, cell = _pieces(SLAB)
    rng = np.random.default_rng(4)
    shape = (1,) + tuple(dense.grid)
    d0 = 0.043
    residual = _band_limited(dense, rng)
    precondition = ldos_preconditioner(dense, cell, shape, beta=0.7)
    precondition.update_ldos(np.full(residual.size, d0))
    kerker = kerker_preconditioner(dense, cell, shape, beta=0.7, screening=8 * np.pi * d0)
    got, expected = np.asarray(precondition(residual)), np.asarray(kerker(residual))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12 * np.max(np.abs(expected)))
    assert precondition.solves[-1][0] <= 1


def test_no_states_at_the_fermi_level_is_the_plain_step_exactly():
    dense, cell = _pieces(SLAB)
    rng = np.random.default_rng(5)
    shape = (1,) + tuple(dense.grid)
    residual = np.concatenate([_band_limited(dense, rng), rng.standard_normal(7)])
    precondition = ldos_preconditioner(dense, cell, shape, beta=0.7)
    precondition.update_ldos(np.zeros(int(np.prod(dense.grid))))
    np.testing.assert_array_equal(precondition(residual), 0.7 * residual)
    assert precondition.solves == []


def test_a_negative_ldos_is_clamped_and_counted():
    """The augmentation charge at ``e_F`` is signed; the solve needs ``D >= 0``."""
    dense, cell = _pieces(SLAB)
    rng = np.random.default_rng(6)
    d = _slab_ldos(dense, cell, rng)
    d[: d.size // 10] -= 1e-3
    precondition = ldos_preconditioner(dense, cell, (1,) + tuple(dense.grid), beta=0.7)
    precondition.update_ldos(d)
    assert float(jnp.min(precondition.ldos)) >= 0.0
    removed = np.sum(np.where(d < 0, -d, 0.0)) / np.sum(np.abs(d))
    assert precondition.clamped == pytest.approx(removed, rel=1e-12)


def test_only_the_charge_is_screened():
    dense, cell = _pieces(SLAB)
    rng = np.random.default_rng(7)
    shape = (2,) + tuple(dense.grid)
    up, down = _band_limited(dense, rng), _band_limited(dense, rng)
    precondition = ldos_preconditioner(dense, cell, shape, beta=0.7)
    precondition.update_ldos(_slab_ldos(dense, cell, rng))
    out = np.asarray(precondition(np.concatenate([up, down]))).reshape(2, -1)
    np.testing.assert_allclose(out[0] - out[1], 0.7 * (up - down), rtol=0, atol=1e-15)
    assert np.linalg.norm(out[0] + out[1] - 0.7 * (up + down)) > 1e-3 * np.linalg.norm(up)


def test_in_g_it_is_the_real_space_solve_at_dual_4():
    """``mixing_space = 'g'`` solves over the smooth sphere, which at dual 4 is the
    dense one, so the two layouts take the same step to round-off."""
    from defuqumat.basis.builder import build_basis
    from defuqumat.io.pwin import read_pw_input
    from defuqumat.scf.mixing import SphereLayout, ldos_preconditioner_g
    from defuqumat.system import build_system

    system = build_system(read_pw_input(SLAB))
    basis, cell = build_basis(system), system.cell
    assert basis.ngms == basis.dense.miller.shape[0]
    layout = SphereLayout(basis.dense, basis.ngms, cell, (1,) + tuple(basis.dense.grid))
    rng = np.random.default_rng(8)
    d = _slab_ldos(basis.dense, cell, rng)
    residual = _band_limited(basis.dense, rng)
    real = ldos_preconditioner(basis.dense, cell, layout.shape, beta=0.7)
    sphere = ldos_preconditioner_g(layout, basis.dense, cell, beta=0.7)
    real.update_ldos(d)
    sphere.update_ldos(d)
    expected = np.asarray(real(residual)).reshape(layout.shape)
    got = np.asarray(layout.field(sphere(layout.stored_of(residual.reshape(layout.shape))
                                         .ravel())))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10 * np.max(np.abs(expected)))


# --- the LDOS itself, from an SCF ------------------------------------------------------


@lru_cache(maxsize=1)
def _slab_state():
    from defuqumat import Calculator

    calc = Calculator.from_file(SLAB, pseudo_dir=PSEUDO, announce=False)
    return calc, calc.get_scf()


def _weights(calc, result):
    calculation = calc.calculation
    eigenvalues = np.asarray(result.eigenvalues_by_spin)
    occupations = np.asarray(result.occupations)
    occupations = occupations if occupations.ndim == 3 else occupations[None]
    levels = {"smearing": 0.0, "fermi_energy": float(result.fermi_energy)}
    return calculation, eigenvalues, occupations, calculation.ldos_weights(eigenvalues, levels)


def test_the_ldos_comes_out_of_the_densitys_own_pass():
    calc, result = _slab_state()
    calculation, eigenvalues, occupations, wl = _weights(calc, result)
    psi = result.wavefunctions
    _, rho, ldos = calculation.density_and_ldos(psi, occupations, wl)
    plain = calculation.density(psi, occupations)
    np.testing.assert_allclose(np.asarray(rho), np.asarray(plain), rtol=0,
                               atol=1e-14 * float(np.max(np.abs(plain))))
    # ``dN/de_F`` from the eigenvalues alone: the same Gaussian, summed.
    sigma = max(float(calc.system.degauss), mixing.LDOS_SIGMA_MIN)
    x = (float(result.fermi_energy) - eigenvalues) / sigma
    dos = np.sum(np.asarray(calc.system.kpoints.weights)[None, :, None]
                 * np.exp(-x * x) / (sigma * np.sqrt(np.pi)))
    states = float(jnp.sum(ldos)) * float(calc.system.cell.volume) / ldos.size
    assert states == pytest.approx(dos, rel=1e-10)
    # In the metal and not in the vacuum: the integral alone would pass a
    # transposed or misindexed D. The midplane of this 16-bohr gap holds 4.2e-3 of
    # the maximum, the tail of states at e_F 8 bohr from each surface under a
    # 0.05 Ry Gaussian; a misplaced D would put the metal's value there.
    planes = np.asarray(ldos).reshape(calculation.basis.dense.grid).mean(axis=(0, 1))
    assert planes[0] < 2e-2 * planes.max()
    assert planes[len(planes) // 2] > 0.1 * planes.max()


def test_the_streamed_store_gives_the_same_ldos():
    calc, result = _slab_state()
    calculation, _, occupations, wl = _weights(calc, result)
    whole = calculation.density_and_ldos(result.wavefunctions, occupations, wl)
    streamed = calculation.density_and_ldos(np.asarray(result.wavefunctions),
                                            occupations, wl)
    for a, b in zip(whole[1:], streamed[1:]):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=0,
                                   atol=1e-13 * float(np.max(np.abs(a))))


# --- the run-level promises ------------------------------------------------------------


def _silicon():
    from defuqumat import Calculator

    return Calculator.from_file(BENCHMARKS / "si-1k.in", pseudo_dir=PSEUDO, announce=False)


def test_ldos_on_an_insulator_is_plain_anderson_to_the_bit():
    """Fixed occupations: no states at ``e_F``, so the step is ``beta R`` exactly."""
    plain = _silicon().get_scf()
    with pytest.warns(RuntimeWarning, match="no states at the Fermi level"):
        ldos = _silicon().get_scf(mixing_mode="ldos")
    assert ldos.iterations == plain.iterations
    assert [h["total_energy"] for h in ldos.history] == [
        h["total_energy"] for h in plain.history]


def test_the_screened_modes_warn_on_an_insulator():
    with pytest.warns(RuntimeWarning, match="has no states at the Fermi level"):
        _silicon().get_scf(mixing_mode="TF")


QE_CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

#: ``tests/data/qe/reference.out.al-tetrahedra:202``, Bloechl's tetrahedra on fcc
#: aluminium, ``pw.x`` 7.5 at ``conv_thr = 1e-10``.
AL_TETRAHEDRA_RY = -4.19789860

#: ``tests/data/qe/reference.out.al10-metal-tetra:686``, Kawamura's tetrahedra on the
#: elongated ten-site aluminium cell, ``pw.x`` at ``conv_thr = 1e-8``, 55 iterations.
AL10_TETRA_RY = -41.82613452


def _aluminium(tmp_path, occupations="occupations = 'tetrahedra'"):
    """``al-tetrahedra.in`` (Bloechl's tetrahedra) with its occupations line replaced."""
    from defuqumat import Calculator

    text = (QE_CASES / "al-tetrahedra.in").read_text()
    text = text.replace("occupations = 'tetrahedra'", occupations)
    assert occupations in text
    path = tmp_path / "al.in"
    path.write_text(text)
    return Calculator.from_file(path, pseudo_dir=PSEUDO, announce=False)


@pytest.mark.parametrize("scheme", ["tetrahedra", "tetrahedra_lin", "tetrahedra_opt"])
@pytest.mark.parametrize("nspin", [1, 2])
def test_the_tetrahedron_ldos_integrates_to_the_tetrahedron_dos(tmp_path, scheme, nspin):
    """The tetrahedron LDOS weights, ``d wg / d e_F``, summed over the states, are the
    tetrahedron DOS at ``e_F``. The two sides differentiate different functions (the
    corner weights, against ``_occupied_fraction``), and a ``w_k`` factor on the weights,
    which the smeared branch has and this one must not, would read 2/nk times smaller.
    Two channels at two levels is the constrained-moment branch."""
    from defuqumat.scf.tetrahedra import tetrahedron_dos

    calculation = _aluminium(tmp_path, f"occupations = {scheme!r}").calculation
    nk = calculation.system.kpoints.nk
    rng = np.random.default_rng(1)
    eigenvalues = jnp.asarray(np.sort(rng.uniform(-0.5, 1.0, (nspin, nk, 6)), axis=-1))
    channel_levels = [0.31] if nspin == 1 else [0.27, 0.36]
    levels = ({"fermi_energy": 0.31} if nspin == 1 else
              {"fermi_energy_up": 0.27, "fermi_energy_down": 0.36, "fermi_energy": 0.315})
    weights_at_ef = calculation.ldos_weights(eigenvalues, levels)
    assert weights_at_ef.shape == eigenvalues.shape
    tetra = calculation._tetrahedra
    wk = jnp.asarray(calculation.system.kpoints.weights)
    for spin, level in enumerate(channel_levels):
        dos, _ = tetrahedron_dos(tetra, eigenvalues[spin], wk, jnp.asarray([level]))
        assert float(dos[0]) > 1.0
        assert float(jnp.sum(weights_at_ef[spin])) == pytest.approx(float(dos[0]),
                                                                    rel=1e-10)


def test_the_tetrahedron_ldos_is_built_without_the_occupations_first(tmp_path):
    """``_tetrahedra`` is dropped by every geometry change; ``ldos_weights`` builds it
    when it is absent, the same object ``occupations`` would have built."""
    calculation = _aluminium(tmp_path, "occupations = 'tetrahedra_opt'").calculation
    nk = calculation.system.kpoints.nk
    eigenvalues = jnp.asarray(np.sort(
        np.random.default_rng(2).uniform(-0.5, 1.0, (1, nk, 6)), axis=-1))
    assert getattr(calculation, "_tetrahedra", None) is None
    cold = calculation.ldos_weights(eigenvalues, {"fermi_energy": 0.3})
    calculation.__dict__.pop("_tetrahedra")
    calculation.occupations(eigenvalues)
    warm = calculation.ldos_weights(eigenvalues, {"fermi_energy": 0.3})
    assert np.array_equal(np.asarray(cold), np.asarray(warm))


def test_tetrahedra_under_ldos_reach_pw_x(tmp_path):
    """Bloechl's tetrahedra with the LDOS preconditioner: ``pw.x``'s total energy, and the
    plain mixing loop's fixed point. Refused until I75 for want of a width the method
    does not need."""
    ldos = _aluminium(tmp_path).get_scf(mixing_mode="ldos")
    plain = _aluminium(tmp_path).get_scf()
    assert ldos.converged and plain.converged
    assert ldos.total_energy == pytest.approx(AL_TETRAHEDRA_RY, abs=TOTAL_ENERGY_RY)
    assert ldos.total_energy == pytest.approx(plain.total_energy, abs=1e-8)


@pytest.mark.slow
def test_optimised_tetrahedra_under_ldos_reach_pw_x_on_ten_sites():
    """Kawamura's tetrahedra on the elongated ten-site aluminium cell, where ``pw.x``
    takes 55 iterations: the LDOS preconditioner reaches ``pw.x``'s energy, in fewer."""
    from defuqumat import Calculator

    calc = Calculator.from_file(QE_CASES / "al10-metal-tetra.in", pseudo_dir=PSEUDO,
                                announce=False)
    result = calc.get_scf(mixing_mode="ldos")
    assert result.converged
    assert result.total_energy == pytest.approx(AL10_TETRA_RY, abs=TOTAL_ENERGY_RY)
    assert len(result.history) < 55


def test_newton_krylov_still_refuses_tetrahedra(tmp_path):
    """The LDOS warm-up in front of a Newton solve with tetrahedra stays refused.

    The residual itself takes tetrahedra since I71; what neither I71 nor I75
    measured is the tetrahedron LDOS as the warm-up, so that is the refusal the
    test now reads (retargeted at the integration of the two)."""
    from defuqumat.scf import run_scf

    calc = _aluminium(tmp_path)
    with pytest.raises(NotImplementedError, match="tetrahedron\\s+LDOS as a Newton warm-up"):
        run_scf(calc.system, calc.pseudos, scf_solver="newton-krylov",
                scf_solver_options={"warmup": 1, "warmup_mixing": "ldos"})


def _newton(calc, warmup_mixing, warmup=3, **options):
    from defuqumat.scf import run_scf

    return run_scf(calc.system, calc.pseudos, scf_solver="newton-krylov",
                   scf_solver_options={"warmup": warmup, "warmup_mixing": warmup_mixing,
                                       **options})


def test_the_ldos_warm_up_on_an_insulator_is_the_anderson_warm_up():
    """No states at ``e_F``: every warm-up step is ``beta R``, so the Newton solve starts
    from the anderson warm-up's point, to the bit."""
    ldos = _newton(_silicon(), "ldos", max_iterations=2)
    plain = _newton(_silicon(), "anderson", max_iterations=2)
    assert ldos.solver.history == plain.solver.history
    assert ldos.total_energy == plain.total_energy


@pytest.mark.slow
def test_the_ldos_warm_up_reaches_the_mixer_s_fixed_point(tmp_path):
    """On a metal the warm-up is screened: the Newton solve starts somewhere else than
    after an anderson warm-up, converges on its own, and lands on the mixing loop's
    fixed point. One warm-up step, because three converge this cell before Newton
    takes a step."""
    calc = _aluminium(
        tmp_path, "occupations = 'smearing', smearing = 'gaussian', degauss = 0.05")
    ldos = _newton(calc, "ldos", warmup=1)
    plain = _newton(calc, "anderson", warmup=1)
    mixed = calc.get_scf()
    assert ldos.solver.converged and mixed.converged
    assert ldos.solver.iterations > 0
    assert ldos.solver.history[0]["accuracy"] != plain.solver.history[0]["accuracy"]
    assert ldos.total_energy == pytest.approx(mixed.total_energy, abs=TOTAL_ENERGY_RY)


def test_places_without_an_ldos_refuse_the_mode():
    from defuqumat.response.mixing import ResponseMixer

    # The ultracell builds its LDOS on the box since I77
    # (``tests/unit/test_ultracell_mixing_modes.py``); an augmented one refuses.
    with pytest.raises(ValueError, match="response loop"):
        ResponseMixer("ldos")
