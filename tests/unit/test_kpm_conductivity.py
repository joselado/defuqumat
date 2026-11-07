"""The optical conductivity by two-index Chebyshev moments (P169).

The moments are checked against the dense route, which shares no machinery
with the recursion: every eigenpair of ``Hamiltonian.matrix`` and the velocity
matrix elements of those eigenvectors from
:meth:`~defuqumat.response.velocity.VelocityOperator.matrix_elements` (one
``jvp`` of the whole ``H(k)`` per direction, contracted inside its own k walk).
The conductivity is checked against the pair sum over the same eigenpairs,
and those elements against the band sum
:func:`~defuqumat.workflows.conductivity.run_conductivity` (nbnd = 80 of the 190
states, which the 0.6 Ry window does not reach past). Silicon at 12 Ry on a 2x2x2 mesh, as in
``test_kpm.py``.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import defuqumat  # noqa: F401
from defuqumat import Calculator
from defuqumat.kpm import get_kernel
from defuqumat.kpm.bounds import spectral_interval
from defuqumat.kpm.conductivity import (
    conductivity_moments, run_kpm_conductivity, velocity_tangents,
)
from defuqumat.kpm.moments import live_mask
from defuqumat.response.velocity import VelocityOperator
from defuqumat.workflows.nscf import fixed_density_hamiltonians

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Free the compiled programs after each test (``.claude/rules/memory-design.md``).

    The KPM files compile a recursion per cell and per moment count, and in the
    one-process fast gate they sat in memory beside the files that follow; the
    gate was killed at its 6 GB cap two files later (2026-11-07).
    """
    yield
    import jax

    jax.clear_caches()

SILICON = dict(
    lattice="fcc", a=5.398, pseudos={"Si": "Si.pz-vbc.UPF"},
    atoms=[("Si", (0.0, 0.0, 0.0)), ("Si", (0.25, 0.25, 0.25))],
    ecut=163.3, kmesh=(2, 2, 2),
)
AXES = np.eye(3)


@functools.lru_cache(maxsize=2)
def _silicon(noncolin: bool = False):
    from pathlib import Path

    pseudo = Path(__file__).parents[1] / "data" / "pseudo"
    keywords = dict(SILICON)
    if noncolin:
        keywords["qe"] = {"system": {"noncolin": True}}
    calc = Calculator.from_keywords(pseudo_dir=pseudo, announce=False, conv_thr=1e-10,
                                    **keywords)
    result = calc.get_scf()
    hamiltonians = fixed_density_hamiltonians(calc.calculation, result.density)
    return calc, result, hamiltonians


def _dense_moments(calc, result, hamiltonian, centre, half, n_moments):
    """``sum_k w_k Tr[v_a T_n v_b T_m]`` from every eigenpair and the velocity operator's elements."""
    calculation = calc.calculation
    mask = np.asarray(live_mask(hamiltonian))
    nk, ndim = mask.shape
    states = np.zeros((1, nk, ndim, ndim), dtype=complex)
    values = np.full((nk, ndim), np.nan)
    for ik in range(nk):
        live = mask[ik]
        matrix = np.asarray(hamiltonian.matrix(ik))[np.ix_(live, live)]
        e, u = np.linalg.eigh(matrix)
        values[ik, :live.sum()] = e
        states[0, ik, :live.sum(), live] = u
    v_scf = calculation.potential(jnp.asarray(result.density), 1.0, None).v_scf
    elements = np.asarray(VelocityOperator(calculation, v_scf).matrix_elements(
        jnp.asarray(states)))[:, 0]  # (3, nk, i, j) = <i|v_a|j>
    weights = np.asarray(calculation.system.kpoints.weights)
    n = np.arange(n_moments)
    total = np.zeros((3, 3, n_moments, n_moments), dtype=complex)
    for ik in range(nk):
        count = mask[ik].sum()
        x = (values[ik, :count] - centre) / half
        cheb = np.cos(n[:, None] * np.arccos(x)[None, :])  # (N, i)
        v = elements[:, ik, :count, :count]
        # Tr[v_a T_n v_b T_m] = sum_ij v_a[i, j] T_n(e_j) v_b[j, i] T_m(e_i)
        total += weights[ik] * np.einsum("aij,nj,bji,mi->abnm", v, cheb, v, cheb)
    return total


def test_the_exact_moments_are_the_dense_trace():
    """Every unit vector of the sphere against the eigenpairs and the velocity matrix, to round-off."""
    calc, result, hamiltonians = _silicon()
    centre, half, _ = spectral_interval(hamiltonians)
    n_moments = 12
    moving, tangents = velocity_tangents(calc.calculation, hamiltonians, AXES)
    weights = jnp.asarray(calc.calculation.system.kpoints.weights)
    mu, norms, _ = conductivity_moments(
        hamiltonians[0], moving[0], tuple(t[0] for t in tangents), centre, half,
        jax.random.PRNGKey(0), weights, n_moments, 1, kind="exact")
    dense = _dense_moments(calc, result, hamiltonians[0], centre, half, n_moments)
    scale = np.abs(dense).max()
    np.testing.assert_allclose(np.asarray(mu), dense, rtol=0, atol=1e-9 * scale)
    assert np.all(np.asarray(norms) <= 1.0 + 1e-9)


def _dense_states(calculation, density):
    """Every eigenpair of the dense ``H(k)`` and ``<i|v_a|j>`` from the velocity operator."""
    hamiltonian = fixed_density_hamiltonians(calculation, density)[0]
    mask = np.asarray(live_mask(hamiltonian))
    nk, ndim = mask.shape
    states = np.zeros((1, nk, ndim, ndim), dtype=complex)
    values = []
    for ik in range(nk):
        live = mask[ik]
        e, u = np.linalg.eigh(np.asarray(hamiltonian.matrix(ik))[np.ix_(live, live)])
        values.append(e)
        states[0, ik, :live.sum(), live] = u
    v_scf = calculation.potential(jnp.asarray(density), 1.0, None).v_scf
    elements = np.asarray(VelocityOperator(calculation, v_scf).matrix_elements(
        jnp.asarray(states)))[:, 0]
    return values, elements


def test_the_conductivity_is_the_dense_kubo_greenwood_sum():
    """``sigma'_xx(w)`` from the moments against the pair sum over every eigenpair.

    The reference is the same energy integral, ``(pi / (Omega w)) sum_k w_k
    sum_ij |v_ij|^2 int dE [f(E) - f(E + w)] D(E; e_i) D(E + w; e_j)``, with
    ``D`` the kernel's delta evaluated at the eigenvalues, so it shares the
    kernel and nothing else: not the recursion, not the moment contraction,
    not the grid arithmetic of ``E_i + w_j``. That the velocity elements are
    the band sum's own is :func:`test_the_dense_elements_are_the_band_sums`.
    """
    from defuqumat.kpm.traces import whole_grid_calculation

    calc, result, _ = _silicon()
    level = result.homo + 0.02
    n_moments, delta_e, window = 48, 0.01, 0.8
    kpm = run_kpm_conductivity(calc.system, calc.pseudos, result.density,
                               calculation=calc.calculation, fermi_energy=level,
                               directions=("x",), kernel="jackson", trace="exact",
                               n_moments=n_moments, window=window, delta_e=delta_e)
    calculation = whole_grid_calculation(calc.calculation)
    values, elements = _dense_states(calculation, result.density)
    weights = np.asarray(calculation.system.kpoints.weights)
    volume = float(calculation.system.cell.volume)
    factors = get_kernel("jackson").factors(n_moments)
    coefficients = factors * np.where(np.arange(n_moments) > 0, 2.0, 1.0)
    energies = kpm.emin + delta_e * np.arange(4000)
    energies = energies[energies < level + window + delta_e]
    x = (energies - kpm.centre) / kpm.half_width
    n = np.arange(n_moments)
    cheb_grid = np.cos(n[:, None] * np.arccos(x)[None, :])  # (N, nE)
    measure = np.pi * kpm.half_width * np.sqrt(1.0 - x**2)
    occupation = np.where(energies < level, 1.0, 0.0)
    w = kpm.frequencies[1:]
    shift = np.arange(1, len(kpm.frequencies))
    reference = np.zeros(len(w))
    for ik, e in enumerate(values):
        xe = (e - kpm.centre) / kpm.half_width
        cheb_e = np.cos(n[:, None] * np.arccos(xe)[None, :])  # (N, i)
        delta = (coefficients[:, None] * cheb_e).T @ cheb_grid / measure  # (i, nE)
        count = len(e)
        v2 = np.abs(elements[0, ik, :count, :count]) ** 2
        gamma = delta.T @ v2 @ delta  # (E, E') = sum_ij D(E; e_i) |v_ij|^2 D(E'; e_j)
        for jw, j in enumerate(shift):
            rows = np.arange(len(energies) - j)
            weight = occupation[rows] - occupation[rows + j]
            reference[jw] += weights[ik] * np.sum(weight * gamma[rows, rows + j])
    reference *= np.pi / volume * delta_e / w
    got = kpm.sigma[1:, 0, 0].real
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-8 * np.abs(reference).max())
    assert np.abs(kpm.sigma[1:, 0, 0].imag).max() < 1e-10 * np.abs(reference).max()


def test_the_off_diagonal_part_has_the_band_sums_order():
    """``kubo_greenwood`` on the moments of random Hermitian matrices, against ``<o|v_x|c><c|v_y|o>``.

    The order of ``a`` and ``b`` changes only the antisymmetric imaginary part,
    which a crystal must make finite for a check to see it. On Si2 with one atom
    moved it was at most 3e-6 of the diagonal at every point of a 3x3x3 grid
    (2e-17 at the time-reversal invariant points of 2x2x2), so the order is
    checked here on random matrices, where it is of the size of the rest; that the
    code's moments are ``Tr[v_a T_n v_b T_m]`` is
    :func:`test_the_exact_moments_are_the_dense_trace`.
    """
    from defuqumat.kpm.conductivity import kubo_greenwood

    rng = np.random.default_rng(3)
    size, n_moments, delta_e, nw = 24, 30, 0.02, 40

    def hermitian():
        m = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
        return 0.5 * (m + m.conj().T)

    values = np.sort(rng.uniform(-0.9, 0.9, size))
    v = np.stack([hermitian(), hermitian()])  # in the eigenbasis, <i|v_a|j>
    centre, half, level, volume = 0.0, 1.0, 0.05, 100.0
    n = np.arange(n_moments)
    cheb = np.cos(n[:, None] * np.arccos(values)[None, :])  # (N, i)
    # Tr[v_a T_n v_b T_m] = sum_ij v_a[i, j] T_n(e_j) v_b[j, i] T_m(e_i)
    mu = np.einsum("aij,nj,bji,mi->abnm", v, cheb, v, cheb)
    factors = get_kernel("jackson").factors(n_moments)
    emin = -1.0 + 1e-3
    got = np.asarray(kubo_greenwood(mu, factors, centre, half, volume, level, 0.0, emin,
                                    delta_e, nw))[1:]
    coefficients = factors * np.where(n > 0, 2.0, 1.0)
    energies = emin + delta_e * np.arange(int((level - emin) / delta_e) + nw + 2)
    grid = np.cos(n[:, None] * np.arccos(energies)[None, :]) / (np.pi * np.sqrt(1 - energies**2))
    delta = (coefficients[:, None] * cheb).T @ grid  # (i, E)
    # the band sum's order: z_ab = <i|v_a|j><j|v_b|i> with i at E (occupied), j at E + w
    pairs = np.einsum("aij,bji->abij", v, v)
    gamma = np.einsum("iE,abij,jF->abEF", delta, pairs, delta, optimize=True)
    occupation = (energies < level).astype(float)
    reference = np.zeros((nw - 1, 2, 2), dtype=complex)
    for j in range(1, nw):
        rows = np.arange(len(energies) - j)
        reference[j - 1] = np.einsum("i,abi->ab", occupation[rows] - occupation[rows + j],
                                     gamma[:, :, rows, rows + j])
    reference *= np.pi / volume * delta_e / (delta_e * np.arange(1, nw))[:, None, None]
    scale = np.abs(reference[:, 0, 0]).max()
    assert np.abs(reference[:, 0, 1].imag).max() > 0.1 * scale
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-10 * scale)


def test_the_dense_elements_are_the_band_sums():
    """The pair sum of the dense elements in the band sum's own form is ``run_conductivity``'s ``sigma``."""
    from defuqumat.kpm.traces import whole_grid_calculation

    calc, result, _ = _silicon()
    level = result.homo + 0.02
    eta = 0.08
    frequencies = np.linspace(0.05, 0.6, 12)
    band = calc.get_optical_conductivity(nbnd=80, broadening=eta, intraband=False,
                                         frequencies=frequencies, fermi_energy=level)
    calculation = whole_grid_calculation(calc.calculation)
    values, elements = _dense_states(calculation, result.density)
    weights = np.asarray(calculation.system.kpoints.weights)
    volume = float(calculation.system.cell.volume)

    def lorentzian(x):
        return eta / np.pi / (x * x + eta * eta)

    reference = np.zeros(len(frequencies))
    for ik, e in enumerate(values):
        count = len(e)
        occupied = e < level
        v2 = np.abs(elements[0, ik, :count, :count]) ** 2
        i, j = np.nonzero(occupied[:, None] & ~occupied[None, :])
        gap = e[j] - e[i]
        reference += weights[ik] * np.pi / volume * np.sum(
            v2[i, j] / gap * (lorentzian(frequencies[:, None] - gap)
                              + lorentzian(frequencies[:, None] + gap)), axis=1)
    np.testing.assert_allclose(band.sigma[:, 0, 0].real, reference, rtol=2e-4)


def test_a_second_call_compiles_nothing():
    """The moments are a module-level ``jit``; a second potential reuses the executable."""
    calc, result, hamiltonians = _silicon()
    centre, half, _ = spectral_interval(hamiltonians)
    moving, tangents = velocity_tangents(calc.calculation, hamiltonians, AXES[:1])
    tangents = (moving[0], tuple(t[0] for t in tangents))
    weights = jnp.asarray(calc.calculation.system.kpoints.weights)
    args = (centre, half, jax.random.PRNGKey(0), weights, 8, 2)
    conductivity_moments(hamiltonians[0], *tangents, *args)
    moved = fixed_density_hamiltonians(calc.calculation, 0.999 * result.density)
    before = conductivity_moments._cache_size()
    conductivity_moments(moved[0], *tangents, *args)
    assert conductivity_moments._cache_size() == before


def test_a_spinor_gives_the_scalar_moments():
    """Silicon as a spinor with no magnetization: each state twice, the weights halved, the same trace."""
    moments = []
    for noncolin in (False, True):
        calc, _, hamiltonians = _silicon(noncolin)
        centre, half, _ = (spectral_interval(_silicon(False)[2]) if noncolin
                           else spectral_interval(hamiltonians))
        moving, tangents = velocity_tangents(calc.calculation, hamiltonians, AXES[:2])
        weights = jnp.asarray(calc.calculation.system.kpoints.weights)
        moments.append(np.asarray(conductivity_moments(
            hamiltonians[0], moving[0], tuple(t[0] for t in tangents), centre, half,
            jax.random.PRNGKey(0), weights, 10, 1, kind="exact")[0]))
    scale = np.abs(moments[0]).max()
    np.testing.assert_allclose(moments[1], moments[0], rtol=0, atol=1e-7 * scale)


def test_the_guard_sees_a_spectrum_outside_the_interval():
    """With half the interval, the Chebyshev vectors grow, and the norms the guard reads say so."""
    calc, _, hamiltonians = _silicon()
    centre, half, _ = spectral_interval(hamiltonians)
    moving, tangents = velocity_tangents(calc.calculation, hamiltonians, AXES[:1])
    weights = jnp.asarray(calc.calculation.system.kpoints.weights)
    args = (moving[0], tuple(t[0] for t in tangents))
    inside = np.asarray(conductivity_moments(hamiltonians[0], *args, centre, half,
                                             jax.random.PRNGKey(0), weights, 16, 2)[1])
    outside = np.asarray(conductivity_moments(hamiltonians[0], *args, centre, 0.5 * half,
                                              jax.random.PRNGKey(0), weights, 16, 2)[1])
    assert inside.max() <= 1.0 + 1e-9
    assert outside.max() > 10.0


def test_the_default_level_of_an_insulator_is_in_its_gap():
    """With no level given, the Ritz values of the variate put it between the dense edges."""
    from defuqumat.kpm.traces import whole_grid_calculation

    calc, result, _ = _silicon()
    sigma = run_kpm_conductivity(calc.system, calc.pseudos, result.density,
                                 calculation=calc.calculation, directions=("x",),
                                 broadening=0.02, n_random=2, window=0.1,
                                 reference_energy=result.homo)
    values, _ = _dense_states(whole_grid_calculation(calc.calculation), result.density)
    homo = max(e[3] for e in values)
    lumo = min(e[4] for e in values)
    assert homo < sigma.fermi_energy < lumo, (homo, sigma.fermi_energy, lumo)


def test_the_stochastic_trace_is_unbiased():
    """The mean over seeds of ``R = 2`` estimates against the exact moments, within its own error."""
    calc, _, hamiltonians = _silicon()
    centre, half, _ = spectral_interval(hamiltonians)
    moving, tangents = velocity_tangents(calc.calculation, hamiltonians, AXES[:1])
    tangents = (moving[0], tuple(t[0] for t in tangents))
    weights = jnp.asarray(calc.calculation.system.kpoints.weights)
    n_moments = 8
    exact = np.asarray(conductivity_moments(hamiltonians[0], *tangents, centre, half,
                                            jax.random.PRNGKey(0), weights, n_moments, 1,
                                            kind="exact")[0])[0, 0]
    samples = np.stack([
        np.asarray(conductivity_moments(hamiltonians[0], *tangents, centre, half,
                                        jax.random.PRNGKey(seed), weights, n_moments, 2)[0])[0, 0]
        for seed in range(16)])
    mean = samples.mean(axis=0)
    error = samples.std(axis=0, ddof=1) / np.sqrt(len(samples))
    deviation = np.abs(mean - exact) / np.maximum(error, 1e-12 * np.abs(exact).max())
    assert np.median(deviation) < 2.0, np.median(deviation)
    assert deviation.max() < 5.0, deviation.max()


def test_an_ultrasoft_dataset_is_refused():
    """``H x = e S x`` is not what the moments of ``H`` describe."""
    from pathlib import Path

    pseudo = Path(__file__).parents[1] / "data" / "pseudo"
    keywords = {**SILICON, "pseudos": {"Si": "Si.pz-n-rrkjus_psl.0.1.UPF"},
                "kmesh": (1, 1, 1), "ecutrho": 4 * 163.3}
    calc = Calculator.from_keywords(pseudo_dir=pseudo, announce=False, **keywords)
    density = calc.get_scf().density
    with pytest.raises(NotImplementedError, match="S\\^-1 H"):
        run_kpm_conductivity(calc.system, calc.pseudos, density, fermi_energy=0.5,
                             calculation=calc.calculation, n_moments=8)


def test_the_control_variate_is_unbiased_and_quieter():
    """Eight seeds at R = 2 with and without the variate, against the exact trace's ``sigma_xx``.

    The variate's mean equals the exact spectrum within its standard error, and its
    error over the spectrum is much smaller than the plain trace's: measured 0.0047
    against 0.064 of the peak at filter degree 16. With the left leg projected
    (``<r|P v_a|y>``, the first form) it was 0.038, which a bound of 0.15 catches
    (``PLAN.md`` P169).
    """
    calc, result, _ = _silicon()
    options = dict(calculation=calc.calculation, directions=("x",), n_moments=64,
                   window=0.6, delta_e=0.01, fermi_energy=result.homo + 0.02)
    exact = run_kpm_conductivity(calc.system, calc.pseudos, result.density, trace="exact",
                                 **options).sigma[1:, 0, 0].real
    runs = {cv: np.stack([run_kpm_conductivity(calc.system, calc.pseudos, result.density,
                                               n_random=2, seed=seed, control_variate=cv,
                                               **options).sigma[1:, 0, 0].real
                          for seed in range(8)])
            for cv in (False, True)}
    errors = {cv: np.sqrt(np.mean((s - exact) ** 2, axis=1)).mean() / np.abs(exact).max()
              for cv, s in runs.items()}
    mean = runs[True].mean(axis=0)
    standard = runs[True].std(axis=0, ddof=1) / np.sqrt(8)
    deviation = np.abs(mean - exact) / np.maximum(standard, 1e-12 * np.abs(exact).max())
    assert np.median(deviation) < 2.0, np.median(deviation)
    assert errors[True] < 0.15 * errors[False], errors


# -- the level, the reference energy and the width (P172) ---------------------

ALUMINIUM = dict(
    lattice="fcc", a=4.05, atoms=[("Al", (0.0, 0.0, 0.0))], pseudos={"Al": "Al.pz-vbc.UPF"},
    ecut=204.1, kmesh=(2, 2, 2), smearing=0.27, smearing_type="methfessel-paxton",
)
QUICK = dict(n_moments=8, n_random=1, directions=("x",), window=0.1)


@functools.lru_cache(maxsize=2)
def _aluminium(moment: float | None = None):
    from pathlib import Path

    keywords = dict(ALUMINIUM)
    if moment is not None:
        keywords.update(atoms=[("Al", (0.0, 0.0, 0.0), moment)], spin="collinear",
                        qe={"system": {"tot_magnetization": moment}})
    calc = Calculator.from_keywords(pseudo_dir=Path(__file__).parents[1] / "data" / "pseudo",
                                    announce=False, conv_thr=1e-10, **keywords)
    return calc, calc.get_scf()


def test_a_constrained_moment_takes_the_pair_of_levels():
    """Al under ``tot_magnetization = 0.5``: the Calculator passes both channel levels, and one number raises.

    The Calculator used to pass ``SCFResult.fermi_energy``, which is only the mean of
    the two, and nothing caught it: ``sigma_xx(0)`` 0.2526 against 2.988 with the pair
    on the 4x4x4 mesh (``PLAN.md`` P172).
    """
    calc, scf = _aluminium(0.5)
    assert calc.calculation.two_fermi_energies
    pair = (scf.fermi_energy_up, scf.fermi_energy_down)
    assert abs(pair[0] - pair[1]) > 1e-3
    sigma = calc.get_kpm_conductivity(**QUICK)
    assert sigma.fermi_energy == pytest.approx(pair, abs=1e-12)
    with pytest.raises(ValueError, match="one level per spin channel"):
        run_kpm_conductivity(calc.system, calc.pseudos, scf.density,
                             calculation=calc.calculation, fermi_energy=scf.fermi_energy,
                             **QUICK)


def test_a_metal_takes_the_level_of_its_scf():
    """Al: the Calculator passes the SCF's level, and a direct call with none raises.

    The count of the random vectors put Al 4x4x4 at 0.57935 Ry against its SCF's
    0.51805, and ``sigma_xx(0)`` at 0.0073 against 10.85 from the same moments.
    The Ritz values of a metal show no gap, so a reference energy alone raises too.
    """
    calc, scf = _aluminium()
    assert calc.get_kpm_conductivity(**QUICK).fermi_energy == pytest.approx(
        scf.fermi_energy, abs=1e-12)
    for extra in ({}, {"reference_energy": scf.fermi_energy}):
        with pytest.raises(ValueError, match="fermi_energy"):
            run_kpm_conductivity(calc.system, calc.pseudos, scf.density,
                                 calculation=calc.calculation, **QUICK, **extra)


def test_the_moments_are_counted_at_the_top_of_the_valence_band():
    """Si2 with fixed occupations through the Calculator: ``N`` at the HOMO, and the width reported is the one used.

    ``N`` was chosen at the middle of the occupied window, ``(bottom + centre) / 2``,
    which gave 990 moments where the HOMO needs 652, and the width reported at that
    energy disagreed with ``broadening_for`` at the level by 1.5x.
    """
    from defuqumat.kpm.kernels import moment_count

    calc, result, _ = _silicon()
    sigma = calc.get_kpm_conductivity(broadening=0.02, n_random=1, window=0.1,
                                      directions=("x",))
    expected = moment_count(get_kernel("jackson"), 0.02, sigma.half_width,
                            (result.homo - sigma.centre) / sigma.half_width)
    assert abs(sigma.n_moments - expected) <= 2, (sigma.n_moments, expected)
    target = sigma.broadening_for("jackson")
    assert abs(sigma.transition_broadening - target) < 1e-3 * target, (
        sigma.transition_broadening, target)
    assert sigma.fermi_energy > result.homo


def _dense_values(calculation, density):
    """``[channel][k]``: every eigenvalue of the dense ``H(k)`` of each channel."""
    out = []
    for hamiltonian in fixed_density_hamiltonians(calculation, density):
        mask = np.asarray(live_mask(hamiltonian))
        out.append([np.linalg.eigvalsh(np.asarray(hamiltonian.matrix(ik))[np.ix_(live, live)])
                    for ik, live in enumerate(mask)])
    return out


def test_the_ritz_level_of_two_channels_is_in_the_gap():
    """Si2 at ``nspin = 2`` with one shared level: both channels' Ritz values in one ladder, mid-gap.

    A fixed-occupation run at ``nspin = 2`` needs ``tot_magnetization``, which makes
    it a constrained moment, so the shared level is reached by a direct call with a
    reference energy on a smeared run.
    """
    from pathlib import Path

    from defuqumat.kpm.traces import whole_grid_calculation

    keywords = {**SILICON, "atoms": [("Si", (0.0, 0.0, 0.0), 0.1),
                                     ("Si", (0.25, 0.25, 0.25), 0.1)],
                "spin": "collinear", "smearing": 0.01}
    calc = Calculator.from_keywords(pseudo_dir=Path(__file__).parents[1] / "data" / "pseudo",
                                    announce=False, conv_thr=1e-10, **keywords)
    scf = calc.get_scf()
    assert not calc.calculation.two_fermi_energies
    sigma = run_kpm_conductivity(calc.system, calc.pseudos, scf.density,
                                 calculation=calc.calculation, directions=("x",),
                                 broadening=0.02, n_random=2, window=0.1,
                                 reference_energy=scf.fermi_energy)
    channels = _dense_values(whole_grid_calculation(calc.calculation), scf.density)
    ladders = [np.sort(np.concatenate([up, down])) for up, down in zip(*channels)]
    homo = max(v[7] for v in ladders)
    lumo = min(v[8] for v in ladders)
    assert homo < sigma.fermi_energy < lumo, (homo, sigma.fermi_energy, lumo)


def test_a_hubbard_u_is_refused_before_the_level_is_asked_for():
    """DFT+U: the velocity of the Hubbard orbitals is not checked, so the run refuses, level or none."""
    from pathlib import Path

    calc = Calculator.from_keywords(pseudo_dir=Path(__file__).parents[1] / "data" / "pseudo",
                                    announce=False, hubbard={"U": {"Si-3p": 1.0}},
                                    **{**SILICON, "kmesh": (1, 1, 1)})
    scf = calc.get_scf()
    with pytest.raises(NotImplementedError, match="Hubbard U"):
        calc.get_kpm_conductivity(**QUICK)
    with pytest.raises(NotImplementedError, match="Hubbard U"):
        run_kpm_conductivity(calc.system, calc.pseudos, scf.density, ns=scf.ns,
                             calculation=calc.calculation, **QUICK)


@pytest.mark.parametrize("temperature", [0.0, 0.004])
def test_the_dc_entry_and_a_temperature_are_the_dense_pair_sum(temperature):
    """``sigma[0]`` and ``T > 0`` against the pair sum over every eigenpair, with a level in the valence band.

    The level 0.03 Ry below the top of the valence band makes the DC entry finite.
    At ``T = 0`` it is ``(pi / Omega) sum_k w_k sum_ij |v_ij|^2 D(E_F; e_i) D(E_F; e_j)``;
    at ``T > 0`` the energy integral of ``-f'(E) Gamma(E, E)``, with the Fermi
    function written here as ``1 / (1 + exp((E - mu) / T))``. The occupied rows
    end ``12 T`` above the level, as in the result.
    """
    from defuqumat.kpm.traces import whole_grid_calculation

    calc, result, _ = _silicon()
    level = result.homo - 0.03
    n_moments, delta_e, window = 48, 0.01, 0.4
    kpm = run_kpm_conductivity(calc.system, calc.pseudos, result.density,
                               calculation=calc.calculation, fermi_energy=level,
                               directions=("x",), kernel="jackson", trace="exact",
                               n_moments=n_moments, window=window, delta_e=delta_e,
                               temperature=temperature)
    calculation = whole_grid_calculation(calc.calculation)
    values, elements = _dense_states(calculation, result.density)
    weights = np.asarray(calculation.system.kpoints.weights)
    volume = float(calculation.system.cell.volume)
    coefficients = get_kernel("jackson").factors(n_moments) * np.where(
        np.arange(n_moments) > 0, 2.0, 1.0)
    nw = len(kpm.frequencies)
    n_low = int(np.floor((level + 12.0 * temperature - kpm.emin) / delta_e)) + 2
    energies = kpm.emin + delta_e * np.arange(n_low + nw)
    if temperature > 0.0:
        with np.errstate(over="ignore"):
            occupation = 1.0 / (1.0 + np.exp((energies - level) / temperature))
    else:
        occupation = np.where(energies < level, 1.0, 0.0)
    n = np.arange(n_moments)

    def deltas(e, at):
        x, xe = (at - kpm.centre) / kpm.half_width, (e - kpm.centre) / kpm.half_width
        grid = np.cos(n[:, None] * np.arccos(x)[None, :])
        own = np.cos(n[:, None] * np.arccos(xe)[None, :])
        return (coefficients[:, None] * own).T @ grid / (
            np.pi * kpm.half_width * np.sqrt(1.0 - x**2))  # (i, E)

    rows = np.arange(n_low)
    reference = np.zeros(nw)
    for ik, e in enumerate(values):
        v2 = np.abs(elements[0, ik, :len(e), :len(e)]) ** 2
        delta = deltas(e, energies)
        gamma = delta.T @ v2 @ delta  # (E, E')
        if temperature > 0.0:
            slope = occupation * (1.0 - occupation) / temperature
            reference[0] += weights[ik] * delta_e * np.sum(slope[rows] * gamma[rows, rows])
        else:
            at = deltas(e, np.asarray([level]))[:, 0]
            reference[0] += weights[ik] * at @ v2 @ at
        for j in range(1, nw):
            reference[j] += weights[ik] * np.sum(
                (occupation[rows] - occupation[rows + j]) * gamma[rows, rows + j]) / j
    reference *= np.pi / volume
    got = kpm.sigma[:, 0, 0].real
    assert reference[0] > 0.05 * np.abs(reference).max(), reference[0]
    np.testing.assert_allclose(got, reference, rtol=0, atol=1e-8 * np.abs(reference).max())
