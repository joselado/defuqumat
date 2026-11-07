"""The KPM density of states of ultrasoft and PAW runs (P171).

The checks share no machinery with the symmetrised operator: the dense overlap
``Hamiltonian.overlap_matrix`` and Hamiltonian ``Hamiltonian.matrix``, whose
generalised eigenvalues come from ``scipy.linalg.eigh(H, S)``, and the SCF's own
Davidson eigenvalues. Si2 at 12 Ry, ``ecutrho`` four times that, on a 2x2x2 mesh,
at ``nspin = 1`` and, for the two-channel test, at ``nspin = 2`` with
``tot_magnetization = 2`` (three cells, two cached at a time).
"""

from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

import defuqumat  # noqa: F401
from defuqumat import Calculator
from defuqumat.kpm import kpm_moments
from defuqumat.kpm.bounds import spectral_interval
from defuqumat.kpm.overlap import apply_power, overlap_power, overlap_powers, symmetrized
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

DATASETS = {"ultrasoft": "Si.pz-n-rrkjus_psl.0.1.UPF", "paw": "Si.pz-n-kjpaw_psl.0.1.UPF"}


@functools.lru_cache(maxsize=2)
def _silicon(kind: str, nspin: int = 1):
    """The converged Si2 cell; ``nspin = 2`` holds two more up electrons than down."""
    from pathlib import Path

    pseudo = Path(__file__).parents[1] / "data" / "pseudo"
    spin = ({"qe": {"system": {"nspin": 2, "tot_magnetization": 2}}} if nspin == 2 else {})
    calc = Calculator.from_keywords(
        pseudo_dir=pseudo, announce=False, conv_thr=1e-10, lattice="fcc", a=5.398,
        pseudos={"Si": DATASETS[kind]},
        atoms=[("Si", (0.0, 0.0, 0.0)), ("Si", (0.25, 0.25, 0.25))],
        ecut=163.3, ecutrho=4 * 163.3, kmesh=(2, 2, 2), **spin)
    with warnings.catch_warnings():
        # five up electrons in Si2 cut the threefold top of the valence band; the
        # trace of the whole spectrum does not see which members are occupied
        warnings.filterwarnings("ignore", message=".*cuts a degenerate multiplet")
        result = calc.get_scf()
    hamiltonians = fixed_density_hamiltonians(calc.calculation, result.density,
                                              becsum=result.becsum)
    return calc, result, hamiltonians


def _chebyshev_trace(spectra, weights, centre, half, n_moments):
    """``sum_k w_k sum_i T_n(e_i')`` of the eigenvalues of each k-point."""
    n = np.arange(n_moments)
    return sum(w * np.cos(n[:, None] * np.arccos((s - centre) / half)[None, :]).sum(axis=1)
               for s, w in zip(spectra, weights))


def _generalised_spectra(hamiltonian):
    mask = np.asarray(hamiltonian.mask)
    spectra = []
    for ik in range(mask.shape[0]):
        live = mask[ik]
        h = np.asarray(hamiltonian.matrix(ik))[np.ix_(live, live)]
        s = np.asarray(hamiltonian.overlap_matrix(ik))[np.ix_(live, live)]
        spectra.append(scipy.linalg.eigh(h, s, eigvals_only=True))
    return spectra


@pytest.mark.parametrize("closeness", [1e-3, 1e-7])
def test_the_half_inverse_holds_with_nearly_parallel_projectors(closeness):
    """``(S^(-1/2))^2 S = 1`` for a random ``B`` with two nearly parallel columns and an indefinite ``Q``.

    Silicon's projectors are well conditioned; this is the case where a form
    through ``(B^dagger B)^(-1/2)`` loses ``cond(B^dagger B)`` in precision.
    """
    rng = np.random.default_rng(1)
    npw, nkb = 60, 6
    table = rng.normal(size=(npw, nkb)) + 1j * rng.normal(size=(npw, nkb))
    table[:, 1] = table[:, 0] + closeness * (rng.normal(size=npw) + 1j * rng.normal(size=npw))
    table /= np.sqrt(2 * npw)  # columns of norm about one
    qq = rng.normal(size=(nkb, nkb))
    qq = 0.15 * (qq + qq.T)  # indefinite, small enough that S stays positive definite
    overlap = np.eye(npw) + table @ qq @ table.conj().T
    assert np.linalg.eigvalsh(overlap).min() > 0
    mask = np.ones(npw, dtype=bool)
    basis, shifted = overlap_power(jnp.asarray(table), jnp.asarray(qq), -0.5)
    v = rng.normal(size=npw) + 1j * rng.normal(size=npw)
    back = apply_power(basis, shifted, apply_power(basis, shifted, jnp.asarray(overlap @ v), mask), mask)
    np.testing.assert_allclose(np.asarray(back), v, rtol=0, atol=1e-11 * np.abs(v).max())


@pytest.mark.parametrize("kind", ["ultrasoft", "paw"])
def test_the_half_inverse_squares_to_the_inverse_overlap(kind):
    """``S^(-1/2) S^(-1/2) S v = v`` with the dense overlap, at every k-point."""
    _, _, hamiltonians = _silicon(kind)
    wrapped = symmetrized(hamiltonians)[0]
    rng = np.random.default_rng(0)
    mask = np.asarray(wrapped.mask)
    for ik in range(mask.shape[0]):
        live = mask[ik]
        v = np.where(live, rng.normal(size=live.size) + 1j * rng.normal(size=live.size), 0.0)
        s = np.asarray(hamiltonians[0].overlap_matrix(ik))
        back = wrapped.half_inverse(wrapped.half_inverse(jnp.asarray(s @ v), ik), ik)
        np.testing.assert_allclose(np.asarray(back)[live], v[live], rtol=0, atol=1e-10)


@pytest.mark.parametrize("kind", ["ultrasoft", "paw"])
def test_the_exact_trace_is_the_generalised_spectrum(kind):
    """Every unit vector against ``sum_i T_n(e_i')`` of ``eigh(H, S)``, to round-off."""
    calc, _, hamiltonians = _silicon(kind)
    wrapped = symmetrized(hamiltonians)
    weights = np.asarray(calc.calculation.system.kpoints.weights)
    centre, half, _ = spectral_interval(wrapped)
    spectra = _generalised_spectra(hamiltonians[0])
    assert min(s.min() for s in spectra) > centre - half
    assert max(s.max() for s in spectra) < centre + half
    n = np.arange(40)
    exact = sum(w * np.cos(n[:, None] * np.arccos((s - centre) / half)[None, :]).sum(axis=1)
                for s, w in zip(spectra, weights))
    moments = kpm_moments(wrapped, weights, centre, half, 40, trace="exact")[0, 0]
    np.testing.assert_allclose(moments, exact, rtol=0, atol=1e-9 * exact[0])


@pytest.mark.parametrize("kind", ["ultrasoft", "paw"])
def test_get_kpm_dos_takes_the_generalised_spectrum(kind):
    """``get_kpm_dos`` on the augmented run: its moments are ``eigh(H, S)``'s, and 8 states lie below the gap.

    The count is read mid-gap between the dense edges, at a moment count whose
    kernel tail across the gap is below the tolerance: the exact trace at N = 1200
    reads 8.0437 for the ultrasoft dataset and 8.0442 for PAW (D22, 2026-11-07), so
    the tolerance is 0.1, where 0.05 left the reading at 87 and 88 per cent of it.
    At N = 40 the same count is 8.596, the Jackson tail of the conduction band
    across the gap.
    """
    calc, _, hamiltonians = _silicon(kind)
    dos = calc.get_kpm_dos(n_moments=40, trace="exact")
    weights = np.asarray(calc.calculation.system.kpoints.weights)
    spectra = _generalised_spectra(hamiltonians[0])
    n = np.arange(40)
    exact = sum(w * np.cos(n[:, None] * np.arccos((s - dos.centre) / dos.half_width)[None, :]).sum(axis=1)
                for s, w in zip(spectra, weights))
    np.testing.assert_allclose(dos.samples[0, 0], exact, rtol=0, atol=1e-9 * exact[0])
    top = max(s[3] for s in spectra)
    bottom = min(s[4] for s in spectra)
    fine = calc.get_kpm_dos(n_moments=1200, trace="exact")
    assert abs(fine.count_at(0.5 * (top + bottom)) - 8.0) < 0.1, (top, bottom)


@pytest.mark.parametrize("kind", ["ultrasoft", "paw"])
def test_the_projected_moments_are_the_s_projections(kind):
    """``get_kpm_pdos`` against ``sum_n |<phi|S|x_n>|^2 T_n(e_n')`` from ``eigh(H, S)`` and the orbitals."""
    from defuqumat.kpm.traces import whole_grid_calculation
    from defuqumat.projwfc.projections import atomic_orbitals

    calc, result, _ = _silicon(kind)
    pdos = calc.get_kpm_pdos(n_moments=30)
    calculation = whole_grid_calculation(calc.calculation)
    hamiltonian = fixed_density_hamiltonians(calculation, result.density,
                                             becsum=result.becsum)[0]
    orbitals = np.asarray(atomic_orbitals(calculation))  # (nk, ndim, r)
    weights = np.asarray(calculation.system.kpoints.weights)
    mask = np.asarray(hamiltonian.mask)
    n = np.arange(30)
    reference = 0.0
    for ik in range(mask.shape[0]):
        live = mask[ik]
        h = np.asarray(hamiltonian.matrix(ik))[np.ix_(live, live)]
        s = np.asarray(hamiltonian.overlap_matrix(ik))[np.ix_(live, live)]
        values, vectors = scipy.linalg.eigh(h, s)
        projections = orbitals[ik][live].conj().T @ s @ vectors  # (r, n)
        x = (values - pdos.centre) / pdos.half_width
        reference = reference + weights[ik] * np.abs(projections) ** 2 @ np.cos(
            np.arccos(x)[:, None] * n[None, :])
    np.testing.assert_allclose(pdos.moments[0], reference, rtol=0,
                               atol=1e-9 * np.abs(reference).max())


def test_the_other_traces_still_refuse_an_augmented_dataset():
    """The LDOS and the conductivity are not in the symmetrised form, and the refusal says which traces are.

    The message used to say the projected DOS "is not implemented" while
    ``get_kpm_pdos`` runs on these datasets to 8e-15; it now names the two traces
    that run and the three that do not.
    """
    calc, _, _ = _silicon("ultrasoft")
    with pytest.raises(NotImplementedError, match="S\\^-1 H") as refused:
        calc.get_kpm_ldos([(0.1, 0.1, 0.1)], n_moments=20)
    message = str(refused.value)
    assert "local DOS, the unfolding and the conductivity" in message, message
    assert "projected and local" not in message, message
    assert "(get_kpm_dos, get_kpm_pdos) run on" in message, message
    with pytest.raises(NotImplementedError, match="S\\^-1 H"):
        calc.get_kpm_conductivity(n_moments=8, n_random=1)


SPIRAL = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1,
  ecutwfc = 10.0, ecutrho = 40.0, noncolin = .true., nosym = .true.,
  starting_magnetization(1) = 0.5, angle1(1) = 90.0,
  spiral_q(1) = 0.0, spiral_q(2) = 0.0, spiral_q(3) = 0.25
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 {dataset}
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


@pytest.mark.parametrize("dataset", [DATASETS["ultrasoft"], "Si.pz-vbc.UPF"])
def test_a_spin_spiral_is_refused_as_a_spiral(dataset):
    """An augmented spiral is told about the spiral, as a norm-conserving one is, not about ``S``.

    The overlap message says the DOS runs on ``S^(-1/2) H S^(-1/2)``, which
    contradicts the call just made; the spiral's two spinor components on two
    spheres are what refuses it whatever the dataset. At the starting density,
    so no SCF is run.
    """
    from pathlib import Path

    from defuqumat.kpm.dos import run_kpm_dos

    pseudo = Path(__file__).parents[1] / "data" / "pseudo"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc = Calculator.from_text(SPIRAL.format(dataset=dataset), pseudo, announce=False)
    calculation = calc.calculation
    assert calculation.spiral
    with pytest.raises(NotImplementedError, match="spin spiral") as refused:
        run_kpm_dos(calc.system, calc.pseudos, calculation.starting_density(),
                    calculation=calculation, fermi_energy=0.5, n_moments=8, n_random=1,
                    control_variate=False)
    assert "S^-1 H" not in str(refused.value)


def test_the_overlap_powers_do_not_depend_on_the_k_chunk():
    """``U`` and ``M^(-1/2) - 1`` with one k-point in flight equal the whole-axis ones."""
    _, _, hamiltonians = _silicon("ultrasoft")
    whole = overlap_powers(hamiltonians[0], -0.5, k_batch=None)
    chunked = overlap_powers(hamiltonians[0], -0.5, k_batch=1)
    for a, b in zip(whole, chunked):
        assert a.shape == b.shape
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=0, atol=1e-12)


def test_the_variate_is_unbiased_on_the_symmetrised_operator():
    """The variate on ``S^(-1/2) H S^(-1/2)``: 8 seeds at R = 4, the mean within 4.5 standard errors of the exact trace.

    The orbitals are the plain ``phi``, not ``S^(1/2) phi``: the variate is
    unbiased for any orthonormal set, so the wrapped operator needs nothing new,
    and this is the check that it got nothing wrong. One interval for every
    seed, so the moments of all runs are of the same operator. The samples'
    ``mu_0`` spread shows the correction is applied (the plain trace's is exact).
    """
    from defuqumat.kpm.control import FILTER_CUT, FILTER_DEGREE
    from defuqumat.kpm.dos import variate_corrections

    calc, result, hamiltonians = _silicon("ultrasoft")
    wrapped = symmetrized(hamiltonians)
    weights = np.asarray(calc.calculation.system.kpoints.weights)
    centre, half, _ = spectral_interval(wrapped)
    n_moments, n_random = 40, 4
    exact = kpm_moments(wrapped, weights, centre, half, n_moments, trace="exact")[0, 0]
    cut = float(result.homo) + FILTER_CUT
    samples = np.concatenate([
        (kpm_moments(wrapped, weights, centre, half, n_moments, n_random=n_random, seed=seed)
         + variate_corrections(calc.calculation, wrapped, weights, centre, half, n_moments,
                               n_random, seed, "stochastic", cut, FILTER_DEGREE))[0]
        for seed in range(8)])  # (32, N)
    assert samples[:, 0].std() > 1e-3 * exact[0], samples[:, 0]
    mean = samples.mean(axis=0)
    spread = samples.std(axis=0, ddof=1) / np.sqrt(samples.shape[0])
    window = spread > 1e-9 * np.abs(exact).max()
    assert window.any()
    z = np.abs(mean - exact)[window] / spread[window]
    assert z.max() < 4.5, (z.max(), mean[0], spread[0], exact[0])


def test_two_augmented_channels_take_their_own_generalised_spectra():
    """``nspin = 2`` ultrasoft: each channel's exact moments are ``eigh(H_s, S)``'s, and the channels differ.

    ``S`` is the same in both channels and ``H`` is not; a channel that took the
    other's Hamiltonian, or a wrapper built once for both, would match one
    channel only, which the difference between the channels makes visible.
    """
    calc, _, hamiltonians = _silicon("ultrasoft", 2)
    assert len(hamiltonians) == 2 and all(h.has_overlap for h in hamiltonians)
    wrapped = symmetrized(hamiltonians)
    weights = np.asarray(calc.calculation.system.kpoints.weights)
    centre, half, _ = spectral_interval(wrapped)
    moments = kpm_moments(wrapped, weights, centre, half, 40, trace="exact")[:, 0]  # (2, N)
    for channel, hamiltonian in enumerate(hamiltonians):
        exact = _chebyshev_trace(_generalised_spectra(hamiltonian), weights, centre, half, 40)
        np.testing.assert_allclose(moments[channel], exact, rtol=0, atol=1e-9 * exact[0])
    assert np.abs(moments[0] - moments[1]).max() > 1e-3 * moments[0, 0]
    dos = calc.get_kpm_dos(n_moments=40, trace="exact")
    assert dos.nspin == 2 and dos.samples.shape == (2, 1, 40)
    again = kpm_moments(wrapped, weights, dos.centre, dos.half_width, 40, trace="exact")
    np.testing.assert_allclose(dos.samples, again, rtol=0, atol=1e-12 * again[0, 0, 0])
