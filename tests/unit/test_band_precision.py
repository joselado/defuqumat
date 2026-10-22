"""A band side in single precision: the SCF of a norm-conserving cell, and what it refuses.

``Calculation(band_precision='single')`` runs ``H|psi>``, the Davidson work arrays
and the wavefunction store in float32 and keeps the density, the potential, the
mixer and the energies in float64; the small subspace solves run in double. It is a
performance mode and no correctness claim is made in it, so what is checked here is
that it is what it says (32-bit where it should be, 64-bit where it should be), that
it lands within its measured floor of the double answer, that a double run continues
from it to the double answer, and that it refuses what it has not been checked on.
"""

import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.config import DOUBLE, SINGLE, resolve_band_precision
from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import Calculation, run_scf
from defuqumat.system import build_system

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Eight distinct cells run here; their compiled code is not kept between tests."""
    yield
    import jax

    jax.clear_caches()

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"


def _cell(pseudo_dir, name):
    system = build_system(read_pw_input(BENCHMARKS / name))
    return system, tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    return _cell(pseudo_dir, "si-1k.in")


def test_the_default_is_double_and_the_environment_moves_it(monkeypatch):
    monkeypatch.delenv("DEFUQUMAT_BAND_PRECISION", raising=False)
    assert resolve_band_precision() is DOUBLE
    monkeypatch.setenv("DEFUQUMAT_BAND_PRECISION", "single")
    assert resolve_band_precision() is SINGLE
    assert resolve_band_precision("double") is DOUBLE


def test_the_band_side_is_32_bit_and_the_grid_side_64(silicon):
    system, pseudos = silicon
    calculation = Calculation(system, pseudos, band_precision="single")
    hamiltonian = calculation.hamiltonian(
        jnp.zeros((calculation.nspin_mag,) + calculation.basis.dense.grid))[0]
    assert hamiltonian.dtype == SINGLE.complex
    assert hamiltonian.potential.dtype == hamiltonian.kinetic.dtype == SINGLE.real
    assert hamiltonian.projectors.at_k(0).dtype == SINGLE.complex
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-7)
    assert result.converged
    assert result.wavefunctions.dtype == SINGLE.complex
    assert result.density.dtype == DOUBLE.real
    assert np.asarray(result.eigenvalues).dtype == DOUBLE.real


def test_it_lands_within_its_floor_and_a_double_run_continues_from_it(silicon):
    """The floor measured on this cell is about 1e-6 Ry; the continuation reaches double."""
    system, pseudos = silicon
    exact = run_scf(system, pseudos, conv_thr=1e-12)
    single = run_scf(system, pseudos, conv_thr=1e-7, band_precision="single")
    assert abs(single.total_energy - exact.total_energy) < 1e-5
    assert np.abs(np.asarray(single.eigenvalues) - np.asarray(exact.eigenvalues)).max() < 1e-4
    finished = run_scf(system, pseudos, conv_thr=1e-12, starting_from=single)
    assert finished.wavefunctions.dtype == DOUBLE.complex
    assert abs(finished.total_energy - exact.total_energy) < 1e-10


def test_the_floors_follow_the_precision(silicon):
    system, pseudos = silicon
    double = Calculation(system, pseudos)
    assert (double.ethr_floor, double.conv_thr_floor) == (1e-13, 0.0)
    single = Calculation(system, pseudos, band_precision="single")
    assert single.ethr_floor == pytest.approx(
        Calculation.SINGLE_ETHR_FACTOR * SINGLE.eps * system.ecutwfc)
    with pytest.raises(ValueError, match="below what band_precision = 'single' delivers"):
        run_scf(system, pseudos, calculation=single, conv_thr=1e-10)


CASES = Path(__file__).resolve().parents[1] / "data" / "qe"

#: The regimes I119 lifted, each as (input, edits) small enough to build here;
#: the same cells the harness ``notes/incompat/I119-harness/harness.py`` measured.
REGIMES = {
    "ultrasoft": (BENCHMARKS / "si2-us-1k.in", ()),
    "paw": (BENCHMARKS / "si2-paw-1k.in", ()),
    "spinor": (CASES / "h-atom-noncolin.in", (("angle1(1) = 0.0", "angle1(1) = 60.0"),)),
    "spiral": (CASES / "h-chain-spiral-elk.in", ()),
    "dftu": (CASES / "alas-ldau.in", (("4 4 4 0 0 0", "2 2 2 0 0 0"),)),
    "tb09": (BENCHMARKS / "si-1k.in", (("ecutwfc=12.0", "ecutwfc=12.0, input_dft='tb09'"),)),
}


def _edited(pseudo_dir, tmp_path, path, edits):
    text = path.read_text()
    for old, new in edits:
        assert old in text
        text = text.replace(old, new)
    edited = tmp_path / path.name
    edited.write_text(text)
    system = build_system(read_pw_input(edited))
    return system, tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)


def _regime(pseudo_dir, tmp_path, name, cheap=False):
    """The regime's cell; ``cheap`` lowers an augmented cell's cutoffs for a setup-only test."""
    path, edits = REGIMES[name]
    if cheap and name in ("ultrasoft", "paw"):
        edits = edits + (("ecutwfc=20.0, ecutrho=160.0", "ecutwfc=12.0, ecutrho=96.0"),)
    return _edited(pseudo_dir, tmp_path, path, edits)


@pytest.mark.parametrize("name", ["spinor", "spiral", "dftu"])
def test_h_psi_stays_32_bit_for_a_spinor_a_spiral_and_dftu(pseudo_dir, tmp_path, name):
    """The casts I119 added: without them ``H|psi>`` came back complex128.

    The spinor's kinetic energy and potential and the Hubbard projectors were
    left in the cell's precision, and the Davidson work array refused the
    promoted product (``dynamic_update_slice`` of complex128 into complex64).
    """
    system, pseudos = _regime(pseudo_dir, tmp_path, name)
    calculation = Calculation(system, pseudos, band_precision="single")
    hubbard = None
    if calculation.is_hubbard:
        hubbard = calculation.hubbard_terms(calculation.starting_ns())[2]
        assert calculation.band_wfcU().dtype == SINGLE.complex
        assert calculation.wfcU.dtype == DOUBLE.complex  # ns is accumulated in double
    hamiltonian = calculation.hamiltonian(
        jnp.zeros((calculation.nspin_mag,) + calculation.basis.dense.grid), hubbard=hubbard)[0]
    psi = jnp.ones((2, hamiltonian.ndim), SINGLE.complex)
    assert hamiltonian.apply(psi, 0).dtype == SINGLE.complex


def test_what_it_has_not_been_checked_on_is_refused(pseudo_dir, tmp_path):
    """DFT+U on a spinor, through a real cell: DFT+U was measured only alone.

    The rest of the combination rule (a meta-GGA beside an augmented dataset or
    a spiral, DFT+U beside an augmented one) is
    :func:`test_the_guard_admits_an_augmented_spinor_and_nothing_beside_dftu_or_meta`.
    """
    system, pseudos = _edited(pseudo_dir, tmp_path, CASES / "alas-ldau.in",
                              (("4 4 4 0 0 0", "1 1 1 0 0 0"),
                               ("noinv = .true.", "noinv = .true., noncolin = .true.")))
    with pytest.raises(NotImplementedError, match="DFT\\+U .*not checked in single"):
        Calculation(system, pseudos, band_precision="single")
    # and the pair stays accepted in double
    Calculation(system, pseudos)


def test_the_guard_admits_an_augmented_spinor_and_nothing_beside_dftu_or_meta():
    """The guard's combination rule, on the attributes it reads.

    An augmented dataset under a spinor or a spiral was measured (spiral +
    ultrasoft, spiral + PAW, spin-orbit + ultrasoft, ``notes/incompat/I119.md``)
    with the guard patched out, so this pins the branch of the committed guard
    that lets the pair through, without building a cell.
    """
    from types import SimpleNamespace

    def stub(ultrasoft=False, noncolin=False, spiral=False, hubbard=None, meta=False):
        return SimpleNamespace(
            pseudos=(SimpleNamespace(is_ultrasoft=ultrasoft),), noncolin=noncolin,
            spiral=spiral, hubbard=hubbard,
            functional=SimpleNamespace(is_meta=meta, name="tb09"), band_precision=SINGLE,
            system=SimpleNamespace(ecutwfc=20.0),
            SINGLE_ETHR_FACTOR=Calculation.SINGLE_ETHR_FACTOR,
            SINGLE_CONV_THR_FLOOR=Calculation.SINGLE_CONV_THR_FLOOR)

    guard = Calculation._require_band_precision_supported
    for accepted in (stub(ultrasoft=True, noncolin=True, spiral=True),
                     stub(ultrasoft=True, noncolin=True), stub(hubbard=object()),
                     stub(meta=True), stub(spiral=True, noncolin=True)):
        guard(accepted)
        assert accepted.conv_thr_floor == Calculation.SINGLE_CONV_THR_FLOOR
    for refused in (stub(hubbard=object(), noncolin=True), stub(hubbard=object(), ultrasoft=True),
                    stub(meta=True, spiral=True, noncolin=True), stub(meta=True, hubbard=object())):
        with pytest.raises(NotImplementedError, match="not checked in single precision"):
            guard(refused)


@pytest.mark.parametrize("name", sorted(REGIMES))
def test_every_derivative_is_still_refused_on_a_lifted_regime(pseudo_dir, tmp_path, name):
    from defuqumat.forces.energy import reject_single_band_side

    system, pseudos = _regime(pseudo_dir, tmp_path, name, cheap=True)
    single = Calculation(system, pseudos, band_precision="single")
    with pytest.raises(NotImplementedError, match="single precision"):
        reject_single_band_side(single)
    reject_single_band_side(Calculation(system, pseudos))


def test_a_spiral_dE_dq_is_refused_on_a_single_band_side(pseudo_dir, tmp_path):
    """``dE/dq`` differentiates its own frozen energy, outside the shared gate."""
    from defuqumat.forces.spiral import _require_a_differentiable_spiral

    system, pseudos = _regime(pseudo_dir, tmp_path, "spiral")
    with pytest.raises(NotImplementedError, match="single precision"):
        _require_a_differentiable_spiral(Calculation(system, pseudos, band_precision="single"))
    _require_a_differentiable_spiral(Calculation(system, pseudos))


def test_the_shared_gate_still_refuses_a_norm_conserving_single_band_side(silicon):
    from defuqumat.forces.energy import reject_potential_only

    system, pseudos = silicon
    with pytest.raises(NotImplementedError, match="single precision"):
        reject_potential_only(Calculation(system, pseudos, band_precision="single"))
    reject_potential_only(Calculation(system, pseudos))


#: Per regime, at ``conv_thr = 1e-7`` (the floor): the bound on ``|E - E_double|``
#: and on the occupied eigenvalues, each about three times the gap measured
#: (``notes/incompat/I119.md``). TB09's total is not the value of anything and
#: moves to first order in the density, so it is bounded loosely and its
#: eigenvalues carry the check.
FLOORS = {
    "ultrasoft": (5e-6, 2e-4),
    "paw": (5e-6, 2e-4),
    "spinor": (5e-6, 1e-4),
    "spiral": (5e-6, 1e-3),
    "dftu": (5e-6, 3e-4),
    "tb09": (1e-3, 2e-4),
}


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(REGIMES))
def test_each_lifted_regime_lands_near_double_and_a_double_run_finishes_it(
        pseudo_dir, tmp_path, name):
    """I119's check: the single state at the floor, and the double state from it.

    The gap at ``conv_thr = 1e-7`` is the SCF's own error at that threshold
    plus the precision's; a double run started from the single result reaches
    the double state (to 1e-11 Ry on every regime but TB09, whose total moves
    with the density at first order) in one to four iterations.
    """
    system, pseudos = _regime(pseudo_dir, tmp_path, name)
    exact = run_scf(system, pseudos, conv_thr=1e-12, max_iterations=200)
    single = run_scf(system, pseudos, conv_thr=1e-7, band_precision="single",
                     max_iterations=200)
    assert single.converged and single.wavefunctions.dtype == SINGLE.complex
    assert single.density.dtype == DOUBLE.real
    energy_bound, eigenvalue_bound = FLOORS[name]
    assert abs(single.total_energy - exact.total_energy) < energy_bound
    occupied = np.asarray(exact.occupations) > 1e-4 * np.asarray(exact.occupations).max()
    gap = np.abs(np.asarray(single.eigenvalues) - np.asarray(exact.eigenvalues))[occupied]
    assert gap.max() < eigenvalue_bound
    finished = run_scf(system, pseudos, conv_thr=1e-11, starting_from=single,
                       max_iterations=200)
    assert finished.wavefunctions.dtype == DOUBLE.complex
    assert len(finished.history) <= 6
    assert abs(finished.total_energy - exact.total_energy) < (1e-6 if name == "tb09" else 1e-10)


def test_mixed_switches_inside_the_run_and_converges_to_the_double_state(silicon):
    """Single while the density is coarse, double for the last iterations, one mixer.

    A double run restarted from a converged single one took 12 iterations to
    1e-10 on 64 atoms against 9 from scratch, its mixer's history gone; the
    switch inside the run keeps it, and the state it converges to is a double
    one whatever ``conv_thr`` was asked for.
    """
    system, pseudos = silicon
    exact = run_scf(system, pseudos, conv_thr=1e-11)
    mixed = run_scf(system, pseudos, conv_thr=1e-11, band_precision="mixed")
    precisions = [entry["band_precision"] for entry in mixed.history]
    assert precisions[0] == "single" and precisions[-1] == "double"
    assert precisions == sorted(precisions, key=("single", "double").index)
    assert mixed.converged and mixed.wavefunctions.dtype == DOUBLE.complex
    assert abs(mixed.total_energy - exact.total_energy) < 1e-10
    # a loose request still finishes in double rather than stopping in single
    loose = run_scf(system, pseudos, conv_thr=1e-5, band_precision="mixed")
    assert loose.history[-1]["band_precision"] == "double"


def test_becsum_is_accumulated_in_the_grid_precision():
    """float32 projections, float64 weights: ``becsum`` is summed in double.

    Before I119 the product was taken in the projections' dtype, and a spiral
    on an ultrasoft dataset carried a float32 ``becsum`` into the mixer.
    """
    from defuqumat.scf.density import becsum, spinor_becsum

    rng = np.random.default_rng(0)
    vkb = jnp.asarray(rng.normal(size=(1, 4, 3)) + 1j * rng.normal(size=(1, 4, 3)),
                      SINGLE.complex)
    channels = (jnp.asarray([[0, 1, 2]]),)
    psi = jnp.asarray(rng.normal(size=(1, 1, 2, 4)), SINGLE.complex)
    weights = jnp.asarray([[[2.0, 1.0]]], DOUBLE.real)
    assert becsum(psi, vkb, weights, channels)[0].dtype == DOUBLE.real
    spinor = jnp.asarray(rng.normal(size=(1, 2, 8)), SINGLE.complex)
    assert spinor_becsum(spinor, vkb, weights[0], channels)[0].dtype == DOUBLE.complex
