"""Real-time propagation at a frozen potential: the step, the field, the current.

Each test here is a property that holds without a reference: the field is the
derivative of the vector potential, the Taylor step is the exponential to its
order, a state with no field is stationary, the work the field does is the
energy it puts in, an unstable step is refused, and the program a block
compiles serves every k-chunk after the first. The comparisons against the
Kubo sum and the dense hierarchy are in ``tests/regression/test_realtime.py``.

The six that propagate a cell (10 to 35 s each on this workstation) are in the
slow set, by the rule that a test above about five seconds that is not a direct
number against a reference code belongs there; the gate keeps the field, the
step, the refusals and the projection.
"""

import logging
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.realtime.propagate import propagate
from defuqumat.realtime.propagators import TAYLOR4_BOUND, taylor4
from defuqumat.realtime.pulse import (
    FS_TO_AU, Adiabatic, Gaussian, Kick, Ramp, Sin2, field_amplitude, get_pulse)
from defuqumat.scf import Calculation
from defuqumat.system import build_system
from defuqumat.system.kpoints import KPoints

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


def test_the_field_is_minus_the_derivative_of_kappa():
    """``E = -dkappa/dt`` by differentiation, against the closed form, for every shape."""
    t = np.linspace(-50.0, 400.0, 301)
    gauss = Gaussian(amplitude=0.1, omega=0.06, fwhm=80.0, peak=150.0, phase=30.0)
    sigma = 80.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    s = t - 150.0
    envelope = np.exp(-0.5 * (s / sigma) ** 2)
    arg = 0.06 * s + np.pi / 6.0
    closed = -0.1 * envelope * (-s / sigma**2 * np.sin(arg) + 0.06 * np.cos(arg))
    np.testing.assert_allclose(gauss.efield(t)[:, 0], closed, atol=1e-14)
    ramp = Ramp(amplitude=2.0, coefficients=(0.5, 0.01), direction=(0, 0, 1), start_time=10.0)
    later = t[t > 10.5]
    np.testing.assert_allclose(ramp.efield(later)[:, 2],
                               -2.0 * (0.5 + 2 * 0.01 * (later - 10.0)), rtol=1e-13)
    adiabatic = Adiabatic(amplitude=1.0, omega=0.3, eta=0.05, eta_t=10.0)
    assert adiabatic.start == pytest.approx(-200.0)
    assert np.all(Kick(0.01).efield(t) == 0.0)


def test_an_intensity_is_the_field_elk_prints():
    """``E0 = sqrt(I / 3.51e16 W/cm^2)``: 1e12 W/cm^2 is 5.338e-3 a.u., as Elk's ``ppd`` reads."""
    assert field_amplitude(3.5094455205905376e16) == pytest.approx(1.0, rel=1e-14)
    assert field_amplitude(1e12) == pytest.approx(5.33807e-3, rel=1e-5)
    pulse = Sin2.from_intensity(1e12, 1.55, 4)
    period = 2 * np.pi / pulse.omega
    assert pulse.natural_duration == pytest.approx(4 * period)
    # the peak field is E0 to the envelope's own slope: 0.966 E0 over four cycles
    t = np.linspace(0.0, pulse.natural_duration, 4001)
    assert np.abs(pulse.efield(t)).max() == pytest.approx(5.33807e-3, rel=5e-2)
    assert FS_TO_AU == pytest.approx(41.3413746, rel=1e-7)  # units.AU_SEC, 24.189 as
    assert isinstance(get_pulse("kick", strength=1e-3), Kick)


def test_the_taylor_step_is_the_exponential_to_fourth_order():
    """Against ``expm`` on a random Hermitian matrix: the error falls as ``dt^5``.

    And the bound is where it says: inside ``|dt rho| <= 2 sqrt 2`` the step
    never grows a norm, just outside it does.
    """
    rng = np.random.default_rng(1)
    a = rng.normal(size=(12, 12)) + 1j * rng.normal(size=(12, 12))
    h = 0.5 * (a + a.conj().T)
    h = h / np.abs(np.linalg.eigvalsh(h)).max()  # spectrum in [-1, 1]
    psi = rng.normal(size=12) + 1j * rng.normal(size=12)
    psi = psi / np.linalg.norm(psi)
    errors = []
    for dt in (0.1, 0.05):
        exact = scipy.linalg.expm(-1j * dt * h) @ psi
        step = np.asarray(taylor4(lambda v: jnp.asarray(h) @ v, jnp.asarray(psi), dt, 0.0))
        errors.append(np.linalg.norm(step - exact))
    assert errors[0] / errors[1] == pytest.approx(32.0, rel=0.05)
    inside = np.asarray(taylor4(lambda v: jnp.asarray(h) @ v, jnp.asarray(psi),
                                0.99 * TAYLOR4_BOUND, 0.0))
    assert np.linalg.norm(inside) <= 1.0 + 1e-12
    eigvec = np.linalg.eigh(h)[1][:, -1]
    outside = np.asarray(taylor4(lambda v: jnp.asarray(h) @ v, jnp.asarray(eigvec),
                                 1.05 * TAYLOR4_BOUND / np.abs(np.linalg.eigvalsh(h)).max(), 0.0))
    assert np.linalg.norm(outside) > 1.0


@lru_cache(maxsize=2)
def _silicon(pseudo_dir, ecut=6.0):
    """Two-atom silicon at a small cutoff on a 2x2x2 grid, with its dense ground states."""
    import re

    from defuqumat.realtime.dense import dense_hamiltonians

    text = re.sub(r"ecutwfc\s*=\s*[0-9.]+", f"ecutwfc = {ecut}",
                  (CASES / "si2-nosym.in").read_text())
    path = Path("/tmp") / f"defuqumat-realtime-si-{ecut}.in"
    path.write_text(text)
    import equinox as eqx

    system = build_system(read_pw_input(path))
    system = eqx.tree_at(lambda s: s.kpoints, system,
                         KPoints.automatic((2, 2, 2), (0, 0, 0), system.cell))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    v_scf = calculation.potential(calculation.starting_density()).v_scf
    terms = calculation.local_terms(v_scf)
    mask = np.asarray(calculation.basis.planewaves.mask)
    nk, npwx = mask.shape
    states = np.zeros((nk, 4, npwx), dtype=complex)
    for ik in range(nk):
        h = dense_hamiltonians(calculation, terms, ik, (1.0, 0.0, 0.0), 0)[0]
        vectors = np.linalg.eigh(h)[1]
        states[ik][:, np.flatnonzero(mask[ik])] = vectors[:, :4].T
    weights = np.repeat(np.asarray(calculation.system.kpoints.weights)[:, None], 4, axis=1)
    return calculation, jnp.asarray(states), weights, v_scf


@pytest.mark.slow
@pytest.mark.parametrize("potential", ["frozen", "hartree", "hxc"])
def test_with_no_field_the_states_are_stationary(pseudo_dir, potential):
    """``J`` and the energy constant to round-off, the norm to the step's ``y^6/144``.

    Measured on two-atom silicon at 6 Ry, 2x2x2, 200 steps of 0.1 from exact
    dense eigenstates: the energy moves by 2.0e-10 Ha, linearly in time, which
    is the norm the step loses at ``(dt (e - centre))^6/144`` a step on each
    band times its energy. The current is zero by symmetry on
    the whole grid, so its value says nothing; what is asserted of it is that
    it does not move. With the potential updated the start is a fixed point by
    the difference form, which this is the smoke test of rather than the
    evidence for: the potential's informative number is the identity against
    the dense self-consistent solve in ``tests/regression/test_realtime.py``.
    """
    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    result = propagate(calculation, states, weights, v_scf, Kick(0.0), dt=0.1,
                       duration=20.0, block_steps=100, potential=potential)
    assert np.abs(result.energy - result.energy[0]).max() < 1e-9
    assert result.norm_drift < 1e-8
    assert np.abs(result.current - result.current[0]).max() < 1e-10
    assert result.excited < 1e-8


@pytest.mark.slow
@pytest.mark.parametrize("potential", ["frozen", "hxc"])
def test_the_work_done_is_the_energy_gained(pseudo_dir, potential):
    """``Omega int J.E dt = Delta E`` for a pulse, to the trapezoid rule's ``dt^2``.

    Blind to the time unit and to any constant on ``J`` or ``E`` (a step
    under ``2H`` satisfies it); what it sees is a current that is not the
    ``kappa`` derivative of the Hamiltonian the step applies, and a step that
    is not unitary. Measured 2.4e-6 relative on the 12 Ry cell at ``dt = 0.1``.
    With the potential updated the energy is the functional the dynamics
    conserves (:mod:`defuqumat.realtime.selfconsistent`), so a potential that is
    not its derivative fails here: measured at 5e12 W/cm^2 on the 6 Ry cell,
    1.4e-6 at ``dt = 0.1`` and 2.8e-7 at 0.05 with the Hartree and
    exchange-correlation potentials updated, against 1.2e-6 and 2.5e-7 frozen,
    and the pulse deposits 8 per cent less energy with them (0.0538 against
    0.0586 Ha).
    """
    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    pulse = Sin2.from_intensity(1e12, 1.55, 2, (1.0, 0.0, 0.0))
    result = propagate(calculation, states, weights, v_scf, pulse, dt=0.1, block_steps=500,
                       potential=potential)
    assert result.energy_gained > 1e-4, "the pulse did work"
    assert result.work == pytest.approx(result.energy_gained, rel=1e-4)


@pytest.mark.slow
def test_after_a_kick_the_updated_potential_conserves_the_energy(pseudo_dir):
    """The energy after a kick of 0.05/bohr, field off, with the potential updated.

    Two-atom silicon at 6 Ry, 2x2x2, 1000 steps of 0.1 and 2000 of 0.05, the
    Hartree and exchange-correlation potentials updated. Measured, the largest
    ``|E(t) - E(0)|`` in Hartree:

    ===========================  ==========  ==========
    step                         dt = 0.1    dt = 0.05
    ===========================  ==========  ==========
    frozen                       1.42e-7     4.5e-9
    extrapolated (corrector 0)   1.60e-6     4.08e-7
    corrected (corrector 1)      1.58e-7     5.0e-9
    ===========================  ==========  ==========

    The extrapolation alone falls by 3.9 per halving, the ``dt^2`` of a
    second-order potential, and stays at 1.6e-6 over a run four times longer,
    so it is bounded and not secular; with the corrector the drift is the
    frozen one, the norm the Taylor step loses (``dt^5`` a run), so the
    potential's own error is below it. The induced potential reaches 1.4e-2 Ry.
    """
    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    kick = Kick(strength=0.05, direction=(1.0, 0.0, 0.0))
    drift = {}
    for corrector in (0, 1):
        for dt in (0.1, 0.05):
            result = propagate(calculation, states, weights, v_scf, kick, dt=dt, start=0.0,
                               duration=100.0, block_steps=50, potential="hxc",
                               corrector=corrector)
            drift[corrector, dt] = np.abs(result.energy - result.energy[0]).max()
            assert result.extras["potential_change"] > 1e-3, "the potential moved"
    ratio = drift[0, 0.1] / drift[0, 0.05]
    assert 3.0 < ratio < 5.0, ratio
    assert drift[1, 0.1] < 0.2 * drift[0, 0.1]
    assert drift[1, 0.1] < 5e-7


def test_a_step_past_the_stability_bound_is_refused(pseudo_dir):
    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    with pytest.raises(ValueError, match="stability bound"):
        propagate(calculation, states, weights, v_scf, Kick(0.0), dt=5.0, duration=10.0)


@pytest.mark.slow
def test_a_second_k_chunk_compiles_nothing(pseudo_dir):
    """The eager-closure trap, checked the way ``CLAUDE.md`` says: count compilations.

    Every chunk's arrays reach the kept programs as arguments, so after the first
    chunk of a pass has compiled them the others compile none, in the pass that
    bounds the spectrum and in the propagation. The counter is validated on the
    first chunk of the spectrum's pass, which must compile something.
    """
    from defuqumat.realtime import propagate as module

    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    count = [0]

    class Counter(logging.Handler):
        def emit(self, record):
            if "Finished XLA compilation" in record.getMessage() or \
                    "Compiling" in record.getMessage():
                count[0] += 1

    handler = Counter()
    logger = logging.getLogger("jax")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    built = []
    original = module._Chunk.build

    def build(*args, **kwargs):
        built.append(count[0])
        return original(*args, **kwargs)

    module._Chunk.build = build
    jax.config.update("jax_log_compiles", True)
    try:
        propagate(calculation, states, weights, v_scf,
                  Sin2.from_intensity(1e11, 1.55, 1), dt=0.2, block_steps=50,
                  k_batch=2)
        built.append(count[0])
    finally:
        jax.config.update("jax_log_compiles", False)
        module._Chunk.build = original
        logger.removeHandler(handler)
        logger.setLevel(previous)
    # Four chunks of two k-points. The builds are: the setup's first chunk, the
    # spectrum's pass over all four, then the propagation's chunks 1 to 3 (its
    # chunk 0 is the setup's), and the end of the run. Each pass compiles on its
    # first chunk and on no other.
    assert len(built) == 9, built
    assert built[2] > built[1], "the counter saw the spectrum's first chunk compile"
    assert built[2] == built[3] == built[4], built
    assert built[6] == built[7] == built[8], built


@pytest.mark.slow
def test_a_second_self_consistent_run_compiles_nothing(pseudo_dir):
    """With the potential updated, a second run of another pulse compiles nothing.

    The stacked chunks reach the kept block as arguments and the density and
    the potential are closed over arrays of the run, so a second run with a
    pulse of another amplitude on the same k-set reuses every program, which
    the counter is validated against on the first run.
    """
    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    count = [0]

    class Counter(logging.Handler):
        def emit(self, record):
            if "Finished XLA compilation" in record.getMessage() or \
                    "Compiling" in record.getMessage():
                count[0] += 1

    handler = Counter()
    logger = logging.getLogger("jax")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    jax.config.update("jax_log_compiles", True)
    counts = []
    try:
        for intensity in (1e11, 2e11):
            before = count[0]
            propagate(calculation, states, weights, v_scf,
                      Sin2.from_intensity(intensity, 1.55, 1), dt=0.2, block_steps=50,
                      k_batch=2, potential="hxc")
            counts.append(count[0] - before)
    finally:
        jax.config.update("jax_log_compiles", False)
        logger.removeHandler(handler)
        logger.setLevel(previous)
    assert counts[0] > 0, "the counter saw the first run compile"
    assert counts[1] == 0, counts


def test_what_is_refused_is_refused_by_name(pseudo_dir):
    """An ultrasoft dataset with the potential updated, and a wedge passed as the k-set.

    An ultrasoft dataset runs at a frozen potential (``PLAN.md`` P141); what is
    refused is the update, whose density, ``D`` and one-centre terms move.
    """
    from defuqumat.realtime.propagate import require_a_realtime_regime
    from defuqumat.realtime.selfconsistent import require_a_potential_mode
    from defuqumat.workflows.realtime import _kset
    from defuqumat.realtime.pulse import Sin2

    system = build_system(read_pw_input(CASES / "si2-us.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    ultrasoft = Calculation(system, pseudos)
    require_a_realtime_regime(ultrasoft)
    with pytest.raises(NotImplementedError, match="ultrasoft or PAW"):
        require_a_potential_mode(ultrasoft, "hxc", None, None)
    silicon = build_system(read_pw_input(CASES / "si2-symmetric.in"))
    pulse = Sin2.from_intensity(1e11, 1.55, 1)
    with pytest.raises(NotImplementedError, match="symmetry-reduced"):
        _kset(silicon, pulse, silicon.kpoints, None, True)
    _, rotations, group = _kset(silicon, pulse, None, (4, 4, 4), True)
    assert len(rotations) == 8, "a [100] field keeps eight of silicon's 48"
    # the density's maps need the translations, and four of the eight have one
    assert group.nsym == 8 and not group.symmorphic


def _orbit_average_sum(kset_crystal, weights, rotations, field):
    """``sum_k w_k (1/|G|) sum_S f(S k)`` over a wedge, as the current's average is."""
    total = 0.0
    for k, w in zip(kset_crystal, weights):
        total += w * np.mean([field(k @ np.asarray(r).T) for r in rotations])
    return total


def test_a_shifted_grid_reduces_by_the_members_of_the_little_group_it_keeps():
    """Silicon at ``4 4 4 1 1 1``: two of the eight operations of a [100] field, 40 of 64 points.

    No SCF. The identity is the one the current rests on, a periodic scalar of
    no symmetry summed over the whole shifted grid against its group average
    summed over the wedge: with the two operations the grid keeps they agree
    to round-off, and with the whole little group of eight, whose wedge
    ``irreducible_wedge`` builds by skipping the off-grid images, they do not,
    which is what the refusal this replaces was guarding against.
    """
    from defuqumat.realtime.pulse import Kick
    from defuqumat.system.kpoints import grid_stabilizer, irreducible_wedge, monkhorst_pack
    from defuqumat.workflows.realtime import _kset, field_symmetries

    silicon = build_system(read_pw_input(CASES / "si2-shifted-symmetric.in"))
    assert silicon.kpoints.shift == (1, 1, 1)
    counts = {}
    for direction in [(1, 0, 0), (1, 1, 0), (1, 1, 1)]:
        kick = Kick(strength=1.0, direction=direction)
        group = field_symmetries(silicon, kick)
        kset, rotations, kept = _kset(silicon, kick, None, None, True)
        counts[direction] = (group.nsym, kept.nsym, kset.nk)
        assert kset.reduced and kset.shift == (1, 1, 1)
        assert len(rotations) == kept.nsym
    assert counts == {(1, 0, 0): (8, 2, 40), (1, 1, 0): (4, 2, 32), (1, 1, 1): (6, 2, 40)}
    # an unshifted grid keeps the whole little group
    _, _, unshifted = _kset(silicon, Kick(strength=1.0, direction=(1, 0, 0)), None,
                            (4, 4, 4), True)
    assert unshifted.nsym == 8
    # little_group=False is the whole shifted grid, unreduced
    whole, rotations, _ = _kset(silicon, Kick(strength=1.0, direction=(1, 0, 0)), None,
                                None, False)
    assert whole.nk == 64 and rotations is None

    # Fourier orders that are multiples of the grid's 4, which a 4x4x4 sum
    # aliases onto the constant: with orders below 4 every one of these sums is
    # the exact zone average and no wedge, right or wrong, could differ from it.
    rng = np.random.default_rng(7)
    orders = 4 * rng.integers(-1, 2, size=(12, 3)) + rng.integers(0, 2, size=(12, 3))
    phases = rng.uniform(0, 2 * np.pi, 12)
    amplitudes = rng.normal(size=12)

    def field(k):
        return float(np.sum(amplitudes * np.cos(2 * np.pi * orders @ k + phases)))

    points, weights = monkhorst_pack((4, 4, 4), (1, 1, 1))
    reference = sum(w * field(k) for k, w in zip(points, weights))
    assert abs(reference) > 1e-2
    little = field_symmetries(silicon, Kick(strength=1.0, direction=(1, 0, 0)))
    every = little.rotation_array()
    kept = every[grid_stabilizer((4, 4, 4), (1, 1, 1), every)]
    assert len(kept) == 2
    wedge, wedge_weights = irreducible_wedge((4, 4, 4), (1, 1, 1), kept, time_reversal=False)
    assert abs(_orbit_average_sum(wedge, wedge_weights, kept, field) - reference) < 1e-13
    wrong, wrong_weights = irreducible_wedge((4, 4, 4), (1, 1, 1), every, time_reversal=False)
    error = abs(_orbit_average_sum(wrong, wrong_weights, every, field) - reference)
    assert error > 1e-3 * abs(reference), "the whole little group would read differently"


def test_a_reduced_k_set_with_no_grid_is_refused_by_name():
    """The sibling of the shifted-grid case: a wedge with no grid to rebuild the zone from."""
    import equinox as eqx

    from defuqumat.realtime.pulse import Kick
    from defuqumat.workflows.realtime import _kset

    silicon = build_system(read_pw_input(CASES / "si2-symmetric.in"))
    assert silicon.kpoints.reduced
    wedge = KPoints(coords=silicon.kpoints.coords, weights=silicon.kpoints.weights,
                    reduced=True)
    gridless = eqx.tree_at(lambda s: s.kpoints, silicon, wedge)
    with pytest.raises(NotImplementedError, match="carries no grid"):
        _kset(gridless, Kick(strength=1.0, direction=(1, 0, 0)), None, None, True)


def test_what_the_updated_potential_refuses_is_refused_by_name(pseudo_dir):
    """An unknown mode, a reduced k-set without its group, and a checkpoint."""
    from defuqumat.realtime.selfconsistent import require_a_potential_mode

    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    with pytest.raises(ValueError, match="one of"):
        require_a_potential_mode(calculation, "rpa", None, None)
    with pytest.raises(ValueError, match="little group"):
        require_a_potential_mode(calculation, "hxc", np.eye(3)[None], None)
    pulse = Sin2.from_intensity(1e11, 1.55, 1)
    with pytest.raises(NotImplementedError, match="checkpoint"):
        propagate(calculation, states, weights, v_scf, pulse, dt=0.2,
                  potential="hxc", checkpoint="unused.npz")


@pytest.mark.slow
def test_a_checkpoint_resumes_after_the_chunks_it_records(pseudo_dir, tmp_path):
    """A run that finds its own checkpoint returns the recorded current unchanged.

    The unit of restart is a finished k-chunk, which in the frozen mode is
    independent of every other: the file holds the current and the energy
    summed over the chunks done, and a run on the same time grid and k-set
    starts after them. A file from another grid is ignored.
    """
    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    pulse = Sin2.from_intensity(1e11, 1.55, 1)
    path = tmp_path / "rt.npz"
    first = propagate(calculation, states, weights, v_scf, pulse, dt=0.2,
                      block_steps=200, k_batch=4, checkpoint=path)
    saved = np.load(path)
    assert int(saved["done"]) == 2
    again = propagate(calculation, states, weights, v_scf, pulse, dt=0.2,
                      block_steps=200, k_batch=4, checkpoint=path)
    np.testing.assert_array_equal(again.current, first.current)
    other = propagate(calculation, states, weights, v_scf, pulse, dt=0.25,
                      block_steps=200, k_batch=4, checkpoint=path)
    assert other.current.shape != first.current.shape


@pytest.mark.slow
def test_a_checkpoint_of_another_run_is_not_resumed(pseudo_dir, tmp_path):
    """The digest holds the field, so a pulse of twice the amplitude at the same length is another run.

    Found in review: the signature was the grid and the k-count alone, and a
    run at twice the amplitude returned the first one's current bit for bit,
    43 per cent off.
    """
    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    path = tmp_path / "rt.npz"
    weak = propagate(calculation, states, weights, v_scf, Sin2.from_intensity(1e11, 1.55, 1),
                     dt=0.2, block_steps=200, k_batch=4, checkpoint=path)
    strong = propagate(calculation, states, weights, v_scf, Sin2.from_intensity(4e11, 1.55, 1),
                       dt=0.2, block_steps=200, k_batch=4, checkpoint=path)
    clean = propagate(calculation, states, weights, v_scf, Sin2.from_intensity(4e11, 1.55, 1),
                      dt=0.2, block_steps=200, k_batch=4)
    np.testing.assert_array_equal(strong.current, clean.current)
    assert np.abs(strong.current - weak.current).max() > 0.5 * np.abs(weak.current).max()


@pytest.mark.slow
def test_the_chunk_size_is_not_in_the_current(pseudo_dir):
    """``k_batch`` moves the current by round-off and nothing more.

    The centre of the step and the spectrum are taken over every k-point; built
    on the first chunk, the centre was 0.224, 0.117 and 0.096 Ry at ``k_batch``
    1, 4 and 8 and the current moved by 3.9e-7 of its size (found in review).
    ``'fit'`` reads the calculation's own chunk.
    """
    calculation, states, weights, v_scf = _silicon(pseudo_dir)
    pulse = Sin2.from_intensity(1e12, 1.55, 1)
    runs = [propagate(calculation, states, weights, v_scf, pulse, dt=0.2, block_steps=200,
                      k_batch=batch) for batch in (1, 4, "fit")]
    scale = np.abs(runs[0].current).max()
    for other in runs[1:]:
        assert np.abs(other.current - runs[0].current).max() < 1e-12 * scale


def test_a_projection_over_a_broken_period_is_refused():
    """``fourier_component`` needs a step that divides the period, or it leaks the other harmonics."""
    from defuqumat.realtime.orders import fourier_component

    omega = 0.05
    times = np.arange(0.0, 400.0, 0.3)
    with pytest.raises(ValueError, match="does not divide the period"):
        fourier_component(times, np.zeros((len(times), 3)), 3, 3, omega, 0.01)
    period = 2 * np.pi / omega
    times = -period * 10 + np.arange(4001) * period / 400
    signal = np.cos(3 * omega * times)[:, None] * np.ones(3)
    value = fourier_component(times, signal, 0, 3, omega, 0.0)
    np.testing.assert_allclose(value, 0.5, atol=1e-12)


def test_the_cutoff_is_the_end_of_an_unbroken_plateau():
    """An isolated peak past a gap does not move the cutoff; found in review, where it read 31 for 9."""
    from defuqumat.realtime.spectra import HarmonicSpectrum

    orders = np.linspace(0, 40, 4001)
    total = np.full_like(orders, 1e-12)
    for n, height in ((1, 1.0), (3, 0.5), (5, 0.3), (7, 0.2), (9, 0.1), (31, 2e-3)):
        total[np.abs(orders - n) < 0.05] = height
    spectrum = HarmonicSpectrum(frequencies=orders, orders=orders,
                                intensity=total[:, None] * np.ones(3), total=total,
                                omega=1.0, window="none")
    assert spectrum.cutoff(floor=1e-3) == 9


def test_a_spin_spiral_is_refused_by_name():
    """The chunk builds one sphere per k-point; a spiral's components have two."""
    from types import SimpleNamespace

    from defuqumat.realtime.propagate import require_a_realtime_regime

    calculation = SimpleNamespace(spiral=True, gamma_only=False, is_hubbard=False,
                                  magnetic_field=None)
    with pytest.raises(NotImplementedError, match="spin spiral"):
        require_a_realtime_regime(calculation)


_GAMMA_SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0, nosym = .true.
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.01 0.00 0.00
 Si 0.26 0.24 0.25
K_POINTS gamma
"""


def test_gamma_storage_is_refused_by_the_driver_and_taken_off_by_the_workflows():
    """``K_POINTS gamma``: the driver refuses half-sphere states, the workflows never hand it any.

    ``H(k + kappa)`` is complex, so a propagated state is not real and one plane
    wave of each ``(G, -G)`` pair cannot carry it. The refusal stays on
    :func:`require_a_realtime_regime` for a half-sphere ``Calculation`` handed
    to the driver directly (a stub is the whole input, as in
    ``test_sibling_refusals.py``), and the k-set every workflow solves on comes
    back on the whole sphere with its weights untouched. The current against an
    explicit ``k = 0`` is ``tests/regression/test_realtime_gamma.py``.
    """
    import types

    from defuqumat.io.pwin import parse_pw_input
    from defuqumat.realtime.propagate import require_a_realtime_regime
    from defuqumat.realtime.pulse import Sin2
    from defuqumat.workflows.realtime import _kset

    system = build_system(parse_pw_input(_GAMMA_SILICON))
    assert system.kpoints.gamma_only
    stub = types.SimpleNamespace(system=system, gamma_only=True, spiral=None,
                                 is_hubbard=False, magnetic_field=None)
    with pytest.raises(NotImplementedError, match="gamma-only storage"):
        require_a_realtime_regime(stub)
    # the siblings still fire on the same stub
    with pytest.raises(NotImplementedError, match="spin spiral"):
        require_a_realtime_regime(types.SimpleNamespace(
            system=system, gamma_only=False, spiral=object(), is_hubbard=False))
    with pytest.raises(NotImplementedError, match="DFT\\+U"):
        require_a_realtime_regime(types.SimpleNamespace(
            system=system, gamma_only=False, spiral=None, is_hubbard=True))
    require_a_realtime_regime(types.SimpleNamespace(**{**vars(stub), "gamma_only": False}))

    pulse = Sin2.from_intensity(1e11, 1.55, 1)
    for kpoints in (None, system.kpoints):
        kset, rotations, group = _kset(system, pulse, kpoints, None, True)
        assert not kset.gamma_only
        assert rotations is None and group is None
        np.testing.assert_array_equal(np.asarray(kset.coords), np.zeros((1, 3)))
        np.testing.assert_array_equal(np.asarray(kset.weights),
                                      np.asarray(system.kpoints.weights))
        assert kset.spin_normalized == system.kpoints.spin_normalized



def test_a_sawtooth_runs_frozen_and_its_siblings_are_refused():
    """``tefield`` propagates at a frozen potential; updated, and ``lelfield``, are refused.

    The saw-tooth is a static term of the ``v_scf`` the driver is handed, so
    :func:`require_a_realtime_regime` lets it through (it refused it at
    6c5b3a36, "a length gauge and a velocity gauge in one Hamiltonian"); the
    stationarity and the polarizability on a saw-tooth ground state are
    ``tests/regression/test_realtime_sawtooth.py`` (``notes/incompat/I101.md``).
    The potential updated in time with a saw-tooth stays refused by
    :func:`require_a_potential_mode`, and the Berry-phase field by the
    realtime guard. Stubs are the whole input, as in the gamma test above.
    """
    import types

    from defuqumat.realtime.propagate import require_a_realtime_regime
    from defuqumat.realtime.selfconsistent import require_a_potential_mode
    from defuqumat.scf.sawtooth import SawtoothField

    saw = SawtoothField(eamp=jnp.asarray(0.01), edir=2, emaxpos=0.9, eopreg=0.2,
                        dipfield=False)
    plain = dict(gamma_only=False, spiral=None, is_hubbard=False, magnetic_field=None,
                 augmentation=None)
    sawtooth = types.SimpleNamespace(sawtooth=saw, berry_field=None, **plain)
    require_a_realtime_regime(sawtooth)
    require_a_potential_mode(sawtooth, "frozen", None, None)
    for mode in ("hartree", "hxc"):
        with pytest.raises(NotImplementedError, match="saw-tooth field \\(tefield\\)"):
            require_a_potential_mode(sawtooth, mode, None, None)
    with pytest.raises(NotImplementedError, match="lelfield"):
        require_a_realtime_regime(types.SimpleNamespace(
            sawtooth=None, berry_field=object(), **plain))
    with pytest.raises(NotImplementedError, match="spin spiral"):
        require_a_realtime_regime(types.SimpleNamespace(
            sawtooth=saw, berry_field=None, **{**plain, "spiral": object()}))
    with pytest.raises(NotImplementedError, match="DFT\\+U"):
        require_a_realtime_regime(types.SimpleNamespace(
            sawtooth=saw, berry_field=None, **{**plain, "is_hubbard": True}))


def test_a_kick_inside_the_run_is_crossed_at_its_grid_point(pseudo_dir):
    """Where an augmented run crosses a kick after its start, and what it refuses.

    A kick at ``times[n]`` is a jump at the end of step ``n - 1`` of its
    ``strength * direction`` (``notes/incompat/I63.md``); one at or before the
    start is the jump the run starts with, a norm-conserving run crosses
    every kick unchanged, and a kick between two grid points gives an error
    for an ultrasoft dataset, whose crossing needs a grid point to happen at.
    """
    import types

    from defuqumat.realtime.propagate import kick_jumps, time_grid

    system = build_system(read_pw_input(CASES / "si2-us.in"))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    ultrasoft = Calculation(system, pseudos)
    plain = types.SimpleNamespace(augmentation=None)
    times = time_grid(Kick(0.01), 0.05, 2.0, 0.0)[0]

    later = Kick(0.02, (0.0, 3.0, 4.0), time=1.0) + Kick(0.01, (1.0, 0.0, 0.0), time=0.0)
    jumps, at = kick_jumps(ultrasoft, later, times)
    assert jumps.shape == (40, 3) and np.flatnonzero(at).tolist() == [19]
    np.testing.assert_allclose(jumps[19], [0.0, 0.012, 0.016], atol=1e-15)
    assert not jumps[np.arange(40) != 19].any()
    # what the grid says the jump is, at the end of that step
    kappa = Kick(0.02, (0.0, 3.0, 4.0), time=1.0).kappa(times)
    np.testing.assert_allclose(kappa[20] - kappa[19], jumps[19], atol=1e-15)

    assert kick_jumps(ultrasoft, Kick(0.01, time=0.0), times) is None
    assert kick_jumps(ultrasoft, Kick(0.01, time=5.0), times) is None
    assert kick_jumps(plain, later, times) is None
    assert kick_jumps(plain, Kick(0.01, time=1.01), times) is None
    with pytest.raises(NotImplementedError, match="between two points of the time grid"):
        kick_jumps(ultrasoft, Kick(0.01, time=1.01), times)
    with pytest.raises(NotImplementedError, match="between two points of the time grid"):
        kick_jumps(ultrasoft, Sin2.from_intensity(1e11, 1.55, 1) + Kick(0.01, time=1.01),
                   times)
