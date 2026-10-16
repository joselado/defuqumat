"""``mixing_mode = 'robust'``: the supervisor's decisions, and the mode end to end.

The decision logic (:class:`~defuqumat.scf.robust.Supervisor`) is pure Python on
scalars and states, so most of it is checked here without an SCF. The end-to-end
checks are on two-atom silicon, where the mode must land on the state plain
Anderson lands on, and on the refusals, each of which is tested to fire.
"""

from pathlib import Path

import numpy as np
import pytest

from defuqumat import Calculator
import defuqumat.scf.robust as robust
from defuqumat.scf.robust import RUNGS, HamiltonianDamping, Supervisor, vacuum_fraction

REPO = Path(__file__).resolve().parents[2]
PSEUDO = REPO / "tests" / "data" / "pseudo"


def _state(value):
    return (np.full(3, float(value)), (), None, None)


def test_a_bulk_cell_has_no_vacuum_and_a_slab_has_it():
    fcc = 7.5 * np.array([[-0.5, 0, 0.5], [0, 0.5, 0.5], [-0.5, 0.5, 0]])
    assert vacuum_fraction(np.zeros((1, 3)), fcc) == 0.0
    slab = np.diag([5.3, 5.3, 47.0])
    layers = np.array([[0, 0, 16.0 + 3.75 * i] for i in range(5)])
    assert vacuum_fraction(layers, slab) == pytest.approx((47.0 - 15.0 - 6.0) / 47.0)


def test_the_first_input_is_kept_and_a_worse_one_is_cut_back():
    sup = Supervisor(nelec=8.0, memory=3)
    assert sup.judge(merit=-10.0, accuracy=1e-2, ethr=1e-6, unsettled=0)
    following = sup.kept(_state(0.0), _state(1.0), -10.0, 1e-2, 1e-6)
    np.testing.assert_allclose(following[0], 1.0)
    # Both the merit and the residual went up: rejected, and the next trial is
    # half way along the same step from the kept input.
    assert not sup.judge(merit=-9.0, accuracy=2e-2, ethr=1e-6, unsettled=0)
    np.testing.assert_allclose(sup.retry()[0], 0.5)
    assert sup.record["reason"] == "rejected"


def test_either_clause_keeps_a_trial():
    sup = Supervisor(nelec=8.0)
    sup.judge(-10.0, 1e-2, 1e-6, 0)
    sup.kept(_state(0.0), _state(1.0), -10.0, 1e-2, 1e-6)
    # The residual fell while the merit rose: kept, Anderson's late steps.
    assert sup.judge(-9.0, 5e-3, 1e-6, 0) and sup.record["reason"] == "residual"
    sup.kept(_state(1.0), _state(2.0), -9.0, 5e-3, 1e-6)
    # The merit fell by more than nelec * ethr while the residual rose above
    # both kept ones: kept.
    assert sup.judge(-10.1, 2e-2, 1e-6, 0) and sup.record["reason"] == "energy"


def test_the_merit_inside_the_eigensolver_noise_or_with_unsettled_bands_keeps_nothing():
    sup = Supervisor(nelec=8.0)
    sup.judge(-10.0, 1e-2, 1e-3, 0)
    sup.kept(_state(0.0), _state(1.0), -10.0, 1e-2, 1e-3)
    # A fall of 1e-3 Ry against a noise of 8 * 1e-3: not evidence.
    assert not sup.judge(-10.001, 2e-2, 1e-3, 0)
    # A fall well outside it, with a band unsettled: not trusted either.
    assert not sup.judge(-11.0, 2e-2, 1e-3, unsettled=3)


def test_a_step_cut_to_its_floor_is_kept_regardless():
    sup = Supervisor(nelec=8.0, max_cuts=2)
    sup.judge(-10.0, 1e-2, 1e-6, 0)
    sup.kept(_state(0.0), _state(1.0), -10.0, 1e-2, 1e-6)
    for _ in range(2):
        assert not sup.judge(-9.0, 2e-2, 1e-6, 0)
        sup.retry()
    assert sup.judge(-9.0, 2e-2, 1e-6, 0) and sup.record["reason"] == "floor"


def test_a_stall_is_detected_and_the_ladder_skips_what_does_not_apply():
    sup = Supervisor(nelec=8.0, window=3, smeared=False)
    sup.judge(-10.0, 1e-3, 1e-6, 0)
    sup.kept(_state(0.0), _state(1.0), -10.0, 1e-3, 1e-6)
    for accuracy in (1.1e-3, 1.2e-3, 1.05e-3):
        sup.judge(-10.0, accuracy, 1e-6, 0)
        sup.kept(_state(0.0), _state(1.0), -10.0, accuracy, 1e-6)
    assert sup.stalled()
    # The same residuals with the merit falling well beyond its noise are a run
    # going downhill, not a stall.
    moving = Supervisor(nelec=8.0, window=3)
    for k, accuracy in enumerate((1e-3, 1.1e-3, 1.2e-3, 1.05e-3)):
        moving.judge(-10.0 - k, accuracy, 1e-6, 0)
        moving.kept(_state(0.0), _state(1.0), -10.0 - k, accuracy, 1e-6)
    assert not moving.stalled()
    # An insulator takes the restart, then the bottom rung: the
    # LDOS rung needs states at the Fermi level.
    assert [sup.next_rung() for _ in range(3)] == ["restart", "hamiltonian", None]
    assert set(RUNGS) == {"restart", "ldos", "hamiltonian"}
    np.testing.assert_allclose(sup.restart_from_best()[0], 0.0)


def test_silicon_lands_where_anderson_lands():
    path = REPO / "benchmarks" / "si-1k.in"
    plain = Calculator.from_file(path, pseudo_dir=PSEUDO, announce=False).get_scf()
    robust = Calculator.from_file(path, pseudo_dir=PSEUDO, announce=False).get_scf(
        mixing_mode="robust")
    assert robust.converged
    assert robust.total_energy == pytest.approx(plain.total_energy, abs=1e-7)
    records = [h["supervisor"] for h in robust.history if "supervisor" in h]
    assert records and records[0]["reason"] == "first"
    # The merit is bounded below by the converged energy, to the noise of the
    # loose thresholds of the first iterations.
    merits = [h["merit"] for h in robust.history if "merit" in h]
    assert min(merits) > robust.total_energy - 1e-4


def test_tetrahedra_are_refused_by_name():
    path = REPO / "tests" / "data" / "qe" / "al-tetrahedra.in"
    calc = Calculator.from_file(path, pseudo_dir=PSEUDO, announce=False)
    with pytest.raises(NotImplementedError, match="tetrahedron"):
        calc.get_scf(mixing_mode="robust", max_iterations=1)


def test_the_hamiltonian_rung_keeps_only_a_lower_merit_and_ends_at_its_floor():
    damping = HamiltonianDamping(alpha_max=0.5, alpha_min=0.1, patience=2)
    h = (np.zeros(2), None, None)
    assert damping.judge(-1.0, 0)
    np.testing.assert_allclose(damping.kept(h, (np.ones(2), None, None), -1.0)[0], 0.5)
    assert not damping.judge(-0.9, 0)
    np.testing.assert_allclose(damping.retry()[0], 0.25)
    assert not damping.judge(-0.9, 0)
    damping.retry()
    damping.retry()
    # At the floor the trial is kept, and twice running is the end.
    assert damping.judge(-0.9, 0) and damping.record["reason"] == "floor"
    damping.kept(h, (np.ones(2), None, None), -0.9)
    assert not damping.at_floor
    assert damping.judge(-0.8, 0)
    damping.kept(h, (np.ones(2), None, None), -0.8)
    assert damping.at_floor


def test_the_bottom_rung_alone_converges_silicon(monkeypatch):
    """Damped Hamiltonian mixing from the second iteration on, every other rung
    skipped: slow, and it must land where Anderson does."""
    monkeypatch.setattr(robust, "RUNGS", ("hamiltonian",))
    monkeypatch.setattr(Supervisor, "stalled", lambda self: True)
    path = REPO / "benchmarks" / "si-1k.in"
    plain = Calculator.from_file(path, pseudo_dir=PSEUDO, announce=False).get_scf()
    result = Calculator.from_file(path, pseudo_dir=PSEUDO, announce=False).get_scf(
        mixing_mode="robust", max_iterations=200)
    assert result.converged
    assert result.total_energy == pytest.approx(plain.total_energy, abs=1e-7)
    rungs = {h["supervisor"]["rung"] for h in result.history if "supervisor" in h}
    assert "hamiltonian" in rungs


def test_the_acceptance_is_measured_against_the_worst_of_the_last_kept_inputs():
    sup = Supervisor(nelec=8.0, memory=3)
    for accuracy in (1e-3, 4e-4, 1e-4):
        sup.judge(-10.0, accuracy, 1e-9, 0)
        sup.kept(_state(0.0), _state(1.0), -10.0, accuracy, 1e-9)
    # Above the last kept residual and below the worst of the last three:
    # Anderson's ordinary non-monotone step, kept.
    assert sup.judge(-10.0, 3e-4, 1e-9, 0) and sup.record["reason"] == "residual"
    # Above all three: cut back.
    assert not sup.judge(-10.0, 2e-3, 1e-9, 0)


def test_a_stall_after_an_action_that_paid_repeats_the_restart():
    sup = Supervisor(nelec=8.0, window=2, smeared=True)
    sup.judge(-10.0, 1e-3, 1e-9, 0)
    sup.kept(_state(0.0), _state(1.0), -10.0, 1e-3, 1e-9)
    assert sup.on_stall() == "restart"
    # The restart bought a tenfold fall in the best residual: restart again.
    sup.judge(-10.0, 1e-4, 1e-9, 0)
    sup.kept(_state(0.0), _state(1.0), -10.0, 5e-5, 1e-9)
    assert sup.on_stall() == "restart"
    # Nothing since: the next rung.
    assert sup.on_stall() == "ldos"


def test_a_smearing_that_is_not_variational_keeps_on_the_residual_alone():
    sup = Supervisor(nelec=8.0, variational=False, smeared=True)
    sup.judge(-10.0, 1e-3, 1e-9, 0)
    sup.kept(_state(0.0), _state(1.0), -10.0, 1e-3, 1e-9)
    # A merit far below with the residual up: a lower bound it is not, so cut.
    assert not sup.judge(-11.0, 1e-2, 1e-9, 0)
    assert "hamiltonian" not in [sup.next_rung() for _ in range(len(RUNGS))]


def test_the_step_length_grows_while_every_trial_is_kept_and_falls_on_a_rejection():
    sup = Supervisor(nelec=8.0, beta=0.1, beta_max=0.8, grow_after=2)
    accuracy = 1e-2
    for _ in range(4):
        sup.judge(-10.0, accuracy, 1e-9, 0)
        sup.kept(_state(0.0), _state(1.0), -10.0, accuracy, 1e-9)
        accuracy *= 0.5
    assert sup.beta == pytest.approx(0.4)
    assert not sup.judge(-10.0, 1.0, 1e-9, 0)
    sup.retry()
    assert sup.beta == pytest.approx(0.2)


def _fake_calculation(**overrides):
    from types import SimpleNamespace

    settings = dict(is_meta=False, berry_field=None, occupations="smearing",
                    magnetic_field=None)
    settings.update(overrides)
    return SimpleNamespace(
        functional=SimpleNamespace(is_meta=settings["is_meta"]),
        berry_field=settings["berry_field"],
        system=SimpleNamespace(occupations=settings["occupations"]),
        magnetic_field=settings["magnetic_field"])


_QUIET = dict(scf_solver="mixing", rotate_moments=False, mixing_beta_mag=None)


def test_the_regimes_the_mode_takes_pass_its_guard():
    from defuqumat.scf.driver import _refuse_robust

    _refuse_robust(_fake_calculation(), **_QUIET)


@pytest.mark.parametrize("calculation,options,words", [
    (dict(is_meta=True), {}, "meta-GGA"),
    (dict(berry_field=object()), {}, "lelfield"),
    (dict(occupations="tetrahedra_opt"), {}, "tetrahedron"),
    ({}, dict(rotate_moments=True), "rotate_moments"),
    ({}, dict(scf_solver="newton-krylov"), "replaces the mixing loop"),
    ({}, dict(mixing_beta_mag=0.1), "magnetization's step"),
])
def test_each_sibling_refusal_still_fires(calculation, options, words):
    from defuqumat.scf.driver import _refuse_robust

    with pytest.raises(NotImplementedError, match=words):
        _refuse_robust(_fake_calculation(**calculation), **{**_QUIET, **options})


def test_a_fading_field_is_still_refused():
    from types import SimpleNamespace

    from defuqumat.scf.driver import _refuse_robust

    field = SimpleNamespace(constraint="none", reducebf=0.5)
    with pytest.raises(NotImplementedError, match="fading"):
        _refuse_robust(_fake_calculation(magnetic_field=field), **_QUIET)


def test_pools_and_a_checkpoint_are_no_longer_arguments_of_the_guard():
    """I76: the clauses are gone, so the guard cannot be handed them at all."""
    import inspect

    from defuqumat.scf.driver import _refuse_robust

    taken = set(inspect.signature(_refuse_robust).parameters)
    assert not taken & {"pooled", "checkpoint_dir", "mixing_from"}


# ------------------------------------------------- the supervisor in a checkpoint


def _exercised_supervisor():
    """A supervisor that has kept, rejected and taken a rung, and a damping."""
    import jax.numpy as jnp

    sup = Supervisor(nelec=8.0, smeared=True, beta=0.3)
    state = (jnp.full(3, 1.0), (np.full(2, 0.5), None), None, None)
    sup.judge(-1.0, 1.0, 1e-6, 0)
    sup.kept(state, (jnp.full(3, 2.0), (np.full(2, 0.7), None), None, None),
             -1.0, 1.0, 1e-6)
    sup.next_rung()
    sup.judge(-0.5, 2.0, 1e-6, 0)
    sup.retry((jnp.full(3, 1.5), (np.full(2, 0.6), None), None, None))
    sup.window = 15
    damping = HamiltonianDamping()
    h = (jnp.zeros(3), None, None)
    damping.judge(-1.0, 0)
    hamiltonian_in = damping.kept(h, (jnp.ones(3), None, None), -1.0)
    return sup, damping, hamiltonian_in


def test_the_supervisor_crosses_the_file_exactly(tmp_path):
    """Every evolving field of both dataclasses comes back, arrays to the bit."""
    import dataclasses

    import jax

    from defuqumat.scf.checkpoint import (_SUPERVISOR_DERIVED, load_supervisor,
                                          save_supervisor)
    from defuqumat.scf.mixing import get_mixer

    sup, damping, hamiltonian_in = _exercised_supervisor()
    mixer = get_mixer("anderson", beta=0.45)
    mixer.condition_limit = 1e5
    path = save_supervisor(tmp_path / "s.npz", sup, damping, hamiltonian_in, mixer, 7)
    saved = load_supervisor(path)
    assert saved["iteration"] == 7
    assert saved["mixer_beta"] == 0.45 and saved["condition_limit"] == 1e5
    names = {f.name for f in dataclasses.fields(sup)} - _SUPERVISOR_DERIVED
    assert set(saved["supervisor"]) == names

    def same(a, b):
        flat_a, tree_a = jax.tree_util.tree_flatten(a)
        flat_b, tree_b = jax.tree_util.tree_flatten(b)
        assert tree_a == tree_b
        for x, y in zip(flat_a, flat_b):
            assert isinstance(x, jax.Array) == isinstance(y, jax.Array)
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))

    for name in names:
        same(getattr(sup, name), saved["supervisor"][name])
    for f in dataclasses.fields(damping):
        same(getattr(damping, f.name), getattr(saved["damping"], f.name))
    same(hamiltonian_in, saved["hamiltonian_in"])


def test_an_unrepresentable_field_raises_rather_than_being_dropped(tmp_path):
    from defuqumat.scf.checkpoint import save_supervisor
    from defuqumat.scf.mixing import get_mixer

    sup, damping, hamiltonian_in = _exercised_supervisor()
    sup._last = {"odd": object()}
    with pytest.raises(NotImplementedError, match="_encode"):
        save_supervisor(tmp_path / "s.npz", sup, damping, hamiltonian_in,
                        get_mixer("anderson"), 1)


def test_a_restore_pours_the_supervisor_back_and_refuses_another_runs(tmp_path):
    """The stamp: a supervisor written at another iteration is not used, and said."""
    from defuqumat.scf.checkpoint import save_supervisor
    from defuqumat.scf.driver import SCF_SUPERVISOR, _restore_supervisor
    from defuqumat.scf.mixing import get_mixer

    sup, damping, hamiltonian_in = _exercised_supervisor()
    sup.rung = 0                      # "restart": no preconditioner to rebuild
    save_supervisor(tmp_path / SCF_SUPERVISOR, sup, damping, hamiltonian_in,
                    get_mixer("anderson", beta=0.45), 7)

    fresh, mixer = Supervisor(nelec=8.0, smeared=True, beta=0.3), get_mixer("anderson")
    restored = _restore_supervisor(fresh, mixer, tmp_path, 7, None, None, None, False)
    assert restored[0] is not None and restored[1] is not None
    assert fresh.scale == sup.scale == 0.5 and fresh.cuts == sup.cuts == 1
    assert fresh.window == 15 and fresh.rung == 0 and mixer.beta == 0.45
    np.testing.assert_array_equal(np.asarray(fresh._kept[0]), np.asarray(sup._kept[0]))

    other = Supervisor(nelec=8.0, smeared=True, beta=0.3)
    with pytest.warns(RuntimeWarning, match="belongs to another run"):
        assert _restore_supervisor(other, get_mixer("anderson"), tmp_path, 9,
                                   None, None, None, False) == (None, None)
    assert other._kept is None and other.scale == 1.0

    assert _restore_supervisor(other, get_mixer("anderson"), tmp_path / "none", 7,
                               None, None, None, False) == (None, None)
    # ``mixing_from`` without a resumed state: the file is there and stamped, and
    # it is still not poured, since this run's input is not the one it kept.
    assert _restore_supervisor(other, get_mixer("anderson"), tmp_path, 0,
                               None, None, None, False) == (None, None)
    assert other._kept is None


def test_a_run_in_another_mode_removes_a_stale_supervisor(tmp_path):
    from defuqumat.scf.driver import SCF_SUPERVISOR, _write_supervisor
    from defuqumat.scf.mixing import get_mixer

    sup, damping, hamiltonian_in = _exercised_supervisor()
    _write_supervisor(tmp_path, (sup, damping, hamiltonian_in), get_mixer("anderson"), 3)
    assert (tmp_path / SCF_SUPERVISOR).exists()
    _write_supervisor(tmp_path, None, get_mixer("anderson"), 4)
    assert not (tmp_path / SCF_SUPERVISOR).exists()
