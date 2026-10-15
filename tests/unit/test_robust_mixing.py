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
    sup = Supervisor(nelec=8.0)
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
    sup = Supervisor(nelec=8.0, window=3, smeared=False, magnetic=False)
    sup.judge(-10.0, 1e-3, 1e-6, 0)
    sup.kept(_state(0.0), _state(1.0), -10.0, 1e-3, 1e-6)
    for accuracy in (9e-4, 8e-4, 9e-4):
        sup.judge(-10.0, accuracy, 1e-6, 0)
        sup.kept(_state(0.0), _state(1.0), -10.0, accuracy, 1e-6)
    assert sup.stalled()
    # The same residuals with the merit falling well beyond its noise are a run
    # going downhill, not a stall.
    moving = Supervisor(nelec=8.0, window=3)
    for k, accuracy in enumerate((1e-3, 9e-4, 8e-4, 9e-4)):
        moving.judge(-10.0 - k, accuracy, 1e-6, 0)
        moving.kept(_state(0.0), _state(1.0), -10.0 - k, accuracy, 1e-6)
    assert not moving.stalled()
    # An insulator without spin takes the restart, then the short history: the
    # LDOS rung needs states at the Fermi level, the magnetic one a moment.
    assert [sup.next_rung() for _ in range(4)] == [
        "restart", "short-history", "hamiltonian", None]
    assert set(RUNGS) == {"restart", "ldos", "magnetization", "short-history",
                          "hamiltonian"}
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
