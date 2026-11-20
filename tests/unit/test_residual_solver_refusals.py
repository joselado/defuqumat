"""What a residual SCF solver refuses before it starts, and what it still runs.

A residual solver (``scf_solver = 'newton-krylov'``) finds the fixed point before
the mixing loop runs, and the loop then takes one hand-over iteration from it.
Two things the loop does at its first iterations therefore never reach the
solve, and each is refused by name rather than applied at the wrong moment:

* ``starting_ns_eigenvalue`` (``pw.x``'s ``ns_adj``), which the loop applies at
  iteration 1 of a fresh run. The hand-over iteration is iteration 1, so the
  request was imposed on the *converged* occupation matrix and the run reported
  convergence with it (audit 2026-11-20c #3).

Each refusal is checked to fire before the solve, and the regimes next to it are
checked to run: the same request on the mixing loop, and a residual solver on a
run that is given its starting state.
"""

from pathlib import Path

import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.scf.driver import Calculation, run_scf

pytestmark = pytest.mark.unit

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"

#: One Newton step and one hand-over iteration: these tests check what is let
#: through, not where it converges.
_ONE_STEP = {"max_iterations": 1, "gmres_maxiter": 2}


@pytest.fixture(scope="module")
def steered_nickel(pseudo_dir):
    """``ni-u-unstable.in`` with one minority eigenvalue of ``ns`` asked for.

    Built once: ``run_scf`` does not change the ``Calculation`` it is handed.
    """
    text = (BENCHMARKS / "ni-u-unstable.in").read_text().replace(
        "nosym = .true.", "nosym = .true.\n    starting_ns_eigenvalue(5,2,1) = 0.5", 1)
    calculator = Calculator.from_text(text, pseudo_dir, announce=False)
    calculation = Calculation(calculator.system, calculator.pseudos)
    assert calculation.hubbard.starting_ns == {(0, 1, 4): 0.5}
    return calculator.system, calculator.pseudos, calculation


def test_starting_ns_eigenvalue_with_a_residual_solver_is_refused(steered_nickel):
    """Refused before the solve, with the routes that work named.

    On ``419b5fb7`` this run converged at the hand-over iteration with the
    requested eigenvalue 0.5 in ``SCFResult.ns``, where the states' own
    occupation matrix has 0.999922, and a total 2.76e-2 Ry below the same
    Newton run without the request.
    """
    system, pseudos, calculation = steered_nickel
    with pytest.raises(NotImplementedError, match="starting_ns_eigenvalue") as raised:
        run_scf(system, pseudos, calculation=calculation, max_iterations=1,
                scf_solver="newton-krylov", scf_solver_options=dict(_ONE_STEP),
                verbose=False)
    message = str(raised.value)
    assert "scf_solver='mixing'" in message and "starting_from=" in message


def test_the_same_request_runs_on_the_mixing_loop(steered_nickel):
    """The loop applies it at iteration 1, as ``electrons_scf`` does."""
    system, pseudos, calculation = steered_nickel
    result = run_scf(system, pseudos, calculation=calculation, max_iterations=1,
                     verbose=False)
    assert result.iterations == 1
    # The requested level is in the minority matrix the first iteration hands on.
    levels = np.linalg.eigvalsh(np.asarray(result.ns)[1, 0])
    assert np.min(np.abs(levels - 0.5)) < 1e-12


def test_a_residual_solver_given_its_starting_ns_runs(steered_nickel):
    """With ``starting_ns`` the loop's ``ns_adj`` gate is closed, so nothing is refused."""
    system, pseudos, calculation = steered_nickel
    result = run_scf(system, pseudos, calculation=calculation, max_iterations=1,
                     starting_ns=calculation.starting_ns(),
                     scf_solver="newton-krylov", scf_solver_options=dict(_ONE_STEP),
                     verbose=False)
    assert result.solver is not None
