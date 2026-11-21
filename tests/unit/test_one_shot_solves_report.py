"""A one-shot solve that leaves bands unsettled says so (audit 2026-11-20b #23).

The NSCF route has read the Davidson step counts since the 1H-NbSe2 stall, and
two other solves that start from random vectors at a tight threshold did not:
the fixed-density states of the topology sources and the states at ``k + q`` of
the phonon at ``q``. A band cut off at the budget went into the invariant, or
into ``P_c^+`` and the Sternheimer solve at ``q``, as if it were solved.

The straggler is forced by handing the calculation an eigensolver with a
two-step budget, so the guard has something to fire on; next to it, the same
solve at its own default threshold and budget settles and must stay silent.
On silicon at ``ethr = 1e-13`` every k-point of the ``k + q`` solve settles in
18 to 36 steps, so the silent case is not a near miss.
"""

import copy
import functools
import warnings

import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response.phononq import states_at_k_plus_q
from defuqumat.scf.driver import Calculation
from defuqumat.solvers.davidson import davidson_eigensolver_all
from defuqumat.system.builder import build_system
from defuqumat.workflows.topology import DFTSource

pytestmark = pytest.mark.unit

_SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0,
  nosym = .true., noinv = .true.
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""

#: A budget no random start settles in.
SHORT_BUDGET = 2


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    system = build_system(parse_pw_input(_SILICON))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    density = calculation.starting_density()
    return system, pseudos, calculation, density


def _short(calculation):
    """A copy of ``calculation`` whose eigensolver stops after two steps."""
    short = copy.copy(calculation)
    short.eigensolver = functools.partial(
        davidson_eigensolver_all, max_iterations=SHORT_BUDGET)
    return short


def _quiet(call):
    """Run ``call`` and return what it returned and the unsettled-band warnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    said = [str(w.message) for w in caught if "did not converge" in str(w.message)]
    return result, said


def _states(system, pseudos, calculation, density):
    source = DFTSource(system=system, pseudos=pseudos, density=density, nocc=4,
                       calculation=calculation)
    points = np.array([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.5, 0.5]])
    return source.states(points)


def test_a_topology_solve_cut_at_the_budget_is_reported(silicon):
    system, pseudos, calculation, density = silicon
    _, said = _quiet(lambda: _states(system, pseudos, _short(calculation), density))
    assert said, "a two-step fixed-density solve passed without a warning"
    message = said[0]
    for expected in ("fixed-density solve", "of 3 k-points", f"took {SHORT_BUDGET} "
                     "Davidson steps", "default budget is 100", "conv_thr"):
        assert expected in message, f"{expected!r} missing from: {message}"


def test_a_settled_topology_solve_stays_silent(silicon):
    system, pseudos, calculation, density = silicon
    _, said = _quiet(lambda: _states(system, pseudos, calculation, density))
    assert not said, said


def _at_q(calculation, density, stream):
    v_scf = calculation.potential(density).v_scf
    q = np.array([0.25, 0.0, 0.0]) * calculation.system.cell.tpiba
    return states_at_k_plus_q(calculation, v_scf, q, nbnd=8, stream=stream)


@pytest.mark.parametrize("stream", [False, True], ids=["device", "stream"])
def test_a_k_plus_q_solve_cut_at_the_budget_is_reported(silicon, stream):
    _, _, calculation, density = silicon
    _, said = _quiet(lambda: _at_q(_short(calculation), density, stream))
    assert said, "a two-step solve at k + q passed without a warning"
    message = said[0]
    for expected in ("k + q", "of 8 k-points", f"took {SHORT_BUDGET} Davidson "
                     "steps", "default budget is 100", "ethr = 1.0e-13"):
        assert expected in message, f"{expected!r} missing from: {message}"


@pytest.mark.parametrize("stream", [False, True], ids=["device", "stream"])
def test_a_settled_k_plus_q_solve_stays_silent(silicon, stream):
    _, _, calculation, density = silicon
    _, said = _quiet(lambda: _at_q(calculation, density, stream))
    assert not said, said
