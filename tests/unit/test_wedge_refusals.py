"""The wedge refusals that stand after the wedge unit, each fired by name.

No SCF here: every guard reads the calculation alone. The transverse spin
susceptibility had no test firing its wedge refusal at all; it is the first one
below.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from defuqumat.io.pwin import parse_pw_input, read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import Calculation
from defuqumat.system import build_system

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _calculation(text: str) -> Calculation:
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return Calculation(system, pseudos)


def test_a_magnon_on_a_symmetry_reduced_k_set_is_refused():
    """``ni-fcc-magnon.in`` with its ``nosym``/``noinv`` taken out.

    ``X_0(G, G')`` at ``q`` needs the states at ``k + q`` from the same set,
    which a wedge does not hold, and a magnetic group besides; the committed
    cell, which carries ``nosym``, passes the same guard.
    """
    from defuqumat.tddft.spinchi0 import require_a_transverse_regime

    text = (CASES / "ni-fcc-magnon.in").read_text()
    reduced = _calculation(text.replace("nosym = .true., noinv = .true.", ""))
    assert reduced.system.kpoints.nk < 64
    with pytest.raises(NotImplementedError, match="full k-grid"):
        require_a_transverse_regime(reduced)
    require_a_transverse_regime(_calculation(text))


SILICON = (CASES / "si-epsilon-unshifted.in").read_text()


@pytest.mark.parametrize("quantity", ["chi_0", "a tunnelling spectrum"])
def test_a_shifted_wedge_is_refused_and_the_unshifted_one_is_not(quantity):
    """A shifted grid's wedge unfolds uniformly, but onto its star closure.

    ``4 4 4 1 1 1`` on fcc silicon: 10 points, which the 96 operations unfold
    onto 256 uniformly weighted points, not the grid's 64. The completion is
    then exact for the closure and not for the grid (D22: chi_0 4.2e-6 and STS
    2.6e-8 against the explicit closure, 0.41 and 0.14 off the shifted grid),
    so the set is refused; the unshifted wedge, which unfolds onto its own 64,
    passes the same guard.
    """
    from defuqumat.system.wedge import wedge_group
    from defuqumat.tddft.chi0 import require_a_sum_over_states_regime
    from defuqumat.workflows.stm import _refuse_a_reduced_k_set

    shifted = _calculation(SILICON.replace(" 4 4 4 0 0 0", " 4 4 4 1 1 1"))
    unshifted = _calculation(SILICON)
    assert shifted.system.kpoints.nk == 10
    if quantity == "chi_0":
        with pytest.raises(NotImplementedError, match="star closure"):
            require_a_sum_over_states_regime(shifted)
        require_a_sum_over_states_regime(unshifted)
    else:
        with pytest.raises(NotImplementedError, match="star closure"):
            _refuse_a_reduced_k_set(shifted.system)
        assert _refuse_a_reduced_k_set(unshifted.system).order == 96
    assert wedge_group(unshifted.system, quantity=quantity).order == 96


def test_a_two_channel_wedge_is_refused_for_chi_0_and_the_spectrum():
    """Collinear ``nspin = 2`` reaches the completion with no number behind it.

    Its group carries no ``t_rev``, so :func:`wedge_group` does not see it as
    magnetic; the two consumers refuse it themselves.
    """
    from defuqumat.tddft.chi0 import require_a_sum_over_states_regime
    from defuqumat.workflows.stm import _refuse_a_reduced_k_set

    text = SILICON.replace(
        "ecutwfc = 18.0", "ecutwfc = 18.0, nspin = 2, starting_magnetization(1) = 0.0,\n    tot_magnetization = 0")
    polarized = _calculation(text)
    assert polarized.system.nspin == 2 and polarized.system.kpoints.nk == 8
    with pytest.raises(NotImplementedError, match="nspin = 1 run only"):
        require_a_sum_over_states_regime(polarized)
    with pytest.raises(NotImplementedError, match="nspin = 1 run only"):
        _refuse_a_reduced_k_set(polarized.system)
