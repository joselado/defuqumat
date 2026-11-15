"""Refusals and branch handling of the topology slice (audit 2026-11-15, ``topology`` F1, F3).

Neither needs an SCF: the orbital magnetization refuses at the door, and the
magnetoelectric step is plain arithmetic on two folded phases.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_the_orbital_magnetization_of_a_smeared_run_is_refused():
    """A metal is refused by name (P64); the direct gap per mesh point did not see a smeared magnet."""
    from defuqumat.workflows.orbital_magnetization import run_orbital_magnetization

    system = SimpleNamespace(nspin=4, noncolin=True, spiral_q=None, spiral=False,
                             occupations="smearing")
    pseudos = (SimpleNamespace(is_ultrasoft=False, is_paw=False),)
    with pytest.raises(NotImplementedError, match="of a metal"):
        run_orbital_magnetization(system, pseudos, None)


def test_a_phase_pinned_at_half_a_quantum_gives_a_small_step():
    """The two ends folded onto opposite edges of [-q/2, q/2) are one small step apart, not a quantum."""
    quantum, eps = 1.0, 1.0e-3
    plus, minus = np.array([-0.5 + eps]), np.array([0.5 - eps])
    step = plus - minus
    reduced = step - quantum * np.rint(step / quantum)
    assert abs(float(reduced[0]) - 2 * eps) < 1e-12
    import inspect

    from defuqumat.response import magnetoelectric

    source = inspect.getsource(magnetoelectric)
    assert "step = step - quantum * np.rint(step / quantum)" in source
