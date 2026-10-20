"""The exchange-correlation spin torque under a potential-only meta-GGA (I110).

``torque_of_result`` used to refuse every meta-GGA run, on the ground that the
potential needs ``tau``. The run already carries its converged ``tau``
(``SCFResult.tau``), so the torque at the run's own state now reads it. What
stays refused: a caller's density with no ``tau`` beside it, and a result that
carries none.

The fast tests here build a real noncollinear TB09 ``Calculation`` and hand it
made-up smooth fields, so ``v_of_rho`` runs once per functional choice and no
SCF is involved. They check that ``tau`` reaches the potential (the torque
moves when ``tau``'s vector part does) and that the two noncollinear
prescriptions behave as they must: ``'kubler'`` attaches the field to
``m``-hat, a null, and ``'grad_m'`` does not. The check against the SCF's own
converged field is a slow test in ``tests/regression/test_spin_torque_meta_run.py``.
"""

import types
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.scf.spin_torque import torque_of_result

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


def _stub(tau=None):
    calculation = types.SimpleNamespace(functional=types.SimpleNamespace(is_meta=True))
    result = types.SimpleNamespace(density=np.zeros((4, 2, 2, 2)), tau=tau)
    return calculation, result


def test_a_result_without_tau_is_refused():
    calculation, result = _stub()
    with pytest.raises(NotImplementedError, match="SCFResult.tau is None"):
        torque_of_result(calculation, result)


def test_a_density_without_its_tau_is_refused():
    """The converged ``tau`` beside a turned magnetization is refused, not defaulted."""
    calculation, result = _stub(tau=np.zeros((4, 2, 2, 2)))
    with pytest.raises(NotImplementedError, match="passed as tau="):
        torque_of_result(calculation, result, density=np.zeros((4, 2, 2, 2)))


@lru_cache(maxsize=2)
def _calculator(choice):
    from defuqumat import Calculator

    text = (CASES / "h-atom-noncolin.in").read_text().replace(
        "    noncolin = .true.",
        f"    noncolin = .true.\n    input_dft = 'tb09'\n    noncollinear_gga = '{choice}'",
        1,
    ).replace("ecutwfc = 25.0", "ecutwfc = 10.0", 1).replace(
        "celldm(1) = 12.0", "celldm(1) = 8.0", 1)
    return Calculator.from_text(text, pseudo_dir=PSEUDO)


def _fields(grid, transverse=1.0):
    """A smooth positive charge, a magnetization that turns in space, and a ``tau``.

    ``tau``'s vector part is given a component off ``m`` scaled by
    ``transverse``, which is what the parity form carries and the local frame
    discards.
    """
    x = 2.0 * np.pi * np.indices(grid) / np.array(grid)[:, None, None, None]
    charge = 0.3 + 0.1 * np.cos(x[0]) * np.cos(x[1]) + 0.05 * np.cos(x[2])
    angle = 0.8 * np.sin(x[2]) + 0.3 * np.cos(x[0])
    size = 0.12 + 0.03 * np.cos(x[1])
    m = np.stack([size * np.sin(angle), 0.02 * np.sin(x[0]), size * np.cos(angle)])
    tau0 = 0.6 * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * charge ** (5.0 / 3.0)
    tau_vec = 2.0 * m + transverse * np.stack(
        [0.03 * np.cos(x[1]), 0.04 * np.sin(x[2]), 0.02 * np.cos(x[0])])
    return np.concatenate([charge[None], m]), np.concatenate([tau0[None], tau_vec])


def _torque(choice, transverse=1.0):
    calc = _calculator(choice)
    grid = tuple(calc.calculation.basis.dense.grid)
    density, tau = _fields(grid, transverse)
    result = types.SimpleNamespace(density=jnp.asarray(density), tau=jnp.asarray(tau))
    return torque_of_result(calc.calculation, result)


def test_kubler_attaches_the_meta_field_to_m():
    torque = _torque("kubler")
    assert float(torque.parallel_fraction) == pytest.approx(1.0, abs=1.0e-12)
    assert np.linalg.norm(np.asarray(torque.total)) < 1.0e-14


def test_grad_m_reads_tau_and_turns_the_moment():
    """The torque is a number under ``'grad_m'``, and it is ``tau``'s.

    Changing only the part of ``tau``'s vector off ``m`` moves the torque,
    which a route that dropped ``tau`` (or rebuilt it from the density) could
    not do.
    """
    with_tau = _torque("grad_m", transverse=1.0)
    without = _torque("grad_m", transverse=0.0)
    assert float(with_tau.parallel_fraction) < 0.999
    a, b = np.asarray(with_tau.total), np.asarray(without.total)
    assert np.linalg.norm(a) > 1.0e-6
    assert np.linalg.norm(a - b) > 1.0e-3 * np.linalg.norm(a)


def test_the_source_free_field_admits_a_meta_gga_and_keeps_its_other_refusals():
    """The sibling guard in ``sourcefree.py``, after I109 lifted its meta-GGA branch.

    A potential-only meta-GGA under ``nosource`` is admitted, so the torque of
    such a run is the one tested above, rebuilt by ``torque_of_result`` with
    ``calculation.source_free`` and the run's own ``tau``. What I109 did not
    lift still fires with a meta-GGA: PAW and a spin spiral, each by its own
    message.
    """
    from defuqumat.scf.sourcefree import refuse_source_free

    system = types.SimpleNamespace(noncolin=True, spiral_q=None)
    meta = types.SimpleNamespace(is_meta=True)
    refuse_source_free(system, meta)
    with pytest.raises(NotImplementedError, match="PAW"):
        refuse_source_free(system, meta, (types.SimpleNamespace(is_paw=True),))
    spiral = types.SimpleNamespace(noncolin=True, spiral_q=(0, 0, 0.25))
    with pytest.raises(NotImplementedError, match="spin spiral"):
        refuse_source_free(spiral, meta)
