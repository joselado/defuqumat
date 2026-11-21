"""``dE/dq`` and ``at_spiral_q`` on a DFT+U spin spiral (audit 2026-11-20b #1, #2, #13).

A norm-conserving spiral with a Hubbard U has its ``wfcU`` built on the two
shifted spheres (``notes/incompat/I28.md``), so the occupation matrix the frozen
states measure moves with ``q``. Two things missed it: ``at_spiral_q`` kept the
old ``wfcU`` in both of its branches, and the frozen energy ``dE/dq``
differentiates had no Hubbard term at all, so its total was the SCF's minus
``E_U`` and its gradient lacked ``dE_U/dq``.

The projector tests need no SCF and run on the NiI chain of I28, whose
fully-relativistic datasets take the spinor orbital route. The energy tests
share one SCF of the hydrogen chain with a U on the 1s shell, at 20 Ry. On it
the Hubbard energy is 3.90e-3 Ry and ``dE_U/dq_z`` is 9.7e-3 Ry, which is what
the old code left out. The check against reconverged SCFs at ``q +- 0.01`` is
the slow test at the end: measured on this cell it differs from the gradient by
3.9e-6 Ry, and by 1.6e-5 at ``q +- 0.02``, the factor of four of a central
difference's truncation.
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.forces.spiral import compute_spiral_gradient
from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation, run_scf
from defuqumat.system import build_system
from tests.conftest import GENERATED

PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: The NiI chain of I28 with its U, at 20 Ry and two k-points: nothing here is
#: converged, so only the projectors' construction is asked of it.
NII = (GENERATED / "nii-chain-spiral.in").read_text().replace(
    "ecutwfc = 34.0", "ecutwfc = 20.0").replace(
    " 1 1 8 0 0 0", " 1 1 2 0 0 0") + "HUBBARD {atomic}\nU Ni-3d 4.0\n"

#: The hydrogen chain of P21 at ``q = 0.3 b3`` and 20 Ry, with a U on its 1s shell.
HCHAIN = (GENERATED / "h-chain-spiral.in").read_text().replace(
    "spiral_q(3) = 0.25", "spiral_q(3) = 0.3").replace(
    "ecutwfc = 25.0", "ecutwfc = 20.0") + "HUBBARD {atomic}\n U H-1s 4.0\n"

SCF = dict(conv_thr=1e-12, mixing_beta=0.3, max_iterations=300)


def _system(text: str):
    system = build_system(parse_pw_input(text))
    return system, tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)


def _at_q(text: str, q3: float) -> str:
    return text.replace("spiral_q(3) = 0.25", f"spiral_q(3) = {q3}").replace(
        "spiral_q(3) = 0.3", f"spiral_q(3) = {q3}")


@lru_cache(maxsize=1)
def _nii():
    system, pseudos = _system(NII)
    return system, Calculation(system, pseudos)


@lru_cache(maxsize=1)
def _converged():
    system, pseudos = _system(HCHAIN)
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, **SCF)
    assert result.converged
    return calculation, result, pseudos


@pytest.mark.unit
def test_a_moved_spiral_rebuilds_its_hubbard_projectors():
    """#2: ``at_spiral_q`` to a new ``q`` gives the ``wfcU`` of a fresh calculation there."""
    _, calculation = _nii()
    assert calculation.spiral and calculation.is_hubbard
    moved = calculation.at_spiral_q((0.0, 0.0, 0.3))
    fresh = Calculation(*_system(_at_q(NII, 0.3)))
    assert moved.wfcU.shape == fresh.wfcU.shape
    assert float(jnp.max(jnp.abs(moved.wfcU - fresh.wfcU))) < 1e-12


@pytest.mark.unit
def test_the_frozen_sphere_carries_the_hubbard_projectors_with_q():
    """#1: the traced branch rebuilds ``wfcU`` at its ``q``, and at its own ``q`` it is the setup's.

    The second half is the check on the convention of the traced build (the
    cartesian ``k +- q/2`` and the splice of the two rows) that does not share
    the builder with the gradient.
    """
    system, calculation = _nii()
    q = np.asarray(system.spiral_q, dtype=float)
    same = calculation.at_spiral_q(jnp.asarray(q), rebuild_basis=False)
    assert float(jnp.max(jnp.abs(same.wfcU - calculation.wfcU))) < 1e-12
    q[2] += 0.05
    turned = calculation.at_spiral_q(jnp.asarray(q), rebuild_basis=False)
    assert float(jnp.max(jnp.abs(turned.wfcU - calculation.wfcU))) > 1e-3


@pytest.mark.unit
def test_the_differentiated_energy_holds_the_hubbard_energy():
    """#1: the frozen energy is the SCF total, Hubbard energy included, on both routes.

    Without the term the two differ by ``E_U``, 3.90e-3 Ry on this cell.
    """
    calculation, result, _ = _converged()
    assert abs(result.energy_terms["hubbard"]) > 1e-3
    whole = compute_spiral_gradient(calculation, result, k_batch=None)
    chunked = compute_spiral_gradient(calculation, result, k_batch=1)
    assert whole.total_energy == pytest.approx(result.total_energy, abs=1e-10)
    assert chunked.total_energy == pytest.approx(result.total_energy, abs=1e-10)
    # ``E_U`` is quadratic in ``ns``: chunking must pull the whole ``ns``'s
    # gradient back through each chunk, or the cross terms are lost.
    np.testing.assert_allclose(chunked.gradient, whole.gradient, rtol=0, atol=1e-12)


@pytest.mark.unit
def test_dE_dq_is_refused_for_a_potential_only_meta_gga():
    """#13: TB09 has no energy for the frozen state to be stationary in.

    The refusal is ``reject_potential_only``'s, the one every consumer of
    ``energy_at`` shares; it fires before the state is read. The same chain with
    a GGA, ``noncollinear_gga = 'kubler'``, is the regime beside it and still
    passes the guard.
    """
    from defuqumat.forces.spiral import _require_a_differentiable_spiral

    base = GENERATED / "h-chain-spiral.in"
    meta = base.read_text().replace(
        "celldm(3) = 0.41666666666667", "celldm(3) = 0.43333333333333").replace(
        "    nosym = .true.",
        "    nosym = .true.\n    input_dft = 'tb09'\n    noncollinear_gga = 'kubler'", 1)
    calculation = Calculation(*_system(meta))
    assert calculation.spiral and calculation.functional.is_meta
    with pytest.raises(NotImplementedError, match="not the derivative of an energy"):
        compute_spiral_gradient(calculation, None)

    gga = base.read_text().replace(
        "    nosym = .true.",
        "    nosym = .true.\n    input_dft = 'pbe'\n    noncollinear_gga = 'kubler'", 1)
    beside = Calculation(*_system(gga))
    assert beside.spiral and not beside.functional.is_meta
    _require_a_differentiable_spiral(beside)


@pytest.mark.slow
def test_the_hubbard_gradient_is_the_slope_of_the_converged_energy():
    """#1: the gradient against two reconverged SCFs on the frozen sphere.

    The old gradient, without ``dE_U/dq``, is -1.43e-2 Ry here and the new one
    -4.61e-3; the difference is 3.9e-6 Ry at ``delta = 0.01``.
    """
    calculation, result, pseudos = _converged()
    gradient = compute_spiral_gradient(calculation, result)
    delta = 0.01
    energies = []
    for step in (+delta, -delta):
        q = np.asarray(calculation.system.spiral_q, dtype=float)
        q[2] += step
        moved = calculation.at_spiral_q(jnp.asarray(q), rebuild_basis=False)
        displaced = run_scf(moved.system, pseudos, calculation=moved, **SCF)
        assert displaced.converged
        energies.append(displaced.total_energy)
    difference = (energies[0] - energies[1]) / (2.0 * delta)
    assert difference == pytest.approx(gradient.gradient[2], abs=2e-5)
