"""The force and the stress of a norm-conserving spin spiral, against its supercell.

``pw.x`` has no spin spiral, so the check is the identity P19 rests on: at a
commensurate ``q`` the spiral *is* a supercell, the generalized Bloch theorem
being exact there. The cell is the nickel chain with an iodine beside each bond
(``tests/data/qe/nii-chain-spiral.in``) at ``q = b3/4``, against the four-cell
supercell ``nii-chain-4cell.in`` diagonalised at the spiral's density unfolded onto
it, exactly as ``test_spiral_soc.py`` builds it. Three things make it the right
cell and the hydrogen chain the wrong one:

* both datasets carry projectors, so the pair ``vkb(k + q/2)``, ``vkb(k - q/2)``
  in the nonlocal term -- the whole of what was missing -- is exercised
  (``H.pz-vbc`` has none);
* the iodine sits off the chain, so the force on each atom is not zero by
  symmetry: an identity between two zeros would test nothing;
* the supercell's two spinor components share one sphere, so it is computed by
  the ordinary noncollinear path that ``test_spinor_forces.py`` holds against
  ``pw.x``.

Both sides are frozen-state derivatives of states that describe the same physical
electrons, so they agree to the eigensolver's tolerance and not to the SCF's: the
supercell is never converged on its own, it is diagonalised once at the unfolded
density. The force is compared atom by atom (each of the four nickel and four
iodine atoms of the supercell against the spiral's one), the stress as it is,
since both are intensive.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.forces import compute_forces
from defuqumat.forces.energy import FrozenState, energy_at, state_from_result
from defuqumat.scf.driver import Calculation
from defuqumat.stress import compute_stress
from defuqumat.workflows.nscf import fixed_density_states
from defuqumat.workflows.spiral import unfold_spiral_density
from tests.conftest import GENERATED

pytestmark = [pytest.mark.regression, pytest.mark.slow]

SPIRAL = GENERATED / "nii-chain-spiral.in"
SUPERCELL = GENERATED / "nii-chain-4cell.in"

#: The spiral against the supercell, Ry/bohr and Ry/bohr^3. Both from this
#: code at one density, so the floor is the eigensolver's (``conv_thr = 1e-10``
#: on the supercell's diagonalisation); the measured figures are in
#: ``notes/incompat/I20.md`` and ``I21.md``.
FORCE_IDENTITY = 1.0e-6
STRESS_IDENTITY = 1.0e-7


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


@lru_cache(maxsize=1)
def _spiral(pseudo_dir: Path):
    calculator = Calculator.from_file(SPIRAL, pseudo_dir, announce=False)
    result = calculator.get_scf()
    assert result.converged
    calculation = Calculation(calculator.system, calculator.pseudos)
    return calculator, calculation, result


@lru_cache(maxsize=1)
def _supercell(pseudo_dir: Path):
    """The supercell at the unfolded spiral density, as a frozen state."""
    spiral, _, result = _spiral(pseudo_dir)
    supercell = Calculator.from_file(SUPERCELL, pseudo_dir, announce=False)
    system = supercell.system.with_soc_scale(0.0)
    calculation = Calculation(system, supercell.pseudos)
    unit = Calculation(spiral.system, spiral.pseudos).basis.dense.grid
    assert calculation.basis.dense.grid == (unit[0], unit[1], 4 * unit[2])
    lab = unfold_spiral_density(np.asarray(result.density), spiral.system.spiral_q,
                                (1, 1, 4), calculation.basis.dense.grid)
    calculation, _, eigenvalues, states = fixed_density_states(
        system, supercell.pseudos, lab, conv_thr=1.0e-10, calculation=calculation,
    )
    eigenvalues = jnp.asarray(eigenvalues)
    weights, levels = calculation.occupations(eigenvalues)
    state = FrozenState(
        wavefunctions=jnp.asarray(states), weights=jnp.asarray(weights),
        eigenvalues=eigenvalues, entropy=float(levels.get("smearing", 0.0)),
    )
    return calculation, state


def test_the_functional_is_the_spiral_total_energy(pseudo_dir):
    """What is differentiated is the energy the SCF converged, pair term included."""
    _, calculation, result = _spiral(pseudo_dir)
    frozen = float(energy_at(calculation, state_from_result(result), spinors=True))
    assert frozen == pytest.approx(result.total_energy, abs=1.0e-8)


def test_the_spiral_force_is_the_supercell_force(pseudo_dir):
    _, calculation, result = _spiral(pseudo_dir)
    spiral = compute_forces(calculation, result)
    supercell_calculation, state = _supercell(pseudo_dir)
    supercell = compute_forces(supercell_calculation, state)

    # The iodine is off the chain, so the force is not a symmetry's zero.
    assert np.abs(spiral.forces[1]).max() > 1.0e-3
    # Nickel then iodine, in both cells.
    unfolded = np.tile(spiral.forces, (4, 1))
    assert np.abs(supercell.forces - unfolded).max() < FORCE_IDENTITY


def test_the_spiral_stress_is_the_supercell_stress(pseudo_dir):
    _, calculation, result = _spiral(pseudo_dir)
    spiral = compute_stress(calculation, result)
    supercell_calculation, state = _supercell(pseudo_dir)
    supercell = compute_stress(supercell_calculation, state)

    assert np.abs(spiral.tensor).max() > 1.0e-5
    assert np.abs(supercell.tensor - spiral.tensor).max() < STRESS_IDENTITY
