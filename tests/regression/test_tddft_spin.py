"""The sum-over-states ``chi_0`` at ``nspin = 2`` (``notes/incompat/I7.md``).

At ``nspin = 2`` the charge ``chi_0`` is the sum of the two channels' Adler-Wiser
sums, each over its own occupied count and with its own ``wg``. Two checks,
neither of which shares the sum it tests:

* **an unpolarized cell at ``nspin = 2`` is the ``nspin = 1`` cell**, the whole
  matrix and not only its head, and the ALDA ``eps_M(0)`` with it. A factor of
  two in the channel weights, or a tangent of ones in the kernel, reads as a
  factor of two here and as nothing anywhere else.
* **a magnetic insulator's RPA ``eps_M(0)`` is the Sternheimer solve's** at
  ``nspin = 2`` with Hartree screening, which never forms an empty state. The
  cell (``h3-chain-lsda-fixed.in``) fills its up channel two bands deep and its
  down channel one, so this is the check of the unequal counts, which an
  unpolarized cell cannot make. The two agree only as ``nbnd`` grows, and the
  tolerance is the truncation measured at the band count used.

Both are marked slow: each is an SCF, a fixed-density run with empty states and
a Sternheimer or second sum.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation, run_scf
from defuqumat.system import build_system
from defuqumat.tddft import alda_matrix, independent_response, solve_dyson
from defuqumat.workflows.nscf import fixed_density_states
from defuqumat.workflows.tddft import run_absorption

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

SILICON = "si-epsilon-unshifted-nosym"
CHAIN = "h3-chain-lsda-fixed"

#: ``max |x(nspin=2) - x(nspin=1)| / max |x|``, measured at 3.0e-9 (8.5e-8 on
#: 28.6, D22, 2026-10-16): two separate SCFs at 1e-12 and fixed-density runs at
#: 1e-10, and nothing else.
MATRIX_TOLERANCE = 1.0e-7
#: ``|eps_M(nspin=2) - eps_M(nspin=1)|`` at ``omega = 0``, measured at 1.6e-11
#: in RPA and 1.5e-11 in ALDA on a constant of 22.33 and 23.62.
EPSILON_TOLERANCE = 1.0e-8
#: The chain at 80 bands and ``ecut_response = 8``: the sum over states reads
#: 8.65363 against the Sternheimer solve's 8.64013, a residual of +1.35e-2
#: (1.8e-3 of ``eps - 1``). At ``ecut_response = 4`` it is +1.31e-1 and does
#: not shrink with bands (8.737, 8.767, 8.772 at 20, 40, 80), which is the
#: local fields; at 8 it is -3.5e-2, +4.5e-3, +1.35e-2. The tolerance is the
#: silicon identity's in ``test_tddft.py``. What a wrong channel count would
#: give was not measured.
CHAIN_NBND = 80
CHAIN_ECUT = 8.0
CHAIN_TOLERANCE = 3.0e-2


@pytest.fixture(autouse=True)
def _clear_compiled():
    yield
    jax.clear_caches()


def _load(case, **spin):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    if spin:
        system = system.with_spin(**spin)
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos


@lru_cache(maxsize=2)
def _silicon(nspin: int):
    """``chi_0`` at 60 bands and ``ecut_response = 8``, the other file's point."""
    spin = {} if nspin == 1 else {"nspin": 2, "tot_magnetization": 0}
    system, pseudos = _load(SILICON, **spin)
    calculation = Calculation(system, pseudos)
    scf = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                  max_iterations=80)
    calculation, system, eigenvalues, wavefunctions = fixed_density_states(
        system, pseudos, scf.density, nbnd=60, conv_thr=1e-10)
    potential = calculation.potential(jnp.asarray(scf.density))
    chi = independent_response(
        calculation, wavefunctions, eigenvalues, potential.v_scf,
        np.array([0.0, 0.1 + 0.01j]), ecut_response=8.0, broadening=0.0)
    alda = alda_matrix(calculation, jnp.asarray(scf.density), chi.sphere)
    return chi, alda


def _static(chi, kernel, alda):
    context = {"alda_matrix": alda} if kernel == "alda" else {}
    solution = solve_dyson(chi, kernel, context, static_index=0)
    return float(np.real(np.diag(np.asarray(solution.epsilon)[0])).mean())


def test_an_unpolarized_cell_at_nspin_2_is_the_nspin_1_cell():
    """The whole matrix, at a static and a broadened frequency, and both kernels.

    Each channel of the unpolarized ``nspin = 2`` cell holds one electron a band
    where the ``nspin = 1`` cell holds two, and the two channels are equal, so
    the sum is the single channel's matrix exactly; what is left is the
    convergence of two separate SCFs and fixed-density runs.
    """
    one, alda_one = _silicon(1)
    two, alda_two = _silicon(2)
    assert two.occupied == (4, 4) and one.occupied == (4,)
    x1, x2 = np.asarray(one.x), np.asarray(two.x)
    assert x1.shape == x2.shape
    assert np.abs(x2 - x1).max() <= MATRIX_TOLERANCE * np.abs(x1).max()
    for kernel, (a1, a2) in (("rpa", (None, None)), ("alda", (alda_one, alda_two))):
        assert abs(_static(two, kernel, a2) - _static(one, kernel, a1)) < EPSILON_TOLERANCE


def test_a_magnetic_insulator_matches_the_sternheimer_solve():
    """Unequal occupied counts, ``(2, 1)``, against a route with no empty state.

    ``run_absorption``'s ``static_residual`` is the RPA ``eps_M(0)`` of the sum
    over states minus the Sternheimer solve's with Hartree screening, at
    ``nspin = 2`` on both sides. If the down channel were filled to the up
    channel's depth, or the weights carried two electrons a band, this would
    be off by the whole of a channel's contribution.
    """
    system, pseudos = _load(CHAIN)
    calculation = Calculation(system, pseudos)
    scf = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                  max_iterations=150)
    spectrum = run_absorption(
        system, pseudos, scf.density, np.array([0.1]), kernel="rpa",
        nbnd=CHAIN_NBND, ecut_response=CHAIN_ECUT, broadening=0.01,
        calculation=calculation)
    sternheimer = spectrum.static_rpa - spectrum.static_residual
    assert sternheimer > 1.0
    assert abs(spectrum.static_residual) < CHAIN_TOLERANCE


def test_alda_on_the_magnetic_chain_is_refused():
    """The sibling refusal: with a moment, ALDA's kernel is a 2x2 in ``(n, m)``."""
    system, pseudos = _load(CHAIN)
    calculation = Calculation(system, pseudos)
    scf = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-8,
                  max_iterations=150)
    with pytest.raises(NotImplementedError, match="magnetized ground state"):
        run_absorption(system, pseudos, scf.density, np.array([0.1]),
                       kernel="alda", nbnd=8, ecut_response=2.0,
                       static_residual=False, calculation=calculation)
