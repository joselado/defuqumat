"""I80: the phonon at ``q != 0`` with a nonlinear core correction.

A core charge reaches the dynamical matrix at ``q`` in three places, and
``ph.x`` has each of them (``addcore`` inside ``dv_of_drho``, the cross term
and the core-core term of ``dynmat_nlcc.f90``): the solve sees ``f_xc
drho_core`` as part of its perturbation, the assembly contracts the response
density against it, and the frozen Hessian holds ``int drho_core_i f_xc
drho_core_j``, which couples two atoms' cores and so depends on ``q``.

There is no ``ph.x`` reference here at ``q != 0`` with a core charge, and the
check is an identity that shares no machinery with the route: a phonon at a
commensurate ``q`` is the ``Gamma`` phonon of the supercell that ``q`` folds
onto. At ``q = b1/2`` on silicon's primitive cell that is the cell doubled
along ``a1``, and

    D(q)_(s,t) = Phi_(s0, t0) + e^{i q . a1} Phi_(s0, t1)

with ``Phi`` the supercell's force constants at ``Gamma`` and ``t1`` atom ``t``
moved by ``a1``. The ``Gamma`` route carries the core correction through one
``jvp`` of the potential through the moved atoms and one Hessian of the
frozen energy, the ``q`` route through :func:`core_charge_at_q` and the swap;
the 4x2x2 grid on the primitive cell is the 2x2x2 grid on the supercell.

The dataset is ONCVPSP silicon (LDA, scalar-relativistic, 0.74 electrons of
model core charge), copied from QE 7.5's ``KCW/examples/example05.1/pseudo/
Si.upf`` as ``Si.pz-oncv-nlcc.UPF``: the only scalar-relativistic
norm-conserving dataset with ``PP_NLCC`` in the QE tree that this cell can use.
``Si.pz-vbc.UPF`` (no core) runs the same identity as the floor of the check.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

pytestmark = [pytest.mark.regression, pytest.mark.slow]

PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

CORE = "Si.pz-oncv-nlcc.UPF"
NO_CORE = "Si.pz-vbc.UPF"

_HEAD = """&control
  calculation = 'scf'
/
&system
  ibrav = 0, celldm(1) = 10.20, nat = {nat}, ntyp = 1, ecutwfc = 16.0,
  nosym = .true., noinv = .true.
/
&electrons
  conv_thr = 1.0d-12
/
ATOMIC_SPECIES
 Si 28.086 {pseudo}
"""

#: Silicon's primitive cell, ``ibrav = 2``'s vectors written out.
_UNIT = _HEAD + """CELL_PARAMETERS alat
  -0.50  0.00  0.50
   0.00  0.50  0.50
  -0.50  0.50  0.00
ATOMIC_POSITIONS alat
 Si  0.00  0.00  0.00
 Si  0.25  0.25  0.25
K_POINTS automatic
 4 2 2 0 0 0
"""

#: The same crystal on ``2 a1, a2, a3``; atoms 3 and 4 are 1 and 2 moved by ``a1``.
_SUPER = _HEAD + """CELL_PARAMETERS alat
  -1.00  0.00  1.00
   0.00  0.50  0.50
  -0.50  0.50  0.00
ATOMIC_POSITIONS alat
 Si  0.00  0.00  0.00
 Si  0.25  0.25  0.25
 Si -0.50  0.00  0.50
 Si -0.25  0.25  0.75
K_POINTS automatic
 2 2 2 0 0 0
"""

#: ``b1 / 2``, the wavevector the doubled cell folds to ``Gamma``.
Q_FOLDED = (0.5, 0.0, 0.0)


@pytest.fixture(autouse=True)
def _bounded_compilation_cache():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _calculator(cell: str, pseudo: str):
    import warnings

    from defuqumat import Calculator

    text = (_UNIT if cell == "unit" else _SUPER).format(
        nat=2 if cell == "unit" else 4, pseudo=pseudo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(text, str(PSEUDO), announce=False)
    calculator.get_scf()
    return calculator


def unfolded(pseudo: str, q=Q_FOLDED) -> np.ndarray:
    """``D(q)`` from the supercell's ``Gamma`` force constants, ``(6, 6)``."""
    from defuqumat.response.phononq import TR2_AT_Q_CAPPED

    phonons = _calculator("super", pseudo).get_phonons(tr2=TR2_AT_Q_CAPPED)
    assert phonons.converged
    phi = np.asarray(phonons.matrix)
    phase = np.exp(2j * np.pi * q[0])
    return (phi[:2, :, :2, :] + phase * phi[:2, :, 2:, :]).reshape(6, 6)


def at_q(pseudo: str, q=Q_FOLDED) -> np.ndarray:
    """``D(q)`` through the two-sphere route on the primitive cell, ``(6, 6)``."""
    phonons = _calculator("unit", pseudo).get_phonons_at_q(q)
    assert phonons.converged
    return np.asarray(phonons.matrix).reshape(6, 6)


#: In Ry/bohr^2, on force constants of order 0.2. To be set from the D22
#: measurement recorded in ``notes/incompat/I80.md``.
SUPERCELL_AGREEMENT = 5e-6


@pytest.mark.parametrize("pseudo", [NO_CORE, CORE])
def test_a_commensurate_q_is_the_supercell_gamma_phonon(pseudo):
    """``D(b1/2)`` against the doubled cell's ``Gamma`` matrix, unfolded."""
    difference = np.abs(at_q(pseudo) - unfolded(pseudo)).max()
    assert difference < SUPERCELL_AGREEMENT, difference


def test_q_zero_reproduces_the_gamma_route_with_a_core_charge():
    """``q = 0`` through two spheres against the ``Gamma`` route on one state.

    The core-core swap is exactly zero here, so this checks the bare term
    (``f_xc drho_core`` on the ``k + q`` sphere) and the cross term in the
    assembly against the ``Gamma`` route's ``addcore`` in ``dV_scf``.
    """
    from defuqumat.response.phononq import TR2_AT_Q_CAPPED

    calculator = _calculator("unit", CORE)
    gamma = np.asarray(calculator.get_phonons(tr2=TR2_AT_Q_CAPPED).matrix).reshape(6, 6)
    difference = np.abs(at_q(CORE, (0.0, 0.0, 0.0)) - gamma).max()
    assert difference < SUPERCELL_AGREEMENT, difference

