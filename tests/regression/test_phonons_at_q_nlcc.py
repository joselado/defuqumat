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
#: :data:`CORE` with its model core replaced by a broad Gaussian, so that two
#: atoms' cores overlap; see :func:`_broad_core`.
BROAD = "broad"

#: The Gaussian of :data:`BROAD`: one electron, ``exp(-r^2 / rc^2)``, in bohr.
BROAD_RADIUS = 1.6


@lru_cache(maxsize=1)
def _broad_core():
    """``CORE`` with ``rho_core(r) = exp(-r^2/rc^2) / (pi^{3/2} rc^3)``.

    The ONCV model core is compact: the core-core term ``C(q) - C(0)`` it gives
    at ``b1/2`` is 9.1e-7 Ry/bohr^2, below the 2.8e-6 at which the supercell
    identity holds, so on it the identity cannot see the swap. At
    ``rc = 1.6`` bohr the Gaussian is 15 per cent of its peak at half the bond
    (2.2 bohr) and the swap is well above the floor (``notes/incompat/I80.md``).
    The dataset is not a physical one; the identity does not need it to be.
    """
    import dataclasses

    from defuqumat.pseudo import read_upf

    pseudo = read_upf(PSEUDO / CORE)
    r = np.asarray(pseudo.r)
    rc = BROAD_RADIUS
    core = np.exp(-(r / rc) ** 2) / (np.pi ** 1.5 * rc ** 3)
    return dataclasses.replace(pseudo, rho_core=core)

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
def _calculator(cell: str, pseudo: str, k_batch=None):
    import warnings

    from defuqumat import Calculator

    from defuqumat.calculator import electrons_defaults
    from defuqumat.io.pwin import parse_pw_input
    from defuqumat.system import build_system

    text = (_UNIT if cell == "unit" else _SUPER).format(
        nat=2 if cell == "unit" else 4, pseudo=CORE if pseudo == BROAD else pseudo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pseudo == BROAD:
            pwin = parse_pw_input(text)
            calculator = Calculator(build_system(pwin), (_broad_core(),),
                                    announce=False, **electrons_defaults(pwin),
                                    **({} if k_batch is None else {"k_batch": k_batch}))
        else:
            calculator = Calculator.from_text(
                text, str(PSEUDO), announce=False,
                **({} if k_batch is None else {"k_batch": k_batch}))
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


#: In Ry/bohr^2, on force constants of 0.30. Measured 2026-10-16 on D22 (CPU,
#: one core, one run each): **4.40e-6** without a core (the floor, which is
#: the Ewald swap :data:`tests.regression.test_phonons_at_q.GAMMA_AGREEMENT`
#: names: the supercell's ``Gamma`` Hessian carries the ground state's Ewald
#: sum, cut at ``4/sqrt(alpha)``), **2.79e-6** on the ONCV core and **4.81e-6**
#: on the broad one. What the alternatives read on the same states: the guard
#: lifted with no core term at all 1.26e-1 (ONCV) and 7.8e-2 (broad); the
#: bare term in and the core-core swap left at ``q = 0`` 2.79e-6 on ONCV,
#: where ``C(q) - C(0)`` is 9.1e-7 and below the floor, and **3.18e-3** on the
#: broad core, which is why that dataset is here.
SUPERCELL_AGREEMENT = 1e-5

#: ``q = 0`` through two spheres against the ``Gamma`` route on one state, in
#: Ry/bohr^2. Measured on D22: 8.8e-7 on the ONCV core, 3.0e-7 on the broad
#: one; :data:`tests.regression.test_phonons_at_q.GAMMA_AGREEMENT`'s bound.
GAMMA_AGREEMENT = 5e-6


@pytest.mark.parametrize("pseudo", [NO_CORE, CORE, BROAD])
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
    assert difference < GAMMA_AGREEMENT, difference


#: The k-chunked route against the whole one, one k-point a chunk, in
#: Ry/bohr^2. Measured on D22: **1.40e-11** on force constants of 0.30, both
#: routes 11 passes. That is above ``test_streamed_phonons.py``'s 1e-11 and
#: about 300 times its no-core cells (5e-15 to 5e-14); the cause is not
#: isolated (one run), and a missing or doubled core term would read 1e-3 or
#: more (the table in ``notes/incompat/I80.md``).
STREAMED_AGREEMENT = 5e-11


def test_the_streamed_route_carries_the_core_correction():
    """The k-chunked route adds ``dv_core`` to each chunk's bare vectors and
    swaps the core-core term; on the broad core, at one k-point a chunk, it is
    the whole route's matrix."""
    from defuqumat.response.phononq import dynamical_matrix_at_q

    calculator = _calculator("unit", BROAD, 1)
    calculation, result = calculator.calculation, calculator.get_scf()

    def matrix(wavefunctions):
        return np.asarray(dynamical_matrix_at_q(
            calculation, wavefunctions, result.eigenvalues, result.density,
            result.becsum, q=Q_FOLDED).matrix)

    whole = matrix(result.wavefunctions)
    streamed = matrix(np.asarray(result.wavefunctions))
    assert np.abs(streamed - whole).max() < STREAMED_AGREEMENT

