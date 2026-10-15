"""Real-time propagation from a ``K_POINTS gamma`` ground state, against an explicit ``k = 0``.

``H(k + kappa)`` is complex at ``kappa != 0``, so the propagated state is not
real and the half sphere cannot carry it. The workflows re-solve the occupied
states from the density on the k-set they propagate, so what a gamma-only
ground state needs is that k-set on the whole sphere
(``defuqumat.workflows.realtime._whole_sphere``); the density itself lives on
the dense set, which gamma-only storage never halves (``build_basis``).

**The reference needs no other code.** The same cell run with an explicit
``K_POINTS automatic 1 1 1 0 0 0`` is the same physics on the whole sphere, so
the current, the band energy and the electron count must agree to what the two
SCF densities and the fixed-density solves agree to, not to a physical
tolerance. A half sphere propagated as if it were whole would be off by about a
factor of two, or raise in ``scatter_to_box_gamma``.

Both routes are checked: the bare :func:`run_realtime`, which builds its own
calculation and is where the refusal fired, and ``Calculator.get_realtime``,
which hands over the SCF's half-sphere calculation for ``at_kpoints`` to move.
"""

import warnings
from functools import lru_cache

import jax
import numpy as np
import pytest

from defuqumat.calculator import Calculator
from defuqumat.realtime.pulse import Sin2
from defuqumat.workflows.realtime import run_realtime

pytestmark = pytest.mark.regression

_TEMPLATE = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0,
  nosym = .true.
/
&electrons
  conv_thr = 1.0d-12
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.01 0.00 0.00
 Si 0.26 0.24 0.25
K_POINTS {kpoints}
"""

#: A one-cycle pulse along a direction no symmetry of the displaced cell fixes,
#: cut at 400 steps: what is compared is two storages, not a converged spectrum.
PULSE = Sin2.from_intensity(1e12, 1.55, 1, polarization=(1.0, 0.5, 0.2))
OPTIONS = dict(dt=0.1, duration=40.0, conv_thr=1.0e-12)

#: Measured on D22 (``notes/incompat/I39.md``), and set above the measurement
#: with room for the fixed-density solve's ``conv_thr``.
CURRENT_RTOL = 1.0e-8
ENERGY_ATOL = 1.0e-10


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _ground(pseudo_dir, kpoints):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(_TEMPLATE.format(kpoints=kpoints), pseudo_dir,
                                          announce=False)
        result = calculator.get_scf()
    assert result.converged
    return calculator, result


def _agree(run, reference):
    current, expected = np.asarray(run.current), np.asarray(reference.current)
    scale = float(np.abs(expected).max())
    assert scale > 0.0
    gap = float(np.abs(current - expected).max()) / scale
    assert gap < CURRENT_RTOL, gap
    np.testing.assert_allclose(run.energy, reference.energy, rtol=0.0, atol=ENERGY_ATOL)
    assert run.nelec == pytest.approx(8.0, abs=1e-12)
    assert run.nelec == pytest.approx(reference.nelec, abs=1e-12)
    assert run.norm_drift < 1e-8
    return gap


@pytest.mark.slow
def test_a_gamma_ground_state_propagates_as_the_explicit_k0(pseudo_dir):
    """The current, the energy and the electron count, both routes, against ``1 1 1 0 0 0``."""
    whole, full = _ground(pseudo_dir, "automatic\n 1 1 1 0 0 0")
    half, gamma = _ground(pseudo_dir, "gamma")
    # Otherwise the SCF did not consume the storage and this proves nothing.
    assert half.calculation.gamma_only is True
    assert whole.calculation.gamma_only is False

    reference = run_realtime(whole.system, whole.pseudos, full.density, PULSE, **OPTIONS)
    bare = run_realtime(half.system, half.pseudos, gamma.density, PULSE, **OPTIONS)
    threaded = half.get_realtime(PULSE, **OPTIONS)
    gaps = (_agree(bare, reference), _agree(threaded, reference))
    print(f"current gap, bare {gaps[0]:.3e}, through the calculator {gaps[1]:.3e}; "
          f"max |J| {float(np.abs(np.asarray(reference.current)).max()):.6e}")
