"""Real-time propagation of a saw-tooth (``tefield``) ground state, at a frozen potential.

``tests/data/qe/h2-box-tefield.in``: H2 along z in a 16-bohr box, ``eamp = 0.01``
along z, two k-points along the field (1x1x2) on the whole sphere
(``notes/incompat/I101.md``).

The saw-tooth is a periodic scalar potential inside the frozen ``v_scf`` the
workflows hand the driver, and the laser couples to the kinetic and nonlocal
terms through ``kappa``, so the two are one ordinary Hamiltonian. Two checks,
each read against the arm with the saw-tooth left out of the propagated
potential:

- **Stationarity.** With no pulse the ground state does not move: 1.1e-12 is
  the largest current over 1000 steps. The same states propagated under
  ``v_scf - v_saw`` are not eigenstates and carry 9.9e-6.
- **The static polarizability through the other gauge.** A slow
  ``E0 sin^2(pi t/T)`` pulse along z, the odd part in ``E0``, gives
  ``P(t) = int J dt``, fitted to ``alpha E(t)/Omega + b int kappa dt``. Here
  ``alpha`` is compared with the frozen-potential polarizability from a central
  difference of the band energy in ``eamp`` (the length gauge,
  ``fixed_density_states``), 10.1187 bohr^3. ``b`` is a Drude-like term,
  ``n = -b Omega = 0.0041``, present at ``eamp = 0`` too. The pulse keeps
  ``E0 T = 0.04``, since the response is nonlinear as ``E0^2 T^4``. Measured
  on D22 at ``T = 200, 400, 800, 1600``: +0.27, +0.065, +0.041 and +0.13 per
  cent, and -45 per cent at ``T = 400`` with the saw-tooth left out.


import math
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.calculator import Calculator
from defuqumat.io.pwin import parse_pw_input
from defuqumat.realtime.propagate import propagate
from defuqumat.realtime.pulse import Kick, Pulse
from defuqumat.scf.sawtooth import sawtooth_potential
from defuqumat.system import build_system
from defuqumat.workflows.nscf import fixed_density_states
from defuqumat.workflows.realtime import _kset, _occupied_states

pytestmark = pytest.mark.regression

INPUT = Path(__file__).resolve().parents[1] / "data" / "qe" / "h2-box-tefield.in"

#: Measured on D22 (``notes/incompat/I101.md``, one run): 1.1e-12 with the
#: saw-tooth and 9.9e-6 without it; 1.1e-10 when the states are solved to 1e-8
#: (12-bohr cell).
NULL_CURRENT_ATOL = 1.0e-9
TERM_OFF_CURRENT_MIN = 1.0e-6
#: The adiabatic limit's convergence parameter, with ``E0 T`` held at 0.04: alpha is
#: +0.27, +0.065, +0.041 and +0.13 per cent off at T = 200, 400, 800, 1600 (the rise
#: past 800 is the ``E0^2 T^4`` nonlinearity). 0.5 per cent holds over the whole
#: series; at 12 bohr (1x1x2) the gap is 1.4 per cent at T = 400.
PULSE_LENGTH = 400.0
PULSE_AREA = 0.04
ALPHA_RTOL = 5.0e-3
#: ``-b Omega``, 0.0041 measured: a basis term, below the 1x1x1 12-bohr cell's 0.022.
DRUDE_MAX = 1.0e-2


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


@dataclass(frozen=True)
class _Smooth(Pulse):
    """``E(t) = e0 sin^2(pi t/T)`` on ``[0, T]`` along z; ``kappa = -int E dt``."""

    e0: float
    length: float
    direction: tuple = (0.0, 0.0, 1.0)

    @property
    def natural_duration(self):
        return self.length

    def kappa_at(self, t):
        s = jnp.clip(t, 0.0, self.length)
        integral = 0.5 * s - self.length * jnp.sin(2.0 * math.pi * s / self.length) / (4.0 * math.pi)
        return -self.e0 * integral * jnp.asarray(self.direction)


def _text(eamp):
    return re.sub(r"eamp = [0-9.]+", f"eamp = {eamp!r}", INPUT.read_text())


@lru_cache(maxsize=2)
def _ground(pseudo_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(_text(0.01), pseudo_dir, announce=False)
        result = calculator.get_scf()
    assert result.converged
    null = Kick(0.0, direction=(0.0, 0.0, 1.0))
    kset, _, _ = _kset(calculator.system, null, None, None, False)
    c, states, weights, v_scf, _ = _occupied_states(
        calculator.system, calculator.pseudos, result.density, kset, nbnd=4,
        conv_thr=1e-12, k_batch="default", calculation=None)
    v_saw = sawtooth_potential(c.sawtooth, jnp.asarray(result.density),
                               c.system.structure.positions, c.charges, c.system.cell)[0]
    return calculator, result, c, states, weights, v_scf, v_saw


def _alpha_static(calculator, density, step=1e-3):
    """``-d^2 E_band/d eamp^2`` at the frozen Hxc potential, bohr^3."""
    energies = []
    for s in (-step, 0.0, step):
        system = build_system(parse_pw_input(_text(0.01 + s)))
        _, moved, eig, _ = fixed_density_states(system, calculator.pseudos, density, nbnd=4,
                                                conv_thr=1e-13)
        # Two electrons at eps/2 Hartree each: the band energy in Ha is eps in Ry,
        # the lowest band averaged over the k-points with their weights.
        lowest = np.asarray(eig).reshape(-1, 4)[:, 0]
        w = np.asarray(moved.kpoints.weights).reshape(-1)
        energies.append(float(np.dot(w, lowest) / w.sum()))
    return -(energies[0] - 2.0 * energies[1] + energies[2]) / step**2


def _polarization(run):
    j, t = np.asarray(run.current)[:, 2], np.asarray(run.times)
    return np.concatenate([[0.0], np.cumsum(0.5 * (j[1:] + j[:-1]) * np.diff(t))])


@pytest.mark.slow
def test_a_sawtooth_ground_state_is_stationary_and_the_term_is_what_holds_it(pseudo_dir):
    """No pulse: no current with the saw-tooth in ``v_scf``, a current without it."""
    _, _, c, states, weights, v_scf, v_saw = _ground(pseudo_dir)
    null = Kick(0.0, direction=(0.0, 0.0, 1.0))
    on = propagate(c, states, weights, v_scf, null, dt=0.1, duration=100.0, block_steps=500)
    off = propagate(c, states, weights, v_scf - v_saw, null, dt=0.1, duration=100.0,
                    block_steps=500)
    on_j, off_j = float(np.abs(on.current).max()), float(np.abs(off.current).max())
    print(f"max |J|: with the saw-tooth {on_j:.3e}, without {off_j:.3e}")
    assert on_j < NULL_CURRENT_ATOL
    assert off_j > TERM_OFF_CURRENT_MIN


@pytest.mark.slow
def test_a_slow_pulse_on_a_sawtooth_ground_state_reads_the_static_polarizability(pseudo_dir):
    """``P(t) = alpha E(t)/Omega + b int kappa`` at ``T = 400``, ``E0 = 1e-4``.

    ``alpha`` is read against the length gauge's static value.
    """
    calculator, result, c, states, weights, v_scf, _ = _ground(pseudo_dir)
    alpha = _alpha_static(calculator, np.asarray(result.density))
    e0 = PULSE_AREA / PULSE_LENGTH
    plus = propagate(c, states, weights, v_scf, _Smooth(e0, PULSE_LENGTH), dt=0.1,
                     duration=PULSE_LENGTH, block_steps=500)
    minus = propagate(c, states, weights, v_scf, _Smooth(-e0, PULSE_LENGTH), dt=0.1,
                      duration=PULSE_LENGTH, block_steps=500)
    p = 0.5 * (_polarization(plus) - _polarization(minus))
    t = np.asarray(plus.times)
    ez = np.asarray(plus.efield)[:, 2]
    kz = np.asarray(plus.kappa)[:, 2]
    drift = np.concatenate([[0.0], np.cumsum(0.5 * (kz[1:] + kz[:-1]) * np.diff(t))])
    (fit, b), *_ = np.linalg.lstsq(np.stack([ez / plus.volume, drift], axis=1), p, rcond=None)
    print(f"alpha: static {alpha:.6f}, from the current {fit:.6f} bohr^3; "
          f"n_drude {-b * plus.volume:.3e}")
    assert float(fit) == pytest.approx(alpha, rel=ALPHA_RTOL)
    assert 0.0 < -b * plus.volume < DRUDE_MAX
