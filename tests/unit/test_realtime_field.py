"""Real-time propagation of a ground state converged under a magnetic field (I64).

The field such a ground state converged under is not the input's: ``reducebf``
multiplies the applied field down over the SCF and the fixed-spin-moment scheme
replaces it, so the pair ``SCFResult.magnetic_field`` and ``.field_scale`` is
state, as ``becsum`` is. The workflows took the states from the fixed-density
solve, which refuses without that pair, and froze ``calc.potential(density)``,
which rebuilds the input's field at full strength; they now take the pair and
freeze it, and the driver's guard on any field is gone.

The gate keeps the guards: the driver admits a field, the workflows take the
pair and refuse without it, and the updated potential refuses a penalty
constraint while it admits an applied or a feedback field. The number is the
slow test at the end, an SCF under a reduced local field and three
zero-amplitude propagations: the workflow's states are stationary, the SCF's
own states are stationary under the frozen potential built from the converged
pair, and under the one built from the input field they are not.
"""

import inspect
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation
from defuqumat.system.builder import build_system

pytestmark = pytest.mark.unit

_SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0
  nspin = 2, starting_magnetization(1) = 0.1, tot_magnetization = 0.0
  {extra}
/
&electrons
  conv_thr = 1.0d-11
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
{cards}
"""

#: A field on the first atom's sphere only, decayed by ``reducebf``: spatially
#: varying, so a wrong strength is not a rigid shift of a channel and the
#: states it was not solved in are not its eigenstates.
_LOCAL = dict(extra="nosym = .true., reducebf = 0.8",
              cards="LOCAL_MAGNETIC_FIELDS\n 0.0 0.0 0.05\n 0.0 0.0 0.0\n")
_BY_HAND = dict(extra="B_field(3) = 0.01", cards="")
_FSM = dict(extra="constrained_magnetization = 'fsm', lambda = 0.02", cards="")
_PENALTY = dict(extra="constrained_magnetization = 'atomic', lambda = 0.5", cards="")


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


def _calculation(case, pseudo_dir):
    system = build_system(parse_pw_input(_SILICON.format(**case)))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, Calculation(system, pseudos)


@pytest.mark.parametrize("case", [_BY_HAND, _LOCAL, _FSM, _PENALTY],
                         ids=["B_field", "reduced local field", "fsm", "penalty"])
def test_the_driver_admits_a_ground_state_under_a_field(case, pseudo_dir):
    """The guard is gone: nothing in the driver rebuilds the field, it is in ``v_scf``."""
    from defuqumat.realtime.propagate import require_a_realtime_regime

    _, _, calculation = _calculation(case, pseudo_dir)
    assert calculation.magnetic_field is not None
    require_a_realtime_regime(calculation)


def test_the_updated_potential_refuses_a_penalty_and_admits_a_linear_field(pseudo_dir):
    """A penalty is quadratic in m and would move with rho(t); a Zeeman term would not."""
    from defuqumat.realtime.selfconsistent import require_a_potential_mode

    for case in (_BY_HAND, _FSM):
        _, _, calculation = _calculation(case, pseudo_dir)
        for potential in ("frozen", "hartree", "hxc"):
            require_a_potential_mode(calculation, potential, None, None)
    _, _, penalty = _calculation(_PENALTY, pseudo_dir)
    require_a_potential_mode(penalty, "frozen", None, None)
    for potential in ("hartree", "hxc"):
        with pytest.raises(NotImplementedError, match="constrained_magnetization = 'atomic'"):
            require_a_potential_mode(penalty, potential, None, None)


def test_every_realtime_workflow_takes_the_converged_field():
    """And forwards it: ``run_hhg`` and ``run_third_harmonic`` call a sibling."""
    from defuqumat.workflows import realtime

    entries = (realtime.run_realtime, realtime.run_realtime_dielectric, realtime.run_hhg,
               realtime.run_harmonic_orders, realtime.run_third_harmonic,
               realtime.run_nonlinear_spectrum)
    for entry in entries:
        parameters = inspect.signature(entry).parameters
        for name in ("field", "field_scale"):
            assert name in parameters, f"{entry.__name__} is missing {name}"
        body = inspect.getsource(entry)
        assert "field=field" in body and "field_scale=field_scale" in body, entry.__name__
    body = inspect.getsource(realtime._solved_states)
    assert body.count("field_scale") >= 3, "the pair goes into the solve and the potential"


def test_a_workflow_refuses_a_field_it_was_not_given(pseudo_dir):
    """The fixed-density solve's refusal, which is what stood before the driver's guard."""
    from defuqumat.realtime.pulse import Kick
    from defuqumat.workflows.realtime import run_realtime

    system, pseudos, calculation = _calculation(_LOCAL, pseudo_dir)
    density = calculation.starting_density()
    with pytest.raises(ValueError, match="needs the field the SCF ended with"):
        run_realtime(system, pseudos, density, Kick(0.0), duration=1.0)


@lru_cache(maxsize=1)
def _ground_state(pseudo_dir):
    from defuqumat.scf import run_scf

    system, pseudos, calculation = _calculation(_LOCAL, pseudo_dir)
    return system, pseudos, calculation, run_scf(system, pseudos, conv_thr=1e-11)


def stationarity_under_a_reduced_field(pseudo_dir, duration=20.0, dt=0.1):
    """The numbers of the slow test, also what ``review/incompat/I64`` runs on D22.

    Returns a dict: the converged ``field_scale``; ``excited`` and the band
    energy's spread for the workflow at a frozen and at an updated (``'hxc'``)
    potential; and ``excited`` for the SCF's own occupied states under the
    frozen potential built from the converged pair and from the input field.
    """
    from defuqumat.realtime.propagate import propagate
    from defuqumat.realtime.pulse import Kick
    from defuqumat.workflows.realtime import run_realtime

    system, pseudos, calculation, result = _ground_state(pseudo_dir)
    out = {"field_scale": float(result.field_scale), "iterations": int(result.iterations)}
    for potential in ("frozen", "hxc"):
        run = run_realtime(system, pseudos, result.density, Kick(0.0), dt=dt,
                           duration=duration, calculation=calculation,
                           field=result.magnetic_field, field_scale=result.field_scale,
                           potential=potential)
        out[f"workflow_{potential}_excited"] = float(run.excited)
        out[f"workflow_{potential}_energy_spread"] = float(np.ptp(run.energy))
        out[f"workflow_{potential}_current_spread"] = float(np.ptp(run.current))
    # The SCF's own states, which were solved under the converged field: the
    # A/B the entry's check stands on, since the workflow re-solves its states
    # under whatever field it is handed and is stationary either way.
    psi = np.asarray(result.wavefunctions)
    psi = psi if psi.ndim == 4 else psi[None]
    wg = np.asarray(result.occupations)
    wg = wg if wg.ndim == 3 else wg[None]
    nocc = int(np.flatnonzero(np.any(wg > 1e-10, axis=(0, 1)))[-1]) + 1
    states, weights = psi[:, :, :nocc], wg[:, :, :nocc]
    density = jnp.asarray(result.density)
    pairs = {"converged": (float(result.field_scale), result.magnetic_field),
             "input": (1.0, None)}
    for name, (scale, field) in pairs.items():
        v_scf = calculation.potential(density, scale, field).v_scf
        run = propagate(calculation, states, weights, v_scf, Kick(0.0), dt=dt,
                        duration=duration)
        out[f"scf_states_{name}_excited"] = float(run.excited)
        out[f"scf_states_{name}_energy_spread"] = float(np.ptp(run.energy))
    return out


@pytest.mark.slow
def test_under_a_reduced_field_a_zero_pulse_leaves_the_states_stationary(pseudo_dir):
    """The check of I64, with the alternative that reads differently beside it.

    Silicon at ``nspin = 2`` with a 0.05 Ry field on one atom's sphere and
    ``reducebf = 0.8``, converged to 1e-11; 200 steps of 0.1 a.u. at zero
    amplitude. The numbers are in ``notes/incompat/I64.md``.
    """
    numbers = stationarity_under_a_reduced_field(pseudo_dir)
    assert numbers["field_scale"] < 0.5, "reducebf must have moved the field"
    for potential in ("frozen", "hxc"):
        assert numbers[f"workflow_{potential}_excited"] < 1e-8
        assert numbers[f"workflow_{potential}_energy_spread"] < 1e-9
        assert numbers[f"workflow_{potential}_current_spread"] < 1e-10
    assert numbers["scf_states_converged_excited"] < 1e-8
    assert numbers["scf_states_input_excited"] > 1e3 * max(
        numbers["scf_states_converged_excited"], 1e-12)


def energy_after_a_kick_under_a_field(pseudo_dir, strength=0.05, dt=0.05, duration=50.0,
                                      input_field_in_u=False):
    """The conserved energy's drift after a kick at ``potential = 'hxc'``, field cell.

    ``input_field_in_u`` puts the input field back into ``U`` at full strength,
    the closure as it stood before I64, by patching ``Calculation.potential``
    for the duration: ``U[rho(t)] - U[rho0]`` still cancels it from the
    potential, but the energy, whose ``E_U`` is ``ehart + etxc``, is then one
    ``int v_field (rho(t) - rho0)`` from the potential the states move in.
    Returns ``(max |E(t) - E(0)|, potential_change)``.
    """
    from defuqumat.realtime.pulse import Kick
    from defuqumat.workflows.realtime import run_realtime

    system, pseudos, calculation, result = _ground_state(pseudo_dir)
    kind = type(calculation)
    original = kind.potential

    def full_strength(self, rho, field_scale=1.0, field=None, tau=None):
        return original(self, rho, 1.0 if field_scale == 0.0 else field_scale, field, tau)

    if input_field_in_u:
        kind.potential = full_strength
    try:
        run = run_realtime(system, pseudos, result.density,
                           Kick(strength=strength, direction=(1.0, 0.0, 0.0)), dt=dt,
                           start=0.0, duration=duration, calculation=calculation,
                           field=result.magnetic_field, field_scale=result.field_scale,
                           potential="hxc", block_steps=50)
    finally:
        kind.potential = original
    return (float(np.abs(run.energy - run.energy[0]).max()),
            float(run.extras["potential_change"]))


@pytest.mark.slow
def test_under_a_field_the_updated_potential_conserves_the_energy(pseudo_dir):
    """A kick at ``'hxc'`` on the field cell: the drift is the step's, not the field's.

    The A/B of the ``field_scale = 0`` in ``_Run``'s closure; the numbers are in
    ``notes/incompat/I64.md``.
    """
    drift, change = energy_after_a_kick_under_a_field(pseudo_dir)
    wrong, _ = energy_after_a_kick_under_a_field(pseudo_dir, input_field_in_u=True)
    assert change > 1e-3, "the potential moved"
    assert drift < 1e-7
    assert wrong > 100.0 * drift
