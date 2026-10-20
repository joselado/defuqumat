"""The potential updated in time on an ultrasoft or PAW dataset (``notes/incompat/I98.md``).

Two statements, each with the term switched off beside it:

* **the start is a fixed point and the energy is conserved.** With no field the
  update's ``v(t) - v_scf`` stays at round-off; after a kick, with the field
  zero from then on, the conserved functional of
  :class:`~defuqumat.realtime.selfconsistent._Run` holds to 1e-10 while the
  potential moves by 5e-3 Ry. That functional's derivative in ``becsum`` is
  ``newd(vltot + v(t)) + ddd(t)``, so its conservation tests the augmentation
  charge at ``k + kappa``, ``D(t)`` and PAW's one-centre ``D(t)`` together.
  With the augmentation charge of ``rho(t)`` held at the start's ``becsum``
  it fails by 2.8e-3 Ha. PAW runs here with ``input_dft = 'sla+pw'``: under
  the dataset's own Perdew-Zunger correlation the one-centre energy jumps by
  2.5e-7 Ry when a point of the radial all-electron density crosses
  ``rs = 1``, where PZ's two branches meet with a small mismatch, which the
  smooth Perdew-Wang form does not have.
* **the local-field shift of eps from a kick is the Sternheimer one.**
  ``eps_hxc(i eta) - eps_frozen(i eta)`` from two propagations on the same
  k-set against ``get_dielectric_tensor`` with ``screening = 'full'`` less
  ``'none'``; the subtraction removes the velocity gauge's band-curvature
  term, which neither propagation can tell from the response on a coarse mesh.
"""

import re
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

from defuqumat import Calculator
from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.realtime.propagate import propagate
from defuqumat.realtime.pulse import EV_TO_HA, Kick
from defuqumat.realtime.spectra import conductivity_from_kick
from defuqumat.scf import Calculation
from defuqumat.system import build_system
from defuqumat.system.kpoints import KPoints

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
#: Gamma and one point of no symmetry.
KSET = (np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.05]]), np.array([0.5, 0.5]))


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


def _input(tmp_path, case, ecut, dual, extra="", mesh=None):
    text = (CASES / f"{case}.in").read_text()
    text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", f"ecutwfc = {ecut}", text)
    text = re.sub(r"ecutrho\s*=\s*[0-9.dD+-]+", f"ecutrho = {dual * ecut}", text)
    if extra:
        text = re.sub(r"&system", "&system\n    " + extra, text, count=1)
    if mesh is not None:
        text = re.sub(r"K_POINTS.*\n\s*[0-9 ]+\n", f"K_POINTS (automatic)\n {mesh}\n", text)
    path = tmp_path / f"{case}-{ecut}-{abs(hash(extra + str(mesh))) % 1000}.in"
    path.write_text(text)
    return path


def _dense_states(calc, v_scf, ddd):
    """The occupied generalised eigenstates of the dense ``H(k)``, ``S(k)`` on :data:`KSET`."""
    nk, nocc = len(KSET[1]), int(round(calc.nelec / 2))
    kcart = np.asarray(calc.system.kpoints.cartesian(calc.system.cell))
    ham = calc.at_kcart(jnp.asarray(kcart)).hamiltonian_from(calc.local_terms(v_scf, ddd))[0]
    mask = np.asarray(ham.state_mask)
    psi = np.zeros((nk, nocc, mask.shape[1]), dtype=complex)
    for ik in range(nk):
        keep = np.flatnonzero(mask[ik])
        h, s = np.asarray(ham.matrix(ik)), np.asarray(ham.overlap_matrix(ik))
        psi[ik][:, keep] = sla.eigh(h[np.ix_(keep, keep)], s[np.ix_(keep, keep)])[1][:, :nocc].T
    return psi, np.repeat(2.0 * KSET[1][:, None], nocc, axis=1)


@pytest.mark.parametrize("case, extra", [
    ("si-epsilon-us", ""),
    ("si-epsilon-paw", "input_dft = 'sla+pw',"),
])
def test_the_augmented_update_is_a_fixed_point_and_conserves_its_energy(pseudo_dir, tmp_path,
                                                                        case, extra,
                                                                        monkeypatch):
    """No field: ``|v(t) - v_scf|`` at 5e-12; a kick of 0.01/bohr: the energy to 1e-10 Ha.

    8 Ry, dual 8, on :data:`KSET` at the superposition of atomic charges, 40
    steps of 0.05 with ``potential = 'hxc'``. Measured: ``|v - v_scf|`` 5.3e-12
    (ultrasoft) and 4.3e-12 (PAW) with no field; after the kick the potential
    moves by 4.7e-3 Ry and the energy by 1.1e-10 and 9.9e-11 Ha, linearly in
    time (the step's). With the augmentation charge of ``rho(t)`` held at the
    start's ``becsum``, the update without the term the refusal named, the
    energy moves by 2.8e-3 Ha in both.
    """
    path = _input(tmp_path, case, 8.0, 8.0, extra)
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    system = eqx.tree_at(lambda s: s.kpoints, system, KPoints(coords=KSET[0], weights=KSET[1]))
    calc = Calculation(system, pseudos)
    v_scf = calc.potential(calc.starting_density()).v_scf
    ddd = None
    psi, w = _dense_states(calc, v_scf, None)
    if calc.is_paw:
        _, ddd = calc.onecenter(calc.becsum(jnp.asarray(psi)[None], jnp.asarray(w)[None]))
        psi, w = _dense_states(calc, v_scf, ddd)

    def run(strength):
        return propagate(calc, jnp.asarray(psi), w, v_scf,
                         Kick(strength=strength, direction=(1.0, 0.0, 0.0)), dt=0.05,
                         start=0.0, duration=2.0, potential="hxc", block_steps=20,
                         k_batch=None, ddd_paw=ddd)

    still = run(0.0)
    assert still.extras["potential_change"] < 1e-10
    assert np.abs(still.current - still.current[0]).max() < 1e-10
    kicked = run(0.01)
    assert kicked.extras["potential_change"] > 1e-3, "the potential moved"
    assert np.ptp(kicked.energy) < 1e-9, kicked.energy
    assert kicked.norm_drift < 1e-9

    held = calc.becsum(jnp.asarray(psi)[None], jnp.asarray(w)[None])
    original = Calculation.augmented
    monkeypatch.setattr(Calculation, "augmented",
                        lambda self, rho, becsum_: original(self, rho, held))
    assert np.ptp(run(0.01).energy) > 1e-4, "without the term the energy is not conserved"


def _shift(pseudo_dir, tmp_path, case, dual, monkeypatch=None):
    """``(hxc - frozen, full - none)``: the kick's local-field shift of ``eps(i eta)`` and the static one.

    2x2x2 unshifted at 12 Ry, ``conv_thr = 1e-12``, a kick of 1e-3/bohr along x,
    ``dt = 0.2``, 22/(0.2 eV) of propagation, ``eps`` read at ``eta = 0.4`` eV
    without the static part of ``J`` removed. For PAW ``screening = 'none'``
    still carries ``PAW_dpotential``, so it is taken with ``paw_response`` set to
    zero, the independent-particle response the frozen propagation computes.
    """
    import defuqumat.response.efield as efield

    path = _input(tmp_path, case, 12.0, dual, mesh="2 2 2 0 0 0")
    calc = Calculator.from_file(path, pseudo_dir=pseudo_dir, announce=False, conv_thr=1e-12)
    calc.get_scf()
    full = float(calc.get_dielectric_tensor(screening="full", born_charges=False,
                                            threshold=1e-14).epsilon[0, 0])
    if calc.calculation.is_paw:
        assert monkeypatch is not None, "a PAW case patches paw_response"
        original = efield.paw_response
        monkeypatch.setattr(efield, "paw_response",
                            lambda c, d, b: jnp.zeros_like(original(c, d, b)))
    none = float(calc.get_dielectric_tensor(screening="none", born_charges=False,
                                            threshold=1e-14).epsilon[0, 0])
    if monkeypatch is not None:
        monkeypatch.undo()
    kappa, direction = 1e-3, (1.0, 0.0, 0.0)
    eps = {}
    for potential in ("frozen", "hxc"):
        run = calc.get_realtime(Kick(strength=kappa, direction=direction), dt=0.2, start=0.0,
                                duration=22.0 / (0.2 * EV_TO_HA), potential=potential,
                                block_steps=500)
        eps[potential] = complex(conductivity_from_kick(
            run.times, run.current, kappa, direction, [0.0], 0.4 * EV_TO_HA).epsilon[0, 0])
    return eps["hxc"].real - eps["frozen"].real, full - none


@pytest.fixture(scope="module")
def norm_conserving_shift(pseudo_dir, tmp_path_factory):
    """The control's ``(kick shift, static shift)`` on the same mesh, computed once for both cases."""
    return _shift(pseudo_dir, tmp_path_factory.mktemp("control"), "si-epsilon-unshifted", 4.0)


@pytest.mark.parametrize("case", ["si-epsilon-us", "si-epsilon-paw"])
def test_the_local_field_shift_of_a_kick_is_the_sternheimer_one(pseudo_dir, tmp_path, case,
                                                                monkeypatch,
                                                                norm_conserving_shift):
    """The augmented shift misses the static one by what the norm-conserving shift misses.

    ``eps_hxc(i eta) - eps_frozen(i eta)`` against ``get_dielectric_tensor``'s
    ``'full'`` less ``'none'``, measured on D22 (``notes/incompat/I98.md``):

    =========  ==============  ==============  =====================
    2x2x2      kick shift      static shift    relative residual
    =========  ==============  ==============  =====================
    ultrasoft  -3.18656        -3.31521        -3.88e-2
    PAW        -3.29523        -3.42789        -3.87e-2
    NC         -2.40127        -2.49258        -3.66e-2
    =========  ==============  ==============  =====================

    The residual is the kick route's on a coarse mesh, the same within 0.2
    points for the three datasets, and it does not move with ``dt`` (0.1 reads
    -3.86e-2 and -3.84e-2) or with ``eta`` (0.2 eV reads -3.86e-2 and
    -3.76e-2). It falls with the k-mesh, the convergence parameter: on 4x4x4
    (8 points) ultrasoft reads -1.03001 against -1.05207 (-2.10e-2), PAW
    -1.05122 against -1.07391 (-2.11e-2) and the norm-conserving control
    -0.86662 against -0.88344 (-1.90e-2), at 0.2 eV -1.65e-2, -1.63e-2 and
    -1.67e-2. With the augmentation charge of
    ``rho(t)`` held at the ground state's ``becsum`` (the term the refusal named,
    switched off) the residual is +1.64e-1 and +1.54e-1, so the test asks that
    the augmented residual sits within 1e-2 of the norm-conserving one.
    """
    shift, static = _shift(pseudo_dir, tmp_path, case, 8.0, monkeypatch)
    control, control_static = norm_conserving_shift
    residual, control_residual = shift / static - 1.0, control / control_static - 1.0
    assert abs(residual) < 5e-2, (shift, static)
    assert abs(residual - control_residual) < 1e-2, (residual, control_residual)
