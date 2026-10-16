"""Route C's pieces, before any SCF: the stepper, the spinor turn, the mixer's history.

``notes/ORIENTATION-NEXT.md`` Route C turns the input density after every mix. What
can be checked without a Hamiltonian is that each piece turns what it should and
nothing else: the spin-1/2 rotation turns a spinor's magnetization by the same
matrix the density is turned by, the mixer is equivariant under a common turn of
its whole history, and the step converges on a quadratic it is handed.
"""

import numpy as np
import pytest

from defuqumat.forces.torque import rotate_texture
from defuqumat.scf.mixing import get_mixer
from defuqumat.scf.orientation import (
    OrientationStepper,
    rotate_spinors,
    rotation_matrix,
    spinor_rotation,
)

pytestmark = pytest.mark.unit

PAULI = (
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]]),
    np.array([[1, 0], [0, -1]], dtype=complex),
)


def test_the_spinor_turn_turns_the_magnetization_by_the_density_s_matrix():
    rng = np.random.default_rng(1)
    omega = np.array([0.3, -0.5, 0.7])
    npwx = 6
    psi = rng.normal(size=(2, 3, 2 * npwx)) + 1j * rng.normal(size=(2, 3, 2 * npwx))
    turned = rotate_spinors(psi, omega)
    for state, image in ((psi[1, 2], turned[1, 2]),):
        pair = np.stack([state[:npwx], state[npwx:]])
        pair_turned = np.stack([image[:npwx], image[npwx:]])
        m = np.array([np.real(np.einsum("sg,st,tg->", pair.conj(), s, pair)) for s in PAULI])
        m2 = np.array([np.real(np.einsum("sg,st,tg->", pair_turned.conj(), s, pair_turned))
                       for s in PAULI])
        np.testing.assert_allclose(m2, rotation_matrix(omega) @ m, atol=1e-12)
    u = spinor_rotation(omega)
    np.testing.assert_allclose(u.conj().T @ u, np.eye(2), atol=1e-15)


def _field(rng, shape=(4, 3, 3, 3)):
    field = rng.normal(size=shape)
    field[0] += 3.0
    return field


def test_the_mixer_is_equivariant_under_a_common_turn_of_its_history():
    """Turning the history and the new pair is turning the mixed density.

    Anderson's coefficients come from inner products of residuals, which a
    common spin rotation leaves alone, so ``mix(R a, R b)`` after
    ``rotate_history(R)`` must be ``R mix(a, b)``: the property that lets Route C
    keep the history across a step instead of paying for a reset.
    """
    rng = np.random.default_rng(2)
    turn = rotation_matrix(np.array([0.4, 0.2, -0.9]))

    def rotated(vector):
        return np.asarray(rotate_texture(vector.reshape(4, 3, 3, 3), turn)).ravel()

    plain = get_mixer("anderson", beta=0.3, history=4)
    turned = get_mixer("anderson", beta=0.3, history=4)
    pairs = [(_field(rng).ravel(), _field(rng).ravel()) for _ in range(4)]
    for rho_in, rho_out in pairs[:3]:
        plain.mix(rho_in, rho_out)
        turned.mix(rho_in, rho_out)
    turned.rotate_history(rotated)
    rho_in, rho_out = pairs[3]
    expected = rotated(plain.mix(rho_in, rho_out))
    got = turned.mix(rotated(rho_in), rotated(rho_out))
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12)


def test_the_step_converges_on_a_quadratic_and_respects_its_bounds():
    """``E = w . K w / 2`` in the accumulated rotation vector, one null direction.

    The first step is ``first_step`` long along ``-G``, no step is longer than
    ``trust``, nothing moves while ``dr2`` is above ``start``, and BFGS then
    finds the minimum of a curvature two orders below one, the size of an
    anisotropy, in a handful of steps. The null direction (a collinear
    texture's turn about its moment) is never moved along.
    """
    curvature = np.diag([8.0e-5, 8.0e-5, 0.0])
    stepper = OrientationStepper(trust=0.1, first_step=0.05, start=1.0e-5)
    w = np.array([0.5, -0.3, 0.2])
    assert stepper.propose(curvature @ w, accuracy=1.0e-3) is None
    first = stepper.propose(curvature @ w, accuracy=1.0e-6)
    assert np.linalg.norm(first) == pytest.approx(0.05)
    w = w + first
    for _ in range(12):
        step = stepper.propose(curvature @ w, accuracy=1.0e-6)
        if step is None:
            break
        assert np.linalg.norm(step) <= 0.1 + 1e-15
        w = w + step
    assert np.linalg.norm(w[:2]) < 1.0e-6
    assert w[2] == pytest.approx(0.2)


def test_the_first_order_spin_turn_is_the_exact_one_to_first_order():
    """``spin_turned`` is what the PAW torque differentiates through the states."""
    from defuqumat.scf.orientation import spin_turned

    rng = np.random.default_rng(4)
    psi = rng.normal(size=(2, 3, 10)) + 1j * rng.normal(size=(2, 3, 10))
    for size in (1.0e-4, 1.0e-6):
        omega = size * np.array([0.3, -0.5, 0.7])
        gap = np.abs(np.asarray(spin_turned(psi, omega)) - rotate_spinors(psi, omega)).max()
        assert gap < 10 * size ** 2 * np.abs(psi).max()


def test_a_measured_flat_direction_is_held_and_the_rest_converges():
    """The curvature freeze, an option: a direction BFGS finds flatter than ``flat_curvature``.

    ``E = w . K w / 2`` with curvatures 8e-5, 8e-5 and 1e-9 Ry/rad^2, the last a
    thousand times below the default threshold of 1e-6: stepping along it would
    need an angle far past the trust bound for a torque at the noise, which is
    how PAW nickel's in-plane angle kicked its SCF. Once the curvature there has
    been measured the stepper holds that direction and still converges the two
    it can resolve; without the freeze (``flat_curvature=None``) it keeps
    stepping along the flat one at the trust bound.
    """
    curvature = np.diag([8.0e-5, 8.0e-5, 1.0e-9])
    for flat, expect_held in ((1.0e-6, True), (None, False)):
        stepper = OrientationStepper(trust=0.1, first_step=0.05, start=1.0e-5,
                                     flat_curvature=flat)
        w = np.array([0.4, -0.3, 0.3])
        held_seen = False
        for _ in range(25):
            step = stepper.propose(curvature @ w, accuracy=1.0e-6)
            held_seen = held_seen or len(stepper.flat) > 0
            if step is None:
                break
            w = w + step
        assert held_seen is expect_held
        if expect_held:
            np.testing.assert_allclose(np.abs(stepper.flat[0]), [0.0, 0.0, 1.0],
                                       atol=1e-6)
            assert np.linalg.norm(w[:2]) < 1.0e-5


# ---------------------------------------------------------------------------
# I78: rotate_moments on a run that keeps its symmetry.
# ---------------------------------------------------------------------------

from pathlib import Path
from types import SimpleNamespace

from defuqumat.scf.driver import _refuse_rotating_moments
from defuqumat.system.symmetry import Symmetries, orientation_free_symmetries

_COBALT = Path(__file__).resolve().parents[1] / "data" / "qe" / "co-tetragonal-relaxed-mae.in"


def _cobalt_system(*, nosym: bool, angle: float = 45.0):
    """Tetragonal cobalt with the coupling, ``angle`` degrees off ``c`` in the xz plane."""
    from defuqumat.system.builder import build_system
    from defuqumat.io.pwin import parse_pw_input

    text = _COBALT.read_text().replace(
        "angle1(1) = 0.0, angle2(1) = 0.0,", f"angle1(1) = {angle}, angle2(1) = 0.0,")
    if not nosym:
        text = text.replace("nosym = .true.,", "")
    return build_system(parse_pw_input(text))


def _stub(*, nosym=False, turning=False, hubbard=False, paw=False):
    return SimpleNamespace(
        is_hubbard=hubbard, is_paw=paw,
        system=SimpleNamespace(nosym=nosym, turning_moments=turning))


def test_a_symmetric_run_without_the_turning_group_is_refused():
    """The starting orientation's group is refused; nosym and the turning group are not.

    Not because the run breaks it (it does not, see the stepper test below) but
    because it pins the moments to its fixed subspace (I78)."""
    with pytest.raises(ValueError, match="with_turning_moments"):
        _refuse_rotating_moments(_stub(), None, True)
    # Either way out of the refusal is accepted.
    _refuse_rotating_moments(_stub(nosym=True), None, True)
    _refuse_rotating_moments(_stub(turning=True), None, True)


def test_the_sibling_refusals_of_rotate_moments_still_fire_on_the_turning_group():
    """Coupling, Hubbard U and field are refused whatever the group."""
    with pytest.raises(ValueError, match="spin-orbit"):
        _refuse_rotating_moments(_stub(turning=True), None, False)
    with pytest.raises(NotImplementedError, match="Hubbard"):
        _refuse_rotating_moments(_stub(turning=True, hubbard=True), None, True)
    with pytest.raises(NotImplementedError, match="magnetic field"):
        _refuse_rotating_moments(_stub(turning=True), object(), True)


def test_the_orientation_free_group_keeps_identity_and_inversion_only():
    """``det(R) R = 1`` and no time reversal; translations are carried."""
    e = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    i = ((-1, 0, 0), (0, -1, 0), (0, 0, -1))
    c2 = ((-1, 0, 0), (0, -1, 0), (0, 0, 1))
    group = Symmetries(rotations=(e, c2, i, e, i),
                       translations=((0, 0, 0), (0, 0, 0), (0, 0, 0.5),
                                     (0.5, 0.5, 0.5), (0, 0, 0)),
                       time_reversed=(0, 0, 0, 0, 1))
    kept = orientation_free_symmetries(group)
    assert kept.rotations == (e, i, e)
    assert kept.translations == ((0, 0, 0), (0, 0, 0.5), (0.5, 0.5, 0.5))
    assert not kept.magnetic


def test_the_turning_group_rebuilds_the_cobalt_k_set_between_the_two_ends():
    """Cobalt 45 degrees off ``c``: 18 points nosym, 10 with ``{E, I}``, fewer at the start's group.

    The 3x3x2 grid has two points equal to their own inverse (``k_z = 0, 1/2``
    at ``k_x = k_y = 0``), so inversion leaves ``(18 - 2) / 2 + 2 = 10``. The
    starting orientation's magnetic group is larger than ``{E, I}`` (the
    two-fold axis along ``y`` with time reversal, and its mirror), so its
    wedge is smaller still: the k-set the refusal keeps out, since that group
    pins the moment to the ``xz`` plane.
    """
    full = _cobalt_system(nosym=True)
    start = _cobalt_system(nosym=False)
    turning = start.with_turning_moments()

    assert full.kpoints.nk == 18
    assert turning.kpoints.nk == 10
    assert start.kpoints.nk < 10
    assert start.symmetry_group().nsym > 2

    group = turning.symmetry_group()
    assert group.nsym == 2 and not group.magnetic
    assert {tuple(map(tuple, r)) for r in group.rotation_array()} == {
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, -1, 0), (0, 0, -1))}
    # The k-set and the symmetrisation group are one group: the grid rebuilt
    # from grid_symmetry is the k-set the System carries.
    rotations, time_reversal, t_rev = turning.grid_symmetry()
    assert len(rotations) == 2 and not time_reversal
    assert np.isclose(float(np.sum(turning.kpoints.weights)),
                      float(np.sum(full.kpoints.weights)))
    # Under nosym the flag changes nothing about the sampling.
    assert full.with_turning_moments().kpoints.nk == 18
    # And switching it off gives the starting group's wedge back.
    assert turning.with_turning_moments(False).kpoints.nk == start.kpoints.nk


def test_a_paw_run_on_the_turning_group_is_refused_and_nosym_paw_is_not():
    """PAW keeps the refusal under symmetry (I78 measured it on ultrasoft only)."""
    with pytest.raises(NotImplementedError, match="PAW"):
        _refuse_rotating_moments(_stub(turning=True, paw=True), None, True)
    _refuse_rotating_moments(_stub(nosym=True, paw=True), None, True)


def test_the_stepper_keeps_the_starting_group_when_the_torque_respects_it():
    """Every step is built from torques alone, so it commutes with their symmetry.

    The 45-degree cobalt start has ``C2y`` and ``sigma_y`` with time reversal;
    each acts on a rotation vector as ``det(R) R = C2y``, so a torque that
    respects the group lies along ``y``, and the turn it gives keeps the moment
    in the ``xz`` plane. This is why the starting group's wedge stays complete
    along the run (I78, the verifier's argument) and also why that group pins
    the moment. Fed torques along ``y`` of varying size, through the first step,
    the BFGS updates and the trust radius, every step stays along ``y``.
    """
    stepper = OrientationStepper()
    for g in (0.3, 0.1, -0.05, 0.02, 0.001, -0.0004):
        step = stepper.propose(np.array([0.0, g, 0.0]), accuracy=1.0e-9)
        assert step is not None
        assert step[0] == 0.0 and step[2] == 0.0
    m = stepper.total @ np.array([np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
    assert abs(m[1]) < 1.0e-15
