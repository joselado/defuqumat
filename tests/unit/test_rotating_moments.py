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
    rotate_ns,
    rotate_spinors,
    rotation_matrix,
    spin_turned,
    spin_turned_ns,
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


def _stub(*, nosym=False, turning=False, hubbard=False, paw=False,
          tensor_moments=False):
    return SimpleNamespace(
        is_hubbard=hubbard, is_paw=paw,
        hubbard=SimpleNamespace(constraints=object() if tensor_moments else None),
        system=SimpleNamespace(nosym=nosym, turning_moments=turning))


def _field_stub(constraint="none", reducebf=1.0):
    return SimpleNamespace(constraint=constraint, reducebf=reducebf, has_field=True)


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
    """The coupling is required whatever the group; a U and a field are admitted (I112, I113).

    What stays refused of the two: a U on PAW and a fixed tensor moment; a
    penalty or feedback constraint, ``reducebf`` and a field without ``nosym``.
    """
    with pytest.raises(ValueError, match="spin-orbit"):
        _refuse_rotating_moments(_stub(turning=True), None, False)
    # A Hubbard U on norm-conserving or ultrasoft is turned with the step.
    _refuse_rotating_moments(_stub(nosym=True, hubbard=True), None, True)
    _refuse_rotating_moments(_stub(turning=True, hubbard=True), None, True)
    with pytest.raises(NotImplementedError, match="Hubbard U on a PAW"):
        _refuse_rotating_moments(_stub(nosym=True, hubbard=True, paw=True), None, True)
    with pytest.raises(NotImplementedError, match="tensor moment"):
        _refuse_rotating_moments(
            _stub(nosym=True, hubbard=True, tensor_moments=True), None, True)
    # A fixed applied field under nosym is in the step.
    _refuse_rotating_moments(_stub(nosym=True), _field_stub(), True)
    for constraint in ("atomic", "atomic direction", "total", "total direction",
                       "atomic texture", "fsm", "atomic fsm"):
        with pytest.raises(NotImplementedError, match="constrained moment"):
            _refuse_rotating_moments(_stub(nosym=True), _field_stub(constraint), True)
    with pytest.raises(NotImplementedError, match="reducebf"):
        _refuse_rotating_moments(_stub(nosym=True), _field_stub(reducebf=0.5), True)
    with pytest.raises(NotImplementedError, match="applied magnetic field needs nosym"):
        _refuse_rotating_moments(_stub(turning=True), _field_stub(), True)


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


def _spinor_projection_problem(rng, nk=2, nbnd=3, npwx=7, ldim=5):
    """Pure up and down projector columns on one shell, and random spinor states."""
    phi = rng.normal(size=(nk, npwx, ldim)) + 1j * rng.normal(size=(nk, npwx, ldim))
    wfcU = np.zeros((nk, 2 * npwx, 2 * ldim), dtype=complex)
    wfcU[:, :npwx, :ldim] = phi
    wfcU[:, npwx:, ldim:] = phi
    columns = np.array([[list(range(ldim)), list(range(ldim, 2 * ldim))]])
    mask = np.ones((1, ldim), dtype=bool)
    psi = (rng.normal(size=(1, nk, nbnd, 2 * npwx))
           + 1j * rng.normal(size=(1, nk, nbnd, 2 * npwx)))
    weights = rng.random((1, nk, nbnd))
    return wfcU, psi, weights, columns, mask


def test_the_occupation_matrix_turns_with_the_states_as_conj_u_n_u_transpose():
    """``ns`` of the turned states is :func:`rotate_ns` of ``ns`` (I112), and to first order too.

    ``new_ns_nc``'s ``N^{st} = sum conj(p_s) p_t`` turns as ``conj(U) N U^T`` when
    every spinor turns by ``U``; on pure up and down projector columns with one
    radial function, which is DFT+U's ``updown`` basis on a norm-conserving
    dataset, the identity is exact. The first-order turn is the derivative of
    the occupation matrix of :func:`spin_turned`'s states, which is what the
    Hubbard share of the torque differentiates.
    """
    import jax
    import jax.numpy as jnp

    from defuqumat.hubbard.occupations import occupation_matrix

    wfcU, psi, weights, columns, mask = _spinor_projection_problem(np.random.default_rng(4))

    def ns_of(states):
        return np.asarray(occupation_matrix(
            jnp.asarray(wfcU), jnp.asarray(states), jnp.asarray(weights),
            jnp.asarray(columns), jnp.asarray(mask)))

    ns = ns_of(psi)
    omega = np.array([0.3, -0.7, 0.4])
    turned = ns_of(rotate_spinors(psi, omega))
    assert np.abs(turned - ns).max() > 1.0
    np.testing.assert_allclose(rotate_ns(ns, omega), turned, atol=1e-12)

    def ns_first_order(o):
        return occupation_matrix(jnp.asarray(wfcU), spin_turned(jnp.asarray(psi), o),
                                 jnp.asarray(weights), jnp.asarray(columns),
                                 jnp.asarray(mask))

    for axis in np.eye(3):
        expected = jax.jvp(ns_first_order, (jnp.zeros(3),), (jnp.asarray(axis),))[1]
        got = jax.jvp(lambda o: spin_turned_ns(jnp.asarray(ns), o),
                      (jnp.zeros(3),), (jnp.asarray(axis),))[1]
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), atol=1e-12)


def test_the_packed_mixer_vector_turns_its_ns_block_and_nothing_else():
    """The ``ns`` block of a history vector, packed as ``_mix`` packs it, turns as ``rotate_ns``.

    A density-like head, a ``becsum``-like block, then the complex ``ns`` viewed
    as interleaved reals, then a ``tau``-like tail. Only the ``ns`` block moves,
    and it unpacks to ``conj(U) N U^T`` exactly; the same call on a block one
    entry off would not unpack to it, which is what an offset error would give.
    """
    from defuqumat.scf.driver import _ns_dtypes, _pack_ns, _turn_packed_ns
    from defuqumat.scf.orientation import rotate_ns

    rng = np.random.default_rng(11)
    ns = rng.normal(size=(4, 1, 5, 5)) + 1j * rng.normal(size=(4, 1, 5, 5))
    dtype, real = _ns_dtypes(ns)
    head, becsum, tail = rng.normal(size=37), rng.normal(size=12), rng.normal(size=9)
    vector = np.concatenate([head, becsum, _pack_ns(ns, dtype, real), tail])
    omega = np.array([0.2, -0.4, 0.3])
    offset = head.size + becsum.size
    turned = _turn_packed_ns(vector, offset, ns.shape, dtype, omega)
    size = 2 * ns.size
    np.testing.assert_array_equal(turned[:offset], vector[:offset])
    np.testing.assert_array_equal(turned[offset + size:], vector[offset + size:])
    got = np.ascontiguousarray(turned[offset:offset + size]).view(dtype).reshape(ns.shape)
    np.testing.assert_allclose(got, rotate_ns(ns, omega), rtol=0, atol=1e-14)
    assert np.abs(got - ns).max() > 0.1
    shifted = _turn_packed_ns(vector, offset + 1, ns.shape, dtype, omega)
    wrong = np.ascontiguousarray(shifted[offset:offset + size]).view(dtype).reshape(ns.shape)
    assert np.abs(wrong - rotate_ns(ns, omega)).max() > 0.1


def test_the_hubbard_torque_is_the_derivative_of_the_pairing_under_the_exact_turn():
    """:func:`_hubbard_torque` against a central difference of ``Re sum(R(w) ns . v)``.

    The difference is taken with the *exact* turn, :func:`rotate_ns`, at
    ``h = 1e-3`` and ``1e-4``; its error falls as ``h^2``, so the two steps
    bracket the forward derivative and the identity is held at the smaller.
    """
    from defuqumat.scf.driver import _hubbard_torque

    rng = np.random.default_rng(7)
    shape = (2, 2, 1, 5, 5)
    a = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    # Hermitian in the combined (spin, m) index, as ns and v_ns are.
    def hermitian(x):
        full = x.transpose(2, 0, 3, 1, 4).reshape(1, 10, 10)
        full = 0.5 * (full + np.conj(np.swapaxes(full, -1, -2)))
        return full.reshape(1, 2, 5, 2, 5).transpose(1, 3, 0, 2, 4).reshape(4, 1, 5, 5)

    ns, v = hermitian(a), hermitian(rng.normal(size=shape) + 1j * rng.normal(size=shape))
    torque = _hubbard_torque(ns, v)
    assert np.linalg.norm(torque) > 1.0e-2

    def pairing(omega):
        return float(np.real(np.sum(rotate_ns(ns, omega) * v)))

    errors = []
    for h in (1.0e-3, 1.0e-4):
        difference = np.array([(pairing(h * e) - pairing(-h * e)) / (2 * h)
                               for e in np.eye(3)])
        errors.append(np.abs(difference - torque).max())
    assert errors[1] < 1.0e-6 * np.linalg.norm(torque)
    assert errors[1] < errors[0]


def test_the_zeeman_torque_is_the_derivative_of_the_field_energy_under_a_turn():
    """:func:`_zeeman_torque` is ``d/dw (-B . R(w) M)`` for a uniform field (I113).

    For a uniform field that is ``B x M`` exactly, the field's potential being
    ``-B`` on the magnetization components; the turn is the one
    :func:`~defuqumat.forces.torque.rotate_texture` applies to the density.
    """
    import jax.numpy as jnp

    from defuqumat.scf.driver import _zeeman_torque
    from defuqumat.scf.fields import MagneticField
    from defuqumat.system.cell import Cell

    rng = np.random.default_rng(3)
    cell = Cell.from_vectors(np.eye(3) * 6.0)
    rho = _field(rng, shape=(4, 4, 4, 4))
    b = np.array([0.02, -0.01, 0.03])
    field = MagneticField(regions=None, uniform=jnp.asarray(b), atomic=None,
                          targets=None, penalty=0.0)
    torque = _zeeman_torque(field, jnp.asarray(rho), cell, 1.0)
    moment = cell.volume / rho[0].size * rho[1:].reshape(3, -1).sum(axis=1)
    np.testing.assert_allclose(torque, np.cross(b, moment), rtol=1e-12, atol=1e-15)

    h = 1.0e-5
    difference = np.array([
        (float(field.field_energy(jnp.asarray(rotate_texture(rho, rotation_matrix(h * e))), cell))
         - float(field.field_energy(jnp.asarray(rotate_texture(rho, rotation_matrix(-h * e))), cell)))
        / (2 * h) for e in np.eye(3)])
    np.testing.assert_allclose(difference, torque, rtol=1e-7, atol=1e-12)
