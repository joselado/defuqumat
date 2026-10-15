"""P145: the Berry-phase field's operator, against the derivative of its own energy.

``h_epsi_her_set``/``h_epsi_her_apply`` build ``H_E = sum_n |a_n><u_n| + h.c.``
with ``a = fact (e- - e+)`` out of dual vectors; ``c_phase_field`` writes the
energy ``-E el_pol`` as a phase of overlap determinants. The two are the same
object only if ``H_E u_nk = (1/w_k) dE_P/d conj(u_nk)``, ``w_k = 2/nk`` at
``nspin = 1``, and that is asserted here with ``jax.grad`` of the energy, which
shares nothing with the duals but the overlaps. The states are random
orthonormal blocks in an abstract basis common to every k, so the gathers are
the identity and the test is about the algebra alone.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.scf.berryfield import (BerryFieldSetup, BerryTerm, phase_energy,
                                      string_layout)
from defuqumat.system import build_system

pytestmark = pytest.mark.unit

NPPSTR, NKORT, NOCC, NPW = 5, 2, 3, 11
NK = NPPSTR * NKORT


def _setup(efield=0.003, length=7.3):
    strings = np.arange(NK).reshape(NKORT, NPPSTR)
    plus = np.roll(strings, -1, axis=1).reshape(-1)[np.argsort(strings.reshape(-1))]
    minus = np.roll(strings, 1, axis=1).reshape(-1)[np.argsort(strings.reshape(-1))]
    gather = jnp.tile(jnp.arange(NPW), (NK, 1))
    found = jnp.ones((NK, NPW), dtype=bool)
    return BerryFieldSetup(
        gdir=2, nppstr=NPPSTR, efield=efield, nberrycyc=1, nocc=NOCC, strings=strings,
        plus_index=plus, minus_index=minus, plus_gather=gather, plus_found=found,
        minus_gather=gather, minus_found=found, length=length,
        charges=np.array([4.0, 6.0]))


def _states(seed=0, smooth=0.3):
    """Orthonormal occupied blocks that vary slowly along each string, as Bloch states do."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(NKORT, NPW, NOCC)) + 1j * rng.normal(size=(NKORT, NPW, NOCC))
    blocks = []
    for s in range(NKORT):
        for j in range(NPPSTR):
            kick = smooth * (rng.normal(size=(NPW, NOCC)) + 1j * rng.normal(size=(NPW, NOCC)))
            q, _ = np.linalg.qr(base[s] + kick)
            blocks.append(q.T)
    return jnp.asarray(np.array(blocks))[None]  # (1, nk, nocc, npw)


def _energy(setup, u):
    plus, _ = setup.neighbours(u)
    prefactor = -setup.efield * 2.0 * setup.fc_pol / setup.strings.shape[0]
    return phase_energy(u, plus, setup.strings, prefactor)


def test_the_operator_is_the_derivative_of_the_energy():
    setup, psi = _setup(), _states()
    u = psi[0]
    term = setup.term(psi)
    grad = jax.grad(lambda x: _energy(setup, x))(u)
    # jax.grad of a real f of complex u is 2 conj(df/du) = 2 df/d conj(u) conjugated.
    derivative = jnp.conj(grad) / 2.0
    w_k = 2.0 / NK
    np.testing.assert_allclose(np.asarray(term.a), np.asarray(derivative / w_k), atol=1e-12)


def test_h_e_u_is_a_and_a_is_orthogonal_to_u():
    setup, psi = _setup(), _states(1)
    term = setup.term(psi)
    for ik in range(NK):
        applied = term.apply(term.u[ik], ik)
        np.testing.assert_allclose(np.asarray(applied), np.asarray(term.a[ik]), atol=1e-13)
        overlap = jnp.conj(term.a[ik]) @ term.u[ik].T
        assert float(jnp.max(jnp.abs(overlap))) < 1e-13


def test_hermitian_and_matrix_agrees_with_apply():
    setup, psi = _setup(), _states(2)
    term = setup.term(psi)
    rng = np.random.default_rng(5)
    x = jnp.asarray(rng.normal(size=NPW) + 1j * rng.normal(size=NPW))
    y = jnp.asarray(rng.normal(size=NPW) + 1j * rng.normal(size=NPW))
    ik = 3
    left = jnp.vdot(x, term.apply(y, ik))
    right = jnp.vdot(term.apply(x, ik), y)
    assert abs(complex(left - right)) < 1e-13
    np.testing.assert_allclose(np.asarray(term.matrix(ik) @ y), np.asarray(term.apply(y, ik)),
                               atol=1e-13)


def test_the_energy_is_blind_to_a_rotation_inside_the_occupied_block():
    setup, psi = _setup(), _states(3)
    u = psi[0]
    rng = np.random.default_rng(9)
    unitary, _ = np.linalg.qr(rng.normal(size=(NOCC, NOCC)) + 1j * rng.normal(size=(NOCC, NOCC)))
    rotated = u.at[4].set(jnp.asarray(unitary.T) @ u[4])
    assert float(_energy(setup, rotated)) == pytest.approx(float(_energy(setup, u)), abs=1e-14)


def test_no_field_is_no_energy():
    setup = _setup(efield=5e-7)
    assert not setup.active


def test_string_layout_reads_consecutive_strings_and_refuses_others():
    k = np.array([[0.0, 0.0, j / 4] for j in range(4)] + [[0.5, 0.0, j / 4] for j in range(4)])
    np.testing.assert_array_equal(string_layout(k, 2, 4), [[0, 1, 2, 3], [4, 5, 6, 7]])
    with pytest.raises(ValueError, match="strings"):
        string_layout(k[[0, 2, 1, 3, 4, 5, 6, 7]], 2, 4)
    with pytest.raises(ValueError, match="nppstr"):
        string_layout(k[:7], 2, 4)


_INPUT = """
 &control
    calculation='scf', lelfield=.true., gdir=3, nppstr=2, nberrycyc=2
 /
 &system
    ibrav=1, celldm(1)=10.0, nat=1, ntyp=1, ecutwfc=20
 /
 &electrons
    efield=0.002 {extra}
 /
ATOMIC_SPECIES
Si 28.0 Si.pbe-rrkj.UPF
ATOMIC_POSITIONS bohr
Si 0.0 0.0 0.0
K_POINTS {kpoints}
"""

_LIST = "crystal\n2\n0.0 0.0 0.0 1\n0.0 0.0 0.5 1\n"


def test_input_is_read_and_symmetry_is_off():
    system = build_system(parse_pw_input(_INPUT.format(extra="", kpoints=_LIST)))
    assert system.berry_field == (2, 2, 0.002, 2)
    assert system.nosym


def test_three_directions_are_refused():
    text = _INPUT.format(extra=", efield_cart(1)=0.001", kpoints=_LIST)
    with pytest.raises(NotImplementedError, match="efield_cart"):
        build_system(parse_pw_input(text))


def test_an_automatic_grid_without_efield_cart_is_refused():
    """``pw.x`` reads ``efield_cart`` on an automatic grid and ignores ``efield``."""
    text = _INPUT.format(extra="", kpoints="automatic\n2 2 2 0 0 0\n")
    with pytest.raises(NotImplementedError, match="efield_cart = 0"):
        build_system(parse_pw_input(text))


def test_efield_cart_along_two_axes_is_refused():
    text = _INPUT.format(extra=", efield_cart(1)=0.001, efield_cart(3)=0.001",
                         kpoints="automatic\n2 2 2 0 0 0\n")
    with pytest.raises(NotImplementedError, match="more than one axis"):
        build_system(parse_pw_input(text))


def test_efield_cart_on_an_oblique_axis_is_refused():
    text = _INPUT.format(extra=", efield_cart(1)=0.001",
                         kpoints="automatic\n2 2 2 0 0 0\n").replace("ibrav=1", "ibrav=2")
    with pytest.raises(NotImplementedError, match="another lattice vector"):
        build_system(parse_pw_input(text))


_SI8 = Path(__file__).resolve().parents[1] / "data" / "qe" / "si8-lelfield.in"


def _si8_automatic(grid, component):
    text = _SI8.read_text()
    text = text[: text.index("K_POINTS")] + f"K_POINTS automatic\n{grid}\n"
    return text.replace("efield=0.001", f"efield_cart({component})=0.001")


def test_an_automatic_grid_is_electric_1s_strings():
    """I57: ``3 3 7 0 0 0`` with ``efield_cart(3)`` is the 63-point list of
    ``pw_electric/electric-1.in`` (9 strings of 7 along z), point by point and in
    order, and the field, its axis and ``nppstr`` are the ``gdir`` form's."""
    automatic = build_system(parse_pw_input(_si8_automatic("3 3 7 0 0 0", 3)))
    explicit = build_system(parse_pw_input(_SI8.read_text()))
    assert automatic.berry_field == explicit.berry_field == (2, 7, 0.001, 3)
    np.testing.assert_allclose(np.asarray(automatic.kpoints.crystal(automatic.cell)),
                               np.asarray(explicit.kpoints.crystal(explicit.cell)), atol=1e-8)
    np.testing.assert_allclose(np.asarray(automatic.kpoints.weights),
                               np.asarray(explicit.kpoints.weights), atol=1e-14)
    assert automatic.nosym


def test_an_automatic_grid_along_x_is_strings_along_x():
    """``nx_el(:, 1)``: the index along x fastest, then z, then y, with the grid's offsets."""
    system = build_system(parse_pw_input(_si8_automatic("5 2 3 1 0 1", 1)))
    assert system.berry_field[:3] == (0, 5, 0.001)
    k = np.asarray(system.kpoints.crystal(system.cell))
    strings = string_layout(k, 0, 5)
    assert strings.shape == (6, 5)
    np.testing.assert_allclose(k[:5], [[(i + 0.5) / 5, 0.0, 0.5 / 3] for i in range(5)])
    np.testing.assert_allclose(k[5, 1:], [0.0, 1.5 / 3])
    grid = {(round(5 * x - 0.5), round(2 * y), round(3 * z - 0.5)) for x, y, z in k}
    assert len(grid) == 30


def _calculation(extra_system="", kpoints=_LIST):
    """A one-atom ``Calculation`` with the field, built and not run: the regime gate fires here."""
    from defuqumat.pseudo import read_upf
    from defuqumat.scf import Calculation

    text = _INPUT.format(extra="", kpoints=kpoints).replace(
        "ecutwfc=20", "ecutwfc=12" + extra_system)
    system = build_system(parse_pw_input(text))
    pseudo = Path(__file__).resolve().parents[1] / "data" / "pseudo"
    pseudos = tuple(read_upf(pseudo / s.pseudo_file) for s in system.structure.species)
    return Calculation(system, pseudos)


def test_nspin_two_is_built_per_channel():
    """I14: a nonmagnetic ``nspin = 2`` run builds one count per channel and drops the factor 2."""
    field = _calculation(", nspin=2, tot_magnetization=0").berry_field
    assert field.nocc == (2, 2)
    assert field.spin_factor == 1.0


def test_unequal_channels_are_refused():
    """``pw.x`` builds the phase on ``nelec/2`` bands per channel, so a magnet has no reference."""
    with pytest.raises(NotImplementedError, match="3 and 1 occupied bands"):
        _calculation(", nspin=2, tot_magnetization=2, starting_magnetization(1)=0.5")


def test_a_nonmagnetic_spinor_is_built():
    """I15: one channel of ``NINT(nelec)`` two-component states, no factor 2."""
    field = _calculation(", noncolin=.true.").berry_field
    assert field.nocc == 4 and field.npol == 2 and field.spin_factor == 1.0


def test_a_magnetized_spinor_is_refused():
    with pytest.raises(NotImplementedError, match="nspin = 4 with a magnetization"):
        _calculation(", noncolin=.true., starting_magnetization(1)=0.5")


def test_a_spinor_with_spin_orbit_coupling_is_refused():
    from defuqumat.scf.berryfield import require_a_berry_field_regime

    calculation = type("C", (), {})()
    system = build_system(parse_pw_input(_INPUT.format(extra="", kpoints=_LIST).replace(
        "ecutwfc=20", "ecutwfc=12, noncolin=.true., lspinorb=.true.")))
    calculation.system, calculation.augmentation, calculation.spiral = system, None, False
    calculation.is_hubbard = False
    with pytest.raises(NotImplementedError, match="spin-orbit coupling"):
        require_a_berry_field_regime(calculation)


def test_each_channel_is_the_derivative_of_its_own_energy():
    """At ``nspin = 2`` the operator of each channel is ``(1/w_k) dE/d conj(u_s)``, ``w_k = 1/nk``.

    The two channels carry different states, so a term built from the wrong
    channel, or a factor 2 left on the energy, fails it.
    """
    setup = _setup()
    setup.nocc, setup.spin_factor = (NOCC, NOCC), 1.0
    psi = jnp.concatenate([_states(11), _states(12)])
    prefactor = -setup.efield * setup.fc_pol / setup.strings.shape[0]
    for spin in range(2):
        term = setup.term(psi, spin)

        def energy(x, spin=spin):
            whole = psi.at[spin].set(x)
            total = 0.0
            for s in range(2):
                plus, _ = setup.neighbours(whole[s])
                total = total + phase_energy(whole[s], plus, setup.strings, prefactor)
            return total

        derivative = jnp.conj(jax.grad(energy)(psi[spin])) / 2.0
        np.testing.assert_allclose(np.asarray(term.a), np.asarray(derivative * NK), atol=1e-12)
    # and the enthalpy is that sum, the ions aside
    positions = jnp.zeros((2, 3))
    total = sum(float(phase_energy(psi[s], setup.neighbours(psi[s])[0], setup.strings,
                                   prefactor)) for s in range(2))
    assert float(setup.enthalpy(psi, positions)) == pytest.approx(total, abs=1e-14)


def test_a_spinor_is_gathered_component_by_component():
    rng = np.random.default_rng(4)
    from defuqumat.scf.berryfield import _gathered

    coefficients = jnp.asarray(rng.normal(size=(3, 2, 2 * NPW)) + 1j * rng.normal(size=(3, 2, 2 * NPW)))
    gather = jnp.asarray(np.stack([rng.permutation(NPW) for _ in range(3)]))
    found = jnp.asarray(rng.random((3, NPW)) > 0.3)
    spinor = _gathered(coefficients, gather, found, npol=2)
    for c in range(2):
        part = _gathered(coefficients[..., c * NPW:(c + 1) * NPW], gather, found)
        np.testing.assert_array_equal(np.asarray(spinor[..., c * NPW:(c + 1) * NPW]),
                                      np.asarray(part))


def test_a_host_store_walked_in_chunks_is_the_whole_set():
    """I56: the operator, its chunk view and the string phases from a host store,
    three k-points at a time, are the whole-set ones. Ten k-points in chunks of
    three leave a padded last chunk, and two channels check the channel index."""
    setup = _setup()
    setup.nocc, setup.spin_factor, setup.k_batch = (NOCC, NOCC), 1.0, 3
    psi = jnp.concatenate([_states(21), _states(22)])
    host = np.array(psi)
    np.testing.assert_allclose(setup.string_phases(host), setup.string_phases(psi),
                               atol=1e-14)
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=NPW) + 1j * rng.normal(size=NPW))
    for spin in range(2):
        whole = setup.term(psi, spin)
        u_host, a_host = setup.host_term(host, spin)
        np.testing.assert_allclose(a_host, np.asarray(whole.a), atol=1e-13)
        np.testing.assert_array_equal(u_host, np.asarray(whole.u))
        rows = np.array([9, 9, 4])
        chunk = setup.chunk_term((u_host, a_host), rows)
        for ik in (9, 4):
            np.testing.assert_allclose(np.asarray(chunk.apply(x, ik)),
                                       np.asarray(whole.apply(x, ik)), atol=1e-13)


def test_efield_cart_against_a_reversed_axis_is_refused():
    """``efield_cry`` is ``E . a_g/|a_g|``: a lattice vector along ``-x`` would flip the field."""
    text = _INPUT.format(extra=", efield_cart(1)=0.001", kpoints="automatic\n2 2 2 0 0 0\n")
    text = text.replace("ibrav=1, celldm(1)=10.0", "ibrav=0").replace(
        "ATOMIC_SPECIES", "CELL_PARAMETERS bohr\n-10.0 0.0 0.0\n0.0 10.0 0.0\n0.0 0.0 10.0\nATOMIC_SPECIES")
    with pytest.raises(NotImplementedError, match="opposite sign"):
        build_system(parse_pw_input(text))
