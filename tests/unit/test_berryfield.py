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


def test_a_magnetized_spinor_is_built():
    """I85: a magnetized spinor runs the nonmagnetic spinor's code (checked against
    ``pw.x`` in ``tests/regression/test_berryfield.py``), one gather for both components."""
    field = _calculation(", noncolin=.true., starting_magnetization(1)=0.5").berry_field
    assert field.nocc == 4 and field.npol == 2 and field.plus_gather.ndim == 2


_SPIRAL = (", noncolin=.true., starting_magnetization(1)=0.5, angle1(1)=90.0, "
           "spiral_q(1)=0.0, spiral_q(2)=0.0, spiral_q(3)=0.25")


def test_a_spin_spiral_gathers_each_component_from_its_own_rows():
    """I85: a spiral's down component is on plane-wave rows ``nk + ik``, ``k - q/2``.

    The up component's gather is the one a plain spinor at ``k + q/2`` would
    have, and the down component's is built from the other half of the rows,
    so the two differ at a string's wrap (the shift ``e_gdir`` lands on
    different Miller sets of the two spheres) or within it.
    """
    from defuqumat.topology.states import build_plane_wave_states

    calculation = _calculation(_SPIRAL)
    field = calculation.berry_field
    nk = field.plus_index.shape[0]
    npwx = calculation.basis.planewaves.mask.shape[1]
    assert field.plus_gather.shape == (nk, 2, npwx)
    assert field.minus_found.shape == (nk, 2, npwx)
    states = build_plane_wave_states(
        calculation, jnp.zeros((nk, 1, 2 * npwx),
                               dtype=calculation.system.cell.precision.complex))
    unit = np.array([0, 0, 1])
    for ik in range(nk):
        jp = field.plus_index[ik]
        shift = unit if jp < ik else None
        for c, (i, j) in enumerate([(ik, jp), (nk + ik, nk + jp)]):
            gather, found = states._alignment(i, j, shift)
            np.testing.assert_array_equal(np.asarray(field.plus_found[ik, c]), np.asarray(found))
            np.testing.assert_array_equal(
                np.where(found, np.asarray(field.plus_gather[ik, c]), -1),
                np.where(found, np.asarray(gather), -1))
    assert not np.array_equal(np.asarray(field.plus_found[:, 0]),
                              np.asarray(field.plus_found[:, 1]))


def test_a_spin_spiral_still_refuses_a_metal_and_an_ultrasoft_dataset():
    """The refusals beside the spiral still fire on a spiral."""
    from defuqumat.scf.berryfield import require_a_berry_field_regime

    with pytest.raises(NotImplementedError, match="occupations = 'smearing'"):
        _calculation(_SPIRAL + ", occupations='smearing', degauss=0.01")
    calculation = type("C", (), {})()
    calculation.system = build_system(parse_pw_input(_INPUT.format(extra="", kpoints=_LIST)
                                                     .replace("ecutwfc=20", "ecutwfc=12" + _SPIRAL)))
    calculation.augmentation, calculation.spiral, calculation.is_hubbard = object(), True, False
    with pytest.raises(NotImplementedError, match="ultrasoft or PAW"):
        require_a_berry_field_regime(calculation)
    calculation.augmentation, calculation.is_hubbard = None, True
    with pytest.raises(NotImplementedError, match="DFT\\+U"):
        require_a_berry_field_regime(calculation)


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


def test_a_spiral_gathers_each_component_on_its_own_sphere():
    """I85: a ``(nk, 2, npwx)`` gather takes each component by its own index,
    and the host path (a streamed store) does the same."""
    rng = np.random.default_rng(5)
    from defuqumat.scf.berryfield import _gathered

    coefficients = jnp.asarray(rng.normal(size=(3, 2, 2 * NPW)) + 1j * rng.normal(size=(3, 2, 2 * NPW)))
    gathers = [jnp.asarray(np.stack([rng.permutation(NPW) for _ in range(3)])) for _ in range(2)]
    founds = [jnp.asarray(rng.random((3, NPW)) > 0.3) for _ in range(2)]
    spinor = _gathered(coefficients, jnp.stack(gathers, axis=1), jnp.stack(founds, axis=1),
                       npol=2)
    for c in range(2):
        part = _gathered(coefficients[..., c * NPW:(c + 1) * NPW], gathers[c], founds[c])
        np.testing.assert_array_equal(np.asarray(spinor[..., c * NPW:(c + 1) * NPW]),
                                      np.asarray(part))
    assert not np.allclose(np.asarray(spinor[..., NPW:]),
                           np.asarray(_gathered(coefficients, gathers[0], founds[0],
                                                npol=2)[..., NPW:]))
    # the host store, walked in chunks, against the whole set
    setup = _setup()
    setup.npol, setup.nocc, setup.spin_factor, setup.k_batch = 2, NOCC, 1.0, 3
    setup.plus_gather = jnp.stack([setup.plus_gather, jnp.roll(setup.plus_gather, 1, axis=1)], 1)
    setup.plus_found = jnp.stack([setup.plus_found, setup.plus_found], 1)
    setup.minus_gather = jnp.stack([setup.minus_gather, jnp.roll(setup.minus_gather, -1, axis=1)], 1)
    setup.minus_found = jnp.stack([setup.minus_found, setup.minus_found], 1)
    psi = jnp.concatenate([_states(31), _states(32)], axis=-1) / np.sqrt(2.0)
    whole = setup.term(psi, 0)
    u_host, a_host = setup.host_term(np.array(psi), 0)
    np.testing.assert_allclose(a_host, np.asarray(whole.a), atol=1e-13)
    np.testing.assert_allclose(setup.string_phases(np.array(psi)), setup.string_phases(psi),
                               atol=1e-14)


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


_US = "Si.pbe-n-rrkjus_psl.0.1.UPF"
_PAW = "Si.pbe-n-kjpaw_psl.0.1.UPF"


def _augmented_calculation(dataset, extra_system="", kpoints=_LIST, nppstr=2):
    """``_calculation`` on an augmented dataset: the cell, the strings and the field are its."""
    from defuqumat.pseudo import read_upf
    from defuqumat.scf import Calculation

    text = _INPUT.format(extra="", kpoints=kpoints).replace(
        "ecutwfc=20", "ecutwfc=12" + extra_system).replace("Si.pbe-rrkj.UPF", dataset)
    text = text.replace("nppstr=2", f"nppstr={nppstr}")
    system = build_system(parse_pw_input(text))
    pseudo = Path(__file__).resolve().parents[1] / "data" / "pseudo"
    pseudos = tuple(read_upf(pseudo / s.pseudo_file) for s in system.structure.species)
    return Calculation(system, pseudos)


_STRING3 = "crystal\n3\n0.0 0.0 0.0 1\n0.0 0.0 0.3333333333333333 1\n0.0 0.0 0.6666666666666666 1\n"


def _augmented_states(calculation, seed=0):
    """Random occupied blocks on each k's sphere, made ``S``-orthonormal there."""
    field = calculation.berry_field
    mask = np.asarray(calculation.basis.planewaves.mask)
    nk, npwx = mask.shape
    rng = np.random.default_rng(seed)
    u = (rng.normal(size=(nk, field.occupied(), npwx))
         + 1j * rng.normal(size=(nk, field.occupied(), npwx))) * mask[:, None, :]
    su, _, _ = field.augmented_neighbours(jnp.asarray(u), calculation)
    for ik in range(nk):
        gram = np.conj(u[ik]) @ np.asarray(su[ik]).T  # <u_m|S|u_n>
        u[ik] = np.linalg.inv(np.linalg.cholesky(gram.T)) @ u[ik]
    return jnp.asarray(u)[None]


@pytest.mark.parametrize("dataset", [_US, _PAW])
def test_an_augmented_field_is_the_derivative_of_its_energy(dataset):
    """I103: on an ultrasoft or PAW dataset ``a = (1/w_k) dE_P/d conj(u_k)``,
    ``H_E u = a`` and ``<a|u> = 0``, with ``M+`` carrying ``q_ij(dk)``.

    ``u_k`` enters ``M+(k)`` as a bra and ``M+(k - dk)`` as a ket, so the
    gradient holds the backward link through ``q_ij(dk)^dagger``, and the duals
    hold it through ``q_ij(-dk)`` computed on its own: the two agree only if the
    backward cross term is the adjoint of the forward one.
    """
    calculation = _augmented_calculation(dataset, kpoints=_STRING3, nppstr=3)
    field = calculation.berry_field
    assert field.augmented
    # dk is b_3/3, unwrapped, read where the string wraps
    bg = np.asarray(calculation.system.cell.bg)
    tpiba = np.linalg.norm(field.dk) * 3 / np.linalg.norm(bg[2])
    np.testing.assert_allclose(field.dk * 3, tpiba * bg[2], atol=1e-12)
    psi = _augmented_states(calculation)
    u = psi[0]
    nk = u.shape[0]
    term = field.term(psi, 0, calculation)
    prefactor = -field.efield * 2.0 * field.fc_pol / field.strings.shape[0]

    def energy(x):
        _, plus, _ = field.augmented_neighbours(x, calculation)
        return phase_energy(x, plus, field.strings, prefactor)

    derivative = jnp.conj(jax.grad(energy)(u)) / 2.0
    scale = float(jnp.max(jnp.abs(term.a)))
    np.testing.assert_allclose(np.asarray(term.a), np.asarray(derivative * nk / 2.0),
                               atol=1e-10 * scale)
    for ik in range(nk):
        np.testing.assert_allclose(np.asarray(term.apply(u[ik], ik)), np.asarray(term.a[ik]),
                                   atol=1e-10 * scale)
        assert float(jnp.max(jnp.abs(jnp.conj(term.a[ik]) @ u[ik].T))) < 1e-10 * scale
    # the enthalpy is the same energy, the ions aside
    positions = jnp.asarray(calculation.system.structure.positions)
    ionic = -field.efield * field.ionic_dipole(positions)
    assert float(field.enthalpy(psi, positions, calculation)) == pytest.approx(
        float(energy(u)) + ionic, abs=1e-12)


def test_the_cross_term_is_what_makes_the_operator_the_derivative():
    """The same identity with ``q_ij(+-dk)`` dropped from the duals only fails:
    the check above can tell the term from its absence."""
    calculation = _augmented_calculation(_US, kpoints=_STRING3, nppstr=3)
    field = calculation.berry_field
    psi = _augmented_states(calculation, seed=1)
    prefactor = -field.efield * 2.0 * field.fc_pol / field.strings.shape[0]

    def energy(x):
        _, plus, _ = field.augmented_neighbours(x, calculation)
        return phase_energy(x, plus, field.strings, prefactor)

    derivative = jnp.conj(jax.grad(energy)(psi[0])) / 2.0 * psi.shape[1] / 2.0
    field.cross_off = True
    try:
        term = field.term(psi, 0, calculation)
    finally:
        field.cross_off = False
    gap = float(jnp.max(jnp.abs(term.a - derivative)) / jnp.max(jnp.abs(derivative)))
    assert gap > 1e-3


def test_an_augmented_dataset_refuses_nspin_two_and_the_streamed_store():
    """The regimes the fix does not cover still refuse by name (I103)."""
    from defuqumat.forces.chunked import walks_chunks

    with pytest.raises(NotImplementedError, match="ultrasoft or PAW dataset at nspin = 2"):
        _augmented_calculation(_US, ", nspin=2, tot_magnetization=0")
    calculation = _augmented_calculation(_US, kpoints=_STRING3, nppstr=3)
    host = np.asarray(_augmented_states(calculation))
    with pytest.raises(NotImplementedError, match="streamed store"):
        calculation.berry_field.host_term(host, 0)
    with pytest.raises(NotImplementedError, match="streamed store"):
        calculation.berry_field.string_phases(host, calculation)
    with pytest.raises(NotImplementedError, match="forces_us_efield"):
        walks_chunks(calculation, host)
    from defuqumat.forces.analytic import analytic_forces
    with pytest.raises(NotImplementedError, match="forces_us_efield"):
        analytic_forces(calculation, None)
