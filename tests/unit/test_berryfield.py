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


def test_an_automatic_grid_is_refused():
    text = _INPUT.format(extra="", kpoints="automatic\n2 2 2 0 0 0\n")
    with pytest.raises(NotImplementedError, match="automatic"):
        build_system(parse_pw_input(text))


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


def test_nspin_two_is_refused():
    with pytest.raises(NotImplementedError, match="nspin = 2"):
        _calculation(", nspin=2, tot_magnetization=0")


def test_a_spinor_is_refused():
    with pytest.raises(NotImplementedError, match="nspin = 4"):
        _calculation(", noncolin=.true.")
