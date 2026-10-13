"""P144: the saw-tooth field and the dipole correction, against ``add_efield.f90``.

:mod:`defuqumat.scf.sawtooth` writes down one energy and takes the potential and
the force from ``jax.grad``. QE writes both out by hand (``add_efield.f90``,
``compute_dip.f90``, ``extfield.f90``), and the two must agree exactly, because
they are the same derivative. The Fortran is transcribed literally below, loop by
loop, and nothing in it is shared with the implementation under test.

The densities are synthetic: a random positive array on a small oblique grid.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.scf.sawtooth import SawtoothField, saw
from defuqumat.system import build_system
from defuqumat.system.cell import Cell

pytestmark = pytest.mark.unit

GRID = (5, 6, 9)
E2 = 2.0
FPI = 4.0 * np.pi


def _qe_saw(emaxpos, eopreg, x):
    """``extfield.f90``'s ``saw``, literally."""
    z = x - emaxpos
    y = z - np.floor(z)
    if y <= eopreg:
        return (0.5 - y / eopreg) * (1.0 - eopreg)
    return (-0.5 + (y - eopreg) / (1.0 - eopreg)) * (1.0 - eopreg)


def _qe_add_efield(rho, tau_alat, zv, cell, edir, eamp, emaxpos, eopreg, dipfield):
    """``add_efield`` with ``compute_el_dip``/``compute_ion_dip``: ``(v, etotefield, tot_dipole)``.

    ``tau_alat`` is in units of ``alat`` and ``bg`` in ``2 pi/alat``, as in QE.
    With ``dipfield = .false.`` the returned energy is QE's printed one, the
    ions' part alone.
    """
    at = np.asarray(cell.at) / cell.alat
    bg = np.linalg.inv(at).T  # rows b_i, a_i . b_j = delta_ij
    alat = cell.alat
    omega = float(cell.volume)
    bmod = np.linalg.norm(bg[edir])
    n = rho.shape
    el_dipole = 0.0
    if dipfield:
        for i in range(n[0]):
            for j in range(n[1]):
                for k in range(n[2]):
                    sawarg = (i, j, k)[edir] / n[edir]
                    el_dipole += (rho[i, j, k] * _qe_saw(emaxpos, eopreg, sawarg)
                                  * (alat / bmod) * (FPI / rho.size))
    ion_dipole = 0.0
    for na in range(len(zv)):
        tvectb = tau_alat[na] @ bg[edir]
        ion_dipole += zv[na] * _qe_saw(emaxpos, eopreg, tvectb) * (alat / bmod) * (FPI / omega)
    tot_dipole = -el_dipole + ion_dipole if dipfield else 0.0
    if dipfield:
        etotefield = -E2 * (eamp - tot_dipole / 2.0) * tot_dipole * omega / FPI
    else:
        etotefield = -E2 * eamp * ion_dipole * omega / FPI
    v = np.zeros(n)
    for i in range(n[0]):
        for j in range(n[1]):
            for k in range(n[2]):
                sawarg = (i, j, k)[edir] / n[edir]
                v[i, j, k] = (E2 * (eamp - tot_dipole) * _qe_saw(emaxpos, eopreg, sawarg)
                              * (alat / bmod))
    forcefield = np.array([
        E2 * (eamp - tot_dipole) * zv[na] * bg[edir] / bmod for na in range(len(zv))
    ])
    return v, etotefield, tot_dipole, forcefield


@pytest.fixture(scope="module")
def cell() -> Cell:
    # Oblique, so that ``alat/bmod`` is not the length of a lattice vector and a
    # wrong plane spacing cannot pass by accident.
    return Cell.from_vectors(np.array([[7.0, 0.0, 0.0], [1.3, 6.2, 0.0],
                                       [0.7, -0.9, 9.4]]), alat=7.0)


@pytest.fixture(scope="module")
def rho():
    return np.random.default_rng(3).uniform(0.01, 0.2, GRID)


#: Two atoms where the saw-tooth rises along every ``edir`` for ``emaxpos =
#: 0.62, eopreg = 0.1``: crystal coordinates kept in (0.05, 0.55).
CRYSTAL = np.array([[0.11, 0.23, 0.31], [0.47, 0.39, 0.52]])
ZV = np.array([4.0, 6.0])


def _field(edir, dipfield, eamp=0.003, emaxpos=0.62, eopreg=0.1):
    return SawtoothField(eamp=jnp.asarray(eamp), edir=edir, emaxpos=emaxpos,
                         eopreg=eopreg, dipfield=dipfield)


@pytest.mark.parametrize("x", np.linspace(-1.3, 2.2, 37))
def test_saw_is_qe_saw(x):
    assert float(saw(0.62, 0.1, x)) == pytest.approx(_qe_saw(0.62, 0.1, x), abs=1e-14)


@pytest.mark.parametrize("dipfield", [False, True])
@pytest.mark.parametrize("edir", [0, 1, 2])
def test_the_potential_and_energy_are_add_efield(cell, rho, edir, dipfield):
    """``v`` is the functional derivative and is QE's explicit saw-tooth.

    Without ``dipfield`` QE prints only the ions' part as ``etotefield``; the
    electrons' part is ``int rho v``, which it keeps in ``eband``. Their sum is
    this code's ``E_f``.
    """
    field = _field(edir, dipfield)
    positions = cell.to_cartesian(CRYSTAL)
    v, energy, dip = field.potential(jnp.asarray(rho)[None], positions, ZV, cell)
    v_qe, e_qe, dip_qe, _ = _qe_add_efield(
        rho, np.asarray(positions) / cell.alat, ZV, cell, edir, 0.003, 0.62, 0.1,
        dipfield,
    )
    np.testing.assert_allclose(np.asarray(v[0]), v_qe, atol=1e-13)
    if dipfield:
        assert float(energy) == pytest.approx(e_qe, abs=1e-12)
        assert float(dip) == pytest.approx(dip_qe, abs=1e-13)
    else:
        electrons = float(cell.volume) / rho.size * np.sum(rho * v_qe)
        assert float(energy) == pytest.approx(e_qe + electrons, abs=1e-12)


@pytest.mark.parametrize("dipfield", [False, True])
def test_the_force_is_forcefield(cell, rho, dipfield):
    """``-dE/dtau`` at fixed density is ``forcefield`` where the saw-tooth rises."""
    field = _field(2, dipfield)
    positions = cell.to_cartesian(CRYSTAL)
    force = -jax.grad(lambda tau: field.energy(jnp.asarray(rho), tau, ZV, cell))(positions)
    _, _, _, forcefield = _qe_add_efield(
        rho, np.asarray(positions) / cell.alat, ZV, cell, 2, 0.003, 0.62, 0.1, dipfield,
    )
    np.testing.assert_allclose(np.asarray(force), forcefield, atol=1e-13)


def test_the_dipole_is_the_derivative_in_the_amplitude(cell, rho):
    """``dE_f/d(eamp) = -(Omega/4 pi) e2 D``: the field couples to the dipole."""
    field = _field(1, True)
    positions = cell.to_cartesian(CRYSTAL)
    derivative = jax.grad(lambda f: f.energy(jnp.asarray(rho), positions, ZV, cell))(field)
    dip, _, _ = field.dipoles(jnp.asarray(rho), positions, ZV, cell)
    assert float(derivative.eamp) == pytest.approx(
        -float(cell.volume) / FPI * E2 * float(dip), rel=1e-13)


@pytest.mark.parametrize("components", [2, 4])
def test_spin_layouts(cell, rho, components):
    """``(up, down)``: both channels feel all of it; ``(n, m)``: the first only."""
    field = _field(2, True)
    positions = cell.to_cartesian(CRYSTAL)
    single, e1, _ = field.potential(jnp.asarray(rho)[None], positions, ZV, cell)
    if components == 2:
        stacked = jnp.stack([0.3 * rho, 0.7 * rho])
    else:
        stacked = jnp.stack([rho, 0.1 * rho, -0.2 * rho, 0.05 * rho])
    v, e, _ = field.potential(stacked, positions, ZV, cell)
    assert float(e) == pytest.approx(float(e1), abs=1e-13)
    if components == 2:
        np.testing.assert_allclose(np.asarray(v), np.asarray(jnp.concatenate([single, single])),
                                   atol=1e-14)
    else:
        np.testing.assert_allclose(np.asarray(v[0]), np.asarray(single[0]), atol=1e-14)
        assert not np.any(np.asarray(v[1:]))


def test_relax_el_printed_correction():
    """``pw_relax/relax-el``: ``electric field correction = 0.01440000`` Ry.

    O at ``z = -1.1``, C at ``z = 1.1`` bohr in a 10 bohr cube, ``eamp =
    0.001``, ``emaxpos = 0.5``, ``eopreg = 0.1``, no ``dipfield``: the ions'
    part alone, which is what QE prints at the first step.
    """
    cell = Cell.from_ibrav(1, [10.0, 0, 0, 0, 0, 0])
    field = SawtoothField(eamp=jnp.asarray(0.001), edir=2, emaxpos=0.5, eopreg=0.1,
                          dipfield=False)
    positions = jnp.array([[0.0, 0.0, -1.1], [0.0, 0.0, 1.1]])
    zero = jnp.zeros((4, 4, 4))
    energy = field.energy(zero, positions, np.array([6.0, 4.0]), cell)
    assert float(energy) == pytest.approx(0.01440000, abs=5e-9)


_INPUT = """
 &control
    calculation='scf', tefield={tefield}, dipfield={dipfield}
 /
 &system
    ibrav=1, celldm(1)=10.0, nat=1, ntyp=1, ecutwfc=20,
    {extra}
 /
 &electrons
 /
ATOMIC_SPECIES
O 16.0 O.pz-rrkjus.UPF
ATOMIC_POSITIONS bohr
O 0.0 0.0 0.0
K_POINTS automatic
2 2 2 0 0 0
"""


def _parse(tefield, dipfield, extra="edir=3, eamp=0.002, emaxpos=0.4, eopreg=0.2"):
    text = _INPUT.format(tefield=tefield, dipfield=dipfield, extra=extra)
    return parse_pw_input(text)


def test_input_is_read_and_symmetry_is_off():
    with pytest.warns(UserWarning, match="no symmetry"):
        system = build_system(_parse(".true.", ".true."))
    assert system.efield == (2, 0.002, 0.4, 0.2, True)
    assert system.nosym


def test_no_field_is_none():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        system = build_system(_parse(".false.", ".false."))
    assert system.efield is None
    assert not system.nosym


def test_dipfield_without_tefield_is_refused():
    with pytest.raises(ValueError, match="dipfield"):
        build_system(_parse(".false.", ".true."))


def test_missing_edir_is_refused():
    with pytest.raises(ValueError, match="edir"):
        build_system(_parse(".true.", ".false.", extra="eamp=0.001"))


def test_gate_is_refused():
    with pytest.raises(NotImplementedError, match="gate"):
        build_system(_parse(".true.", ".false.", extra="edir=3, gate=.true."))
