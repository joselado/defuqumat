"""I80: the phonon at ``q != 0`` with a nonlinear core correction, at gate speed.

The guard of :func:`~defuqumat.response.phononq.require_a_two_sphere_regime` is
a function of a handful of flags, so it is checked here on a stand-in
calculation rather than on a ground state: a core charge passes since I80, and
each sibling refusal still fires with the core charge present. The shifted
core charge the route differentiates is checked against the unshifted one at
``q = 0`` and against its own analytic tangent. The number the route is
validated by is ``tests/regression/test_phonons_at_q_nlcc.py``.
"""

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.response.phononq import require_a_two_sphere_regime

PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _calculation(**changes):
    """A stand-in carrying exactly the fields the guard reads, every one in the
    regime the route carries and with a core charge; ``changes`` moves one of
    them out of it."""
    system = SimpleNamespace(nspin=1, noncolin=False, spiral_q=None)
    fields = dict(
        is_ultrasoft=False, rho_core_g=object(), use_symmetry=False,
        functional=SimpleNamespace(is_meta=False), source_free=False,
        is_hubbard=False, system=system,
    )
    for name, value in changes.items():
        if name in ("nspin", "noncolin", "spiral_q"):
            setattr(system, name, value)
        else:
            fields[name] = value
    return SimpleNamespace(**fields)


def test_a_core_correction_is_admitted():
    require_a_two_sphere_regime(_calculation(), None)


@pytest.mark.parametrize("change, message", [
    (dict(is_ultrasoft=True), "ultrasoft or PAW"),
    (dict(nspin=2), "one spin channel"),
    (dict(noncolin=True), "one spin channel"),
    (dict(use_symmetry=True), "small group of q"),
    (dict(spiral_q=(0.0, 0.0, 0.1)), "spin spiral"),
    (dict(functional=SimpleNamespace(is_meta=True)), "meta-GGA"),
    (dict(source_free=True), "nosource"),
    (dict(is_hubbard=True), "Hubbard"),
])
def test_the_sibling_refusals_still_fire_with_a_core_charge(change, message):
    with pytest.raises(NotImplementedError, match=message):
        require_a_two_sphere_regime(_calculation(**change), None)


_INPUT = """&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0,
  nosym = .true., noinv = .true.
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-oncv-nlcc.UPF
ATOMIC_POSITIONS alat
 Si  0.00  0.00  0.00
 Si  0.25  0.27  0.21
K_POINTS automatic
 2 2 2 0 0 0
"""


def _pieces():
    """``(pseudos, structure, cell, dense G set)`` for silicon with a core
    charge, the second atom off its site so no phase is special."""
    from defuqumat.basis.gvectors import generate_gvectors
    from defuqumat.io.pwin import parse_pw_input
    from defuqumat.pseudo import read_upf
    from defuqumat.system import build_system

    system = build_system(parse_pw_input(_INPUT))
    pseudos = (read_upf(PSEUDO / "Si.pz-oncv-nlcc.UPF"),)
    return pseudos, system.structure, system.cell, generate_gvectors(system.cell, 48.0)


def test_the_shifted_core_charge_is_the_core_charge_at_q_zero():
    from defuqumat.pseudo.potentials import core_charge, core_charge_at_q

    pseudos, structure, cell, dense = _pieces()
    plain = np.asarray(core_charge(pseudos, structure, cell, dense))
    shifted = np.asarray(core_charge_at_q(pseudos, structure, cell, dense,
                                          np.zeros(3)))
    assert np.abs(plain).max() > 1e-3
    assert np.abs(shifted - plain).max() < 1e-14 * np.abs(plain).max() + 1e-16


def test_the_tangent_brings_down_minus_i_g_plus_q():
    """``d/dtau_a`` of the shifted charge is ``-i (G+q)`` times atom ``a``'s own
    term, which is ``addcore.f90``'s ``drc * gu * fact * gtau``."""
    import equinox as eqx

    from defuqumat.pseudo.formfactors import core_charge_of_g
    from defuqumat.basis.gvectors import modulus
    from defuqumat.pseudo.potentials import core_charge_at_q

    pseudos, structure, cell, dense = _pieces()
    q_cart = np.asarray(cell.k_to_cartesian(np.array([0.5, 0.0, 0.0]))) * cell.tpiba
    positions = jnp.asarray(structure.positions)

    def charge(pos):
        moved = eqx.tree_at(lambda s: s.positions, structure, pos)
        return core_charge_at_q(pseudos, moved, cell, dense, q_cart)

    g = np.asarray(dense.cartesian(cell)) + q_cart[None, :]
    radial = np.asarray(core_charge_of_g(pseudos[0], modulus(jnp.asarray(g)),
                                         cell.volume))
    for atom, cart in ((0, 0), (1, 2)):
        tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
        derivative = np.asarray(jax.jvp(charge, (positions,), (tangent,))[1])
        phase = np.exp(-1j * g @ np.asarray(positions[atom]))
        expected = -1j * g[:, cart] * radial * phase
        assert np.abs(derivative - expected).max() < 1e-12 * np.abs(expected).max()
