"""Real-time propagation with a Hubbard U at a frozen occupation matrix (``notes/incompat/I100.md``).

The Hubbard term ``sum |wfcU> v <wfcU|`` moves with ``k + kappa`` as the beta
functions do, so the chunk rebuilds the atomic orbitals from a Chebyshev table
of ``chi_l(q)/q^l`` at every ``kappa``. Checked here without a long propagation:

* the refusals that stay, each fired;
* the chunk's ``wfcU`` at ``kappa = 0`` against the calculation's, built by
  the radial transform, and at a shifted ``kappa`` against the calculation
  moved there by ``at_kcart``;
* the chunk's Hamiltonian at ``kappa = 0`` against the calculation's own with
  the Hubbard term, on the states;
* the chunk's current at a shifted ``kappa`` against
  :class:`~defuqumat.response.velocity.VelocityOperator` with ``ns``, which
  differentiates ``at_kcart`` and shares only the definition with the chunk.

The identity on a propagation, the first order of a kick against the Kubo sum
plus the band curvature, is ``tests/regression/test_realtime_regimes.py``'s
DFT+U case (slow).
"""

import re
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.realtime.propagate import _prepare, propagate
from defuqumat.realtime.orders import propagate_orders
from defuqumat.realtime.pulse import Kick
from defuqumat.scf import Calculation
from defuqumat.system import build_system
from defuqumat.system.kpoints import KPoints, for_spin

pytestmark = pytest.mark.unit

#: Gamma and one point of no symmetry.
KSET = (np.array([[0.0, 0.0, 0.0], [0.25, 0.1, -0.05]]), np.array([0.5, 0.5]))

CELL = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0, nosym = .true.
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 1 1 1 0 0 0
HUBBARD {{{kind}}}
 U Si-3p 6.0
"""


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


def _calculation(pseudo_dir, tmp_path, kind="atomic", extra=""):
    path = tmp_path / f"si-ldau-{kind}-{abs(hash(extra)) % 1000}.in"
    path.write_text(CELL.format(kind=kind, extra=extra))
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    system = eqx.tree_at(lambda s: s.kpoints, system,
                         for_spin(KPoints(coords=KSET[0], weights=KSET[1]), system.nspin))
    return Calculation(system, pseudos)


def _states(calc, nbnd=4):
    """Random orthonormal states on each sphere, ``(nk, nbnd, npwx)``; nothing here needs eigenstates."""
    mask = np.asarray(calc.basis.planewaves.mask)
    rng = np.random.default_rng(3)
    psi = np.zeros((mask.shape[0], nbnd, mask.shape[1]), dtype=complex)
    for ik in range(mask.shape[0]):
        keep = np.flatnonzero(mask[ik])
        block = rng.normal(size=(len(keep), nbnd)) + 1j * rng.normal(size=(len(keep), nbnd))
        q, _ = np.linalg.qr(block)
        psi[ik][:, keep] = q.T
    return psi


def _frozen(calc):
    v_scf = calc.potential(calc.starting_density()).v_scf
    ns = calc.starting_ns()
    return v_scf, ns


def test_a_hubbard_run_without_its_occupation_matrix_is_refused(pseudo_dir, tmp_path):
    """The driver handed no ``ns`` refuses: the term cannot be rebuilt from the density."""
    calc = _calculation(pseudo_dir, tmp_path)
    v_scf, _ = _frozen(calc)
    psi = _states(calc)
    weights = np.full(psi.shape[:2], 0.5)
    with pytest.raises(NotImplementedError, match="occupation matrix"):
        propagate(calc, psi, weights, v_scf, Kick(strength=0.01, direction=(1, 0, 0)),
                  dt=0.05, duration=0.1, k_batch=None)
    with pytest.raises(NotImplementedError, match="occupation matrix"):
        propagate_orders(calc, psi, weights, v_scf, Kick(strength=1.0, direction=(1, 0, 0)),
                         dt=0.05, order=1, start=0.0, duration=0.1, k_batch=None)


def _setup(calc, psi, weights, v_scf, ns, kappa_max=0.3):
    return _prepare(calc, psi, weights, v_scf, kappa_max, None, "taylor4", None, None,
                    bounds=False, ns=ns)


@pytest.mark.parametrize("kind", ["atomic", "norm-atomic"])
def test_the_moving_projectors_are_the_calculations_at_k_and_at_k_plus_kappa(
        pseudo_dir, tmp_path, kind):
    """``wfcU`` from the table against the transform's, at ``kappa = 0`` and moved by ``at_kcart``.

    The table is a Chebyshev series in ``q^2`` of the same radial transform, so
    the two agree to its round-off; a wrong column order, phase or atom
    gather would be an error of the order of the orbitals themselves.
    """
    calc = _calculation(pseudo_dir, tmp_path, kind)
    v_scf, ns = _frozen(calc)
    psi = _states(calc)
    weights = np.full(psi.shape[:2], 0.5)
    chunk = _setup(calc, psi, weights, v_scf, ns).first
    kcart = np.asarray(calc.system.kpoints.cartesian(calc.system.cell))
    for kappa in (np.zeros(3), np.array([0.13, -0.07, 0.21])):
        mine = np.asarray(chunk.hubbard_at(jnp.asarray(kappa)).wfcU)
        theirs = np.asarray(calc.at_kcart(jnp.asarray(kcart + kappa[None])).wfcU)
        assert mine.shape == theirs.shape
        assert np.abs(mine - theirs).max() < 1e-11 * np.abs(theirs).max()


def test_the_chunks_hamiltonian_is_the_calculations_with_its_hubbard_term(pseudo_dir, tmp_path):
    """``H(k)|psi>`` from the chunk against ``Calculation.hamiltonian`` with ``hubbard_terms(ns)``.

    And the term is there: the same comparison without it differs by the
    Hubbard term's size, ``U (1/2 - 1/3)`` on Hund's rule ``n = 1/3``.
    """
    calc = _calculation(pseudo_dir, tmp_path)
    v_scf, ns = _frozen(calc)
    psi = _states(calc)
    weights = np.full(psi.shape[:2], 0.5)
    chunk = _setup(calc, psi, weights, v_scf, ns).first
    zero = jnp.zeros(3)
    mine = np.asarray(chunk.applied(zero, jnp.asarray(psi)))
    _, _, terms = calc.hubbard_terms(jnp.asarray(ns))
    ham = calc.hamiltonian(v_scf, None, terms)[0]
    bare = calc.hamiltonian(v_scf)[0]
    theirs = np.stack([np.asarray(ham.apply(jnp.asarray(psi[ik]), ik)) for ik in range(2)])
    without = np.stack([np.asarray(bare.apply(jnp.asarray(psi[ik]), ik)) for ik in range(2)])
    scale = np.abs(theirs).max()
    assert np.abs(mine - theirs).max() < 1e-10 * scale
    assert np.abs(without - theirs).max() > 1e-3 * scale
    # the chunk's kinetic, nonlocal and Hubbard form is the same operator less the local part
    form = np.asarray(chunk.band_form(zero, jnp.asarray(psi), jnp.asarray(psi), jnp.asarray(weights)))
    local = theirs - np.asarray(chunk.kinetic_nonlocal(zero, jnp.asarray(psi)))
    total = np.einsum("kn,kng,kng->", weights, psi.conj(), theirs)
    local_part = np.einsum("kn,kng,kng->", weights, psi.conj(), local)
    assert abs(form - (total - local_part)) < 1e-10 * abs(total)


def test_the_current_holds_the_hubbard_velocity(pseudo_dir, tmp_path):
    """``jax.grad`` of the chunk's band energy at ``kappa`` against ``VelocityOperator`` with ``ns``.

    ``VelocityOperator`` differentiates ``Calculation.at_kcart``, which rebuilds
    ``wfcU`` by the radial transform: the two share the definition of the
    Hubbard projectors and nothing else. Without the Hubbard term moving (the
    chunk's ``wfcU`` held at ``kappa = 0``) the same comparison is off by the
    term's velocity, which is asserted to be well above the agreement.
    """
    from defuqumat.response.velocity import VelocityOperator

    calc = _calculation(pseudo_dir, tmp_path)
    v_scf, ns = _frozen(calc)
    psi = _states(calc)
    weights = np.full(psi.shape[:2], 0.5)
    chunk = _setup(calc, psi, weights, v_scf, ns).first
    kappa = np.array([0.13, -0.07, 0.21])
    kcart = np.asarray(calc.system.kpoints.cartesian(calc.system.cell))
    mine = np.asarray(chunk.current(jnp.asarray(kappa), jnp.asarray(psi), jnp.asarray(weights)))
    elements = np.asarray(VelocityOperator(calc, v_scf, ns=jnp.asarray(ns),
                                           kcart=kcart + kappa[None]).matrix_elements(
        jnp.asarray(psi)[None]))[:, 0]
    theirs = np.einsum("kn,aknn->a", weights, elements).real
    assert np.abs(mine - theirs).max() < 1e-9 * np.abs(theirs).max()
    frozen_term = chunk.hubbard_at(jnp.zeros(3))
    held = eqx.tree_at(lambda c: c.hubbard, chunk, None)

    def energy(k):
        a = held._components(jnp.asarray(psi))
        wfcU = frozen_term.wfcU.astype(a.dtype)
        p = held._project(wfcU, a)
        hub = jnp.einsum("kn,knai,knai->", jnp.asarray(weights, dtype=a.dtype), jnp.conj(p),
                         held._contract(frozen_term.vns[None, None], p))
        return held.kappa_energy(k, jnp.asarray(psi), jnp.asarray(weights)) + jnp.real(hub)

    still = np.asarray(jax.grad(energy)(jnp.asarray(kappa)))
    assert np.abs(still - theirs).max() > 1e-4 * np.abs(theirs).max()


def test_the_refusals_that_stay_fire(pseudo_dir, tmp_path):
    """Ortho-atomic projectors, an updated potential, and an ``ns`` on a run without a U."""
    calc = _calculation(pseudo_dir, tmp_path, "ortho-atomic")
    v_scf, ns = _frozen(calc)
    psi = _states(calc)
    weights = np.full(psi.shape[:2], 0.5)
    with pytest.raises(NotImplementedError, match="ortho-atomic"):
        _setup(calc, psi, weights, v_scf, ns)
    calc = _calculation(pseudo_dir, tmp_path)
    for potential in ("hartree", "hxc"):
        with pytest.raises(NotImplementedError, match="ns\\(t\\)"):
            propagate(calc, psi, weights, v_scf, Kick(strength=0.01, direction=(1, 0, 0)),
                      dt=0.05, duration=0.1, k_batch=None, potential=potential, ns=ns)
        with pytest.raises(NotImplementedError, match="ns\\(t\\)"):
            propagate_orders(calc, psi, weights, v_scf, Kick(strength=1.0, direction=(1, 0, 0)),
                             dt=0.05, order=1, start=0.0, duration=0.1, k_batch=None,
                             potential=potential, ns=ns)
    path = tmp_path / "si-plain.in"
    path.write_text(re.sub(r"HUBBARD.*\n.*\n", "", CELL.format(kind="atomic", extra="")))
    system = build_system(read_pw_input(path))
    plain = Calculation(system, calc.pseudos)
    with pytest.raises(ValueError, match="no Hubbard U"):
        _setup(plain, psi[:1], weights[:1], plain.potential(plain.starting_density()).v_scf, ns)


def test_the_augmented_and_spinor_refusals_fire(pseudo_dir, tmp_path):
    """A U on an ultrasoft dataset and on spinors stays refused, with ``ns`` handed in."""
    from defuqumat.realtime.propagate import require_a_realtime_regime

    text = (Path(__file__).resolve().parents[1] / "data" / "qe" / "si2-us.in").read_text()
    path = tmp_path / "si2-us-ldau.in"
    path.write_text(text + "HUBBARD {atomic}\n U Si-3p 2.0\n")
    system = build_system(read_pw_input(path))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    augmented = Calculation(system, pseudos)
    spinor = _calculation(pseudo_dir, tmp_path, extra="noncolin = .true.")
    for calc, match in ((augmented, "ultrasoft or PAW"), (spinor, "spinors")):
        assert calc.is_hubbard
        with pytest.raises(NotImplementedError, match=match):
            require_a_realtime_regime(calc, calc.starting_ns())
