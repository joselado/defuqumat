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
