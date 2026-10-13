"""P144: the saw-tooth field and the dipole correction inside a real run.

The references are the vendored ``pw.x`` on the same inputs
(``tests/data/qe/efield-co*.in``, generated with ``tools/generate_reference.py``):
QE's ``pw_relax/relax-el.in`` at its first geometry, an ultrasoft CO molecule in
a 10 bohr cube in a field of 0.001 Ha a.u. along ``z``, once as it is and once
with ``dipfield`` on top. Both sides run at ``conv_thr = 1e-13``, because the
cell is mostly vacuum and at 1e-10 the force still moves by 1e-5 Ry/bohr under
``dipfield`` and by 3e-6 without it, on either code; at 1e-13 the autodiff and
analytic routes agree to 2e-8 and 2e-7.

**Without ``dipfield`` the one-electron term and the printed correction are
not this code's**, by construction: ``pw.x`` keeps the electrons' share of the
field inside ``eband`` through ``vltot`` and prints the ions' share alone
(0.01440000 Ry), where here the field sits in ``v_scf`` and its whole energy is
one term. The total is the same, and the test says by how much the two
splittings differ: by ``int rho v``, the electrons' share.

Two identities share no machinery with ``pw.x``. The derivative of the total
energy with respect to the amplitude is the dipole, ``dE/d(eamp) = -(Omega/4
pi) e2 D``, a Hellmann-Feynman statement about the self-consistent state taken
by a central difference of two SCF runs. And an unpolarized ``nspin = 2`` run
is the ``nspin = 1`` one, which is the check that the spin-independent
potential reaches both channels in full.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defuqumat.forces import compute_forces
from defuqumat.io import read_qe_output
from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation, run_scf
from defuqumat.system import build_system
from defuqumat.workflows.relax import run_relax
from tests.tolerances import USPP_TERM_RY

pytestmark = [pytest.mark.regression, pytest.mark.slow,
              pytest.mark.filterwarnings("ignore:tefield")]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: Achieved 2e-10 and 4e-10 Ry against ``pw.x``'s eight printed decimals.
ENERGY = 1e-8
#: Achieved 3.0e-7 (plain) and 3.7e-7 (``dipfield``) Ry/bohr, autodiff route;
#: QE prints eight decimals and its own ``SCF correction`` reads 1e-6.
FORCE = 1e-6
#: Autodiff against analytic, measured 2e-8 and 2e-7 at ``conv_thr = 1e-13``.
ROUTES = 5e-7


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


def _system(case, **changes):
    text = (CASES / f"{case}.in").read_text()
    for old, new in changes.items():
        assert old in text, old
        text = text.replace(old, new)
    from defuqumat.io.pwin import parse_pw_input
    return build_system(parse_pw_input(text))


@lru_cache(maxsize=2)
def _converged(case, conv_thr=1e-13, **changes):
    system = _system(case, **dict(changes))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=conv_thr,
                     max_iterations=100)
    assert result.converged
    return calculation, result


def _reference(case):
    path = CASES / f"reference.out.{case}"
    if not path.is_file():
        pytest.skip(f"no generated reference for {case}")
    return read_qe_output(path)


@pytest.mark.parametrize("case", ["efield-co", "efield-co-dipole"])
def test_total_energy(case):
    _, result = _converged(case)
    assert result.total_energy == pytest.approx(_reference(case).total_energy, abs=ENERGY)


@pytest.mark.parametrize("case", ["efield-co", "efield-co-dipole"])
@pytest.mark.parametrize("method", ["autodiff", "analytic"])
def test_forces(case, method):
    calculation, result = _converged(case)
    forces = compute_forces(calculation, result, method=method).forces
    np.testing.assert_allclose(forces, _reference(case).forces, atol=FORCE)


@pytest.mark.parametrize("case", ["efield-co", "efield-co-dipole"])
def test_the_two_force_routes_agree(case):
    calculation, result = _converged(case)
    autodiff = compute_forces(calculation, result, method="autodiff").forces
    analytic = compute_forces(calculation, result, method="analytic").forces
    np.testing.assert_allclose(autodiff, analytic, atol=ROUTES)


def test_dipfield_energy_and_dipole():
    """Under ``dipfield`` ``etotefield`` is the whole term, and the dipole is printed."""
    _, result = _converged("efield-co-dipole")
    reference = _reference("efield-co-dipole")
    assert result.efield_energy == pytest.approx(reference.energy_terms["efield"], abs=1e-8)
    text = (CASES / "reference.out.efield-co-dipole").read_text()
    printed = float(text.rsplit("Dipole   ", 1)[1].split()[0])
    assert result.efield_dipole == pytest.approx(printed, abs=1e-4)


def test_plain_field_splits_its_energy_as_described():
    """``pw.x``'s one-electron term holds ``int rho v_saw`` that this code's does not."""
    calculation, result = _converged("efield-co")
    reference = _reference("efield-co")
    assert reference.energy_terms["efield"] == pytest.approx(0.01440000, abs=1e-8)
    electrons = result.efield_energy - reference.energy_terms["efield"]
    # A single term against ``pw.x``, so ``USPP_TERM_RY`` and not the total's
    # tolerance: measured 1.5e-6 Ry here, against a share of 0.0143 Ry.
    assert (reference.energy_terms["one-electron"] - result.energy_terms["one-electron"]
            == pytest.approx(electrons, abs=USPP_TERM_RY))


def test_the_amplitude_derivative_is_the_dipole():
    """``dE/d(eamp) = -(Omega/4 pi) e2 D`` across two self-consistent runs.

    A central difference at ``eamp = 0.001 +- 0.0005``: the energy is quadratic
    in the field to the order this resolves (the polarizability), so the
    central difference carries no first-order error.
    """
    plus = _converged("efield-co-dipole", 1e-12, **{"eamp=0.001": "eamp=0.0015"})[1]
    minus = _converged("efield-co-dipole", 1e-12, **{"eamp=0.001": "eamp=0.0005"})[1]
    _, centre = _converged("efield-co-dipole", 1e-12)
    derivative = (plus.total_energy - minus.total_energy) / 0.001
    # ``efield_dipole`` is ``D Omega/(4 pi)``, so the right-hand side is ``-e2`` times it.
    assert derivative == pytest.approx(-2.0 * centre.efield_dipole, rel=1e-4)


def test_an_unpolarized_lsda_run_is_the_spinless_one():
    _, single = _converged("efield-co-dipole", 1e-11)
    _, double = _converged("efield-co-dipole", 1e-11,
                           **{"ecutrho =300,": "ecutrho =300, nspin=2, "
                                               "tot_magnetization=0,"})
    assert double.total_energy == pytest.approx(single.total_energy, abs=1e-8)
    assert double.efield_dipole == pytest.approx(single.efield_dipole, abs=1e-6)


def test_relax_el_relaxes_to_the_benchmark(qe_testsuite):
    """QE's own ``pw_relax/relax-el.in``, run as it is: the field stretches CO.

    The committed benchmark (``pw.x`` 6.1 at ``conv_thr = 1e-8``) ends at
    -43.1264735566 Ry with O at ``z = -1.0716`` bohr after three BFGS steps.
    """
    system = build_system(read_pw_input(qe_testsuite / "pw_relax" / "relax-el.in"))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    relaxed = run_relax(system, pseudos, conv_thr=1e-10)
    benchmark = read_qe_output(
        qe_testsuite / "pw_relax" / "benchmark.out.git.inp=relax-el.in")
    assert relaxed.converged
    assert relaxed.scf.total_energy == pytest.approx(benchmark.final_energy, abs=2e-6)
    z = np.asarray(relaxed.system.structure.positions)[:, 2]
    assert z[0] == pytest.approx(-1.071599420, abs=2e-3)
