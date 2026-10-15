"""P145: the finite field by the Berry phase inside a real run, against ``pw.x``.

QE's ``pw_electric`` chain on Si8 (norm-conserving PBE, 63 k-points as 9
strings of 7 along ``z``), regenerated with the vendored ``pw.x`` 7.5 at
``conv_thr = 1e-11``: ``electric.in`` converges the crystal, ``electric-1``
switches the Berry operator on at ``efield = 0`` and ``electric-2`` turns the
field to 0.001 Ry a.u. with ``nberrycyc = 3``. Here the same chain: the
zero-field state, then the field switched on from it (``starting_from``),
because from random states the neighbouring occupied subspaces are nearly
orthogonal and the operator is O(1) Ry, so no QE input starts the field cold.

The total energy is mostly ``-E ion_pol`` (0.1152 of the 0.1156 Ry the field
moves it), so the electronic polarization is compared on its own, and the
identity that shares nothing with ``pw.x`` is on it alone:
``d(E + E ion_pol)/dE = -el_pol``, the Hellmann-Feynman statement for the
enthalpy, by a central difference of two self-consistent runs.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defuqumat.forces import compute_forces
from defuqumat.io import read_qe_output
from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation, run_scf
from defuqumat.system import build_system

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: Measured 1.2e-8 (field) and 1e-10 (zero field) Ry. ``pw.x``'s own total
#: moves by 2e-8 between its last two iterations (-63.06610339, -63.06610337).
ENERGY = 3e-8
#: ``el_pol``: measured 8.6e-6, inside the 1.6e-5 by which ``pw.x``'s own value
#: moves between its last two iterations.
DIPOLE = 3e-5
#: Measured 1e-7 Ry/bohr on both routes, which agree with each other to 1e-8.
FORCE = 1e-6


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


def _load(case, efield=None, spin=""):
    text = (CASES / f"{case}.in").read_text()
    if efield is not None:
        text = text.replace("efield=0.001", f"efield={efield}")
    if spin:
        text = text.replace("nosym=.true.", f"nosym=.true., {spin}")
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, Calculation(system, pseudos)


@lru_cache(maxsize=1)
def _zero():
    system, pseudos, calculation = _load("si8-lelfield-zero")
    return calculation, run_scf(system, pseudos, calculation=calculation, conv_thr=1e-11,
                                mixing_beta=0.5, max_iterations=100)


@lru_cache(maxsize=3)
def _field(efield=None):
    _, zero = _zero()
    system, pseudos, calculation = _load("si8-lelfield", efield)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-11,
                     mixing_beta=0.5, max_iterations=100, starting_from=zero)
    assert result.converged
    return calculation, result


def test_zero_field():
    _, zero = _zero()
    reference = read_qe_output(CASES / "reference.out.si8-lelfield-zero")
    assert zero.total_energy == pytest.approx(reference.total_energy, abs=ENERGY)


def test_total_energy_and_dipoles():
    _, result = _field()
    reference = read_qe_output(CASES / "reference.out.si8-lelfield")
    assert result.total_energy == pytest.approx(reference.total_energy, abs=ENERGY)
    text = (CASES / "reference.out.si8-lelfield").read_text()
    electronic = float(text.rsplit("Electronic Dipole per cell (Ry a.u.)", 1)[1].split()[0])
    ionic = float(text.rsplit("Ionic Dipole per cell (Ry a.u.)", 1)[1].split()[0])
    el, ion, acc = result.berry_dipoles
    assert acc == 0.0
    assert ion == pytest.approx(ionic, abs=1e-10)
    assert el == pytest.approx(electronic, abs=DIPOLE)


@pytest.mark.parametrize("method", ["autodiff", "analytic"])
def test_forces(method):
    calculation, result = _field()
    reference = read_qe_output(CASES / "reference.out.si8-lelfield")
    forces = compute_forces(calculation, result, method=method).forces
    np.testing.assert_allclose(forces, reference.forces, atol=FORCE)


def test_the_field_derivative_is_the_electronic_dipole():
    """``E(E+h) - E(E-h) = -int el_pol dE`` after the ions' exact linear term is removed.

    Three self-consistent runs at 0.0005, 0.001 and 0.0015 Ry a.u. The
    integral is Simpson's rule on their own ``el_pol``, exact to ``h^4``. The
    plain central difference against ``-el_pol`` at the centre is off by
    3.0e-4, which is the third-order response: ``el = chi1 E + chi3 E^3``
    (the cell keeps inversion, so there is no ``E^2``) leaves the central
    difference ``chi3 E h^2`` short, and ``chi3`` fitted from the five-field
    residual of notebook 54 predicts 2.9e-4. The test would be about ``chi3``
    rather than about the Hellmann-Feynman statement.
    """
    _, centre = _field()
    _, plus = _field(0.0015)
    _, minus = _field(0.0005)
    ion = centre.berry_dipoles[1]
    shifted = lambda result, e: result.total_energy + e * ion
    difference = shifted(plus, 0.0015) - shifted(minus, 0.0005)
    el = [r.berry_dipoles[0] for r in (minus, centre, plus)]
    simpson = 0.001 * (el[0] + 4.0 * el[1] + el[2]) / 6.0
    assert difference == pytest.approx(-simpson, abs=1e-7)


def test_the_frozen_energy_is_the_total_energy():
    """The force functional, Berry enthalpy included, evaluated at the converged
    state is the SCF total: the identity every force here rests on."""
    from defuqumat.forces.energy import energy_at, state_from_result

    calculation, result = _field()
    frozen = float(energy_at(calculation, state_from_result(result)))
    assert frozen == pytest.approx(result.total_energy, abs=1e-8)


#: The same chain in another spin regime with no moment: the identity with
#: ``nspin = 1`` (I14, I15). Measured at conv_thr = 1e-11 on D22, see the tests.
NSPIN2 = "nspin=2, tot_magnetization=0"
SPINOR = "noncolin=.true."


@lru_cache(maxsize=2)
def _regime(spin):
    """The zero-field run and the field switched on from it, both in ``spin``'s regime.

    Its own zero-field start, never ``_zero()``: ``starting_from`` across a
    change of spin regime seeds a moment when the source has none (P23).
    """
    system, pseudos, calculation = _load("si8-lelfield-zero", spin=spin)
    zero = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-11,
                   mixing_beta=0.5, max_iterations=100)
    assert zero.converged
    system, pseudos, calculation = _load("si8-lelfield", spin=spin)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-11,
                     mixing_beta=0.5, max_iterations=100, starting_from=zero)
    assert result.converged
    return calculation, result, zero


@pytest.mark.parametrize("spin, energy, dipole", [
    (NSPIN2, 1e-8, 1e-6),
    (SPINOR, 1e-8, 1e-6),
])
def test_a_nonmagnetic_spin_regime_is_the_unpolarized_field(spin, energy, dipole):
    """I14 and I15: ``nspin = 2`` at ``tot_magnetization = 0``, and a spinor without
    spin-orbit coupling, are the ``nspin = 1`` field run: ``el_pol`` (a sum of two
    channels' phases, or one channel of twice the bands, where ``nspin = 1`` doubles
    one), ``ion_pol``, and what the field does to the total, ``E(E) - E(0)`` of each
    regime's own chain. The totals themselves are compared too, at zero field and
    with it, since a spin regime's exchange-correlation need not be the unpolarized
    one's to the last digit."""
    _, reference = _field()
    _, reference_zero = _zero()
    calculation, result, zero = _regime(spin)
    shift = (result.total_energy - zero.total_energy) - (
        reference.total_energy - reference_zero.total_energy)
    print(f"\n{spin}: dE(0) = {zero.total_energy - reference_zero.total_energy:.3e}, "
          f"dE(E) = {result.total_energy - reference.total_energy:.3e}, "
          f"d[E(E) - E(0)] = {shift:.3e}, "
          f"d el_pol = {result.berry_dipoles[0] - reference.berry_dipoles[0]:.3e}, "
          f"iterations {result.iterations}")
    assert shift == pytest.approx(0.0, abs=energy)
    el, ion, acc = result.berry_dipoles
    assert acc == 0.0
    assert ion == pytest.approx(reference.berry_dipoles[1], abs=1e-12)
    assert el == pytest.approx(reference.berry_dipoles[0], abs=dipole)


@pytest.mark.parametrize("spin", [NSPIN2, SPINOR])
def test_a_nonmagnetic_spin_regime_has_the_unpolarized_forces(spin):
    calculation, result, _ = _regime(spin)
    reference = read_qe_output(CASES / "reference.out.si8-lelfield")
    for method in ("autodiff", "analytic"):
        forces = compute_forces(calculation, result, method=method).forces
        np.testing.assert_allclose(forces, reference.forces, atol=FORCE)


def test_an_automatic_grid_is_pw_x_l3dstring_field():
    """I57: ``K_POINTS automatic 3 3 7`` with ``efield_cart(3)``, against ``pw.x``'s own
    ``l3dstring`` run of the same input (``reference.out.si8-lelfield-automatic``), whose
    total, forces and ``el_pol`` along z match its explicit-list run to every printed
    digit. Here the grid becomes the same 63 points, so the run is the explicit one's."""
    _, zero = _zero()
    system, pseudos, calculation = _load("si8-lelfield-automatic")
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-11,
                     mixing_beta=0.5, max_iterations=100, starting_from=zero)
    assert result.converged
    reference = read_qe_output(CASES / "reference.out.si8-lelfield-automatic")
    text = (CASES / "reference.out.si8-lelfield-automatic").read_text()
    # each iteration prints the three directions in order; the last is z
    electronic = float(text.rsplit("Electronic Dipole per cell (Ry a.u.)", 1)[1].split()[0])
    ionic = float(text.rsplit("Ionic Dipole per cell (Ry a.u.)", 1)[1].split()[0])
    _, explicit = _field()
    print(f"\nautomatic: dE(pw.x) = {result.total_energy - reference.total_energy:.3e}, "
          f"dE(explicit) = {result.total_energy - explicit.total_energy:.3e}, "
          f"d el_pol(pw.x) = {result.berry_dipoles[0] - electronic:.3e}")
    assert result.total_energy == pytest.approx(reference.total_energy, abs=ENERGY)
    el, ion, acc = result.berry_dipoles
    assert ion == pytest.approx(ionic, abs=1e-10)
    assert el == pytest.approx(electronic, abs=DIPOLE)
    forces = compute_forces(calculation, result, method="analytic").forces
    np.testing.assert_allclose(forces, reference.forces, atol=FORCE)
    assert result.total_energy == pytest.approx(explicit.total_energy, abs=1e-10)


def test_a_streamed_store_is_the_in_memory_run():
    """I56: ``wfc_store = 'stream'`` (the states in host memory, walked a k-point at a
    time) against the in-memory field run: the total, ``el_pol``, the iteration count,
    and the force, which on the streamed store takes the chunked route with the
    field's ions in its global term and its electrons as a constant."""
    from defuqumat.forces.energy import state_from_result
    from defuqumat.forces.chunked import wants_chunks

    calculation, reference = _field()
    _, zero = _zero()
    system, pseudos, streamed_calculation = _load("si8-lelfield")
    result = run_scf(system, pseudos, calculation=streamed_calculation, conv_thr=1e-11,
                     mixing_beta=0.5, max_iterations=100, starting_from=zero,
                     wfc_store="stream")
    assert result.converged
    assert isinstance(result.wavefunctions, np.ndarray)
    state = state_from_result(result)
    assert wants_chunks(streamed_calculation, state)
    forces = compute_forces(streamed_calculation, result).forces
    expected = compute_forces(calculation, reference).forces
    print(f"\nstreamed: dE = {result.total_energy - reference.total_energy:.3e}, "
          f"d el_pol = {result.berry_dipoles[0] - reference.berry_dipoles[0]:.3e}, "
          f"dF = {np.abs(np.asarray(forces) - np.asarray(expected)).max():.3e}, "
          f"iterations {result.iterations} and {reference.iterations}")
    assert result.total_energy == pytest.approx(reference.total_energy, abs=1e-9)
    assert result.berry_dipoles[0] == pytest.approx(reference.berry_dipoles[0], abs=1e-7)
    np.testing.assert_allclose(forces, expected, atol=1e-8)
