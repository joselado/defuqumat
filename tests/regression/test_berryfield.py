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


def _load(case, efield=None):
    text = (CASES / f"{case}.in").read_text()
    if efield is not None:
        text = text.replace("efield=0.001", f"efield={efield}")
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
