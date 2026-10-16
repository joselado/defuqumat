"""The transverse spin susceptibility in three regimes beyond plain ``nspin = 2``.

``notes/INCOMPATIBILITIES.md`` I58, I59 and I61, each closed on the Goldstone
identity ``X_0 B = m`` of :mod:`defuqumat.tddft.spinkernel`, read where the
regime puts it:

* **a uniform field** (I58): at the Larmor frequency ``2 B``, not at zero, and
  only there. The other two candidates, ``0`` and ``-2 B``, are asserted to
  fail, which is what makes the passing one a measurement;
* **a Hubbard U** (I61): over the plane waves *and* the Hubbard channel, and
  the plane waves alone are asserted to fail by the Hubbard field's size;
* **a collinear moment run as a spinor** (I59): the same numbers as the
  ``nspin = 2`` run of the same cell.

Every cell is ``h-fcc-magnon.in`` or a one-line change of it, at
``ecut_response = 12`` and ``nbnd = 30`` (60 for the spinor), where that cell's
zero-field residual is 0.0057.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.tddft.spinchi0 import larmor_frequency, transverse_response
from defuqumat.tddft.spinkernel import (
    exchange_field,
    goldstone_residual,
    response_kernel,
    transverse_kernel_matrix,
)
from defuqumat.workflows.nscf import fixed_density_states

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"
ECUT = 12.0


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(text: str):
    calculator = Calculator.from_text(text, pseudo_dir=PSEUDO)
    scf = calculator.get_scf()
    scf.require_converged("the magnon regime tests")
    return calculator, scf


def _lambda(chi, kernel, index):
    return float(np.linalg.eigvals(np.asarray(chi.x[index]) @ kernel).real.max())


# --- I58: Larmor --------------------------------------------------------------


def test_the_uniform_mode_sits_at_the_larmor_frequency():
    """``h-fcc-magnon-field.in``: ``X_0(0, 2B) B_xc = m``, and not at 0 or -2B.

    D22, one sample: 0.00585 at ``+2B = 0.02`` Ry against 0.171 at zero and
    0.286 at ``-2B``; ``lambda_max(X_0 F)`` 1.0184 at ``2B``. The cell is a
    metal whose moment (0.321) is there because of the field, so this is not
    the zero-field state relabelled.
    """
    calculator, scf = _converged((CASES / "h-fcc-magnon-field.in").read_text())
    larmor = larmor_frequency(scf.magnetic_field, scf.field_scale)
    assert larmor == pytest.approx(0.02)
    calculation, _, eigenvalues, wavefunctions = fixed_density_states(
        calculator.system, calculator.pseudos, scf.density, nbnd=30,
        field=scf.magnetic_field, field_scale=scf.field_scale)
    chi = transverse_response(
        calculation, wavefunctions, eigenvalues, (0.0, 0.0, 0.0),
        np.array([larmor, 0.0, -larmor]), ecut_response=ECUT, broadening=0.0)
    density = jnp.asarray(scf.density)
    residuals = [goldstone_residual(chi, calculation, density, index=i) for i in range(3)]
    assert residuals[0] < 0.01
    assert min(residuals[1:]) > 10 * residuals[0]
    kernel = np.asarray(transverse_kernel_matrix(calculation, density, chi.sphere))
    assert _lambda(chi, kernel, 0) == pytest.approx(1.0, abs=0.03)


def test_the_workflow_reads_the_identity_at_the_larmor_frequency():
    """The same through ``Calculator.get_spin_susceptibility``, which must
    forward the converged field: ``.larmor`` is ``2B`` and ``.goldstone`` the
    residual there, while ``.enhancement`` stays at zero frequency, below one
    for a gapped mode."""
    calculator, _ = _converged((CASES / "h-fcc-magnon-field.in").read_text())
    result = calculator.get_spin_susceptibility(
        (0.0, 0.0, 0.0), np.linspace(0.001, 0.04, 40), nbnd=30,
        ecut_response=ECUT, crossing_points=40)
    assert result.larmor == pytest.approx(0.02)
    assert result.goldstone < 0.01
    assert result.enhancement < 0.9
    assert result.larmor_enhancement == pytest.approx(1.0, abs=0.03)
    assert result.method == "crossing"
    assert abs(result.magnon - 0.02) < 0.005


# --- I61: the Hubbard channel -------------------------------------------------


def test_the_hubbard_channel_closes_the_goldstone_identity():
    """``h-fcc-magnon-ldau.in``: with the channel 0.0047, without it 0.197.

    D22, one sample. ``lambda_max`` reads 1.0143 with the channel and 0.818
    from the plane waves alone, so the term is the whole of the difference.
    """
    calculator, scf = _converged((CASES / "h-fcc-magnon-ldau.in").read_text())
    ns = np.asarray(scf.ns)
    calculation, _, eigenvalues, wavefunctions = fixed_density_states(
        calculator.system, calculator.pseudos, scf.density, nbnd=30, ns=scf.ns)
    chi = transverse_response(
        calculation, wavefunctions, eigenvalues, (0.0, 0.0, 0.0), np.array([0.0]),
        ecut_response=ECUT, broadening=0.0)
    assert chi.hubbard == ((0, 0, 0),)
    density = jnp.asarray(scf.density)
    full = goldstone_residual(chi, calculation, density, ns=ns)

    nm = chi.sphere.nm
    grid = np.asarray(calculation.basis.dense.grid)
    miller = np.asarray(chi.sphere.miller) % grid

    def plus(field):
        coefficients = np.fft.fftn(field) / float(np.prod(grid))
        return np.conj(coefficients[miller[:, 0], miller[:, 1], miller[:, 2]])

    rho = np.asarray(scf.density)
    target = plus(rho[0] - rho[1])
    bxc = np.asarray(exchange_field(calculation, density))
    alone = np.linalg.norm(np.asarray(chi.x[0])[:nm, :nm] @ plus(bxc) - target) / np.linalg.norm(target)
    assert full < 0.01
    assert alone > 10 * full

    kernel = response_kernel(calculation, density, chi, ns=ns)
    assert _lambda(chi, kernel, 0) == pytest.approx(1.0, abs=0.03)


# --- I59: a collinear spinor --------------------------------------------------


def test_a_collinear_spinor_run_is_the_nspin_2_run():
    """``h-fcc-magnon.in`` as a spinor with its moment along z, at twice the bands.

    The identity on the spinor run itself, and ``lambda_max`` and the head of
    ``X_0`` at two wavevectors against the ``nspin = 2`` run of the same cell.
    """
    text = (CASES / "h-fcc-magnon.in").read_text()
    spinor_text = text.replace("    nspin = 2\n", "    noncolin = .true.\n    angle1(1) = 0.0\n")
    readings = []
    for source, nbnd in ((text, 30), (spinor_text, 60)):
        calculator, scf = _converged(source)
        calculation, _, eigenvalues, wavefunctions = fixed_density_states(
            calculator.system, calculator.pseudos, scf.density, nbnd=nbnd)
        density = jnp.asarray(scf.density)
        row = []
        for q in ((0.0, 0.0, 0.0), (0.0, 0.0, 0.25)):
            chi = transverse_response(
                calculation, wavefunctions, eigenvalues, q, np.array([0.0, 0.02]),
                ecut_response=ECUT, broadening=0.0)
            kernel = np.asarray(transverse_kernel_matrix(calculation, density, chi.sphere))
            row.append((_lambda(chi, kernel, 0), complex(np.asarray(chi.x[1])[0, 0])))
            if q == (0.0, 0.0, 0.0):
                assert goldstone_residual(chi, calculation, density) < 0.01
        readings.append(row)
    for (lam_c, head_c), (lam_s, head_s) in zip(*readings):
        assert lam_s == pytest.approx(lam_c, abs=1.0e-4)
        assert abs(head_s - head_c) < 1.0e-4 * abs(head_c)
