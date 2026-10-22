"""I121: the displaced tabulated augmentation table takes a traced wavevector.

A spin spiral's transverse augmentation charge is ``Q_ij(G - q)``, and
``dE/dq`` rebuilds it at a traced ``q``. On a cell too large to store
``Q_ij(G)`` the table is QE's radial one, whose extent is
``cell_factor G_rho + |q|``, and ``|q|`` was read on the host, which a tracer
cannot give. ``shift_extent`` passes it from the host instead (the frozen
wavevector's modulus, in ``Calculation.at_spiral_q``).

What is checked here, with no SCF: the derivative in the shift of a scalar
built from the tabulated table is the stored table's derivative up to the
four-point stencil's interpolation error, and that error falls as the knot
spacing ``AUG_DQ`` is refined, at the rate a cubic Lagrange stencil gives. The
end-to-end ``dE/dq`` on a converged spiral is
``tests/regression/test_spiral_relaxation.py``
``test_a_tabulated_augmentation_table_gives_the_stored_gradient`` (slow).
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import defuqumat.pseudo.augmentation as augmentation
from defuqumat.calculator import Calculator
from defuqumat.pseudo.augmentation import TabulatedAugmentation, build_augmentation

pytestmark = pytest.mark.unit

SILICON_US = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1,
  ecutwfc = 10.0, ecutrho = 40.0
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-n-rrkjus_psl.0.1.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.27 0.25 0.23
K_POINTS automatic
 2 2 2 0 0 0
"""

#: The wavevector the derivative is taken at, cartesian in bohr^-1: a general
#: direction, so no component of the gradient is zero by symmetry.
Q0 = np.array([0.11, -0.07, 0.19])


@pytest.fixture(scope="module")
def calculation(pseudo_dir):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Calculator.from_text(SILICON_US, pseudo_dir, announce=False).calculation


def _scalar(calculation, max_bytes, extent):
    """``q -> Re sum_ij V_ij(q)``, the D_ij integrals of a fixed potential."""
    system = calculation.system
    dense = calculation.basis.dense
    rng = np.random.default_rng(3)
    potential = jnp.asarray(rng.normal(size=dense.ngm) + 1j * rng.normal(size=dense.ngm))

    def f(q):
        table = build_augmentation(calculation.pseudos, system.structure, system.cell,
                                   dense, max_bytes=max_bytes, shift=-q,
                                   shift_extent=extent)
        return sum(jnp.sum(jnp.real(block) + 0.3 * jnp.imag(block))
                   for block in table.cross_integrals(potential))
    return f


@pytest.mark.slow
def test_the_tabulated_derivative_converges_to_the_stored_one(calculation, monkeypatch):
    """Slow (about 15 s on one core): three tabulated builds and their gradients.

    Measured on this cell (2026-10-22, one core), the largest component of the
    gradient being 110.5: the tabulated derivative differs from the stored one
    by 5.3e-05, 2.6e-06 and 4.1e-07 at ``AUG_DQ = 0.04, 0.02, 0.01``, and
    8.5e-08 at 0.005, so it falls by 21 and then 6.4 and stops near 1e-7,
    where the stencil error meets the rest of the table's. Stopping at 0.01,
    QE's spacing and the code's, is deliberate. With the table's ``q``
    dependence dropped the derivative of this scalar would be zero, an error
    of 110.
    """
    q = jnp.asarray(Q0)
    extent = float(np.linalg.norm(Q0))
    stored = np.asarray(jax.grad(_scalar(calculation, None, None))(q))
    scale = float(np.abs(stored).max())
    assert scale > 1.0  # a slope to compare

    errors = {}
    for dq in (0.04, 0.02, 0.01):
        monkeypatch.setattr(augmentation, "AUG_DQ", dq)
        jax.clear_caches()
        assert isinstance(build_augmentation(
            calculation.pseudos, calculation.system.structure, calculation.system.cell,
            calculation.basis.dense, max_bytes=1, shift=-q, shift_extent=extent),
            TabulatedAugmentation)
        tabulated = np.asarray(jax.grad(_scalar(calculation, 1, extent))(q))
        assert np.all(np.isfinite(tabulated))
        errors[dq] = float(np.abs(tabulated - stored).max())
    # 0.04 -> 0.02 is in the stencil's regime (measured 21); 0.02 -> 0.01 is
    # where the floor begins (measured 6.4), so only a fall of 2 is asked there.
    assert errors[0.02] < errors[0.04] / 8
    assert errors[0.01] < errors[0.02] / 2
    assert errors[0.01] < 1e-8 * scale


def test_a_traced_shift_without_its_extent_is_still_refused(calculation):
    """The host read is still there when nobody passes the extent: it must fail loudly."""
    f = _scalar(calculation, 1, None)
    with pytest.raises((jax.errors.ConcretizationTypeError,
                        jax.errors.TracerArrayConversionError)):
        jax.grad(f)(jnp.asarray(Q0))


def test_a_shift_past_the_table_reads_nan_rather_than_a_clamp(calculation):
    """The one regime the extent does not cover: a traced ``q`` far from ``q0``.

    The table reaches ``2 G_rho + |q0|`` (12.9 bohr^-1 here, ``G_rho = 6.32``);
    a shift of 8 more along x puts ``|G - q|`` up to 14.3, past its end, and
    the stencil must say so with NaN rather than return a clamped, plausible
    number.
    """
    extent = float(np.linalg.norm(Q0))
    f = _scalar(calculation, 1, extent)
    assert np.isfinite(float(f(jnp.asarray(Q0))))
    assert not np.isfinite(float(f(jnp.asarray(Q0 + np.array([8.0, 0.0, 0.0])))))
