"""``dE_F``: the derivative of a bisection, which is not the bisection's own.

``bisect_fermi`` finds the Fermi level by halving a bracket and then refining
with Newton. Differentiating that *search* gives nothing useful -- every number
in it is a midpoint chosen by a comparison -- so the derivative is written down
from the implicit function theorem instead (``scf/occupations.py``). These are
its tests, and they matter beyond the residual solver that motivated them: the
same term is the Fermi-level shift of metallic linear response, and any future
DFPT or implicit-differentiation route through a metal goes through it.

Central differences are the reference, as ``PLAN.md`` D5 asks.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.scf.occupations import _wgauss_prime, bisect_fermi, w0gauss

SMEARINGS = [(0, "gaussian"), (1, "methfessel-paxton"), (-1, "cold"), (-99, "fermi-dirac")]


@pytest.fixture
def levels():
    """A metallic spectrum: eight bands at four k-points, six electrons."""
    rng = np.random.default_rng(0)
    return (
        jnp.asarray(np.sort(rng.normal(size=(4, 8)))),
        jnp.asarray(np.full(4, 0.5)),
        6.0,
        0.05,
    )


@pytest.mark.unit
@pytest.mark.parametrize("ngauss,name", SMEARINGS)
def test_wgauss_derivative_is_w0gauss(ngauss, name):
    """The rule uses ``jax.grad(wgauss)``; ``w0gauss`` is the transcription of
    the same derivative from ``Modules/w0gauss.f90``. They must be the same
    function, and this is what pins the sign convention that
    ``wgauss``'s own docstring warns about."""
    x = jnp.linspace(-4.0, 4.0, 41)
    assert np.allclose(_wgauss_prime(x, ngauss), w0gauss(x, ngauss), atol=1e-14)


@pytest.mark.unit
@pytest.mark.parametrize("ngauss,name", SMEARINGS)
def test_fermi_level_responds_to_the_eigenvalues(levels, ngauss, name):
    eigenvalues, weights, nelec, degauss = levels
    rng = np.random.default_rng(1)
    direction = jnp.asarray(rng.normal(size=eigenvalues.shape))

    def level(e):
        return bisect_fermi(e, weights, nelec, degauss, ngauss)

    _, tangent = jax.jvp(level, (eigenvalues,), (direction,))
    h = 1.0e-6
    difference = (level(eigenvalues + h * direction) - level(eigenvalues - h * direction)) / (2 * h)
    assert float(tangent) == pytest.approx(float(difference), abs=1e-8)
    # ...and it is not the zero the bisection would have given on its own.
    assert abs(float(tangent)) > 1e-3


@pytest.mark.unit
def test_fermi_level_responds_to_nelec_and_degauss(levels):
    """The other two tangent slots. They are not exercised by the SCF residual,
    which moves only the eigenvalues, and are written because a rule that is
    silently wrong in a slot nobody uses is a trap for whoever uses it next."""
    eigenvalues, weights, nelec, degauss = levels
    h = 1.0e-6

    def by_electrons(n):
        return bisect_fermi(eigenvalues, weights, n, degauss, 0)

    def by_smearing(g):
        return bisect_fermi(eigenvalues, weights, nelec, g, 0)

    for function, point in ((by_electrons, nelec), (by_smearing, degauss)):
        _, tangent = jax.jvp(function, (point,), (1.0,))
        difference = (function(point + h) - function(point - h)) / (2 * h)
        assert float(tangent) == pytest.approx(float(difference), abs=1e-7)


@pytest.mark.unit
def test_gapped_spectrum_has_no_fermi_response(levels):
    """With a wide gap and a narrow smearing there is no density of states at
    the Fermi level, the implicit derivative's denominator vanishes, and the
    tangent is forced to zero rather than to an infinity."""
    _, weights, nelec, _ = levels
    eigenvalues = jnp.asarray(np.tile([-5.0, -5.0, -5.0, 5.0, 5.0, 5.0, 5.0, 5.0], (4, 1)))
    rng = np.random.default_rng(2)
    direction = jnp.asarray(rng.normal(size=eigenvalues.shape))
    _, tangent = jax.jvp(
        lambda e: bisect_fermi(e, weights, nelec, 0.01, 0), (eigenvalues,), (direction,)
    )
    assert float(tangent) == 0.0


@pytest.mark.unit
@pytest.mark.parametrize("noncolin,channels", [(False, 1), (False, 2), (True, 4)])
def test_the_fermi_level_shift_counts_the_charge_and_not_the_magnetization(noncolin, channels):
    """``ef_shift`` fixes the electron count, so only the charge enters it.

    At ``nspin_mag = 4`` the channel axis is ``(n, m_x, m_y, m_z)``, and summing
    it put the magnetization into the count (``OPEN.md`` Part XXVI item 3). The
    probe is a response with a charge of 0.3 electrons and a magnetization of
    0.7 per row: the shift has to be ``-0.3 / N(e_F)`` regardless of the
    magnetization, and collinear the count is up plus down.
    """
    from types import SimpleNamespace

    from defuqumat.response.sternheimer import SternheimerSolver

    grid, volume, dos_ef = (2, 3, 4), 12.0, 1.7
    element = volume / np.prod(grid)
    rows = np.full((channels,) + grid, 0.3 / (channels * volume))
    if noncolin:
        rows[0] = 0.3 / volume
        rows[1:] = 0.7 / volume
    drho = jnp.asarray(np.stack([rows, 2.0 * rows]))
    ldos = jnp.asarray(np.full((channels,) + grid, dos_ef / (volume * (1 if noncolin else channels))))
    if noncolin:
        ldos = ldos.at[1:].set(0.0)
    stub = SimpleNamespace(calculation=SimpleNamespace(
        noncolin=noncolin, system=SimpleNamespace(cell=SimpleNamespace(volume=volume))))
    corrected, shift = SternheimerSolver.fermi_level_shift(stub, drho, levels=(ldos, dos_ef))
    np.testing.assert_allclose(np.asarray(shift), [-0.3 / dos_ef, -0.6 / dos_ef], rtol=1e-12)
    charge = corrected[:, 0] if noncolin else jnp.sum(corrected, axis=1)
    np.testing.assert_allclose(np.asarray(jnp.sum(charge, axis=(1, 2, 3))) * element, 0.0, atol=1e-12)
