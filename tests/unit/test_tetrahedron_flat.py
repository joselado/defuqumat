"""The tetrahedron integrand of a nearly flat band (audit 2026-11-15, ``scf-small`` F1).

The denominators of the integrand are products of corner differences. They
were floored at 1e-12 as a whole, so four corners within 1e-5 Ry, a flat band
or a localized level in a large cell, gave a live branch with a clamped
denominator: the occupied fraction read 7.3e-4 where it is 0.1215. The checks
are against the closed form of each branch, computed here with no clamp, and
against the sum rule that the four Bloechl weights add up to the occupied
fraction.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import defuqumat  # noqa: F401
from defuqumat.scf.tetrahedra import _bloechl_weights, _occupied_fraction

pytestmark = pytest.mark.unit


def _exact(e, energy):
    e1, e2, e3, e4 = e
    if energy < e1:
        return 0.0
    if energy < e2:
        return (energy - e1) ** 3 / ((e2 - e1) * (e3 - e1) * (e4 - e1))
    if energy < e3:
        a = energy - e2
        return ((e2 - e1) ** 2 + 3 * (e2 - e1) * a + 3 * a ** 2
                - (e3 - e1 + e4 - e2) / ((e3 - e2) * (e4 - e2)) * a ** 3) / (
                    (e3 - e1) * (e4 - e1))
    if energy < e4:
        return 1.0 - (e4 - energy) ** 3 / ((e4 - e1) * (e4 - e2) * (e4 - e3))
    return 1.0


@pytest.mark.parametrize("scale", [1.0, 1e-5, 1e-7])
def test_the_occupied_fraction_of_a_flat_tetrahedron_is_the_closed_form(scale):
    e = np.array([0.0, 1.0, 2.0, 3.0]) * scale
    for x in (0.3, 0.9, 1.5, 2.2, 2.9):
        energy = x * scale
        got = float(_occupied_fraction(jnp.asarray(e), energy))
        assert got == pytest.approx(_exact(e, energy), rel=1e-9, abs=1e-14), (scale, x)


def test_the_gradient_is_finite_on_a_degenerate_tetrahedron():
    """Two pairs of equal corners: every dead branch divides by zero if it is not masked."""
    e = jnp.asarray([0.0, 0.0, 1.0, 1.0])
    for energy in (-0.5, 0.5, 1.5):
        de = jax.grad(lambda c: _occupied_fraction(c, energy))(e)
        dE = jax.grad(lambda x: _occupied_fraction(e, x))(energy)
        assert np.all(np.isfinite(np.asarray(de))) and np.isfinite(float(dE))


def test_the_bloechl_weights_of_a_flat_tetrahedron_add_up_to_its_fraction():
    """The four corner weights sum to the fraction (a full tetrahedron gives 0.25 per corner)."""
    e = np.array([0.0, 1.0, 2.0, 3.0]) * 1e-5
    for x in (0.9, 1.5, 2.2):
        ef = x * 1e-5
        weights = np.asarray(_bloechl_weights(jnp.asarray(e), ef))
        # the correction sum_i (sum e - 4 e_i) / 40 is zero, so it drops out of the sum
        assert weights.sum() == pytest.approx(_exact(e, ef), rel=1e-9), x
