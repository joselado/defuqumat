"""The ultracell's stopping test carries tau's residual (audit 2026-11-15, ``ultracell`` F2).

``tauk_ddot`` on the box is checked against Parseval: with every ``G + Q``
kept, the sum over the box of ``|r_G|^2`` with ``r_G = FFT(r)/points`` is
``sum_r r^2 / points``, so the term is ``0.5 V w sum_r r^2 / points``. The
FFT side and the real-space side share nothing but the field.
"""

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import defuqumat  # noqa: F401
from defuqumat.ultracell.driver import _tau_accuracy
from defuqumat.units import E2, FPI

pytestmark = pytest.mark.unit


def test_the_box_tau_residual_is_tauk_ddot():
    rng = np.random.default_rng(3)
    residual = rng.normal(size=(1, 6, 6, 8))
    points = 6 * 6 * 8
    ultracell = SimpleNamespace(points=points, volume=lambda cell: 270.0)
    keep = jnp.ones((6, 6, 8), dtype=bool)
    weight = E2 * FPI / (2.0 * np.pi) ** 2
    expected = 0.5 * 270.0 * weight * float(np.sum(residual ** 2)) / points
    assert _tau_accuracy(jnp.asarray(residual), ultracell, None, keep) == pytest.approx(expected, rel=1e-12)
    # and it is in the stopping test, not only defined
    import inspect

    from defuqumat.ultracell import driver

    assert "accuracy += _tau_accuracy(new_tau - tau_state" in inspect.getsource(driver)
