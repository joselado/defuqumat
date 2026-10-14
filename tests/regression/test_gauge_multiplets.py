"""A band-resolved first-order quantity inside a degenerate multiplet, under a random gauge.

``PLAN.md`` P151: the gauge sweep over the front door (``tools/gauge_sweep.py``) rotated the
converged states by a random unitary inside every multiplet and found three quantities that moved
by order one, each a per-band diagonal read inside a multiplet (rule D4): the band velocities, the
strain response's first-order eigenvalues and two diagnostics of the effective mass. The first two
now take the spectrum of the multiplet's block, which is what degenerate perturbation theory gives
for the slopes of the bands the level splits into; these tests pin that on ``si2-lda-gap.in`` at
2x2x2 with ``Gamma`` (a threefold block at ``Gamma``, pairs at ``L``), and pin that the rotation
reaches the assembly, since a test whose rotation did nothing would pass the same way.
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _calculator():
    from defuqumat import Calculator

    text = (CASES / "si2-lda-gap.in").read_text()
    assert "\n 6 6 6 0 0 0" in text
    return Calculator.from_text(text.replace("\n 6 6 6 0 0 0", "\n 2 2 2 0 0 0"),
                                pseudo_dir=PSEUDO, announce=False)


@lru_cache(maxsize=1)
def _silicon():
    calc = _calculator()
    return calc, calc.get_scf()


def _rotated(result, seed, tol=1e-5):
    """``result`` with a random unitary inside every multiplet of its states."""
    import dataclasses

    from defuqumat.scf.occupations import multiplet_labels

    e = np.asarray(result.eigenvalues_by_spin)
    labels = multiplet_labels(e, tol)
    psi = np.array(result.wavefunctions)
    rng = np.random.default_rng(seed)
    blocks = 0
    for index in np.ndindex(labels.shape[:-1]):
        for value in np.unique(labels[index]):
            bands = np.flatnonzero(labels[index] == value)
            if bands.size > 1:
                a = rng.normal(size=(bands.size,) * 2) + 1j * rng.normal(size=(bands.size,) * 2)
                q, _ = np.linalg.qr(a)
                psi[index][bands] = q.T @ psi[index][bands]
                blocks += 1
    assert blocks >= 3
    return dataclasses.replace(result, wavefunctions=jnp.asarray(psi))


def test_the_band_velocities_do_not_see_the_basis_of_a_multiplet():
    """Measured before the change: the per-band diagonal moved by 0.12 to 0.86 of the largest
    velocity here. After it, the block spectra agree to round-off, and the raw diagonal of the
    kernel still moves, which is the check that the rotation reached the states."""
    from defuqumat.response.velocity import VelocityOperator, band_velocities

    calc, result = _silicon()
    rotated = _rotated(result, 7)
    plain = band_velocities(calc.calculation, result)
    turned = band_velocities(calc.calculation, rotated)
    scale = np.abs(plain.velocities_by_spin).max()
    assert plain.degenerate_by_spin.any()
    assert np.abs(turned.velocities_by_spin - plain.velocities_by_spin).max() < 1e-10 * scale
    assert np.array_equal(np.isnan(plain.speeds), plain.degenerate_by_spin)

    # the raw diagonal, which is what the code returned before P151, moves by order one
    potential = calc.calculation.potential(result.density)
    operator = VelocityOperator(calc.calculation, potential.v_scf, None, result.ns)
    raw = [operator.band_velocities(r.wavefunctions, r.eigenvalues).velocities_by_spin
           for r in (result, rotated)]
    assert np.abs(raw[1] - raw[0]).max() > 1e-2 * scale


def test_the_strain_eigenvalue_response_does_not_see_the_basis_of_a_multiplet():
    """Measured before the change: ``deigenvalues`` moved by 0.27 to 0.42 here. After it the
    move is the self-consistency's own noise at ``tr2 = 1e-14``."""
    calc, result = _silicon()
    plain = calc.get_strain_response()
    # a second calculator holding the rotated states, so the cached response is not reused
    other = _calculator()
    other._scf, other._scf_options = _rotated(result, 11), calc._scf_options
    turned = other.get_strain_response()
    scale = np.abs(plain.deigenvalues).max()
    assert np.abs(turned.deigenvalues - plain.deigenvalues).max() < 1e-6 * scale
