"""Route U step 1, the magnetic ultracell (Wang and Vafek, PRB 108, 245109 (2023)).

The fast checks: the parallel-transport gauge on a model whose centres are known, the
zero-flux limit (the magnetic machinery must give back the zero-field bands to round-off),
the free Landau problem on a small cell, the Hermiticity and rank guards, and the
refusals. The acceptance numbers on the graphene cell are probes
(``plans/orbital-field/mag_route_u.py``) and ``tests/regression/test_magnetic_ultracell_slow.py``.
"""

import jax
import numpy as np
import pytest

from defuqumat.magnetic.hybrid import outside_weight, half_width, transport_gauge
from defuqumat.magnetic.ultracell_field import assemble, hermiticity, hermiticity_guard, solve


@pytest.fixture(autouse=True)
def _clear():
    yield
    jax.clear_caches()


def _orbital_links(taus, n1, seed=0):
    """Links of a model whose Bloch states are point orbitals at ``taus`` (cells along a1).

    The periodic part of orbital ``s`` at ``k1 = i/n1`` is ``e^{-2 pi i k1 tau_s}`` on its
    own site, so ``<u_i|u_{i+1}> = diag(e^{-2 pi i tau/n1})``, and the closing link (``u_N(G)
    = u_0(G + b1)``) is the same. Each k-point gets a random unitary mixing.
    """
    rng = np.random.default_rng(seed)
    nb = len(taus)
    mix = []
    for _ in range(n1):
        a = rng.normal(size=(nb, nb)) + 1j * rng.normal(size=(nb, nb))
        q, _ = np.linalg.qr(a)
        mix.append(q)
    base = np.diag(np.exp(-2j * np.pi * np.asarray(taus) / n1))
    links = [mix[i].conj().T @ base @ mix[(i + 1) % n1] for i in range(n1)]
    return np.stack(links), mix


def test_the_transport_gauge_finds_the_centres_and_is_periodic():
    taus = [0.1, -0.3, 0.45]
    n1 = 7
    links, mix = _orbital_links(taus, n1)
    gauge = transport_gauge(links)
    assert np.allclose(np.sort(gauge.centres), np.sort(taus), atol=1e-12)
    # every link in the new gauge, the closing one included, is diag(e^{i phi/N1}) times a
    # Hermitian positive matrix, and for this model the positive part is the identity
    r = gauge.rotation
    for i in range(n1):
        new = r[i].conj().T @ links[i] @ r[(i + 1) % n1]
        assert np.allclose(new, np.diag(np.exp(1j * gauge.phases / n1)), atol=1e-12)
    assert gauge.singular == pytest.approx(1.0, abs=1e-12)


def test_the_other_sign_of_the_phase_is_not_periodic():
    """The design's exp(-i phi i/N1) leaves a jump of 2 phi on the closing link."""
    taus = [0.2]
    n1 = 6
    links, _ = _orbital_links(taus, n1, seed=3)
    gauge = transport_gauge(links)
    wrong = np.stack([gauge.rotation[i] * np.exp(-2j * gauge.phases * i / n1)
                      for i in range(n1)])
    closing = (wrong[-1].conj().T @ links[-1] @ wrong[0])[0, 0]
    inner = (wrong[0].conj().T @ links[0] @ wrong[1])[0, 0]
    jump = np.angle(closing / inner)
    assert abs(abs(jump) - abs(2 * gauge.phases[0])) < 1e-10
    assert abs(jump) > 1.0


def test_the_outside_weight_and_the_half_width():
    xi = np.linspace(-4.5, 4.5, 91)
    profile = np.exp(-xi ** 2 / 0.05)[None, :]
    widths = np.arange(4)
    weights = outside_weight(profile, xi, widths)
    assert weights[0, 0] > 1e-6
    assert half_width(weights, widths, 1e-8) == 1
    assert half_width(weights, widths, 1e-300) is None


def test_assemble_is_the_bloch_sum_of_a_banded_chain():
    """One band, q = 3, nearest neighbour t: H(kappa1) is the 3-site ring with phase on the wrap."""
    q, w, t = 3, 1, -0.7
    B = np.zeros((q, 2 * w + 1, 1, 1), complex)
    S = np.zeros_like(B)
    B[:, 0], B[:, 2] = t, t
    S[:, 1] = 1.0
    for kappa1 in (0.0, 0.25, 0.5):
        h, s = assemble(kappa1, B, S, w, q)
        assert hermiticity(h) < 1e-15
        eps, rank, d_min, d_all = solve(h, s)
        k = 2 * np.pi * (kappa1 + np.arange(q)) / q
        assert rank == q and d_min == pytest.approx(1.0)
        assert np.allclose(np.sort(eps), np.sort(2 * t * np.cos(k)), atol=1e-14)


def test_assemble_folds_a_wide_band_onto_a_small_magnetic_cell():
    """q = 1 with w = 2: every Delta lands on the same block, each with its own phase."""
    B = np.zeros((1, 5, 1, 1), complex)
    S = np.zeros_like(B)
    B[0, :, 0, 0] = [0.1, -0.5, 1.0, -0.5, 0.1]
    S[0, 2] = 1.0
    h, s = assemble(0.3, B, S, 2, 1)
    k = 2 * np.pi * 0.3
    assert h[0, 0] == pytest.approx(1.0 - 1.0 * np.cos(k) + 0.2 * np.cos(2 * k))


def test_the_hermiticity_guard_fires():
    h = np.array([[1.0, 0.5], [0.5 + 1e-6, 2.0]])
    assert hermiticity_guard(np.array([[1.0, 0.5], [0.5, 2.0]])) == 0.0
    with pytest.raises(ValueError, match="not Hermitian"):
        hermiticity_guard(h, tol=1e-8)


def test_the_rank_is_recorded_when_a_direction_is_dropped():
    s = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]])
    h = np.diag([1.0, 2.0])
    eps, rank, d_min, d_all = solve(h, s, drop=1e-8)
    assert rank == 1 and d_all < 1e-8 and d_min == pytest.approx(1.0)
