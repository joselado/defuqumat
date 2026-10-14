"""The regularised tangent of the subspace ``eigh`` (P148).

JAX's ``eigh`` tangent carries ``1/(w_j - w_i)`` over every off-diagonal pair,
unprotected, so an exactly degenerate pair gives ``inf`` and a pair split by
round-off gives ``1/round-off``. Davidson's subspace solve meets both at every
step on a cell with a symmetry-forced multiplet, and the SCF map's autodiff
Jacobian action was 28 per cent off its central difference there while agreeing
to 1.5e-8 on a cell without one (`notes/DEGENERACY-NEXT.md` item 4). The kernel
``(w_j - w_i)/((w_j - w_i)^2 + eta^2)`` is what replaces it, and these are its
four properties: the primal is unchanged, the eigenvalue tangent is unchanged,
far from a degeneracy the eigenvector tangent is JAX's to ``(eta/Delta)^2``,
and at an exact degeneracy it is finite and the tangent of the block's
*projector* is still the derivative, which is the quantity a density reads.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.solvers.subspace import _eigh, _eigh_regularised

pytestmark = pytest.mark.unit


def _hermitian(rng, n):
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    return jnp.asarray(0.5 * (a + a.conj().T))


def _with_spectrum(rng, values):
    """A Hermitian matrix with exactly these eigenvalues, in a random basis."""
    q, _ = np.linalg.qr(rng.standard_normal((len(values), len(values)))
                        + 1j * rng.standard_normal((len(values), len(values))))
    return jnp.asarray(q @ np.diag(values) @ q.conj().T)


def test_the_primal_is_the_plain_solve():
    rng = np.random.default_rng(0)
    x = _hermitian(rng, 7)
    values, vectors = _eigh(x)
    reg_values, reg_vectors = _eigh_regularised(x, 1e-5)
    assert np.array_equal(np.asarray(values), np.asarray(reg_values))
    assert np.array_equal(np.asarray(vectors), np.asarray(reg_vectors))


def test_far_from_a_degeneracy_the_tangent_is_jaxs_to_eta_over_delta_squared():
    rng = np.random.default_rng(1)
    x = _with_spectrum(rng, np.array([0.0, 1.0, 2.5, 4.0, 7.0]))
    dx = _hermitian(rng, 5)
    eta = 1e-3
    (_, v), (dw, dv) = jax.jvp(lambda a: _eigh(a), (x,), (dx,))
    (_, v_reg), (dw_reg, dv_reg) = jax.jvp(lambda a: _eigh_regularised(a, eta), (x,), (dx,))
    assert np.array_equal(np.asarray(v), np.asarray(v_reg))
    assert np.allclose(np.asarray(dw), np.asarray(dw_reg), atol=1e-14), "the eigenvalue tangent moved"
    # the smallest gap is 1.0, so the kernel differs from 1/Delta by eta^2 of itself
    relative = np.linalg.norm(np.asarray(dv - dv_reg)) / np.linalg.norm(np.asarray(dv))
    assert relative < 3 * eta ** 2
    assert relative > 1e-8, "a null comparison: the kernel did nothing"


def test_at_an_exact_degeneracy_the_tangent_is_finite_and_the_projector_tangent_is_the_derivative():
    """JAX's rule gives ``inf`` or ``NaN`` here; the kernel gives the derivative of
    what a density reads, the projector onto the degenerate block."""
    rng = np.random.default_rng(2)
    x = _with_spectrum(rng, np.array([0.0, 1.0, 1.0, 1.0, 3.0, 5.0]))
    dx = _hermitian(rng, 6)
    eta = 1e-6

    def projector(a):
        _, v = _eigh_regularised(a, eta)
        block = v[:, 1:4]
        return block @ block.conj().T

    primal, tangent = jax.jvp(projector, (x,), (dx,))
    assert np.all(np.isfinite(np.asarray(tangent)))
    h = 1e-5
    central = (np.asarray(projector(x + h * dx)) - np.asarray(projector(x - h * dx))) / (2 * h)
    scale = np.linalg.norm(central)
    assert scale > 1e-3, "a null test: the perturbation does not move the block"
    assert np.linalg.norm(np.asarray(tangent) - central) < 1e-6 * scale

    # and the plain rule on the same matrix is the defect: ``inf`` when LAPACK
    # returns two members bit-equal, ``1/round-off`` when it splits them by ulps
    _, plain = jax.jvp(lambda a: _eigh(a)[1], (x,), (dx,))
    plain = np.asarray(plain)
    assert (not np.all(np.isfinite(plain))) or np.max(np.abs(plain)) > 1e9


def test_reverse_mode_transposes_the_rule():
    """The rule is linear in the tangent, so ``jax.grad`` through it exists, is
    finite at an exact degeneracy, and away from one equals JAX's own gradient
    to ``(eta/Delta)^2``; ``eta`` rides as a traced argument with no cotangent
    asked of it."""
    rng = np.random.default_rng(4)
    probe = _hermitian(rng, 5)

    def overlap_of_the_lowest_two(v):
        # a real scalar of the eigenvectors: the overlap of the lowest two with a probe
        block = v[:, :2]
        return jnp.real(jnp.trace(block.conj().T @ probe @ block))

    def energy_like(a, eta):
        return overlap_of_the_lowest_two(_eigh_regularised(a, eta)[1])

    def plain_energy_like(a):
        return overlap_of_the_lowest_two(_eigh(a)[1])

    x = _with_spectrum(rng, np.array([0.0, 1.0, 2.5, 4.0, 7.0]))
    eta = 1e-3
    g_reg = np.asarray(jax.grad(energy_like)(x, jnp.asarray(eta)))
    g_plain = np.asarray(jax.grad(plain_energy_like)(x))
    assert np.all(np.isfinite(g_reg))
    assert np.linalg.norm(g_reg - g_plain) < 3 * eta ** 2 * np.linalg.norm(g_plain)

    degenerate = _with_spectrum(rng, np.array([0.0, 1.0, 1.0, 1.0, 3.0]))
    g_deg = np.asarray(jax.grad(energy_like)(degenerate, jnp.asarray(1e-6)))
    assert np.all(np.isfinite(g_deg))


def test_a_pair_split_by_round_off_no_longer_amplifies():
    """A pair 1e-13 apart gives JAX's rule a mixing tangent of 1e13; the kernel
    bounds it by ``1/(2 eta)``."""
    rng = np.random.default_rng(3)
    x = _with_spectrum(rng, np.array([0.0, 1.0, 1.0 + 1e-13, 3.0]))
    dx = _hermitian(rng, 4)
    eta = 1e-5
    _, dv_plain = jax.jvp(lambda a: _eigh(a)[1], (x,), (dx,))
    _, dv_reg = jax.jvp(lambda a: _eigh_regularised(a, eta)[1], (x,), (dx,))
    assert np.max(np.abs(np.asarray(dv_plain))) > 1e9
    assert np.max(np.abs(np.asarray(dv_reg))) < 1.0 / eta
