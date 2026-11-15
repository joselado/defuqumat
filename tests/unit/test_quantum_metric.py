"""The quantum metric: the assembly against a closed form, and the guards.

The metric of a two-band ``h(k) = d(k) . sigma`` model is ``g_ab = (1/4) d_a n .
d_b n`` with ``n = d/|d|`` (Provost and Vallee), a closed form that shares
nothing with a sum over empty states. :func:`_metric_sum` is checked against it
on the plain model and on the same model written as a generalised eigenproblem
``H = A^dag H0 A``, ``S = A^dag A``, where the states a metric is about are
``A c`` and the velocity needs the connection ``K = c^dag A^dag dA c``. Without
``K`` the generalised model is off by an order-one fraction of the metric, so
that half cannot pass by accident.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg as sla

import defuqumat  # noqa: F401
from defuqumat.response.geometry import (
    QuantumMetric, _metric_sum, require_an_insulator, wannier_count,
)
from tests.models import nonorthogonal, qwz

pytestmark = pytest.mark.unit

MASS = 1.3
TWOPI = 2.0 * np.pi


def _unit_d(k):
    k1, k2 = TWOPI * k[0], TWOPI * k[1]
    d = jnp.stack([jnp.sin(k1), jnp.sin(k2), MASS - jnp.cos(k1) - jnp.cos(k2)])
    return d / jnp.linalg.norm(d)


def _closed_form(k):
    jacobian = np.asarray(jax.jacfwd(_unit_d)(jnp.asarray(k)))   # (3 components, 3 directions)
    return 0.25 * jacobian.T @ jacobian


def _points(count=8, seed=5):
    points = np.random.default_rng(seed).random((count, 3))
    points[:, 2] = 0.0
    return points


def test_the_metric_of_a_two_band_model_is_its_closed_form():
    """``(1/4) d_a n . d_b n``, measured to 1e-14 on a scale of order one."""
    h = qwz(mass=MASS)
    worst, scale = 0.0, 0.0
    for k in _points():
        energies, vectors = np.linalg.eigh(np.asarray(h(jnp.asarray(k))))
        jac = np.asarray(jax.jacfwd(h)(jnp.asarray(k)))           # (2, 2, 3)
        element = np.stack([vectors.conj().T @ jac[:, :, a] @ vectors for a in range(3)])
        g, _, dropped = _metric_sum(jnp.asarray(element), jnp.asarray(energies),
                                    jnp.array([1.0, 0.0]), 1.0e-8)
        exact = _closed_form(k)
        worst = max(worst, float(np.max(np.abs(np.asarray(g) - exact))))
        scale = max(scale, float(np.max(np.abs(exact))))
        assert int(dropped) == 0
    assert scale > 0.1
    assert worst < 1.0e-12


def test_a_moving_overlap_needs_the_connection_and_has_it():
    """The generalised velocity ``<n|dH - e_m dS|m> + (e_m - e_n) K`` gives H0's metric.

    ``K_nm = <c_n|A^dag dA|c_m>`` is the augmentation connection's model
    counterpart. With it the metric is the closed form of ``H0``; without it the
    error is a large fraction of the metric, which is what says the check sees
    the term.
    """
    h0 = qwz(mass=MASS)
    h, s, a = nonorthogonal(h0, s=0.25)
    with_k, without_k, scale = 0.0, 0.0, 0.0
    for k in _points(6, seed=7):
        kj = jnp.asarray(k)
        matrix, metric, frame = (np.asarray(f(kj)) for f in (h, s, a))
        energies, vectors = sla.eigh(matrix, metric)
        gap = energies[None, :] - energies[:, None]               # [n, m] = e_m - e_n
        full, bare = [], []
        for d in range(3):
            tangent = jnp.asarray(np.eye(3)[d])
            dh, ds, da = (np.asarray(jax.jvp(f, (kj,), (tangent,))[1]) for f in (h, s, a))
            tangents = vectors.conj().T @ dh @ vectors - energies[None, :] * (vectors.conj().T @ ds @ vectors)
            connection = vectors.conj().T @ frame.conj().T @ da @ vectors
            full.append(tangents + gap * connection)
            bare.append(tangents)
        exact = _closed_form(k)
        fill = jnp.array([1.0, 0.0])
        g = np.asarray(_metric_sum(jnp.asarray(np.stack(full)), jnp.asarray(energies), fill, 1e-8)[0])
        g0 = np.asarray(_metric_sum(jnp.asarray(np.stack(bare)), jnp.asarray(energies), fill, 1e-8)[0])
        with_k = max(with_k, float(np.max(np.abs(g - exact))))
        without_k = max(without_k, float(np.max(np.abs(g0 - exact))))
        scale = max(scale, float(np.max(np.abs(exact))))
    assert with_k < 1.0e-12
    assert without_k > 0.05 * scale


def test_the_trimmed_sum_drops_exactly_the_top_band():
    """The truncation diagnostic is the sum without the highest band, nothing else."""
    rng = np.random.default_rng(2)
    nb = 5
    raw = rng.normal(size=(3, nb, nb)) + 1j * rng.normal(size=(3, nb, nb))
    element = raw + np.conj(np.swapaxes(raw, -1, -2))
    energies = np.array([-1.0, -0.5, 0.4, 0.9, 1.7])
    fill = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    full, trimmed, _ = _metric_sum(jnp.asarray(element), jnp.asarray(energies), jnp.asarray(fill), 1e-8)
    less, _, _ = _metric_sum(jnp.asarray(element[:, :-1, :-1]), jnp.asarray(energies[:-1]),
                             jnp.asarray(fill[:-1]), 1e-8)
    assert np.allclose(np.asarray(trimmed), np.asarray(less), atol=1e-13)
    assert not np.allclose(np.asarray(full), np.asarray(trimmed))
    # Symmetric and positive semi-definite, as a metric must be.
    g = np.asarray(full)
    assert np.allclose(g, g.T, atol=1e-13)
    assert np.min(np.linalg.eigvalsh(g)) > -1e-12


def test_a_metal_is_refused_and_a_smeared_insulator_is_not():
    """The guard reads the fillings, so it fires on a partial band whatever the scheme."""
    insulator = np.zeros((1, 4, 6))
    insulator[..., :3] = 1.0
    insulator[..., :3] -= 1.0e-14                       # what a smearing leaves
    require_an_insulator(insulator, insulator)
    metal = insulator.copy()
    metal[0, 2, 3] = 0.3
    with pytest.raises(ValueError, match="refused on a metal"):
        require_an_insulator(metal, metal)


def test_the_spread_counts_spatial_functions_at_nspin_1_and_spin_orbitals_otherwise():
    assert wannier_count(8.0, 1) == 4.0
    assert wannier_count(8.0, 2) == 8.0
    assert wannier_count(8.0, 4) == 8.0
    tensor = np.diag([1.0, 2.0, 3.0])
    result = QuantumMetric(metric=np.zeros((1, 3, 3)), metric_by_spin=np.zeros((1, 1, 3, 3)),
                           kpoints=np.zeros((1, 3)), localization_tensor=tensor,
                           nelec=8.0, nwann=4.0)
    assert result.spread_invariant == pytest.approx(24.0)
    assert np.allclose(result.localization_length, np.sqrt([1.0, 2.0, 3.0]))


def _silicon_metric(pseudo, ecut, kmesh, **system):
    from defuqumat import Calculator
    from defuqumat.units import RY_TO_EV

    calc = Calculator.from_keywords(
        lattice="fcc", a=10.2 * 0.529177210903,
        atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
        pseudos={"Si": pseudo}, pseudo_dir="tests/data/pseudo", ecut=ecut * RY_TO_EV,
        kmesh=kmesh, qe={"system": {"nosym": True, "noinv": True, **system}})
    calc.get_scf(conv_thr=1e-10)
    return calc.get_quantum_metric(nbnd=8)


def _check_metric(metric, trace):
    """Symmetric, positive, and isotropic as a cubic crystal on a cubic mesh must be.

    The run uses no symmetry (``nosym``, ``noinv``), so the isotropy of the
    localization tensor is not something the code imposed. The trace is pinned
    (2026-11-15), so a change in the assembly, the fillings or the connection shows.
    """
    g = np.asarray(metric.metric)
    assert np.abs(g - np.swapaxes(g, 1, 2)).max() < 1e-12
    assert np.linalg.eigvalsh(0.5 * (g + np.swapaxes(g, 1, 2))).min() > 0.0
    tensor = np.asarray(metric.localization_tensor)
    mean = np.trace(tensor) / 3.0
    assert np.abs(tensor - mean * np.eye(3)).max() < 1e-6 * mean
    assert np.trace(tensor) == pytest.approx(trace, abs=1e-5)


# The audit of 2026-11-15 (``tests`` hole on P161) found no test of the plane-wave
# assembly in the gate. These are the cheapest: 6.5 s and 8.5 s on this workstation
# with a warm kernel cache, above the gate's five seconds for a check that is not a
# number against a reference code, so they are slow. Written from a probe with the
# same calls and parameters (2026-11-15); the tests themselves were not run.
@pytest.mark.slow
def test_the_plane_wave_metric_of_silicon_is_isotropic():
    """Norm-conserving silicon, 12 Ry, the whole 2x2x2 grid: tensor isotropic to 1.6e-8."""
    _check_metric(_silicon_metric("Si.pz-vbc.UPF", 12.0, (2, 2, 2)), 25.827379755580736)


@pytest.mark.slow
def test_the_ultrasoft_metric_runs_its_connection_stage():
    """Ultrasoft silicon at Gamma, 20/160 Ry: the augmented stage list, with the connection."""
    _check_metric(_silicon_metric("Si.pz-n-rrkjus_psl.0.1.UPF", 20.0, (1, 1, 1), ecutrho=160.0),
                  75.65657570777523)
