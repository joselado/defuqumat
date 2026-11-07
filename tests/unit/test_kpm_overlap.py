"""The folded ``S^(-1/2) H S^(-1/2)`` of ultrasoft and PAW runs (P174).

The folded apply (four projector products) is checked against the unfolded one
(``S^(-1/2)`` on each side of ``Hamiltonian.apply``), which shares none of the
fold's small matrices; the independent check of both against ``eigh(H, S)`` is
``test_kpm_augmented.py``. Si2 at 12 Ry, ``ecutrho`` four times that, on a 2x2x2
mesh, and the same cell at ``nspin = 2`` with ``tot_magnetization = 2`` (0.2 eV smearing).
"""

from __future__ import annotations

import contextlib
import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import defuqumat  # noqa: F401
from defuqumat import Calculator
from defuqumat.kpm.overlap import apply_power, overlap_powers, raised_vectors, symmetrized
from defuqumat.workflows.nscf import fixed_density_hamiltonians

pytestmark = pytest.mark.unit

SETUP_KERNELS = ("_overlap_setup", "_fold", "_raised")
DATASETS = {"ultrasoft": "Si.pz-n-rrkjus_psl.0.1.UPF", "paw": "Si.pz-n-kjpaw_psl.0.1.UPF"}


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Free the compiled programs after each test (``.claude/rules/memory-design.md``)."""
    yield
    jax.clear_caches()


@functools.lru_cache(maxsize=2)
def _silicon(kind: str, magnetized: bool = False):
    from pathlib import Path

    pseudo = Path(__file__).parents[1] / "data" / "pseudo"
    # a smearing, so that the moment of 2 does not cut the valence-band multiplet
    extra = (dict(smearing=0.2, qe={"system": {"nspin": 2, "tot_magnetization": 2}})
             if magnetized else {})
    calc = Calculator.from_keywords(
        pseudo_dir=pseudo, announce=False, conv_thr=1e-10, lattice="fcc", a=5.398,
        pseudos={"Si": DATASETS[kind]},
        atoms=[("Si", (0.0, 0.0, 0.0)), ("Si", (0.25, 0.25, 0.25))],
        ecut=163.3, ecutrho=4 * 163.3, kmesh=(2, 2, 2), **extra)
    result = calc.get_scf()
    hamiltonians = fixed_density_hamiltonians(calc.calculation, result.density,
                                              becsum=result.becsum)
    return calc, result, hamiltonians


def _random_block(mask, rows, seed):
    rng = np.random.default_rng(seed)
    shape = (rows, mask.size)
    return jnp.asarray(np.where(mask, rng.normal(size=shape) + 1j * rng.normal(size=shape), 0.0))


def _assert_fold_is_the_unfolded_apply(wrapped):
    mask = np.asarray(wrapped.mask)
    for ik in range(mask.shape[0]):
        v = _random_block(mask[ik], 4, ik)
        unfolded = np.asarray(wrapped.apply_unfolded(v, ik))
        folded = np.asarray(wrapped.apply(v, ik))
        np.testing.assert_allclose(folded, unfolded, rtol=0,
                                   atol=1e-12 * np.abs(unfolded).max())


@pytest.mark.parametrize("kind", ["ultrasoft", "paw"])
def test_the_folded_apply_is_the_unfolded_one(kind):
    """Four products against six, on 4 complex random vectors at every k-point, to 1e-12."""
    _, _, hamiltonians = _silicon(kind)
    wrapped = symmetrized(hamiltonians)[0]
    assert wrapped.is_folded
    _assert_fold_is_the_unfolded_apply(wrapped)
    assert not symmetrized(hamiltonians, fold=False)[0].is_folded


@pytest.mark.parametrize("kind", ["ultrasoft", "paw"])
def test_raised_vectors_reuse_the_stored_factor(kind):
    """``S^(1/2) v`` from the wrapped ``U`` and ``(1 + W)^-1 - 1`` against a fresh QR at ``p = 1/2``."""
    _, _, hamiltonians = _silicon(kind)
    wrapped = symmetrized(hamiltonians)[0]
    mask = np.asarray(wrapped.mask)
    vectors = jnp.stack([_random_block(mask[ik], 3, 10 + ik) for ik in range(mask.shape[0])])
    raised = np.asarray(raised_vectors(wrapped, vectors))
    basis, shifted = overlap_powers(hamiltonians[0], 0.5)
    for ik in range(mask.shape[0]):
        reference = np.asarray(apply_power(basis[ik], shifted[ik], vectors[ik], mask[ik]))
        np.testing.assert_allclose(raised[ik], reference, rtol=0,
                                   atol=1e-12 * np.abs(reference).max())


@contextlib.contextmanager
def _setup_compiles():
    """The compilations of the three setup kernels inside the block, read off the ``jax`` logger."""
    compiled = []
    handler = logging.Handler()
    handler.emit = lambda record: compiled.append(record.getMessage())
    logger = logging.getLogger("jax")
    logger.addHandler(handler)
    previous = logger.level
    logger.setLevel(logging.DEBUG)
    jax.config.update("jax_log_compiles", True)
    names = []
    try:
        yield names
    finally:
        jax.config.update("jax_log_compiles", False)
        logger.removeHandler(handler)
        logger.setLevel(previous)
        names.extend(name for m in compiled for name in SETUP_KERNELS
                     if name in m and ("Compiling" in m or "Finished XLA compilation" in m))


def test_a_second_setup_compiles_nothing():
    """The QR, the fold and the raise are module-level jits: a new potential reuses them.

    The first call is counted too, so that a logger that saw nothing cannot pass.
    """
    import equinox as eqx

    _, _, hamiltonians = _silicon("paw")
    mask = np.asarray(hamiltonians[0].mask)
    vectors = jnp.stack([_random_block(mask[ik], 3, ik) for ik in range(mask.shape[0])])
    with _setup_compiles() as first:
        raised_vectors(symmetrized(hamiltonians)[0], vectors)
    assert set(first) == set(SETUP_KERNELS), first
    shifted = tuple(eqx.tree_at(lambda h: h.potential, h, h.potential + 0.01)
                    for h in hamiltonians)
    with _setup_compiles() as second:
        raised_vectors(symmetrized(shifted)[0], vectors + 1.0)
    assert not second, second


def test_the_two_spin_channels_share_one_overlap_factor():
    """``S`` does not depend on spin: one ``U`` for both channels, and each channel's own fold."""
    _, _, hamiltonians = _silicon("ultrasoft", magnetized=True)
    assert len(hamiltonians) == 2
    wrapped = symmetrized(hamiltonians)
    assert wrapped[0].basis is wrapped[1].basis
    assert wrapped[0].shifted is wrapped[1].shifted
    # the channels differ (a moment of 2), so their folds must too
    assert float(jnp.max(jnp.abs(wrapped[0].folded - wrapped[1].folded))) > 1e-4
    for channel in wrapped:
        _assert_fold_is_the_unfolded_apply(channel)
