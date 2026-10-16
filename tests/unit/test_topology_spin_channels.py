"""A collinear ``nspin = 2`` source: which channel, and what is still refused.

A collinear spin-polarized run has two Hamiltonians, one per channel, and so two
occupied manifolds and two sets of invariants. :class:`DFTSource` takes the
channel as ``spin`` and refuses to guess it. None of these needs an SCF: every
refusal here fires at construction or before the first diagonalisation, so the
density handed in is a placeholder of the right shape.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo.upf import read_upf
from defuqumat.system.builder import build_system
from defuqumat.workflows.topology import (
    DFTSource,
    _occupied_bands,
    run_berry_curvature,
    run_z2,
)

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _system(name):
    system = build_system(read_pw_input(CASES / name))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file)
                    for s in system.structure.species)
    return system, pseudos


def _placeholder(system, up=None, down=None):
    """A ``(nspin, n1, n2, n3)`` density holding ``up`` and ``down`` electrons."""
    shape = (4, 4, 4)
    volume = float(system.cell.volume)
    if system.nspin == 1:
        return jnp.ones((1,) + shape)
    return jnp.stack([jnp.full(shape, up / volume), jnp.full(shape, down / volume)])


def test_a_polarized_source_without_a_channel_is_refused():
    """The two manifolds are two answers; which one is meant must be said."""
    system, pseudos = _system("o2-fixed-lsda.in")
    assert system.nspin == 2
    with pytest.raises(NotImplementedError, match="spin = 0 or spin = 1"):
        DFTSource(system=system, pseudos=pseudos,
                  density=_placeholder(system, 7, 5), nocc=7)


def test_a_channel_outside_a_polarized_run_or_out_of_range_is_refused():
    system, pseudos = _system("o2-fixed-lsda.in")
    with pytest.raises(ValueError, match="spin = 2"):
        DFTSource(system=system, pseudos=pseudos,
                  density=_placeholder(system, 7, 5), nocc=7, spin=2)
    unpolarized, upf = _system("si2-nc-pbe.in")
    with pytest.raises(ValueError, match="nspin = 2"):
        DFTSource(system=unpolarized, pseudos=upf,
                  density=_placeholder(unpolarized), nocc=4, spin=1)


def test_the_kubo_route_is_refused_per_channel():
    """``VelocityOperator`` builds every channel's ``dH/dk`` and the Kubo sum
    reads channel 0 of it, so at ``nspin = 2`` it would differentiate the up
    Hamiltonian whatever ``spin`` says."""
    system, pseudos = _system("o2-fixed-lsda.in")
    source = DFTSource(system=system, pseudos=pseudos,
                       density=_placeholder(system, 7, 5), nocc=5, spin=1)
    with pytest.raises(NotImplementedError, match="kubo"):
        source.states(np.zeros((1, 3)), keep_velocity=True)


def test_the_channel_count_comes_from_the_density():
    """``nelup`` and ``neldw`` are what the density integrates to per channel;
    a channel holding a fractional count has no gapped manifold to name."""
    system, pseudos = _system("o2-fixed-lsda.in")
    density = _placeholder(system, 7, 5)
    assert _occupied_bands(system, pseudos, density, 0) == 7
    assert _occupied_bands(system, pseudos, density, 1) == 5
    with pytest.raises(ValueError, match="explicitly"):
        _occupied_bands(system, pseudos, _placeholder(system, 6.6, 5.4), 0)


def test_the_z2_refusals_still_fire_on_a_polarized_run():
    """A collinear channel has no spin-orbit coupling, so ``_require_spinors``
    refuses the Z2 whatever ``spin`` says, before any source is built."""
    system, pseudos = _system("o2-fixed-lsda.in")
    with pytest.raises(ValueError, match="spin-orbit"):
        run_z2(system, pseudos, _placeholder(system, 7, 5), nocc=7)


def test_run_berry_curvature_forwards_the_channel():
    """Without ``spin`` the guard fires through the workflow, which is what a
    ``Calculator.get_berry_curvature()`` call on a polarized run meets."""
    system, pseudos = _system("o2-fixed-lsda.in")
    with pytest.raises(NotImplementedError, match="spin = 0 or spin = 1"):
        run_berry_curvature(system, pseudos, _placeholder(system, 7, 5), nocc=7)
