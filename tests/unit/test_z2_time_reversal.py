"""The Z2 invariant needs time reversal: a magnet and an off-TRIM plane are refused.

Both Z2 routes assume a time-reversal-invariant Hamiltonian on a
time-reversal-invariant plane. The Wilson route counts crossings of Wannier
centres that are Kramers pairs at the two ends of the pump, and the parity route
multiplies the parities of Kramers pairs. A magnetic spinor run (``domag``), a
run under a field or a constrained moment, or a plane at ``offset = 0.3`` gave
an integer with no warning (audit 2026-11-20c #4).

None of these needs an SCF. The DFT entry points are tested with a sentinel in
place of the state source, so a run the guard lets through stops at the
sentinel, and the plane is tested on a model whose Z2 is known.
"""

from pathlib import Path

import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo.upf import read_upf
from defuqumat.system.builder import build_system
from defuqumat.topology import ModelSource, z2_invariant
from defuqumat.workflows import topology
from tests.models import doubled_qwz

pytestmark = pytest.mark.unit

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


class _Reached(Exception):
    """The state source was asked for, so the guard let the run through."""


def _spinor_silicon(extra: str = ""):
    """Silicon as a two-component spinor run, with ``extra`` in ``&system``."""
    text = (CASES / "si2-nosym.in").read_text().replace(
        "ecutwfc = 12.0, nosym = .true.",
        "ecutwfc = 12.0, nosym = .true., noncolin = .true.,\n    " + extra)
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos


def _sentinel(*args, **kwargs):
    raise _Reached


def _placeholder(system):
    return np.zeros((system.nspin_mag, 4, 4, 4))


@pytest.mark.parametrize("extra, reason", [
    ("starting_magnetization(1) = 0.3,", "carries a magnetization"),
    ("b_field(3) = 0.01,", "carries a magnetization"),
    ("constrained_magnetization = 'total', fixed_magnetization(3) = 0.5,",
     "magnetic field or a constrained moment"),
])
@pytest.mark.parametrize("entry", ["run_z2", "run_z2_3d"])
def test_a_run_that_breaks_time_reversal_is_refused(monkeypatch, extra, reason, entry):
    """A magnet, a field, and a constraint on a run that has no moment of its own."""
    system, pseudos = _spinor_silicon(extra)
    assert system.domag or system.has_applied_field
    monkeypatch.setattr(topology, "_source", _sentinel)
    with pytest.raises(ValueError, match="time-reversal symmetry") as raised:
        getattr(topology, entry)(system, pseudos, _placeholder(system), nocc=8)
    assert reason in str(raised.value)
    assert "get_chern" in str(raised.value)


@pytest.mark.parametrize("extra", ["", "starting_magnetization(1) = 0.0,"])
@pytest.mark.parametrize("entry", ["run_z2", "run_z2_3d"])
def test_a_nonmagnetic_spinor_run_passes_the_guard(monkeypatch, extra, entry):
    """The regime next to the refusal: zero starting moment, no field."""
    system, pseudos = _spinor_silicon(extra)
    assert not system.domag and not system.has_applied_field
    monkeypatch.setattr(topology, "_source", _sentinel)
    with pytest.raises(_Reached):
        getattr(topology, entry)(system, pseudos, _placeholder(system), nocc=8)


def test_an_off_trim_plane_is_refused_on_the_dft_entry(monkeypatch):
    """``get_z2(offset=0.3)``: the plane is not mapped onto itself by time reversal.

    The source is a model with ``nu = 1`` on every plane, so the integer the
    old code returned for ``offset = 0.3`` would have looked like an answer.
    """
    system, pseudos = _spinor_silicon()
    model = ModelSource(hamiltonian=doubled_qwz(1.0), nocc=2)
    monkeypatch.setattr(topology, "_source", lambda *a, **k: model)
    with pytest.raises(ValueError, match="not a time-reversal-invariant plane"):
        topology.run_z2(system, pseudos, _placeholder(system), offset=0.3,
                        nloop=21, npump=11)
    flow = topology.run_z2(system, pseudos, _placeholder(system), offset=0.5,
                           nloop=21, npump=11)
    assert flow.z2 == 1


@pytest.mark.parametrize("offset", [0.3, 0.25, 0.5 + 1e-6, -0.1])
def test_the_wilson_route_refuses_a_plane_that_is_not_time_reversal_invariant(offset):
    source = ModelSource(hamiltonian=doubled_qwz(1.0), nocc=2)
    with pytest.raises(ValueError, match="not a time-reversal-invariant plane"):
        z2_invariant(source, offset=offset, nloop=21, npump=11)


@pytest.mark.parametrize("offset", [0.0, 0.5, 1.0, -0.5, 1.5])
def test_the_wilson_route_still_runs_on_both_time_reversal_invariant_planes(offset):
    """0 and 1/2 modulo 1; the model does not depend on the third axis, so nu = 1."""
    source = ModelSource(hamiltonian=doubled_qwz(1.0), nocc=2)
    assert z2_invariant(source, offset=offset, nloop=21, npump=11).z2 == 1
