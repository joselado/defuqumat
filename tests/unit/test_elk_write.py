"""Writing an Elk ``STATE.OUT``: the inverse of the reader, byte for byte.

Each committed Elk run directory is read into an :class:`ElkState` and written
back; the file must be Elk's own, including the uninitialised padding rows past
``nrmt`` of the shorter species (``sic_zb``), the magnetic records of a
spin-polarized state (``h_sc_spin``) and the DFT+U potential matrix
(``h_sc_dftu``). The potentials are halved on the way out (Rydberg to Hartree),
which is exact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from defuqumat.io.elk import ElkState
from defuqumat.io.elk_write import HANDOFF_INPUT, write_elk_state

ELK = Path(__file__).resolve().parents[1] / "data" / "elk"
RUNS = sorted(p.name for p in ELK.iterdir() if (p / "STATE.OUT").exists())


@pytest.mark.parametrize("name", RUNS)
def test_a_state_read_and_written_is_elk_own_file(name, tmp_path):
    state = ElkState.read(ELK / name)
    path = write_elk_state(tmp_path / "STATE.OUT", state)
    assert path.read_bytes() == (ELK / name / "STATE.OUT").read_bytes()


def test_the_handoff_input_keeps_the_density_and_writes_the_potential():
    """``beta0 = 0`` keeps the density, and ``nwrite = 1`` is what writes it.

    Without ``nwrite`` the one-loop run kept the density and wrote nothing,
    since ``gndstate.f90`` writes after the loop only when ``maxscl > 1``.
    """
    words = HANDOFF_INPUT.split()
    for key, value in (("tasks", "1"), ("mixrho", ".true."), ("mixtype", "0"),
                       ("beta0", "0.0"), ("maxscl", "1"), ("nwrite", "1")):
        assert words[words.index(key) + 1] == value


def test_a_fixed_spin_moment_state_is_refused(tmp_path):
    import dataclasses

    state = dataclasses.replace(ElkState.read(ELK / "h_sc"), fsmtype=1)
    with pytest.raises(NotImplementedError, match="fixed spin moment"):
        write_elk_state(tmp_path / "STATE.OUT", state)
