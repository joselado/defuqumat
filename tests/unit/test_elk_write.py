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

import numpy as np
import pytest

from defuqumat.io.elk import ElkState
from defuqumat.io.elk_write import HANDOFF_INPUT, write_elk_state

ELK = Path(__file__).resolve().parents[1] / "data" / "elk"
RUNS = sorted(p.name for p in ELK.iterdir() if (p / "STATE.OUT").exists())


@pytest.mark.parametrize("name", RUNS)
def test_a_state_read_and_written_is_elk_own_file(name, tmp_path):
    state = ElkState.read(ELK / name, allow_noncollinear=True)
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


@pytest.mark.parametrize("change, words", [
    ({"xcgrad": 3}, "meta-GGA"),
    ({"ftmtype": 1}, "tensor moment"),
])
def test_records_this_state_does_not_hold_are_refused(change, words, tmp_path):
    import dataclasses

    state = dataclasses.replace(ElkState.read(ELK / "h_sc"), **change)
    with pytest.raises(NotImplementedError, match=words):
        write_elk_state(tmp_path / "STATE.OUT", state)


@pytest.mark.parametrize("version, records", [
    ((10, 4, 0), ("xcgrad", "efermi")),
    ((9, 2, 0), ("xcgrad",)),
    ((5, 0, 0), ()),
])
def test_a_state_of_an_older_elk_is_written_with_its_own_records(version, records, tmp_path):
    """``readstate.f90`` reads ``xcgrad`` from 5.1, ``efermi`` from 9.6 and ``dlefe`` from 10.7.

    A state of such a version read and written back must carry only the records
    that version reads (AUDIT-2026-11-20 #16). Before, ``dlefe`` was always
    written, so the 10.4 file was read back with the ``dlefe`` real where the
    density pair is, and failed.
    """
    import dataclasses
    import math
    import shutil

    from scipy.io import FortranFile

    state = dataclasses.replace(ElkState.read(ELK / "h_sc"), version=version)
    write_elk_state(tmp_path / "STATE.OUT", state)
    shutil.copy(ELK / "h_sc" / "GEOMETRY.OUT", tmp_path)
    back = ElkState.read(tmp_path)
    assert back.version == version
    np.testing.assert_array_equal(back.rhomt, state.rhomt)
    np.testing.assert_array_equal(back.vsir, state.vsir)
    assert math.isnan(back.dlefe)
    assert math.isnan(back.efermi) == ("efermi" not in records)
    # The header is the version, spinpol, nspecies, lmmaxo, nrmtmax, nrcmtmax,
    # five records per species, ngridg, ngvec, ndmag, nspinor, fsmtype, ftmtype,
    # dftu and lmmaxdm, then the gated ones, then the four pairs.
    with FortranFile(tmp_path / "STATE.OUT", "r") as f:
        count = 0
        while True:
            try:
                f.read_record(np.uint8)
            except (OSError, ValueError, TypeError):
                break
            count += 1
    assert count == 6 + 5 + 8 + len(records) + 4
    # The same state written twice is the same file: the writer is the reader's inverse.
    again = write_elk_state(tmp_path / "STATE2.OUT", back)
    assert again.read_bytes() == (tmp_path / "STATE.OUT").read_bytes()
