"""Continuing a converged ``pw.x`` run from its save directory.

The fixtures under ``tests/data/qe/save/`` are the ``charge-density.dat``,
``paw.txt`` and ``data-file-schema.xml`` of three ``pw.x`` 7.5 SCFs at
``conv_thr = 1e-10``, with the ``pw.in`` that made them: Si2 norm-conserving
under ``K_POINTS gamma`` (the half-sphere file), Si2 PAW on a 2x2x2 mesh, and
an O2 PAW triplet at ``nspin = 2`` (the ``(n, m_z)`` layout and the
``(up, down)`` ``becsum``).

What makes the check discriminating is that both codes hold the same pseudo
valence density: a wrong grid map, spin split, ``becsum`` packing or
projector order would leave the seeded run iterating, where the right one
converges in one or two iterations at the ``pw.x`` energy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.io.qe_density import QESave

SAVE = Path(__file__).resolve().parents[1] / "data" / "qe" / "save"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: ``!    total energy`` of each ``pw.x`` run, Ry.
QE_ENERGY = {
    "si2-nc-gamma": -14.56060838,
    "si2-paw": -89.09136627,
    "o2-paw-lsda": -80.72096086,
}


def _calculator(name: str, text: str | None = None) -> Calculator:
    text = (SAVE / name / "pw.in").read_text() if text is None else text
    return Calculator.from_text(text, pseudo_dir=PSEUDO)


@pytest.mark.parametrize("name, counts", [
    ("si2-nc-gamma", [8.0]),
    ("si2-paw", [8.0]),
    ("o2-paw-lsda", [12.0, 2.0]),
])
def test_g0_is_the_electron_count_and_the_moment(name, counts):
    """``rho(G = 0) omega`` is N, and for LSDA the second component is M, not N_down."""
    save = QESave.read(SAVE / name)
    np.testing.assert_allclose(save.rho_g[:, 0].real * save.omega, counts, atol=1e-8)
    assert save.gamma_only == (name == "si2-nc-gamma")


def test_lsda_density_is_split_into_up_and_down():
    calc = _calculator("o2-paw-lsda")
    seed = calc.get_qe_seed(SAVE / "o2-paw-lsda")
    rho = np.asarray(seed["starting_density"])
    omega = QESave.read(SAVE / "o2-paw-lsda").omega
    channels = rho.reshape(2, -1).sum(axis=1) * omega / rho[0].size
    np.testing.assert_allclose(channels, [7.0, 5.0], atol=1e-8)
    becsum = [np.asarray(b) for b in seed["starting_becsum"] if b is not None]
    assert becsum[0].shape == (2, 2, 8, 8)
    np.testing.assert_array_equal(becsum[0], np.swapaxes(becsum[0], -1, -2))


@pytest.mark.parametrize("name", ["si2-nc-gamma", "si2-paw"])
def test_seeded_run_converges_at_once_to_the_pwx_energy(name):
    calc = _calculator(name)
    result = calc.get_scf(**calc.get_qe_seed(SAVE / name))
    assert result.converged
    # The first iteration lands at dr2 = 2.5e-11 (NC) and 4.0e-11 (PAW) against
    # conv_thr = 1e-10, a factor of 3 under it, so a second is allowed; a wrong
    # map took 9 against 7 from scratch.
    assert result.iterations <= 2
    assert abs(result.total_energy - QE_ENERGY[name]) < 1e-7


def test_paw_becsum_is_the_converged_one():
    """``paw.txt``'s packing (upper triangle, doubled off-diagonals) against our own."""
    calc = _calculator("si2-paw")
    seed = calc.get_qe_seed(SAVE / "si2-paw")
    own = calc.get_scf()
    for loaded, converged in zip(seed["starting_becsum"], own.becsum):
        if loaded is None:
            continue
        loaded, converged = np.asarray(loaded), np.asarray(converged)
        off = ~np.eye(loaded.shape[-1], dtype=bool)
        assert np.abs(converged[..., off]).max() > 1e-2  # the off-diagonals are tested
        np.testing.assert_allclose(loaded, converged, atol=2e-5)


@pytest.mark.parametrize("edit, error, words", [
    (("ecutrho=160.0", "ecutrho=120.0"), NotImplementedError, "ecutrho"),
    (("celldm(1)=10.20", "celldm(1)=10.30"), ValueError, "same cell"),
    (("ecutrho=160.0",
      "ecutrho=160.0, nspin=2, tot_magnetization=0, starting_magnetization(1)=0.2"),
     ValueError, "components"),
])
def test_a_save_directory_of_another_calculation_is_refused(edit, error, words):
    text = (SAVE / "si2-paw" / "pw.in").read_text()
    assert edit[0] in text
    calc = _calculator("si2-paw", text.replace(*edit))
    with pytest.raises(error, match=words):
        calc.get_qe_seed(SAVE / "si2-paw")


@pytest.mark.slow
def test_lsda_paw_seeded_run_converges_at_once():
    calc = _calculator("o2-paw-lsda")
    result = calc.get_scf(**calc.get_qe_seed(SAVE / "o2-paw-lsda"))
    assert result.converged
    assert result.iterations <= 2
    assert abs(result.total_energy - QE_ENERGY["o2-paw-lsda"]) < 1e-7
