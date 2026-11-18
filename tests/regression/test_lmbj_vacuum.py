"""P182: the local mBJ on a sheet with vacuum, against the vacuum's size.

Graphane (CH, chair) on the HGH datasets at 60 Ry, 6x6x1, at cell heights of 20
and 30 bohr (about 15 and 25 bohr of vacuum). What the local form is for:

* the sheet's ``c`` does not depend on how much vacuum the cell has (measured
  1.2705 and 1.2701), where TB09's single ``c`` is the vacuum tail's (1.456);
* ``c`` goes to 1 in the vacuum (1.004 at 30 bohr; 1.079 at 20, where the
  vacuum is not much wider than the Gaussian's reach);
* the occupied bands move rigidly with the vacuum (2.8 meV);
* the vacuum state at ``Gamma`` moves by about PBE's confinement shift (0.11 eV
  against PBE's 0.21), which needs the meta potential's gate at 1e-9: at the
  old 1e-6 it moved 1.53 eV, the cut leaving a well in the vacuum.

Slow (six SCFs of a few minutes each, the 30 bohr local run 81 iterations), and
run only when asked for.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.units import RY_TO_EV

pytestmark = [pytest.mark.regression, pytest.mark.slow]

PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


def _text(dft: str, height: float) -> str:
    a = 2.54 * BOHR_PER_ANGSTROM
    zc, zh = 0.23 * BOHR_PER_ANGSTROM, 1.33 * BOHR_PER_ANGSTROM
    dft_line = "" if dft == "pbe" else f", input_dft='{dft}'"
    return f"""&control
 calculation='scf'
/
&system
 ibrav=4, celldm(1)={a:.6f}, celldm(3)={height / a:.6f}, nat=4, ntyp=2,
 ecutwfc=60.0, nbnd=8{dft_line}
/
&electrons
 conv_thr=1.0d-9, mixing_beta=0.4, electron_maxstep=200
/
ATOMIC_SPECIES
 C 12.011 C.pbe-hgh.UPF
 H 1.008 H.pbe-hgh.UPF
ATOMIC_POSITIONS bohr
 C 0.0 {a / np.sqrt(3):.6f} {zc:.6f}
 C {a / 2:.6f} {a / (2 * np.sqrt(3)):.6f} {-zc:.6f}
 H 0.0 {a / np.sqrt(3):.6f} {zh:.6f}
 H {a / 2:.6f} {a / (2 * np.sqrt(3)):.6f} {-zh:.6f}
K_POINTS automatic
 6 6 1 0 0 0
"""


@lru_cache(maxsize=None)
def _run(dft: str, height: float):
    calc = Calculator.from_text(_text(dft, height), pseudo_dir=PSEUDO, announce=False)
    result = calc.get_scf()
    assert result.converged
    return result


def _bands(result):
    return np.asarray(result.eigenvalues) * RY_TO_EV


def test_the_sheets_c_does_not_depend_on_the_vacuum_and_the_vacuums_is_one():
    short, tall = _run("lmbj", 20.0), _run("lmbj", 30.0)
    profile_short = np.asarray(short.meta_c_field).mean(axis=(0, 1))
    profile_tall = np.asarray(tall.meta_c_field).mean(axis=(0, 1))
    assert abs(profile_short[0] - profile_tall[0]) < 1e-3
    assert abs(profile_tall[len(profile_tall) // 2] - 1.0) < 1e-2
    # and it is not TB09's single c, which the vacuum tail sets
    assert _run("tb09", 30.0).meta_c - profile_tall[0] > 0.1


@pytest.mark.parametrize("dft, cbm_shift", [("pbe", 0.3), ("lmbj", 0.2)])
def test_the_bands_move_rigidly_and_the_vacuum_state_by_its_confinement(dft, cbm_shift):
    short, tall = _bands(_run(dft, 20.0)), _bands(_run(dft, 30.0))
    occupied = slice(0, 5)  # ten valence electrons
    shift = (tall[:, occupied] - short[:, occupied]).mean()
    assert np.ptp(tall[:, occupied] - short[:, occupied] - shift) < 5e-3
    assert abs(tall[0, 5] - short[0, 5] - shift) < cbm_shift
