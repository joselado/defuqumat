"""Wannier functions of silicon against ``pw2wannier90.x`` + ``wannier90.x``, three datasets.

Slow: one SCF and one fixed-density solve on the 4x4x4 mesh per dataset, and
an 8x8x8 mesh for the interpolation. Written from the numbers measured on
2026-11-04 (``PLAN.md``, P162), where each ran as a script; the tolerances are
about 3x those measured.

- norm-conserving, 12 bands to 8 sp3: our pipeline 16.053194854 A^2 against
  ``wannier90.x``'s 16.053195110 (4e-7, two SCFs);
- ultrasoft: ``wannier90.x`` on our matrices 18.358070728 against
  18.358070827 on ``pw2wannier90.x``'s; our engine 18.358070873;
- PAW: ``wannier90.x`` on ours 18.362772978 against 18.362772917; our engine
  18.362773138;
- the bands interpolated with ``use_ws_distance`` against ``wannier90.x``'s
  ``si_band.dat``: 9.2e-6 eV;
- the interpolation error of the four valence bands at 20 random k-points
  against a direct solve: 0.267, 0.049 and 0.020 eV on 4, 6 and 8 cubed.
"""

from pathlib import Path

import dataclasses

import numpy as np
import pytest

import defuqumat  # noqa: F401
from defuqumat.units import RY_TO_EV

pytestmark = [pytest.mark.slow, pytest.mark.regression]

DATA = Path(__file__).resolve().parents[1] / "data" / "wannier"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

CASES = {
    "si": (6.4, 16.053195110, 11.813496347),
    "si-us": (6.6, 18.358070827, 12.514071305),
    "si-paw": (6.6, 18.362772917, 12.515758031),
}


def _wannier(name, frozen, divisions=(4, 4, 4)):
    from defuqumat import Calculator

    calc = Calculator.from_file(DATA / name / "si.scf", pseudo_dir=PSEUDO)
    calc.get_scf()
    w = calc.get_wannier(8, ["Si:sp3"], num_bands=12, divisions=divisions,
                         dis_win=(None, 17.0), dis_froz=(None, frozen),
                         engine_options=dict(num_iter=200, conv_window=3, dis_num_iter=200))
    return calc, w


@pytest.mark.parametrize("name", sorted(CASES))
def test_silicon_is_wannier90s(name):
    frozen, total, invariant = CASES[name]
    _, w = _wannier(name, frozen)
    terms = w.spread_terms_angstrom2()
    assert w.converged
    assert abs(terms["omega_total"] - total) < 1.5e-6
    assert abs(terms["omega_i"] - invariant) < 1.5e-6


def test_the_interpolated_bands_are_wannier90s():
    _, w = _wannier("si", 6.4)
    k = np.loadtxt(DATA / "si" / "si_band.kpt", skiprows=1)[:, :3]
    ref = np.loadtxt(DATA / "si" / "si_band.dat").reshape(8, -1, 2)[:, :, 1].T
    assert np.max(np.abs(w.bands_ev(k) - ref)) < 3e-5


def test_the_interpolation_converges_with_the_mesh():
    from defuqumat.workflows.topology import _source

    points = np.random.default_rng(0).random((20, 3))
    errors = []
    for divisions in ((4, 4, 4), (8, 8, 8)):
        calc, w = _wannier("si", 6.4, divisions)
        result = calc.get_scf()
        source = dataclasses.replace(
            _source(calc.system, calc.pseudos, result.density, 4, 6, 1e-12, None), gap_tol=0.0)
        direct = np.asarray(source.states(points).energies)[:, :4] * RY_TO_EV
        errors.append(np.max(np.abs(w.bands_ev(points)[:, :4] - direct)))
    assert errors[0] < 0.35 and errors[1] < 0.03
