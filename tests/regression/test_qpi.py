"""T-matrix quasiparticle interference, on converged surfaces.

``PLAN.md`` P157. Neither ``pw.x`` nor Elk computes this quantity, so the
checks close inside the package, and each shares as little as possible with
the route it checks. The algebra against the dense supercell inverse is in
``tests/unit/test_qpi_machinery.py``; what needs an SCF is here:

* the clean surface's ``landauer`` background is
  :func:`~defuqumat.workflows.transport.run_vertical_transport`'s map, which is
  P66's code and not this one;
* a scatterer of zero strength changes nothing, exactly;
* the maps converge with the band window (P66's trap is that the literal exit
  resolvent does not, which is asserted too, so the guard is seen to fire);
* graphene with a scatterer on one atom keeps the C3 axis through that atom,
  once ``eta`` exceeds the level spacing on the contour.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defuqumat import Calculator

pytestmark = pytest.mark.slow

DATA = Path(__file__).resolve().parents[1] / "data"
FIGURES = Path(__file__).resolve().parents[2] / "docs" / "figures"

#: The hydrogen sheet's atom is at 0.5 of c = 15 bohr.
SHEET = dict(height=0.7, exit_height=0.3, readers=("ldos", "landauer"),
             grid=(6, 6, 1), shape=(4, 4), broadening=0.05)


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(path: str):
    calculator = Calculator.from_file(path, pseudo_dir=DATA / "pseudo")
    calculator.get_scf()
    return calculator


def _sheet():
    return _converged(str(DATA / "qe" / "h-sheet.in"))


def test_the_clean_landauer_background_is_the_vertical_transmission():
    calculator = _sheet()
    energy = calculator.get_scf().fermi_energy + 0.05
    qpi = calculator.get_qpi(0.5, energies=[energy], **SHEET)
    flow = calculator.get_vertical_transport(
        exit_height=0.3, height=0.7, shape=(4, 4), grid=(6, 6, 1),
        broadening=0.05, energies=energy)
    want = np.asarray(flow.values)
    got = qpi.backgrounds["landauer"][0]
    assert np.abs(got - want).max() < 1e-12 * np.abs(want).max()


def test_a_scatterer_of_zero_strength_changes_nothing():
    qpi = _sheet().get_qpi(0.0, **SHEET)
    for values in qpi.maps.values():
        assert np.abs(values).max() == 0.0


@pytest.mark.parametrize("exit_method, bound", [("spectral", 2e-3),
                                                ("resolvent", None)])
def test_the_maps_converge_with_the_window(exit_method, bound):
    """32 bands against 64, ``q != 0``: measured 3.5e-4 apart (``spectral``).

    The literal resolvent on the exit leg measured 3.8e-2 apart in the same
    comparison, so it must fail where the on-shell leg passes.
    """
    calculator = _sheet()
    maps = {nbnd: calculator.get_qpi(0.5, nbnd=nbnd, exit_method=exit_method,
                                     **SHEET).transforms["landauer"][0]
            for nbnd in (32, 64)}
    centre = tuple(n // 2 for n in maps[32].shape)
    for value in maps.values():
        value[centre] = 0.0
    error = np.abs(maps[32] - maps[64]).max() / np.abs(maps[64]).max()
    if bound is None:
        assert error > 1e-2
    else:
        assert error < bound


def _c3_residue(values):
    """``max |f - C3 f| / max |f|`` on a hexagonal ``q`` grid centred at 0."""
    n = values.shape[0]
    centre = n // 2
    p = np.arange(n) - centre
    p1, p2 = np.meshgrid(p, p, indexing="ij")
    # C3 on crystal coordinates of r is (x1, x2) -> (-x2, x1 - x2); on q it is
    # the inverse transpose of that matrix.
    rotation = np.linalg.inv(np.array([[0, 1], [-1, -1]])).T.round().astype(int)
    turned = np.stack([p1, p2], axis=-1) @ rotation
    image = values[(turned[..., 0] + centre) % n, (turned[..., 1] + centre) % n]
    return np.abs(values - image).max() / np.abs(values).max()


def test_graphene_keeps_the_c3_axis_through_the_scatterer():
    """Measured 2.1e-4 (``ldos``) and 5.3e-3 (``landauer``) at ``eta = 0.08``.

    At ``eta = 0.03`` the ``landauer`` residue was 0.35: the on-shell exit leg
    then samples the contour on a handful of k-points, which is what
    :attr:`~defuqumat.stm.qpi.QPI.contour_points` reports.
    """
    calculator = _converged(str(FIGURES / "stm_transport" / "graphene.in"))
    fermi = calculator.get_scf().fermi_energy
    qpi = calculator.get_qpi(1.0, 0.62, grid=(18, 18, 1), nbnd=12,
                             readers=("ldos", "landauer"), exit_height=0.38,
                             shape=(6, 6), broadening=0.08,
                             energies=[fermi - 0.08])
    assert _c3_residue(np.abs(qpi.transforms["ldos"][0])) < 2e-3
    assert _c3_residue(np.abs(qpi.transforms["landauer"][0])) < 2e-2
