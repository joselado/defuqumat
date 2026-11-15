"""The three ``Calculator`` entry points of the uniform orbital field.

Each is a delegation, so what is checked is that the bound method gives the functional
entry point's numbers, and the refusals the facade adds: ``get_landau_levels`` checks the
regime of the calculator itself (the Peierls workflow sees only the Wannier set, so it
cannot), the ultracell route needs a flux and a band count, and the band routes need
``nbnd``. Nothing here runs an SCF: the Peierls model comes from the committed ``p_z``
set of the shared graphene state and the bands from the free Landau problem on a small box.
"""

from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.units import RY_TO_EV
from defuqumat.wannier.interpolation import interpolate
from defuqumat.workflows.landau import run_landau_levels
from defuqumat.workflows.magnetic_bands import run_magnetic_bands

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "plans" / "orbital-field"


@pytest.fixture(autouse=True)
def _clear():
    yield
    jax.clear_caches()


def _graphene(**extra):
    return Calculator.from_keywords(
        lattice="hexagonal", a=2.46, c=15.0,
        atoms=extra.pop("atoms", [("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))]),
        pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir=str(ROOT / "tests" / "data" / "pseudo"),
        ecut=50 * RY_TO_EV, kmesh=(1, 1, 1), **extra)


@pytest.fixture(scope="module")
def pz_set():
    path = DATA / "graphene-wannier-pz.npz"
    if not path.exists():
        pytest.skip("graphene-wannier-pz.npz is not in plans/orbital-field")
    z = np.load(path)
    shifted, count = z["ws_shifted"], z["ws_count"]
    ws = None if shifted.size == 0 else (shifted, count)

    def bands(kappa):
        return interpolate(np.atleast_2d(kappa), z["rvectors"], z["degeneracy"], z["hoppings"],
                           *(ws if ws is not None else (None, None)))[0]

    return SimpleNamespace(centres=z["centres"], rvectors=z["rvectors"],
                           degeneracy=z["degeneracy"], hoppings=z["hoppings"],
                           lattice=z["lattice"], ws_shifts=ws, spin=None,
                           edge=float(z["fermi_energy"]), bands=bands)


def test_get_landau_levels_is_the_peierls_workflow(pz_set):
    calc = _graphene()
    options = dict(flux=(1, 40), kappa=((0.11, 0.23),), nlevels=8, dirac=True)
    facade = calc.get_landau_levels(wannier=pz_set, **options)
    direct = run_landau_levels(pz_set, geometry="torus", **options)
    assert facade.flux == (1, 40)
    assert np.array_equal(facade.levels, direct.levels)
    assert facade.dirac["E_D"] == direct.dirac["E_D"]
    # a field in tesla is rounded to a flux, and the field used is on the result
    in_tesla = calc.get_landau_levels(wannier=pz_set, B=2000.0, nlevels=4, dirac=False)
    # 2000 T is 1/39.46 flux quanta per cell, so q = 39 and B = 78912 T / 39
    assert in_tesla.flux == (1, 39) and abs(in_tesla.field_tesla * 39 - 78912.0) < 1.0


def test_a_set_with_no_edge_is_centred_from_the_fermi_level(pz_set):
    """``get_wannier`` with explicit projections leaves ``edge`` at ``None``.

    The first facade run on such a set stopped in ``float(None)``; the workflow now takes
    ``fermi_energy`` for the start of the Dirac search, and refuses by name without one.
    """
    no_edge = SimpleNamespace(**{**vars(pz_set), "edge": None})
    options = dict(flux=(1, 40), kappa=((0.11, 0.23),), nlevels=8, dirac=True)
    with pytest.raises(ValueError, match="no edge"):
        run_landau_levels(no_edge, **options)
    given = run_landau_levels(no_edge, fermi_energy=pz_set.edge, **options)
    assert np.array_equal(given.levels, run_landau_levels(pz_set, **options).levels)


def test_get_landau_levels_refuses_the_regime_of_the_calculator(pz_set):
    magnetic = _graphene(spin="collinear", smearing=0.1,
                         atoms=[("C", (0, 0, 0), 0.5), ("C", (1 / 3, 2 / 3, 0), 0.5)])
    with pytest.raises(NotImplementedError, match="nspin = 2"):
        magnetic.get_landau_levels(wannier=pz_set, flux=(1, 40))
    calc = _graphene()
    with pytest.raises(NotImplementedError, match="Zeeman"):
        calc.get_landau_levels(wannier=pz_set, flux=(1, 40), zeeman=True)
    with pytest.raises(ValueError, match="nbnd"):
        calc.get_landau_levels(route="ultracell", flux=(1, 10))
    with pytest.raises(ValueError, match="q <= 100"):
        calc.get_landau_levels(route="ultracell", flux=(1, 200), nbnd=8)
    with pytest.raises(ValueError, match="route"):
        calc.get_landau_levels(route="ribbon", wannier=pz_set, flux=(1, 40))


def test_get_magnetic_bands_is_the_route_p_workflow():
    calc = _graphene()
    options = dict(flux=(1, 2), kmesh=(2, 2), nbnd=4, free=True, local_only=True,
                   grid=(4, 4, 8), david=4, ethr=1e-11, max_iterations=400)
    facade = calc.get_magnetic_bands(**options)
    direct = run_magnetic_bands(calc.system, calc.pseudos, None, **options)
    assert np.max(np.abs(facade.eigenvalues - direct.eigenvalues)) < 1e-12
    assert calc.scf_result is None, "the free problem needs no ground state"
    with pytest.raises(ValueError, match="nbnd"):
        calc.get_magnetic_bands(flux=(1, 2))
    with pytest.raises(ValueError, match="nbnd"):
        calc.get_magnetizability()


def test_each_route_refuses_the_options_it_does_not_read(pz_set):
    """A named facade option one branch dropped is now refused there (audit ``magnetic`` F2, F3, N1).

    ``zeeman=True`` on the ultracell route came back with no Zeeman term; with
    ``wannier=`` given every extra keyword was dropped, ``qmax`` and ``energy``
    among them; ``get_magnetic_bands`` dropped ``chern`` under ``scf=True`` and
    ``conv_thr`` without it. Each refusal comes before any SCF.
    """
    calc = _graphene()
    with pytest.raises(NotImplementedError, match="Zeeman"):
        calc.get_landau_levels(route="ultracell", flux=(1, 10), nbnd=8, zeeman=True)
    with pytest.raises(TypeError, match="geometry"):
        calc.get_landau_levels(route="ultracell", flux=(1, 10), nbnd=8, geometry="ribbon")
    with pytest.raises(TypeError, match="nbnd and N1"):
        calc.get_landau_levels(wannier=pz_set, flux=(1, 40), nbnd=8)
    with pytest.raises(TypeError, match="qmaxx"):
        calc.get_landau_levels(wannier=pz_set, B=10.0, qmaxx=500)
    with pytest.raises(TypeError, match="chern and local_only"):
        calc.get_magnetic_bands(flux=(1, 2), nbnd=4, scf=True, chern=True)
    with pytest.raises(TypeError, match="conv_thr"):
        calc.get_magnetic_bands(flux=(1, 2), nbnd=4, conv_thr=1e-10)


def test_the_peierls_options_reach_the_workflow(pz_set):
    """``qmax`` given to the facade is the one ``run_landau_levels`` rounds ``B`` with (``magnetic`` F3)."""
    calc = _graphene()
    facade = calc.get_landau_levels(wannier=pz_set, B=2000.0, nlevels=4, dirac=False, qmax=50)
    direct = run_landau_levels(pz_set, B=2000.0, nlevels=4, qmax=50)
    assert facade.flux == direct.flux and facade.flux[1] <= 50
