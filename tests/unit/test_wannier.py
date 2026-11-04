"""Maximally localized Wannier functions, against ``pw2wannier90.x`` and ``wannier90.x``.

The references in ``tests/data/wannier/`` are QE 7.5's ``pw.x`` and
``pw2wannier90.x`` with Wannier90 3.1.0's ``wannier90.x``, run on the inputs
stored beside them, with the cell given to ``wannier90.x`` in bohr to twelve
digits (an angstrom cell to six decimals moves the spread by 1e-6 A^2). The
checks are split as the two codes are: the engine on ``pw2wannier90.x``'s own
matrices, which needs no state of ours, and the whole pipeline on ours, which
is the Bloch side. Both have to land on ``wannier90.x``'s printed spread.
"""

from pathlib import Path

import numpy as np
import pytest

import defuqumat  # noqa: F401
from defuqumat.units import BOHR_TO_ANGSTROM
from defuqumat.wannier import files
from defuqumat.wannier.engine import EngineOptions, kmesh_get, run_engine
from defuqumat.wannier.trial import _COMBINATIONS, angular_parts, guiding_functions, radial_transform
from defuqumat.workflows.wannier import uniform_mesh

pytestmark = pytest.mark.unit

DATA = Path(__file__).resolve().parents[1] / "data" / "wannier"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: ``wannier90.x`` on ``pw2wannier90.x``'s files, as printed in each ``.wout``.
DIAMOND_W90 = {"omega_i": 1.973863427, "omega_od": 0.375282848, "omega_total": 2.349146274,
               "spread": 0.58728657}
SI_W90 = {"omega_i": 11.813496347, "omega_d": 0.137522691, "omega_od": 4.102176072,
          "omega_total": 16.053195110}
#: ``diamond.win``'s cell, a = 6.1 bohr fcc, and the four bond centres.
DIAMOND_PROJECTIONS = ["f=0,0,0:s", "f=0,0,0.5:s", "f=0,0.5,0:s", "f=0.5,0,0:s"]


def _cell_and_mesh(win: Path):
    text = win.read_text().lower()
    rows = [l for l in text.split("begin unit_cell_cart")[1].split("end unit_cell_cart")[0].split("\n")
            if l.strip() and l.strip() != "bohr"]
    at_bohr = np.array([[float(x) for x in l.split()] for l in rows])
    lines = text.split("begin kpoints")[1].split("end kpoints")[0].split("\n")
    kpt = np.array([[float(x) for x in l.split()[:3]] for l in lines if l.strip()])
    return at_bohr, kpt


def _engine_on_reference(directory: Path, seed: str, nb: int, nw: int, options: EngineOptions,
                         suffix: str = ""):
    at, kpt = _cell_and_mesh(directory / f"{seed}.win")
    recip = 2.0 * np.pi * np.linalg.inv(at * BOHR_TO_ANGSTROM).T
    kmesh = kmesh_get(kpt.T, recip)
    nk = len(kpt)
    mmn = files.read_mmn(directory / f"{seed}.mmn{suffix}", nb, nk, kmesh.nnlist, kmesh.nncell)
    amn = files.read_amn(directory / f"{seed}.amn{suffix}", nb, nk, nw)
    eig = files.read_eig(directory / f"{seed}.eig", nb, nk)
    return run_engine(mmn, amn, eig, kmesh, nw, options)


def test_the_kpoint_order_is_wannier90s():
    """The mesh of ``run_wannier`` is ``kmesh.pl``'s, the last index fastest."""
    _, kpt = _cell_and_mesh(DATA / "si" / "si.win")   # written by kmesh.pl 4 4 4
    assert np.allclose(uniform_mesh((4, 4, 4)), kpt, atol=1e-12)


def test_the_engine_on_pw2wannier90s_matrices_is_wannier90s_isolated():
    """Diamond, four bond-centred s orbitals: the four valence bands, no window."""
    r = _engine_on_reference(DATA / "diamond", "diamond", 4, 4,
                             EngineOptions(num_iter=100, conv_window=3))
    assert not r.disentangled and r.converged
    assert abs(r.terms.om_tot - DIAMOND_W90["omega_total"]) < 1.0e-7
    assert abs(r.terms.om_i - DIAMOND_W90["omega_i"]) < 1.0e-7
    assert np.max(np.abs(r.spreads - DIAMOND_W90["spread"])) < 1.0e-7


def test_the_engine_on_pw2wannier90s_matrices_is_wannier90s_disentangled():
    """Silicon, 8 sp3 orbitals out of 12 bands, frozen below 6.4 eV.

    Measured 1.3e-7 A^2 above ``wannier90.x`` on the same file; the two atoms
    end in different local minima (spreads 1.818 and 2.196 A^2), so a port that
    lands elsewhere is caught by the per-function spreads too.
    """
    r = _engine_on_reference(DATA / "si", "si", 12, 8,
                             EngineOptions(dis_win_max=17.0, dis_froz_max=6.4, num_iter=200,
                                           conv_window=3), suffix=".gz")
    assert r.disentangled and r.converged and r.dis_converged
    for key, value in (("om_tot", "omega_total"), ("om_i", "omega_i"), ("om_d", "omega_d"),
                       ("om_od", "omega_od")):
        assert abs(getattr(r.terms, key) - SI_W90[value]) < 5.0e-7, key
    assert np.allclose(np.sort(r.spreads), [1.8177444] * 4 + [2.1955544] * 4, atol=2e-5)


def test_the_conv_window_default_runs_every_step():
    """Wannier90's ``conv_window = -1`` takes all ``num_iter`` steps (wannierpy
    replaced it by 5)."""
    r = _engine_on_reference(DATA / "diamond", "diamond", 4, 4, EngineOptions(num_iter=7))
    assert r.iterations == 7


def test_the_angular_functions_are_orthonormal():
    """Each family of Wannier90's real functions is orthonormal on the sphere,
    hybrids included, by Gauss-Legendre in cos(theta) times a uniform phi."""
    x, w = np.polynomial.legendre.leggauss(24)
    phi = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    cost, ph = np.meshgrid(x, phi, indexing="ij")
    sint = np.sqrt(1.0 - cost ** 2)
    directions = np.stack([sint * np.cos(ph), sint * np.sin(ph), cost], axis=-1).reshape(-1, 3)
    weights = (w[:, None] * np.full(len(phi), 2.0 * np.pi / len(phi))[None, :]).ravel()
    for l in (0, 1, 2, 3, -1, -2, -3, -4, -5):
        members = sorted(mr for (ll, mr) in _COMBINATIONS if ll == l)
        values = np.stack([angular_parts(l, mr, directions).sum(axis=0) for mr in members])
        gram = (values * weights) @ values.T
        assert np.allclose(gram, np.eye(len(members)), atol=1e-12), l


def test_the_mesh_radial_table_is_the_per_k_transform():
    """The one table over the distinct ``|k + G|`` gives the same ``g``."""
    class P:
        site_frac, l, m, radial, zona = np.array([0.1, 0.2, 0.3]), -3, 2, 2, 1.7
        z_axis, x_axis = np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])
    at = np.array([[-5.1, 0.0, 5.1], [0.0, 5.1, 5.1], [-5.1, 5.1, 0.0]])
    kg = np.random.default_rng(3).normal(size=(40, 3))
    kg[0] = 0.0
    volume = abs(np.linalg.det(at))
    direct = guiding_functions([P], kg, at, volume)
    table = {(1.7, 2): radial_transform(np.linalg.norm(kg, axis=1), 1.7, 2, volume)}
    assert np.allclose(direct, guiding_functions([P], kg, at, volume, radial=table), atol=1e-15)
    assert np.all(np.isfinite(direct))


def test_the_files_round_trip(tmp_path):
    at, kpt = _cell_and_mesh(DATA / "diamond" / "diamond.win")
    kmesh = kmesh_get(kpt.T, 2.0 * np.pi * np.linalg.inv(at * BOHR_TO_ANGSTROM).T)
    mmn = files.read_mmn(DATA / "diamond" / "diamond.mmn", 4, 64, kmesh.nnlist, kmesh.nncell)
    amn = files.read_amn(DATA / "diamond" / "diamond.amn", 4, 64, 4)
    files.write_mmn(tmp_path / "x.mmn", mmn, kmesh.nnlist, kmesh.nncell)
    files.write_amn(tmp_path / "x.amn", amn)
    assert np.allclose(files.read_mmn(tmp_path / "x.mmn", 4, 64, kmesh.nnlist, kmesh.nncell), mmn,
                       atol=1e-12)
    assert np.allclose(files.read_amn(tmp_path / "x.amn", 4, 64, 4), amn, atol=1e-12)


@pytest.fixture(scope="module")
def diamond():
    from defuqumat import Calculator

    calc = Calculator.from_file(DATA / "diamond" / "diamond.scf", pseudo_dir=PSEUDO)
    calc.get_scf()
    return calc


def test_the_pipeline_is_pw2wannier90_and_wannier90(diamond):
    """Our states, matrices and engine against ``pw.x``, ``pw2wannier90.x`` and
    ``wannier90.x``: two SCFs, so a tolerance of 1e-6 A^2 (measured 2e-9 on the
    ecut-40 cell, and 4e-7 on disentangled silicon)."""
    w = diamond.get_wannier(4, DIAMOND_PROJECTIONS, divisions=(4, 4, 4),
                            engine_options=dict(num_iter=100, conv_window=3))
    terms = w.spread_terms_angstrom2()
    assert abs(terms["omega_total"] - DIAMOND_W90["omega_total"]) < 1.0e-6
    assert abs(terms["omega_i"] - DIAMOND_W90["omega_i"]) < 1.0e-6
    assert np.max(np.abs(w.spreads_angstrom2 - DIAMOND_W90["spread"])) < 1.0e-6
    # The centres are the bond centres of the .wout, in angstrom.
    expected = np.array([[0, 0, 0], [-0.806995, 0.806995, 0], [0, 0.806995, 0.806995],
                         [-0.806995, 0, 0.806995]])
    assert np.max(np.abs(w.centres_angstrom - expected)) < 2e-6
    # The interpolation is exact on the mesh for an isolated set, with and
    # without use_ws_distance: H(R) is the transform of H(k) there.
    assert np.max(np.abs(w.bands(w.kpoints) - w.eigenvalues)) < 1e-10


def test_the_interpolation_without_ws_distance_is_exact_on_the_mesh(diamond):
    w = diamond.get_wannier(4, DIAMOND_PROJECTIONS, divisions=(4, 4, 4), use_ws_distance=False,
                            engine_options=dict(num_iter=10))
    assert w.ws_shifts is None
    assert np.max(np.abs(w.bands(w.kpoints) - w.eigenvalues)) < 1e-10


def _si_text(extra: str) -> str:
    return f"""&control
 calculation='scf', pseudo_dir='{PSEUDO}'
/
&system
 ibrav=2, celldm(1)=10.2, nat=2, ntyp=1, ecutwfc=12.0 {extra}
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS {{crystal}}
Si 0.00 0.00 0.00
Si 0.25 0.25 0.25
K_POINTS {{automatic}}
 2 2 2 0 0 0
"""


def test_a_spinor_calculation_is_refused():
    from defuqumat import Calculator
    from defuqumat.workflows.wannier import run_wannier

    calc = Calculator.from_text(_si_text(", noncolin=.true."), pseudo_dir=PSEUDO)
    with pytest.raises(NotImplementedError, match="spinor"):
        run_wannier(calc.system, calc.pseudos, None, num_wann=8, projections=["Si:sp3"])


def test_a_collinear_magnet_needs_its_channel():
    from defuqumat import Calculator
    from defuqumat.workflows.wannier import run_wannier

    calc = Calculator.from_text(_si_text(", nspin=2, starting_magnetization(1)=0.5, tot_magnetization=0"),
                                pseudo_dir=PSEUDO)
    with pytest.raises(ValueError, match="spin = 0 or 1"):
        run_wannier(calc.system, calc.pseudos, None, num_wann=8, projections=["Si:sp3"])


def test_fewer_bands_than_functions_is_refused():
    from defuqumat import Calculator
    from defuqumat.workflows.wannier import run_wannier

    calc = Calculator.from_text(_si_text(""), pseudo_dir=PSEUDO)
    with pytest.raises(ValueError, match="fewer than num_wann"):
        run_wannier(calc.system, calc.pseudos, None, num_wann=8, num_bands=4,
                    projections=["Si:sp3"])
