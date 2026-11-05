"""Symmetry-adapted Wannier functions (site symmetry), against ``wannier90.x`` and ``pw2wannier90.x``.

Two halves, checked apart as in P162. **The engine** on Wannier90 3.1.0's own
test fixtures (``test-suite/tests/testw90_example21_As_sp``, GaAs, four
As-centred s and p functions, isolated; ``testw90_disentanglement_sawfs``, H3S,
twelve functions out of twenty bands), with their ``.dmn`` files, against
``wannier90.x`` run on them (the same numbers as the fixtures' benchmarks). The
cell goes in with Wannier90's own bohr, 0.52917720859 A: with ours, every
spread moves by ``2 x 4.4e-9`` relative, 8.8e-8 A^2 on GaAs. **The
representations** ``D`` and ``d`` of :mod:`defuqumat.wannier.dmatrix`, against
the GaAs file's ``D`` and against the identity ``A(gk) = d A(k) D^+`` on our
own states, which diamond's nonsymmorphic group makes a check of the
fractional-translation phases.
"""

from pathlib import Path

import numpy as np
import pytest

import defuqumat  # noqa: F401
from defuqumat.wannier import files
from defuqumat.wannier.engine import EngineOptions, kmesh_get, run_engine
from defuqumat.wannier.sitesym import from_dmn

pytestmark = pytest.mark.unit

DATA = Path(__file__).resolve().parents[1] / "data" / "wannier"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"
W90_BOHR = 0.52917720859

#: ``wannier90.x`` (3.1.0) with ``site_symmetry = .true.`` on the fixtures, as printed.
GAAS_W90 = {"omega_i": 6.506264726, "omega_d": 0.015874955, "omega_od": 3.614352981,
            "omega_total": 10.136492662, "spreads": [1.63710679, 2.83312863, 2.83312863, 2.83312862]}
H3S_W90 = {"omega_i": 3.408923571, "omega_od": 2.893033737, "omega_total": 6.301957309,
           "spreads": [0.51280268, 0.56343517, 0.56343517, 0.56343517, 0.61535175, 0.66536369,
                       0.66536350, 0.61535175, 0.66536347, 0.29068499, 0.29068499, 0.29068499]}


def _fixture(name, seed, nb, nw, shells=36):
    d = DATA / name
    text = (d / f"{seed}.win").read_text().lower()
    rows = [l for l in text.split("begin unit_cell_cart")[1].split("end unit_cell_cart")[0].split("\n")
            if l.strip() and l.strip() != "bohr"]
    at = np.array([[float(x) for x in l.split()] for l in rows])
    lines = text.split("begin kpoints")[1].split("end kpoints")[0].split("\n")
    kpt = np.array([[float(x) for x in l.split()[:3]] for l in lines if l.strip()])
    kmesh = kmesh_get(kpt.T, 2.0 * np.pi * np.linalg.inv(at * W90_BOHR).T, search_shells=shells)
    nk = len(kpt)
    mmn = files.read_mmn(d / f"{seed}.mmn.gz", nb, nk, kmesh.nnlist, kmesh.nncell)
    amn = files.read_amn(d / f"{seed}.amn.gz", nb, nk, nw)
    eig = files.read_eig(d / f"{seed}.eig.gz", nb, nk)
    dmn = files.read_dmn(d / f"{seed}.dmn.gz", nw)
    return at, kpt, kmesh, mmn, amn, eig, dmn


def _identity_error(amn, sym):
    err = 0.0
    for ir in range(sym.nkptirr):
        ik = sym.ir2ik[ir]
        for s in range(sym.nsymmetry):
            rhs = sym.d_matrix_band[:, :, s, ir] @ amn[:, :, ik] @ sym.d_matrix_wann[:, :, s, ir].conj().T
            err = max(err, float(np.abs(amn[:, :, sym.kptsym[s, ir]] - rhs).max()))
    return err


def test_the_engine_is_wannier90s_on_gaas_isolated():
    """Four As-centred s, p functions of GaAs, 24 operations, 10 of 64 points
    irreducible. Measured equal to every printed digit."""
    _, _, kmesh, mmn, amn, eig, dmn = _fixture("gaas-sitesym", "GaAs", 4, 4, shells=12)
    sym = from_dmn(dmn, 4, symmetrize_eps=1e-9)
    assert (sym.nsymmetry, sym.nkptirr) == (24, 10)
    r = run_engine(mmn, amn, eig, kmesh, 4, EngineOptions(num_iter=100), sym=sym)
    assert abs(r.terms.om_tot - GAAS_W90["omega_total"]) < 1e-8
    assert abs(r.terms.om_i - GAAS_W90["omega_i"]) < 1e-8
    assert abs(r.terms.om_d - GAAS_W90["omega_d"]) < 1e-8
    assert np.max(np.abs(r.spreads - GAAS_W90["spreads"])) < 2e-8
    # The fixture's own matrices obey the constraint the engine imposes.
    assert _identity_error(amn, sym) < 1e-7


def test_the_engine_is_wannier90s_on_h3s_disentangled():
    """H3S, 12 of 20 bands, 48 operations, ten steps of each loop (the
    fixture's), so the match is a step-by-step one. Measured equal to ten
    digits once the 2x2 eigenvector phase is LAPACK's: with numpy's it was
    2.1e-7 A^2 off."""
    _, _, kmesh, mmn, amn, eig, dmn = _fixture("h3s-sitesym", "H3S", 20, 12)
    sym = from_dmn(dmn, 12, symmetrize_eps=1e-8)
    options = EngineOptions(dis_win_min=-100, dis_win_max=95, dis_num_iter=10,
                            dis_mix_ratio=0.2, num_iter=10)
    with pytest.warns(UserWarning) as record:
        r = run_engine(mmn, amn, eig, kmesh, 12, options, sym=sym)
    messages = " ".join(str(w.message) for w in record)
    assert "did not converge" in messages
    # The fixture's 20 bands cut a multiplet at Gamma; wannier90.x runs on, and a
    # .dmn read from a file is warned about rather than refused.
    assert "cuts a degenerate multiplet" in messages
    assert abs(r.terms.om_tot - H3S_W90["omega_total"]) < 1e-8
    assert abs(r.terms.om_i - H3S_W90["omega_i"]) < 1e-8
    assert np.max(np.abs(r.spreads - H3S_W90["spreads"])) < 2e-8
    # Without symmetry the same ten steps land elsewhere (wannier90.x: 6.234424363).
    with pytest.warns(UserWarning):
        free = run_engine(mmn, amn, eig, kmesh, 12, options)
    assert abs(free.terms.om_tot - 6.234424363) < 1e-8


def test_a_frozen_window_with_site_symmetry_is_refused():
    _, _, kmesh, mmn, amn, eig, dmn = _fixture("h3s-sitesym", "H3S", 20, 12)
    sym = from_dmn(dmn, 12, symmetrize_eps=1e-8)
    with pytest.raises(NotImplementedError, match="frozen window"):
        run_engine(mmn, amn, eig, kmesh, 12,
                   EngineOptions(dis_win_max=95, dis_froz_max=20.0, dis_num_iter=2, num_iter=2),
                   sym=sym)


def test_the_dmn_file_round_trips(tmp_path):
    dmn = files.read_dmn(DATA / "gaas-sitesym" / "GaAs.dmn.gz", 4)
    files.write_dmn(tmp_path / "x.dmn", dmn["ik2ir"] - 1, dmn["ir2ik"] - 1, dmn["kptsym"] - 1,
                    dmn["d_matrix_wann"], dmn["d_matrix_band"])
    back = files.read_dmn(tmp_path / "x.dmn", 4)
    for key in ("ik2ir", "ir2ik", "kptsym"):
        assert np.array_equal(back[key], dmn[key])
    for key in ("d_matrix_wann", "d_matrix_band"):
        assert np.max(np.abs(back[key] - dmn[key])) < 1e-9


def _gaas_operations():
    """GaAs.sym (compute_dmn's own list): ``sr`` column-major, ``tvec`` (zero here)."""
    from defuqumat.wannier.dmatrix import Operations

    txt = (DATA / "gaas-sitesym" / "GaAs.sym").read_text().split()
    nsym = int(txt[0])
    vals = np.array([float(x) for x in txt[1:]]).reshape(nsym, 12)
    rotations = np.array([v[:9].reshape(3, 3, order="F") for v in vals])
    assert np.abs(vals[:, 9:]).max() == 0.0
    return Operations(rotations=rotations, translations=np.zeros((nsym, 3)))


def test_d_wann_and_the_stars_are_compute_dmns_on_gaas():
    """``D`` from :func:`wannier_representation` on the file's 24 operations is
    the file's ``d_matrix_wann``, and the stars are its ``ik2ir``/``ir2ik``/
    ``kptsym``. Measured 4e-15."""
    from defuqumat.wannier.dmatrix import kpoint_stars, wannier_representation
    from defuqumat.wannier.projections import apply_projection_selection, parse_projections

    at, kpt, _, _, _, _, dmn = _fixture("gaas-sitesym", "GaAs", 4, 4, shells=12)
    bg = 2.0 * np.pi * np.linalg.inv(at).T
    ops = _gaas_operations()
    stars = kpoint_stars(kpt, ops, at, bg)
    ik2ir, ir2ik, kptsym, _ = stars
    assert np.array_equal(ik2ir, dmn["ik2ir"] - 1)
    assert np.array_equal(ir2ik, dmn["ir2ik"] - 1)
    assert np.array_equal(kptsym, dmn["kptsym"] - 1)
    crystal = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    parsed = parse_projections(["f=0.25,0.25,0.25:s", "f=0.25,0.25,0.25:p"], ["Ga", "As"],
                               crystal.T, bg)
    trial = apply_projection_selection(parsed, 4, None)
    D = wannier_representation(trial, ops, at, bg, kpt, stars)
    assert np.max(np.abs(D - dmn["d_matrix_wann"])) < 1e-10


def test_a_projection_set_that_is_not_a_representation_is_refused():
    """One sp3 lobe on one atom is not mapped onto the set by the tetrahedral group."""
    from defuqumat.wannier.dmatrix import wannier_rotations
    from defuqumat.wannier.projections import apply_projection_selection, parse_projections

    at = 10.68 / 2 * np.array([[-1.0, 0, 1], [0, 1, 1], [-1, 1, 0]])
    bg = 2.0 * np.pi * np.linalg.inv(at).T
    crystal = np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]])
    parsed = parse_projections(["As:l=-3,mr=1"], ["Ga", "As"], crystal.T, bg)
    trial = apply_projection_selection(parsed, 1, None)
    with pytest.raises(ValueError, match="not a representation"):
        wannier_rotations(trial, _gaas_operations(), at)
    # A centre that is not on an orbit of the group: refused by name too.
    parsed = parse_projections(["f=0.1,0.0,0.0:s"], ["Ga", "As"], crystal.T, bg)
    with pytest.raises(ValueError, match="whole orbits"):
        wannier_rotations(apply_projection_selection(parsed, 1, None), _gaas_operations(), at)


# --- the pipeline on our own states ------------------------------------------------------

DIAMOND_PROJECTIONS = ["f=0,0,0:s", "f=0,0,0.5:s", "f=0,0.5,0:s", "f=0.5,0,0:s"]


@pytest.fixture(scope="module")
def diamond():
    from defuqumat import Calculator

    calc = Calculator.from_file(DATA / "diamond" / "diamond.scf", pseudo_dir=PSEUDO)
    calc.get_scf()
    return calc


def test_diamond_with_symmetry_obeys_the_constraint_and_writes_the_dmn(diamond, tmp_path):
    """Four bond-centred s functions, 48 operations of which 24 carry the
    fractional translation: ``A(gk) = d A(k) D^+`` on our own states (measured
    5.8e-10), the spread unchanged (2.349146410 A^2 with and without), and
    ``write_w90`` writes the ``.dmn`` that ``wannier90.x`` reads (2.349146389
    from those files, Wannier90's bohr)."""
    w = diamond.get_wannier(4, DIAMOND_PROJECTIONS, divisions=(4, 4, 4), symmetry=True,
                            engine_options=dict(num_iter=100, conv_window=3))
    sym = w.site_symmetry
    assert (w.nsym, sym.nsymmetry, sym.nkptirr) == (48, 48, 8)
    assert _identity_error(w.amn, sym) < 1e-8
    assert abs(w.spread_terms_angstrom2()["omega_total"] - 2.349146410) < 1e-6
    assert np.ptp(w.spreads_angstrom2) < 1e-8
    w.write_w90(str(tmp_path / "di"), projections=DIAMOND_PROJECTIONS)
    assert "site_symmetry = .true." in (tmp_path / "di.win").read_text()
    back = files.read_dmn(tmp_path / "di.dmn", 4)
    assert np.array_equal(back["kptsym"] - 1, sym.kptsym)
    assert np.max(np.abs(back["d_matrix_band"] - sym.d_matrix_band)) < 1e-9


def _si_text(extra: str) -> str:
    return f"""&control
 calculation='scf', pseudo_dir='{PSEUDO}'
/
&system
 ibrav=2, celldm(1)=10.2, nat=2, ntyp=1, ecutwfc=12.0 {extra}
/
&electrons
 conv_thr=1e-10
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS {{crystal}}
Si 0.00 0.00 0.00
Si 0.25 0.25 0.25
K_POINTS {{automatic}}
 2 2 2 0 0 0
"""


#: The four bonds of the atom at the origin, in crystal units of the fcc cell.
SI_BONDS = ["f=0.125,0.125,0.125:s", "f=-0.375,0.125,0.125:s", "f=0.125,-0.375,0.125:s",
            "f=0.125,0.125,-0.375:s"]


def test_with_nosym_the_option_is_a_no_op_bit_for_bit():
    """The group of a ``nosym`` run is the identity alone, and ``symmetry=True``
    then changes nothing at all."""
    from defuqumat import Calculator

    calc = Calculator.from_text(_si_text(", nosym=.true."), pseudo_dir=PSEUDO)
    calc.get_scf()
    a = calc.get_wannier(4, SI_BONDS, divisions=(2, 2, 2), engine_options=dict(num_iter=20))
    b = calc.get_wannier(4, SI_BONDS, divisions=(2, 2, 2), symmetry=True,
                         engine_options=dict(num_iter=20))
    assert (b.nsym, b.site_symmetry) == (1, None)
    assert np.array_equal(a.spreads, b.spreads)
    assert np.array_equal(a.hoppings, b.hoppings)
    assert np.array_equal(a.u_matrix, b.u_matrix)


def test_what_the_site_symmetry_does_not_treat_is_refused():
    from types import SimpleNamespace

    from defuqumat import Calculator
    from defuqumat.workflows.wannier import _refuse_for_symmetry, run_wannier

    calc = Calculator.from_text(_si_text(""), pseudo_dir=PSEUDO)
    with pytest.raises(NotImplementedError, match="frozen window"):
        run_wannier(calc.system, calc.pseudos, None, num_wann=8, num_bands=12,
                    projections=["Si:sp3"], symmetry=True, dis_froz=(None, 6.0))
    with pytest.raises(NotImplementedError, match="frozen window"):
        run_wannier(calc.system, calc.pseudos, None, num_wann=8, num_bands=12,
                    projections=["Si:sp3"], symmetry=True,
                    engine_options=dict(dis_froz_max=6.0))
    with pytest.raises(NotImplementedError, match="projection lines"):
        run_wannier(calc.system, calc.pseudos, None, num_wann=8, projections="auto",
                    symmetry=True)
    with pytest.raises(NotImplementedError, match="precond"):
        run_wannier(calc.system, calc.pseudos, None, num_wann=8, projections=["Si:sp3"],
                    symmetry=True, engine_options=dict(precond=True))
    with pytest.raises(NotImplementedError, match="spinor"):
        spinor = Calculator.from_text(_si_text(", noncolin=.true."), pseudo_dir=PSEUDO)
        run_wannier(spinor.system, spinor.pseudos, None, num_wann=8, projections=["Si:sp3"],
                    symmetry=True)
    # A group with a spin flip only arises with spinors, refused above; the guard
    # itself is checked on a stand-in system whose group carries one.
    group = SimpleNamespace(magnetic=True)
    fake = SimpleNamespace(nosym=False, symmetry_group=lambda nosym: group)
    with pytest.raises(NotImplementedError, match="magnetic group"):
        _refuse_for_symmetry(fake, ["Si:sp3"], None, None)
