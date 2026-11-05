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


def _frozen_checks(u_opt, windows, sym, lwindow):
    """max_k |(1 - U U^+) E_F| and max |P(gk) - d P(k) d^+| over the stars."""
    from defuqumat.wannier.sitesym import slim_d_matrix_band

    defect = 0.0
    P = []
    for k in range(u_opt.shape[2]):
        nd, nf = windows.ndimwin[k], windows.ndimfroz[k]
        U = u_opt[:nd, :, k]
        e = np.zeros((nd, nf))
        e[windows.indxfroz[:nf, k], np.arange(nf)] = 1.0
        if nf:
            defect = max(defect, float(np.abs(e - U @ (U.conj().T @ e)).max()))
        P.append(U @ U.conj().T)
    sw = slim_d_matrix_band(sym, lwindow)
    cov = 0.0
    for ir in range(sw.nkptirr):
        ik = sw.ir2ik[ir]
        nd = windows.ndimwin[ik]
        for s in range(sw.nsymmetry):
            d = sw.d_matrix_band[:nd, :nd, s, ir]
            cov = max(cov, float(np.abs(P[sw.kptsym[s, ir]] - d @ P[ik] @ d.conj().T).max()))
    return defect, cov


def test_a_frozen_window_with_site_symmetry_keeps_both():
    """H3S, outer window to 65 eV (95 cuts a multiplet at Gamma), frozen to 20 eV:
    the block form (P164b) keeps the frozen states exact (measured 1.2e-11) and
    ``P(gk) = d P(k) d^+`` (1.5e-11), and lands on ``Omega_I`` 3.6431179138 A^2,
    the same as the plain frozen route here, whose minimum is symmetric. The
    energy windows and the same sets given as masks give the same subspace."""
    from defuqumat.wannier import disentangle as dis

    _, _, kmesh, mmn, amn, eig, dmn = _fixture("h3s-sitesym", "H3S", 20, 12)
    sym = from_dmn(dmn, 12, symmetrize_eps=1e-8)
    windows = dis.dis_windows(eig, 12, -100.0, 65.0, True, -100.0, 20.0)
    assert windows.linner and windows.ndimfroz.min() == 5
    by_mask = dis.windows_from_masks(eig, 12, (eig >= -100.0) & (eig <= 65.0), eig <= 20.0)
    wbtot = float(np.sum(kmesh.wb))
    omega = []
    for win in (None, by_mask):
        u_opt, _, lwindow, _, converged = dis.dis_main(
            amn, mmn, eig, kmesh.nnlist, kmesh.wb, 12, -100.0, 65.0, True, -100.0, 20.0,
            400, 0.5, 1e-10, 3, windows=win, sym=sym)
        assert converged
        defect, cov = _frozen_checks(u_opt, windows, sym, lwindow)
        assert defect < 1e-10 and cov < 1e-9
        omega.append(dis._womegai(u_opt, dis.slim_m(mmn, windows, kmesh.nnlist), windows,
                                  kmesh.nnlist, kmesh.wb, wbtot, 12))
    assert abs(omega[0] - 3.6431179138) < 1e-8
    assert abs(omega[1] - omega[0]) < 1e-10


def _toy(d_band, d_wann, frozen_rows):
    """A one-point SymmetryData and its windows: ``d_band`` (nb, nb, nsym),
    ``d_wann`` (nw, nw, nsym), every operation in the little group, the frozen
    bands given by index."""
    from defuqumat.wannier.disentangle import windows_from_masks
    from defuqumat.wannier.sitesym import SymmetryData

    nb, nw, nsym = d_band.shape[0], d_wann.shape[0], d_band.shape[2]
    eig = np.arange(nb, dtype=float)[:, None]
    frozen = np.zeros((nb, 1), dtype=bool)
    frozen[frozen_rows, 0] = True
    windows = windows_from_masks(eig, nw, np.ones((nb, 1), dtype=bool), frozen)
    sym = SymmetryData(nsymmetry=nsym, nkptirr=1, num_kpts=1, ik2ir=np.zeros(1, dtype=np.int64),
                       ir2ik=np.array([0]), kptsym=np.zeros((nsym, 1), dtype=np.int64),
                       d_matrix_wann=d_wann[:, :, :, None].astype(complex),
                       d_matrix_band=d_band[:, :, :, None].astype(complex))
    return sym, windows


def test_a_frozen_irrep_the_trial_orbitals_lack_is_refused():
    """A mirror (two operations): the frozen band is odd and both trial orbitals
    are even, so no U keeps it frozen and symmetric; in the second case a trial
    orbital is odd and the free bands are all even. Both guards are the rank of
    a generic intertwiner (Schur's lemma)."""
    from defuqumat.wannier.sitesym import frozen_blocks, intertwiner_rank

    A = np.ones((3, 2, 1))
    d_band = np.stack([np.eye(3), np.diag([-1.0, 1.0, 1.0])], axis=-1)
    even = np.stack([np.eye(2), np.eye(2)], axis=-1)
    sym, windows = _toy(d_band, even, [0])
    with pytest.raises(ValueError, match="carry an irrep the trial orbitals do not"):
        frozen_blocks(sym, windows, A, 2)
    sym, windows = _toy(d_band, even, [1])          # the even band 1 frozen: allowed
    assert frozen_blocks(sym, windows, A, 2)[0].nfroz == 1
    sym, windows = _toy(np.stack([np.eye(3), np.eye(3)], axis=-1),
                        np.stack([np.eye(2), np.diag([1.0, -1.0])], axis=-1), [0])
    with pytest.raises(ValueError, match="bands left free"):
        frozen_blocks(sym, windows, A, 2)
    odd = np.array([1.0, -1.0]).reshape(1, 1, 2)
    rng = np.random.default_rng(0)
    assert intertwiner_rank(odd, np.ones((1, 1, 2)), rng, [0, 1])[0] == 0
    assert intertwiner_rank(odd, odd, rng, [0, 1])[0] == 1


def test_a_frozen_set_that_differs_across_a_star_is_refused():
    """Two k-points of one star (an operation sends 0 to 1): band 0 frozen at
    one and not at the other, which an edge at round-off from a band gives."""
    from defuqumat.wannier.disentangle import windows_from_masks
    from defuqumat.wannier.sitesym import SymmetryData, check_frozen_star

    nb, nw = 3, 2
    eig = np.tile(np.arange(nb, dtype=float)[:, None], (1, 2))
    frozen = np.zeros((nb, 2), dtype=bool)
    frozen[0, 0] = True
    windows = windows_from_masks(eig, nw, np.ones((nb, 2), dtype=bool), frozen)
    d = np.stack([np.eye(nb), np.eye(nb)], axis=-1)[:, :, :, None].astype(complex)
    D = np.stack([np.eye(nw), np.eye(nw)], axis=-1)[:, :, :, None].astype(complex)
    sym = SymmetryData(nsymmetry=2, nkptirr=1, num_kpts=2, ik2ir=np.zeros(2, dtype=np.int64),
                       ir2ik=np.array([0]), kptsym=np.array([[0], [1]]), d_matrix_wann=D,
                       d_matrix_band=d)
    with pytest.raises(ValueError, match="frozen set holds different bands at k and g k"):
        check_frozen_star(windows, sym)


def test_the_automatic_selection_with_symmetry_is_the_same_on_a_star():
    """``select_states`` with ``sym`` decides at the representative and copies:
    a band 1e-12 below the frozen edge at k and 1e-12 above it at g k is frozen
    at both (the plain selection splits the star), and a pair the little group
    mixes is one block although it is 1e-3 apart in energy."""
    from defuqumat.wannier.pao import select_states
    from defuqumat.wannier.sitesym import SymmetryData

    edge = 1.0
    energies = np.array([[0.0, edge - 1e-12, 2.0, 2.001, 3.0, 4.0],
                         [0.0, edge + 1e-12, 2.0, 2.001, 3.0, 4.0]])
    p = np.array([[0.99, 0.5, 0.97, 0.90, 0.5, 0.5]] * 2)
    kw = dict(num_wann=4, proj_min=0.01, proj_max=0.95, froz_max=edge, tol=1e-6)
    _, plain = select_states(energies, p, **kw)
    assert plain[1, 0] and not plain[1, 1] and plain[2, 0] and not plain[3, 0]
    swap = np.eye(6)
    swap[2:4, 2:4] = [[0.0, 1.0], [1.0, 0.0]]
    # identity and the swap keep k = 0; the third operation sends it to k = 1
    d = np.stack([np.eye(6), swap, np.eye(6)], axis=-1)[:, :, :, None].astype(complex)
    D = np.stack([np.eye(4)] * 3, axis=-1)[:, :, :, None].astype(complex)
    sym = SymmetryData(nsymmetry=3, nkptirr=1, num_kpts=2, ik2ir=np.zeros(2, dtype=np.int64),
                       ir2ik=np.array([0]), kptsym=np.array([[0], [0], [1]]), d_matrix_wann=D,
                       d_matrix_band=d)
    outer, frozen = select_states(energies, p, sym=sym, **kw)
    assert np.array_equal(frozen[:, 0], frozen[:, 1]) and frozen[1].all()
    # 2 and 3 are one block for d: mean p 0.935 < 0.95, so neither is frozen
    assert not frozen[2].any() and not frozen[3].any() and outer[2].all() and outer[3].all()


def test_two_shells_of_one_l_on_one_atom_are_a_representation():
    """The fit of D is blocked by radial part: a 3S beside a 4S on one atom each
    map onto themselves (a fit over both gave sum_j W_jn^2 = 2, refused)."""
    from defuqumat.wannier.dmatrix import Operations, wannier_rotations
    from defuqumat.wannier.pao import PAOProjection

    site = np.zeros(3)
    shells = [PAOProjection(0, 0, 0, "3S", 3, 0, 1, site), PAOProjection(0, 0, 1, "4S", 4, 0, 1, site)]
    shells += [PAOProjection(0, 0, 2, "3P", 3, 1, m, site) for m in (1, 2, 3)]
    c3 = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])
    ops = Operations(rotations=np.stack([np.eye(3), c3]), translations=np.zeros((2, 3)))
    W, _ = wannier_rotations(shells, ops, np.eye(3))
    assert np.allclose(W[0], np.eye(5), atol=1e-10)
    assert np.allclose(W[1][:2, :2], np.eye(2), atol=1e-10)
    assert np.allclose(np.abs(W[1][2:, 2:]).sum(axis=0), 1.0, atol=1e-10)


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
    then changes nothing at all; the default (``a``) says nothing about it."""
    from defuqumat import Calculator

    calc = Calculator.from_text(_si_text(", nosym=.true."), pseudo_dir=PSEUDO)
    calc.get_scf()
    a = calc.get_wannier(4, SI_BONDS, divisions=(2, 2, 2), engine_options=dict(num_iter=20))
    b = calc.get_wannier(4, SI_BONDS, divisions=(2, 2, 2), symmetry=True,
                         engine_options=dict(num_iter=20))
    assert (b.nsym, b.site_symmetry) == (1, None)
    assert (a.nsym, a.site_symmetry, a.symmetry_fallback) == (1, None, None)
    assert np.array_equal(a.spreads, b.spreads)
    assert np.array_equal(a.hoppings, b.hoppings)
    assert np.array_equal(a.u_matrix, b.u_matrix)


def test_what_the_site_symmetry_does_not_treat_is_refused():
    from types import SimpleNamespace

    from defuqumat import Calculator
    from defuqumat.workflows.wannier import _refuse_for_symmetry, run_wannier

    calc = Calculator.from_text(_si_text(""), pseudo_dir=PSEUDO)
    with pytest.raises(NotImplementedError, match="projection lines"):
        run_wannier(calc.system, calc.pseudos, None, num_wann=8, projections="bonds",
                    symmetry=True)
    with pytest.raises(ValueError, match="engine_options would not be read"):
        run_wannier(calc.system, calc.pseudos, None, symmetry=True,
                    engine_options=dict(dis_froz_max=6.0))
    with pytest.raises(NotImplementedError, match="exclude_bands"):
        run_wannier(calc.system, calc.pseudos, None, num_wann=8, num_bands=12,
                    projections=["Si:sp3"], symmetry=True, exclude_bands=[1])
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


@pytest.fixture(scope="module")
def silicon():
    from defuqumat import Calculator

    calc = Calculator.from_text(_si_text(""), pseudo_dir=PSEUDO)
    calc.get_scf()
    return calc


def _star_consistent(mask, sym):
    return all(np.array_equal(mask[:, sym.kptsym[s, ir]], mask[:, sym.ir2ik[ir]])
               for ir in range(sym.nkptirr) for s in range(sym.nsymmetry))


def _window_u(w, windows):
    nk = len(w.kpoints)
    lwindow = np.zeros((w.num_bands, nk), dtype=bool)
    u_opt = np.zeros((w.num_bands, w.num_wann, nk), dtype=complex)
    for k in range(nk):
        rows = windows.indxwin[:windows.ndimwin[k], k]
        lwindow[rows, k] = True
        u_opt[:len(rows), :, k] = w.u_matrix[k][rows]
    return u_opt, lwindow


def test_automatic_projections_with_symmetry_on_silicon(silicon):
    """``projections="auto"`` with ``symmetry=True`` (P164b): the selection is the
    same at every point of a star, the frozen states are exact, ``P`` is
    covariant and the constraint holds on ``A``."""
    from defuqumat.units import RY_TO_EV
    from defuqumat.wannier import disentangle as dis

    w = silicon.get_wannier(divisions=(2, 2, 2), symmetry=True,
                            engine_options=dict(num_iter=50, conv_window=3))
    sym = w.site_symmetry
    assert w.nsym == 48 and sym.d_matrix_band.shape[0] == w.num_bands
    assert _star_consistent(w.outer.T, sym) and _star_consistent(w.frozen.T, sym)
    assert w.frozen.sum() > 0
    # on the outer set: the 16th band at Gamma cuts the 30.5 eV multiplet (measured
    # 0.066 there and 0 on every other band), and the selection drops it
    err = 0.0
    for ir in range(sym.nkptirr):
        ik = sym.ir2ik[ir]
        rows = np.flatnonzero(w.outer[ik])
        for s in range(sym.nsymmetry):
            d = sym.d_matrix_band[np.ix_(rows, rows)][:, :, s, ir]
            rhs = d @ w.amn[rows, :, ik] @ sym.d_matrix_wann[:, :, s, ir].conj().T
            err = max(err, float(np.abs(w.amn[rows, :, sym.kptsym[s, ir]] - rhs).max()))
    assert err < 1e-8
    windows = dis.windows_from_masks(w.eigenvalues.T * RY_TO_EV, w.num_wann, w.outer.T,
                                     w.frozen.T)
    u_opt, lwindow = _window_u(w, windows)
    defect, cov = _frozen_checks(u_opt, windows, sym, lwindow)
    assert defect < 1e-9 and cov < 1e-8


def test_a_frozen_energy_window_with_symmetry_on_silicon(silicon):
    """``Si:sp3`` from 12 bands, outer to 17 eV, frozen to 6.4 eV, with symmetry:
    the frozen valence states are in the subspace exactly and ``P`` is covariant."""
    from defuqumat.units import RY_TO_EV
    from defuqumat.wannier import disentangle as dis

    w = silicon.get_wannier(8, ["Si:sp3"], num_bands=12, divisions=(2, 2, 2), symmetry=True,
                            dis_win=(None, 17.0), dis_froz=(None, 6.4),
                            engine_options=dict(num_iter=50, conv_window=3))
    e = w.eigenvalues.T * RY_TO_EV
    windows = dis.dis_windows(e, 8, float(e.min()), 17.0, True, float(e.min()), 6.4)
    assert windows.ndimfroz.min() >= 1
    u_opt, lwindow = _window_u(w, windows)
    defect, cov = _frozen_checks(u_opt, windows, w.site_symmetry, lwindow)
    assert defect < 1e-9 and cov < 1e-8


# --- symmetry=None, the default (P166) -----------------------------------------------------

#: What the default cannot impose, one entry per mechanism, on the 2x2x2 cell: the
#: input (decided before any state is solved), the D matrices (pz alone is not a
#: representation), the engine (twelve bands cut the triplet at 14.485 eV at Gamma),
#: and the rank refusals of a frozen set (s orbitals on the atoms and on the empty
#: tetrahedral sites carry no triplet at Gamma, so the free bands lack an irrep the
#: trial orbitals need; p orbitals with the Gamma1 band frozen, which
#: symmetrize_ukirr cannot match). Each is the exception symmetry=True raises.
_SITES_S = ["Si:s", "f=0.5,0.5,0.5:s", "f=0.75,0.75,0.75:s"]
FALLBACKS = {
    "input, precond": (dict(num_wann=8, projections=["Si:sp3"],
                            engine_options=dict(precond=True, num_iter=20)),
                       NotImplementedError, "precond together with symmetry"),
    "D matrix, pz alone": (dict(num_wann=2, projections=["Si:pz"],
                                engine_options=dict(num_iter=20)),
                           "SiteSymmetryError", "not a representation of the group"),
    "engine, a cut multiplet": (dict(num_wann=8, projections=["Si:sp3"], num_bands=12,
                                     engine_options=dict(num_iter=20)),
                                "SiteSymmetryError", "cuts a degenerate multiplet"),
    "rank, the free bands": (dict(num_wann=4, projections=_SITES_S, num_bands=8,
                                  dis_froz=(None, 6.4), engine_options=dict(num_iter=20)),
                             "SiteSymmetryError", "do not carry the irreps"),
    "rank, symmetrize_ukirr": (dict(num_wann=6, projections=["Si:p"], num_bands=8,
                                    dis_froz=(None, -4.0), engine_options=dict(num_iter=20)),
                               "SiteSymmetryError", "symmetrize_ukirr: not converged"),
}


@pytest.mark.parametrize("case", sorted(FALLBACKS))
def test_the_default_falls_back_to_the_free_run_and_says_why(silicon, case):
    """``symmetry=None`` warns with the refusal's own message and returns the free
    result bit for bit; ``symmetry=True`` still raises it."""
    import re

    from defuqumat.wannier.sitesym import SiteSymmetryError

    options, error, reason = FALLBACKS[case]
    error = SiteSymmetryError if error == "SiteSymmetryError" else error
    free = silicon.get_wannier(divisions=(2, 2, 2), symmetry=False, **options)
    with pytest.warns(UserWarning, match="without the site symmetry of the crystal") as caught:
        default = silicon.get_wannier(divisions=(2, 2, 2), **options)
    assert any(re.search(reason, str(c.message)) for c in caught)
    assert re.search(reason, default.symmetry_fallback)
    assert free.symmetry_fallback is None
    assert (default.nsym, default.site_symmetry) == (1, None)
    for name in ("u_matrix", "hoppings", "spreads", "centres"):
        assert np.array_equal(getattr(default, name), getattr(free, name)), name
    with pytest.raises(error, match=reason):
        silicon.get_wannier(divisions=(2, 2, 2), symmetry=True, **options)


@pytest.mark.parametrize("projections", ["auto", "Si:sp3"])
def test_the_default_on_silicon_is_symmetry_true(silicon, projections):
    """Where the input allows it the default imposes the group, with no warning
    about it: the result is the one of ``symmetry=True`` bit for bit."""
    import warnings

    if projections == "auto":
        options = dict(projections="auto")
    else:
        options = dict(num_wann=8, projections=[projections], num_bands=12,
                       dis_win=(None, 17.0), dis_froz=(None, 6.4))
    options.update(divisions=(2, 2, 2), engine_options=dict(num_iter=50, conv_window=3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        default = silicon.get_wannier(**options)
    assert not [c for c in caught if "site symmetry" in str(c.message)]
    explicit = silicon.get_wannier(symmetry=True, **options)
    assert (default.nsym, default.symmetry_fallback) == (48, None)
    assert default.site_symmetry is not None
    for name in ("u_matrix", "hoppings", "spreads"):
        assert np.array_equal(getattr(default, name), getattr(explicit, name)), name
    if projections == "auto":
        assert np.array_equal(default.frozen, explicit.frozen)
        assert np.array_equal(default.outer, explicit.outer)


def test_a_symmetry_that_is_not_a_switch_is_refused(silicon):
    with pytest.raises(ValueError, match="symmetry must be None, True or False"):
        silicon.get_wannier(divisions=(2, 2, 2), symmetry="yes")
    # 0 is False, as it was before the default changed (it tested truthiness)
    options = dict(num_wann=4, projections=SI_BONDS, divisions=(2, 2, 2),
                   engine_options=dict(num_iter=20))
    zero = silicon.get_wannier(symmetry=0, **options)
    assert zero.nsym == 1
    assert np.array_equal(zero.u_matrix, silicon.get_wannier(symmetry=False, **options).u_matrix)
