"""Wannier functions without chosen trial orbitals: projectability disentanglement.

The reference is ``pw2wannier90.x`` with ``atom_proj = .true.`` and
``atom_proj_ortho = .true.`` (QE 7.5) on the inputs in
``tests/data/wannier/si-pdwf/``: its ``.amn`` gives ``p_mk = sum_n |A_mn|^2``
on the Loewdin-orthonormalised pseudo-atomic orbitals, stored as
``p_atom_proj.dat``. That code builds its orbitals with QE's own harmonics and
phase and these use Wannier90's, so only gauge-free numbers are compared: ``p``
averaged over each degenerate multiplet. The engine half is checked on
``pw2wannier90.x``'s silicon matrices of P162: a set given band by band must give
what the energy windows give, and a hole in the outer set must give what removing
that band from the files gives.
"""

from pathlib import Path

import numpy as np
import pytest

import defuqumat  # noqa: F401
from defuqumat.units import BOHR_TO_ANGSTROM, RY_TO_EV
from defuqumat.wannier import files
from defuqumat.wannier.engine import EngineOptions, absolute_u_opt, kmesh_get, run_engine
from defuqumat.wannier.pao import (
    block_average,
    pao_projections,
    select_states,
    semicore_candidates,
    semicore_cut,
)

pytestmark = pytest.mark.unit

DATA = Path(__file__).resolve().parents[1] / "data" / "wannier"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _si_reference():
    text = (DATA / "si" / "si.win").read_text().lower()
    rows = [l for l in text.split("begin unit_cell_cart")[1].split("end unit_cell_cart")[0].split("\n")
            if l.strip() and l.strip() != "bohr"]
    at = np.array([[float(x) for x in l.split()] for l in rows])
    lines = text.split("begin kpoints")[1].split("end kpoints")[0].split("\n")
    kpt = np.array([[float(x) for x in l.split()[:3]] for l in lines if l.strip()])
    kmesh = kmesh_get(kpt.T, 2.0 * np.pi * np.linalg.inv(at * BOHR_TO_ANGSTROM).T)
    mmn = files.read_mmn(DATA / "si" / "si.mmn.gz", 12, 64, kmesh.nnlist, kmesh.nncell)
    amn = files.read_amn(DATA / "si" / "si.amn.gz", 12, 64, 8)
    eig = files.read_eig(DATA / "si" / "si.eig", 12, 64)
    return kmesh, mmn, amn, eig


OPTIONS = dict(dis_num_iter=200, num_iter=200, conv_window=3)


def test_masks_give_what_the_energy_windows_give():
    """The same outer and frozen sets, once as windows and once as masks."""
    kmesh, mmn, amn, eig = _si_reference()
    windows = run_engine(mmn, amn, eig, kmesh, 8,
                         EngineOptions(dis_win_min=-3.0, dis_win_max=17.0, dis_froz_max=6.4,
                                       **OPTIONS))
    outer = (eig >= -3.0) & (eig <= 17.0)
    frozen = outer & (eig <= 6.4)
    masks = run_engine(mmn, amn, eig, kmesh, 8, EngineOptions(**OPTIONS), outer=outer,
                       frozen=frozen)
    assert masks.terms.om_tot == windows.terms.om_tot
    assert np.array_equal(masks.lwindow, windows.lwindow)


def test_a_hole_in_the_outer_set_is_the_band_taken_out():
    """Band 11 of 12 left out of the outer set at every k is band 11 removed from
    ``M``, ``A`` and the eigenvalues: the gathers of the five places that took a
    band range are checked against an input that never had the band."""
    kmesh, mmn, amn, eig = _si_reference()
    keep = np.array([b for b in range(12) if b != 10])
    outer = np.ones_like(eig, dtype=bool)
    outer[10] = False
    frozen = eig <= 6.4
    hole = run_engine(mmn, amn, eig, kmesh, 8, EngineOptions(**OPTIONS), outer=outer,
                      frozen=frozen)
    removed = run_engine(mmn[np.ix_(keep, keep)], amn[keep], eig[keep], kmesh, 8,
                         EngineOptions(dis_froz_max=6.4, **OPTIONS))
    assert abs(hole.terms.om_tot - removed.terms.om_tot) < 1e-10
    assert abs(hole.terms.om_i - removed.terms.om_i) < 1e-10
    assert not hole.lwindow[10].any()
    v_hole = np.einsum("bwk,wnk->kbn", absolute_u_opt(hole), hole.u_matrix)
    v_removed = np.einsum("bwk,wnk->kbn", absolute_u_opt(removed), removed.u_matrix)
    assert np.allclose(v_hole[:, keep], v_removed, atol=1e-8)
    assert np.all(v_hole[:, 10] == 0.0)


def test_a_hole_that_moves_with_k_keeps_the_frozen_states():
    """Band 10 left out at even k and band 11 at odd k: the frozen eigenvalues are
    still exactly those of ``V^H E V`` at each mesh point, and nothing left out is
    in the window."""
    kmesh, mmn, amn, eig = _si_reference()
    outer = np.ones_like(eig, dtype=bool)
    outer[9, 0::2] = False
    outer[10, 1::2] = False
    frozen = eig <= 6.4
    r = run_engine(mmn, amn, eig, kmesh, 8, EngineOptions(**OPTIONS), outer=outer, frozen=frozen)
    assert np.array_equal(r.lwindow, outer)
    v = np.einsum("bwk,wnk->kbn", absolute_u_opt(r), r.u_matrix)
    for k in range(64):
        bands = np.linalg.eigvalsh(v[k].conj().T @ np.diag(eig[:, k]) @ v[k])
        assert np.min(np.abs(bands[:, None] - eig[frozen[:, k], k][None, :]), axis=0).max() < 1e-8


def test_a_frozen_state_outside_the_outer_set_is_refused():
    kmesh, mmn, amn, eig = _si_reference()
    outer = np.ones_like(eig, dtype=bool)
    outer[0, 5] = False
    with pytest.raises(ValueError, match="outside the outer set at k-point 5"):
        run_engine(mmn, amn, eig, kmesh, 8, EngineOptions(**OPTIONS), outer=outer,
                   frozen=eig <= 6.4)


def test_the_selection_drops_freezes_and_refuses():
    energies = np.array([[0.0, 1.0, 2.0, 3.0, 4.0, 4.0]])
    p = np.array([[1.0, 0.5, 0.005, 0.99, 0.99, 0.90]])
    outer, frozen = select_states(energies, p, num_wann=4, proj_min=0.01, proj_max=0.95,
                                  froz_max=None, tol=1e-6)
    # the state with p below proj_min is dropped, the others kept
    assert outer[:, 0].tolist() == [True, True, False, True, True, True]
    # p at 4.0 is 0.99 and 0.90 on one doublet: its mean 0.945 freezes neither
    assert frozen[:, 0].tolist() == [True, False, False, True, False, False]
    # the energy edge freezes everything at or below it, whatever its p
    _, frozen = select_states(energies, p, num_wann=4, proj_min=0.01, proj_max=0.95,
                              froz_max=2.5, tol=1e-6)
    assert frozen[:, 0].tolist() == [True, True, True, True, False, False]
    with pytest.raises(ValueError, match=r"6 states are frozen at k-point 0 \(0.1000"):
        select_states(energies, p, num_wann=4, proj_min=0.01, proj_max=0.95, froz_max=4.5,
                      tol=1e-6, kpoints=np.array([[0.1, 0.2, 0.3]]))


def test_a_multiplet_cut_by_the_top_band_is_dropped_whole():
    energies = np.array([[0.0, 1.0, 2.0, 3.0]])
    p = np.array([[1.0, 0.9, 0.8, 0.7]])
    outer, _ = select_states(energies, p, num_wann=3, proj_min=0.01, proj_max=0.95,
                             froz_max=None, tol=1e-6, energy_above=np.array([3.0]))
    assert outer[:, 0].tolist() == [True, True, True, False]
    with pytest.raises(ValueError, match="cut by the top"):
        select_states(energies, p, num_wann=4, proj_min=0.01, proj_max=0.95, froz_max=5.0,
                      tol=1e-6, energy_above=np.array([3.0]))


def test_the_block_average_is_over_multiplets():
    energies = np.array([[0.0, 1.0, 1.0 + 1e-9, 2.0]])
    values = np.array([[0.2, 0.4, 0.8, 1.0]])
    assert np.allclose(block_average(values, energies, 1e-6), [[0.2, 0.6, 0.6, 1.0]])


def _pseudo(name):
    from defuqumat.pseudo.upf import read_upf

    return read_upf(PSEUDO / name)


def test_the_semicore_candidates_are_read_from_the_labels():
    """Cr's 3S has a 4S partner; its 3P has none; silicon has no semicore."""
    assert semicore_candidates(_pseudo("Cr.pbe-spn-rrkjus_psl.1.0.0.UPF")) == {0}
    assert semicore_candidates(_pseudo("Si.pz-vbc.UPF")) == set()


def test_the_semicore_cut_fires_only_with_a_gap():
    rng = np.random.default_rng(1)
    nk, nb = 3, 4
    amn = np.zeros((nb, 2, nk), dtype=complex)
    amn[0, 0, :] = 1.0                     # band 0 is all orbital 0, the semicore one
    amn[1:, 1, :] = rng.random((nb - 1, nk)) * 0.5
    below = np.array([[-50.0, 0.0, 1.0, 2.0]] * nk)
    assert semicore_cut(below, amn, [0], 1e-6)[0] == 1
    assert semicore_cut(below, amn, [0], 1e-6, mode=False)[0] == 0
    touching = below.copy()
    touching[0, 0] = -1.0                  # above band 1 at another k: no gap
    touching[1, 1] = -2.0
    n, note = semicore_cut(touching, amn, [0], 1e-6)
    assert n == 0 and "overlaps the valence" in note
    assert semicore_cut(touching, amn, [0], 1e-6, mode=True)[0] == 1
    mixed = amn.copy()
    mixed[0, 0, 1] = 0.3                   # band 0 is not semicore at k = 1
    with pytest.warns(UserWarning, match="not the lowest"):
        assert semicore_cut(below, mixed, [0], 1e-6)[0] == 0
    with pytest.raises(ValueError, match="semicore=True cannot exclude"):
        semicore_cut(below, mixed, [0], 1e-6, mode=True)


def test_a_dataset_without_orbitals_is_refused():
    from defuqumat import Calculator

    calc = Calculator.from_text(_text("Cr 51.996 Cr.pbe-nc-sg15.UPF", "Cr"), pseudo_dir=PSEUDO)
    crystal = np.zeros((1, 3))
    with pytest.raises(ValueError, match="carries no pseudo-atomic orbital"):
        pao_projections(calc.pseudos, calc.system.structure, crystal)


def _text(species, symbol, extra=""):
    return f"""&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=10.2, nat=1, ntyp=1, ecutwfc=12.0 {extra}
/
&electrons
/
ATOMIC_SPECIES
 {species}
ATOMIC_POSITIONS {{crystal}}
{symbol} 0.00 0.00 0.00
K_POINTS {{automatic}}
 2 2 2 0 0 0
"""


def test_auto_refuses_windows_and_gamma():
    from defuqumat import Calculator
    from defuqumat.workflows.wannier import run_wannier

    calc = Calculator.from_text(_text("Si 28.086 Si.pz-vbc.UPF", "Si"), pseudo_dir=PSEUDO)
    with pytest.raises(ValueError, match="dis_win and dis_froz are for given projections"):
        run_wannier(calc.system, calc.pseudos, None, dis_win=(None, 10.0))
    with pytest.raises(ValueError, match="sets exclude_bands by the semicore cut"):
        run_wannier(calc.system, calc.pseudos, None, exclude_bands=[1])
    gamma = Calculator.from_text(_text("Si 28.086 Si.pz-vbc.UPF", "Si").replace(
        "K_POINTS {automatic}\n 2 2 2 0 0 0", "K_POINTS gamma"), pseudo_dir=PSEUDO)
    with pytest.raises(NotImplementedError, match="K_POINTS gamma"):
        run_wannier(gamma.system, gamma.pseudos, None)


@pytest.fixture(scope="module")
def silicon():
    from defuqumat import Calculator

    calc = Calculator.from_file(DATA / "si-pdwf" / "scf.in", pseudo_dir=PSEUDO)
    calc.get_scf()
    w = calc.get_wannier(divisions=(4, 4, 4), engine_options=dict(num_iter=100, conv_window=3))
    return calc, w


def test_the_projectability_is_pw2wannier90s(silicon):
    """Measured 1.06e-6 at most and 1.7e-7 on average against ``atom_proj``; the
    eigenvalues of the two SCFs differ by 1.8e-5 eV."""
    _, w = silicon
    reference = np.loadtxt(DATA / "si-pdwf" / "p_atom_proj.dat")
    eig = files.read_eig(DATA / "si-pdwf" / "w.eig", 16, 64).T
    assert np.max(np.abs(w.eigenvalues * RY_TO_EV - eig)) < 1e-4
    ours = block_average(w.projectability, eig, 1e-4)
    theirs = block_average(reference, eig, 1e-4)
    assert np.max(np.abs(ours - theirs)) < 3e-6
    assert w.projectability.max() <= 1.0 + 1e-12


def test_the_automatic_set_is_the_one_claimed(silicon):
    """Eight functions, s and p on each atom; the frozen states are reproduced
    exactly on the mesh; a dropped state is not in the outer set."""
    _, w = silicon
    assert w.num_wann == 8 and w.num_bands == 16 and w.excluded_bands == ()
    assert w.projection_labels[:4] == ("Si1 3S:s", "Si1 3P:pz", "Si1 3P:px", "Si1 3P:py")
    edge = w.edge * RY_TO_EV
    e = w.eigenvalues * RY_TO_EV
    p = block_average(w.projectability, e, 1e-4)
    assert np.all(w.frozen[e <= edge + 2.0 - 1e-6])
    assert np.all(w.frozen[p >= 0.95])
    dropped = (p < 0.01) & (e > edge + 2.0)
    assert dropped.any() and not w.outer[dropped].any()
    mesh = w.bands_ev(w.kpoints)
    for k in range(len(w.kpoints)):
        frozen = e[k, w.frozen[k]]
        assert np.min(np.abs(mesh[k][:, None] - frozen[None, :]), axis=0).max() < 1e-8


def test_the_automatic_masks_reduce_to_the_energy_windows(silicon):
    """With no projectability threshold the masks are the windows: the same
    matrices through both engine paths give one spread."""
    from defuqumat.wannier.engine import EngineOptions, run_engine
    from defuqumat.workflows.wannier import uniform_mesh

    calc, w = silicon
    e = (w.eigenvalues * RY_TO_EV).T
    froz = w.edge * RY_TO_EV + 2.0
    bg = np.asarray(calc.system.cell.bg) / BOHR_TO_ANGSTROM
    kmesh = kmesh_get(uniform_mesh((4, 4, 4)).T, bg)
    windows = run_engine(w.mmn, w.amn, e, kmesh, 8,
                         EngineOptions(dis_froz_max=froz, num_iter=50, conv_window=3))
    masks = run_engine(w.mmn, w.amn, e, kmesh, 8, EngineOptions(num_iter=50, conv_window=3),
                       outer=np.ones_like(e, dtype=bool), frozen=e <= froz)
    assert masks.terms.om_tot == windows.terms.om_tot
