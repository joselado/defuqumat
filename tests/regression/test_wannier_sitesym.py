"""Symmetry-adapted Wannier functions of silicon against ``pw.x`` + ``pw2wannier90.x``
(``write_dmn``) + ``wannier90.x`` (``site_symmetry``). Written from the numbers measured
on D22 on 2026-11-05 (``PLAN.md``, P164), where each ran as a script; not run as tests.

Slow: one SCF and one fixed-density solve on the 4x4x4 mesh per case. ``si.scf``,
12 bands to 8, an outer window to 17 eV and no frozen window (refused with
symmetry, as in Wannier90), 48 operations, 24 with a fractional translation.

- ``Si:sp3``: ours 15.773696950 A^2 against the QE chain's 15.773697207;
- the turned set: ours 13.787202775 against 13.787202895, eight equal spreads;
- ``H(gR) = W H(R + ...) W^T`` on the 4^3 box, every operation: 1.4e-13 eV with
  symmetry, 2.4e-2 eV without (``Si:sp3``);
- the interpolated bands at 12 random k and their 48 images: 7.5e-9 eV with
  symmetry, 1.2e-6 eV without.
"""

from pathlib import Path

import numpy as np
import pytest

import defuqumat  # noqa: F401
from defuqumat.units import RY_TO_EV

pytestmark = [pytest.mark.slow, pytest.mark.regression]

DATA = Path(__file__).resolve().parents[1] / "data" / "wannier"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"
SETS = {"sp3": (["Si:sp3"], 15.773697207),
        "turned": (["f=-0.25,0.75,-0.25:sp3", "f=0,0,0:sp3:z=0,0,1:x=0,1,0"], 13.787202895)}


@pytest.fixture(scope="module")
def silicon():
    from defuqumat import Calculator

    calc = Calculator.from_file(DATA / "si" / "si.scf", pseudo_dir=PSEUDO)
    calc.get_scf()
    return calc


def _wannier(calc, proj, symmetry):
    return calc.get_wannier(8, proj, num_bands=12, divisions=(4, 4, 4), dis_win=(None, 17.0),
                            symmetry=symmetry,
                            engine_options=dict(num_iter=200, conv_window=3, dis_num_iter=200,
                                                conv_tol=1e-10, dis_conv_tol=1e-10))


def _hopping_error(calc, w, proj):
    """max over g and R of |H(R) - sum W H(gR + T_n - T_m) W^T| on the 4^3 box, eV."""
    from defuqumat.wannier.dmatrix import operations_of, wannier_rotations
    from defuqumat.wannier.projections import apply_projection_selection, parse_projections

    system = calc.system
    at = np.asarray(system.cell.at, float)
    bg = np.asarray(system.cell.bg, float)
    ops = operations_of(system.cell, system.symmetry_group(nosym=system.nosym))
    crystal = np.asarray(system.structure.positions_crystal(system.cell), float)
    trial = apply_projection_selection(parse_projections(proj, list(system.structure.symbols),
                                                         crystal.T, bg), 8, None)
    W, T = wannier_rotations(trial, ops, at)
    hk = np.einsum("kbm,kb,kbn->kmn", w.u_matrix.conj(), w.eigenvalues * RY_TO_EV, w.u_matrix)
    rs = np.array([(a, b, c) for a in range(4) for b in range(4) for c in range(4)])
    h = np.einsum("rk,kmn->rmn", np.exp(-2j * np.pi * rs @ w.kpoints.T), hk) / len(w.kpoints)
    index = {tuple(r): i for i, r in enumerate(rs)}
    inv_at = np.linalg.inv(at)
    worst = 0.0
    for s in range(ops.nsym):
        rot = np.rint(at @ ops.rotations[s].T @ inv_at).astype(int)
        tc = np.rint(T[s] @ inv_at).astype(int)
        support = [np.flatnonzero(np.abs(W[s, :, n]) > 1e-8) for n in range(8)]
        for ir, r in enumerate(rs):
            gr = r @ rot
            rhs = np.zeros((8, 8), complex)
            for m in range(8):
                for n in range(8):
                    for j in support[m]:
                        for i in support[n]:
                            key = tuple(np.mod(gr + tc[n] - tc[m], 4))
                            rhs[m, n] += W[s, j, m] * W[s, i, n] * h[index[key], j, i]
            worst = max(worst, float(np.abs(h[ir] - rhs).max()))
    return worst


@pytest.mark.parametrize("name", sorted(SETS))
def test_silicon_with_symmetry_is_the_qe_chains(silicon, name):
    proj, total = SETS[name]
    w = _wannier(silicon, proj, True)
    assert w.nsym == 48
    assert abs(w.spread_terms_angstrom2()["omega_total"] - total) < 1e-6
    assert abs(w.spread_terms_angstrom2()["omega_i"] - 11.642688389) < 1e-6
    if name == "turned":
        assert np.ptp(w.spreads_angstrom2) < 1e-7


def test_the_hoppings_and_the_bands_carry_the_group_only_with_symmetry(silicon):
    proj = SETS["sp3"][0]
    free, sym = _wannier(silicon, proj, False), _wannier(silicon, proj, True)
    assert _hopping_error(silicon, sym, proj) < 1e-10
    assert _hopping_error(silicon, free, proj) > 1e-3   # the check can fail
    rng = np.random.default_rng(7)
    ks = rng.random((12, 3))
    ops = silicon.system.symmetry_group(nosym=False)
    bg = np.asarray(silicon.system.cell.bg, float)
    from defuqumat.wannier.dmatrix import operations_of
    cart = operations_of(silicon.system.cell, ops).rotations

    def spread_over_images(w):
        e0 = np.sort(w.bands_ev(ks), axis=1)
        return max(np.abs(np.sort(w.bands_ev((ks @ bg @ R.T) @ np.linalg.inv(bg)), axis=1)
                          - e0).max() for R in cart)

    assert spread_over_images(sym) < 1e-7
    assert spread_over_images(free) > 1e-7
