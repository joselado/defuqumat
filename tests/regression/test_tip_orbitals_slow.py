"""The orbital of the tip on converged sheets: what needs an SCF.

``PLAN.md`` P179. Each test is a probe that was run on this workstation on
2026-11-15 (``plans/tip-orbitals-review/`` has the record), with the numbers it
read in the docstring; the algebra is ``tests/unit/test_tip_orbitals.py``.

* V6, the sign of a hybrid on a real vacuum tail: ``s - pz`` points its lobe at
  the sample, in both geometries (tip above and below).
* V9, a symmetry-reduced k-set against the whole grid, with the orbital turned
  by each operation, on graphene's ``D6h`` -- and the same with the rotation
  transposed, which must fail.
* V3 on the real code paths: the map integrated over the tip plane is the
  momentum route's sum, raw and continued.
* Graphene's contrast: the hollow is dark for ``s`` and ``pz`` by symmetry and
  brighter than the atoms for ``pxy``; with the vacuum continuation the ratio
  ``pz/s`` over an atom settles at ``(kappa_K / kappa)^2``, and the WKB
  continuation meets the raw tail where the raw tail is still good.

Peaks measured: graphene's SCF and two 6x6x1 band solves are about a minute
each on four cores here, under 2 GB.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.transport.substrate import surface_area
import defuqumat.workflows.transport as transport
from defuqumat.workflows.nscf import denser_grid
from defuqumat.workflows.transport import run_vertical_transport, whole_grid_states

pytestmark = pytest.mark.slow

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"
GRAPHENE_C = 20.0          # bohr, celldm(3) * celldm(1) of graphene-monolayer.in


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(case: str):
    calculator = Calculator.from_file(CASES / f"{case}.in", pseudo_dir=PSEUDO)
    calculator.get_scf()
    return calculator


@lru_cache(maxsize=2)
def _graphene_states(nbnd: int, reduced: bool = False):
    calculator = _converged("graphene-monolayer")
    scf = calculator.get_scf()
    if reduced:
        return whole_grid_states(calculator.system, calculator.pseudos, scf,
                                 kpoints=denser_grid(calculator.system, (6, 6, 1)),
                                 nbnd=nbnd, conv_thr=1e-10,
                                 calculation=calculator.calculation)
    return whole_grid_states(calculator.system, calculator.pseudos, scf,
                             grid=(6, 6, 1), nbnd=nbnd, conv_thr=1e-10,
                             calculation=calculator.calculation)


def test_a_hybrid_with_its_lobe_at_the_sample_tunnels_more_on_both_sides():
    """V6: ``T(s - pz) / T(s + pz) > 20`` at every pixel (30.3 measured).

    The hydrogen sheet, the tip 3 bohr above it and the exit plane 3 bohr
    below, then the mirror of that geometry: the frame's z points from the
    material to the tip in each, so the ratio is the same (30.3 to 7787 in
    both) and the side recorded is +1 and -1.
    """
    calculator = _converged("h-sheet")
    options = dict(shape=(8, 8), broadening=0.03, incoherent=False)
    for exit_h, tip_h, side in ((0.3, 0.7, 1), (0.7, 0.3, -1)):
        down = calculator.get_vertical_transport(
            exit_height=exit_h, height=tip_h, tip_orbital={"s": 1, "pz": -1}, **options)
        up = calculator.get_vertical_transport(
            exit_height=exit_h, height=tip_h, tip_orbital={"s": 1, "pz": 1}, **options)
        ratio = np.asarray(down.values) / np.asarray(up.values)
        assert ratio.min() > 20.0
        assert down.notes["tip"]["side"] == side


@pytest.mark.parametrize("spec", ["px", {"s": 1.0, "px": 0.7}, {"s": 1.0, "pz": -0.5},
                                  "pxy", "dz2", {"dxy": 1.0, "s": 0.4},
                                  {"py": 1.0, "dxz": 0.5}], ids=str)
def test_the_wedge_turns_the_orbital_with_each_operation(spec):
    """V9: graphene's 7-point wedge against the 36-point grid, below 1e-6 of
    the peak, the bound the s tip's wedge test asserts; measured 3.0e-9 to
    7.3e-9 for these tips and 3.3e-9 for the s tip. A 6-fold cell, so ``g``
    and ``g^-1`` give different images.
    """
    calculator = _converged("graphene-monolayer")
    scf = calculator.get_scf()
    options = dict(exit_height=0.38, height=0.62, shape=(5, 5), broadening=0.03,
                   energies=float(scf.fermi_energy) + np.array([-0.35, -0.2, 0.15]),
                   incoherent=False, tip_orbital=spec)
    right = np.asarray(run_vertical_transport(
        calculator.system, calculator.pseudos, scf, states=_graphene_states(12),
        **options).values)
    got = np.asarray(run_vertical_transport(
        calculator.system, calculator.pseudos, scf,
        states=_graphene_states(12, reduced=True), **options).values)
    assert np.max(np.abs(got - right)) / np.max(np.abs(right)) < 1e-6


def test_the_wedge_test_fails_with_the_rotation_transposed(monkeypatch):
    """The guard: ``g`` for ``g^-1`` reads 0.30 of the peak for ``s + 0.7 px``."""
    calculator = _converged("graphene-monolayer")
    scf = calculator.get_scf()
    original = transport.rotation_representation
    monkeypatch.setattr(transport, "rotation_representation",
                        lambda basis, rotation: original(basis, np.asarray(rotation).T))
    options = dict(exit_height=0.38, height=0.62, shape=(5, 5), broadening=0.03,
                   energies=float(scf.fermi_energy) + np.array([-0.35, -0.2, 0.15]),
                   incoherent=False, tip_orbital={"s": 1.0, "px": 0.7})
    right = np.asarray(run_vertical_transport(
        calculator.system, calculator.pseudos, scf, states=_graphene_states(12),
        **options).values)
    got = np.asarray(run_vertical_transport(
        calculator.system, calculator.pseudos, scf,
        states=_graphene_states(12, reduced=True), **options).values)
    assert np.max(np.abs(got - right)) / np.max(np.abs(right)) > 1e-2


@pytest.mark.parametrize("spec, match", [("pz", None), ("s", 2.0), ("pz", 2.0),
                                         ({"s": 1, "pz": -1}, 2.0), ("pxy", 2.0)],
                         ids=str)
def test_the_map_over_the_tip_plane_is_the_momentum_sum(spec, match):
    """V3 on the real code paths, raw and continued: 6e-16 to 1.1e-15 measured.

    The hydrogen sheet on its 4x4x1 grid, the tip 4 bohr above, a 16 x 16 map,
    which integrates the trigonometric polynomial exactly at this cutoff.
    """
    calculator = _converged("h-sheet")
    c, n = 15.0, 16
    tip_h = 0.5 + 4.0 / c
    match_h = None if match is None else 0.5 + match / c
    flow = calculator.get_vertical_transport(
        exit_height=0.3, height=tip_h, shape=(n, n), grid=(4, 4, 1),
        broadening=0.03, incoherent=False, tip_orbital=spec, match_height=match_h)
    run = calculator.get_momentum_transport(
        exit_height=0.3, height=tip_h, grid=(4, 4, 1), broadening=0.03,
        tip_orbital=spec, match_height=match_h)
    area = surface_area(calculator.system.cell, 2)
    integral = np.asarray(flow.values).sum() * area / n**2
    assert integral == pytest.approx(float(np.asarray(run.weight).sum()), rel=1e-11)


def test_the_hollow_of_graphene_is_dark_for_s_and_bright_for_pxy():
    """The contrast inversion, at 2.4, 4 and 5 bohr above the sheet.

    At ``E_F`` only the Dirac states at K carry the map, and they are zero at
    the hollow by C3, so ``s``, ``pz`` and ``s - pz`` read 1e-16 of an atom
    there; their lateral gradient is not zero, so ``pxy`` reads 1.57, 1.81 and
    2.00 times an atom at the three heights (raw tail, 40 Ry).
    """
    calculator = _converged("graphene-monolayer")
    scf = calculator.get_scf()
    states = _graphene_states(20)
    for height, low in ((2.4, 1.4), (4.0, 1.6), (5.0, 1.8)):
        z = 0.5 + height / GRAPHENE_C
        sites = np.array([[0, 0, z], [1 / 3, 2 / 3, z], [2 / 3, 1 / 3, z]])
        for spec in ("s", "pz", {"s": 1, "pz": -1}):
            v = np.asarray(run_vertical_transport(
                calculator.system, calculator.pseudos, scf, states=states,
                exit_height=0.38, tip=sites, broadening=0.02, tip_orbital=spec,
                incoherent=False).values)
            assert v[2] < 1e-12 * v[0]
            assert v[1] == pytest.approx(v[0], rel=1e-3)
        v = np.asarray(run_vertical_transport(
            calculator.system, calculator.pseudos, scf, states=states,
            exit_height=0.38, tip=sites, broadening=0.02, tip_orbital="pxy",
            incoherent=False).values)
        assert low < v[2] / v[0] < 2.1


def test_the_continued_tail_follows_the_dirac_decay():
    """With ``match_height`` 3 bohr above graphene, ``pz/s`` over an atom goes to
    ``(kappa_K / kappa)^2 = 1 + |K|^2 / kappa^2`` far away: 3.41, 3.77, 3.92 at
    4, 6, 10 bohr (40 Ry), approaching it from below because the WKB decay is
    slower where the potential is below the plateau, where the raw tail read
    5.4 at 4 bohr and 49.8 at 8. The ``pxy`` ratio hollow/A goes to 2.0000.
    """
    calculator = _converged("graphene-monolayer")
    scf = calculator.get_scf()
    states = _graphene_states(20)
    cell = calculator.system.cell
    corner = np.linalg.norm(np.array([1 / 3, 1 / 3, 0.0]) @ np.asarray(cell.bg))
    gaps = []
    for height in (4.0, 6.0, 10.0):
        z = 0.5 + height / GRAPHENE_C
        sites = np.array([[0, 0, z], [2 / 3, 1 / 3, z]])
        values = {}
        for spec in ("s", "pz", "pxy"):
            flow = run_vertical_transport(
                calculator.system, calculator.pseudos, scf, states=states,
                exit_height=0.38, tip=sites, broadening=0.02, tip_orbital=spec,
                incoherent=False, match_height=0.5 + 3.0 / GRAPHENE_C)
            values[spec] = np.asarray(flow.values)
            if spec == "pz":
                kappa = flow.notes["tip"]["kappa"][0]
        analytic = 1.0 + corner**2 / kappa**2
        gaps.append(abs(values["pz"][0] / values["s"][0] - analytic))
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[2] < 2e-3 * analytic
    assert values["pxy"][1] / values["pxy"][0] == pytest.approx(2.0, rel=1e-3)


def test_the_wkb_continuation_meets_the_raw_tail_where_that_tail_is_good():
    """The check of the continuation that symmetry does not decide.

    Over an atom of graphene, 3.5 and 4 bohr up (40 Ry), the raw s map is still
    good, and the s map continued from 3 bohr through the planar average reads
    1.010 and 0.984 of it. Continued with the plateau's decay alone (a flat
    vacuum, which ``tip_kappa`` gives) it reads 0.885 and 0.778, because the
    potential above the plane is still below the plateau.
    """
    calculator = _converged("graphene-monolayer")
    scf = calculator.get_scf()
    states = _graphene_states(20)
    match = 0.5 + 3.0 / GRAPHENE_C
    for height, flat_bound in ((3.5, 0.92), (4.0, 0.85)):
        site = np.array([[0, 0, 0.5 + height / GRAPHENE_C]])
        common = dict(states=states, exit_height=0.38, tip=site, broadening=0.02,
                      incoherent=False)
        raw = float(np.asarray(run_vertical_transport(
            calculator.system, calculator.pseudos, scf, **common).values)[0])
        wkb = run_vertical_transport(calculator.system, calculator.pseudos, scf,
                                     match_height=match, **common)
        kappa = wkb.notes["tip"]["kappa"][0]
        flat = float(np.asarray(run_vertical_transport(
            calculator.system, calculator.pseudos, scf, match_height=match,
            tip_kappa=kappa, **common).values)[0])
        assert float(np.asarray(wkb.values)[0]) / raw == pytest.approx(1.0, abs=0.03)
        assert flat / raw < flat_bound
