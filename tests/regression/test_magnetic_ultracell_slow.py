"""Route U step 1 at the acceptance sizes, on the shared graphene cell (slow).

The design's acceptance cases (``plans/orbital-field/u1.md``): the free Landau problem at
``q = 10`` and graphene local-only at ``q = 6``. What these assert is what was measured on
D22 at 2026-11-07: the bare hybrid basis of an entangled band set is refused by the
Hermiticity guard and is not converged in ``N1``, and the windowed basis with the symmetric
``a.p + p.a`` is Hermitian and variational. Each case is 40 s to 2 min on one core.
"""

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _clear():
    yield
    jax.clear_caches()


def _graphene():
    from defuqumat import Calculator
    from defuqumat.units import RY_TO_EV

    return Calculator.from_keywords(
        lattice="hexagonal", a=2.46, c=15.0,
        atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
        pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
        ecut=50 * RY_TO_EV, kmesh=(12, 12, 1), smearing=0.01 * RY_TO_EV, smearing_type="mv")


def _reference(calc, p, q):
    from defuqumat.magnetic.conventions import MagneticCell
    from defuqumat.magnetic.landau import free_landau_levels

    levels, mult, _ = free_landau_levels(MagneticCell.build(calc.system.cell, p, q), 12, g3max=4)
    return np.repeat(levels, mult)


def test_the_free_problem_at_q10_on_bare_hybrid_states_is_refused():
    """Measured: max |H - H^dagger|/max|H| = 5.7e-2 at nbnd = 2 (review/mag/u1/free1.log)."""
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _graphene()
    with pytest.raises(ValueError, match="not Hermitian"):
        run_magnetic_ultracell(calc.system, calc.pseudos, None, flux=(1, 10), nbnd=2, N1=10,
                               kmesh=(2, 2), free=True)


def test_the_windowed_free_problem_at_q10_is_hermitian_and_variational():
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _graphene()
    r = run_magnetic_ultracell(calc.system, calc.pseudos, None, flux=(1, 10), nbnd=4, N1=10,
                               kmesh=(2, 2), free=True, h0="box", window=(2.0, 2.5),
                               form="symmetric")
    ref = _reference(calc, 1, 10)
    n = min(len(ref), 12)
    assert np.all(r.hermiticity < 1e-12)
    assert np.min(r.levels[:, :n] - ref[None, :n]) > -1e-8
    assert np.all(r.rank == 40)


def test_graphene_local_only_is_not_converged_in_n1_on_bare_hybrid_states():
    """Measured: the lowest 24 levels at N1 = 8 and 12 differ by 2.3e-2 Ry (q = 6, nbnd = 8,
    review/mag/u1/acc3-n8.log, acc3-n12.log), against the design's 1e-5 and its refusal at
    1e-3."""
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _graphene()
    scf = "plans/orbital-field/graphene-scf.npz"
    runs = [run_magnetic_ultracell(calc.system, calc.pseudos, scf, flux=(1, 6), nbnd=8, N1=n,
                                   kmesh=(2, 2), strict=False) for n in (8, 12)]
    diff = np.max(np.abs(runs[0].levels[:, :24] - runs[1].levels[:, :24]))
    assert diff > 1e-3
    assert all(r.w_measured is None for r in runs)


# --- step 2: the GIPAW-dressed nonlocal term (plans/orbital-field/u2.md) ----------------

SCF = "plans/orbital-field/graphene-scf.npz"


def test_zero_flux_with_the_full_hamiltonian_gives_back_the_bands():
    """Acceptance 1: p = 0 on the full basis, h0 = eigen: 1.8e-15 Ry against the zero-field
    eigenvalues (review/mag/u2/zero-full.log); the grid projection against vkb 4.4e-6 Ry."""
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _graphene()
    r = run_magnetic_ultracell(calc.system, calc.pseudos, SCF, flux=(0, 1), nbnd=4, N1=5, w=2,
                               kmesh=(1, 2), local_only=False)
    for ik, (_, k2) in enumerate(r.kappa):
        assert np.max(np.abs(r.levels[ik, :4] - r.eigenvalues[int(round(k2 * 2)), 0, 0])) < 1e-9
    assert r.nonlocal_error < 1e-5


@pytest.mark.parametrize("term, bound", [("none", (1e-6, 3e-6)), ("gipaw", (1e-6, 3e-6)),
                                         ("bare", (1e-4, 1.0))])
def test_the_dressed_term_keeps_the_hermiticity_floor_of_a_localized_basis(term, bound):
    """The isolated deep composite of the local-only basis (nbnd = 2), q = 10, N1 = 12,
    h0 = box: 1.7e-6 with no nonlocal term, 1.5e-6 with V_NL^B, 3.2e-4 with the bare V_NL^0
    (review/mag/u2/iso-{none,gipaw,bare}.log)."""
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _graphene()
    r = run_magnetic_ultracell(calc.system, calc.pseudos, SCF, flux=(1, 10), nbnd=2, N1=12,
                               kmesh=(2, 2), h0="box", nonlocal_term=term, strict=False)
    assert bound[0] < np.max(r.hermiticity) < bound[1]


def test_the_full_hamiltonian_at_q10_is_refused_by_the_hermiticity_guard():
    """Acceptance 2: 2.1e-2 to 2.3e-2 (review/mag/u2/q10-full.log), the same as the bare
    control (2.1e-2 to 3.0e-2, q10-bare.log): the basis, not the nonlocal term."""
    from defuqumat.workflows.landau_ultracell import run_magnetic_ultracell

    calc = _graphene()
    with pytest.raises(ValueError, match="not Hermitian"):
        run_magnetic_ultracell(calc.system, calc.pseudos, SCF, flux=(1, 10), nbnd=8, N1=10,
                               kmesh=(1, 1), local_only=False)
