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
