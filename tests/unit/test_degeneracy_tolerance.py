"""The one rule for a degeneracy tolerance, and the sites that read it (``PLAN.md`` P149).

A site's constant used to be a step a solver-split symmetry pair could fall on
either side of; the rule resolves it against the promise the eigensolver made
on the states the site reads, so the guard is a function of where the
eigenvalues came from. Every test here is a synthetic pair placed *between*
the old constant and the resolved value, the one place the two differ, since
on real states at the defaults both catch the split by two decades
(``notes/DEGENERACY-NEXT.md`` item 3, fable's review in
``plans/degeneracy-p149.md``).
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.solvers.davidson import (
    ARITHMETIC_FLOOR,
    EMPTY_ETHR_FLOOR,
    SPLIT_PER_ETHR,
    degeneracy_tolerance,
    empty_band_threshold,
    resolve_degeneracy_tol,
)
from defuqumat.workflows.nscf import fixed_density_ethr


def test_the_rule_reads_the_states_provenance():
    # Unknown provenance: B1's floor, what an SCF's empty bands carry.
    assert resolve_degeneracy_tol(1.0e-8) == EMPTY_ETHR_FLOOR
    assert resolve_degeneracy_tol(2.0e-5) == 2.0e-5
    # A fixed-density solve at a loose conv_thr raises the constant with ethr;
    # at the defaults it leaves it alone.
    loose = fixed_density_ethr(1.0e-4, 8.0)          # 1.25e-6
    assert resolve_degeneracy_tol(1.0e-8, loose) == pytest.approx(SPLIT_PER_ETHR * loose)
    tight = fixed_density_ethr(1.0e-10, 8.0)         # 1.25e-12
    assert resolve_degeneracy_tol(1.0e-8, tight) == 1.0e-8
    assert resolve_degeneracy_tol(1.0e-6, tight) == 1.0e-6
    # ...but never below the arithmetic floor, B1's own defect.
    assert resolve_degeneracy_tol(5.56e-13, tight) == ARITHMETIC_FLOOR
    # An SCF's states carry the empty-band floor on top.
    assert resolve_degeneracy_tol(1.0e-6, 1.0e-10, empty_floor=True) == empty_band_threshold(1.0e-10)


def test_fixed_density_ethr_is_setup_f90s_rule():
    assert fixed_density_ethr(1.0e-6, 8.0) == pytest.approx(1.25e-8)
    assert fixed_density_ethr(1.0, 8.0) == pytest.approx(1.0e-3)      # the 1e-2 cap
    assert fixed_density_ethr(1.0e-14, 8.0) == 1.0e-13                 # ETHR_MIN


def test_an_explicit_number_is_honoured_and_warned_about_below_the_promise():
    # None resolves.
    assert degeneracy_tolerance(None, 1.0e-8) == EMPTY_ETHR_FLOOR
    # A number above the promise is simply the number, silently.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert degeneracy_tolerance(2.0e-5, 1.0e-8) == 2.0e-5
    # Below it, the number still, with the warning naming both (B1's 42-pair
    # reading at 5.56e-13 is a measurement a silently raised knob could not make).
    with pytest.warns(UserWarning, match="5.56e-13.*below.*1e-05"):
        assert degeneracy_tolerance(5.56e-13, 1.0e-8, site="a test") == 5.56e-13


def test_the_transport_block_decision_reads_the_resolved_tolerance():
    from defuqumat.transport.green import DEGENERACY_TOL, multiplets

    # An SCF's empty band in the window, split from its symmetry partner by
    # 3e-6, inside what the 1e-5 floor promises and above the old 1e-6: one
    # block under the resolved tolerance, two under the old constant.
    levels = np.array([0.1, 0.1 + 3.0e-6, 0.4])
    old = multiplets(levels, DEGENERACY_TOL)
    assert [len(b) for b in old] == [1, 1, 1] and DEGENERACY_TOL == 1.0e-6
    resolved = resolve_degeneracy_tol(DEGENERACY_TOL, 1.0e-10, empty_floor=True)
    assert resolved == EMPTY_ETHR_FLOOR
    assert [len(b) for b in multiplets(levels, resolved)] == [2, 1]
    # Fixed-density states at the default conv_thr: the constant, unchanged.
    assert resolve_degeneracy_tol(DEGENERACY_TOL, fixed_density_ethr(1.0e-6, 8.0)) == DEGENERACY_TOL


def test_the_photocurrent_cut_warning_fires_at_the_resolved_tolerance():
    from defuqumat.response.photocurrent import DEGENERACY_TOL, _warn_about_the_cut

    # A band set cut inside a multiplet the frozen solve at conv_thr = 1e-4
    # left split by 3e-8 (0.02 ethr there): silent under the constant, and
    # the warning under the promise.
    ethr = fixed_density_ethr(1.0e-4, 8.0)
    tol = resolve_degeneracy_tol(DEGENERACY_TOL, ethr)
    assert DEGENERACY_TOL < 3.0e-8 < tol
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_about_the_cut(3.0e-8, "a test tensor", "nothing")
    with pytest.warns(RuntimeWarning, match="cut inside a degenerate multiplet"):
        _warn_about_the_cut(3.0e-8, "a test tensor", "nothing", tol=tol)


def test_the_kubo_per_band_curvature_drops_a_pair_inside_the_promise():
    from defuqumat.topology.kubo import DEGENERACY_TOL, kubo_from_matrices

    # Two bands split by 3e-6 at one k-point, with a velocity coupling them:
    # the 1/gap^2 weight of the per-band curvature is 1e11 under the old
    # constant and nothing under the tolerance resolved at conv_thr = 1e-4
    # (6.25e-6); the total, occupied/empty only, is the same either way.
    rng = np.random.default_rng(3)
    e = jnp.asarray([[0.0, 3.0e-6, 1.0]])
    v = rng.normal(size=(2, 1, 3, 3)) + 1j * rng.normal(size=(2, 1, 3, 3))
    v = jnp.asarray(v + np.conj(np.swapaxes(v, -1, -2)))
    ds = jnp.zeros_like(v[0])        # norm-conserving: S = 1, dS/dk = 0
    resolved = resolve_degeneracy_tol(DEGENERACY_TOL, fixed_density_ethr(1.0e-4, 8.0))
    assert DEGENERACY_TOL < 3.0e-6 < resolved
    total_old, by_band_old = kubo_from_matrices(v[0], ds, v[1], ds, e, nocc=2,
                                                degeneracy_tol=DEGENERACY_TOL)
    total_new, by_band_new = kubo_from_matrices(v[0], ds, v[1], ds, e, nocc=2,
                                                degeneracy_tol=resolved)
    assert float(jnp.max(jnp.abs(by_band_old))) > 1.0e9
    assert np.allclose(np.asarray(total_old), np.asarray(total_new))
    assert float(jnp.max(jnp.abs(by_band_new))) < 1.0e3


def test_the_plane_wave_states_carry_their_threshold_through_a_selection():
    from defuqumat.topology.states import PlaneWaveStates

    assert "ethr" in PlaneWaveStates.__dataclass_fields__
    assert PlaneWaveStates.__dataclass_fields__["ethr"].default is None


def test_the_option_is_shared_and_named_at_every_entry_point():
    import inspect

    from defuqumat.calculator import SHARED_OPTIONS
    from defuqumat.scf.driver import run_scf
    from defuqumat.ultracell.driver import run_ultracell
    from defuqumat.workflows.conductivity import run_conductivity
    from defuqumat.workflows.photocurrent import run_shift_current
    from defuqumat.workflows.shg import run_shg
    from defuqumat.workflows.topology import run_berry_curvature
    from defuqumat.workflows.transport import run_vertical_transport
    from defuqumat.workflows.ultracell import run_ultracell_transport

    assert "degeneracy_tol" in SHARED_OPTIONS
    for entry in (run_scf, run_conductivity, run_shift_current, run_shg,
                  run_berry_curvature, run_vertical_transport,
                  run_ultracell_transport, run_ultracell):
        assert "degeneracy_tol" in inspect.signature(entry).parameters, entry.__name__
