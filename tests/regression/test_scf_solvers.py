"""The SCF as a root-find: the residual, its Jacobian action, and the solver.

Three things are checked, in the order they can go wrong:

1. **The Jacobian action.** ``ScfResidual.jvp`` differentiates one SCF step with
   ``jax.jvp``; ``jvp_finite_difference`` central-differences the same residual.
   They share no machinery, so agreement means both are right -- the same
   argument the autodiff forces and QE's transcribed ones stand on (P15).
   Where they *disagree* is itself recorded here, because it is the finding that
   decides the default backend (see ``test_autodiff_jvp_needs_a_warm_start``).
2. **The fixed point.** A different solver reaching a different answer is a bug
   in one of them, so Newton-Krylov's converged energy is compared against the
   mixing loop's on the same input, to the tolerance the QE comparison uses.
3. **The preconditioner.** Kerker changes how the mixer gets there and must not
   change where it arrives.
"""

from functools import lru_cache
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation, run_scf
from defuqumat.scf.driver import default_nbnd
from defuqumat.scf.residual import make_residual
from defuqumat.hubbard import uniform_ns
from defuqumat.system import build_system
from tests.tolerances import TOTAL_ENERGY_RY

BENCHMARKS = Path(__file__).resolve().parents[2] / "benchmarks"

#: Silicon, two atoms, one k-point -- the smallest thing that exercises the
#: whole path, and gapped, so the occupations are constant and the Jacobian is
#: the density response alone.
SILICON = BENCHMARKS / "si-1k.in"

#: fcc aluminium with Bloechl's tetrahedra on the 4x4x4 shifted grid; ``pw.x``
#: 7.5's own run is ``reference.out.al-tetrahedra``.
AL_TETRAHEDRA = Path(__file__).resolve().parents[1] / "data" / "qe" / "al-tetrahedra.in"
AL_TETRAHEDRA_REFERENCE = AL_TETRAHEDRA.with_name("reference.out.al-tetrahedra")

#: bcc iron from a *small* starting moment: two self-consistent solutions, one
#: of them unstable. See ``test_newton_krylov_reaches_an_unstable_solution``.
IRON = BENCHMARKS / "fe-unstable.in"

#: How far to kick the symmetric iron solution, in the atomic magnetization's
#: shape. The middle of a *measured* window, not a tuned number: re-measured on
#: ``D22-0161`` on 2026-10-12, after ``22842dc`` changed Davidson's collapse,
#: mixing comes back below 0.10 and runs away from 0.10 up, the
#: tightly solved Newton comes back from 0.10 to 0.14 and not at 0.15, and the
#: loosely solved one runs away from 0.10 up with Kerker, 0.12 without -- see
#: :func:`test_newton_krylov_reaches_an_unstable_solution`. It was 0.10 until
#: then, the middle of the window measured on the other workstation before
#: the collapse fix, and on D22 at that commit 0.10 was the only point of it.
KICK = 0.12

#: The inner solve that delivers the Newton direction: GMRES to a relative
#: residual of 0.1 in up to 20 iterations. This, and not the preconditioner,
#: is what returns the kicked iron to its symmetric root (measured with Kerker
#: off as well: back at 0.10 to 0.13, 0.42 at 0.14).
TIGHT_INNER_SOLVE = {"forcing": 0.1, "gmres_maxiter": 20, "max_iterations": 30}

#: The inexact one: a relative residual of 0.5 in at most 8 GMRES iterations,
#: which is what P22 ran. With Kerker on or off it reaches the ferromagnet from
#: 0.12 to 0.14 on D22, under one core and under two.
LOOSE_INNER_SOLVE = {"forcing": 0.5, "gmres_maxiter": 8, "max_iterations": 12}
IRON_NONMAGNETIC = BENCHMARKS / "fe-unstable-nonmagnetic.in"

#: fcc nickel with U = 3 eV: the DFT+U counterpart of ``IRON``, and the case
#: that needs a custom starting occupation matrix to set up at all.
NICKEL_U = BENCHMARKS / "ni-u-unstable.in"
NICKEL_U_NONMAGNETIC = BENCHMARKS / "ni-u-nonmagnetic.in"

#: An aluminium slab with vacuum: metallic, so the Fermi level moves and its
#: ``custom_jvp`` is on the path, and *inhomogeneously screened*, which is the
#: regime this solver was built for. ``PLAN.md`` P22 has the measurement.
SLAB = BENCHMARKS / "al-slab.in"


@lru_cache(maxsize=1)
def _kicked_iron():
    """The iron saddle, and it displaced by ``KICK`` along the magnetization.

    **Both tests below start here rather than from the input's own
    ``starting_magnetization``**, and that is the difference between a
    demonstration and a knife edge. From the bare 0.05 seed which root an
    *unpreconditioned* inexact Newton reaches is decided by round-off: measured
    on this machine, the same commit gives ``M = -3.4050`` with the default
    thread pool and ``M = -0.0003`` -- the saddle -- under
    ``OMP_NUM_THREADS=1``, because a different reduction order is a different
    perturbation and the seed is small enough for that to choose. From the
    kicked state, which is a *measured* distance outside the saddle's basin,
    both arithmetics give -3.405 and -3.405.

    **That was measured with the Davidson of before ``22842dc``** (2026-10-10),
    which collapsed its subspace on every step at ``david = 2``. Collapsing on
    ``cegterg``'s condition changes every evaluation of ``F`` at the level of
    ``ethr``, and at the old kick of 0.10 both Newton tests flipped on D22: the
    Kerker run reached the ferromagnet (-3.4052) and the unpreconditioned one
    the saddle (-0.0005). The kick and the inner solves were re-measured then;
    at 0.12 every column of the window agrees under one core and under two.

    Returns ``(system, pseudos, saddle, kicked)``.
    """
    system, pseudos, calculation, _ = _setup(IRON)

    # The saddle itself: a spin-symmetric density, which the SCF map preserves
    # exactly -- nothing in it breaks spin symmetry on its own.
    rho = np.asarray(calculation.starting_density())
    symmetric = jnp.asarray(
        np.repeat(rho.mean(axis=0, keepdims=True), rho.shape[0], axis=0)
    )
    saddle = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                     conv_thr=1e-8, max_iterations=200,
                     starting_density=symmetric)
    assert saddle.converged
    assert abs(float(saddle.magnetization)) < 1e-6

    # Kick it along the magnetization direction. Both the **shape** and the
    # **size** of the kick matter, and measuring them is what made this
    # reproducible -- see the docstring of the test below.
    atomic = np.asarray(calculation.starting_density())
    shape = atomic[0] - atomic[1]
    shape = shape / np.abs(shape).max()
    density = np.asarray(saddle.density)
    kick = KICK * shape
    kicked = jnp.asarray(
        np.stack([density[0] + 0.5 * kick, density[1] - 0.5 * kick])
    )
    return system, pseudos, saddle, kicked


def _setup(path, ethr=1.0e-11):
    system = build_system(read_pw_input(path))
    pseudos = tuple(
        read_upf(Path(__file__).parents[1] / "data" / "pseudo" / s.pseudo_file)
        for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    nbnd = system.nbnd or default_nbnd(
        calculation.nelec, system.occupations,
        *((calculation.nelup, calculation.neldw) if system.nspin == 2 else (None, None)),
        noncolin=system.noncolin,
    )
    return system, pseudos, calculation, make_residual(calculation, nbnd, ethr)


@pytest.mark.regression
@pytest.mark.parametrize("case", [SILICON, SLAB], ids=["silicon", "al-slab"])
@pytest.mark.slow
def test_jacobian_action_matches_finite_differences(case):
    """``J v`` two ways, from a self-consistent point.

    The comparison is made at a *converged* density, which is where a Newton
    solver spends the steps that decide whether it converges at all, and where
    Davidson's warm start is the one the solver actually uses.
    """
    system, pseudos, calculation, residual = _setup(case)
    converged = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-10)
    x = residual.pack(converged.density, calculation.starting_becsum())
    _, psi = residual.residual(x, converged.wavefunctions)

    rng = np.random.default_rng(0)
    direction = rng.normal(size=x.shape)
    direction /= np.linalg.norm(direction)

    autodiff, _ = residual.jvp(x, direction, psi)
    difference = residual.jvp_finite_difference(x, direction, psi)
    relative = np.linalg.norm(autodiff - difference) / np.linalg.norm(difference)
    # Loose on purpose, and the looseness is the point: the autodiff Jacobian is
    # the Jacobian of Davidson *as implemented*, and from a converged start
    # Davidson exits in one or two steps, so its tangent is a one-step
    # approximation to the eigenvector response rather than the response itself.
    # A percent is enough for a Krylov solver, which needs a direction; it is
    # not the exact response, which is what the Sternheimer rule of P22c is for.
    assert relative < 0.05


@pytest.mark.regression
@pytest.mark.slow
def test_autodiff_jvp_needs_a_warm_start():
    """From a *cold* start the two backends disagree completely, and that is a
    statement about the method rather than a tolerance to be tuned.

    With no starting wavefunctions Davidson runs from the pseudo-atomic
    orbitals, and differentiating that trajectory gives the derivative of a
    different map -- one that happens to land in the same place. The finite
    difference is immune because it relies on Davidson *converging*, not on how
    it got there. This is why ``finite-difference`` is the default backend, and
    it is asserted rather than commented so that a future exact-response rule
    (P22c) has to come and delete it.

    **The size of the disagreement belongs to Davidson's trajectory, so it moved
    when the trajectory did.** Measured on D22 on 2026-10-12 over three random
    directions (seeds 0, 1, 2): 1.08, 1.89 and 3.42 just before ``22842dc``, and
    0.242, 0.422 and 0.687 from it on, once the subspace stopped collapsing on
    every step and the tape became a longer Krylov expansion. The bound was 0.5,
    which seed 0 now misses; 0.1 is twice the 0.05 the warm start is held to in
    :func:`test_jacobian_action_matches_finite_differences` and 2.4 times below
    the smallest of the three, so the assertion still separates the cold tangent
    from a usable Jacobian, which is the claim.

    **P148 moved it again, and named the mechanism** (`PLAN.md` P148, `OPEN.md`
    Part XXVI): the cold disagreement is the tangent of a degenerate Davidson
    subspace, half of it JAX's ``eigh`` rule and half the gauge of one member's
    correction vector, and the kernel in ``solvers/subspace.py`` removed the
    first half. Seed 0 on this slab reads **0.051** since, against 0.242 before,
    while a cell with no degenerate block agrees cold to 1.5e-9. So the bound is
    0.02: still an order above what a Jacobian reads where the tangent is exact,
    and now only a hair above the 0.05 the warm start is held to, which is the
    honest size of what a cold start still costs here.
    """
    system, pseudos, calculation, residual = _setup(SLAB)
    x = residual.pack(calculation.starting_density(), calculation.starting_becsum())
    rng = np.random.default_rng(0)
    direction = rng.normal(size=x.shape)
    direction /= np.linalg.norm(direction)

    autodiff, _ = residual.jvp(x, direction, None)
    difference = residual.jvp_finite_difference(x, direction, None)
    relative = np.linalg.norm(autodiff - difference) / np.linalg.norm(difference)
    assert relative > 0.02


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.parametrize("case", [SILICON, SLAB], ids=["silicon", "al-slab"])
def test_newton_krylov_finds_the_mixer_s_fixed_point(case):
    system, pseudos, _, _ = _setup(case)
    mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-9)
    newton = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-9,
        scf_solver="newton-krylov",
        scf_solver_options={"forcing": 0.5, "warmup": 6},
    )
    assert newton.converged and mixed.converged
    assert newton.total_energy == pytest.approx(mixed.total_energy, abs=TOTAL_ENERGY_RY)
    assert newton.solver is not None and newton.solver.steps > 0


@pytest.mark.regression
@pytest.mark.parametrize("case", [SILICON, SLAB], ids=["silicon", "al-slab"])
@pytest.mark.slow
def test_kerker_changes_the_path_and_not_the_answer(case):
    system, pseudos, _, _ = _setup(case)
    plain = run_scf(system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-9)
    kerker = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                     conv_thr=1e-9, mixing_mode="kerker")
    assert kerker.converged
    assert kerker.total_energy == pytest.approx(plain.total_energy, abs=TOTAL_ENERGY_RY)


@pytest.mark.regression
@pytest.mark.parametrize("case", [SILICON, SLAB], ids=["silicon", "al-slab"])
@pytest.mark.slow
def test_local_tf_changes_the_path_and_not_the_answer(case):
    """``approx_screening2``: the same fixed point as any other mixer.

    A preconditioner is an approximate inverse Jacobian, so it may change how
    many iterations a run takes and must not change what it converges to. That
    is the whole of what makes one safe to switch on, and it is the check that
    a density-dependent one especially needs -- its operator is rebuilt at every
    iteration, so a bug in it moves with the density rather than sitting still.
    """
    system, pseudos, _, _ = _setup(case)
    plain = run_scf(system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-9)
    local = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                    conv_thr=1e-9, mixing_mode="local-TF")
    assert local.converged
    assert local.total_energy == pytest.approx(plain.total_energy, abs=TOTAL_ENERGY_RY)


@pytest.mark.regression
def test_local_tf_screens_the_vacuum_less_than_the_metal():
    """The one thing that distinguishes it from Kerker, measured directly.

    ``alpha(r) = 3 (2 pi / 3)^(5/3) r_s(r)`` is large where the density is
    small, so the operator ``4 pi e2 + |G|^2 alpha`` is dominated by its second
    term in vacuum and by its first in the metal -- which is the statement that
    the vacuum is screened less. Kerker has one number for both and cannot make
    the distinction at all, so the test is that the two preconditioners disagree
    *and* that they disagree in the vacuum rather than everywhere.
    """
    from defuqumat.scf.mixing import kerker_preconditioner, local_tf_preconditioner

    system, pseudos, calculation, _ = _setup(SLAB)
    rho = np.asarray(calculation.starting_density())
    shape = rho.shape
    rng = np.random.default_rng(0)
    residual = 0.01 * rng.standard_normal(rho.size)

    local = local_tf_preconditioner(
        calculation.basis.dense, system.cell, shape, beta=0.7)
    kerker = kerker_preconditioner(
        calculation.basis.dense, system.cell, shape, beta=0.7, nelec=calculation.nelec)
    a = np.asarray(local(residual, rho.ravel()))[: int(np.prod(shape))].reshape(shape)
    b = np.asarray(kerker(residual, rho.ravel()))[: int(np.prod(shape))].reshape(shape)

    charge = rho.reshape(shape[0], -1).sum(axis=0)
    vacuum = charge < np.quantile(charge, 0.25)
    metal = charge > np.quantile(charge, 0.75)
    difference = np.abs(a - b).reshape(shape[0], -1).sum(axis=0)
    assert np.any(difference > 0), "local-TF reproduced Kerker exactly"
    assert difference[vacuum].mean() > difference[metal].mean(), (
        "local-TF differs from Kerker mostly in the dense region, which is the "
        "opposite of what a locally-screened preconditioner does"
    )


@pytest.mark.regression
@pytest.mark.parametrize("case", [SILICON, SLAB, AL_TETRAHEDRA],
                         ids=["silicon", "al-slab", "al-tetrahedra"])
@pytest.mark.slow
def test_weights_agree_with_the_driver(case):
    """``residual._weights`` is a second copy of the driver's occupation
    dispatch, kept because ``Calculation.occupations`` syncs the Fermi level to
    the host and cannot be traced. A second copy is a second thing to get wrong,
    so it is pinned against the original rather than trusted."""
    from defuqumat.scf.residual import _weights

    system, pseudos, calculation, _ = _setup(case)
    converged = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-9)
    eigenvalues = np.asarray(converged.eigenvalues_by_spin)
    expected, _ = calculation.occupations(eigenvalues)
    assert np.allclose(_weights(calculation, eigenvalues), np.asarray(expected), atol=1e-14)



#: How far the autodiff ``dF v`` may sit from the central difference on the
#: tetrahedron cell, warm, relative to ``|dF v|`` (not to ``|J v| = |dF v - v|``,
#: which is about 1 here and would hide the whole occupation term). Measured on
#: D22 (``notes/incompat/I71.md``): 2.6e-2, 1.1e-2, 2.3e-2 on three random
#: directions with the implicit tangent, 1.22, 0.045, 1.38 with the bare
#: bisection's.
AUTODIFF_TETRAHEDRA_BOUND = 0.1


def _occupation_cell(name, qe_testsuite=None):
    """``al-tetrahedra.in`` (committed) or ``pw_scf/scf-occ.in`` (test-suite)."""
    if name == "tetrahedra":
        return _setup(AL_TETRAHEDRA)
    return _setup(qe_testsuite / "pw_scf" / "scf-occ.in")


@pytest.mark.regression
@pytest.mark.parametrize("name", ["tetrahedra", "from_input"])
def test_weights_take_tetrahedra_and_from_input_like_the_driver(request, name):
    """``_weights`` against ``Calculation.occupations`` for the two schemes I71 added.

    Any eigenvalues of the right shape will do -- the dispatch is what is being
    compared, not a converged state -- so this runs no SCF. Random sorted
    levels put the Fermi level inside bands, where the tetrahedron weights are
    least trivial.
    """
    from defuqumat.scf.residual import _weights

    testsuite = None if name == "tetrahedra" else request.getfixturevalue("qe_testsuite")
    system, _, calculation, residual = _occupation_cell(name, testsuite)
    nk = len(system.kpoints.weights)
    rng = np.random.default_rng(0)
    eigenvalues = np.sort(rng.uniform(-0.5, 1.5, size=(1, nk, residual.nbnd)), axis=-1)
    expected, _ = calculation.occupations(eigenvalues)
    ours = _weights(calculation, jnp.asarray(eigenvalues))
    assert np.allclose(np.asarray(ours), np.asarray(expected), atol=1e-14)
    assert float(np.sum(ours)) == pytest.approx(calculation.nelec, abs=1e-9)


@pytest.mark.regression
@pytest.mark.parametrize("name", ["tetrahedra", "from_input"])
def test_the_sternheimer_jacobian_still_refuses_these_occupations(request, name):
    """The residual takes both schemes; the implicit Jacobian does not.

    It needs ``dfpt_tetra_beta`` for tetrahedra and a projector for a filling
    fixed by hand, neither of which is here, so ``jvp = 'sternheimer'`` keeps
    its refusal by name while the finite-difference and autodiff backends run.
    """
    testsuite = None if name == "tetrahedra" else request.getfixturevalue("qe_testsuite")
    _, _, _, residual = _occupation_cell(name, testsuite)
    match = "tetrahedron" if name == "tetrahedra" else "from_input"
    with pytest.raises(NotImplementedError, match=match):
        residual._refuse_sternheimer(None)


@pytest.mark.regression
@pytest.mark.slow
def test_newton_krylov_with_tetrahedra_is_pw_x_s_fixed_point():
    """Newton-Krylov on Bloechl tetrahedra: ``pw.x``'s total, and the mixer's.

    ``reference.out.al-tetrahedra:202`` is -4.19789860 Ry and ``:200`` the Fermi
    level 8.5257 eV, from ``pw.x`` 7.5 at ``conv_thr = 1e-10``.
    """
    from defuqumat.io import read_qe_output

    system, pseudos, _, _ = _setup(AL_TETRAHEDRA)
    reference = read_qe_output(AL_TETRAHEDRA_REFERENCE)
    mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                    conv_thr=1e-10)
    newton = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-10,
        scf_solver="newton-krylov", scf_solver_options={"forcing": 0.5, "warmup": 3},
    )
    assert newton.converged and mixed.converged
    assert newton.solver is not None and newton.solver.steps > 0
    assert newton.total_energy == pytest.approx(reference.total_energy, abs=TOTAL_ENERGY_RY)
    assert newton.total_energy == pytest.approx(mixed.total_energy, abs=1e-8)
    assert newton.fermi_energy == pytest.approx(mixed.fermi_energy, abs=1e-5)


@pytest.mark.regression
@pytest.mark.slow
def test_newton_krylov_with_from_input_is_pw_x_s_fixed_point(qe_testsuite):
    """Newton-Krylov on ``pw_scf/scf-occ.in``: ``pw.x``'s -15.79449557 Ry.

    ``reference.out.pw_scf-scf-occ:185``, the regenerated 7.5 reference
    ``test_scf.py`` reads for the same cell.
    """
    from defuqumat.io import read_qe_output
    from tests.conftest import reference_output

    path = qe_testsuite / "pw_scf" / "scf-occ.in"
    system, pseudos, _, _ = _setup(path)
    reference = read_qe_output(reference_output("pw_scf", "scf-occ.in", qe_testsuite))
    mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                    conv_thr=1e-10)
    newton = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-10,
        scf_solver="newton-krylov", scf_solver_options={"forcing": 0.5, "warmup": 3},
    )
    assert newton.converged and mixed.converged
    assert newton.solver is not None and newton.solver.steps > 0
    assert newton.total_energy == pytest.approx(reference.total_energy, abs=TOTAL_ENERGY_RY)
    assert newton.total_energy == pytest.approx(mixed.total_energy, abs=1e-8)


@pytest.mark.regression
@pytest.mark.slow
def test_autodiff_jacobian_with_tetrahedra_matches_finite_differences(monkeypatch):
    """``J v`` by ``jax.jvp`` through the tetrahedron weights, against ``r``'s difference.

    What the custom tangent of the tetrahedron Fermi level buys: without it the
    level's tangent is the bracket's, and the autodiff action is off by the
    whole Fermi-shift term (``test_bz_integration.py`` measures that alone).
    """
    import defuqumat.scf.tetrahedra as tetrahedra

    system, pseudos, calculation, residual = _setup(AL_TETRAHEDRA)
    converged = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-11)
    x = residual.pack(converged.density, calculation.starting_becsum())
    _, psi = residual.residual(x, converged.wavefunctions)
    rng = np.random.default_rng(0)
    direction = rng.normal(size=x.shape)
    direction /= np.linalg.norm(direction)
    difference = residual.jvp_finite_difference(x, direction, psi)
    response = np.linalg.norm(difference + direction)

    def relative():
        autodiff, _ = residual.jvp(x, direction, psi)
        return np.linalg.norm(autodiff - difference) / response

    assert relative() < AUTODIFF_TETRAHEDRA_BOUND
    # The guard fires: with the bare bisection's tangent the same check fails.
    monkeypatch.setattr(tetrahedra, "_fermi_level", tetrahedra._bisect)
    import jax

    jax.clear_caches()
    assert relative() > 5 * AUTODIFF_TETRAHEDRA_BOUND


@pytest.mark.regression
@pytest.mark.slow
def test_newton_krylov_reaches_an_unstable_solution():
    """The one thing the residual solver does that no mixer can.

    Iron has two self-consistent solutions here: the ferromagnetic ground state
    and a non-magnetic one, which is a *saddle* of the energy in the
    magnetization direction. Damped mixing is a discrete relaxation dynamics, so
    from a small starting moment it runs downhill into the stable one. Newton is
    stability-blind -- it converges on whichever root it started nearest -- and
    from the same guess it collapses the moment instead.

    **The validation is free and independent**: that ``nspin = 2`` non-magnetic
    energy has to equal a plain ``nspin = 1`` run's on the same cell, and no part
    of the two calculations is shared. The difference between the two roots is
    iron's magnetic stabilisation energy, which is the number such a reference
    state exists to give.

    **The state is perturbed away from the symmetric root and handed to both
    solvers**, rather than both being started from the atomic superposition.
    That is the protocol
    :func:`test_a_hubbard_saddle_is_unstable_to_mixing_and_stable_to_newton`
    uses, and it is the one ``PLAN.md`` P22 says is robust: from far away,
    *which* root Newton-Krylov lands on depends on the inner-solve accuracy in
    no systematic way, and no setting makes it otherwise.

    **Why it had to be rewritten, and what the rewrite measured.** An earlier
    version started both solvers from the atomic superposition, and it was
    passing on a knife edge: changing how ``|psi|^2`` is evaluated --
    ``Re(conj(psi) psi)`` rather than ``abs(psi)**2``, the same number to **3.5
    eps** (:func:`defuqumat.scf.density.band_density`) -- was enough to send
    Newton to the ferromagnet instead. Rebuilding it around a perturbed root
    then turned up something about the *physics* that P22 had got slightly
    wrong. The symmetric solution of this cell is **not a saddle in the linear
    sense**: it is metastable, with a finite basin, and the kick has to clear it.
    Measured here, with the kick in the atomic magnetization's own shape (a
    uniform ``1 +- eps`` scaling of the two channels is the wrong direction and
    decays by a factor of 300):

    ======  ====================  ====================
    kick    mixing ends at        Newton ends at
    ======  ====================  ====================
    0.05    ``7e-6`` (comes back)  --
    0.08    ``+3.4052``            ``-0.0003``
    0.10    ``+3.4052``            ``+0.0007``
    0.12    ``+3.4053``            ``+0.0009``
    0.15    ``+3.4053``            ``+3.4052``
    0.20    ``+3.4052``            ``-3.4052``
    0.50    ``+3.4051``            ``+3.4052``
    ======  ====================  ====================

    So the demonstration lives in a **window**, and both of its edges are
    physical rather than numerical: below ~0.08 the kick is inside the
    symmetric root's basin and mixing returns to it too, and above ~0.12 the
    perturbed state is nearer the ferromagnet than the symmetric root and Newton
    converges on *that* -- which is what "converges on whichever root it started
    nearest" means, said quantitatively. Three consecutive points in between
    behave identically, which is what makes 0.10 a reproducible choice rather
    than another knife edge.

    "A root no mixer can hold" is still exactly true of the Hubbard saddle
    below, whose 2% kick runs away. For iron the honest statement is a root no
    mixer *reaches*, which Newton returns to from outside its basin.

    **Re-measured on D22, 2026-10-12, after ``22842dc``**, because the table
    above stopped holding: at 0.10 the Kerker run with the loose inner solve
    (``LOOSE_INNER_SOLVE``) went to the ferromagnet. Under ``taskset -c 2-3``,
    with ``TIGHT_INNER_SOLVE`` for Newton:

    ======  ====================  ======================  ======================
    kick    mixing ends at        tight Newton ends at    loose Newton ends at
    ======  ====================  ======================  ======================
    0.085   ``-0.0000``           --                      --
    0.095   ``+0.0001`` (back)    --                      --
    0.10    ``+3.4052``           ``+0.0004``             ``-3.4052``
    0.11    ``+3.4052``           ``+0.0011``             ``+3.4053``
    0.12    ``+3.4053``           ``+0.0003``             ``+3.4052``
    0.13    ``+3.4053``           ``+0.0001``             ``+3.4053``
    0.14    ``+3.4052``           ``+0.0001``             ``+3.4052``
    0.15    ``+3.4052``           ``+3.4053``             ``+3.4053``
    ======  ====================  ======================  ======================

    (both Newton columns with Kerker on). Under one core, ``taskset -c 2``,
    the tight column is the same to the sign of the residual moment from 0.10
    to 0.14 and the loose one reaches the ferromagnet at every point from 0.10
    to 0.14, as it does under two. So the window is now
    0.10 to 0.14, two points either side of 0.12, and its upper edge is set
    by the inner solve: the loose one, which is what this test ran until then,
    reaches the ferromagnet wherever mixing does, and at 0.095 (``-3.4053``)
    where mixing comes back. A third root at
    ``|M| = 0.42`` turns up in the same scans (the tight solve at 0.10 and 0.11
    on the Davidson before ``22842dc``, the tight unpreconditioned one at 0.14), and
    is not the one asserted here.
    """
    system, pseudos, saddle, kicked = _kicked_iron()

    # The saddle is the non-magnetic solution, by an independent nspin = 1 run.
    reference_system, reference_pseudos, _, _ = _setup(IRON_NONMAGNETIC)
    reference = run_scf(
        reference_system, reference_pseudos, conv_thr=1e-8, max_iterations=200,
        calculation=Calculation(reference_system, reference_pseudos),
    )
    assert reference.converged
    assert saddle.total_energy == pytest.approx(reference.total_energy, abs=1e-6)

    # The *same* perturbed state goes to both solvers (:func:`_kicked_iron`).
    ran_away = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                       conv_thr=1e-8, max_iterations=200,
                       starting_density=kicked)
    came_back = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-8,
        max_iterations=200, scf_solver="newton-krylov",
        # The inner solve is load-bearing and not a tuning choice: see
        # ``test_an_inexact_newton_is_only_as_stability_blind_as_its_inner_solve``.
        scf_solver_options={**TIGHT_INNER_SOLVE, "kerker": True},
        starting_density=kicked,
    )

    assert ran_away.converged and came_back.converged
    # *Newton* converged, not the mixing loop that ``run_scf`` hands an
    # unconverged residual solve to: ``SCFResult.iterations`` counts that loop
    # and reads 1 here; the Newton count is ``came_back.solver.iterations``.
    assert came_back.solver.converged
    # Mixing amplified the kick into the ferromagnetic ground state -- which is
    # *what makes the symmetric solution a saddle*...
    assert abs(float(ran_away.magnetization)) > 3.0
    # ...and Newton put it back, which no mixer can be made to do.
    assert abs(float(came_back.magnetization)) < 1e-2
    assert came_back.total_energy == pytest.approx(reference.total_energy, abs=1e-6)
    # The gap between the two roots is iron's magnetic stabilisation energy.
    assert ran_away.total_energy < came_back.total_energy - 0.02


@pytest.mark.regression
@pytest.mark.slow
def test_an_inexact_newton_is_only_as_stability_blind_as_its_inner_solve():
    """Loosening the inner solve changes *which root* is found.

    This is the trap, and it is silent: both runs converge, both report an
    accuracy below ``conv_thr``, and they land on different physics. A Newton
    method is stability-blind only to the extent that its inner solve actually
    delivers the Newton direction; with a badly conditioned or truncated Krylov
    solve the inexact step degrades towards a damped-mixing step, and a
    damped-mixing step flows to the *stable* fixed point. So the inner solve
    here is not a tuning knob for speed -- it decides the answer.

    **Until 2026-10-12 the difference was the preconditioner**, Kerker on in
    ``came_back`` and off here, both at ``LOOSE_INNER_SOLVE``, and P22 read the
    root as decided by Kerker. Re-measured on D22 after ``22842dc`` that is not
    what decides it: with the tight solve both preconditioners return to the
    symmetric root from 0.10 to 0.13, and with the loose one both reach the
    ferromagnet from 0.12 to 0.14, under one core and under two (the table in
    :func:`test_newton_krylov_reaches_an_unstable_solution`). At the old kick
    of 0.10 this test read ``M = -0.0005`` on D22 after ``22842dc`` and
    ``-0.0072`` already at ``30cffd2`` before it, which is the knife edge
    ``OPEN.md`` recorded on 2026-10-04.

    **It starts from the kicked state and not from the input's own
    ``starting_magnetization``**, which is what made this reproducible at all.
    From the bare 0.05 seed the answer is decided by round-off: the same commit
    gave ``M = -3.4050`` with the default thread pool and ``M = -0.0003`` under
    ``OMP_NUM_THREADS=1``, since a different reduction order is a different
    perturbation and a seed that small lets it choose.

    Starting both runs from the same state, with the same preconditioner, is
    what makes this a *comparison*: the inner-solve budget is the only
    difference between this run and ``came_back`` above, which reaches the
    saddle from that same start.
    """
    system, pseudos, _, kicked = _kicked_iron()
    inexact = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-8,
        max_iterations=200, scf_solver="newton-krylov",
        scf_solver_options={**LOOSE_INNER_SOLVE, "kerker": True},
        starting_density=kicked,
    )
    assert inexact.converged
    assert inexact.solver.converged
    assert abs(float(inexact.magnetization)) > 3.0


# --- DFT+U ------------------------------------------------------------------
#
# ``ns`` joins the packed state exactly as ``mix_rho.f90`` puts it in
# ``mix_type``: it is not a function of the density, because the Hubbard
# potential is built from it before the Hamiltonian exists. Two things are
# checked -- that the solver still lands on the mixer's fixed point with a U in
# play, and that it reaches a solution the mixer cannot.


#: bcc iron at one k-point under a uniform field: ``tests/data/qe/fe-fsm.in``
#: with its feedback constraint replaced by ``b_field(3) = 0.01``, which the
#: mixing loop holds fixed, so the residual's ``F`` is the loop's map (I73).
FIELD_EDITS = (
    ("constrained_magnetization = 'fsm'", "b_field(3) = 0.01"),
    ("lambda = 0.02", ""),
    ("fixed_magnetization(3) = 2.0", ""),
)


def _iron_under_a_field(edits):
    from defuqumat.io.pwin import parse_pw_input

    text = (Path(__file__).resolve().parents[1] / "data" / "qe" / "fe-fsm.in").read_text()
    for old, new in edits:
        assert old in text, old
        text = text.replace(old, new)
    system = build_system(parse_pw_input(text))
    pseudos = tuple(
        read_upf(Path(__file__).parents[1] / "data" / "pseudo" / s.pseudo_file)
        for s in system.structure.species
    )
    return system, pseudos


@pytest.mark.regression
@pytest.mark.slow
def test_newton_krylov_finds_the_mixer_s_fixed_point_under_a_field():
    """I73: an applied field is part of ``F``, so Newton lands where mixing does.

    Measured on D22 (``review/incompat/I73/harnessBC.log``, one sample): total
    energy 1.7e-8 Ry apart, field energy 1.7e-8, moment 1.7e-6 mu_B, with
    Newton at 7 outer steps and 58 evaluations of ``F`` against the mixer's 14
    iterations. The field-free cell is the alternative a solver that dropped
    the Zeeman term would land on, and it is asserted to be far away.

    **The Newton run's loop iteration count is what carries the claim.** A
    Newton whose ``F`` dropped the field converges on the field-free residual,
    and the mixing loop after it (``max_iterations = 300``) still carries the
    field and finishes at the mixer's fixed point, so the energy, field energy
    and moment asserts all pass. Measured on D22 with that mutation
    (``make_residual`` handed the calculation with ``magnetic_field = None``):
    the loop takes 11 iterations, against 1 clean.
    """
    system, pseudos = _iron_under_a_field(FIELD_EDITS)
    assert Calculation(system, pseudos).magnetic_field is not None
    mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                    conv_thr=1e-10, max_iterations=300, mixing_beta=0.3)
    newton = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-10,
        max_iterations=300, mixing_beta=0.3, scf_solver="newton-krylov",
        scf_solver_options={"forcing": 0.5, "warmup": 6},
    )
    assert mixed.converged and newton.converged
    assert newton.solver is not None and newton.solver.converged
    # A Newton converged on the wrong problem is finished by the mixing loop,
    # which would then take more than a step or two (11 measured).
    assert newton.iterations <= 2
    # The reported total leaves out the Zeeman term, so it is not the
    # stationary functional: E + E_field is, and dE = -dE_field to 1e-10 at
    # b_field(3) = 0.01 and 0.02. The E and E_field asserts are one first-order
    # check in the moment, read twice; 1.7e-8 Ry and 1.7e-6 mu_B were measured,
    # and the tolerances leave room for a different stopping point.
    assert newton.total_energy == pytest.approx(mixed.total_energy, abs=TOTAL_ENERGY_RY)
    assert newton.field_energy == pytest.approx(mixed.field_energy, abs=TOTAL_ENERGY_RY)
    assert newton.magnetization == pytest.approx(mixed.magnetization, abs=1e-4)

    free_system, _ = _iron_under_a_field(
        (("constrained_magnetization = 'fsm'", ""),) + FIELD_EDITS[1:]
    )
    free = run_scf(free_system, pseudos, conv_thr=1e-10, max_iterations=300,
                   mixing_beta=0.3)
    assert abs(free.magnetization - mixed.magnetization) > 1e-3
    assert abs(free.total_energy - mixed.total_energy) > 1e-5


#: The fixed-spin-moment cells of the I73 feedback round: ``fe-fsm.in`` as
#: committed (collinear, ``'fsm'``, 2.0 mu_B), and the canted iron pair of
#: ``test_holding_a_texture._iron`` under the two per-atom schemes, with Elk's
#: adaptive mixer the loop needs there.
FEEDBACK_CASES = ("fsm", "atomic fsm", "atomic fsm direction")


def _feedback_text(case):
    root = Path(__file__).resolve().parents[1] / "data" / "qe"
    if case == "fsm":
        return (root / "fe-fsm.in").read_text()
    text = (root / "fe2-canted-nosoc.in").read_text().replace(
        "    nosym = .true.\n",
        f"    nosym = .true.\n    constrained_magnetization = '{case}'\n"
        "    lambda = 0.02\n")
    return text + "STARTING_MOMENTS\n  1.868 0 0\n  0 1.868 0\n"


def _energy_at_target(result):
    """The reported total moved to the target moment, to second order.

    The total leaves out the Zeeman term and ``dE/dm = B`` along the
    constrained family, so a run that stopped ``dm`` off its target reads
    ``B . dm`` off in ``E``, first order; ``E + B . (m_fix - m)`` removes it.
    """
    field = result.magnetic_field
    b = np.asarray(field.driven(), dtype=float)
    targets = np.asarray(field.targets, dtype=float)
    if field.driven_leaf == "atomic":
        m = np.asarray(result.site_moments, dtype=float)
    else:
        m = np.atleast_1d(np.asarray(result.magnetization, dtype=float))
    return float(result.total_energy + np.sum(b * (targets - m)))


@pytest.mark.regression
@pytest.mark.slow
@pytest.mark.parametrize("case", FEEDBACK_CASES)
def test_newton_krylov_finds_the_mixer_s_fixed_point_under_a_feedback_field(
        case, monkeypatch):
    """I73 feedback round: the driven field in Newton's state lands where mixing does.

    The reference is the mixing loop held to its own fixed point:
    ``FSM_TOLERANCE`` (1e-3 mu_B by default, and the loop stops there, 1.5e-4
    mu_B off on ``fe-fsm.in``) is set to 1e-12 for that run only, which holds
    the moment to 1.6e-13 mu_B on ``fe-fsm.in`` and 1.1e-12 on the canted pair
    under ``'atomic fsm'``. The direction scheme's angle cannot be held below
    about 8.5e-7 degrees, since ``site_residuals`` reads it as an ``arccos``
    of a cosine within one ulp of 1, so that reference is 2.5e-8 mu_B off in
    the perpendicular moment and 1.0e-9 Ry off in its own raw total.

    Newton's field share is ``|dm| (|B| + tau |dm|)`` (the user's decision of
    2026-10-17), a bound on the first-order error ``B . dm`` of the reported
    total, which leaves out the Zeeman term. On ``fe-fsm.in`` Newton reaches
    it and the raw totals agree to 7.8e-13 Ry; on the canted pair it stalls
    at 7e-10 to 7e-9 within its 40 outer steps and hands the state over, and
    the loop accepts it in one iteration at its own 1e-3 tolerance, so the run
    reports converged with the raw total 1.2e-10 Ry (``'atomic fsm'``) and
    2.5e-9 Ry (``'atomic fsm direction'``) off the loop's. Measured on D22
    (``notes/incompat/I73.md``, "The feedback field").

    **The loop iteration count carries the claim.** With the field row of
    ``F`` replaced by the identity Newton holds the starting field and solves
    the unconstrained problem, and the loop after the hand-over then drives
    the field itself: 45 iterations on ``fe-fsm.in`` and 93 on the canted pair
    under ``'atomic fsm'``, against 1 clean (``'atomic fsm direction'`` was not
    mutated; measured under the earlier measure ``tau |dm|^2``).
    """
    from defuqumat import Calculator
    from defuqumat.scf import fields

    pseudo_dir = str(Path(__file__).resolve().parents[1] / "data" / "pseudo")
    text = _feedback_text(case)
    options = ({"mixing_beta": 0.3} if case == "fsm"
               else {"mixing_mode": "adaptive", "mixing_beta": 0.05})
    options.update(conv_thr=1e-11, max_iterations=400, verbose=False)
    with monkeypatch.context() as patch:
        patch.setattr(fields, "FSM_TOLERANCE", 1e-12)
        patch.setattr(fields, "FSM_ANGLE_TOLERANCE", 1e-12)
        mixed = Calculator.from_text(text, pseudo_dir, announce=False).get_scf(**options)
    newton = Calculator.from_text(text, pseudo_dir, announce=False).get_scf(
        scf_solver="newton-krylov",
        scf_solver_options={"forcing": 0.5, "warmup": 6, "max_iterations": 40},
        **options)
    assert mixed.converged and newton.converged
    assert newton.solver is not None
    assert newton.iterations <= 2
    assert _energy_at_target(newton) == pytest.approx(_energy_at_target(mixed), abs=1e-11)
    if case == "fsm":
        # Converged numbers: measured 7.8e-13 Ry, a moment 6.2e-11 off and the
        # field 6.0e-11 Ry apart.
        assert newton.solver.converged
        assert newton.total_energy == pytest.approx(mixed.total_energy, abs=1e-11)
        assert newton.magnetization == pytest.approx(2.0, abs=1e-9)
        np.testing.assert_allclose(np.asarray(newton.magnetic_field.driven()),
                                   np.asarray(mixed.magnetic_field.driven()), atol=1e-9)
    else:
        # **Stall levels, not convergence claims.** Newton does not reach the
        # measure on the canted pair, so these differences are set by where it
        # wanders, one sample each on one D22 core: raw totals 1.2e-10 and
        # 2.5e-9 Ry (3.5e-10 and 8.4e-10 in the run with the field scaled),
        # fields 1.6e-9 and 4.8e-9 Ry, site moments 1.5e-8 and 3.6e-7 mu_B.
        # The bounds are about ten times those.
        raw = {"atomic fsm": 5e-9, "atomic fsm direction": 3e-8}[case]
        assert newton.total_energy == pytest.approx(mixed.total_energy, abs=raw)
        np.testing.assert_allclose(np.asarray(newton.magnetic_field.driven()),
                                   np.asarray(mixed.magnetic_field.driven()), atol=5e-8)
        np.testing.assert_allclose(np.asarray(newton.site_moments),
                                   np.asarray(mixed.site_moments), atol=4e-6)


@pytest.mark.regression
@pytest.mark.slow
def test_newton_krylov_matches_the_mixer_with_a_hubbard_u():
    """The packing check: ``ns`` in the state must not move the answer."""
    system, pseudos, _, _ = _setup(NICKEL_U)
    mixed = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                    conv_thr=1e-8, max_iterations=150)
    newton = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-8,
        max_iterations=150, scf_solver="newton-krylov",
        scf_solver_options={"forcing": 0.5, "gmres_maxiter": 8, "max_iterations": 12},
    )
    assert mixed.converged and newton.converged
    assert newton.total_energy == pytest.approx(mixed.total_energy, abs=TOTAL_ENERGY_RY)
    # ``ns`` is compared in ``ns_ddot``, the metric the SCF itself converges on,
    # rather than elementwise. An elementwise tolerance here would be a number
    # invented for the test: ``conv_thr`` bounds ``U/2 sum |dns|^2``, so at
    # 1e-8 with U = 3 eV the individual entries are only pinned to about 3e-4,
    # and the two solvers do differ at 1e-4. Judging them by the convergence
    # criterion is both the right comparison and one that scales with it.
    _, _, calculation, _ = _setup(NICKEL_U)
    assert float(calculation.ns_accuracy(newton.ns - mixed.ns)) < 10.0 * 1e-8


@pytest.mark.regression
@pytest.mark.slow
def test_a_hubbard_saddle_is_unstable_to_mixing_and_stable_to_newton():
    """The definition of an unstable fixed point, tested as such.

    Perturb the non-magnetic solution along the magnetization direction and the
    two solvers do opposite things: damped mixing runs away to the ferromagnet,
    which is *what makes it a saddle*, and Newton comes back to it. This is
    pyqula's own check on the solutions it found (``scftk/vjinteraction_jax.py``)
    and it is stronger than merely landing on an unusual answer, because the
    same perturbed input is handed to both.

    **It needs the custom starting occupation matrix**, and that is the point of
    ``uniform_ns``: ``init_ns`` reads Hund's rule off ``starting_magnetization``,
    so the default start is already deep in the ferromagnetic basin whatever
    that number is, and the perturbation has to be applied to ``ns`` as well as
    to the density -- a kick in the density alone is undone by the Hubbard
    potential the polarised ``ns`` is still generating.
    """
    system, pseudos, calculation, _ = _setup(NICKEL_U)

    # The saddle: spin-symmetric density and an orbitally uniform ns, which the
    # SCF map preserves exactly.
    rho = np.asarray(calculation.starting_density())
    symmetric = jnp.asarray(np.repeat(rho.mean(axis=0, keepdims=True), rho.shape[0], axis=0))
    start = dict(starting_density=symmetric,
                 starting_becsum=calculation.starting_becsum(),
                 starting_ns=uniform_ns(calculation.hubbard, calculation.nspin))
    saddle = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                     conv_thr=1e-9, max_iterations=150, **start)
    assert saddle.converged
    assert abs(float(saddle.magnetization)) < 1e-6

    # ...and it is the non-magnetic solution, by an independent nspin = 1 run.
    reference_system, reference_pseudos, _, _ = _setup(NICKEL_U_NONMAGNETIC)
    reference = run_scf(reference_system, reference_pseudos, conv_thr=1e-9,
                        max_iterations=150,
                        calculation=Calculation(reference_system, reference_pseudos))
    assert saddle.total_energy == pytest.approx(reference.total_energy, abs=1e-6)

    # Kick it 2% along the magnetization direction, in both parts of the state.
    epsilon = 0.02
    density = np.asarray(saddle.density)
    density = density * np.array([1.0 + epsilon, 1.0 - epsilon])[:, None, None, None]
    ns = np.asarray(saddle.ns) * np.array([1.0 + epsilon, 1.0 - epsilon])[:, None, None, None]
    kicked = dict(starting_density=jnp.asarray(density),
                  starting_becsum=calculation.starting_becsum(),
                  starting_ns=jnp.asarray(ns))

    ran_away = run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                       conv_thr=1e-9, max_iterations=150, **kicked)
    came_back = run_scf(
        system, pseudos, calculation=Calculation(system, pseudos), conv_thr=1e-9,
        max_iterations=150, scf_solver="newton-krylov",
        scf_solver_options={"forcing": 0.1, "gmres_maxiter": 30, "max_iterations": 15},
        **kicked,
    )
    assert ran_away.converged and came_back.converged
    # Mixing amplified the kick into the ferromagnetic solution...
    assert abs(float(ran_away.magnetization)) > 1.0
    # ...and Newton put it back.
    assert abs(float(came_back.magnetization)) < 1e-4
    assert came_back.total_energy == pytest.approx(saddle.total_energy, abs=1e-7)


@pytest.mark.regression
def test_starting_ns_is_checked_rather_than_broadcast():
    """A wrong-shaped occupation matrix is a mistake worth a message: the shape
    has one slot per *correlated* atom and zero-padded manifolds, so it is not
    something to infer from the structure."""
    system, pseudos, calculation, _ = _setup(NICKEL_U)
    with pytest.raises(ValueError, match="nslot"):
        run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                max_iterations=1, starting_ns=np.zeros((2, 1, 3, 3)))


@pytest.mark.regression
def test_starting_ns_without_a_u_is_refused():
    system, pseudos, calculation, _ = _setup(SILICON)
    with pytest.raises(ValueError, match="no Hubbard U"):
        run_scf(system, pseudos, calculation=Calculation(system, pseudos),
                max_iterations=1, starting_ns=np.zeros((1, 1, 5, 5)))


# ---- the implicit Jacobian (P150) -------------------------------------------

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


@lru_cache(maxsize=2)
def _converged_text(name: str):
    """A converged cell built from an input's text, its residual and its state.

    ``si-gamma`` is ``si2-lda-gap.in`` on a 2x2x2 mesh with ``Gamma`` in it, a
    threefold block at ``Gamma`` and pairs at ``L``, the cell ``OPEN.md`` Part
    XXVI measured the autodiff Jacobian on.
    """
    from defuqumat.io.pwin import parse_pw_input

    nbnd = None
    if name == "si-gamma":
        text = (CASES / "si2-lda-gap.in").read_text()
        assert "\n 6 6 6 0 0 0" in text
        text = text.replace("\n 6 6 6 0 0 0", "\n 2 2 2 0 0 0")
        # Six bands above the four occupied ones, as ``Calculator`` gives this
        # cell. With ``nbnd = 4`` the top of the threefold block at ``Gamma`` is
        # Davidson's topmost requested root, the one it converges worst, and the
        # eigenpairs the action is built from read 2.6e-3 off the reference.
        nbnd = 10
    else:
        text = (CASES / f"{name}.in").read_text()
    system = build_system(parse_pw_input(text))
    pseudos = tuple(
        read_upf(Path(__file__).parents[1] / "data" / "pseudo" / s.pseudo_file)
        for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=100, nbnd=nbnd)
    assert result.converged
    nbnd = int(np.asarray(result.eigenvalues_by_spin).shape[-1])
    residual = make_residual(calculation, nbnd, float(result.ethr))
    x = residual.pack(result.density, tuple(result.becsum or ()))
    return calculation, result, residual, np.asarray(x), nbnd


#: ``(buffer bands, bound on the Sternheimer action, bound the warm difference
#: stays above)``, both relative to ``|dF v|`` and both the median over the
#: three steps ``STERNHEIMER_STEPS`` of ``max(1, |x|)``. Measured on D22
#: (2026-10-14, P150): the Sternheimer action 4.9e-7, 2.6e-3 and 1.4e-7 off the
#: reference on ``si-gamma`` (the middle one the reference's own: the cold
#: difference at 1e-5 is 2.6e-3 from the two beside it, a Davidson solve on a
#: degenerate block landing elsewhere at one of its two points) and 9.6e-6,
#: 5.8e-5 and 9.2e-6 on ``al-metal``; 4.6e-7 on silicon at one generic k-point,
#: where no degeneracy exists. The warm difference was 0.35 off on
#: ``si-gamma`` and 4.3e-3 to 3.9e-2 on ``al-metal``.
STERNHEIMER_CASES = {
    "si-gamma": (4, 5e-6, 1e-2),
    "al-metal": (0, 1e-4, 1e-3),
}
STERNHEIMER_STEPS = (1e-4, 1e-5, 1e-6)


@pytest.mark.regression
@pytest.mark.parametrize("name", sorted(STERNHEIMER_CASES))
def test_the_sternheimer_jacobian_is_the_derivative_of_the_exact_eigenpair_map(name):
    """``jvp = 'sternheimer'`` is the derivative of ``F`` with exact eigenpairs.

    The reference is the central difference of the same ``F`` diagonalised from
    a cold start at ``ethr = 1e-13``, on the insulator with four buffer bands,
    since Davidson converges its topmost requested root worst (on ``si-gamma``
    without them the reference itself moves by 1e-4 to 7e-3 with the step). The
    two share the Hamiltonian and the assembly and nothing else. **The
    warm-started difference is not the reference, and the test says so**: from
    a converged start Davidson stops once the eigenvalues have settled, which is
    second order in the perturbation, so its eigenvectors carry a first-order
    error, and the warm map's difference is a derivative of a different map
    (P150).
    """
    buffer, bound, warm_above = STERNHEIMER_CASES[name]
    calculation, result, residual, x, nbnd = _converged_text(name)
    rng = np.random.default_rng(7)
    v = rng.normal(size=x.shape)
    v /= np.linalg.norm(v)
    size = max(1.0, float(np.linalg.norm(x)))

    jv = residual.sternheimer_jacobian(x, result.wavefunctions)(v)
    exact = make_residual(calculation, nbnd + buffer, 1e-13)
    errors, warm_errors = [], []
    for step in STERNHEIMER_STEPS:
        reference = np.asarray(
            exact.jvp_finite_difference(x, v, None, epsilon=step * size))
        warm = np.asarray(residual.jvp_finite_difference(
            x, v, result.wavefunctions, epsilon=step * size))
        scale = np.linalg.norm(reference + v)
        errors.append(np.linalg.norm(jv - reference) / scale)
        warm_errors.append(np.linalg.norm(warm - reference) / scale)
    assert np.median(errors) < bound
    assert np.median(warm_errors) > warm_above


@pytest.mark.regression
def test_the_sternheimer_jacobian_does_not_see_the_basis_of_a_degenerate_block():
    """Rule D4 for the Jacobian: the same eigenpairs rotated by a random unitary
    inside every degenerate block give the same ``J v``. Measured at 3.0e-10 of
    ``|dF v|`` on ``si-gamma`` over ten blocks; the autodiff action through
    Davidson is not even defined there (``OPEN.md`` Part XXVI)."""
    import jax.numpy as jnp

    calculation, result, residual, x, _ = _converged_text("si-gamma")
    rng = np.random.default_rng(7)
    v = rng.normal(size=x.shape)
    v /= np.linalg.norm(v)
    jacobian = residual.sternheimer_jacobian(x, result.wavefunctions)
    jv = jacobian(v)
    eigenvalues, psi = (np.asarray(a) for a in jacobian.eigenpairs)
    rotated = psi.copy()
    blocks = 0
    for k in range(eigenvalues.shape[1]):
        start = 0
        while start < eigenvalues.shape[2]:
            stop = start + 1
            while (stop < eigenvalues.shape[2]
                   and eigenvalues[0, k, stop] - eigenvalues[0, k, stop - 1] < 1e-8):
                stop += 1
            if stop - start > 1:
                size = stop - start
                q, _ = np.linalg.qr(rng.normal(size=(size, size))
                                    + 1j * rng.normal(size=(size, size)))
                rotated[0, k, start:stop] = q.T @ psi[0, k, start:stop]
                blocks += 1
            start = stop
    assert blocks >= 3
    turned = residual.sternheimer_jacobian_at(
        x, jnp.asarray(eigenvalues), jnp.asarray(rotated))(v)
    assert np.linalg.norm(turned - jv) / np.linalg.norm(jv + v) < 1e-8


@pytest.mark.unit
def test_the_sternheimer_jacobian_refuses_a_hubbard_u_and_a_meta_gga():
    """The two parts of the state whose perturbation of ``H`` is not a local
    potential are refused by name, before anything is computed."""
    from defuqumat.scf.residual import ScfResidual

    hubbard = ScfResidual(None, 4, 1e-10, ((1, 2, 2, 2),), 8, ns_shape=(1, 1, 5, 5))
    with pytest.raises(NotImplementedError, match="Hubbard U"):
        hubbard.sternheimer_jacobian(np.zeros(8), None)
    meta = ScfResidual(None, 4, 1e-10, ((1, 2, 2, 2),), 8, tau_shape=(1, 2, 2, 2))
    with pytest.raises(NotImplementedError, match="meta-GGA"):
        meta.sternheimer_jacobian(np.zeros(8), None)
