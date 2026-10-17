"""Solving ``r(rho) = 0`` with the Jacobian, rather than iterating ``F``.

``scf/residual.py`` writes self-consistency as a residual; this module solves it.
The default remains mixing -- ``run_scf``'s loop, unchanged -- and this is the
other entry in the registry (rule R4).

**Newton-Krylov.** The step solves ``J d = -r`` for the Newton direction, with
``J`` never formed: GMRES asks only for ``J v``, which is one forward-mode
differentiation of one SCF iteration. Forming ``J`` densely is not an option and
is worth saying explicitly -- it is ``(nspin * nr)^2``, which on the smallest
slab here is a 90000 x 90000 matrix and on anything real is astronomically
worse. pyqula measured the same wall (``scftk/densitydensity_jax.py``: 32
orbitals did not finish three dense-Jacobian Newton steps in 280 s) on a problem
thousands of times smaller.

**Two things pyqula established that this does not repeat** (``PLAN.md`` P22):

* minimising the *physical energy* off the self-consistency surface fails --
  the SCF solution is a saddle of that off-shell functional, not a minimum, so
  a descent method walks away from it with a correct gradient;
* minimising ``||r||^2`` with a scalar-loss optimiser (L-BFGS) stalls, because
  it sees only ``J^T r`` and discards the residual *vector*. A Krylov method
  uses that vector directly, which is why this is a root-finder and not a
  minimiser.

**Two JVP backends, and finite differences won.** ``autodiff`` is ``jax.jvp``
through the step, Davidson's ``lax.while_loop`` included; ``finite-difference``
is a central difference of ``r``, which is classic Jacobian-free Newton-Krylov
and shares no machinery with the first. Measured on the aluminium slab
(``tests/regression/test_scf_solvers.py`` pins both numbers):

* from a **cold** start the two disagree by **24%** along the test's direction
  (24 to 69% over three; 109 to 342% before ``22842dc`` changed when Davidson
  collapses its subspace) -- differentiating Davidson's trajectory from the
  pseudo-atomic orbitals is the derivative of a different map, one that merely
  lands in the same place, and how different depends on the trajectory;
* from a **warm** start they agree to **0.8%**, which is a Jacobian good enough
  to give a Krylov solver a direction and is not the response operator;
* and the finite difference is **4-7x faster**, because forward mode through the
  ``while_loop`` costs more than the two extra primal solves do when both start
  from a converged guess.

So ``finite-difference`` is the default. This is the empirical case for the
Sternheimer rule of ``PLAN.md`` P22c: the response has to be *written down*, not
taken from the eigensolver's tape.

**Cost, stated up front because the wall-clock case is not the case for this.**
One outer iteration is one residual evaluation, plus one line-search
evaluation per backtrack, plus **two** per GMRES iteration -- the default
Jacobian action is a central difference, so it costs two evaluations of ``F``,
and each evaluation is a diagonalisation, 84% of an SCF step
(``PERFORMANCE.md``). Anderson's whole iteration is one such diagonalisation and
a negligible least-squares solve. So Newton-Krylov is ahead only when it cuts
the *outer* count by more than the inner Krylov work adds -- and measured over a
sweep of increasingly ill-conditioned aluminium slabs it never is, because the
outer count stays flat at the cost of GMRES iterations that grow to replace it.
The numbers are in ``PERFORMANCE.md``; the summary is 19 to 139 evaluations
against Kerker-preconditioned Anderson's 14 to 36.

**So this is not the default and should not become one until the Jacobian action
is cheap.** The Sternheimer rule of ``PLAN.md`` P22c, a projected CG solve per
occupied band rather than another diagonalisation, is in since P150
(``jvp = 'sternheimer'``), and it does not make it cheap: on the aluminium slab
from the atomic density it takes 4 outer steps and 9 evaluations of ``F``
against the finite difference's 5 and 82, and 10.3 s of Newton against 3.3 s,
because one CG solve to 1e-11 costs more than two diagonalisations from a
converged start (D22, two cores, after the first call; four more cells read the
same way). What the solver buys today is a *capability*: Newton is
stability-blind and converges on whichever root it starts nearest, so it reaches
**unstable** SCF solutions that a damped mixing dynamics cannot hold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, lgmres

__all__ = ["NewtonKrylovResult", "newton_krylov", "SCF_SOLVERS", "get_scf_solver"]


@dataclass
class NewtonKrylovResult:
    """The solution and, as importantly, what it cost."""

    x: np.ndarray
    psi: object
    converged: bool
    accuracy: float
    iterations: int
    #: Evaluations of ``F``. This is the currency: each is one diagonalisation,
    #: and it is the only fair thing to compare against a mixer's iteration
    #: count.
    steps: int
    jvps: int
    history: list = field(default_factory=list)
    #: The solved feedback (fixed-spin-moment) field, the
    #: :class:`~defuqumat.scf.fields.MagneticField` whose driven leaf is
    #: ``x``'s, or ``None`` without one. Set by the driver, which hands it to
    #: the mixing loop that runs after the solve (I73).
    field: object = None


def newton_krylov(
    residual,
    x0,
    psi0,
    accuracy_of,
    conv_thr: float = 1.0e-8,
    max_iterations: int = 30,
    gmres_maxiter: int = 20,
    forcing: float = 0.1,
    precondition=None,
    jvp: str = "finite-difference",
    steps_already_taken: int = 0,
    verbose: bool = False,
) -> NewtonKrylovResult:
    """Drive ``r(x)`` to zero by inexact Newton with a GMRES inner solve.

    ``accuracy_of(r, x)`` is the *same* measure the mixing loop converges on --
    QE's ``dr2``, the Hartree energy of the residual -- so ``conv_thr`` means
    here what it means in a ``pw.x`` input and the two solvers can be compared
    at all. It takes the state ``x`` beside the residual ``r`` because a
    feedback field's share is weighted by the field the state holds
    (``ScfResidual.field_accuracy``).

    ``forcing`` is the inexact-Newton forcing term: the inner solve is only
    asked for a relative residual of ``forcing``, because an exactly solved
    Newton direction is wasted while ``x`` is still far from the root. It is the
    single most important knob for the cost, since GMRES iterations *are* the
    cost.

    ``precondition`` is applied to the Krylov system, and Kerker is the natural
    choice: it is an approximation to ``-J^-1``, which is precisely what a
    preconditioner should be. **This is the honest comparison** -- a
    preconditioner that helps GMRES helps the mixer too, so both sides of any
    measurement get it.

    ``jvp`` chooses how ``J v`` is formed, and the default is the central
    difference of ``r`` for a measured reason (P148, `notes/DEGENERACY-NEXT.md`
    item 4). ``'autodiff'`` differentiates Davidson as implemented, and that
    action is the Jacobian only on a cell whose subspace holds no degenerate
    block: norm-conserving silicon at one generic k-point agrees with the
    central difference to 1.5e-8, following its ``eps^2`` law, while the same
    cell with ``Gamma`` in the mesh (a threefold block) was 28 per cent off and
    ``NaN`` at ``ethr = 1e-13`` before the subspace ``eigh``'s tangent was
    regularised (``solvers/subspace.py``) and is 1.2e-2 off warm after it,
    growing with the number of Davidson steps (1.9x to 4.7x on the eigensolver
    alone, 119x from a cold start), because one member's correction vector on
    a degenerate block is not a smooth function of the density (``OPEN.md``
    Part XXVI). The finite difference is a derivative wherever the state is
    single-valued, which since P148's averaged ultrasoft ``becsum`` includes a
    fixed filling that cuts a degenerate shell with the point group on: there
    it converges in four undamped steps to 5e-12 (38 ``F`` evaluations) where
    the autodiff arm takes five (26 JVPs, each about two to three ``F``).

    ``'sternheimer'`` does not differentiate the eigensolver at all
    (:class:`~defuqumat.scf.residual.SternheimerJacobian`, ``PLAN.md`` P150): the
    action is the static response of the eigenpairs of ``H[x]``, one
    Sternheimer solve per matvec after one warm diagonalisation per outer
    iteration, and it does not depend on the basis the eigensolver returned
    inside a degenerate block. Refused with a Hubbard U, a meta-GGA, a
    magnetic noncollinear run, a streamed store and a fixed filling that cuts
    a degenerate shell. **It also showed what the two other backends agree
    on**: both differentiate the *warm-started* map, whose Davidson stops once
    the eigenvalues have settled, and that map's derivative is 5.3e-2 of
    ``|dF v|`` off the exact-eigenpair one on the generic silicon cell above
    (3.4e-3 to 0.40 on ten more), where the Sternheimer action agrees with a
    cold, tight difference to 4.6e-7. The default stays the finite difference
    because it is faster on every cell measured (P150).
    """
    if jvp not in ("autodiff", "finite-difference", "sternheimer"):
        raise ValueError(f"unknown jvp backend {jvp!r}")
    if jvp == "autodiff" and getattr(
        getattr(residual, "calculation", None), "magnetic_field", None
    ) is not None:
        # I73: not measured under a field. On the one-k-point iron cell at the
        # mixing fixed point it reads 0.53 to 0.81 of |J v| off the central
        # difference with b_field(3) = 0.01, and 1.7 to 2.0 off without it, so
        # the disagreement is not the field's, but no number backs this regime.
        raise NotImplementedError(
            "jvp='autodiff' under a magnetic field or a constrained moment is "
            "not yet checked: every residual-solver number with a field was "
            "taken with the central difference. Use jvp='finite-difference' "
            "(the default)"
        )

    x = np.asarray(x0, dtype=float).copy()
    psi = psi0
    # Warm-up evaluations of ``F`` taken by the caller before the solve started
    # count here too: ``steps`` is what a mixing run's iteration count is
    # compared against, and a currency that quietly omits part of the bill is
    # worse than no currency.
    counters = {"steps": int(steps_already_taken), "jvps": 0}
    history = []

    def evaluate(y, warm):
        counters["steps"] += 1
        r, psi_new = residual.residual(y, warm)
        return np.asarray(r, dtype=float), psi_new

    r, psi = evaluate(x, psi)
    accuracy = float(accuracy_of(r, x))
    converged = accuracy < conv_thr
    # The starting point is recorded so that a convergence history plotted
    # against ``steps`` begins where the mixer's does, and so that a solve that
    # converges before its first outer iteration is still visible as something
    # rather than as an empty history.
    history.append(
        {"iteration": 0, "accuracy": accuracy, "step": 0.0, "gmres": 0,
         "steps": counters["steps"], "jvps": 0, "seconds": 0.0}
    )

    for iteration in range(1, max_iterations + 1):
        if converged:
            break
        started = time.perf_counter()

        # ``psi`` is the converged eigenvector at the current ``x``, so it is
        # the right warm start for every JVP taken here: ``x`` does not move
        # during the inner solve.
        warm = psi

        def matvec(v):
            counters["jvps"] += 1
            if jvp == "sternheimer":
                return jacobian(v)
            if jvp == "autodiff":
                out, _ = residual.jvp(x, v, warm)
                return np.asarray(out, dtype=float)
            counters["steps"] += 2
            return np.asarray(
                residual.jvp_finite_difference(x, v, warm), dtype=float
            )

        if not np.all(np.isfinite(r)):
            # The *residual* rather than the direction, which the guard below
            # catches. What puts a NaN here is a step that left the region where
            # ``F`` is a function of the density at all. The suspect this guard
            # was written for is fixed occupations whose filled/empty boundary
            # cuts a **degenerate multiplet** -- which member of the multiplet
            # the eigensolver hands back is arbitrary -- and the account of it
            # here used to say that the oxygen atom at ``tot_magnetization = 2``
            # (one electron in a threefold 2p shell) "dies at the first line
            # search". **Measured on 2026-10-14 (PLAN.md P147 and P148), it
            # does not, under either Jacobian.** With the point group on,
            # ``sym_rho`` averages the occupied member into the spherical shell,
            # so ``F`` is single-valued in its density rows; its ``becsum`` rows
            # were not (the raw one-k sum moved by 0.34 on a maximum of 1.11
            # under a rotation of the shell, the density by 9.1e-8), which is
            # why the finite-difference JVP once reached an accuracy of 7e33 at
            # the first step, without a NaN. Since P148 the ultrasoft ``becsum``
            # carries the same group average as PAW's and the state is
            # single-valued to the eigensolver's accuracy on the states
            # (4.1e-7): both Jacobians then converge on that atom in four and
            # five undamped Newton steps, to 5.2e-12 and 2.3e-12. So a cut shell
            # is not what reaches this branch; it is diagnosed where it is made
            # (``SCFResult.cut_gap``, ``warn_about_a_cut_shell``), the band-
            # resolved quantities and the response on it stay gauge dependent,
            # and their cure is a smearing. The sentence about a *gapped* fixed
            # LSDA cell agreeing with the mixer to 5e-12 stands.
            raise FloatingPointError(
                "the SCF residual is not finite, so there is no Newton step to "
                "take: a step left the region where the density map is a "
                "function of the density. Read the SCF's own diagnostics first "
                "(a RuntimeWarning that occupations='fixed' cuts a degenerate "
                "multiplet, SCFResult.cut_gap, names the one case measured so "
                "far, which both Jacobians now converge), and fall back to "
                "scf_solver='mixing' or a smearing"
            )
        if jvp == "sternheimer":
            # After the guard, so that a non-finite residual raises its own
            # message rather than a diagonalisation of it: one warm solve at
            # ``x``, counted as an evaluation of ``F``, and then every matvec
            # is one Sternheimer solve.
            counters["steps"] += 1
            jacobian = residual.sternheimer_jacobian(x, warm)
        operator = LinearOperator((x.size, x.size), matvec=matvec, dtype=float)
        # ``-P`` rather than ``P``: the Jacobian of ``r = F - x`` is close to
        # ``-1`` where the iteration is well behaved, so the approximate inverse
        # is the *negative* of the mixer's preconditioner. Getting this sign
        # wrong does not fail -- GMRES simply makes no progress.
        inverse = (
            None if precondition is None
            else LinearOperator((x.size, x.size),
                                matvec=lambda v: -np.asarray(precondition(v), dtype=float),
                                dtype=float)
        )
        direction, info = lgmres(
            operator, -r, rtol=forcing, atol=0.0, maxiter=gmres_maxiter, M=inverse,
        )
        if not np.all(np.isfinite(direction)):
            raise FloatingPointError(
                "the Newton direction is not finite; the Jacobian action has "
                "diverged. This is what an unconverged inner eigensolver looks "
                "like from here -- tighten ethr"
            )

        # Backtracking on the residual norm. A full Newton step is right near
        # the root and can be wild far from it, and the density is a quantity
        # with a sign: an overshoot that drives it negative makes the next
        # exchange-correlation evaluation meaningless rather than merely worse.
        norm = float(np.linalg.norm(r))
        length = 1.0
        for _ in range(4):
            # ``taken`` is the length of the step ``trial`` actually is, which is
            # not ``length`` once the loop has run out: the halving at the foot
            # of the body moves ``length`` past the last point evaluated, so
            # reporting ``length`` would name a step the solver never took.
            taken = length
            trial = x + taken * direction
            r_trial, psi_trial = evaluate(trial, warm)
            if float(np.linalg.norm(r_trial)) < (1.0 - 1.0e-4 * taken) * norm:
                break
            length *= 0.5
        else:
            # Four halvings and still no decrease. The shortest step tried is
            # taken anyway rather than raising: the residual norm is not the
            # quantity being converged (``accuracy_of`` is), and a Newton
            # direction from an inexact inner solve can fail this test while
            # still being an improvement in the measure that decides
            # convergence. What must not happen is silence, so it is reported in
            # ``history`` as ``step`` = 0.125 and the caller can see the solver
            # was crawling.
            pass
        x, r, psi = trial, r_trial, psi_trial

        accuracy = float(accuracy_of(r, x))
        converged = accuracy < conv_thr
        history.append(
            {"iteration": iteration, "accuracy": accuracy, "norm": float(np.linalg.norm(r)),
             "step": taken,
             "gmres": info, "steps": counters["steps"], "jvps": counters["jvps"],
             "seconds": time.perf_counter() - started}
        )
        if verbose:
            print(f"  newton {iteration:3d}   accuracy = {accuracy:.2e}   "
                  f"lambda = {taken:.3f}   F evaluations = {counters['steps']:4d}   "
                  f"jvps = {counters['jvps']:4d}   {history[-1]['seconds']:.1f}s",
                  flush=True)

    return NewtonKrylovResult(
        x=x, psi=psi, converged=converged, accuracy=accuracy,
        iterations=len(history) - 1, steps=counters["steps"], jvps=counters["jvps"],
        history=history,
    )


#: Name -> solver of the SCF fixed point, as written in ``scf_solver``.
#: ``"mixing"`` is the loop in ``driver.py`` and is the default; it is in the
#: registry as a name rather than a function because it is not written as a
#: solver of a residual at all.
SCF_SOLVERS = {
    "mixing": None,
    "newton-krylov": newton_krylov,
    "newton_krylov": newton_krylov,
}


def get_scf_solver(name: str):
    try:
        return SCF_SOLVERS[name.lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown scf_solver {name!r}; expected one of {sorted(SCF_SOLVERS)}"
        ) from error
