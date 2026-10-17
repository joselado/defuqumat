"""The SCF as a root-find: the residual whose zero is self-consistency.

``scf/driver.py`` solves ``rho = F(rho)`` by iterating ``F`` and damping the
iteration with a mixer. This module writes the same fixed point as a *residual*

    r(x) = F(x) - x,   x = the packed (rho, becsum) state,

so that a solver which knows the derivative of ``r`` can drive it to zero
instead. ``F`` is one SCF iteration with every host-side decision frozen -- one
diagonalisation at a fixed ``ethr``, no retry, no ``ethr`` schedule, no
``reducebf`` decay (refused), and a feedback field carried in the state rather
than stepped between iterations -- which is what makes it a *function* rather
than a step of a loop.

**Calibration, because this is easy to oversell.** Anderson mixing already is a
quasi-Newton method on this residual: ``mixing.py`` builds a secant
approximation to ``dr/dx`` from the residual history and takes the Newton step
inside its span, and QE's Broyden (``mix_rho.f90``) is the same idea. Nothing
here replaces iteration with a closed form; what changes is that the Jacobian is
*exact* rather than fitted, which matters only where the fit is bad.

**Why the Jacobian comes from ``jax.jvp`` and not from a tape.** ``r`` is built
from ``Calculation``'s own methods, so it is one pure JAX function from the
density to the density, Davidson's ``lax.while_loop`` included. Forward-mode
differentiation of a ``while_loop`` is supported (reverse-mode is not, and its
tape would be ``n_davidson_steps * nvecx * nbnd * npwx`` complex numbers -- see
``PLAN.md`` P22), and forward mode is all a Krylov solver needs: it asks only for
``J v``.

Two things had to be written down rather than differentiated:

* **The Fermi level.** ``bisect_fermi`` is a bisection, and a bisection's
  tangent is zero or garbage. It now carries a ``custom_jvp`` with the implicit
  derivative of ``N(E_F) = nelec`` (``scf/occupations.py``), which is the
  Fermi-shift term of metallic linear response.
* **Davidson, half way.** Davidson is differentiated as implemented, so the
  Jacobian is the Jacobian of the *approximate* eigensolver. This used to say
  "accurate to about ``ethr``", and that is measured true only on a cell whose
  subspace holds no degenerate block (P148, `notes/DEGENERACY-NEXT.md` item 4):
  norm-conserving silicon at one generic k-point agrees with a central
  difference of ``r`` to 1.5e-8, following its ``eps^2`` law. With a degenerate
  block (the same cell with ``Gamma`` in the mesh) it was 28 per cent off warm,
  10x off cold and ``NaN`` at ``ethr = 1e-13``, from JAX's ``eigh`` tangent,
  ``1/(w_j - w_i)`` unprotected on Ritz values the block splits only at the
  solver's noise; the kernel in ``solvers/subspace.py`` takes that to 1.2e-2
  warm, and the rest, growing with the Davidson step count, is the gauge of one
  member's correction vector (``OPEN.md`` Part XXVI). The exact route is the
  implicit tangent of the converged eigenproblem, the Sternheimer response
  ``PLAN.md`` P22c planned, and it is :class:`SternheimerJacobian` (P150).
* **What the warm start does to ``F``, measured with that route** (P150). The
  sentence below, that ``psi0`` "changes how many iterations the solver takes
  and not what it converges to", is true of the fixed point and false of the
  derivative. From a converged start Davidson stops once the eigenvalues have
  settled, which is second order in the change of ``x``, while the states are
  first order, so the warm map has a different derivative from the
  exact-eigenpair one: its central difference was 5.3e-2 of ``|dF v|`` off on
  silicon at one generic k-point (with autodiff agreeing with it to 1.5e-3,
  both derivatives of the same warm map) and 3.4e-3 to 0.40 on ten other
  cells, while the Sternheimer action agreed with a cold, tight difference to
  2e-7 to 1e-5 (on the spinor cell to the scatter of the reference,
  about 5e-4). Newton converges under either,
  since the fixed point is the same; the default stays the finite difference
  on wall time (P150).

**DFT+U joins the state rather than being refused.** ``ns`` is not a function of
the density -- the Hubbard potential is built from it before the Hamiltonian
exists -- so ``mix_rho.f90`` carries it inside ``mix_type`` and the mixing loop
mixes it alongside ``rho`` and ``becsum``. A root-finder solves for it on the
same footing, for the same reason, and nothing about the Jacobian action changes:
``v_hubbard`` is already ``jax.grad`` of the Hubbard energy (P20) and the
projectors ``wfcU`` are fixed while the atoms are.

**An applied field or a penalty constraint is part of ``F`` already** (I73):
``potential(rho)`` applies ``calculation.magnetic_field`` at full strength and a
penalty's field is ``jax.grad`` of the penalty, so both are functions of the
density. A uniform field and the ``atomic``, ``atomic direction`` and ``total``
penalties run, each measured: the field in a collinear run, the penalties in
a noncollinear one (the two together are refused at input, as ``pw.x`` does).
A ``LOCAL_MAGNETIC_FIELDS`` card, the ``total direction`` and ``atomic texture``
penalties, a noncollinear uniform field, a collinear penalty and
``jvp='autodiff'`` under any field (``solvers.py``) are refused until they are
measured.

**A feedback (fixed-spin-moment) field joins the state, as ``ns`` did** (I73,
the user's decision of 2026-10-16). The mixing loop moves it between
iterations by Elk's rule ``B <- B - tau (m(rho_out) - m_fix)``, so it is a
second unknown beside the density; carried in ``x``, with that rule as its row
of ``F`` (:meth:`MagneticField.elk_driven`), the residual's field rows are
``-tau (m - m_fix)`` and vanish exactly where the moment of the self-consistent
density is the target, which is the loop's own fixed point. ``tau`` is kept in
the row rather than solving ``m - m_fix = 0`` directly because it is Elk's
estimate of ``1/chi`` (0.02 against a measured 0.022 on ``fe-fsm.in``), so the
field block of the Jacobian sits near ``-1``, the scale of the density block,
and the identity the Kerker preconditioner applies past the density is right
for it. The convergence measure adds ``|dm| (|B| + tau |dm|)``, ``dm = m -
m_fix``, a Rydberg number like ``rho_ddot`` that bounds the first-order error
``B . dm`` of the reported total (:meth:`ScfResidual.field_accuracy`). The secant
controller (``fsm_update = 'secant'``) has memory and is refused; its zero is
the same field. **Refused rather than approximated:** ``reducebf < 1``, whose
decay ``s -> reducebf s`` has the one root ``s = 0``, the field-free problem,
while the loop stops at whatever field is left, so the two answers differ by
that field rather than by ``conv_thr``.

A spin spiral runs (``notes/incompat/I72.md``): its map is the mixing loop's
own, with no term of its own here, except under a Hubbard U, which is refused:
a spinor ``ns`` is complex and :meth:`ScfResidual.pack` keeps its real part
only. A spiral under a meta-GGA is refused too, as not measured, and so is a
spiral under a field or a constraint, a pair neither I72 nor I73 measured.

**Tetrahedron and ``from_input`` occupations are in** (I71). Both were refused
on the grounds that the first is "built on the host" and the second "is not a
function of the eigenvalues at all"; neither stops ``F`` being a function of
``x``. The tetrahedra are a table of the k-grid and the group, built once like
``wk`` (``Calculation.tetrahedron_table``), and the weights on them were
already JAX; ``from_input`` is a constant, whose zero tangent is the right one.
What was missing for the autodiff Jacobian was the Fermi level's tangent: the
tetrahedron bisection now carries the implicit derivative of
``N(E_F) = nelec``, as ``bisect_fermi`` does for a smearing. The Sternheimer
Jacobian still refuses both (``require_a_sternheimer_regime``): it needs the
tetrahedron response weight ``dfpt_tetra_beta``, and for ``from_input`` a
projector for a filling that is not a function of the levels.

Memory: the packed state is ``nspin * dense_grid`` floats plus ``becsum``,
``ns`` and, under a feedback constraint, the driven field (a few floats), and a Krylov subspace holds a few tens of those -- tens of MB,
negligible beside the wavefunctions. One ``jvp`` doubles the wavefunction
working set for the duration of the call, because the tangent ``psi`` has the
same shape as ``psi``; that is the number to watch, and it is why the JVP goes
through the same ``k_batch`` dial the primal does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.eager import compiled
from defuqumat.scf.fields import FEEDBACK, FEEDBACK_ATOMIC
from defuqumat.scf.occupations import (
    fixed_occupations, input_occupations, smeared_occupations,
)
from defuqumat.scf.tetrahedra import tetrahedron_occupations_spin

__all__ = ["PooledScfResidual", "ScfResidual", "SternheimerJacobian", "make_residual"]


@dataclass(frozen=True)
class ScfResidual:
    """``F`` and ``r = F - x`` as flat real vectors, plus the packing.

    ``psi0`` is a warm start for Davidson carried *outside* the differentiated
    function: it changes how many iterations the solver takes and not what it
    converges to, so it is a constant of ``F`` at every point where ``F`` is
    evaluated, and updating it between outer iterations is legitimate. It is
    also the reason ``step`` returns the wavefunctions: the caller feeds them
    back.
    """

    calculation: object
    nbnd: int
    ethr: float
    shapes: tuple
    size: int
    #: Shape of the DFT+U occupation matrix, or ``None`` without a ``U``.
    #: ``ns`` is part of the *mixed state* rather than a function of the
    #: density (``mix_rho.f90`` carries it in ``mix_type`` for that reason), so
    #: it is part of the state a root-finder solves for on exactly the same
    #: footing.
    ns_shape: tuple | None = None
    #: Shape of the kinetic energy density, or ``None`` when the functional is
    #: not a meta-GGA. ``tau`` is packed with the density and the fixed point is
    #: sought in ``(rho, tau)`` jointly, because a root-finder needs ``F`` to be
    #: a function of its argument and a lagged ``tau`` makes it depend on
    #: something outside it. What is gained by that is the Jacobian's
    #: ``d v / d tau`` block, which is where a meta-GGA's convergence trouble
    #: lives and which no density mixer can model.
    #:
    #: The sentence that stood here until 2026-09-20 called this "a deviation
    #: from QE and it is the point", on the grounds that "QE recomputes
    #: ``kin_r`` from the output states and never mixes it". Both halves are
    #: false: ``mix_type`` carries ``kin_g`` under
    #: ``IF (xclib_dft_is('meta') .OR. lxdm)`` and ``assign_mix_to_scf_type``
    #: rebuilds ``kin_r`` from the mixed copy (``scf_mod.f90:368-375``). The
    #: mixing loop here now mixes it too, so the two routes agree with each
    #: other and with ``pw.x``, and what is left of the deviation is the
    #: Jacobian block, which is a property of the solver rather than of the
    #: state.
    tau_shape: tuple | None = None
    #: The loop's starting :class:`~defuqumat.scf.fields.MagneticField` under a
    #: feedback (fixed-spin-moment) constraint, and the shape of the field it
    #: drives; both ``None`` otherwise. The driven field joins the state as
    #: ``ns`` did, last in the vector, and its row of ``F`` is Elk's update
    #: ``B - tau (m(rho_out) - m_fix)`` (:meth:`MagneticField.elk_driven`), so
    #: the residual's field rows are ``-tau (m - m_fix)`` and vanish exactly
    #: where the moment of the self-consistent density is the target (I73).
    feedback_field: object = None
    field_shape: tuple | None = None

    # ---- packing -------------------------------------------------------
    def pack(self, rho, becsum_, ns=None, tau=None, driven=None) -> np.ndarray:
        """The mixed state as one flat real vector, exactly ``_mix``'s packing.

        ``driven`` is the feedback field, the starting field's when ``None``.
        """
        flat = [np.asarray(rho, dtype=float).ravel()]
        for part in becsum_:
            if part is not None:
                flat.append(np.asarray(part, dtype=float).ravel())
        if self.ns_shape is not None:
            if ns is None:
                raise ValueError("this calculation has a Hubbard U; ns is part of the state")
            flat.append(np.asarray(ns, dtype=float).ravel())
        if self.tau_shape is not None:
            if tau is None:
                raise ValueError(
                    "this calculation runs a meta-GGA; tau is part of the state"
                )
            flat.append(np.asarray(tau, dtype=float).ravel())
        if self.field_shape is not None:
            if driven is None:
                driven = self.feedback_field.driven()
            flat.append(np.asarray(driven, dtype=float).ravel())
        return np.concatenate(flat)

    def unpack(self, x):
        """The inverse of :meth:`pack`. Returns ``(rho, becsum, ns, tau)``."""
        x = jnp.asarray(x)
        rho_shape = self.shapes[0]
        n = int(np.prod(rho_shape))
        rho = x[:n].reshape(rho_shape)
        offset = n
        becsum_ = []
        for shape in self.shapes[1:]:
            if shape is None:
                becsum_.append(None)
                continue
            n = int(np.prod(shape))
            becsum_.append(x[offset : offset + n].reshape(shape))
            offset += n
        ns = None
        if self.ns_shape is not None:
            n = int(np.prod(self.ns_shape))
            ns = x[offset : offset + n].reshape(self.ns_shape)
            offset += n
        tau = None
        if self.tau_shape is not None:
            n = int(np.prod(self.tau_shape))
            tau = x[offset : offset + n].reshape(self.tau_shape)
        return rho, tuple(becsum_), ns, tau

    def unpack_field(self, x):
        """The driven field's block of ``x``, or ``None`` without a feedback scheme.

        Last in the vector, so :meth:`unpack` reads the same offsets with or
        without it.
        """
        if self.field_shape is None:
            return None
        n = int(np.prod(self.field_shape))
        return jnp.asarray(x)[self.size - n:].reshape(self.field_shape)

    def field_accuracy(self, r, x) -> float:
        """The feedback field's share of the convergence measure, in Ry.

        ``|dm| (|B| + tau |dm|)``, with ``B`` the field ``x`` holds and ``dm =
        m - m_fix`` read off the field rows of ``r``, which are ``-tau dm``;
        both norms are over the whole driven leaf (every atom and component).
        The reported total leaves out the Zeeman term and along the
        constrained family ``dE/dm = B``, so a state that stopped ``dm`` off
        its target reads ``B . dm`` off in the total, first order, and
        ``|B . dm| <= |B| |dm|``: the first term bounds that error in the
        quantity ``conv_thr`` is a statement about (the user's decision,
        2026-10-17). The second, ``tau |dm|^2``, is the field's step to the
        target, ``|B - B'|`` with ``B' = B - tau dm`` Elk's update, times
        ``|dm|``, so the sum bounds ``|B'| |dm|`` and stays above zero at
        ``B = 0``, where ``|B| |dm|`` alone would pass a state converged in
        the density at no field and any moment. At ``conv_thr = 1e-11`` and
        the 1.1e-2 Ry field of ``fe-fsm.in`` it asks the moment to 9e-10 mu_B
        and Newton stopped at 6.2e-11, where the loop's own test
        (``FSM_TOLERANCE``) is 1e-3. On the canted iron pair (8 k-points,
        fields of 2.6e-2 to 3.0e-2 Ry) Newton does not reach it: it stalls at
        7e-10 to 7e-9 Ry, the moment 1.7e-8 to 1.6e-7 mu_B off, and hands over
        to the loop (``notes/incompat/I73.md``, "The feedback field").
        Zero without a feedback scheme. Host-side, never traced.
        """
        if self.field_shape is None:
            return 0.0
        tau = float(self.feedback_field.penalty)
        moment_error = float(np.linalg.norm(
            np.asarray(self.unpack_field(r), dtype=float))) / tau
        field = float(np.linalg.norm(np.asarray(self.unpack_field(x), dtype=float)))
        return moment_error * (field + tau * moment_error)

    def field_at(self, x):
        """The :class:`MagneticField` whose driven leaf is ``x``'s, or ``None``."""
        if self.field_shape is None:
            return None
        return self.feedback_field.with_driven(self.unpack_field(x))

    # ---- the map -------------------------------------------------------
    def step(self, x, psi0):
        """One SCF iteration as a pure function: ``x -> F(x)``, and the ``psi``."""
        fx, psi, _ = self.step_states(x, psi0)
        return fx, psi

    def step_states(self, x, psi0):
        """:meth:`step`, with the eigenvalues as well: ``(F(x), psi, eigenvalues)``.

        For a caller outside the differentiated map that needs more of the
        iteration than ``F``: the Newton-Krylov warm-up under
        ``warmup_mixing = 'ldos'`` builds the LDOS at the Fermi level from these
        states and eigenvalues. :meth:`jvp` differentiates :meth:`step`, whose
        first return is this one's.
        """
        calculation = self.calculation
        rho, becsum_, ns, tau = self.unpack(x)
        # ``None`` reads ``calculation.magnetic_field``, which is right for every
        # field the loop holds fixed; a feedback field is ``x``'s own.
        field = self.field_at(x)
        potential = calculation.potential(rho, field=field, tau=tau)
        # The spheres must use the *same* Tran-Blaha ``c`` the grid used, and
        # only the grid can compute it.
        _, ddd_paw = calculation.onecenter(
            becsum_, None if tau is None else potential.meta_c
        )
        hubbard_terms = None
        if ns is not None:
            # ``v_hubbard`` is ``jax.grad`` of the Hubbard energy (P20), so this
            # whole branch is differentiable without anything being added for it.
            _, _, hubbard_terms = calculation.hubbard_terms(ns)
        hamiltonians = calculation.hamiltonian(potential.v_scf, ddd_paw, hubbard_terms)
        eigenvalues, psi = calculation.diagonalize(
            hamiltonians, self.nbnd, psi0, self.ethr
        )
        wg = _weights(calculation, eigenvalues)
        becsum_out = calculation.becsum(psi, wg)
        rho_out = calculation.density(psi, wg, becsum_out)
        ns_out = None if ns is None else calculation.occupation_matrix(psi, wg)
        tau_out = None if tau is None else calculation.kinetic_energy_density(psi, wg)
        # The moment read off the *output* density, as the loop's feedback reads
        # it (``run_scf``, after the mix) and as Elk's ``bfieldfsm`` reads
        # ``mommt``.
        driven_out = (None if field is None
                      else field.elk_driven(rho_out, calculation.system.cell))
        return (self.flatten(rho_out, becsum_out, ns_out, tau_out, driven_out),
                psi, eigenvalues)

    def flatten(self, rho, becsum_, ns=None, tau=None, driven=None):
        """:meth:`pack`, inside a trace -- ``jnp`` rather than ``np``."""
        flat = [jnp.reshape(jnp.real(rho), (-1,))]
        for part in becsum_:
            if part is not None:
                flat.append(jnp.reshape(jnp.real(part), (-1,)))
        if ns is not None:
            flat.append(jnp.reshape(jnp.real(ns), (-1,)))
        if tau is not None:
            flat.append(jnp.reshape(jnp.real(tau), (-1,)))
        if driven is not None:
            flat.append(jnp.reshape(jnp.real(driven), (-1,)))
        return jnp.concatenate(flat)

    def residual(self, x, psi0):
        """``r(x) = F(x) - x``, and the wavefunctions ``F`` produced."""
        fx, psi = self.step(x, psi0)
        return fx - jnp.asarray(x), psi

    def jvp(self, x, v, psi0):
        """``J v = d r / d x . v``, by forward-mode differentiation of ``F``.

        ``psi0`` is passed as a closed-over constant, so no tangent flows
        through the warm start -- which is right: it is not part of the map
        being differentiated, only of how fast it is evaluated.
        """
        def f(y):
            return self.step(y, psi0)[0]

        primal, tangent = jax.jvp(f, (jnp.asarray(x),), (jnp.asarray(v),))
        return np.asarray(tangent - jnp.asarray(v)), np.asarray(primal)

    def jvp_finite_difference(self, x, v, psi0, epsilon=None):
        """The same ``J v``, by a central difference of ``r``.

        This is the *test* of :meth:`jvp`, in the same spirit as QE's
        hand-derived forces standing as the test of the autodiff ones: the two
        share no machinery, so agreeing means both are right. It is also the
        fallback for a regime where the autodiff route is refused, and it costs
        two evaluations of ``F`` against one ``jvp``.
        """
        x = np.asarray(x)
        v = np.asarray(v)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return np.zeros_like(x)
        if epsilon is None:
            # Nocedal-Wright's forward-difference step, scaled by the state's
            # own magnitude so that a nearly converged density is not perturbed
            # below the eigensolver's own noise floor.
            epsilon = 1.0e-6 * max(1.0, float(np.linalg.norm(x))) / norm
        plus, _ = self.residual(x + epsilon * v, psi0)
        minus, _ = self.residual(x - epsilon * v, psi0)
        return np.asarray((plus - minus) / (2.0 * epsilon))

    # ---- the implicit Jacobian -----------------------------------------
    def sternheimer_jacobian(self, x, psi0, threshold=None):
        """``v -> J v`` at ``x``, from the response of the converged eigenproblem.

        Built once per point and applied to as many directions as GMRES asks
        for, since the eigenpairs of ``H[x]``, the solver and the density of
        states at the Fermi level depend on ``x`` alone. ``psi0`` is the warm
        start: the converged states at ``x``, so the diagonalisation here is
        ``F``'s own, one Davidson step from the answer and not bit for bit the
        eigenpairs the last evaluation of ``F`` returned. See
        :class:`SternheimerJacobian` for what is differentiated.
        """
        calculation = self.calculation
        self._refuse_sternheimer(psi0)
        rho, becsum_, _, _ = self.unpack(x)
        potential = calculation.potential(rho)
        _, ddd_paw = calculation.onecenter(becsum_)
        hamiltonians = calculation.hamiltonian(potential.v_scf, ddd_paw)
        eigenvalues, psi = calculation.diagonalize(
            hamiltonians, self.nbnd, psi0, self.ethr
        )
        return SternheimerJacobian(self, x, eigenvalues, psi, threshold,
                                   built=(potential, hamiltonians))

    def sternheimer_jacobian_at(self, x, eigenvalues, psi, threshold=None):
        """:meth:`sternheimer_jacobian` on eigenpairs of ``H[x]`` the caller has.

        The gauge test's entry: the same eigenpairs rotated inside a degenerate
        block give the same ``J v``.
        """
        self._refuse_sternheimer(psi)
        return SternheimerJacobian(self, x, eigenvalues, psi, threshold)

    def _refuse_sternheimer(self, psi) -> None:
        """Every regime the implicit Jacobian does not carry, by name."""
        from defuqumat.response.sternheimer import require_a_sternheimer_regime
        from defuqumat.scf.streaming import is_host_store

        calculation = self.calculation
        if self.ns_shape is not None:
            raise NotImplementedError(
                "jvp='sternheimer' with a Hubbard U is not implemented: ns is part "
                "of the state and its perturbation of H (adddvhubscf.f90) is not a "
                "local potential. Use jvp='finite-difference'"
            )
        if self.tau_shape is not None:
            raise NotImplementedError(
                "jvp='sternheimer' with a meta-GGA is not implemented: tau is part "
                "of the state and dH from dtau is a kinetic-like operator, not a "
                "local potential. Use jvp='finite-difference'"
            )
        if calculation.noncolin and calculation.nspin_mag == 4:
            raise NotImplementedError(
                "jvp='sternheimer' on a magnetic noncollinear run is not measured: "
                "no cell checks its four-component response against a central "
                "difference of F. Use jvp='finite-difference'"
            )
        if psi is not None and is_host_store(psi):
            raise NotImplementedError(
                "jvp='sternheimer' on a streamed store is not implemented: the "
                "density of a host store is walked by stream_densities, which has "
                "no derivative. Use jvp='finite-difference'"
            )
        # The solve and its assembly are this one's own, so the response
        # stack's per-quantity gates (spin, spinors, metals, half-sphere
        # storage) are lifted; what stays is every regime whose operator or
        # occupations are not here (fields, tetrahedra, two smeared levels).
        require_a_sternheimer_regime(
            calculation, metals=True, spin_polarized=True, gamma_ok=True,
            noncollinear=True,
        )


#: The command codes of :class:`PooledScfResidual`'s protocol, the first entry
#: of the vector rank 0 broadcasts before every collective step.
_EVALUATE, _FINISH, _ABORT = 1.0, 0.0, -1.0


@dataclass(frozen=True)
class PooledScfResidual(ScfResidual):
    """``F`` under k-point pools, each pool solving its own rows.

    The same map as :meth:`ScfResidual.step`, evaluated the way the mixing
    loop's pooled iteration is: each pool diagonalises its rows of a streamed
    :class:`~defuqumat.parallel.PoolStore` at the fixed ``ethr``, the
    eigenvalues are gathered so that the occupations see the whole set, and
    the raw sums over k are all-reduced before they are finished
    (:func:`~defuqumat.scf.streaming.stream_densities`). It is a host-side
    function and not a traceable one, so only the central-difference Jacobian
    (``jvp = 'finite-difference'``, the default) is available; ``autodiff`` and
    ``sternheimer`` raise by name. Nothing here is differentiated, which is why
    the ``np.asarray`` calls in :meth:`_evaluate` are harmless.

    **Rank 0 runs the solver and the other pools serve it.** GMRES, the line
    search and the convergence test are host decisions, and pools whose thread
    counts differ can differ in the last bit, so a solver run on every rank
    could take one more GMRES iteration on one of them and leave the others
    waiting at a collective. So the solver runs on rank 0 alone, every
    evaluation of ``F`` is a collective it opens by broadcasting
    ``[command, warm, x]`` (every pool evaluates at rank 0's ``x`` to the
    bit), and the others sit in :meth:`serve` until :meth:`finish` hands them
    the result: the pooled loop's rule, every decision is rank 0's, carried to
    the whole solver.

    **The warm start is an integer token, the same on every pool.** Each
    evaluation is collective, so a counter numbers the stores alike everywhere,
    and rank 0's solver passes the tokens where the serial one passes arrays.
    A pool keeps two stores, the warm start just used and the one just made,
    which is all ``newton_krylov`` reads: every matvec and line-search trial of
    an outer iteration starts from the accepted point's states, and an
    accepted trial is always the last one evaluated. The warm store is
    **copied** before the solve, since ``stream_diagonalize`` writes in place
    and the central difference evaluates ``x + eps v`` and ``x - eps v`` from
    the same start (the warm start changes ``F``'s derivative, module
    docstring).

    Memory: three of the pool's stores, the two kept and the copy being solved.
    """

    pools: object = None
    rows: object = None
    layout: object = None
    held: dict = field(default_factory=dict, compare=False)

    @classmethod
    def of(cls, residual: ScfResidual, pools, rows, layout) -> "PooledScfResidual":
        if residual.field_shape is not None:
            # ``_evaluate`` builds its potential from the calculation's field
            # and flattens no field row, so a feedback state would be dropped
            # on the pools' side. Refused rather than half carried.
            raise NotImplementedError(
                f"constrained_magnetization = {residual.feedback_field.constraint!r} with "
                "a residual solver under k-point pools is not implemented: the "
                "driven field is part of the residual's state and the pooled "
                "evaluation of F does not carry it. Run without pools "
                "(DEFUQUMAT_POOLS unset) or use scf_solver='mixing'"
            )
        return cls(residual.calculation, residual.nbnd, residual.ethr,
                   residual.shapes, residual.size, residual.ns_shape,
                   residual.tau_shape, pools=pools, rows=np.asarray(rows),
                   layout=layout)

    # ---- the collective evaluation --------------------------------------
    def _evaluate(self, x, warm):
        from defuqumat.parallel import PoolStore
        from defuqumat.scf.streaming import (
            stream_densities, stream_diagonalize, stream_start,
        )

        calculation = self.calculation
        rho, becsum_, ns, tau = self.unpack(x)
        potential = calculation.potential(rho, tau=tau)
        _, ddd_paw = calculation.onecenter(
            becsum_, None if tau is None else potential.meta_c
        )
        hubbard_terms = None
        if ns is not None:
            _, _, hubbard_terms = calculation.hubbard_terms(ns)
        hamiltonians = calculation.hamiltonian(potential.v_scf, ddd_paw, hubbard_terms)
        if warm is None:
            store = stream_start(calculation, hamiltonians, self.nbnd, rows=self.rows)
        else:
            source = self.held[warm]
            store = PoolStore(source.array.copy(), source.rows, source.nk)
        eigenvalues, _, _ = stream_diagonalize(
            calculation, hamiltonians, self.nbnd, store, self.ethr)
        eigenvalues = self.pools.gather_k(
            eigenvalues, calculation.system.kpoints.nk, layout=self.layout)
        wg = np.asarray(_weights(calculation, jnp.asarray(eigenvalues)))
        becsum_out, rho_out, tau_out, ns_out = stream_densities(
            calculation, store, wg, kinetic=tau is not None,
            hubbard=ns is not None, reduce=self.pools.allreduce_sum,
        )
        fx = np.asarray(self.flatten(rho_out, becsum_out, ns_out, tau_out))
        token = self.held.pop("next", 0)
        kept = {} if warm is None else {warm: self.held[warm]}
        self.held.clear()
        self.held.update(kept)
        self.held[token] = store
        self.held["next"] = token + 1
        return fx, token

    def _command(self, code, warm, x):
        vector = np.empty(self.size + 2, dtype=np.float64)
        vector[0] = code
        vector[1] = -1.0 if warm is None else float(warm)
        vector[2:] = np.asarray(x, dtype=np.float64)
        return np.asarray(self.pools.broadcast(vector))

    # ---- rank 0: the solver's view ---------------------------------------
    def step(self, x, psi0):
        """``F(x)`` and a token for the pools' states; a collective rank 0 opens."""
        if self.pools.rank != 0:
            raise RuntimeError("only rank 0 drives a pooled residual; the others serve")
        sent = self._command(_EVALUATE, psi0, x)
        return self._evaluate(sent[2:], psi0)

    def jvp(self, x, v, psi0):
        raise NotImplementedError(
            "jvp='autodiff' is not available with k-point pools: a pool's F is "
            "evaluated on the host through the streamed store, which jax.jvp "
            "cannot trace. Use jvp='finite-difference' (the default)")

    def sternheimer_jacobian(self, x, psi0, threshold=None):
        raise NotImplementedError(
            "jvp='sternheimer' is not available with k-point pools: its "
            "Sternheimer solve reads the whole set's states. Use "
            "jvp='finite-difference' (the default)")

    def store(self, token):
        """The pool's states behind ``token``."""
        return self.held[token]

    def finish(self, result) -> None:
        """Hand the solver's result to the serving pools and release them."""
        self._command(_FINISH, result.psi, result.x)
        self.pools.broadcast(np.asarray(
            [result.accuracy, float(result.converged), result.iterations,
             result.steps, result.jvps], dtype=np.float64))
        result.psi = self.store(result.psi)

    def abort(self) -> None:
        """Release the serving pools after rank 0's solver has raised."""
        self._command(_ABORT, None, np.zeros(self.size))

    # ---- every other pool --------------------------------------------------
    def serve(self):
        """Evaluate ``F`` whenever rank 0 asks, until it finishes; its result."""
        from defuqumat.scf.solvers import NewtonKrylovResult

        while True:
            # The placeholder is overwritten by rank 0's command.
            sent = self._command(_ABORT, None, np.zeros(self.size))
            code, warm = float(sent[0]), int(sent[1])
            warm = None if warm < 0 else warm
            if code == _EVALUATE:
                self._evaluate(sent[2:], warm)
            elif code == _FINISH:
                record = np.asarray(self.pools.broadcast(np.zeros(5, dtype=np.float64)))
                return NewtonKrylovResult(
                    x=np.asarray(sent[2:]), psi=self.store(warm),
                    converged=bool(record[1]), accuracy=float(record[0]),
                    iterations=int(record[2]), steps=int(record[3]),
                    jvps=int(record[4]))
            else:
                raise RuntimeError(
                    "rank 0's residual solver raised, so this pool stops with it; "
                    "rank 0's log has the reason")


class SternheimerJacobian:
    """``J v`` of ``r = F - x`` at one point, without differentiating the eigensolver.

    ``F(x)`` is built from the eigenpairs of ``H[x]``, and the derivative of
    its assembly along ``v`` is the static independent-particle response of
    those eigenpairs to ``dH = dH/dx . v``: one Sternheimer solve
    (:class:`~defuqumat.response.sternheimer.SternheimerSolver`, ``PLAN.md``
    P22c) and one ``jvp`` of ``F``'s own assembly along the first-order states.
    The insulator branch needs only ``P_c``, which no rotation inside the
    occupied manifold changes, and the metal branch carries the divided
    difference ``(f_n - f_m)/(e_n - e_m)`` with its limit ``-f'`` at a
    degeneracy (``_smeared_projection``), so the action does not depend on the
    basis the eigensolver returned inside a degenerate block. What is
    differentiated is the ``ethr -> 0`` limit of ``F``, the exact-eigenpair
    map, not Davidson as implemented (``OPEN.md`` Part XXVI shows that one is
    not differentiable on a degenerate block): it approximates ``F``'s
    derivative to Davidson's ``ethr`` on the eigenpairs and the CG threshold
    on ``dpsi``.

    Term by term against :meth:`ScfResidual.step`: ``dV`` is one ``jvp`` of
    ``potential``; the ultrasoft ``D_ij(V)`` moves through ``int3``
    (:func:`~defuqumat.response.sternheimer.perturbed_coefficients`); PAW's
    one-centre term moves with the *input* ``becsum`` (``paw_response``); ``S``
    holds only the dataset's ``q_ij``, so there is no ``dS``. A smeared Fermi
    level is re-bisected at every ``x``, so the tangent carries ``ef_shift``,
    applied to the states (``fermi_level_shift_states``) so that ``dbecsum``
    gets its share as well as ``drho``. The assembly is ``F``'s own two calls
    (``becsum`` with its group average and ``density`` with ``sym_rho``), both
    linear in the tangent, with ``density_weights`` as the weight: ``wg`` on
    the occupied block for an insulator and ``wk`` for a metal, whose
    occupation is already inside ``dpsi``.

    **Two departures from the response stack's metal solver.** Its level shift
    and its projector mask run up to ``ef + xmax degauss``, and a band above
    that cutoff then loses the diagonal term ``-wk f'_i <i|dV|i>``, which the
    ``alpha_pv`` channel delivers; ``F`` fills every one of its ``nbnd`` bands,
    so here the mask covers them all and ``alpha_pv = 2 (e_max - e_min)`` over
    the whole set, which keeps ``e_j - e_i + alpha_pv > 0`` on every row
    (fable's review of the plan, 2026-10-14: ``w0gauss(-3) = 7e-5`` per band
    otherwise). And the solve sees ``v`` normalised, since the CG's stopping
    test is absolute, and the answer is scaled back.

    Memory: three wavefunction sets, ``psi``, the solver's sliced copy and
    ``dpsi``, plus the CG's four band blocks at the ``k_batch`` dial.
    """

    def __init__(self, residual, x, eigenvalues, psi, threshold=None, built=None):
        from defuqumat.response.sternheimer import (
            THRESHOLD, Smearing, SternheimerSolver, _require_a_gap_at_the_cut,
            occupied_counts,
        )
        from defuqumat.scf.occupations import smearing_order

        calculation = residual.calculation
        system = calculation.system
        self.residual = residual
        self.x = np.asarray(x)
        rho, becsum_, _, _ = residual.unpack(x)
        self.rho, self.becsum = rho, becsum_
        if built is None:
            potential = calculation.potential(rho)
            _, ddd_paw = calculation.onecenter(becsum_)
            hamiltonians = calculation.hamiltonian(potential.v_scf, ddd_paw)
        else:
            potential, hamiltonians = built
        eigenvalues = jnp.asarray(eigenvalues)
        psi = jnp.asarray(psi)
        if eigenvalues.ndim == 2:
            eigenvalues, psi = eigenvalues[None], psi[None]
        self.eigenpairs = (eigenvalues, psi)
        nbnd = int(eigenvalues.shape[-1])

        weights, levels = calculation.occupations(eigenvalues)
        counts = occupied_counts(calculation)
        smearing = None
        if system.occupations == "fixed":
            # No derivative exists where the filling cuts a degenerate shell
            # (``PLAN.md`` P147): which member is occupied is the solver's
            # choice. The response stack checks this only with a spin axis.
            _require_a_gap_at_the_cut(np.asarray(eigenvalues), counts)
        else:
            smearing = Smearing(
                ef=float(levels["fermi_energy"]),
                degauss=float(system.degauss),
                ngauss=smearing_order(system.smearing),
            )
        solver = SternheimerSolver(
            calculation, hamiltonians, psi, eigenvalues, jnp.asarray(weights),
            counts, THRESHOLD if threshold is None else threshold,
            v_scf=potential.v_scf, becsum=becsum_, smearing=smearing,
            kpoint_weights=system.kpoints.weights,
        )
        if smearing is not None:
            solver.projector_mask = jnp.ones(solver.eigenvalues.shape, dtype=bool)
            values = np.asarray(eigenvalues)
            solver.alpha_pv = max(2.0 * float(values.max() - values.min()), 1.0e-2)
            self.levels = solver.local_density_of_states()
        self.solver = solver
        self.smearing = smearing
        keep = solver.nocc
        pad = ((0, 0), (0, 0), (0, nbnd - keep))
        self.weights = jnp.pad(solver.density_weights, pad)
        self.pad = pad + ((0, 0),)
        self.last_iterations = None

    def _dv(self, drho):
        calculation = self.residual.calculation
        return compiled(
            lambda rho, t: jax.jvp(
                lambda q: calculation.potential(q).v_scf, (rho,), (t,))[1],
            self.rho, drho,
        )

    def _assembly(self, tangent):
        calculation = self.residual.calculation
        weights = self.weights

        def assemble(states):
            becsum_ = calculation.becsum(states, weights)
            return calculation.density(states, weights, becsum_), becsum_

        return compiled(
            lambda psi, t: jax.jvp(assemble, (psi,), (t,))[1],
            self.eigenpairs[1], tangent,
        )

    def __call__(self, v):
        from defuqumat.response.sternheimer import paw_response

        residual = self.residual
        calculation = residual.calculation
        solver = self.solver
        v = np.asarray(v, dtype=float)
        norm = float(np.linalg.norm(v))
        if norm == 0.0:
            return np.zeros_like(v)
        drho_in, dbecsum_in, _, _ = residual.unpack(v / norm)
        dv = self._dv(drho_in)
        dddd_paw = paw_response(calculation, dbecsum_in, self.becsum)
        solution = solver.solve(solver.perturbation(dv, dddd_paw))
        self.last_iterations = solution.iterations
        dpsi = solution.dpsi
        if self.smearing is not None:
            # ``ef_shift``: ``-int drho / N(ef)``, the implicit derivative of
            # ``N(E_F) = nelec`` that ``bisect_fermi``'s custom tangent writes.
            # The charge is taken by ``charge_channel``, not by summing every
            # channel, which would add the magnetization at ``nspin_mag = 4``.
            ldos, dos_ef = self.levels
            drho_raw = solver.response_density(dpsi)
            cell = calculation.system.cell
            element = cell.volume / int(np.prod(drho_raw.shape[1:]))
            delta_n = jnp.sum(calculation.charge_channel(drho_raw)) * element
            shift = jnp.where(jnp.abs(dos_ef) > 1.0e-18, -delta_n / dos_ef, 0.0)
            dpsi = solver.fermi_level_shift_states(dpsi, shift)
        drho_out, dbecsum_out = self._assembly(jnp.pad(dpsi, self.pad))
        tangent = residual.flatten(drho_out, dbecsum_out)
        return np.asarray(norm * tangent) - v


def refuse_residual_occupations(system) -> None:
    """Raise by name for the occupation schemes ``F`` has no *pooled* form for.

    On one process tetrahedra and ``from_input`` are in (I71, :func:`_weights`).
    Under k-point pools they are not measured: ``PooledScfResidual`` gathers the
    eigenvalues but nothing built the tetrahedron table's pooled use, so the
    driver's pool guard calls this on every rank before
    :class:`PooledScfResidual`'s protocol opens. Raised inside ``_evaluate``
    instead, after the eigenvalue gather, the serving pools leave ``serve()``
    while rank 0 waits in ``abort()``'s broadcast, and the run hangs until the
    pool heartbeat with rank 0's traceback never printed (measured on D22, Si
    8k tetrahedra at 2 pools: both ranks exit -6 after 315 s; I53).
    """
    scheme = system.occupations
    if scheme.startswith("tetrahedra") or scheme == "from_input":
        raise NotImplementedError(
            f"occupations={scheme!r} has no pooled form here: a residual "
            "solver takes it on one process (the tetrahedron table is a "
            "constant of the k-grid and 'from_input' a constant of the "
            "eigenvalues), but under k-point pools the pair has not been "
            "measured. Run it on one process, or use occupations='smearing' "
            "or 'fixed' with a pooled residual solver"
        )


def _weights(calculation, eigenvalues):
    """``wg`` alone, and traceable.

    ``Calculation.occupations`` cannot be used here: it converts the Fermi level
    and the entropy to Python floats for the iteration report, which is a host
    sync and a trace error. This is the same dispatch with only the weights kept
    -- the two must agree, and ``test_weights_agree_with_the_driver`` checks
    that they do rather than leaving it to inspection.
    """
    system = calculation.system
    scheme = system.occupations
    degeneracy = 1 if calculation.noncolin else 2
    weights = system.kpoints.weights

    if scheme == "from_input":
        # The card's numbers times ``wk``: a constant of the eigenvalues, so its
        # tangent is zero, which is the right derivative of a filling fixed by
        # hand. ``F`` is still a function of ``x`` alone.
        if system.input_occupations is None:
            raise ValueError("occupations='from_input' needs an OCCUPATIONS card")
        return input_occupations(
            system.input_occupations, eigenvalues, weights, degeneracy
        )
    if scheme.startswith("tetrahedra"):
        # The tetrahedra are host-built from the k-grid and the group, not from
        # the density, so they are a constant here like ``wk``; the weights
        # themselves are JAX on the eigenvalues, and the Fermi level carries the
        # implicit tangent of ``N(E_F) = nelec`` (``scf/tetrahedra.py``,
        # ``_fermi_level``), which is what ``jvp = 'autodiff'`` reads.
        counts = (
            (calculation.nelup, calculation.neldw)
            if calculation.two_fermi_energies else None
        )
        return tetrahedron_occupations_spin(
            calculation.tetrahedron_table(), eigenvalues, weights,
            calculation.nelec, counts=counts,
        )[0]

    if scheme == "fixed":
        # The same ``counts`` the smeared branch below takes, for the same
        # reason: with ``tot_magnetization`` the two channels are filled
        # independently, and forwarding it on one path and not the other is the
        # sibling-guard shape this file is a mirror of.
        counts = (
            (calculation.nelup, calculation.neldw)
            if calculation.two_fermi_energies else None
        )
        return fixed_occupations(
            eigenvalues, weights, calculation.nelec, degeneracy, counts=counts
        )[0]
    counts = (calculation.nelup, calculation.neldw) if calculation.two_fermi_energies else None
    wg, _ = smeared_occupations(
        eigenvalues, weights, calculation.nelec, system.degauss,
        system.smearing, counts=counts,
    )
    return wg


#: Penalty constraints whose field is a function of the density, as for the
#: three measured against ``pw.x`` (I73), but which no reference input carries,
#: so they stay refused until a Newton-against-mixing pair is measured on them.
UNMEASURED_PENALTIES = frozenset({"total direction", "atomic texture"})


#: The feedback schemes measured with a residual solver, each in the one regime
#: it was measured in (``notes/incompat/I73.md``, "The feedback field"):
#: ``'fsm'`` on the collinear ``fe-fsm.in``, the two per-atom schemes on the
#: noncollinear canted iron pair. The value is whether the run is noncollinear.
MEASURED_FEEDBACK = {"fsm": False, "atomic fsm": True, "atomic fsm direction": True}


def make_residual(calculation, nbnd: int, ethr: float, field=None) -> ScfResidual:
    """Build the residual for a calculation, refusing the regimes it cannot hold.

    ``field`` is the loop's starting field, which under a feedback scheme can
    differ from ``calculation.magnetic_field`` (a driven field carried over from
    a starting state, ``driver._carried_driven_field``); ``None`` takes the
    calculation's.
    """
    system = calculation.system
    if field is None:
        field = calculation.magnetic_field
    if field is not None and getattr(calculation, "spiral", False):
        # Neither unit measured the pair: I72 measured a spiral with no field,
        # I73 a field and the penalties on cells without a spiral.
        raise NotImplementedError(
            "a spin spiral under a magnetic field or a constraint has no "
            "measured residual form here: Newton on a spiral (I72) and Newton "
            "under a field (I73) were each checked on their own, the pair was "
            "not; use the mixing loop"
        )
    if field is not None:
        # ``step`` builds its potential with ``calculation.potential(rho)``,
        # which applies this field at full strength, so a field the mixing loop
        # holds fixed -- a uniform ``B_field``, a ``LOCAL_MAGNETIC_FIELDS`` card,
        # every *penalty* constraint, whose field is ``jax.grad`` of the
        # penalty and so a function of the density -- needs nothing here (I73).
        # ``run_scf`` changes the field in exactly two places. The feedback
        # field joins the state, as ``ns`` does, with Elk's update as its row
        # of ``F``; ``reducebf < 1`` stays refused (below). A local-field card
        # and the two penalty schemes no reference input carries are refused
        # only because they have not been measured.
        feedback = field.constraint in FEEDBACK
        if feedback:
            if field.fsm_update != "elk":
                raise NotImplementedError(
                    f"fsm_update = {field.fsm_update!r} has no residual form "
                    "here: the secant controller steps only on converged pairs "
                    "and carries the previous (field, moment) pair as memory, "
                    "so it is not a function of the state. The residual solver "
                    "carries the driven field in its state with Elk's row "
                    "B - tau (m - m_fix), whose zero is the same field, and "
                    "its Jacobian is the susceptibility the secant estimates. "
                    "Use fsm_update = 'elk' (the default) or scf_solver='mixing'"
                )
            if not field.penalty > 0.0:
                raise ValueError(
                    f"lambda = {field.penalty} is the feedback gain of "
                    f"constrained_magnetization = {field.constraint!r} and must "
                    "be positive: the residual's field rows are -lambda (m - "
                    "m_fix), and its convergence measure divides by lambda"
                )
            noncolin = bool(getattr(calculation, "noncolin", False))
            if (field.constraint not in MEASURED_FEEDBACK
                    or MEASURED_FEEDBACK[field.constraint] != noncolin):
                raise NotImplementedError(
                    f"constrained_magnetization = {field.constraint!r} in a "
                    f"{'noncollinear' if noncolin else 'collinear'} run is not "
                    "yet checked with a residual solver: 'fsm' is measured "
                    "against the mixing loop in a collinear run, 'atomic fsm' "
                    "and 'atomic fsm direction' in a noncollinear one, and "
                    "nothing else. Use the mixing loop (scf_solver='mixing')"
                )
        if field.constraint in UNMEASURED_PENALTIES:
            raise NotImplementedError(
                f"constrained_magnetization = {field.constraint!r} is not yet "
                "checked with a residual solver: its field is jax.grad of a "
                "penalty like 'atomic', 'atomic direction' and 'total', which "
                "match pw.x under Newton to 8e-8 Ry, but no reference input "
                "carries this scheme and it has not been measured. Use the "
                "mixing loop (scf_solver='mixing')"
            )
        if field.atomic is not None and field.constraint not in FEEDBACK_ATOMIC:
            raise NotImplementedError(
                "a LOCAL_MAGNETIC_FIELDS card is not yet checked with a "
                "residual solver: the field is held fixed by the loop, as a "
                "uniform b_field is (which runs, and matches the mixing loop "
                "to 2e-8 Ry), but the per-atom field has not been measured. "
                "Use the mixing loop (scf_solver='mixing')"
            )
        if field.reducebf != 1.0:
            # The scale's own map, ``s -> reducebf s``, has one root, s = 0, so
            # a residual solver carrying the scale would solve the field-free
            # problem, while the loop stops at a finite scale (FADED_FIELD's
            # table: 1.8e-3 Ry left and 2.2e-8 Ry of error at reducebf = 0.9).
            # The two answers then differ by the field the loop had not shed,
            # not by conv_thr, and the field's one job, breaking a symmetry on
            # the way, belongs to the start rather than to the fixed point.
            raise NotImplementedError(
                f"reducebf = {field.reducebf} has no residual form here: the "
                "mixing loop multiplies the applied field by reducebf after "
                "every iteration and stops at whatever field is left, so its "
                "answer depends on how many iterations it took. The decay's "
                "only fixed point is the field at zero, which is the run "
                "without the field; a residual solver would solve that and "
                "drop the symmetry breaking the field is for. Run the mixing "
                "loop, or run the residual solver without the field (from a "
                "starting state that already breaks the symmetry). A field "
                "held fixed (reducebf = 1) runs with a residual solver"
            )
        # What was measured is a collinear uniform field alone (Newton against
        # the mixing loop) and a noncollinear penalty alone (against pw.x).
        # The other two combinations run the same code, but are refused until
        # they are measured, as the local-field card is. A field together with
        # a penalty never reaches here: build_system refuses it, as pw.x does.
        noncolin = bool(getattr(calculation, "noncolin", False))
        uniform = getattr(field, "uniform", None)
        # A feedback scheme's uniform field is its driven field, not an
        # applied one, and its regimes are checked above.
        has_uniform = (not feedback and uniform is not None
                       and bool(np.any(np.asarray(uniform) != 0.0)))
        has_penalty = field.constraint != "none" and not feedback
        if has_uniform and noncolin:
            raise NotImplementedError(
                "a uniform b_field in a noncollinear run is not yet checked "
                "with a residual solver: only the collinear field is measured "
                "(against the mixing loop, to 2e-8 Ry). Use the mixing loop "
                "(scf_solver='mixing')"
            )
        if has_penalty and not noncolin:
            raise NotImplementedError(
                f"constrained_magnetization = {field.constraint!r} in a "
                "collinear run is not yet checked with a residual solver: the "
                "penalties are measured against pw.x on noncollinear inputs "
                "only, and on the one collinear cell tried neither Newton nor "
                "the mixing loop converged. Use the mixing loop "
                "(scf_solver='mixing')"
            )
    if calculation.spiral and calculation.is_hubbard:
        # Measured (``notes/incompat/I72.md``): on ``nii-chain-spiral.in`` with
        # U = 4 eV on Ni-3d, ``accuracy_of(F(x*) - x*)`` at the mixing loop's
        # converged state is 2.3e-4 against the loop's own 8.9e-10, and Newton
        # lands 7.3e-3 Ry away. A spinor ``ns`` is complex; the mixing loop
        # carries it as real and imaginary pairs (``driver._pack_ns``), while
        # :meth:`ScfResidual.pack` and :meth:`ScfResidual.flatten` keep only its
        # real part, so ``F`` here is not the loop's map.
        raise NotImplementedError(
            "a spin spiral with a Hubbard U has no residual form here: the "
            "occupation matrix of a spinor run is complex and the residual's "
            "packed state keeps only its real part, so F is not the mixing "
            "loop's map (notes/incompat/I72.md); use the mixing loop"
        )
    if calculation.spiral and calculation.functional.is_meta:
        # Not measured, so not admitted: ``tau`` joins the packed state, and the
        # one harness run of Newton on the hydrogen chain under ``tb09`` and
        # ``'kubler'`` had not returned after 36 minutes on one core, where the
        # same cell without the meta-GGA takes 4 s (``notes/incompat/I72.md``).
        raise NotImplementedError(
            "a spin spiral under a meta-GGA has no measured residual form here: "
            "tau joins the packed state and Newton on a spiral with it has not "
            "been checked against the mixing loop (notes/incompat/I72.md); use "
            "the mixing loop"
        )
    # A spin spiral is not refused (``notes/incompat/I72.md``): every call in
    # :meth:`ScfResidual.step` is the mixing loop's own, and the spiral lives
    # inside them -- the two spheres at ``k +- q/2`` in ``hamiltonian`` and
    # ``diagonalize``, the rotated-frame density and its displaced augmentation
    # in ``becsum`` and ``density`` -- so ``F`` is the loop's map. The packed
    # state is the rotated-frame ``(n, m_x, m_y, m_z)`` on the dense grid, which
    # the real-space Kerker screens on the charge alone and ``accuracy_of``
    # measures as the loop's ``dr2`` does; ``_weights`` takes the unshifted
    # k-set's weights, as ``Calculation.occupations`` does. ``jvp='sternheimer'``
    # stays refused (a magnetic noncollinear run, and ``response/sternheimer``
    # refuses a spiral by name behind it).

    rho = calculation.starting_density()
    becsum_ = calculation.starting_becsum()
    shapes = (tuple(np.shape(rho)),) + tuple(
        None if part is None else tuple(np.shape(part)) for part in becsum_
    )
    ns_shape = (
        tuple(np.shape(calculation.starting_ns())) if calculation.is_hubbard else None
    )
    tau_shape = None
    if calculation.functional.is_meta:
        # Exactly the density's shape, read off the density rather than rebuilt
        # from a spin number. The channel count of ``tau`` is ``nspin_mag`` and
        # never ``nspin``, and the two come apart at 4: a *nonmagnetic*
        # spin-orbit run has spinor wavefunctions and a one-channel density, so
        # ``(calculation.nspin,) + ...`` asked for four channels of a quantity
        # produced with one and the packing died on the reshape.
        tau_shape = tuple(np.shape(rho))
    size = int(np.prod(shapes[0])) + sum(
        int(np.prod(s)) for s in shapes[1:] if s is not None
    )
    if ns_shape is not None:
        size += int(np.prod(ns_shape))
    if tau_shape is not None:
        size += int(np.prod(tau_shape))
    feedback_field = field_shape = None
    if field is not None and field.constraint in FEEDBACK:
        feedback_field = field
        field_shape = tuple(np.shape(field.driven()))
        size += int(np.prod(field_shape))
    return ScfResidual(calculation, nbnd, ethr, shapes, size, ns_shape, tau_shape,
                       feedback_field, field_shape)
