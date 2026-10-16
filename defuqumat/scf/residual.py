"""The SCF as a root-find: the residual whose zero is self-consistency.

``scf/driver.py`` solves ``rho = F(rho)`` by iterating ``F`` and damping the
iteration with a mixer. This module writes the same fixed point as a *residual*

    r(x) = F(x) - x,   x = the packed (rho, becsum) state,

so that a solver which knows the derivative of ``r`` can drive it to zero
instead. ``F`` is one SCF iteration with every host-side decision frozen -- one
diagonalisation at a fixed ``ethr``, no retry, no ``ethr`` schedule, no field
feedback -- which is what makes it a *function* rather than a step of a loop.

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

**Refused rather than approximated:** external and constrained magnetic fields
(the field is driven by a secant *outside* the density, so ``F`` would not be a
function of ``x`` alone), and tetrahedron and ``from_input`` occupations (built on
the host, and the second is not a function of the eigenvalues at all). Each
raises by name. A spin spiral runs (``notes/incompat/I72.md``): its map is the
mixing loop's own, with no term of its own here, except under a Hubbard U, which
is refused: a spinor ``ns`` is complex and :meth:`ScfResidual.pack` keeps its real
part only. A spiral under a meta-GGA is refused too, as not measured.

Memory: the packed state is ``nspin * dense_grid`` floats plus ``becsum`` and
``ns``, and a Krylov subspace holds a few tens of those -- tens of MB,
negligible beside the wavefunctions. One ``jvp`` doubles the wavefunction
working set for the duration of the call, because the tangent ``psi`` has the
same shape as ``psi``; that is the number to watch, and it is why the JVP goes
through the same ``k_batch`` dial the primal does.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.eager import compiled
from defuqumat.scf.occupations import fixed_occupations, smeared_occupations

__all__ = ["ScfResidual", "SternheimerJacobian", "make_residual"]


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

    # ---- packing -------------------------------------------------------
    def pack(self, rho, becsum_, ns=None, tau=None) -> np.ndarray:
        """The mixed state as one flat real vector, exactly ``_mix``'s packing."""
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

    # ---- the map -------------------------------------------------------
    def step(self, x, psi0):
        """One SCF iteration as a pure function: ``x -> F(x)``, and the ``psi``."""
        calculation = self.calculation
        rho, becsum_, ns, tau = self.unpack(x)
        potential = calculation.potential(rho, tau=tau)
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
        return self.flatten(rho_out, becsum_out, ns_out, tau_out), psi

    def flatten(self, rho, becsum_, ns=None, tau=None):
        """:meth:`pack`, inside a trace -- ``jnp`` rather than ``np``."""
        flat = [jnp.reshape(jnp.real(rho), (-1,))]
        for part in becsum_:
            if part is not None:
                flat.append(jnp.reshape(jnp.real(part), (-1,)))
        if ns is not None:
            flat.append(jnp.reshape(jnp.real(ns), (-1,)))
        if tau is not None:
            flat.append(jnp.reshape(jnp.real(tau), (-1,)))
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
    if scheme.startswith("tetrahedra") or scheme == "from_input":
        raise NotImplementedError(
            f"occupations={scheme!r} has no differentiable form here: the "
            "tetrahedron weights are built on the host from the k-grid and the "
            "symmetry, and 'from_input' is not a function of the eigenvalues at "
            "all. Use occupations='smearing' or 'fixed' with a residual solver"
        )
    degeneracy = 1 if calculation.noncolin else 2
    weights = system.kpoints.weights

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


def make_residual(calculation, nbnd: int, ethr: float) -> ScfResidual:
    """Build the residual for a calculation, refusing the regimes it cannot hold."""
    if calculation.magnetic_field is not None:
        raise NotImplementedError(
            "an external or constraining magnetic field has no residual form "
            "here: the field is driven by a secant *outside* the density "
            "(reducebf, the fixed-spin-moment feedback), so F changes between "
            "evaluations and is not a function of the state alone"
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
    return ScfResidual(calculation, nbnd, ethr, shapes, size, ns_shape, tau_shape)
