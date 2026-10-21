"""The dynamical matrix at ``q != 0``: the perturbed states live on a second sphere.

``PLAN.md`` P71. Everything the response stack has computed so far is a
zone-centre quantity -- the dielectric constant, the Born charges, the
``Gamma`` dynamical matrix -- and what separates those from a phonon
*dispersion* is one fact: a displacement pattern

    u_s(R) = u_s e^{i q . R}

is not a displacement of the unit cell. It is periodic only on a supercell,
and a supercell is what a frozen-phonon calculation pays for. Linear response
does not pay it, because the *first-order* quantities are still lattice
periodic once their ``e^{i q . r}`` is factored out: the perturbation is
``e^{i q . r}`` times a periodic function, so ``dV_q |psi_k>`` is a Bloch state
at ``k + q`` and the whole calculation stays in the unit cell with one extra
plane-wave sphere. That is the same trick the spin spirals use (``PLAN.md``
P19, Elk's ``gengkqvec``), one perturbation instead of one spinor component.

**What changes, term by term, and what does not.**

| | ``q = 0`` (P25) | ``q != 0`` (here) |
|---|---|---|
| where ``dpsi`` lives | the ``k`` sphere | the ``k + q`` sphere |
| ``H - eps S`` | both at ``k`` | ``H``, ``S`` at ``k+q``, ``eps`` at ``k`` |
| ``P_c^+`` | occupied at ``k`` | occupied at ``k+q`` |
| the bare local term | ``jvp`` of ``V_loc(G)`` | ``jvp`` of ``V_loc(G+q)`` |
| the Hartree kernel | ``8 pi / |G|^2``, ``G = 0`` dropped | ``8 pi / |G+q|^2``, **``G = 0`` kept** |
| ``drho`` | ``2 Re[psi* dpsi]`` | ``2 psi* dpsi``, complex |
| the frozen Hessian | ``dynmat_us + d2ionq(0)`` | ``dynmat_us`` unchanged ``+ d2ionq(q)``, and ``C(q)`` for a core charge (below) |

The last row is the one worth reading twice, and it was checked in the Fortran
rather than assumed. ``dynmat_us.f90:105-124`` fills ``dynwrk(na_icart,
na_jcart)`` -- **the same atom twice** -- from ``g(:,ng)`` and ``tau(na)``, with
no ``xq`` anywhere in the routine. The second derivative of the external
potential at frozen density is diagonal in the atom index, so under a modulated
displacement its two phases cancel and it is *the same matrix at every* ``q``.
The whole ``q`` dependence of the part that does not involve the response is
therefore in the Ewald sum, ``d2ionq`` -- **without a core charge**. With one
(I80) the frozen Hessian also holds ``int drho_core_i f_xc drho_core_j``, which
couples two atoms' cores inside the nonlinear ``E_xc`` and is swapped to ``q``
like the Ewald sum (:func:`swap_core`), and the bare perturbation carries
``f_xc drho_core`` (:func:`core_displacements_at_q`), ``ph.x``'s ``addcore``
and ``dynmat_nlcc``.
That is a norm-conserving statement: for an ultrasoft or PAW dataset
``addusdynmat`` adds ``dvanqq``'s ``int2`` and ``int5``, which pair the local
potential of one atom with the augmentation charge of another at ``G + q``, so
the frozen Hessian is not atom-diagonal there and is a function of ``q``
(``notes/incompat/I81.md``; the dataset is refused).

**What is refused, and it is most things.** This lands the norm-conserving
insulator and the norm-conserving metal on a full k-grid: no ultrasoft or PAW
(``S`` moves with the atoms and the multiplier matrix has no two-sphere form),
no metal at a *nonzero* reciprocal lattice vector ``q = G`` (``ef_shift_wfc``
would add the states at ``k`` to a ``dpsi`` on the ``k + G`` sphere; a metal at
``q = 0`` carries ``ef_shift`` here as at ``Gamma``), no spin-polarized metal
(a ``nspin = 2`` insulator runs under the LDA, I82: the response density keeps
one sum per channel and the kernel acts on the two apart), no nonlinear core
correction at ``nspin = 2`` (the core term runs at ``nspin = 1``, I80; the pair
was measured by neither), no spinor, and no symmetry -- the small group of ``q``
and the star of ``q`` are
what a dispersion needs and are not written. Each is named at the door.

References: Baroni, de Gironcoli, Dal Corso and Giannozzi, *Rev. Mod. Phys.*
**73**, 515 (2001), §II.B for the two-sphere structure; ``PHonon/PH`` for the
transcribed halves (``dvqpsi_us``, ``compute_dvloc``, ``incdrhoscf``,
``d2ionq``, ``drhodv``, ``dynmat_us``).
"""

from __future__ import annotations

import copy

import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.basis.fft import g_to_r, r_to_g
from defuqumat.eager import compiled
from defuqumat.response.sternheimer import (
    SternheimerResult, SternheimerSolver, pass_threshold,
)
from defuqumat.scf.occupations import w0gauss, wgauss
from defuqumat.scf.streaming import is_host_store
from defuqumat.system.cell import Cell
from defuqumat.system.kpoints import KPoints

__all__ = [
    "dynamical_matrix_at_q", "kpoints_plus_q", "states_at_k_plus_q",
    "TwoSphereSolver", "bare_displacements_at_q", "self_consistent_response_at_q",
    "hartree_at_q", "induced_potential_at_q", "induced_perturbation_at_q",
    "ewald_dynamical_matrix", "frozen_force_constants", "response_force_constants",
    "require_a_two_sphere_regime", "core_displacements_at_q",
    "core_force_constants", "swap_core", "local_at_q",
]


def kpoints_plus_q(kpoints: KPoints, q_cart, cell: Cell) -> KPoints:
    """The list ``k + q``. ``q_cart`` is in **1/bohr**, and that matters.

    **Two cartesian conventions meet here and only this function sees both.**
    Everything else in this module works against
    :meth:`~defuqumat.basis.gvectors.GVectors.cartesian`, which is 1/bohr, so
    ``q_cart`` is in 1/bohr throughout -- while :class:`KPoints` stores
    ``coords`` in units of ``2 pi / alat``, which is what ``pw.x`` prints.
    Adding one to the other is dimensionally silent: the k-points move by
    ``tpiba`` times too much, every array is the right shape, every solve
    converges, and the answer is **exactly right at** ``q = 0`` because zero
    scales to zero. It was worth 1822 cm^-1 at ``X`` and nothing at ``Gamma``,
    which is why ``PLAN.md`` P71 has a regression at ``q = 0`` *and* a number at
    the zone boundary rather than only the first.

    The shifted points are kept **literal** -- not wrapped back into the first
    Brillouin zone -- for the reason :mod:`defuqumat.system.spiral` gives for the
    same choice: the wrapped point is the same physics through a different
    G-index map, and reconciling the two is P16's zone-edge Miller shift. A
    sphere built directly at ``k + q`` needs none of it.

    The weights are carried through unchanged. They are the weights of ``k``,
    which is what every sum over the zone here is over; the ``k + q`` list is a
    place to put a second sphere and a second diagonalisation, not a second
    integration grid.
    """
    shift = np.asarray(q_cart, dtype=float) / cell.tpiba
    coords = np.asarray(kpoints.coords) + shift[None, :]
    moved = KPoints.from_cartesian(
        coords, np.asarray(kpoints.weights), precision=kpoints.precision
    )
    # ``from_cartesian`` renormalises to the unpolarized convention (weights
    # summing to 2), so a polarized run's set takes ``for_spin`` back at this
    # boundary, read off the set it came from (the trap list's caller-built
    # k-set; nothing at ``k + q`` reads the weights today, I82).
    if kpoints.spin_normalized:
        from defuqumat.system.kpoints import for_spin

        moved = for_spin(moved, 2)
    return moved


def states_at_k_plus_q(calculation, v_scf, q_cart, nbnd, ethr: float = 1e-13,
                       stream: bool = False):
    """The ground state again, on the ``k + q`` sphere, at the converged potential.

    ``phq_init``'s job: ``evq``, the unperturbed states the projector ``P_c^+``
    and the preconditioner are built from. It is a non-self-consistent
    diagonalisation and nothing about it is new -- the density, the potential
    and every radial table are the ones the SCF ended with, and only the things
    carrying a ``k`` index are rebuilt.

    That is what :meth:`~defuqumat.scf.driver.Calculation.at_kpoints` is for, and
    using it rather than a fresh :class:`~defuqumat.scf.driver.Calculation` is a
    memory decision as much as a speed one: the dense G set, the FFT box, the
    local potential, the Ewald neighbour list and the augmentation tables are
    *shared* between the two calculations rather than built twice.

    ``stream`` writes the states into a numpy store a k-chunk at a time
    (:func:`~defuqumat.scf.streaming.stream_states`, ``c_bands_nscf`` with
    ``save_buffer``), for the k-chunked route
    (:mod:`defuqumat.response.chunked_phonon`). The two solve the same problem
    and agree to round-off at one k-point a chunk; at larger chunks the
    eigensolver's compiled program differs, and the states agree inside each
    degenerate multiplet only up to the rotation the solver is free in.

    Returns ``(calculation_kq, hamiltonians_kq, eigenvalues_kq, psi_kq)``.
    """
    moved = calculation.at_kpoints(kpoints_plus_q(
        calculation.system.kpoints, q_cart, calculation.system.cell
    ))
    hamiltonians = moved.hamiltonian(v_scf)
    if stream:
        from defuqumat.scf.streaming import stream_states

        eigenvalues, wavefunctions = stream_states(moved, hamiltonians, nbnd, ethr)[:2]
    else:
        eigenvalues, wavefunctions = moved.diagonalize(hamiltonians, nbnd, None, ethr)
    return moved, hamiltonians, np.asarray(eigenvalues), wavefunctions


def metallic_rows(eigenvalues, smearing) -> int:
    """``max_k nbnd_occ(k)``: the bands a metal solves for, the grid's widest cut.

    ``setup_nbnd_occ``'s count, ``eps < ef + xmax degauss``, maximised over k
    and spin so it is one static width. Host-side, on concrete eigenvalues.
    """
    below = np.asarray(eigenvalues) < smearing.cutoff
    return max(1, int(np.max(np.sum(below, axis=-1))))


class TwoSphereSolver(SternheimerSolver):
    """``(H_{k+q} - eps_k S_{k+q} + alpha Q) dpsi = -P_c^{k+q} dV_q |psi_k>``.

    The Sternheimer equation with its two sides on different spheres, and the
    asymmetry is the whole of it: the operator, the projector and the
    preconditioner belong to ``k + q``, while the eigenvalue subtracted is the
    one of the band being perturbed, at ``k``. QE writes exactly that split --
    ``ch_psi_all.f90`` subtracts ``e(ibnd)`` from ``et(:,ikk)`` while applying
    ``h_psi`` at ``ikq``, ``orthogonalize.f90`` takes ``evq``, and
    ``h_prec.f90``'s docstring says "evq wavefunction at k+q point" in as many
    words -- and getting either half from the wrong sphere gives an equation
    that still converges.

    A **subclass** rather than a pair of optional arguments on
    :class:`~defuqumat.response.sternheimer.SternheimerSolver`, because that class
    is what P24, P25, P28, P45 and P70 are validated through and threading a
    second state set into six of its methods would put every one of those
    numbers behind an untested branch. What this owes in exchange is that
    ``q = 0`` through *this* class must reproduce the parent exactly, which is
    the first thing its tests assert.
    """

    def __init__(self, base: SternheimerSolver, hamiltonians_kq, psi_kq,
                 eigenvalues_kq, calculation_kq, rows: int | None = None):
        # Built from a converged parent rather than from scratch: everything
        # that is a property of the ground state at ``k`` -- the occupied
        # counts, ``alpha_pv``, the weights, the thresholds -- is already
        # decided there, and duplicating that logic is how the two would drift.
        self.__dict__.update(base.__dict__)
        self.calculation_kq = calculation_kq
        # ``self.hamiltonians`` **is** the second sphere's here. The solve is
        # the only thing that reads it -- the operator, the projector, the
        # preconditioner and the band mask -- and all four belong at ``k + q``,
        # so replacing it wholesale is what keeps ``solve_at``'s sixty lines of
        # conjugate gradient inherited rather than copied. The states at ``k``
        # are still ``self.psi``, which is what the right-hand side and the
        # response density are built from.
        self.hamiltonians = tuple(hamiltonians_kq)
        keep = self.psi.shape[2]
        if self.smearing is not None:
            # **A metal solves fewer rows than it projects out.** The rows are
            # the bands at ``k`` below the cut, ``nbnd_occ(ikk)``, and nothing
            # past the largest such count over the grid has a right-hand side
            # (``orthogonalize`` and ``cgsolve_all`` stop at ``nbnd_occ(ikk)``),
            # so the block is cut there: a static width, the grid's maximum.
            # The columns are every band at ``k + q``, ``orthogonalize``'s
            # ``DO jbnd = 1, nbnd``, whose weights fall off with ``wgp`` rather
            # than at the cut. On ``al-elph-nosym.in`` the cut leaves 1 to 4
            # bands per k-point of 6, 1.88 on average.
            # ``rows`` is that count when the caller has it already, as a pass
            # compiled over a k-chunk must (:func:`metallic_rows`).
            if rows is None:
                rows = metallic_rows(self.eigenvalues, self.smearing)
            self.psi = self.psi[:, :, :rows]
            self.eigenvalues = self.eigenvalues[:, :, :rows]
            self.weights = self.weights[:, :, :rows]
            self.density_weights = self.density_weights[:, :, :rows]
            self.projector_mask = self.projector_mask[:, :, :rows]
            self.nocc = rows
        else:
            self.psi_kq = jnp.asarray(psi_kq)[:, :, :keep]
            self.eigenvalues_kq = jnp.asarray(eigenvalues_kq)[:, :, :keep]
        if self.smearing is not None:
            self.psi_kq = jnp.asarray(psi_kq)
            self.eigenvalues_kq = jnp.asarray(eigenvalues_kq)
        # The projector runs over the occupied manifold **at k + q**, and the
        # right-hand side runs over the bands being solved for, **at k**. They
        # are the same count for an insulator with a gap everywhere, but they
        # are not the same object, and writing one mask for both is the kind of
        # thing that is right until a band crosses -- which in a metal is every
        # k-point on the Fermi surface. A metal's mask is ``setup_nbnd_occ``'s
        # cut, ``eps < ef + xmax degauss``, evaluated on the k + q eigenvalues
        # (``nbnd_occ(ikq)`` in ``orthogonalize.f90`` and ``ch_psi_all.f90``).
        if self.smearing is not None:
            self.projector_mask_kq = self.eigenvalues_kq < self.smearing.cutoff
        else:
            bands = jnp.arange(self.eigenvalues_kq.shape[2])
            counts = jnp.asarray(self.occupied_counts)
            self.projector_mask_kq = jnp.broadcast_to(
                (bands[None, :] < counts[:, None])[:, None, :],
                self.eigenvalues_kq.shape,
            )

    # -- the three pieces that move to the second sphere -------------------

    def _occupied_overlapped(self, ik, spin):
        """``S_{k+q}|psi_occ(k+q)>``, the level shift's fixed half on the second sphere."""
        return self.hamiltonians[spin].apply_s(self.psi_kq[spin][ik], ik)

    def _operator(self, vectors, ik, spin, s_occupied=None):
        """``ch_psi_all`` with ``H`` and ``S`` at ``k+q`` and ``eps`` at ``k``;
        ``s_occupied`` as :meth:`SternheimerSolver._operator` takes it."""
        hamiltonian = self.hamiltonians[spin]
        occupied = self.psi_kq[spin][ik]
        eps = self.eigenvalues[spin][ik][:, None]

        h = hamiltonian.apply(vectors, ik)
        s = hamiltonian.apply_s(vectors, ik)
        out = h - eps * s

        overlaps = jnp.einsum("mg,ng->mn", jnp.conj(occupied), s)
        overlaps = jnp.where(self.projector_mask_kq[spin][ik][:, None], overlaps, 0.0)
        if s_occupied is not None:
            return out + self.alpha_pv * jnp.einsum("mn,mg->ng", overlaps, s_occupied)
        lifted = jnp.einsum("mn,mg->ng", overlaps, occupied)
        return out + self.alpha_pv * hamiltonian.apply_s(lifted, ik)

    def project(self, rhs, ik, spin):
        """``-P_c^+ rhs`` with the occupied manifold taken at ``k + q``.

        Two masks rather than one: the **row** mask is over the bands being
        solved for and is the ground state's at ``k``; the **column** mask is
        over the manifold projected out and is the one at ``k + q``. A metal
        takes :meth:`_smeared_projection_at_q` for the columns instead.
        """
        hamiltonian = self.hamiltonians[spin]
        occupied = self.psi_kq[spin][ik]
        s_occupied = hamiltonian.apply_s(occupied, ik)

        rows = self.projector_mask[spin][ik][:, None]
        columns = self.projector_mask_kq[spin][ik][:, None]
        rhs = jnp.where(rows, rhs, 0.0)
        if self.smearing is not None:
            rhs, overlaps = self._smeared_projection_at_q(
                rhs, jnp.einsum("mg,ng->mn", jnp.conj(occupied), rhs), ik, spin)
            return -(rhs - jnp.einsum("mn,mg->ng", overlaps, s_occupied))
        overlaps = jnp.where(
            columns, jnp.einsum("mg,ng->mn", jnp.conj(occupied), rhs), 0.0
        )
        return -(rhs - jnp.einsum("mn,mg->ng", overlaps, s_occupied))

    def _smeared_projection_at_q(self, rhs, overlaps, ik, spin):
        """``orthogonalize``'s metal branch with its two eigenvalue sets apart.

        :meth:`SternheimerSolver._smeared_projection` with ``eps_i`` (the band
        solved for, the rows of ``rhs``) at ``k`` and ``eps_j`` (the band
        projected out, the rows of ``overlaps``) at ``k + q``, which is
        ``orthogonalize.f90:115-138`` read as written: ``wg1`` and ``w0g`` from
        ``et(ibnd, ikk)``, ``wgp`` from ``et(jbnd, ikq)``, and
        ``deltae = et(jbnd, ikq) - et(ibnd, ikk)``. The ``alpha_pv`` piece is
        admitted for ``j`` inside ``nbnd_occ(ikq)``, the column mask.

        The Fermi level's shift is not here: ``ph.x`` shifts it only at
        ``q = 0`` (``lmetq0``), where a uniform change of the electron count is
        possible, and it acts on the density and on ``dpsi`` after the solve,
        not on the projector (:func:`screening_loop_at_q`).
        """
        occupation, weights = self._projection_at(
            self.eigenvalues[spin][ik], self.eigenvalues_kq[spin][ik],
            self.projector_mask_kq[spin][ik])
        return occupation[:, None] * rhs, weights * overlaps

    def _projection_at(self, eps, eps_kq, columns):
        """``(wg1_i, wwg_(j i))`` at one k-point, from its eigenvalues at ``k``
        (``i``) and at ``k + q`` (``j``) and the ``j`` admitted to the shift."""
        smearing = self.smearing
        degauss, ngauss = smearing.degauss, smearing.ngauss
        occupation = wgauss((smearing.ef - eps) / degauss, ngauss)      # wg1_i, at k
        delta = w0gauss((smearing.ef - eps) / degauss, ngauss) / degauss
        occupation_kq = wgauss((smearing.ef - eps_kq) / degauss, ngauss)  # wgp_j, at k+q

        difference = eps_kq[:, None] - eps[None, :]                     # deltae_(j,i)
        step = wgauss(difference / degauss, 0)                          # theta
        mixed = occupation[None, :] * (1.0 - step) + occupation_kq[:, None] * step

        close = jnp.abs(difference) <= 1.0e-5
        safe = jnp.where(close, 1.0, difference)
        ratio = (occupation_kq[:, None] - occupation[None, :]) / safe
        shift = self.alpha_pv * step * jnp.where(close, -delta[None, :], ratio)
        return occupation, mixed + jnp.where(columns[:, None], shift, 0.0)

    def _preconditioner(self, ik, spin):
        """``h_prec`` on the second sphere: ``|k+q+G|^2`` and ``evq``."""
        kinetic = self.hamiltonians[spin].state_kinetic[ik]
        # ``h_prec``'s ``eprec(ibnd)`` from ``evq(:, ibnd)``: one per solved row.
        occupied = self.psi_kq[spin][ik][: self.psi.shape[2]]
        eprec = 1.35 * jnp.real(
            jnp.einsum("ng,g,ng->n", jnp.conj(occupied), kinetic, occupied))
        return 1.0 / jnp.maximum(1.0, kinetic[None, :] / eprec[:, None])

    # -- a metal's k-points, grouped by how many bands each one solves -------

    def row_groups(self, spin: int) -> list:
        """``[(width, k indices)]``: the k-points of one channel by ``nbnd_occ(k)``.

        ``cgsolve_all`` is called with ``nbnd_occ(ikk)`` bands
        (``response_kernels.f90:261``), a different block at every k-point of a
        metal. Here each distinct count is one static width and its k-points
        are solved together at that width, so a k-point with one band below the
        cut applies ``H`` to one band and not to the grid's widest block: on
        ``al-elph-nosym.in`` 167, 243, 96 and 6 k-points at widths 1 to 4, 1.88
        bands a k-point against 4. ``None`` widths for an insulator, whose
        block is one width already.
        """
        cached = self.__dict__.setdefault("_row_groups", {})
        if spin not in cached:
            below = np.asarray(self.eigenvalues[spin]) < self.smearing.cutoff
            counts = np.maximum(1, np.sum(below, axis=-1))
            cached[spin] = [(int(width), np.flatnonzero(counts == width))
                            for width in np.unique(counts)]
        return cached[spin]

    def _groups_by_width(self) -> bool:
        """Whether :meth:`solve` and the response density walk :meth:`row_groups`:
        a metal on device arrays with concrete eigenvalues, which a pass
        compiled over a k-chunk does not have (it solves at one width)."""
        return (self.smearing is not None and not is_host_store(self.psi)
                and not isinstance(self.eigenvalues, jax.core.Tracer))

    def narrowed(self, width: int) -> "TwoSphereSolver":
        """This solver with the solved block cut to its first ``width`` rows."""
        cached = self.__dict__.setdefault("_narrowed", {})
        if width not in cached:
            view = copy.copy(self)
            view.__dict__.pop("_narrowed", None)
            view.psi = self.psi[:, :, :width]
            view.eigenvalues = self.eigenvalues[:, :, :width]
            view.weights = self.weights[:, :, :width]
            view.density_weights = self.density_weights[:, :, :width]
            view.projector_mask = self.projector_mask[:, :, :width]
            view.nocc = width
            cached[width] = view
        return cached[width]

    def solve(self, perturbation, start=None, threshold=None) -> SternheimerResult:
        """:meth:`SternheimerSolver.solve`, a metal's k-points one width at a time.

        Each group of :meth:`row_groups` is one walk over its k-points with the
        block cut to its width, and the solutions are put back into the whole
        ``(nk, nocc, npwx)`` array with zero rows past each k-point's count,
        which is what the whole-width solve returns there. ``perturbation`` is
        called with the narrowed block and must answer at its width.
        """
        if not self._groups_by_width():
            return super().solve(perturbation, start=start, threshold=threshold)
        from defuqumat.batching import map_k

        batch = self.calculation.k_batch
        level = jnp.asarray(self.threshold if threshold is None else threshold)
        nk, rows = self.psi.shape[1], self.psi.shape[2]
        ndim = self.psi_kq.shape[-1]
        blocks, iterations, residuals = [], [], []
        for spin in range(self.nspin):
            groups = self.row_groups(spin)
            # Sliced here, outside the trace, so the cached views hold arrays.
            views = [self.narrowed(width) for width, _ in groups]
            initial = (jnp.zeros((nk, rows, ndim), dtype=self.psi.dtype)
                       if start is None else jnp.asarray(start[spin]))

            def walk(level, initial, indices, spin=spin, groups=groups,
                     views=views):
                out = jnp.zeros((nk, rows, ndim), dtype=initial.dtype)
                steps, worst = [], []
                for (width, _), chosen, view in zip(groups, indices, views):

                    def one_k(ik, view=view, width=width):
                        rhs = view.project(
                            perturbation(view.psi[spin][ik], ik, spin)[:width],
                            ik, spin)
                        return view.solve_at(rhs, ik, spin,
                                             start=initial[ik][:width],
                                             threshold=level)

                    dpsi, count, residual = map_k(one_k, chosen, batch=batch)
                    out = out.at[chosen].set(
                        jnp.pad(dpsi, ((0, 0), (0, rows - width), (0, 0))))
                    steps.append(jnp.max(count))
                    worst.append(jnp.max(residual))
                return out, jnp.max(jnp.stack(steps)), jnp.max(jnp.stack(worst))

            dpsi, steps, residual = compiled(
                walk, level, initial,
                tuple(jnp.asarray(chosen) for _, chosen in groups))
            blocks.append(dpsi)
            iterations.append(int(steps))
            residuals.append(float(residual))
        return SternheimerResult(dpsi=jnp.stack(blocks), iterations=max(iterations),
                                 residual=max(residuals))

    # -- the density it produces -------------------------------------------

    def response_density_at_q(self, dpsi) -> jnp.ndarray:
        """``drho_q(r)``, finished: :meth:`response_parts_at_q`, then
        :meth:`finish_response_at_q`."""
        return self.finish_response_at_q(self.response_parts_at_q(dpsi))

    def response_parts_at_q(self, dpsi) -> jnp.ndarray:
        """``drho_q(r)``'s sum over k: ``incdrhoscf``, and here it is transcribed.

        The sum alone, on the smooth grid and not normalised;
        :meth:`finish_response_at_q` does the rest, and is linear, so a response
        walked a k-chunk at a time adds the chunks' sums and finishes once.

        At ``Gamma`` the response density is one ``jvp`` of the *code that
        builds a density* -- ``rho = sum wg Re[psi* psi]``, differentiated along
        ``psi -> psi + dpsi`` -- and every ultrasoft term comes with it (P24).
        At ``q != 0`` that route is not available and the reason is physical
        rather than technical: the perturbed crystal has no density in this
        cell at all. What is periodic is the ``+q`` Fourier component,

            drho_q(r) = (2 / Omega) sum_k w_k sum_v conj(psi_vk(r)) dpsi_vk(r)

        with ``psi`` on the ``k`` sphere and ``dpsi`` on the ``k+q`` one, and
        the two factors in it are **not** a modulus. The leading 2 is the
        ``-q`` half of a real perturbation, which contributes the same amount
        (``incdrhoscf.f90:76``, ``wgt = 2 weight / omega``); at ``Gamma`` the
        two halves are complex conjugates of each other and collapse to the
        ``2 Re[...]`` the ``jvp`` produces, which is what makes the two
        expressions agree there.
        """
        from defuqumat.batching import sum_bands, sum_k

        calculation, kq = self.calculation, self.calculation_kq
        grid = calculation.basis.smooth.grid
        dpsi = jnp.asarray(dpsi)
        index_k, index_kq = calculation.fft_index, kq.fft_index

        def one_k(item):
            states, tangent, here, there, weight = item

            # **The bands are walked, not batched**, which is
            # ``incdrhoscf.f90``'s own ``DO ibnd`` and is the same working-set
            # argument the ground-state ``sum_band`` makes -- with *three*
            # band-sized real-space arrays here rather than one, since both
            # factors are transformed and then multiplied (``PLAN.md`` P74).
            def one_band(arrays):
                state, tangent_band, occupation = arrays
                psi_r = g_to_r(state, here, grid)
                dpsi_r = g_to_r(tangent_band, there, grid)
                return occupation.astype(states.dtype) * jnp.conj(psi_r) * dpsi_r

            return sum_bands(one_band, (states, tangent, weight))

        # **One sum per channel**, ``incdrhoscf``'s ``drhoscf(:, current_spin)``:
        # the channels are not added, because at ``nspin = 2`` the kernel acts
        # on the up and down response apart (I82), and adding them is a shape
        # error in its ``jvp`` rather than a wrong number. ``(nspin, *grid)``.
        def summed(psi, dpsi):
            return jnp.stack([
                sum_k(
                    one_k,
                    (psi[spin], dpsi[spin], index_k, index_kq,
                     self.density_weights[spin]),
                    batch=calculation.k_batch,
                )
                for spin in range(self.nspin)
            ])

        def grouped(psi, dpsi, groups):
            # A metal's rows past ``nbnd_occ(k)`` hold zeros (:meth:`solve`), so
            # each group of :meth:`row_groups` is summed at its own width:
            # ``incdrhoscf``'s ``DO ibnd = 1, nbnd_occ(ikk)``.
            channels = []
            for spin in range(self.nspin):
                total = jnp.zeros(grid, dtype=psi.dtype)
                for (width, _), chosen in zip(self.row_groups(spin), groups[spin]):
                    total = total + sum_k(
                        one_k,
                        (psi[spin][chosen, :width], dpsi[spin][chosen, :width],
                         index_k[chosen], index_kq[chosen],
                         self.density_weights[spin][chosen, :width]),
                        batch=calculation.k_batch,
                    )
                channels.append(total)
            return jnp.stack(channels)

        # Called once per displacement per iteration with new closures, so it is
        # compiled by its structure (:mod:`defuqumat.eager`).
        if self._groups_by_width():
            groups = tuple(tuple(jnp.asarray(chosen) for _, chosen in
                                 self.row_groups(spin))
                           for spin in range(self.nspin))
            return compiled(grouped, self.psi, dpsi, groups)
        return compiled(summed, self.psi, dpsi)

    def finish_response_at_q(self, total) -> jnp.ndarray:
        """``2 total / Omega`` on the **dense** grid, like every other density
        here, channel by channel: ``total`` is ``(nspin, *smooth grid)`` and the
        result ``(nspin_mag, *dense grid)``, the leading axis every consumer of
        one expects (collinear only, where ``nspin_mag = nspin``). Linear in
        ``total``."""
        calculation = self.calculation
        volume = calculation.system.cell.volume
        smooth, dense = calculation.basis.smooth, calculation.basis.dense
        total = jnp.asarray(total)
        return jnp.stack([complex_to_dense(2.0 * channel / volume, smooth, dense)
                          for channel in total])


def _one_grid(smooth, dense) -> bool:
    """The interpolation helpers' own test for a field that needs no resampling."""
    return smooth is dense or (smooth.grid == dense.grid and smooth.ngm == dense.ngm)


def complex_to_dense(field, smooth, dense):
    """:func:`~defuqumat.basis.interpolate.to_dense` for a **complex** field.

    The interpolation helpers are written for a real field and end in
    ``jnp.real``, which is right for every density and potential at ``Gamma``
    and silently wrong for the ``+q`` component here, whose imaginary part is
    half of it. On a cell whose two grids coincide the helpers return their
    argument untouched, which is why the one-grid validation could not see it;
    with ``ecutrho = 8 ecutwfc`` on two-atom silicon at ``q = (1/2, 0, 0)`` every
    mode came out imaginary, -1899 to -120 cm^-1 against -94 to 482 on one grid.
    The resampling is linear, so the two parts go through it apart.
    """
    from defuqumat.basis.interpolate import to_dense

    field = jnp.asarray(field)
    if not jnp.iscomplexobj(field) or _one_grid(smooth, dense):
        return to_dense(field, smooth, dense)
    return (to_dense(jnp.real(field), smooth, dense)
            + 1j * to_dense(jnp.imag(field), smooth, dense))


def complex_to_smooth(field, dense, smooth):
    """:func:`~defuqumat.basis.interpolate.to_smooth` for a complex field; see
    :func:`complex_to_dense`."""
    from defuqumat.basis.interpolate import to_smooth

    field = jnp.asarray(field)
    if not jnp.iscomplexobj(field) or _one_grid(smooth, dense):
        return to_smooth(field, dense, smooth)
    return (to_smooth(jnp.real(field), dense, smooth)
            + 1j * to_smooth(jnp.imag(field), dense, smooth))


# ---------------------------------------------------------------------------
# The bare perturbation.
# ---------------------------------------------------------------------------

def bare_displacements_at_q(calculation, calculation_kq, solver, q_cart, positions,
                            atoms=None, core=None) -> np.ndarray:
    """``dV_bare_q/du |psi_k>`` for every atom and direction, on the ``k+q`` sphere.

    ``dvqpsi_us.f90``, and it stays what P25 made it: **one ``jvp`` through the
    positions of code that applies a potential**, rather than a second,
    hand-derived expression. What ``q`` changes is the code being
    differentiated, not the way the derivative is taken -- two substitutions and
    nothing else:

    * the local potential is built on the shifted argument
      (:func:`~defuqumat.pseudo.potentials.local_potential_at_q`), so
      differentiating its phase brings down ``-i(G+q)`` where at ``Gamma`` it
      brought down ``-iG``, and the product with ``psi`` is gathered out of the
      box onto the ``k+q`` sphere rather than back onto ``k``;
    * the nonlocal term is written across the two spheres,
      ``|beta_{k+q}> D <beta_k|psi_k>``. **Both** factors carry the positions,
      so one ``jvp`` produces both halves of ``dvqpsi_us_only`` --
      ``|dbeta_{k+q}> D <beta_k|psi>`` and ``|beta_{k+q}> D <dbeta_k|psi>`` --
      without either being written down.

    The *value* of that function is not a physical operator: it applies
    ``V_loc(G+q)`` as if it were a potential, and its nonlocal part pairs
    projectors at two different k. Only the tangent is used, and the tangent is
    right, which is the same trade
    :func:`~defuqumat.response.phonon._bare_displacements` already makes at
    ``Gamma``.

    ``core`` is :func:`core_displacements_at_q`'s ``dv_core``, ``(nat, 3,
    nspin, n1, n2, n3)``, or ``None`` without a core charge: the XC potential
    the travelling core charge changes at a frozen valence density
    (``dvqpsi_us``'s ``addnlcc`` branch), applied as a local operator and added.

    Returns an object array of shape ``(nat, 3)`` -- or ``(len(atoms), 3)`` --
    whose entries are ``(nspin, nk, nocc, npwx_kq)``.
    """
    psi = solver.psi
    positions = jnp.asarray(positions)
    dij = tuple(h.coefficients for h in solver.hamiltonians)

    def applied(pos):
        return applied_at_q(calculation, calculation_kq, psi, pos, q_cart, dij)

    nat = positions.shape[0]
    chosen = tuple(range(nat)) if atoms is None else tuple(atoms)
    bare = np.empty((len(chosen), 3), dtype=object)
    for row, atom in enumerate(chosen):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            bare[row, cart] = compiled(
                lambda u, du: jax.jvp(applied, (u,), (du,))[1], positions, tangent)
            if core is not None:
                bare[row, cart] = bare[row, cart] + local_at_q(
                    calculation, calculation_kq, psi, core[atom, cart])
    return bare


def applied_at_q(calculation, calculation_kq, psi, pos, q_cart, dij):
    """The function :func:`bare_displacements_at_q` differentiates in ``pos``.

    ``V_loc(G+q)`` applied to ``psi_k`` and gathered onto the ``k+q`` sphere,
    plus ``|beta_{k+q}> D <beta_k|psi_k>``, at every k-point ``psi`` holds.
    Module level so that the k-chunked route
    (:mod:`defuqumat.response.chunked_phonon`) differentiates the same expression
    on a chunk's two row-subset calculations. ``dij`` is one ``D`` per channel,
    which is k-independent.
    """
    import equinox as eqx

    from defuqumat.basis.fft import gather_from_box
    from defuqumat.batching import map_bands, map_k
    from defuqumat.pseudo.potentials import local_potential_at_q

    smooth = calculation.basis.smooth
    grid = smooth.grid
    points = grid[0] * grid[1] * grid[2]
    cell = calculation.system.cell
    structure = calculation.system.structure
    pseudos = calculation.pseudos
    batch = calculation.k_batch
    index_k = calculation.fft_index
    index_kq = calculation_kq.fft_index
    mask_kq = calculation_kq.basis.planewaves.mask

    moved = eqx.tree_at(lambda s: s.positions, structure, pos)
    potential = g_to_r(
        local_potential_at_q(pseudos, moved, cell, smooth, q_cart),
        smooth.fft_index, grid,
    )
    vkb_k = calculation.projector_core.at_positions(pos).vkb
    vkb_kq = calculation_kq.projector_core.at_positions(pos).vkb

    blocks = []
    for spin in range(psi.shape[0]):
        coefficients = dij[spin].astype(vkb_kq.dtype)

        def one_k(item, coefficients=coefficients):
            states, here, there, beta_k, beta_kq, keep = item

            # The local term is a band's box in and a band's box out, so
            # it takes the band dial; the two einsums below are
            # ``(n, npwx) x (npwx, nkb)`` and hold nothing grid-sized.
            def local_block(block):
                field = g_to_r(block, here, grid)
                box = jnp.fft.fftn(
                    field * potential, axes=(-3, -2, -1)) / points
                return gather_from_box(box, there)

            local = map_bands(local_block, states)
            projected = jnp.einsum("gk,ng->nk", beta_k.conj(), states)
            nonlocal_ = jnp.einsum(
                "gk,nk->ng", beta_kq, projected @ coefficients.T
            )
            return jnp.where(keep, local + nonlocal_, 0.0)

        blocks.append(map_k(
            one_k,
            (psi[spin], index_k, index_kq, vkb_k, vkb_kq, mask_kq),
            batch=batch,
        ))
    return jnp.stack(blocks)


# ---------------------------------------------------------------------------
# The screening kernel, and the loop that applies it.
# ---------------------------------------------------------------------------

def hartree_at_q(calculation, drho, q_cart) -> jnp.ndarray:
    """``8 pi drho(G) / |G+q|^2`` on the dense grid, in real space.

    The one term of the screening kernel that knows about ``q``, and the ``G =
    0`` entry is where it shows. At ``Gamma`` that entry is dropped -- the
    average electrostatic potential of a periodic solid is not defined, and its
    divergence cancels against the Ewald and local-potential ones
    (``dv_of_drho.f90`` drops it, and so does
    :func:`defuqumat.scf.potential.hartree`). At ``q != 0`` there is nothing to
    drop: ``|G+q|`` is bounded away from zero, the ``G = 0`` coefficient is an
    ordinary finite number, and **leaving it out is a real error** rather than
    a convention -- it is the long-wavelength part of the screening, which is
    the largest part.

    The guard below is written as ``|G+q|^2 > 0`` rather than as a branch on
    ``q``, so ``q = 0`` reproduces :func:`~defuqumat.scf.potential.hartree`'s own
    exclusion exactly. That is what makes the substitution in
    :func:`induced_potential_at_q` a piece of arithmetic rather than an
    approximation.
    """
    from defuqumat.basis.fft import g_to_r as _g_to_r
    from defuqumat.units import E2, FPI

    dense = calculation.basis.dense
    cell = calculation.system.cell
    g = dense.cartesian(cell) + jnp.asarray(q_cart)[None, :]
    g2 = jnp.sum(g * g, axis=-1)
    finite = g2 > 1e-12
    inverse = jnp.where(finite, 1.0 / jnp.where(finite, g2, 1.0), 0.0)

    charge = jnp.sum(jnp.asarray(drho), axis=0)
    v = E2 * FPI * r_to_g(charge, dense.fft_index) * inverse
    return _g_to_r(v, dense.fft_index, dense.grid)[None]


def induced_potential_at_q(calculation, density, drho, q_cart) -> jnp.ndarray:
    """``dV_Hxc`` at wavevector ``q``: ``dv_of_drho.f90``.

    Built out of the ``Gamma`` kernel rather than beside it, and the two edits
    are the two ways a finite ``q`` reaches a *local* functional of the density:

    * the **exchange-correlation** part does not know about ``q`` at all. It is
      ``f_xc(r) drho(r)``, pointwise in real space, so the operator is the same
      one at every wavevector. What changes is only that ``drho`` is now
      complex, and a real linear kernel applied to a complex field is the same
      kernel on its two parts -- which is also why the ``jvp`` below is taken
      twice rather than handed a complex tangent it would refuse: the primal
      density is real, and JAX requires a tangent of the primal's dtype.
    * the **Hartree** part does, through ``1/|G+q|^2`` and through the ``G = 0``
      term it no longer drops. It is subtracted at ``q = 0`` and added back at
      ``q``, which is exact because :func:`hartree_at_q` at ``q = 0`` *is* the
      expression inside ``v_of_rho`` -- and the check on that claim is that
      ``q = 0`` through this function returns the P25 kernel to round-off.
    """
    density = jnp.asarray(density)
    drho = jnp.asarray(drho)

    def potential(rho):
        return calculation.potential(rho).v_scf

    real = jax.jvp(potential, (density,), (jnp.real(drho),))[1]
    imaginary = jax.jvp(potential, (density,), (jnp.imag(drho),))[1]
    kernel = real + 1j * imaginary
    return (
        kernel
        - hartree_at_q(calculation, drho, jnp.zeros(3))
        + hartree_at_q(calculation, drho, q_cart)
    )


def induced_perturbation_at_q(calculation, calculation_kq, dv):
    """``dV_scf(r) |psi_k>`` gathered onto the ``k+q`` sphere.

    The self-consistent half of the perturbation. ``dV_scf`` is the *periodic
    part* -- the ``e^{i q . r}`` is carried by which sphere the answer is
    gathered onto, not by the field -- so this is an ordinary local operator
    applied through the FFT box, with one index set going in and another coming
    out. That asymmetry is the only difference from
    :func:`~defuqumat.response.sternheimer.local_perturbation`.
    """
    from defuqumat.basis.fft import gather_from_box
    from defuqumat.batching import map_bands

    smooth, dense = calculation.basis.smooth, calculation.basis.dense
    grid = smooth.grid
    points = grid[0] * grid[1] * grid[2]
    field = jnp.stack([complex_to_smooth(component, dense, smooth)
                       for component in dv])
    index_k, index_kq = calculation.fft_index, calculation_kq.fft_index
    mask = calculation_kq.basis.planewaves.mask

    def apply(states, ik, spin):
        def block(chunk):
            box = jnp.fft.fftn(
                g_to_r(chunk, index_k[ik], grid) * field[spin],
                axes=(-3, -2, -1),
            ) / points
            return gather_from_box(box, index_kq[ik])

        return jnp.where(mask[ik], map_bands(block, states), 0.0)

    return apply


# ---------------------------------------------------------------------------
# The nonlinear core correction at q (I80).
# ---------------------------------------------------------------------------

def core_displacements_at_q(calculation, density, q_cart, positions,
                            host: bool = False):
    """``(drho_core, dv_core)`` per mode at ``q``, or ``None`` without a core charge.

    Each is ``(nat, 3, nspin, n1, n2, n3)`` on the dense grid, complex, and the
    periodic part of a field carrying ``e^{i q . r}``:

    * ``drho_core[a, i]`` is ``addcore.f90``: the core charge travelling with
      atom ``a`` displaced along ``i`` with ``e^{i q R}``, one ``jvp`` through the
      positions of :func:`~defuqumat.pseudo.potentials.core_charge_at_q`;
    * ``dv_core[a, i]`` is the exchange-correlation response to it,
      ``f_xc drho_core``, at the ground state's ``rho + rho_core``. ``ph.x``
      adds ``drhoc`` to ``drho`` inside ``dv_of_drho`` for the XC part alone
      and never in the Hartree term; this is :func:`induced_potential_at_q`
      with its Hartree term at ``q`` subtracted again, so the XC kernel is the
      one the valence response uses, gradient terms included.

    ``dv_core`` goes into the **bare** perturbation (:func:`bare_displacements_at_q`),
    not into ``dV_scf`` as ``ph.x`` has it: the fixed point is the same, and
    with it there :func:`response_force_constants` carries the cross term
    ``int conj(drho_i) f_xc drho_core_j`` of ``dynmat_nlcc.f90`` with its
    measured factor 2 and conjugation, and the electron-phonon ``g`` sees it
    once. What is left of ``dynmat_nlcc`` is the core-core term,
    :func:`core_force_constants`. Memory: ``2 x 3 nat`` complex dense grids,
    in host memory with ``host``.
    """
    if calculation.rho_core_g is None:
        return None
    import equinox as eqx

    from defuqumat.pseudo.potentials import core_charge_at_q

    density = jnp.asarray(density)
    if density.shape[0] != 1:
        raise NotImplementedError(
            "the core correction's response at q is written for one spin "
            "channel; require_a_two_sphere_regime refuses the others first")
    dense = calculation.basis.dense
    cell = calculation.system.cell
    structure = calculation.system.structure
    pseudos = calculation.pseudos
    positions = jnp.asarray(positions)
    q_cart = jnp.asarray(np.asarray(q_cart, dtype=float))

    def charge(pos, shift):
        moved = eqx.tree_at(lambda s: s.positions, structure, pos)
        return g_to_r(core_charge_at_q(pseudos, moved, cell, dense, shift),
                      dense.fft_index, dense.grid)

    nat = positions.shape[0]
    charges, potentials = [], []
    for atom in range(nat):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            drhoc = compiled(
                lambda u, du, shift: jax.jvp(lambda p: charge(p, shift),
                                             (u,), (du,))[1],
                positions, tangent, q_cart)[None]
            dv = (induced_potential_at_q(calculation, density, drhoc, q_cart)
                  - hartree_at_q(calculation, drhoc, q_cart))
            charges.append(np.asarray(drhoc) if host else drhoc)
            potentials.append(np.asarray(dv) if host else dv)
    stack = np.stack if host else jnp.stack
    shape = (nat, 3) + tuple(density.shape)
    return stack(charges).reshape(shape), stack(potentials).reshape(shape)


def core_force_constants(calculation, core) -> np.ndarray:
    """``(3 nat, 3 nat)``: ``int conj(drho_core_i) f_xc drho_core_j``, the core-core
    term of ``dynmat_nlcc.f90`` (its ``dot_product(dvaux, drhoc) omega / nrtot``
    with ``drho`` set to zero). ``core`` is :func:`core_displacements_at_q`'s pair.

    It is the one part of the frozen Hessian that couples the cores of two
    *different* atoms, so it is a function of ``q``; the same-atom term
    ``int v_xc d^2 rho_core`` (``dynmatcc.f90``, which says "q-independent" in
    its own header) is not.
    """
    drhoc, dvcore = core
    nat = np.shape(drhoc)[0]
    grid = calculation.basis.dense.grid
    points = grid[0] * grid[1] * grid[2]
    left = np.conj(np.asarray(drhoc)).reshape(3 * nat, -1)
    right = np.asarray(dvcore).reshape(3 * nat, -1)
    return (left @ right.T) * (calculation.system.cell.volume / points)


def swap_core(calculation, density, positions, frozen, q_cart, core=None):
    """``frozen - C(0) + C(q)``: the core-core term moved to ``q``, ``C`` being
    :func:`core_force_constants`; ``frozen`` unchanged without a core charge.

    The Gamma functional's frozen Hessian is the second derivative of
    ``E_xc[rho + rho_core(tau)]`` at a fixed ``rho``, so it holds ``C(0)``
    beside the same-atom term; ``C(0)`` is taken here by the same expression as
    ``C(q)``, so the swap is exact at ``q = 0`` and its residue against the
    Hessian's own ``C(0)`` is what the supercell check measures. ``core`` is
    :func:`core_displacements_at_q` at ``q``, built here when ``None``.
    """
    if calculation.rho_core_g is None:
        return frozen
    if core is None:
        core = core_displacements_at_q(calculation, density, q_cart, positions)
    at_zero = core_displacements_at_q(calculation, density, np.zeros(3), positions)
    return (frozen - core_force_constants(calculation, at_zero)
            + core_force_constants(calculation, core))


def local_at_q(calculation, calculation_kq, psi, field):
    """``field(r) psi_k`` gathered onto the ``k + q`` sphere, at every k-point
    ``psi`` holds: ``(nspin, nk, nbnd, npwx_kq)``.

    :func:`induced_perturbation_at_q` applied to a whole block through
    :func:`~defuqumat.batching.map_k`, for a field fixed before the loop (the
    core correction's ``dv_core``). ``field`` is ``(nspin, n1, n2, n3)`` on the
    dense grid, complex.
    """
    from defuqumat.basis.fft import gather_from_box
    from defuqumat.batching import map_bands, map_k

    smooth, dense = calculation.basis.smooth, calculation.basis.dense
    grid = smooth.grid
    points = grid[0] * grid[1] * grid[2]
    smooth_field = jnp.stack([complex_to_smooth(component, dense, smooth)
                              for component in jnp.asarray(field)])
    index_k, index_kq = calculation.fft_index, calculation_kq.fft_index
    mask_kq = calculation_kq.basis.planewaves.mask

    blocks = []
    for spin in range(psi.shape[0]):
        def one_k(item, potential=smooth_field[spin]):
            states, here, there, keep = item

            def block(chunk):
                box = jnp.fft.fftn(g_to_r(chunk, here, grid) * potential,
                                   axes=(-3, -2, -1)) / points
                return gather_from_box(box, there)

            return jnp.where(keep, map_bands(block, states), 0.0)

        blocks.append(map_k(one_k, (psi[spin], index_k, index_kq, mask_kq),
                            batch=calculation.k_batch))
    return jnp.stack(blocks)


def self_consistent_response_at_q(
    calculation, calculation_kq, solver, bare, density, q_cart,
    alpha_mix: float = 0.7, tr2: float | None = None, max_iterations: int = 100,
    mixing_mode: str = "anderson", verbose: bool = False,
    threshold_cap: float | None | str = "default",
):
    """``solve_linter``'s loop at ``q != 0``.

    The same fixed point as :func:`~defuqumat.response.phonon.self_consistent_response`
    -- solve for ``dpsi``, build ``drho``, screen it, mix, repeat -- with three
    differences and no fourth:

    * ``dpsi`` and ``drho`` are **complex** and stay complex. At ``Gamma`` the
      response of a real perturbation is real and the code takes the real part;
      here the ``+q`` and ``-q`` halves of the perturbation are different
      operators and their responses are complex conjugates rather than equal.
      The mixer sees the two parts as one real vector of twice the length, which
      is what ``mix_pot.f90`` does with the same array.
    * the kernel is :func:`induced_potential_at_q`;
    * **nothing is symmetrised.** ``symdvscf`` averages a response over the
      small group of ``q`` -- the operations with ``S q = q + G`` -- which is
      not the crystal's group and is not written here. Until it is, this needs
      the whole k-grid, and :func:`dynamical_matrix_at_q` refuses a reduced one
      by name.

    Returns ``(dpsi, drho, history, average_iterations, converged)``.
    """
    nat = np.asarray(calculation.system.structure.positions).shape[0]
    if isinstance(threshold_cap, str):
        threshold_cap = SCHEDULE_CAP_AT_Q
    tr2 = resolve_tr2_at_q(tr2, threshold_cap, getattr(solver, "threshold", None)
                           if not getattr(solver, "schedule", False) else None)
    return screening_loop_at_q(
        calculation, _WholeDisplacementsAtQ(calculation, calculation_kq, solver,
                                            bare, nat),
        density, q_cart, alpha_mix=alpha_mix, tr2=tr2,
        max_iterations=max_iterations, mixing_mode=mixing_mode, verbose=verbose,
        threshold_cap=threshold_cap,
    )


def _modes(nat: int, modes):
    """``modes`` as a list of ``(atom, cart)``, every mode when ``None``."""
    if modes is None:
        return [(atom, cart) for atom in range(nat) for cart in range(3)]
    return list(modes)


#: The CG threshold a *scheduled* solve at ``q`` is capped at, in the units of
#: :func:`~defuqumat.response.sternheimer.pass_threshold`: every pass runs at
#: ``min(0.1 sqrt(dr2), 1e-2, cap)``. ``ph.x``'s loop is the uncapped one
#: (``threshold_cap=None``), and this code's default departs from it on a
#: measurement (``PLAN.md`` P149, item 5b; ``al-elph-nosym.in`` at
#: q = (1/4, 0, 0), D22, 2026-10-14): under the bare schedule the transverse
#: pair comes out 1.5e-2 cm^-1 apart and moves by 1.4e-2 under a random
#: unitary inside the degenerate pairs at ``k`` (rule D4), while ``ph.x``'s
#: own pair is 7.8e-5 apart; a loose early pass leaves an error inside each
#: pair that the mixer folds into ``dV_scf``, the metal's response map at
#: small ``q`` amplifies it into the fixed point, and the self-consistency
#: test is blind to it, so neither a warm nor a cold tight pass at the
#: converged potential removes it (1.42e-2 either way), nor every pass from
#: zero (5.6e-4). With every pass at 1e-8 the move is 2.2e-7 and the pair is
#: 3e-6 from the one at 1e-12, at 301 against 178 CG steps (1.7x); 1e-6 is
#: gauge-safe (6e-5) but 1.4e-2 off the tight pair, and a fixed 1e-2 or 1e-4
#: never converges the loop.
SCHEDULE_CAP_AT_Q = 1.0e-8

#: The loop's ``tr2`` at ``q`` when the scheduled threshold is capped, and when
#: it is not (``ph.x``'s ``tr2_ph``). **The pair is measured together** (P149,
#: silicon ``si-epsilon-unshifted-nosym`` at q = (1/4, 1/4, 0), D22): with the
#: CG noise gone the pass-to-pass change ``|ddv_scf|^2`` falls below 1e-14
#: before the potential is converged, so at 1e-14 the stopping test reads the
#: change and not the distance to the fixed point, the capped loop stops two
#: passes short, ``D(q)`` against ``D(q + G)`` reads 1.03e-5 where the Ewald
#: floor is 8.2e-7, and the degenerate TA pair at ``q`` is split 1.4e-2 cm^-1
#: (1.0e-2 under the bare schedule at 1e-14, the aluminium disease on an
#: insulator). At 1e-16 the capped loop reads 8.2e-7 and 4.3e-5 in 11 passes,
#: the bare schedule's count at 1e-14, at 16.8 against 12.0 CG steps a solve
#: (1.4x); on aluminium the capped loop is identical at 1e-14 and 1e-16
#: (2.2e-7 under rotation, 9 passes, 301 steps), and the bare schedule at 1e-16
#: still moves the pair by 5.1e-4, so ``tr2`` alone is not the cure either.
TR2_AT_Q_CAPPED = 1.0e-16
TR2_AT_Q = 1.0e-14


def resolve_tr2_at_q(tr2, threshold_cap, threshold=None) -> float:
    """An explicit ``tr2`` as given; ``None`` is the pair's value for this loop."""
    if tr2 is not None:
        return float(tr2)
    scheduled = threshold is None
    capped = scheduled and threshold_cap is not None
    return TR2_AT_Q_CAPPED if capped else TR2_AT_Q


def _capped(threshold, cap):
    """``min(threshold, cap)`` where both are numbers, else ``threshold``."""
    if threshold is None or cap is None:
        return threshold
    return min(float(threshold), float(cap))


def _levels(threshold, count: int) -> list:
    """One CG threshold per mode from a scalar, ``None`` or a list."""
    if isinstance(threshold, (list, tuple)):
        return list(threshold)
    return [threshold] * count


class _WholeDisplacementsAtQ:
    """The ``3 nat`` solves at ``q`` with every k-point in one array: the route as it was.

    :class:`~defuqumat.response.chunked_phonon.StreamedDisplacementsAtQ` is the
    other one; :func:`screening_loop_at_q` drives either.
    """

    def __init__(self, calculation, calculation_kq, solver, bare, nat: int):
        self.calculation = calculation
        self.calculation_kq = calculation_kq
        self.solver = solver
        self.bare = bare
        self.nat = nat
        self.dpsi = np.empty((nat, 3), dtype=object)
        self.iterations = 0
        self.solves = 0

    def respond(self, dvscf, include_induced: bool, threshold=None, modes=None):
        """One iteration's solves: the complex response density per mode.

        Each solve starts from the previous pass's ``dpsi``, zeros on the first
        (``iudwf``). ``modes`` is the ``(atom, cart)`` pairs still being solved,
        every one when ``None``; ``threshold`` this pass's CG threshold, one
        value or one per mode in ``modes``.
        """
        solver = self.solver
        response = []
        modes = _modes(self.nat, modes)
        levels = _levels(threshold, len(modes))
        for (atom, cart), level in zip(modes, levels):
            if not include_induced:
                perturbation = (
                    lambda psi, ik, spin, b=self.bare[atom, cart]:
                    b[spin][ik][: psi.shape[0]]
                )
            else:
                induced = induced_perturbation_at_q(
                    self.calculation, self.calculation_kq, dvscf[atom, cart]
                )
                perturbation = (
                    lambda psi, ik, spin, b=self.bare[atom, cart], f=induced:
                    b[spin][ik][: psi.shape[0]] + f(psi, ik, spin)
                )
            previous = self.dpsi[atom, cart]
            start = (jnp.zeros_like(self.bare[atom, cart]) if previous is None
                     else previous)
            solution = solver.solve(perturbation, start=start,
                                    threshold=level)
            self.dpsi[atom, cart] = solution.dpsi
            self.iterations += solution.iterations
            self.solves += 1
            response.append(solver.response_density_at_q(solution.dpsi))
        return response

    def shift_states(self, shifts) -> None:
        """``ef_shift_wfc`` on every mode's ``dpsi``, ``shifts`` in mode order.

        Only at ``q = 0``, where the ``k + q`` sphere is the ``k`` sphere and
        ``psi_k`` is added to ``dpsi`` index by index; the shapes are checked
        rather than assumed. The rows are the solver's, cut at the grid's
        widest ``nbnd_occ``; a row past a k-point's own cut has
        ``delta(ef - eps)`` below the cut's threshold, where ``ef_shift_wfc``
        stops at ``nbnd_occ(ik)``, and the Gamma route keeps every band.
        """
        solver = self.solver
        for index in range(3 * self.nat):
            atom, cart = divmod(index, 3)
            dpsi = self.dpsi[atom, cart]
            if jnp.shape(dpsi) != jnp.shape(solver.psi):
                raise ValueError(
                    f"ef_shift_wfc needs dpsi on the k sphere: dpsi is "
                    f"{jnp.shape(dpsi)} and psi {jnp.shape(solver.psi)}")
            self.dpsi[atom, cart] = solver.fermi_level_shift_states(
                dpsi, shifts[index])


def screening_loop_at_q(
    calculation, displacements, density, q_cart, alpha_mix: float = 0.7,
    tr2: float = 1e-14, max_iterations: int = 100, mixing_mode: str = "anderson",
    verbose: bool = False, threshold_cap: float | None = None,
    shift_fermi_level: bool = False,
):
    """:func:`self_consistent_response_at_q`'s loop around a set of solves.

    ``displacements`` holds the bare perturbations and does the solves --
    every k-point in one array, or a k-chunk at a time
    (:mod:`defuqumat.response.chunked_phonon`) -- and everything here acts on
    whole-grid objects: the kernel at ``q``, the convergence test and the mixer.

    ``shift_fermi_level`` is ``ph.x``'s ``lmetq0``, a metal at ``q = 0``, and
    it is three terms of ``dfpt_kernels.f90``, not one: inside the loop each
    mode's response density takes ``def ldos`` before it is screened
    (``ef_shift``, ``:359-365``), with ``def = -int drho / N(ef)`` complex as
    ``efermi_shift.f90`` declares it; after the loop ``def`` is subtracted from
    that mode's ``dV_scf`` (``:542-551``), which only a consumer of
    ``dvscf`` reads; and ``dpsi`` takes ``(1/2) def delta(ef - eps) psi``
    (``ef_shift_wfc``), which is what the assembly contracts. Each mode keeps
    the ``def`` of the pass its ``dpsi`` was solved at, so a mode that stopped
    early is shifted by its own last value. Only the whole route has the
    displacements' ``shift_states``; :func:`dynamical_matrix_at_q` refuses the
    k-chunked one at ``q = 0`` for a metal.
    """
    from defuqumat.response.mixing import ResponseMixer, ddv_scf

    nat = displacements.nat
    grid_shape = tuple(np.shape(density))
    # **The ``3 nat`` complex grids in host memory**, one mode at a time on the
    # device, where the displacements ask for it (the k-chunked route, which at
    # ``q`` always runs without symmetry): every operation here is per mode
    # except the mixer, whose history is in host memory already. The
    # convergence test sums each mode's change, in numpy here and on the device
    # on the other route, so the two histories agree to round-off.
    host = bool(getattr(displacements, "host_fields", False))
    stack = np.stack if host else jnp.stack
    xp = np if host else jnp
    dtype = np.dtype(displacements.solver.psi.dtype)
    dvscf = xp.zeros((nat, 3) + grid_shape, dtype=dtype)
    history = []
    converged = False

    def screened(field):
        out = induced_potential_at_q(
            calculation, density, jax.device_put(field) if host else field, q_cart)
        return np.asarray(out) if host else out

    # **One mixer per mode**, which is ``ph.x``'s: ``solve_linter`` mixes each
    # irreducible representation with its own ``mix_potential`` history, and
    # with no symmetry every mode is one. Nothing physical couples the modes
    # here (no symmetrisation at q), and one Anderson history across all of
    # them is unstable at the floor for a metal at small q: on
    # ``al-elph-nosym.in`` at q = (1/4, 0, 0) the joint history reached
    # 1.5e-14 at the scheduled CG thresholds and then grew to 1e+4, where the
    # per-mode one converges in 11 iterations to ph.x 7.5's frequencies to
    # 5e-4 cm^-1 (``PLAN.md`` EP1).
    mixers = [ResponseMixer(mixing_mode, beta=alpha_mix) for _ in range(3 * nat)]
    # **And one loop per mode**, which is the rest of ``ph.x``'s: each
    # representation runs its own ``solve_linter``, with its own CG threshold
    # from its own ``dr2`` and its own stop. A mode below ``tr2`` keeps the
    # input it was solved at and is not solved again while the others finish;
    # on ``al-elph-nosym.in`` at q = (1/4, 0, 0) that is 10, 8 and 8 passes
    # rather than three times the slowest mode's 11. ``history`` stays the
    # worst mode's ``|ddv_scf|^2`` per pass, the test the loop as a whole meets.
    modes = 3 * nat
    changes = [None] * modes
    shifts = [0.0] * modes
    active = list(range(modes))
    drho = None
    if shift_fermi_level:
        # ``localdos``, once: the ground state does not change in the loop.
        levels = displacements.solver.local_density_of_states()
    for iteration in range(max_iterations):
        thresholds = [_capped(pass_threshold(displacements.solver,
                                             [] if changes[index] is None
                                             else [changes[index]]),
                              threshold_cap)
                      for index in active]
        response = displacements.respond(
            dvscf, iteration > 0, threshold=thresholds,
            modes=[divmod(index, 3) for index in active])

        if drho is None:
            drho = xp.zeros((nat, 3) + tuple(np.shape(response[0])), dtype=dtype)
        for index, field in zip(active, response):
            atom, cart = divmod(index, 3)
            if shift_fermi_level:
                # ``ef_shift``: ``drho <- drho + def ldos`` before the kernel.
                field, shifts[index] = displacements.solver.fermi_level_shift(
                    field, levels=levels)
            drho = _set(drho, (atom, cart), field)

        # ``ph.x``'s ``|ddv_scf|^2`` of each mode, as the Gamma loop tests it
        # (:data:`defuqumat.response.phonon.TR2`); the potential is complex, and
        # ``ndimtot`` counts its entries as two reals each, as at ``Gamma``.
        induced = {}
        for index in active:
            atom, cart = divmod(index, 3)
            induced[index] = screened(drho[atom, cart])
            if grid_shape[0] == 2:
                # The LSDA kernel at a point a channel empties (I82): a NaN
                # here never meets ``tr2`` and would run the whole budget.
                from defuqumat.response.efield import _require_a_finite_kernel

                _require_a_finite_kernel(calculation, induced[index], density)
            changes[index] = ddv_scf([induced[index] - dvscf[atom, cart]],
                                     joint=False)
        change = max(changes)
        history.append(change)
        if verbose:
            print(f"  response iteration {iteration + 1}: |ddv_scf|^2 = {change:.3e}"
                  f" ({len(active)} modes)")
        # Tested before mixing, and on convergence the input is kept, as
        # ``mix_potential`` keeps it; ``drho`` is this pass's output density,
        # which is ``ph.x``'s ``drhop`` too.
        limit = tr2 / calculation.system.npol
        active = [index for index in active if changes[index] >= limit]
        if not active:
            converged = True
            break

        for index in active:
            atom, cart = divmod(index, 3)
            real, imaginary = mixers[index].mix(
                [np.real(dvscf[atom, cart]), np.imag(dvscf[atom, cart])],
                [np.real(induced[index]), np.imag(induced[index])],
                host=host,
            )
            dvscf = _set(dvscf, (atom, cart), real + 1j * imaginary)

    if shift_fermi_level:
        # ``dfpt_kernels.f90:542-551``: ``dvscfp(:, is, ipert) - def(ipert)``
        # on every channel, then ``ef_shift_wfc`` on ``dpsi``.
        for index in range(modes):
            atom, cart = divmod(index, 3)
            dvscf = _set(dvscf, (atom, cart), dvscf[atom, cart] - shifts[index])
        displacements.shift_states(shifts)

    # The input ``dV_scf`` the last ``dpsi`` was solved at, which on
    # convergence is the one kept (``ph.x``'s ``dvscfins``, what ``elphel``
    # applies).
    displacements.dvscf = dvscf
    return (displacements.dpsi, drho, history,
            displacements.iterations / max(1, displacements.solves), converged)


# ---------------------------------------------------------------------------
# The ionic second derivative: d2ionq.
# ---------------------------------------------------------------------------

def _set(array, index, value):
    """``array[index] = value`` on a numpy or a JAX array, returning the array."""
    if isinstance(array, np.ndarray):
        array[index] = np.asarray(value)
        return array
    return array.at[index].set(value)


def ewald_dynamical_matrix(calculation, q_cart, alpha: float | None = None):
    """``d2ionq``: the Ewald sum's second derivative under a modulated displacement.

    **The only term of the frozen Hessian that knows about ``q``.** The
    electronic half of ``dynmat0`` -- the second derivative of the external
    potential at frozen density -- is diagonal in the atom index
    (``dynmat_us.f90`` fills ``dynwrk(na_icart, na_jcart)``, the same atom
    twice), so its two modulation phases cancel and it is the same matrix at
    every wavevector. The ion-ion sum is not diagonal, and this is what
    replaces it.

    Both halves of Ewald's split have the same shape, and it is the shape a
    lattice sum of a *pair* function always has:

        D_(ai)(bj)(q) = C_(ai)(bj)(q)  -  delta_ab sum_c C_(ai)(cj)(0)

    -- the cross term evaluated at ``q``, minus the same object at ``q = 0``
    summed over the other atom. The second piece is the atom's own recoil: an
    atom displaced alone feels the force its neighbours no longer exert, and
    that term is unmodulated because both derivatives are taken at the same
    site. It is also why the ionic matrix obeys the acoustic sum rule at
    ``q = 0`` identically rather than to the accuracy of the sum.

    Transcribed from ``PHonon/PH/d2ionq.f90``, including its choices, because
    matching them is what makes agreement with ``ph.x`` exact rather than
    approximate: ``alpha`` stepped down from 2.9 until the reciprocal
    truncation error is below **1e-9** (the ground-state ``ewald.f90`` uses 1e-7
    -- the sum is independent of ``alpha``, but the two truncations are not),
    and a real-space cutoff of ``5/sqrt(alpha)`` rather than ``4/sqrt(alpha)``.

    Returns ``(3 nat, 3 nat)`` complex, in Ry/bohr^2, with the row index
    conjugated -- the convention ``D(q)^dagger = D(q)`` is stated in.
    """
    from jax.scipy.special import erfc

    from defuqumat.scf.ewald import ewald_alpha
    from defuqumat.system.cell import lattice_translations, pair_separation_bound
    from defuqumat.units import E2, FPI

    cell = calculation.system.cell
    structure = calculation.system.structure
    dense = calculation.basis.dense
    positions = jnp.asarray(structure.positions)
    charges = jnp.asarray(calculation.charges)
    nat = positions.shape[0]
    q = jnp.asarray(q_cart)

    if alpha is None:
        tpiba2 = cell.tpiba**2
        alpha = ewald_alpha(
            float(np.sum(np.asarray(charges))), dense.ecut / tpiba2, tpiba2,
            tolerance=1.0e-9,
        )

    # -- reciprocal space --------------------------------------------------
    g = dense.cartesian(cell)
    volume = cell.volume

    def reciprocal(shift):
        return _ewald_reciprocal(g, positions, jnp.asarray(shift),
                                 jnp.asarray(volume), jnp.asarray(alpha))

    pairs = charges[:, None] * charges[None, :]
    cross = pairs[:, :, None, None] * reciprocal(q)
    self_term = pairs[:, :, None, None] * reciprocal(jnp.zeros(3))

    # -- real space --------------------------------------------------------
    rmax = 5.0 / np.sqrt(alpha)
    at = np.asarray(cell.at)
    radius = rmax + pair_separation_bound(at, np.asarray(structure.positions))
    translations = jnp.asarray(lattice_translations(at, radius))

    separation = positions[:, None, :] - positions[None, :, :]      # (nat, nat, 3)
    vectors = translations[None, None, :, :] - separation[:, :, None, :]
    square = jnp.sum(vectors * vectors, axis=-1)
    keep = (square > 1.0e-16) & (square <= rmax**2)
    distance = jnp.sqrt(jnp.where(keep, square, 1.0))
    ar = np.sqrt(alpha) * distance
    root = 2.0 / np.sqrt(np.pi)
    second = jnp.where(
        keep,
        (3.0 * erfc(ar) + root * ar * (3.0 + 2.0 * ar**2) * jnp.exp(-ar**2))
        / distance**5,
        0.0,
    )
    first = jnp.where(
        keep, (-erfc(ar) - root * ar * jnp.exp(-ar**2)) / distance**3, 0.0
    )
    # ``qrg = q . (r + dtau)``, and ``r + dtau`` is the lattice vector itself.
    phase = jnp.exp(1j * (translations @ q))[None, None, :]
    outer = vectors[..., :, None] * vectors[..., None, :]
    identity = jnp.eye(3)

    def assemble(weight):
        return E2 * pairs[:, :, None, None] * (
            jnp.einsum("abt,abtij->abij", weight * second, outer)
            + jnp.einsum("abt,ij->abij", weight * first, identity)
        )

    cross = cross + assemble(phase)
    self_term = self_term + assemble(jnp.ones_like(phase))

    matrix = cross - jnp.eye(nat)[:, :, None, None] * jnp.sum(
        self_term, axis=1
    )[:, None, :, :]
    return -jnp.transpose(matrix, (0, 2, 1, 3)).reshape(3 * nat, 3 * nat)


#: How many G vectors :func:`_ewald_reciprocal` contracts at once.
EWALD_G_CHUNK = 4096


@jax.jit
def _ewald_reciprocal(g, positions, shift, volume, alpha):
    """``sum_G w(G+q) (G+q)_i (G+q)_j e^{i(G+q).(tau_a - tau_b)}``, ``(nat, nat, 3, 3)``.

    The reciprocal half of :func:`ewald_dynamical_matrix`, **walked over G in
    blocks of** :data:`EWALD_G_CHUNK`. Contracted in one piece the einsum's
    intermediate is ``(ngm, nat, 3, 3)`` complex, which grows as ``ngm x nat``:
    on a slab of a hundred atoms and a million dense G vectors, tens of GB. On
    eight-atom silicon at 20 Ry it is small enough that the RTX A2000's peak did
    not move either way. A padded block's rows carry a weight of zero; the
    blocks are added in order, so the sum differs from the single contraction
    in its rounding only.
    """
    from defuqumat.units import E2, FPI

    ngm = g.shape[0]
    blocks = -(-ngm // EWALD_G_CHUNK)
    padded = blocks * EWALD_G_CHUNK
    valid = jnp.arange(padded) < ngm
    g = jnp.concatenate([g, jnp.zeros((padded - ngm, 3), dtype=g.dtype)])
    nat = positions.shape[0]

    def block(total, item):
        gb, live = item
        gq = gb + shift[None, :]
        g2 = jnp.sum(gq * gq, axis=-1)
        finite = (g2 > 1.0e-8) & live
        safe = jnp.where(finite, g2, 1.0)
        factor = jnp.where(
            finite, -E2 * FPI / volume * jnp.exp(-safe / alpha / 4.0) / safe, 0.0
        )
        phases = jnp.exp(1j * (gq @ positions.T))       # (chunk, nat)
        weighted = factor[:, None, None] * gq[:, :, None] * gq[:, None, :]
        return total + jnp.einsum("ga,gij,gb->abij", phases, weighted,
                                  jnp.conj(phases)), None

    total = jnp.zeros((nat, nat, 3, 3), dtype=jnp.result_type(1j, g.dtype))
    total, _ = jax.lax.scan(
        block, total,
        (g.reshape(blocks, EWALD_G_CHUNK, 3),
         valid.reshape(blocks, EWALD_G_CHUNK)))
    return total


# ---------------------------------------------------------------------------
# The assembly.
# ---------------------------------------------------------------------------

def _diagonalize_at_q(matrix: np.ndarray, masses: np.ndarray):
    """``dyndia`` for a complex hermitian ``D(q)``.

    :func:`~defuqumat.response.phonon._diagonalize` casts to ``float``, which is
    right at ``Gamma`` -- the force constants of a real crystal are real there
    -- and is not right anywhere else. Everything else is the same: mass-weight
    with ``amu_ry``, diagonalise, and report an imaginary frequency as a
    negative number.
    """
    from defuqumat.units import AMU_TO_RY, RY_TO_CMM1

    nat = masses.shape[0]
    flat = np.asarray(matrix, dtype=complex).reshape(3 * nat, 3 * nat)
    scale = 1.0 / np.sqrt(np.repeat(masses, 3) * AMU_TO_RY)
    weighted = flat * scale[:, None] * scale[None, :]
    omega2, vectors = np.linalg.eigh(0.5 * (weighted + weighted.conj().T))
    return np.sign(omega2) * np.sqrt(np.abs(omega2)) * RY_TO_CMM1, vectors


def response_force_constants(solver, dpsi, bare, nat) -> np.ndarray:
    """``drhodv``: the electronic half of ``D(q)``, in one contraction.

    QE splits this in two because its two halves live in different places --
    ``drhodvloc`` integrates ``conj(drho_i) dV_loc_j`` on the FFT grid, and
    ``drhodvnl`` rebuilds the nonlocal half in the projector basis from
    ``dbecq`` and ``dalpq``. Here they are one inner product, because the
    object QE has to rebuild is one this code already holds: ``bare[j]`` **is**
    ``dV_bare_j |psi>`` on the ``k+q`` sphere, local and nonlocal together, and
    contracting it against ``dpsi_i`` on that same sphere gives both halves at
    once.

        D^resp_(ai)(bj) = 2 sum_{k,n} w_kn <dpsi_(ai) | dV_bare_(bj) | psi_n k>

    **The leading 2 is not a derivation, it is a measurement**, and so is the
    index order. Both are the shape a Hermitian-looking answer can be wrong in
    (``PLAN.md`` P54, P66), so they were fixed at ``q = 0`` against the P25
    matrix on the 64-point unshifted silicon grid rather than argued for: the
    ratio of ``D_P25 - D_frozen`` to this contraction is **2.000000** on every
    element the contraction reaches, with the off-diagonal blocks agreeing at
    the 1e-5 level where both are numerically zero. It is the same 2 that
    ``incdrhoscf`` carries and for the same reason -- the ``-q`` half of a real
    perturbation contributes as much as the ``+q`` one.

    The conjugate sits on ``dpsi`` and the row index is the perturbation
    ``dpsi`` solves, which is ``drhodvloc``'s ``dot_product(drhos(ipert), dvloc(nu_j))``
    read through ``drho ~ conj(psi) dpsi``.

    **What this costs**, since a design is not finished until its working set is
    known: the loop is ``(3 nat)^2`` passes over an ``(nspin, nk, nocc, npwx)``
    block, which is the same flop count a single ``(3 nat, N)`` Gram matrix
    would be and worse constants. It allocates nothing -- ``dpsi`` and ``bare``
    are already held -- and stacking them into that matrix would hold a second
    copy of the largest arrays a phonon run has. On the cells this phase admits
    the Sternheimer solves dominate it by two orders; on a large cell the trade
    is worth revisiting, and the stack is where to start.
    """
    from defuqumat.response.phonon import _state_weights

    # ``wg`` for an insulator, ``wk`` for a metal, whose occupation is already
    # inside ``dpsi`` (``orthogonalize``'s ``wg1 dvpsi``): ``drhodvnl.f90:181``
    # contracts with ``2 wk``. An insulator cannot tell the two apart.
    weights = _state_weights(solver, solver.weights)
    matrix = np.zeros((3 * nat, 3 * nat), dtype=complex)
    for atom in range(nat):
        for cart in range(3):
            row = 3 * atom + cart
            tangent = jnp.conj(dpsi[atom, cart])
            for other in range(nat):
                for direction in range(3):
                    matrix[row, 3 * other + direction] = complex(2.0 * jnp.sum(
                        weights * jnp.einsum(
                            "skng,skng->skn", tangent, bare[other, direction]
                        )
                    ))
    return matrix


def frozen_force_constants(calculation, solver, positions, density, q_cart,
                           core=None):
    """``dynmat0`` at ``q``: the second derivative at a frozen electronic state.

    Three terms, and without a core charge only the last of them knows about
    ``q``:

        dynmat0(q) = dynmat_us + dynmatcc + d2ionq(q)

    ``dynmat_us`` and ``dynmatcc`` are taken from the ``Gamma`` machinery
    unchanged -- :func:`~defuqumat.response.phonon._force_constants` with the
    state and density tangents set to zero is exactly ``dynmat0(0)``, which is
    those two plus ``d2ionq(0)``. The ionic term is then swapped: the **same**
    ``jax.hessian`` of the **same** Ewald energy that is inside that functional
    is subtracted, so the swap is exact to round-off and carries no
    disagreement of its own, and :func:`ewald_dynamical_matrix` is added in its
    place.

    That the other two survive the swap is the fact this whole phase rests on
    and it was read in the Fortran rather than assumed:
    ``dynmat_us.f90:105-124`` writes into ``dynwrk(na_icart, na_jcart)`` -- one
    atom index, used twice -- and ``init_us_2`` there is called at ``ikk``, so
    nothing in the routine is a function of ``xq``. A second derivative taken
    twice at the *same* site carries ``e^{iqR} e^{-iqR} = 1``. That reading
    covers the norm-conserving routine only: ``dynmat_us.f90:236`` then calls
    ``addusdynmat``, whose ``int2``/``int5`` terms are cross-atom and carry
    ``xq`` (``dvanqq.f90:106-191``), which is one of the reasons an ultrasoft or
    PAW dataset is refused at ``q != 0``.

    **A core charge adds one term that does know about** ``q`` (I80). The
    Gamma functional differentiates ``E_xc[rho + rho_core(tau)]`` twice at a
    fixed ``rho``, which is ``dynmatcc``'s same-atom ``int v_xc d^2 rho_core``
    (q-independent, as ``dynmatcc.f90``'s header says; its
    ``set_drhoc(xq, drc)`` at ``:105`` only prepares the table for
    ``addcore``) **plus** ``int drho_core_i f_xc drho_core_j``, which couples
    two *different* atoms' cores inside a nonlinear ``E_xc`` and is the
    core-core half of ``dynmat_nlcc.f90``. That one is swapped like the Ewald
    sum, :func:`swap_core`; ``core`` is :func:`core_displacements_at_q` at
    ``q``, built when ``None``.
    """
    from defuqumat.response.phonon import _force_constants, _state_weights

    nat = np.asarray(positions).shape[0]
    empty = np.empty((nat, 3), dtype=object)
    for atom in range(nat):
        for cart in range(3):
            empty[atom, cart] = jnp.zeros_like(solver.psi)
    frozen = np.asarray(_force_constants(
        calculation, jnp.asarray(positions), solver.psi, solver.weights,
        _state_weights(solver, solver.weights), solver.eigenvalues,
        jnp.asarray(density), empty,
        jnp.zeros((nat, 3) + jnp.asarray(density).shape), solver.nocc,
    )).reshape(3 * nat, 3 * nat)

    return swap_core(calculation, density, positions,
                     swap_ewald(calculation, positions, frozen, q_cart),
                     q_cart, core)


def swap_ewald(calculation, positions, frozen, q_cart):
    """``frozen - d2ionq(0) + d2ionq(q)``: the frozen Hessian moved to ``q``.

    ``frozen`` is the ``(3 nat, 3 nat)`` second derivative of the frozen-state
    functional at ``Gamma``, which carries the Ewald sum's Hessian there; the
    rest of it is the same at every ``q`` (see :func:`frozen_force_constants`).
    """
    from defuqumat.response.chunked import _jacobian_by_columns

    nat = np.asarray(positions).shape[0]
    cell = calculation.system.cell
    dense = calculation.basis.dense
    gradient = jax.grad(lambda pos: calculation.ewald_sum.energy(cell, pos, dense))
    # **A column at a time**, where ``jax.hessian`` takes all ``3 nat`` tangents
    # at once and holds that many copies of the Ewald sum's structure factors,
    # ``(ngm, nat)`` each: that was the peak of the whole k-chunked phonon at
    # ``q`` on the RTX A2000, 103.0 MB against 42.5 for everything else,
    # eight-atom silicon at 20 Ry with the SCF's own peak at 35.1. Compiled by
    # its structure, so a second call compiles nothing (:mod:`defuqumat.eager`).
    ionic = np.asarray(compiled(
        lambda pos: _jacobian_by_columns(gradient, pos), jnp.asarray(positions)
    )).reshape(3 * nat, 3 * nat)

    return frozen - ionic + np.asarray(ewald_dynamical_matrix(calculation, q_cart))


def require_a_two_sphere_regime(calculation, q_crystal) -> None:
    """What a phonon at ``q != 0`` refuses, and why each one is named.

    Every entry here is a term that is *absent* rather than a feature that is
    unwritten, which is the distinction ``PLAN.md`` asks a refusal to make.
    """
    system = calculation.system
    if calculation.is_ultrasoft:
        raise NotImplementedError(
            "a phonon at q != 0 with an ultrasoft or PAW dataset is not "
            "implemented, and it is a second implementation rather than four "
            "missing terms (notes/incompat/I81.md, read in PHonon/PH). At "
            "Gamma every augmented term falls out of one jvp of the force's "
            "gradient (P39); at q != 0 there is no cell energy to "
            "differentiate, and response_force_constants is the "
            "hand-transcribed drhodv contraction, so each term is a "
            "transcription with no tangent to check it against. Absent here: "
            "(1) addusddens's sum_ij dbecsum_ij Q_ij(q+G) and the moving "
            "Q_ij's own term in drho, (2) adddvscf's int3 = int dV_scf Q_ij "
            "e^{iqr} in the perturbation, (3) dvanqq's int1 and int2 in the "
            "bare term (bare_displacements_at_q freezes D_ij), (4) P39's whole "
            "orthogonality set at q: dpsi^ort pairs evq at k+q with evc at k "
            "(compute_drhous.f90 builds vkb at ikq), drhous and becsumort in "
            "the screening, dLambda, drhodvus in the assembly and the int1/"
            "int2 half of drhodvnl, (5) a frozen Hessian that is not "
            "atom-diagonal: addusdynmat contracts dvanqq's int2 and int5, "
            "which pair V_loc(G+q) of one atom with Q_ij(G+q) of another under "
            "eigqts(na) conj(eigqts(nb)), so frozen_force_constants' 'the "
            "same matrix at every q' is a norm-conserving fact, and (6) for "
            "PAW the one-centre response at q. q_ij(q+G) itself is written "
            "(defuqumat.tddft.spinchi0.augmentation_factors). Use a "
            "norm-conserving dataset, or, for an insulator, the Gamma phonon "
            "of a commensurate supercell"
        )
    if system.noncolin:
        # ``nspin = 2`` runs (I82): the solve, the bare term and the assembly
        # already walked the channel axis, and the response density keeps one
        # sum per channel (:meth:`TwoSphereSolver.response_parts_at_q`).
        raise NotImplementedError(
            "a phonon at q != 0 of a noncollinear or spin-orbit (spinor) "
            "calculation is not implemented, and it is three absent terms: the "
            "response density is a spinor pair density across two spheres, "
            "conj(psi_k) sigma dpsi_{k+q}, with four components and no "
            "counterpart of spinor_sum_band at q (response_parts_at_q sums "
            "channels of a collinear density); applied_at_q's nonlocal term "
            "would pair the two spheres' projectors through D as a 2x2 matrix "
            "in spin space; and the Gamma phonon of a spinor, which the frozen "
            "Hessian is taken from, is itself refused. nspin = 2 collinear runs"
        )
    if system.nspin == 2 and system.occupations != "fixed":
        raise NotImplementedError(
            "a spin-polarized metal at q != 0 is not implemented: the two "
            "channels' solve and response density are in place and the "
            "insulator is checked (a magnetic hydrogen chain at q = b3/2 against "
            "its doubled cell at Gamma), but the smeared two-sphere projector "
            "with two channels sharing one Fermi level has not been run against "
            "anything. Use occupations = 'fixed' with tot_magnetization, or "
            "nspin = 1"
        )
    if system.nspin == 2 and calculation.rho_core_g is not None:
        # I80 measured the core term at nspin = 1 and I82 the two channels on
        # datasets without a core; the core charge split between two channels
        # inside the spin-polarized kernel was measured by neither.
        raise NotImplementedError(
            "a phonon at q != 0 of a spin-polarized (nspin = 2) calculation "
            "with a nonlinear core correction is not measured: the core term "
            "at q runs at nspin = 1 and the two channels run on datasets "
            "without a core, but the pair has not been checked. Use a dataset "
            "without PP_NLCC, or nspin = 1"
        )
    if calculation.use_symmetry:
        raise NotImplementedError(
            "a phonon at q != 0 needs the whole k-grid: the response has to be "
            "symmetrised over the small group of q -- the operations with "
            "S q = q + G -- which is not the crystal's group and is not "
            "written here. Run with nosym = .true. and noinv = .true. on an "
            "unshifted grid, which is the grid that is closed under the point "
            "group (see si-epsilon-unshifted-nosym.in)"
        )
    if system.spiral_q is not None:
        raise NotImplementedError(
            "a phonon at q != 0 on a spin spiral is not implemented: both "
            "already put the states on shifted spheres and the two shifts "
            "compose, so the perturbation would need four"
        )
    if calculation.functional.is_meta:
        raise NotImplementedError(
            "a phonon under a potential-only meta-GGA is not implemented, at "
            "any q: tb09 and bj06 are potentials with no energy functional, so "
            "there is no second derivative of an energy to take (P30)"
        )
    if calculation.functional.is_gradient:
        # Measured (I82 verifier, D22): the supercell identity that holds to
        # 1.5e-5 Ry/bohr^2 under the LDA reads 7.97e-3 and 60.55 cm^-1 under
        # PBE on the magnetic H3 chain at b3/2 (nspin = 2), and 1.33e-2 and
        # 77 cm^-1 on a dimerized H2 chain at nspin = 1, silently.
        raise NotImplementedError(
            "a phonon at q != 0 under a gradient-corrected functional is not "
            "implemented, at any nspin: induced_potential_at_q takes the jvp "
            "of v_scf on the periodic part of drho e^{iqr}, which is the whole "
            "kernel only for a functional local in the density. A GGA kernel "
            "differentiates the field, and on drho e^{iqr} every gradient is "
            "e^{iqr} (grad + iq), so the iq terms of dgradcorr.f90 (its "
            "fft_qgradient and fft_qgraddot, both called with xq) are absent. Measured against the "
            "doubled cell at Gamma, the error is 1e-2 Ry/bohr^2 and 60 to 77 "
            "cm^-1 where the LDA agrees to 1.5e-5. Use an LDA functional"
        )
    if getattr(calculation, "source_free", False):
        raise NotImplementedError(
            "a phonon under nosource is not implemented, at any q: the "
            "source-free projection changes the potential and not the energy, "
            "so there is no second derivative of an energy to take"
        )
    if calculation.is_hubbard:
        raise NotImplementedError(
            "a phonon at q != 0 with a Hubbard U is not implemented: the "
            "projectors move with the atoms and their q-modulated response is "
            "dnsq_bare/dnsq_scf, which is a second implementation rather than "
            "a shift of this one"
        )


def _require_one_channel_when_walked(calculation, streamed: bool) -> None:
    """``nspin = 2`` on the k-chunked route at ``q``: refused, as at ``Gamma``.

    The route's frozen Hessian is the ``Gamma`` route's walked passes, which
    refuse two channels (:func:`~defuqumat.response.phonon._require_one_spin_channel`),
    and its own passes have not been run at ``nspin = 2``.
    """
    if streamed and calculation.system.nspin == 2:
        raise NotImplementedError(
            "the dynamical matrix at q of a spin-polarized calculation is "
            "implemented on the whole-k route only; this run walks the k axis "
            "a chunk at a time (the k-chunked route: a streamed store, memory "
            "mode with a chunk smaller than the mesh, or memory mode on a card), "
            "whose frozen Hessian is the Gamma route's walked passes, and those "
            "refuse nspin = 2. The whole-k route is memory_mode='speed' with "
            "the states held whole rather than streamed"
        )


def _reciprocal_lattice_vector(cell, q_cart):
    """``q`` in crystal coordinates rounded to integers when it is a reciprocal
    lattice vector (to 1e-8), else ``None``. ``q_cart`` in 1/bohr."""
    crystal = np.asarray(cell.k_to_crystal(np.asarray(q_cart) / cell.tpiba))
    nearest = np.rint(crystal)
    if np.max(np.abs(crystal - nearest)) < 1.0e-8:
        return nearest
    return None


def at_zone_centre(cell, q_cart) -> bool:
    """Whether ``q`` is ``0`` itself (to 1e-8 in crystal coordinates), where a
    metal's Fermi level moves: ``ph.x``'s ``lgamma`` for ``lmetq0``."""
    nearest = _reciprocal_lattice_vector(cell, q_cart)
    return nearest is not None and not np.any(nearest)


def _require_a_metallic_q(cell, q_cart) -> None:
    """Where a metal is refused on this route: at a nonzero ``q = G``, by its missing term.

    Both routes, whole and k-chunked, carry the metal branch of the two-sphere
    projector, and the whole route carries ``ef_shift`` at ``q = 0`` (I4), so
    what is left is a reciprocal lattice vector other than zero.
    """
    nearest = _reciprocal_lattice_vector(cell, q_cart)
    if nearest is not None and np.any(nearest):
        raise NotImplementedError(
            "a metal at a nonzero reciprocal lattice vector q = G through the "
            "q != 0 route is not implemented: the perturbation there is lattice "
            "periodic, so the Fermi level moves as at q = 0 (ph.x's ef_shift), "
            "but the k + q sphere is the k + G sphere, and ef_shift_wfc adds "
            "(1/2) def delta(ef - eps) psi_k, a state on the k sphere, to dpsi on "
            "it, which needs the Miller-index map between the two spheres; "
            "ph.x itself shifts the level only at xq = 0 (lmetq0 = lgauss .AND. "
            "lgamma). Use q = 0, which this route carries, or get_phonons()"
        )


def dynamical_matrix_at_q(
    calculation, wavefunctions, eigenvalues, density, becsum=(),
    q=(0.0, 0.0, 0.0), q_cartesian: bool = False, nbnd: int | None = None,
    threshold: float | None = None, alpha_mix: float = 0.7, tr2: float | None = None,
    max_iterations: int = 100, verbose: bool = False,
    keep_internals: bool = False, threshold_cap: float | None = SCHEDULE_CAP_AT_Q,
):
    """``D(q)``: the dynamical matrix at one wavevector.

    ``q`` is in **crystal** coordinates of the reciprocal lattice unless
    ``q_cartesian`` is set, in which case it is in units of ``2 pi / alat`` --
    the units ``ph.x`` prints ``xq`` in and the units
    :class:`~defuqumat.system.kpoints.KPoints` stores.

    The sequence is ``phonon.f90``'s: diagonalise the converged potential again
    on the ``k+q`` sphere, build the bare perturbations there, run
    ``solve_linter``'s fixed point, and assemble. What differs from ``Gamma``
    is only where things live, which is the subject of the module docstring.

    Returns a :class:`~defuqumat.response.phonon.Phonons` whose ``matrix`` is
    **complex**: ``D(q)`` is hermitian rather than symmetric, and only at
    ``q = 0`` (or at a wavevector where the crystal has inversion) does it
    collapse to a real matrix.

    **There is no** ``atoms=`` **here**, which is the first lever a large cell
    would reach for at ``Gamma`` (:func:`~defuqumat.response.phonon.dynamical_matrix`
    has it). It is left out rather than half-written: the response half of the
    assembly would index a subset's rows while the frozen half and
    :func:`ewald_dynamical_matrix` are whole-cell objects, and a partial
    dynamical matrix that silently mixed the two would be a plausible wrong
    answer rather than an error. Adding it means slicing all three together.

    ``tr2`` is the loop's stop on ``|ddv_scf|^2``; ``None`` is :data:`TR2_AT_Q_CAPPED` (1e-16) under the cap and :data:`TR2_AT_Q` (1e-14, ``ph.x``'s ``tr2_ph``) without it, a pair measured together (their docstring), and a number given is used as it is.

    ``threshold_cap`` caps the *scheduled* CG threshold at every pass, ``min(0.1 sqrt(dr2), 1e-2, cap)``, :data:`SCHEDULE_CAP_AT_Q` by default; ``None`` is ``ph.x``'s uncapped loop, for a like-for-like comparison, and a number given as ``threshold`` is never scheduled and never capped. Why the default departs from ``ph.x`` is :data:`SCHEDULE_CAP_AT_Q`'s docstring.

    ``keep_internals`` attaches the two-sphere solver, the bare perturbations
    and the converged ``dV_scf`` to the result's ``internals``, which is what
    :mod:`defuqumat.response.elph` builds the coupling from. On the streamed
    route they are the host stores (``"displacements"``), and the coupling is
    walked a chunk at a time from them.
    """
    tr2 = resolve_tr2_at_q(tr2, threshold_cap, threshold)
    from defuqumat.response.phonon import Phonons
    from defuqumat.response.sternheimer import make_sternheimer

    cell = calculation.system.cell
    q_crystal = None if q_cartesian else np.asarray(q, dtype=float)
    require_a_two_sphere_regime(calculation, q_crystal)

    if q_cartesian:
        q_cart = np.asarray(q, dtype=float) * cell.tpiba
    else:
        q_cart = np.asarray(
            cell.k_to_cartesian(np.asarray(q, dtype=float))
        ) * cell.tpiba

    structure = calculation.system.structure
    positions = jnp.asarray(structure.positions)
    nat = structure.nat

    # **The route is decided before anything converts the states**, by the
    # rule every response shares (:func:`~defuqumat.response.efield._streams`):
    # where it says to walk, the states at ``k`` and at ``k + q``, the bare
    # perturbations and ``dpsi`` stay in host memory and every pass is one
    # k-chunk's (:mod:`defuqumat.response.chunked_phonon`).
    from defuqumat.response.efield import _streams

    metal = calculation.system.occupations != "fixed"
    if metal:
        _require_a_metallic_q(cell, q_cart)
    shift_fermi_level = metal and at_zone_centre(cell, q_cart)
    # The two regimes the walked passes do not carry take the whole-k route
    # instead (:func:`~defuqumat.response.efield._streams`), which is where
    # they were measured.
    lacks = ("the second spin channel" if calculation.system.nspin == 2 else
             "the metal's Fermi-level shift at q = 0 (ldos, ef_shift_wfc)"
             if shift_fermi_level else None)
    streamed = _streams(calculation, wavefunctions, keep_internals=False,
                        what="the dynamical matrix at q", whole_k=lacks)
    if not streamed:
        wavefunctions = jnp.asarray(wavefunctions)
    # The core correction's two fields per mode, ``None`` without one (I80),
    # in host memory beside the loop's own grids on the streamed route.
    core = core_displacements_at_q(calculation, density, q_cart, positions,
                                   host=streamed)
    dv_core = None if core is None else core[1]
    _require_one_channel_when_walked(calculation, streamed)
    if shift_fermi_level and streamed:
        raise NotImplementedError(
            "a metal at q = 0 through the k-chunked route of the dynamical "
            "matrix at q is not implemented: its displacements have no ldos or "
            "ef_shift_wfc pass at q (the Gamma route's StreamedDisplacements "
            "has both), so the Fermi level's shift is not carried. Use "
            "get_phonons(), whose k-chunked route does"
        )
    result = _GroundState(wavefunctions, eigenvalues, density, becsum)
    # A metal's Fermi level is re-derived from these eigenvalues by the call
    # the SCF made, as the ``Gamma`` route does (``phonon._fermi_level``).
    if metal:
        from defuqumat.response.phonon import _fermi_level

        levels = _fermi_level(calculation, jnp.asarray(eigenvalues).reshape(
            (-1,) + jnp.shape(eigenvalues)[-2:]))
        result.fermi_energy = levels.fermi_energy
    # The bare solves stay at the fixed 1e-14 this entry always used; the
    # loop's are scheduled unless a number is given (``pass_threshold``).
    solver = make_sternheimer(
        calculation, result, threshold=1.0e-14 if threshold is None else threshold,
        metals=True, spin_polarized=True, host_store=streamed)
    solver.schedule = threshold is None
    potential = calculation.potential(density)

    kq, hamiltonians_kq, eigenvalues_kq, psi_kq = states_at_k_plus_q(
        calculation, potential.v_scf, q_cart,
        # ``jnp.shape`` rather than ``np.asarray(...).shape``: the second pulls
        # the whole ground-state block to the host to read one integer, and JAX
        # then keeps that host copy on the array for its lifetime.
        nbnd=nbnd or jnp.shape(wavefunctions)[2], stream=streamed,
    )
    if streamed:
        from defuqumat.response.chunked_phonon import StreamedDisplacementsAtQ

        displacements = StreamedDisplacementsAtQ(
            calculation, kq, solver, hamiltonians_kq, eigenvalues_kq, psi_kq,
            q_cart, positions, core=dv_core,
        )
        displacements.prepare()
        dpsi, drho, history, average, converged = screening_loop_at_q(
            calculation, displacements, density, q_cart,
            alpha_mix=alpha_mix, tr2=tr2, max_iterations=max_iterations,
            verbose=verbose, threshold_cap=threshold_cap,
        )
        matrix = (
            swap_core(calculation, density, positions,
                      swap_ewald(calculation, positions,
                                 displacements.frozen_force_constants(density),
                                 q_cart),
                      q_cart, core)
            + displacements.response_force_constants()
        )
    else:
        two = TwoSphereSolver(solver, hamiltonians_kq, psi_kq, eigenvalues_kq, kq)

        # The coupling's matrix elements run over every band at ``k``
        # (``elphel``), where a metal's solve stops at its cut, so with
        # ``keep_internals`` the bare vectors are built on the whole block and
        # the solve takes their leading rows.
        whole = bare_displacements_at_q(
            calculation, kq, solver if keep_internals else two, q_cart, positions,
            core=dv_core)
        rows = two.psi.shape[2]
        bare = np.empty_like(whole)
        for index in np.ndindex(whole.shape):
            bare[index] = whole[index][:, :, :rows]
        displacements = _WholeDisplacementsAtQ(calculation, kq, two, bare, nat)
        dpsi, drho, history, average, converged = screening_loop_at_q(
            calculation, displacements, density, q_cart,
            alpha_mix=alpha_mix, tr2=tr2, max_iterations=max_iterations,
            verbose=verbose, threshold_cap=threshold_cap,
            shift_fermi_level=shift_fermi_level,
        )

        matrix = (
            frozen_force_constants(calculation, solver, positions, density, q_cart,
                                   core=core)
            + response_force_constants(two, dpsi, bare, nat)
        )
    asymmetry = float(np.max(np.abs(matrix - matrix.conj().T)))
    matrix = 0.5 * (matrix + matrix.conj().T)

    frequencies, vectors = _diagonalize_at_q(matrix, np.asarray(structure.masses))
    internals = None
    if keep_internals and streamed:
        # ``g`` runs over every band the stores hold at ``k`` and every band
        # at ``k + q``, and so do the eigenvalues ``elphsum`` weights it with.
        internals = {"displacements": displacements, "calculation_kq": kq,
                     "dvscf": displacements.dvscf, "q_cart": q_cart,
                     "psi": solver.psi, "eigenvalues": solver.eigenvalues,
                     "eigenvalues_kq": np.asarray(eigenvalues_kq)}
    elif keep_internals:
        internals = {"solver": two, "bare": whole, "calculation_kq": kq,
                     "dvscf": displacements.dvscf, "q_cart": q_cart,
                     "psi": solver.psi, "eigenvalues": solver.eigenvalues,
                     "eigenvalues_kq": np.asarray(two.eigenvalues_kq)}
    return Phonons(
        matrix=matrix.reshape(nat, 3, nat, 3),
        frequencies=frequencies,
        eigenvectors=vectors,
        induced_density=np.asarray(drho),
        asymmetry=asymmetry,
        history=history,
        average_iterations=average,
        converged=converged,
        internals=internals,
    )


class _GroundState:
    """The four fields :func:`make_sternheimer` reads off an ``SCFResult``.

    A shim rather than a refactor: the solver's constructor takes a result
    object, and this entry point takes the four arrays directly, as
    :func:`~defuqumat.response.phonon.dynamical_matrix` does.
    """

    def __init__(self, wavefunctions, eigenvalues, density, becsum):
        self.wavefunctions = wavefunctions
        self.eigenvalues = eigenvalues
        self.density = density
        self.becsum = becsum
        self.occupations = None
        self.fermi_energy = None
