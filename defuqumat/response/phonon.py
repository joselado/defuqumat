"""Phonons at ``Gamma``: the force constants as a second derivative of the energy.

``PLAN.md`` P25. This is the other half of what the Sternheimer solver was
built for (P24): the electric field's response gave ``epsilon_infinity`` and the
Born charges, and the *ionic* perturbation gives the dynamical matrix.

**The whole assembly is one derivative of code that already exists.** The force
is ``jax.grad`` of :func:`~defuqumat.forces.energy.frozen_energy` with respect to
the positions at a frozen electronic state (P15); the force *constants* are that
same gradient differentiated once more, and the only new ingredient is which
direction to differentiate it in. Write the energy as a function of both the
coordinate and the state, ``L(u, psi)``, chosen so that it is stationary in
``psi`` at the solution -- which is what carrying the orthonormality constraint
with its multipliers buys, and what makes the force a *partial* derivative. Then

    dE/du_j            = d_j L                              (the force, P15)
    d^2E/du_i du_j     = d_i d_j L + (d_psi d_j L) . dpsi_i

exactly: no second-order wavefunction, no ``<dpsi|H - eps S|dpsi>`` term, and no
factor to get right. The second piece is a *directional* derivative of the same
gradient along ``dpsi_i``, so **one ``jvp`` of ``jax.grad(L)`` along the tangent
``(e_i, dpsi_i)`` returns a whole column of the matrix** -- the frozen Hessian
and the electronic response together, in one call, with nothing separating them
by hand.

What that replaces on the Fortran side is most of ``PHonon/PH``:

| QE | here |
|---|---|
| ``dynmat0`` + ``d2ionq`` + ``dvloc`` second derivatives | the ``u`` half of the ``jvp`` |
| ``drhodv`` | the ``psi`` half of the same ``jvp`` |
| ``dvqpsi_us`` (the bare perturbation) | one ``jvp`` through ``at_positions`` |
| ``dv_of_drho`` (the screening kernel) | one ``jvp`` of ``v_of_rho`` |
| ``dyndia`` | mass-weighting and an ``eigh`` |

``solve_linter``'s self-consistent loop, ``symdvscf``, ``symdynph_gq`` and the
diagonalisation are transcribed; the perturbations and the second derivative are
not.

**The trap this phase adds is the third appearance of ``abs``.**
:func:`~defuqumat.forces.energy.frozen_energy` was written to be differentiated
with respect to the *positions*, and it contained ``jnp.abs(psi)**2`` in its
kinetic and constraint terms because on that path nothing ever differentiated it
with respect to the states. Here something does, and ``abs``'s derivative is
``Re(conj z t)/|z|`` -- ``0/0`` at a coefficient that vanishes, which by symmetry
happens on the nose. It is
:func:`defuqumat.scf.density.band_density`'s trap and
:func:`defuqumat.basis.gvectors.modulus`'s, in a third place, and the symptom is
a NaN in every force constant rather than a wrong number.

**The symmetrisation is P24's trap with one more index**, and it appears twice
for two different reasons. Inside the self-consistent loop, the response density
of a perturbation on a symmetry-reduced k-set is not what the whole grid would
give, and what has to be averaged is not three scalar fields but ``3 nat``
fields carrying *both* a direction and an atom label
(:meth:`~defuqumat.scf.driver.Calculation.symmetrize_atom_displacement`,
``symdvscf``): an operation rotates the displacement and carries it to the atom
it maps onto. That fixes the screening each perturbation sees. It does **not**
fix the wedge sum in the assembled matrix, which is a rank-2 tensor with two
atom indices and needs
:func:`~defuqumat.system.symmetry.symmetrize_atom_pair_tensor` (``symdynph_gq``)
-- the same argument that makes ``symvector`` non-cosmetic for a force, two
ranks up. Both are measured in ``PLAN.md`` P25.

**Insulators and metals, and the metal is one weight rather than a routine.**
The frozen energy weights its states by ``wg = wk f``, which is right for the
frozen Hessian and wrong for the electronic half, because a metal's ``dpsi``
already carries ``f`` from ``orthogonalize``'s smeared right-hand side. So the
one ``jvp`` above is two: the coordinate and the density at ``wg``, the states
at ``wk``. That is what ``dynmat_us.f90`` (``wg``) and ``drhodvnl.f90``
(``2 wk``) are, and it is the whole of ``PLAN.md`` P28 --
:func:`_state_weights`. Two-atom aluminium's modes come out at 146.711240 and
311.033545 cm^-1 against ``ph.x``'s 146.710511/146.714378 and 311.035401, and
the acoustic sum rule holds to 1.06e-5 Ry/bohr^2 where the unsplit assembly
violated it by half the spectrum.

**Norm-conserving only, and the reason is in the formula rather than in a
missing routine.** The identity above holds because ``L`` is stationary in
``psi`` at *fixed* multipliers, and the multipliers are attached to the
constraint ``<psi|S(u)|psi> - 1``. Differentiating a second time leaves a term
``-<psi|dS/du_j|psi> . deps_i`` which vanishes identically when ``S`` does not
move with the atoms -- that is, for a norm-conserving dataset, where ``S = 1``.
For an ultrasoft or PAW one it does not vanish, and there is a second gap beside
it: the augmentation charge ``Q_ij(r - tau)`` moves at frozen ``becsum``, which
``addusdynmat`` and ``drhodvus`` account for. Both are written now (P39), and
the elastic constants carry the strain coordinate's counterparts of the first
since I87 (``ort`` and ``dLambda``, :func:`~defuqumat.response.strain.
multiplier_response`). The measurement that stood behind refusing them is the
same shape as ``zstar_eu_us``'s: with the guard lifted and
the terms unwritten, ultrasoft silicon's
optical mode comes out at **-504.3 cm^-1** against ``ph.x``'s **+513.3** --
imaginary where the crystal is stable -- from a run that converges and gives a
cubic, symmetric matrix.

**What this costs in memory**, since a design is not finished until its peak
working set is known: ``3 nat`` bare perturbations and ``3 nat`` first-order
wavefunctions are held at once, each ``(nspin, nk, nocc, npwx)`` complex. On the
two-atom silicon of ``si-epsilon.in`` that is 2 MB; on a 16-atom cell with 100
k-points and 3000 plane waves it is 7 GB, and the way down is QE's -- solve one
irreducible representation at a time, which cuts the count as well as the
storage. **``atoms=`` is that lever in its simplest form**: it solves a chosen
subset's perturbations and nothing else, so the storage falls in proportion --
a 57-atom molecule on a 100-atom fixed slab is 171 perturbations rather than
471 -- and ``on_row`` streams each row of the force constants out as it is
assembled, so a run that outlives its job keeps what it paid for. The bare perturbations are stored rather than recomputed because the
self-consistent loop re-uses them at every iteration, which is the same trade
:mod:`defuqumat.response.efield` makes for its three.

**An ultrasoft or PAW run adds three more arrays of that same shape and two
smaller ones**, and they are all built once rather than per iteration: the
occupied block ``dpsi^ort`` (``3 nat`` of ``(nspin, nk, nocc, npwx)``, so the
state storage goes from two such sets to three), the mixed state's own change
``drhous`` (``3 nat`` dense grids, which is the *density* shape and an order
smaller), and ``becsumort`` plus ``dLambda``, which are ``nh^2`` and ``nbnd^2``
per mode and negligible beside either. Held together the ultrasoft peak is
about **1.5x** the norm-conserving one at the same cell -- 3 MB on
``si-epsilon-us.in``. PAW adds the one-centre response beside ``dvscf`` in the
mixer, ``3 nat`` times ``(nspin_mag, nkb, nkb)``, which is kilobytes. **An
ultrasoft or PAW metal doubles the bands inside the assembly**
(:func:`_force_constants`, ``notes/incompat/I86.md``): the ground state, the
tangent and the multipliers each go from ``nbnd`` to ``2 nbnd`` bands for the
``jvp`` of one mode, so the assembly's working set is twice the
undoubled one, while the response loop and the stored ``3 nat`` sets do not
change.

References for the method rather than the code: Baroni, de Gironcoli, Dal Corso
and Giannozzi, *Rev. Mod. Phys.* **73**, 515 (2001), whose Eq. (14) is the
identity above.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.forces.energy import FrozenState, frozen_energy
from defuqumat.response.efield import _streams, require_a_symmetrisable_response
from defuqumat.response.mixing import DEFAULT_RESPONSE_MIXING, ResponseMixer, ddv_scf
from defuqumat.response.sternheimer import (
    SternheimerSolver,
    paw_response,
    occupied_counts,
    pass_threshold,
    require_a_sternheimer_regime,
    smearing_of,
)
from defuqumat.response.velocity import over_kpoints
from defuqumat.batching import map_k
from defuqumat.scf.occupations import wgauss
from defuqumat.system.symmetry import (
    atom_mapping, cartesian_rotations, orbit_closure, subset_atom_mapping,
)
from defuqumat.units import AMU_TO_RY, RY_TO_CMM1, RY_TO_THZ
from defuqumat.eager import compiled_jvp

__all__ = ["Phonons", "DisplacementResponse", "dynamical_matrix",
           "self_consistent_response", "screening_loop",
           "orthogonality_states", "non_variational_response",
           "multiplier_response", "overlap_derivatives",
           "symmetrize_becsum_modes"]

#: QE's ``alpha_mix(1)``: the weight the mixer gives the residual. It is no
#: longer the *whole* of the mixing -- :mod:`defuqumat.response.mixing` builds an
#: Anderson history on top of it -- which is what makes 0.7 a safe default here
#: rather than a value each system has to be tuned to.
ALPHA_MIX = 0.7

#: The loop's stop, in the units of ``ph.x``'s ``tr2_ph``
#: (:data:`defuqumat.response.efield.TR2` says what they are). The test here is
#: the **largest single-mode** ``|ddv_scf|^2`` below ``tr2 / npol``, which is
#: ``ph.x``'s test for a one-dimensional irreducible representation and never
#: looser than its test for a larger one: this loop mixes every mode together
#: where ``ph.x`` mixes a representation at a time, and a test over the joint
#: vector would loosen the per-mode one by ``(3 nat)^2``. A subset run
#: (``atoms=``) tests over the modes it solves.
#:
#: **1e-18, four orders below ``ph.x``'s 1e-14**, the user's decision of
#: 2026-10-15 (``OPEN.md`` Part XXVII item 1, ``PLAN.md`` P149): the acoustic
#: residue is a 1e-4 fraction of the force constants and is the first number a
#: loop that stops early moves. On ``docs/figures/phonons_silicon_q/si.in`` it
#: reads 5.68 cm^-1 at 1e-14 and 4.326 at 1e-18, against 4.325 from a finite
#: difference of the energy, for 9 passes against 6; the optical mode moves by
#: 1.5e-3 cm^-1. **``tr2=1e-14`` is ``ph.x``'s stop** and is what to pass for a
#: like-for-like comparison with ``ph.x``; every P24 and P25 number before
#: 2026-10-15 was taken there. The CG schedule stays the bare one
#: (:func:`~defuqumat.response.sternheimer.pass_threshold`, uncapped), measured
#: to reach 1e-18 on every cell the phonon tests build (P149's note).
TR2 = 1.0e-18

MAX_ITERATIONS = 60

#: The CG threshold of every solve that is not scheduled: the bare
#: ``P_c r|psi>`` always, and the loop's when a caller asks for a fixed one.
#: It was the loop's too until 2026-10-03.
FIXED_THRESHOLD = 1.0e-12


@dataclass
class Phonons:
    """The force constants at ``Gamma``, and what they diagonalise to."""

    #: ``(nat, 3, nat, 3)`` in Ry/bohr^2 -- ``d^2E/du_(a i) du_(b j)``,
    #: hermitised and symmetrised over the crystal's point group. This is the
    #: matrix QE writes to ``fildyn``, in its own units and layout.
    matrix: np.ndarray
    #: ``(3 nat,)`` in cm^-1, signed: an imaginary frequency is reported as a
    #: negative number, exactly as ``dyndia`` prints one.
    frequencies: np.ndarray
    #: ``(3 nat, 3 nat)`` displacement patterns, columns matching
    #: :attr:`frequencies`, in the mass-weighted basis ``eigh`` returns.
    eigenvectors: np.ndarray
    #: ``(nat, 3, nspin_mag, n1, n2, n3)``: the induced charge density per unit
    #: displacement, symmetrised, on the dense grid. Summed over the atoms it is
    #: the ground-state density's own gradient, which is what
    #: :func:`self_consistent_response` checks.
    induced_density: np.ndarray | None = None
    #: ``max|D - D^T|`` after the group average and **before** the
    #: hermitisation, in Ry/bohr^2. The force constants are symmetric exactly,
    #: so this is a measurement and not a construction: it is the free
    #: diagnostic of whether the linear solves converged and whether the
    #: response half of the derivative is right, and nothing else here shows
    #: either. Measured *before* the group average it would show the wedge
    #: residue instead, which is large and says nothing.
    asymmetry: float = 0.0
    #: ``|ddv_scf|^2`` at each iteration of the self-consistent loop.
    history: list = field(default_factory=list)
    #: Mean CG iterations per band per solve, QE's ``av.it.``.
    average_iterations: float = 0.0
    converged: bool = False
    #: Which atoms were displaced, or ``None`` for the whole cell. When it is a
    #: tuple, :attr:`matrix` and :attr:`frequencies` are that subset's block
    #: with every other atom held fixed -- so the modes are the frozen-substrate
    #: ones and there is no acoustic triplet among them to look for.
    atoms: tuple | None = None
    #: What :func:`~defuqumat.response.phononq.dynamical_matrix_at_q` keeps when
    #: asked to (``keep_internals``): the solver, the bare perturbations and the
    #: converged ``dV_scf``, which the electron-phonon matrix element is built
    #: from (:mod:`defuqumat.response.elph`). ``None`` otherwise.
    internals: dict | None = field(default=None, repr=False)

    @property
    def omega2(self) -> np.ndarray:
        """``(3 nat,)`` eigenvalues of the mass-weighted matrix, in Ry^2."""
        signs = np.sign(self.frequencies)
        return signs * (self.frequencies / RY_TO_CMM1) ** 2

    @property
    def frequencies_thz(self) -> np.ndarray:
        """:attr:`frequencies` in THz, the other unit ``dyndia`` prints."""
        return self.frequencies / RY_TO_CMM1 * RY_TO_THZ

    @property
    def acoustic_residue(self) -> float:
        """The largest of the three lowest frequencies, in cm^-1.

        Translating the whole crystal costs no energy, so three frequencies are
        zero exactly and what comes out instead measures the calculation --
        chiefly the finite plane-wave basis, whose energy depends on where the
        atoms sit relative to the FFT grid. ``ph.x`` prints 2.045258 cm^-1 for
        the silicon of ``si-epsilon.in`` and does **not** impose the sum rule,
        so this number is a target rather than a nuisance: reproducing it means
        the same basis-set error is being made, which is a sharper statement
        than reproducing the optical mode.
        """
        return float(np.max(np.abs(self.frequencies[:3])))


@dataclass
class DisplacementResponse:
    """``solve_linter``'s output: the ``3 nat`` first-order states and densities.

    A carrier, so that a caller which has already solved these does not solve
    them again -- which is the expensive half of both a dynamical matrix and a
    Raman tensor, and they are the *same* half.
    :class:`~defuqumat.response.strain.StrainResponse` is the same idea for the
    strain perturbation.
    """

    #: ``(nat, 3, nspin, nk, nocc, ndim)`` -- ``dpsi`` per mode.
    dpsi: object
    #: ``(nat, 3, nspin_mag, n1, n2, n3)`` -- the induced density per mode.
    drho: object
    #: ``|ddv_scf|^2`` per iteration.
    history: list
    converged: bool = True
    #: The bare perturbations, kept only because an ultrasoft or PAW assembly
    #: needs to rebuild ``dLambda`` from the converged perturbation and cannot
    #: do that from ``dpsi`` alone. ``None`` on the norm-conserving path, where
    #: the term it feeds is zero.
    bare: object = None
    #: ``{"dvscf", "onecentre", "dbecsum"}`` at convergence, for the same
    #: reason.
    extras: dict | None = None
    #: The :class:`~defuqumat.response.chunked_phonon.StreamedDisplacements` that
    #: solved it, when the solve walked the k axis: ``dpsi`` is then its host
    #: store, and :func:`dynamical_matrix` assembles from it a chunk at a time
    #: rather than putting the stores on the device whole.
    walked: object = None


def dynamical_matrix(
    calculation,
    wavefunctions,
    eigenvalues,
    density,
    becsum=(),
    alpha_mix: float = ALPHA_MIX,
    tr2: float = TR2,
    max_iterations: int = MAX_ITERATIONS,
    threshold: float | None = None,
    acoustic_sum_rule: bool = False,
    response: "DisplacementResponse | None" = None,
    atoms=None,
    on_row=None,
    verbose: bool = False,
) -> Phonons:
    """The ``Gamma``-point force constants and phonon frequencies.

    Args:
        calculation: the :class:`~defuqumat.scf.driver.Calculation` the states
            belong to, with its own symmetry-reduced k-set. A ``nosym`` run is
            accepted only on an **unshifted** grid, for the reason
            :func:`~defuqumat.response.efield.require_a_symmetrisable_response`
            gives.
        wavefunctions: ``(nspin, nk, nbnd, ndim)`` from the converged run.
        eigenvalues: ``(nspin, nk, nbnd)`` or the squeezed ``(nk, nbnd)``.
        density: the converged density the fixed potential is built from.
        becsum: unused for a norm-conserving dataset, which is the only kind
            this accepts; present so the signature matches
            :func:`~defuqumat.response.efield.dielectric_tensor`.
        threshold: the CG threshold of the linear solves. ``None``, the
            default, is ``ph.x``'s schedule (:func:`~defuqumat.response.
            sternheimer.pass_threshold`); a number holds that threshold on every
            pass (:func:`~defuqumat.response.efield.dielectric_tensor` says when
            that is wanted).
        tr2: the loop's stop on ``|ddv_scf|^2``, in the units of ``ph.x``'s
            ``tr2_ph``. The default, :data:`TR2` (1e-18), converges the acoustic
            residue; ``tr2=1e-14`` is ``ph.x``'s own stop, for a like-for-like
            comparison with it.
        acoustic_sum_rule: whether to impose ``sum_b D_(a i)(b j) = 0`` before
            diagonalising. **Off by default, because ``ph.x`` does not impose
            it** and the residue is the diagnostic described in
            :attr:`Phonons.acoustic_residue`.
        atoms: solve only these atoms' displacements, as indices into the
            structure. The cost of this function is ``3 nat`` perturbations held
            at once -- see the module docstring -- so a molecule on a fixed slab
            is the case this exists for: 171 perturbations rather than 471, and
            the memory in proportion. What comes back is then the subset's own
            block, ``(len(atoms), 3, len(atoms), 3)``, diagonalised with the
            subset's masses, which is the frozen-substrate approximation and is
            what holding the other atoms fixed means.

            **Ultrasoft and PAW work**, which took subsetting the four tangents
            P39 adds -- the overlap derivatives, the occupied block ``ort``, the
            mixed state's own change and the multiplier response. Two of those
            needed no change at all once the first two were done, because they
            take their shape from what they are handed (``multiplier_response``
            from ``bare``, ``orthogonality_states`` from ``derivatives``) rather
            than from ``nat``, and that is the reason to prefer such a shape.
            An ultrasoft or PAW **metal**'s subset is refused, because only its
            whole cell is measured (:func:`_require_a_measured_overlap_metal`).

            **With symmetry the subset must be a union of orbits**, because
            ``symdvscf`` symmetrises the responses *together* -- an operation
            rotates the direction and permutes the atom, so for any other subset
            it mixes a mode inside with one outside, and the refusal names the
            smallest closed subset that contains the one asked for. A closed
            subset is symmetrised with the group restricted to it, and its block
            is the whole cell's block of the same calculation.

            The acoustic sum rule is unavailable for the same reason the columns
            outside the subset are not solved -- it is a sum over *all* atoms --
            and it is not wanted here anyway: the modes a fixed substrate leaves
            are frustrated translations, which are physical rather than an
            artefact to be projected out.
        on_row: called as ``on_row(atom, cart, row)`` as each row of the force
            constants is assembled, so a long run can persist what it has paid
            for instead of holding it to the end.
        response: a :class:`DisplacementResponse` solved earlier, if there is
            one. It is steps 1 and 2 below and the whole cost of this function;
            :func:`~defuqumat.response.nonlinear.raman_tensors` solves the same
            object and hands it back through its ``keep_internals``, which is
            what lets a spectrum cost one displacement response rather than two.
            With ``atoms``, a response solved for every atom (which both of
            those are) is cut to those atoms' rows -- :func:`_response_rows`
            on the whole-k route, ``force_constants(atoms=)`` on the walked
            one -- and gives the block of the whole matrix exactly; a response
            whose row count is neither the cell's nor the subset's is refused.
    """
    eigenvalues = jnp.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    require_a_symmetrisable_response(calculation)
    # ``metals = True``: a Fermi surface does not stop a dynamical matrix
    # existing, and the solve handles one (``PLAN.md`` P24c). What it adds is
    # ``ef_shift``, inside the loop below.
    # Before the generic guard, so that the message names the dynamical matrix.
    _require_no_spinor(calculation)
    # ``gamma_ok``: validated against an explicit k = 0 on the whole sphere,
    # which is the same calculation in the other storage (``PLAN.md`` P68a).
    # ``spin_polarized``: the whole-k assembly is checked against ``ph.x`` on
    # triplet O2 (:func:`occupied_pairs`); the walked one is refused below.
    # ``noncollinear``: the spinor assembly is written and measured, and
    # :func:`_require_no_spinor` is what refuses it, with the numbers.
    require_a_sternheimer_regime(calculation, metals=True, gamma_ok=True,
                                 spin_polarized=True, noncollinear=True)
    # The ultrasoft or PAW *metal* is implemented on the whole-k route
    # (``notes/incompat/I86.md``); what is still refused of it is decided here,
    # and the walked route refuses it below.
    _require_a_measured_overlap_metal(calculation, atoms)

    structure = calculation.system.structure
    positions = jnp.asarray(structure.positions)
    chosen = _require_a_subsettable_regime(calculation, atoms, structure.nat)
    if atoms is not None and acoustic_sum_rule:
        raise ValueError(
            "the acoustic sum rule is a sum over every atom of the cell and "
            "cannot be imposed on a subset's block. It is also not what a "
            "frozen substrate wants: the modes left are frustrated "
            "translations, which are physical"
        )

    walked = None if response is None else getattr(response, "walked", None)
    if walked is not None:
        _require_one_spin_channel(calculation)
        _require_no_walked_overlap_metal(calculation)
        if chosen is not None and calculation.use_symmetry:
            # I51 admits a closed subset with symmetry on the whole-k route
            # only; I54's row map on the walked route was measured on ``nosym``
            # cells. The two meet here, unmeasured, so the walked subset keeps
            # the streamed route's refusal (``notes/INCOMPATIBILITIES-T2.md``).
            raise NotImplementedError(_SYMMETRISED_SUBSET_WALKED)
        # **A response solved a k-chunk at a time is assembled the same way**,
        # from its own stores (:meth:`~defuqumat.response.chunked_phonon.
        # StreamedDisplacements.force_constants`): the states it was solved at
        # are the ones it holds, so ``wavefunctions`` is not read. It solved
        # every atom, and a subset is its rows for those atoms: the subset
        # is on a ``nosym`` run (refused otherwise, just above), where nothing
        # couples the modes, so
        # the rows of the whole response *are* the subset's response, and each
        # row of the assembly reads only its own (``force_constants(atoms=)``).
        matrix = walked.force_constants(response.drho, response.extras,
                                        on_row=on_row, atoms=chosen)
        drho, history = response.drho, response.history
        if chosen is not None:
            drho = drho[np.asarray([walked.atoms.index(a) for a in chosen])]
        average_iterations, converged = float("nan"), response.converged
        return _phonons_of(calculation, matrix, drho, history, average_iterations,
                           converged, chosen, acoustic_sum_rule)

    # **The route is decided before anything converts the states**, by the
    # field response's rule (:func:`~defuqumat.response.efield._streams`): a
    # streamed store, memory mode with a chunk smaller than the mesh, or memory
    # mode on a card, walk the k axis a chunk at a time with every
    # per-perturbation array in host memory (:mod:`defuqumat.response.
    # chunked_phonon`). A response handed in was solved whole and is assembled
    # whole. Before this route existed, a streamed store did not run at all:
    # the solver keeps it in host memory and the whole-k helpers below index it
    # with a traced k index.
    streamed = response is None and _streams(
        calculation, wavefunctions, keep_internals=False,
        what="the dynamical matrix")
    if streamed:
        _require_one_spin_channel(calculation)
        _require_no_walked_overlap_metal(calculation)
    if streamed and chosen is not None and calculation.use_symmetry:
        # The closed-subset symmetrisation is threaded through
        # :class:`~defuqumat.response.chunked_phonon.StreamedDisplacements` but
        # was measured only on the whole-k route (``notes/incompat/I51.md``).
        raise NotImplementedError(_SYMMETRISED_SUBSET_WALKED)
    if streamed and calculation.noncolin:
        # **The walked route has a collinear copy of two of the terms** this
        # module writes for a spinor: :mod:`defuqumat.response.chunked_phonon`'s
        # ``overlap_matrix`` contracts the scalar ``qq`` on an ``npwx``-long
        # band, and its frozen energies do not ask for ``spinors=True``.
        raise NotImplementedError(
            "the dynamical matrix of a noncollinear or spin-orbit run walked a "
            "k-chunk at a time is not implemented: the chunked route "
            "(response/chunked_phonon.py) has the collinear overlap qq and no "
            "spinor energy. The whole-k route is implemented; run with "
            "memory_mode='speed' or a k_batch that holds the whole mesh"
        )
    weights, _ = calculation.occupations(eigenvalues)
    weights = jnp.asarray(weights)
    nocc = occupied_counts(calculation)
    potential = calculation.potential(density)
    _, ddd_paw = calculation.onecenter(becsum)
    hamiltonians = calculation.hamiltonian(potential.v_scf, ddd_paw)
    if not streamed:
        wavefunctions = jnp.asarray(wavefunctions)
    solver = SternheimerSolver(
        calculation, hamiltonians, wavefunctions, eigenvalues, weights,
        nocc, FIXED_THRESHOLD if threshold is None else threshold,
        v_scf=potential.v_scf, becsum=becsum,
        smearing=smearing_of(calculation, _fermi_level(calculation, eigenvalues)),
        kpoint_weights=calculation.system.kpoints.weights,
    )
    solver.schedule = threshold is None

    if streamed:
        from defuqumat.response.chunked_phonon import StreamedDisplacements

        displacements = StreamedDisplacements(
            calculation, solver, potential.v_scf, positions,
            tuple(range(structure.nat)) if chosen is None else chosen,
            wavefunctions, eigenvalues, weights, density, becsum,
        )
        # Steps 0 and 1 below, a chunk at a time.
        displacements.prepare()
        dpsi, drho, history, average_iterations, converged, extras = (
            screening_loop(
                calculation, displacements, density, positions=positions,
                alpha_mix=alpha_mix, tr2=tr2, max_iterations=max_iterations,
                becsumort=displacements.becsumort, drhous=displacements.drhous,
                atoms=chosen, verbose=verbose,
            )
        )
        # Steps 2b and 3.
        matrix = displacements.force_constants(drho, extras, on_row=on_row)
    else:
        matrix, drho, history, average_iterations, converged = _whole_k(
            calculation, solver, positions, wavefunctions, eigenvalues, weights,
            density, becsum, potential, chosen, response, alpha_mix, tr2,
            max_iterations, verbose, on_row,
        )

    return _phonons_of(calculation, matrix, drho, history, average_iterations,
                       converged, chosen, acoustic_sum_rule)


def _phonons_of(calculation, matrix, drho, history, average_iterations,
                converged, chosen, acoustic_sum_rule) -> "Phonons":
    """The force constants averaged, hermitised and diagonalised, for any route."""
    structure = calculation.system.structure
    # ``symdynph_gq`` first and the hermitisation second, which is the order
    # that makes the second one a *measurement*. A column of the raw matrix is a
    # sum over the irreducible wedge, and such a sum is not symmetric in
    # ``(a i) <-> (b j)`` until the group has put back what the reduction left
    # out -- on the ten-point wedge of ``si-epsilon.in`` the raw asymmetry is
    # 5.1e-2 against force constants of 0.28, and essentially all of it is that.
    # After the average it is 2e-16 on every case here, shifted wedge included,
    # so the hermitisation has nothing left to do and what it would have removed
    # is a report on the linear solves. Whether the average *must* leave a
    # symmetric matrix is not claimed -- it is measured.
    if chosen is None:
        matrix = calculation.symmetrize_atom_pair_tensor(matrix)
        masses = structure.masses
    else:
        # **The subset's own block, and it is square by construction.** A row is
        # the gradient against every atom, so the columns outside the subset are
        # computed and then dropped: they are the force the molecule's
        # displacement exerts on the frozen substrate, which is a real quantity
        # and not one a frozen-substrate mode can respond to. Then
        # ``symdynph_gq`` on the block: the guard established that the subset is
        # a union of orbits, so the group average of an entry inside it reads
        # only entries inside it, and is the whole matrix's average restricted to
        # the block exactly. Under ``nosym`` it is the identity.
        matrix = matrix[:, :, list(chosen), :]
        matrix = calculation.symmetrize_atom_pair_tensor(matrix, atoms=chosen)
        masses = np.asarray(structure.masses)[list(chosen)]
    asymmetry = float(np.abs(matrix - matrix.transpose(2, 3, 0, 1)).max())
    matrix = 0.5 * (matrix + matrix.transpose(2, 3, 0, 1))
    if acoustic_sum_rule:
        matrix = _impose_acoustic_sum_rule(matrix)

    frequencies, vectors = _diagonalize(matrix, masses)
    return Phonons(
        matrix=np.asarray(matrix),
        frequencies=frequencies,
        eigenvectors=vectors,
        induced_density=np.asarray(drho),
        asymmetry=asymmetry,
        history=history,
        average_iterations=average_iterations,
        converged=converged,
        atoms=chosen,
    )


def _whole_k(calculation, solver, positions, wavefunctions, eigenvalues,
             weights, density, becsum, potential, chosen, response, alpha_mix,
             tr2, max_iterations, verbose, on_row):
    """Steps 0 to 3 of :func:`dynamical_matrix` with every k-point in one array."""
    nbnd = wavefunctions.shape[2]
    # 0. What ``S`` moving with the atoms adds, and all of it is zero for a
    #    norm-conserving dataset: the occupied block of the first-order state,
    #    and the mixed state's own change at frozen states (``drho.f90``).
    derivatives = overlap_derivatives(calculation, solver, positions, chosen)
    ort = orthogonality_states(calculation, solver, positions, chosen)
    # **The occupied block is a tangent of the *stored* states and has to be as
    # wide as they are.** ``solver.psi`` is truncated to ``max(occupied_counts)``
    # -- 7 for triplet O2, whose channels hold 7 and 5 -- while
    # ``wavefunctions`` carries all ``nbnd`` of them, and a ``jvp`` wants the two
    # the same shape. Every cell this module was validated on had
    # ``nbnd == nocc``, so the pad is a no-op there and the mismatch only
    # appears once a run asks for empty bands or fills its two channels to
    # different depths.
    ort = _pad_to_bands(ort, nbnd)
    (rho_moved, bec_moved), (rho_ort, bec_ort) = non_variational_response(
        calculation, positions, wavefunctions, weights,
        jnp.asarray(density), becsum, ort, chosen,
        state_weights=_state_weights(solver, weights),
    )
    drhous_stacked = moved_stacked = becsumort = bec_moved_sym = None
    if rho_moved is not None:
        moved_stacked = _stack_modes(
            _symmetrize_modes(calculation, rho_moved, chosen))
        drhous_stacked = moved_stacked + _stack_modes(
            _symmetrize_modes(calculation, rho_ort, chosen)
        )
        bec_moved_sym = symmetrize_becsum_modes(calculation, bec_moved, chosen)
        becsumort = _add_becsum(
            bec_moved_sym, symmetrize_becsum_modes(calculation, bec_ort, chosen)
        )

    if response is None:
        # 1. The bare perturbation ``(dH/du - eps dS/du)|psi>``, once per mode
        #    and stored: the loop below drives on it at every iteration.
        bare = _bare_displacements(
            calculation, solver, potential.v_scf, positions, chosen
        )

        # 2. ``solve_linter``'s loop, one perturbation per (atom, direction).
        dpsi, drho, history, average_iterations, converged, extras = (
            self_consistent_response(
                calculation, solver, bare, density, positions=positions,
                alpha_mix=alpha_mix, tr2=tr2, max_iterations=max_iterations,
                becsumort=becsumort, drhous=drhous_stacked, atoms=chosen,
                verbose=verbose,
            )
        )
    else:
        dpsi, drho, bare, extras = _response_rows(
            response, chosen, calculation.system.structure.nat)
        history, converged = response.history, response.converged
        average_iterations = float("nan")

    # 2b. The three tangents ``S``'s motion adds to the assembly. ``dLambda``
    #     is a matrix element of the *same* perturbation the last solve was
    #     driven by, rebuilt at the converged ``dV_scf`` --
    #     :func:`~defuqumat.response.born._multiplier_response`'s argument, with
    #     a displacement in place of the field.
    multipliers = dbecsum = None
    if calculation.is_ultrasoft and bare is not None:
        multipliers = multiplier_response(
            calculation, solver, bare, extras, weights, nbnd, derivatives,
        )
        # The assembly rebuilds the raw mixed-state response itself, so what
        # it is handed is the *symmetrised* total; the difference between the
        # two is the wedge sum's own correction and is zero on a closed grid.
        dbecsum = extras.get("dbecsum")

    # 3. The second derivative, two jvp of the force's own gradient per mode:
    #    the frozen Hessian at ``wg`` and the electronic response at ``wk``.
    matrix = _force_constants(
        calculation, positions, wavefunctions, weights,
        _state_weights(solver, weights), eigenvalues, jnp.asarray(density),
        dpsi, drho, solver.nocc, becsum=becsum, dbecsum=dbecsum,
        multipliers=multipliers, ort=ort, atoms=chosen, on_row=on_row,
    )
    return matrix, drho, history, average_iterations, converged


def _response_rows(response, chosen, nat):
    """``(dpsi, drho, bare, extras)`` of a handed-in response, at ``chosen``'s rows.

    Every per-mode store of a :class:`DisplacementResponse` leads with the
    perturbed atom, in the order it was solved, and the assembly below reads
    row ``r`` of each as the response to ``chosen[r]``. A response solved for
    every atom (what :func:`~defuqumat.response.nonlinear.raman_tensors` keeps)
    handed in with a subset therefore has to be cut to the subset's rows first:
    read whole, row 0 is atom 0's response whichever atom ``chosen[0]`` is, and
    the matrix comes out wrong with no error. A response with exactly
    ``len(chosen)`` rows (fewer than ``nat``) is taken to be the subset's own,
    as before; any other row count cannot be matched to the atoms and is
    refused.
    """
    held = int(np.shape(response.drho)[0])
    wanted = nat if chosen is None else len(chosen)
    if chosen is None or (held == wanted and held != nat):
        if held != wanted:
            raise ValueError(
                f"the response holds {held} atoms' perturbations and the "
                f"matrix asked for needs {wanted}")
        return response.dpsi, response.drho, response.bare, response.extras
    if held != nat:
        raise ValueError(
            f"the response holds {held} atoms' perturbations, neither the cell's "
            f"{nat} nor the {len(chosen)} asked for, so its rows cannot be "
            "matched to atoms")
    # An index array, not a list: a jax array refuses a list as an index.
    rows = np.asarray(chosen)

    def cut(stack):
        return None if stack is None else stack[rows]

    extras = None if response.extras is None else {
        key: cut(value) for key, value in response.extras.items()}
    return (cut(response.dpsi), cut(response.drho), cut(response.bare),
            extras)


class _Levels:
    """The one field :func:`~defuqumat.response.sternheimer.smearing_of` reads.

    :func:`dynamical_matrix` is handed a density and eigenvalues rather than an
    ``SCFResult``, so the Fermi level is recomputed from the occupations the
    calculation itself would assign -- the same call the SCF made, on the same
    eigenvalues, so it is the same number.
    """

    def __init__(self, fermi_energy):
        self.fermi_energy = fermi_energy


def _fermi_level(calculation, eigenvalues):
    if calculation.system.occupations == "fixed":
        return _Levels(None)
    _, levels = calculation.occupations(eigenvalues)
    return _Levels(levels["fermi_energy"])


def self_consistent_response(
    calculation,
    solver,
    bare,
    density,
    positions=None,
    alpha_mix: float = ALPHA_MIX,
    tr2: float = TR2,
    max_iterations: int = MAX_ITERATIONS,
    mixing_mode: str = DEFAULT_RESPONSE_MIXING,
    becsumort=None,
    drhous=None,
    atoms=None,
    verbose: bool = False,
):
    """``solve_linter``'s loop for the ``3 nat`` displacement patterns.

    Split out from :func:`dynamical_matrix` because the first-order
    wavefunctions have a check the assembled matrix does not offer: **displace
    every atom by the same vector and the response must be the ground-state
    density's own gradient**, ``sum_a drho_(a i) = -d(rho)/dr_i``, because a
    rigid translation of the crystal *is* a translation of its self-consistent
    solution. The identity holds only for the fully screened response -- the
    bare one is 52% off on silicon -- so it tests the loop, the kernel and the
    symmetrisation together against a quantity obtained by differentiating the
    converged density on the grid, which shares no machinery with any of them.
    It is the phonon's counterpart of the ``chi_0`` finite-difference check
    :mod:`defuqumat.response.sternheimer` rests on.

    **The modes are independent.** Each perturbation screens only itself, so
    nothing in the physics couples them; the only thing that mixes them is the
    symmetrisation, which is why they are iterated together rather than one
    after another.

    **A metal goes through this loop unchanged**, and that was true one phase
    before the matrix above could consume the result: P24c put ``ef_shift`` and
    ``orthogonalize``'s smearing branch here and the assembly still counted the
    occupation twice, so the loop was kept metallic while
    ``dynamical_matrix`` refused one at the door. That is what let the refusal
    be *measured* against ``ph.x`` rather than asserted, and lifting it (P28)
    changed nothing in this function: the fix was the weight the **matrix**
    contracts ``dpsi`` with, not anything the loop produces
    (:func:`_state_weights`).

    Returns ``(dpsi, drho, history, average_iterations, converged)``, with
    ``dpsi`` an object array of shape ``(nat, 3)``.
    """
    nat = calculation.system.structure.nat
    chosen = tuple(range(nat)) if atoms is None else tuple(atoms)
    return screening_loop(
        calculation, _WholeDisplacements(solver, bare, len(chosen)), density,
        positions=positions, alpha_mix=alpha_mix, tr2=tr2,
        max_iterations=max_iterations, mixing_mode=mixing_mode,
        becsumort=becsumort, drhous=drhous, atoms=atoms, verbose=verbose,
    )


class _WholeDisplacements:
    """The ``3 nat`` solves with every k-point in one array: the route as it was.

    :class:`~defuqumat.response.chunked_phonon.StreamedDisplacements` is the other
    one, with the same four methods; :func:`screening_loop` drives either, so the
    self-consistent loop around the solves is one loop for both routes.
    """

    def __init__(self, solver, bare, rows: int):
        self.solver = solver
        self.bare = bare
        self.rows = rows
        self.dpsi = np.empty((rows, 3), dtype=object)
        self.iterations = 0
        self.solves = 0

    def respond(self, dvscf, onecentre, include_induced: bool, threshold=None):
        """One iteration's solves: the response density per mode, and for PAW
        the raw ``becsum`` response the one-centre potential is built from.

        Each solve starts from the previous pass's ``dpsi``, zeros on the first
        (``iudwf``); ``threshold`` is this pass's CG threshold. The store is the
        unshifted solution inside the loop: :meth:`shift_states` runs once after
        it and rewrites the store in place, so an object used for one loop is
        not reused for another after it.
        """
        solver = self.solver
        response, becsum_response = [], []
        for row in range(self.rows):
            for cart in range(3):
                perturbation = _bare_plus_induced(
                    solver, self.bare[row, cart], dvscf[row, cart],
                    include_induced,
                    None if onecentre is None else onecentre[row, cart],
                )
                previous = self.dpsi[row, cart]
                start = (jnp.zeros_like(self.bare[row, cart]) if previous is None
                         else previous)
                solution = solver.solve(perturbation, start=start,
                                        threshold=threshold)
                self.dpsi[row, cart] = solution.dpsi
                self.iterations += solution.iterations
                self.solves += 1
                response.append(solver.response_density(solution.dpsi))
                if onecentre is not None:
                    becsum_response.append(solver.response_becsum(solution.dpsi))
        return response, becsum_response

    def fermi_level_shift(self, drho):
        return self.solver.fermi_level_shift(drho)

    def shift_states(self, shifts) -> None:
        """``ef_shift_wfc`` on every mode's ``dpsi``, ``shifts`` in mode order."""
        for index, (row, cart) in enumerate(
            (r, c) for r in range(self.rows) for c in range(3)
        ):
            self.dpsi[row, cart] = self.solver.fermi_level_shift_states(
                self.dpsi[row, cart], shifts[index]
            )


def screening_loop(
    calculation,
    displacements,
    density,
    positions=None,
    alpha_mix: float = ALPHA_MIX,
    tr2: float = TR2,
    max_iterations: int = MAX_ITERATIONS,
    mixing_mode: str = DEFAULT_RESPONSE_MIXING,
    becsumort=None,
    drhous=None,
    atoms=None,
    verbose: bool = False,
):
    """:func:`self_consistent_response`'s loop around a set of displacement solves.

    ``displacements`` holds the bare perturbations and does the solves --
    :class:`_WholeDisplacements` with every k-point in one array, or
    :class:`~defuqumat.response.chunked_phonon.StreamedDisplacements` a k-chunk at
    a time -- and everything here acts on whole-grid objects with no k index:
    the Fermi level's shift, the symmetrisation, the screening kernel and the
    mixing. Returns :func:`self_consistent_response`'s tuple, with ``dpsi`` the
    object's own store.
    """
    solver = displacements.solver
    nat = calculation.system.structure.nat
    # ``rows`` is how many perturbations are being solved and ``chosen`` which
    # atoms they belong to. They are the same thing only for a whole-cell run;
    # with symmetry a subset is accepted only when it is a union of orbits,
    # because an operation *mixes* the modes and the symmetrisation below can
    # act on a partial stack only when it never leaves it (see
    # :func:`_require_a_subsettable_regime`).
    chosen = tuple(range(nat)) if atoms is None else tuple(atoms)
    rows = len(chosen)
    grid_shape = tuple(np.shape(density))
    # **The ``3 nat`` grids in host memory**, one mode at a time on the device,
    # where the displacements ask for it: the k-chunked route on a run with no
    # symmetry, where ``symmetrize_atom_displacement`` is the identity and so
    # nothing below needs the modes together. ``dvscf``, the response, the
    # induced potential, ``drhous`` and the core term are each ``P`` dense grids
    # -- six to seven of them at once at an iteration's peak, which on a
    # 57-atom molecule over a 6M-point grid is about 40 GB -- and every
    # operation on them here is per mode except the mixer, whose history is in
    # host memory already.
    host = bool(getattr(displacements, "host_fields", False))
    core = _core_charge_response(calculation, density, positions, chosen,
                                 host=host)
    if host:
        dvscf = np.zeros((rows, 3) + grid_shape)
        symmetrised = np.zeros_like(dvscf)
    else:
        dvscf = jnp.zeros((rows, 3) + grid_shape)
        symmetrised = jnp.zeros_like(dvscf)
    history = []
    converged = False
    # PAW's one-centre coefficients respond too, and they are *not* a function
    # of the density -- they come from ``becsum``. Carried and mixed beside
    # ``dvscf`` exactly as :mod:`defuqumat.response.efield` carries its three,
    # which is ``dfpt_kernels``' ``int3_paw`` beside ``dvscfin``.
    onecentre = None if solver.ddd_paw is None else jnp.zeros(
        (rows, 3) + solver.ddd_paw.shape
    )

    mixer = ResponseMixer(mixing_mode, beta=alpha_mix)
    fermi_becsum = None
    for iteration in range(max_iterations):
        response, becsum_response = displacements.respond(
            dvscf, onecentre, iteration > 0,
            threshold=pass_threshold(solver, history),
        )

        # ``ef_shift``: a displacement at ``q = 0`` moves charge in and out of
        # the cell, so a metal's Fermi level moves with it and the response
        # density has to be corrected by ``def ldos`` before it screens
        # anything. Applied to the raw per-mode densities and *before* the
        # symmetrisation, which is where QE applies it too -- ``ldos`` is a
        # scalar under the group, so symmetrising ``def ldos`` as part of the
        # displacement-labelled vector field is ``sym_def`` by another route.
        shifts = None
        if solver.smearing is not None and drhous is not None and not host:
            # **The count includes ``drhous``**, as ``solve_linter`` counts it:
            # ``ef_shift`` there acts on ``drhoscfh`` after ``addusddens`` has
            # added the orthogonality part. Its two halves integrate to zero
            # together in exact arithmetic (``Tr[S' gamma]`` against
            # ``-sum wk W_nn S'_nn``), so this is QE's order rather than a term.
            flat = drhous.reshape((rows * 3,) + grid_shape)
            corrected = [displacements.fermi_level_shift(r + extra)
                         for r, extra in zip(response, flat)]
            response = [r - extra for (r, _), extra in zip(corrected, flat)]
            shifts = [float(d) for _, d in corrected]
        elif solver.smearing is not None:
            corrected = [displacements.fermi_level_shift(r) for r in response]
            response = [r for r, _ in corrected]
            shifts = [float(d) for _, d in corrected]
        if shifts is not None and becsum_response:
            # ``ef_shift``'s PAW branch: the level's motion fills the Fermi
            # surface's projector occupations too, ``dbecsum += def becsum1``,
            # and the one-centre potential responds to them. ``becsum1`` is the
            # raw ``becsum`` at the smeared delta's weight -- the same builder
            # :meth:`~defuqumat.response.sternheimer.SternheimerSolver.
            # local_density_of_states` augments its ``ldos`` with -- and it is
            # what :meth:`~defuqumat.response.sternheimer.SternheimerSolver.
            # fermi_level_shift_states` will put into ``dpsi`` at the end.
            if fermi_becsum is None:
                fermi_becsum = solver._raw_becsum(solver.psi,
                                                  solver._delta_weights())
            becsum_response = [
                tuple(None if b is None else b + shift * f
                      for b, f in zip(parts, fermi_becsum))
                for parts, shift in zip(becsum_response, shifts)
            ]

        # ``symdvscf``: the 3 nat responses are symmetrised *together*, after
        # the loop over modes and before the kernel, because an operation mixes
        # them -- rotating the direction and permuting the atom.
        stack = np.stack if host else jnp.stack
        stacked = stack(response).reshape((rows, 3) + grid_shape)
        if drhous is not None:
            # The first-order density the potential responds to is the *whole*
            # of it, and for an ultrasoft or PAW dataset the variational part is
            # not the whole: the augmentation charge moves with its atom and the
            # occupied block of ``dpsi`` is not zero, so ``drho.f90``'s
            # "change at fixed wavefunctions" screens beside it.
            stacked = stacked + drhous
        # In host mode the run has no symmetry and the average is the identity
        # (``host_fields`` is set only then), so it is not called on a stack the
        # device would have to hold.
        symmetrised = (stacked if host
                       else calculation.symmetrize_atom_displacement(
                           stacked, atoms=atoms))

        # ``dv_of_drho``: one jvp of the potential this code already writes.
        def screened(field):
            out = compiled_jvp(
                lambda r: calculation.potential(r).v_scf,
                (jnp.asarray(density),),
                (jax.device_put(field) if host else field,),
            )[1]
            return np.asarray(out) if host else out

        induced = stack([
            screened(symmetrised[row, cart])
            for row in range(rows) for cart in range(3)
        ]).reshape(dvscf.shape)

        induced_onecentre = None
        if onecentre is not None:
            # ``dfpt_kernels``: for a *phonon* the one-centre potential
            # responds to ``2 dbecsum + becsumort`` and not to the variational
            # part alone -- the orthogonality correction changes the one-centre
            # occupations, and PAW's energy is a second, independent functional
            # of them. (The grid density is the other way round: ``drhop``
            # there is the variational response only, and ``drhous`` is carried
            # separately into the assembly.) The factor of two in QE's line is
            # a storage convention rather than a term: ``addusdbec``
            # accumulates one of the two cross terms and
            # :meth:`SternheimerSolver.response_becsum` -- a ``jvp``, so
            # ``2 Re`` -- accumulates both.
            per_mode = np.empty((rows, 3), dtype=object)
            index = 0
            for row in range(rows):
                for cart in range(3):
                    per_mode[row, cart] = becsum_response[index]
                    index += 1
            if becsumort is not None:
                per_mode = _add_becsum(per_mode, becsumort)
            per_mode = symmetrize_becsum_modes(calculation, per_mode, atoms)
            induced_onecentre = jnp.stack([
                paw_response(calculation, per_mode[row, cart], solver.becsum)
                for row in range(rows) for cart in range(3)
            ]).reshape(onecentre.shape)

        if core is not None:
            # ``addcore``/``drhoc``: the core charge travels with its atom, so
            # ``v_xc`` changes even at a frozen valence density. It is
            # independent of the iteration -- it is not a response to anything
            # -- so it is built once and added to the induced potential here.
            induced = induced + core

        # In host mode a numpy sum, whose order differs from the device's, so
        # the history agrees with the device route's to round-off rather than to
        # the bit.
        difference = induced - dvscf
        onecentre_difference = (None if onecentre is None
                                else induced_onecentre - onecentre)
        change = ddv_scf(
            [difference[row, cart] for row in range(rows) for cart in range(3)],
            None if onecentre is None else [
                onecentre_difference[row, cart]
                for row in range(rows) for cart in range(3)
            ],
            joint=False,
        )
        history.append(change)
        if verbose:
            print(f"  iter {iteration + 1}: |ddv_scf|^2 = {change:.3e}")
        # Tested before mixing, and on convergence the *input* is kept, as
        # ``mix_potential`` returns it (``mix_pot.f90:85-113``): the potential
        # this pass's ``dpsi`` was solved at, which the multipliers' response
        # and the assembly rebuild their perturbation from.
        if change < tr2 / calculation.system.npol:
            converged = True
            break
        if onecentre is None:
            dvscf = mixer.mix(dvscf, induced, host=host)
        else:
            # **One Anderson problem over both.** The one-centre potential and
            # ``dV_scf`` are coupled through the same ``dbecsum``, which is why
            # ``mix_pot`` concatenates them rather than mixing them apart.
            dvscf, onecentre = mixer.mix(
                [dvscf, onecentre], [induced, induced_onecentre], host=host
            )

    # ``ef_shift_wfc``: the level's motion belongs to the first-order *states*
    # as well, and it is applied once at the end, as QE applies it -- the loop
    # itself is driven by the density, which already carries the correction.
    # It matters because the second derivative consumes ``dpsi`` as a tangent
    # rather than through the density it builds.
    if solver.smearing is not None and shifts is not None:
        # **Over the rows solved, not over the cell's atoms.** ``dpsi`` and
        # ``shifts`` are both indexed by *perturbation*, so their leading size
        # is ``len(atoms)`` and using an absolute atom index here reads off the
        # end of them -- or, for a subset that happens to start at 0, reads the
        # wrong row silently. It is the only place in this module that got the
        # two confused, and it is a **metals-only** path, which is why an
        # insulating validation could not see it.
        displacements.shift_states(shifts)

    extras = {
        "dvscf": dvscf,
        "onecentre": onecentre,
        "dbecsum": None if onecentre is None else per_mode,
    }
    return (displacements.dpsi, symmetrised, history,
            displacements.iterations / max(displacements.solves, 1), converged,
            extras)


def _core_charge_response(calculation, density, positions, atoms=None,
                          host: bool = False):
    """``dv_xc`` from the core charge travelling with its atom -- ``addcore``.

    ``(nat, 3, nspin_mag, n1, n2, n3)``, or ``None`` when no species has a
    nonlinear core correction.

    **This is not an ultrasoft term and it was missing before the ultrasoft
    ones.** :func:`_bare_displacements` builds ``dV_bare|psi>`` at a *frozen*
    ``v_scf``, which is right for the local potential and the projectors and
    wrong for the exchange-correlation potential: ``rho_core(r - tau)`` moves
    with the atom, so ``v_xc[rho + rho_core]`` changes even at a frozen valence
    density. QE keeps it as ``drhoc`` and hands it to ``dv_of_drho`` beside the
    response density (``solve_linter.f90``'s ``addcore``); here it is one
    ``jvp`` of the potential through
    :meth:`~defuqumat.scf.driver.Calculation.at_positions` at a *fixed* density,
    which is the same object and needs no second expression.

    It is independent of the self-consistent iteration -- nothing responds to
    it -- so it is built once and added to the induced potential every step.

    **Every committed phonon case before this had no core charge**, which is
    why it went unnoticed: ``Si.pz-vbc`` and ``Al.pz-vbc`` are
    ``core_correction="false"`` and the ultrasoft and PAW silicon datasets that
    first exercised it are ``"T"``. Its size, on ``si-epsilon-us``: leaving it
    out puts the response density **45%** away from a finite difference of
    re-converged densities, and the optical mode at 785 cm^-1 against
    ``ph.x``'s 513. It is the same omission ``force_cc`` exists for one
    derivative down, and :func:`~defuqumat.forces.energy.frozen_energy` already
    carries it there -- so the assembly was right and only the response was not.
    """
    if calculation.rho_core is None:
        return None
    if positions is None:
        raise ValueError(
            "the displacement response of a dataset with a nonlinear core "
            "correction needs the positions: the core charge travels with its "
            "atom, so v_xc changes at a frozen valence density (addcore). "
            "Pass positions=... to self_consistent_response"
        )
    positions = jnp.asarray(positions)
    nat = positions.shape[0]
    rho = jnp.asarray(density)

    def potential_at(pos):
        return calculation.at_positions(pos).potential(rho).v_scf

    chosen = tuple(range(nat)) if atoms is None else tuple(atoms)
    fields = []
    for atom in chosen:
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            field = compiled_jvp(potential_at, (positions,), (tangent,))[1]
            # ``host``: the loop keeps its ``3 nat`` grids in host memory, and
            # this is one more set of them.
            fields.append(np.asarray(field) if host else field)
    stack = np.stack if host else jnp.stack
    return stack(fields).reshape((len(chosen), 3) + rho.shape)


def _bare_displacements(calculation, solver, v_scf, positions, atoms=None) -> np.ndarray:
    """``dV_bare/du |psi>`` for every atom and cartesian direction.

    ``dvqpsi_us.f90`` builds this term by term from ``dvloc/dtau``, the
    projectors' derivatives and (for a core correction) ``drhoc``. Here it is
    one ``jvp`` through
    :meth:`~defuqumat.scf.driver.Calculation.at_positions` at frozen ``v_scf`` --
    the same method the force differentiates, and the same call
    :func:`~defuqumat.response.born.born_effective_charges` already makes for
    ``Z*`` (``efield.py`` imports it and calls it; the shared frozen-potential
    ``at_positions`` ``jvp`` is in ``born.py``), which is why this is the one
    piece of the phase that was written before it.

    "Bare" is the whole point: the self-consistent part of the perturbation is
    the induced ``dV_scf`` that the loop above adds, so the potential handed to
    ``at_positions`` here is held *fixed* while the atoms move under it.

    ``atoms`` restricts the perturbations to a subset, in which case the result
    is ``(len(atoms), 3)`` and its first index runs over the subset rather than
    over the cell. That is the whole of what makes a partial dynamical matrix
    cheap: this array and ``dpsi`` beside it are the memory the phase costs.

    Returns an object array of shape ``(nat, 3)`` whose entries are
    ``(nspin, nk, nocc, npwx)`` -- see the module docstring on what that costs.
    """
    batch = calculation.k_batch
    psi = solver.psi
    nat = positions.shape[0]
    chosen = tuple(range(nat)) if atoms is None else tuple(atoms)

    eigenvalues = solver.eigenvalues

    def h_psi(moved_positions):
        moved = calculation.at_positions(moved_positions)
        hamiltonians = moved.hamiltonian(v_scf, solver.ddd_paw)
        applied = []
        for spin, hamiltonian in enumerate(hamiltonians):
            values = over_kpoints(hamiltonian, psi[spin], batch)
            if hamiltonian.has_overlap:
                # ``compute_deff``: the right-hand side of the Sternheimer
                # equation for a *displacement* is ``(dH/du - eps dS/du)|psi>``,
                # which ``dvqpsi_us_only`` builds from
                # ``deff = deeq - et qq`` rather than from ``deeq``. The second
                # half is identically zero when ``S`` is the identity, so this
                # branch is the ultrasoft and PAW one and nothing else here
                # changes. Leaving it out solves a different equation whose
                # solution is still orthogonal to the occupied manifold and
                # still converges.
                overlap = over_kpoints(hamiltonian, psi[spin], batch, overlap=True)
                values = values - eigenvalues[spin][..., None] * overlap
            applied.append(values)
        return jnp.stack(applied)

    bare = np.empty((len(chosen), 3), dtype=object)
    for row, atom in enumerate(chosen):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            bare[row, cart] = compiled_jvp(h_psi, (positions,), (tangent,))[1]
    return bare


def orthogonality_states(calculation, solver, positions, atoms=None) -> np.ndarray | None:
    """``dpsi^ort_n = -1/2 sum_m psi_m <psi_m|dS/du|psi_n>``, one per mode.

    ``compute_drhous.f90``'s ingredient, and the piece of the first-order
    wavefunction the Sternheimer solve does **not** produce. ``solve``'s answer
    is orthogonal to the occupied manifold in the ``S`` metric -- that is what
    ``orthogonalize``'s projector imposes -- but the physical first-order state
    is not, because the constraint it has to satisfy is
    ``<psi + dpsi|S(u + du)|psi + dpsi> = 1`` and ``S`` itself has moved. The
    occupied-occupied block is therefore fixed rather than free, and this is it.

    ``None`` for a norm-conserving dataset, where ``S`` does not move and the
    block is zero. Returns an object array of shape ``(nat, 3)`` whose entries
    have ``dpsi``'s own shape.
    """
    derivatives = overlap_derivatives(calculation, solver, positions, atoms)
    if derivatives is None:
        return None
    psi = solver.psi
    # A metal's block carries ``compute_weight``'s ``wgg`` on each pair
    # (:func:`pair_weights`), and with it the occupation: the tangent is then
    # contracted at ``wk``, like the Sternheimer part beside it.
    pairs = pair_weights(solver)
    out = np.empty(derivatives.shape, dtype=object)
    for index in np.ndindex(derivatives.shape):
        block = derivatives[index] if pairs is None else derivatives[index] * pairs
        out[index] = -0.5 * jnp.einsum("skmg,skmn->skng", psi, block)
    return out


def overlap_derivatives(calculation, solver, positions, atoms=None) -> np.ndarray | None:
    """``S'_mn = <psi_m|dS/du|psi_n>`` over the occupied block, per mode.

    ``None`` for a norm-conserving dataset. The object two other things are
    built from -- :func:`orthogonality_states` and the gauge correction in
    :func:`multiplier_response` -- so it is computed once and shared.
    """
    if not calculation.is_ultrasoft:
        return None
    psi = solver.psi
    nat = positions.shape[0]
    batch = calculation.k_batch

    noncolin = bool(calculation.noncolin)

    def overlap_matrix(pos):
        """``<psi_m|S(u)|psi_n>`` -- only the augmentation half moves.

        For a spinor the band is ``2 npwx`` long and the metric is ``qq_so``,
        a 2x2 matrix in spin space, which is
        :func:`~defuqumat.response.born.constraint_sandwich_at`'s contraction:
        a fully-relativistic dataset's overlap couples the two components
        through ``fcoef``, and the scalar ``qq`` on each block alone is the
        ``j``-averaged operator.
        """
        moved = calculation.at_positions(pos)
        vkb = moved.projectors.vkb
        npwx = vkb.shape[1]
        qq = jnp.asarray(
            moved.qq_so if noncolin else moved.projectors.qq
        ).astype(psi.dtype)
        blocks = []
        for spin in range(psi.shape[0]):
            def one_k(ik, spin=spin):
                if noncolin:
                    bands = psi[spin][ik]
                    becp = jnp.einsum(
                        "gc,nag->nac", vkb[ik].conj(),
                        bands.reshape(bands.shape[:-1] + (2, npwx)),
                    )
                    return jnp.einsum("mai,abij,nbj->mn", becp.conj(), qq, becp)
                becp = jnp.einsum("gc,ng->nc", vkb[ik].conj(), psi[spin][ik])
                return jnp.einsum("mi,ij,nj->mn", becp.conj(), qq, becp)

            blocks.append(map_k(one_k, jnp.arange(psi.shape[1]), batch=batch))
        return jnp.stack(blocks)

    chosen = tuple(range(nat)) if atoms is None else tuple(atoms)
    pairs = occupied_pairs(solver.occupied_counts, solver.smearing, psi.shape[2])
    out = np.empty((len(chosen), 3), dtype=object)
    for row, atom in enumerate(chosen):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            value = compiled_jvp(overlap_matrix, (positions,), (tangent,))[1]
            out[row, cart] = (value if pairs is None
                              else jnp.where(pairs, value, 0.0))
    return out


def occupied_pairs(counts, smearing, width):
    """``(nspin, 1, width, width)``: which ``(m, n)`` of ``S'_mn`` are occupied pairs.

    **The occupied block of a spin channel is that channel's own**, and for a
    magnetic insulator the two are not the same size. The solver keeps
    ``max(occupied_counts)`` bands in every channel because the shape has to be
    static, so on triplet O2, at counts ``(7, 5)``, the down channel's block
    carries its two empty pi*_g bands. Left in ``S'``, they enter
    :func:`orthogonality_states` as partners ``m`` of an occupied ``n`` --
    ``-1/2 psi_pi* <pi*|dS/du|psi_n>`` added to a first-order state whose
    component along ``pi*`` the Sternheimer solve has *already* produced, since
    ``P_c`` keeps a band that is conduction in its own channel -- and they enter
    :func:`multiplier_response`'s gauge correction as rows. QE cannot write
    either down: ``compute_drhous.f90`` and ``drhodvus.f90`` run their band
    loops over ``nbnd_occ(ikk)``, the spin-resolved k-point's own count.

    Measured on triplet O2 against ``reference.dyn.o2-fixed-lsda-phonon``
    (``notes/incompat/I2.md``). Nothing validated before could see it: ``S'``
    is zero for a norm-conserving dataset, and the mask is all ones wherever
    the two channels are filled to the same depth.

    ``None`` -- no mask -- for a metal, where every band is in the block and the
    occupation is carried by :func:`pair_weights` instead, and wherever the mask
    would be all ones, which keeps those paths bit for bit what they were.
    """
    if smearing is not None:
        return None
    counts = np.asarray(counts)
    if np.all(counts >= width):
        return None
    bands = np.arange(width)
    occupied = bands[None, :] < counts[:, None]
    return jnp.asarray(occupied[:, None, :, None] & occupied[:, None, None, :])


def pair_weights(solver):
    """``(nspin, nk, nbnd, nbnd)``: ``compute_weight``'s ``wgg``, as ``W[m, n]``.

    ``W_mn = f_n (1 - theta_mn) + f_m theta_mn`` with
    ``theta_mn = wgauss((eps_m - eps_n)/degauss, 0)`` and ``f`` the smeared
    occupation, which is the ``mixed`` weight of
    :meth:`~defuqumat.response.sternheimer.SternheimerSolver._smeared_projection`
    without its ``alpha_pv`` piece (that piece cancels in the solution). It is
    symmetric, because ``theta_mn + theta_nm = 1``, and it is the whole of what
    a metal adds to the three tangents a moving ``S`` brings
    (``notes/incompat/I86.md``):

    the Sternheimer solve of a metal puts ``theta_mn (f_n - f_m)/(eps_n - eps_m)
    <m|dH - eps_n dS|n>`` on the occupied pair ``(m, n)``, and the exact
    first-order density matrix differs from what that builds by
    ``-W_mn S'_mn``. So the occupied block of the first-order state is
    ``-1/2 sum_m psi_m S'_mn W_mn`` at the k-point weight ``wk``
    (``incdrhous``), and the multipliers' tangent is
    ``wk W_mn [<m|dH|n> - 1/2 (eps_m + eps_n) S'_mn]``. For an insulator ``W``
    is 1 on the occupied block and 0 elsewhere, which is
    :func:`occupied_pairs`' mask, and both expressions are the ones validated
    there.

    ``None`` for an insulator.
    """
    smearing = solver.smearing
    if smearing is None:
        return None
    eps = solver.eigenvalues
    occupation = wgauss((smearing.ef - eps) / smearing.degauss, smearing.ngauss)
    step = wgauss((eps[..., :, None] - eps[..., None, :]) / smearing.degauss, 0)
    return occupation[..., None, :] * (1.0 - step) + occupation[..., :, None] * step



def _pad_to_bands(blocks, nbnd):
    """Widen each ``(nspin, nk, nocc, npwx)`` tangent to ``nbnd`` with zeros.

    ``blocks`` is the object array :func:`orthogonality_states` returns, indexed
    ``[row, cart]``. ``None`` passes through, which is the norm-conserving case.
    """
    if blocks is None:
        return None
    padded = np.empty(blocks.shape, dtype=object)
    for index, block in np.ndenumerate(blocks):
        if block is None:
            padded[index] = None
            continue
        block = jnp.asarray(block)
        if block.shape[2] == nbnd:
            padded[index] = block
            continue
        padded[index] = jnp.zeros(
            block.shape[:2] + (nbnd,) + block.shape[3:], dtype=block.dtype
        ).at[:, :, :block.shape[2]].set(block)
    return padded


def non_variational_response(calculation, positions, psi, weights, density,
                             becsum, ort, atoms=None, state_weights=None):
    """``drhous`` and ``becsumort``: the mixed state's change at frozen ``dpsi``.

    ``PHonon/PH/drho.f90``, whose own summary is the definition -- "the change
    of the charge density due to the displacement, at fixed wavefunctions; the
    orthogonality part is included in the computed change". Two things move it
    and neither exists for a norm-conserving dataset:

    * the augmentation charge ``Q_ij(r - tau)`` and the projectors ``beta(r -
      tau)`` travel with their atom, so ``rho`` and ``becsum`` depend on the
      positions explicitly (``addusddens``'s ``alpha bb`` term and
      ``alphasum``);
    * the occupied block of the first-order state is not zero
      (:func:`orthogonality_states`), and it changes both.

    Both fall out of ``jvp`` of the raw mixed-state builders --
    :func:`~defuqumat.response.born._raw_mixed_state`'s, reused rather than
    restated -- and **the two halves are returned apart**, because they are
    consumed in different places. Their sum is what screens
    (:func:`self_consistent_response`) and what the assembly's density tangent
    has to carry; but the ``moved`` half is *also* generated by the assembly's
    own position tangent, since :func:`_force_constants` hands the mixed state
    to :func:`~defuqumat.forces.energy.frozen_energy` as a function of where the
    atoms are rather than as an array -- which it must, because the gradient
    being differentiated is otherwise not the force at all. It would then be
    counted twice.

    **That is the term ``addusforce`` is**, one derivative down: at a frozen
    density the ultrasoft force is missing the augmentation charge's own
    motion, and P25 could freeze it because a norm-conserving ``rho`` has no
    explicit position dependence. Measured on ultrasoft silicon, freezing it
    puts a whole column of the force constants at **0.26** of a
    finite-differenced one, with the acoustic sum rule holding throughout --
    which is P28a's lesson again, that an atom-sum is blind to a transfer
    between atoms.

    **The two halves take different weights in a metal.** The ``moved`` half is
    the mixed state's change at frozen states and reads ``wg``
    (``compute_alphasum``); the ``ort`` half is a state tangent that already
    carries its occupation (:func:`pair_weights`), so it is accumulated at
    ``wk`` (``incdrhous`` and ``addusdbec`` at ``weight = wk``), which is
    ``state_weights``. Both builders are linear in the weight, so the tangent
    does not depend on which primal it is taken at. ``None`` is ``weights``,
    which is the insulator, where the two coincide.

    Returns ``((drho_moved, becsum_moved), (drho_ort, becsum_ort))`` as object
    arrays of shape ``(nat, 3)``, or ``(None, None)`` for a norm-conserving
    dataset.
    """
    if ort is None:
        return (None, None), (None, None)
    from defuqumat.response.born import _raw_mixed_state

    density_of, becsum_of = _raw_mixed_state(
        calculation, positions, psi, weights, density, becsum
    )
    tangent_weights = {"moved": weights,
                       "ort": weights if state_weights is None else state_weights}

    def mixed(pos, states, occupations):
        moved = calculation.at_positions(pos)
        parts = becsum_of(moved, states, occupations)
        return density_of(moved, states, occupations, parts), parts

    nat = positions.shape[0]
    chosen = tuple(range(nat)) if atoms is None else tuple(atoms)
    zero_p, zero_s = jnp.zeros_like(positions), jnp.zeros_like(psi)
    halves = {}
    for name in ("moved", "ort"):
        halves[name] = (np.empty((len(chosen), 3), dtype=object),
                        np.empty((len(chosen), 3), dtype=object))
    for row, atom in enumerate(chosen):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            for name, pair in (("moved", (tangent, zero_s)),
                               ("ort", (zero_p, ort[row, cart]))):
                _, (rho, parts) = compiled_jvp(
                    lambda pos, states, w=tangent_weights[name]: mixed(
                        pos, states, w),
                    (positions, psi), pair)
                halves[name][0][row, cart] = rho
                halves[name][1][row, cart] = parts
    return halves["moved"], halves["ort"]


def symmetrize_becsum_modes(calculation, per_mode, atoms=None):
    """``PAW_dusymmetrize`` on the ``3 nat`` displacement patterns.

    The counterpart of :meth:`~defuqumat.scf.driver.Calculation.
    symmetrize_atom_displacement` one level down, and the reason it is not
    optional is P24a's: a ``becsum`` response on a reduced k-set is a **polar**
    object carrying a direction, so a wedge sum of it is not the crystal's until
    the group has put back what the reduction left out. Worth 1.6e-2 on the
    dielectric constant of PAW silicon.

    ``per_mode`` is an object array of shape ``(nat, 3)`` whose entries are
    per-species ``becsum`` tuples; the result has the same layout. A no-op when
    there is no PAW species or no symmetry to average over.

    ``atoms``: the displaced atoms when ``per_mode`` is a subset's
    ``(len(atoms), 3)``, a union of orbits. Only the *displaced* axis is
    restricted: the ``becsum`` blocks inside each entry still run over every
    atom of their species, and ``apply_atom_displacement`` permutes those with
    its own full table.
    """
    if per_mode is None or calculation._becsum_symmetry is None:
        return per_mode
    nat, ncart = per_mode.shape
    sample = per_mode[0, 0]
    if not sample:
        return per_mode
    stacked = tuple(
        None if sample[species] is None else jnp.stack([
            jnp.stack([per_mode[atom, cart][species] for cart in range(ncart)])
            for atom in range(nat)
        ])
        for species in range(len(sample))
    )
    rotations = cartesian_rotations(calculation.system.cell, calculation.symmetries)
    mapping = atom_mapping(
        calculation.system.cell, calculation.system.structure, calculation.symmetries
    )
    if atoms is not None:
        mapping = subset_atom_mapping(mapping, atoms)
    symmetrised = calculation._becsum_symmetry.apply_atom_displacement(
        stacked, rotations, mapping
    )
    out = np.empty((nat, ncart), dtype=object)
    for atom in range(nat):
        for cart in range(ncart):
            out[atom, cart] = tuple(
                None if values is None else values[atom, cart]
                for values in symmetrised
            )
    return out


def _symmetrize_modes(calculation, per_mode, atoms=None):
    """``symdvscf`` on an object array of ``3 nat`` grid responses, or of a
    subset's ``3 len(atoms)`` when ``atoms`` is a union of orbits."""
    nat, ncart = per_mode.shape
    stacked = jnp.stack([
        jnp.stack([per_mode[atom, cart] for cart in range(ncart)])
        for atom in range(nat)
    ])
    symmetrised = calculation.symmetrize_atom_displacement(stacked, atoms=atoms)
    out = np.empty((nat, ncart), dtype=object)
    for atom in range(nat):
        for cart in range(ncart):
            out[atom, cart] = symmetrised[atom, cart]
    return out


def _ground_state_multipliers(weights, eigenvalues, dtype):
    """``Lambda_mn = delta_mn w_n eps_n``: the multipliers at the solution.

    :func:`~defuqumat.response.born._ground_state_multipliers`, and the same
    argument for passing it explicitly: it is what puts the matrix on the
    tangent's argument list, so that ``dLambda`` can be a tangent of it.
    """
    identity = jnp.eye(weights.shape[-1], dtype=dtype)
    return (weights * eigenvalues)[..., :, None].astype(dtype) * identity


def multiplier_response(calculation, solver, bare, extras, weights, nbnd,
                        derivatives):
    """``dLambda_mn``: the multipliers' own tangent, in the gauge ``ort`` fixes.

    **The multipliers are variables of the functional and they move.** P15
    writes the energy with the orthonormality constraint carried explicitly and
    its multipliers among the *frozen* variables, which is what makes the force
    a partial derivative. Differentiating that gradient a second time needs
    their tangent beside the states' and the density's:

        d^2E/du_i du_j = d_i d_j L + (d_psi d_j L).dpsi_i + (d_Lambda d_j L).dLambda_i

    and ``d_Lambda d_j L = -<psi|dS/du_j|psi>``, which vanishes identically when
    ``S`` is the identity. That is the whole of why P25 was norm-conserving: not
    a missing routine, a missing tangent.

    **It is a matrix and not a diagonal, and the reason is a gauge.** Write the
    constraint with a diagonal multiplier ``w_n eps_n`` and the functional stops
    being invariant under a unitary mixing of the occupied states -- the
    eigenvalues weight the bands differently -- while the *state* tangent is
    only defined up to exactly such a mixing: the constraint fixes
    ``c + c^dagger = -S'`` and leaves the antihermitian part of ``c`` free.
    :func:`orthogonality_states` picks the hermitian representative; the
    physical branch picks another; and a diagonal multiplier can tell them
    apart. **The acoustic sum rule is what says so**: with the diagonal form it
    stops at 1.7e-2 Ry/bohr^2 on ultrasoft silicon whatever else is switched on,
    where the matrix form takes it to the basis-set floor. Carrying the full
    ``Lambda_mn`` restores the invariance, because the pair ``(dpsi, dLambda)``
    then transforms together and the gauge cancels between them.

    In that gauge, differentiating ``Lambda_mn = w_n <psi_m|H|psi_n>`` along the
    branch gives

        dLambda_mn = w_n [ <psi_m|dH|psi_n> - 1/2 (eps_m + eps_n) S'_mn ]

    -- the ``1/2 (eps_m + eps_n)`` and not ``eps_n``, which is
    :func:`~defuqumat.response.born._multiplier_response`'s expression and is
    right *there* because a field leaves ``dpsi`` orthogonal to the occupied
    manifold, so the two terms carrying it drop out. A displacement does not,
    and using the field's expression here is wrong by ``(eps_n - eps_m)/2`` on
    every off-diagonal entry. Since the perturbation handed in already carries
    ``-eps_n dS`` (``compute_deff``), what has to be added back is
    ``+1/2 (eps_n - eps_m) S'_mn``.
    """
    from defuqumat.response.born import _multiplier_response

    nat = bare.shape[0]
    dvscf, onecentre = extras["dvscf"], extras["onecentre"]
    nocc = solver.nocc
    eps = solver.eigenvalues[..., :nocc]
    # ``(nspin, nk, nocc, nocc)``: (eps_n - eps_m) with m the row.
    gaps = eps[..., None, :] - eps[..., :, None]
    pairs = pair_weights(solver)
    out = np.empty((nat, 3), dtype=object)
    for atom in range(nat):
        for cart in range(3):
            perturbation = _bare_plus_induced(
                solver, bare[atom, cart], dvscf[atom, cart], True,
                None if onecentre is None else onecentre[atom, cart],
            )
            if pairs is not None:
                # **A metal: ``wk W_mn`` in place of ``w_n``**
                # (:func:`pair_weights`). The Sternheimer state of a metal is not
                # in the hermitian gauge -- its occupied block is split between
                # the two partners by ``theta`` -- and the exact energy-weighted
                # density matrix minus what that state and the occupied block
                # generate is this matrix, worked out in
                # ``notes/incompat/I86.md``. No row mask: the weight is the one
                # the solve used, on every band it kept.
                elements = _occupied_matrix_elements(solver, perturbation)
                k_weight = solver.density_weights[..., :1, None]
                out[atom, cart] = k_weight * pairs * (
                    elements + 0.5 * gaps * derivatives[atom, cart])
                continue
            matrix = _multiplier_response(
                solver, perturbation, weights, nbnd, nocc
            )
            correction = 0.5 * gaps * derivatives[atom, cart] * (
                weights[:, :, None, :nocc]
            )
            out[atom, cart] = matrix.at[:, :, :nocc, :nocc].add(correction)
    return out


def _occupied_matrix_elements(solver, perturbation):
    """``<psi_m|perturbation|psi_n>`` over the solver's bands, unweighted.

    :func:`~defuqumat.response.born._multiplier_response`'s contraction without
    its weight and row mask, for :func:`multiplier_response`'s metal branch.
    """
    from defuqumat.eager import compiled

    psi = solver.psi
    blocks = []
    for spin in range(solver.nspin):
        def one_k(ik, spin=spin):
            applied = perturbation(psi[spin][ik], ik, spin)
            return jnp.einsum("mg,ng->mn", jnp.conj(psi[spin][ik]), applied)

        blocks.append(compiled(
            lambda indices, one_k=one_k: map_k(
                one_k, indices, batch=solver.calculation.k_batch),
            jnp.arange(psi.shape[1]),
        ))
    return jnp.stack(blocks)


def _stack_modes(per_mode):
    """``(nat, 3)`` object array of grid fields -> one stacked array."""
    nat, ncart = per_mode.shape
    return jnp.stack([
        jnp.stack([per_mode[atom, cart] for cart in range(ncart)])
        for atom in range(nat)
    ])


def _subtract_becsum(left, right):
    """``left - right`` on two ``(nat, 3)`` object arrays of ``becsum`` tuples."""
    out = np.empty(left.shape, dtype=object)
    for index in np.ndindex(left.shape):
        out[index] = tuple(
            None if a is None else a - b
            for a, b in zip(left[index], right[index])
        )
    return out


def _add_becsum(left, right):
    """Add two ``(nat, 3)`` object arrays of per-species ``becsum`` tuples."""
    out = np.empty(left.shape, dtype=object)
    for index in np.ndindex(left.shape):
        out[index] = tuple(
            None if a is None else a + b
            for a, b in zip(left[index], right[index])
        )
    return out


def _bare_plus_induced(solver, bare_mode, dv, include_induced: bool,
                       dddd_paw=None):
    """``dV_bare|psi> + dV_scf|psi>`` as the solver's callback wants it.

    The sign convention is :meth:`SternheimerSolver.chi0`'s and not
    :func:`~defuqumat.response.efield._solve_stored`'s: what is handed in is an
    honest ``dH|psi>``, and ``project`` carries ``orthogonalize``'s sign
    already. The extra minus in the electric-field path belongs to
    ``dvpsi_e``'s commutator, which is a different right-hand side.
    """
    if not include_induced:
        return lambda psi, ik, spin: bare_mode[spin][ik]

    induced = solver.perturbation(dv, dddd_paw)

    def perturbation(psi, ik, spin):
        return bare_mode[spin][ik] + induced(psi, ik, spin)

    return perturbation


def _state_weights(solver, weights):
    """The weight the *state* tangent is contracted with -- ``wg`` or ``wk``.

    An insulator's ``dpsi`` is a plain wavefunction response and the functional's
    own ``wg`` is right for it. A metal's is not: ``orthogonalize``'s smearing
    branch scales the right-hand side by ``wg1 = f``, so the occupation is
    already inside the tangent and contracting against ``wg = wk f`` would apply
    it twice. ``incdrhoscf`` is called with ``wk`` for the same reason, which is
    what :attr:`SternheimerSolver.density_weights` holds -- and reusing that
    array rather than rebuilding the broadcast is deliberate: it is the one the
    ``response_density == def ldos`` identity was measured against, so a
    normalisation convention cannot differ between the density and the matrix.

    It is ``nocc``-sliced for an insulator and full-length for a metal (where
    ``keep = nbnd``), which is why the insulator branch returns ``weights``
    whole instead: :func:`_force_constants` contracts a tangent padded to
    ``nbnd``, and the padding is zero exactly where the two would disagree.
    """
    if solver.smearing is None:
        return weights
    return jnp.broadcast_to(solver.density_weights[:, :, :1], weights.shape)


def _force_constants(
    calculation, positions, psi, weights, state_weights, eigenvalues, density,
    dpsi, drho, nocc, becsum=(), dbecsum=None, multipliers=None, ort=None,
    atoms=None, on_row=None,
) -> np.ndarray:
    """``d^2E/du_i du_j``: two ``jvp`` of the force's gradient per mode.

    The frozen-state energy is a function of three things that move -- where the
    atoms are, what the states are and what the density is -- and the total
    second derivative is its derivative along the tangent that carries all
    three:

        D[:, i] = jvp( grad_u L )(u, psi, rho ; e_i, dpsi_i, drho_i)

    The ``e_i`` half is the frozen Hessian (``dynmat0``, ``d2ionq``, the local
    potential's and the projectors' second derivatives); the ``dpsi_i`` and
    ``drho_i`` halves are the electronic response (``drhodv``). Nothing
    separates them in the mathematics: they are components of one tangent
    vector, and for an insulator one ``jvp`` is what this was.

    **The state tangent is contracted with a different weight from the rest,
    and that is the whole of what a metal adds.** ``L`` weights the states by
    ``wg = wk f``, which is right for the frozen Hessian -- ``dynmat_us.f90``
    reads ``wg(ibnd, ikk)`` -- and wrong for the electronic half, because a
    metal's ``dpsi`` already carries its occupation: ``orthogonalize``'s
    smearing branch scales the right-hand side by ``wg1 = f`` and
    ``incdrhoscf`` then accumulates with ``wk``. Contracting such a tangent
    against a ``wg``-weighted functional counts ``f`` twice. QE never does,
    because its two halves are two routines: ``drhodvnl.f90`` contracts with
    ``2 wk(ikk)`` while ``dynmat_us.f90`` uses ``wg``. So the ``jvp`` is split
    in two and the state tangent is taken against ``L[wk]``:

        D[:, i] = jvp_(u, rho)( grad_u L[wg] )(e_i, drho_i)
                + jvp_psi(     grad_u L[wk] )(dpsi_i)

    which is de Gironcoli's Eq. (B19) structure -- the frozen Hessian at ``wg``,
    the electronic response with the metal's own weights.

    **The occupations' own first-order change needs no term of its own**, which
    is the part of this that is not obvious. ``df_n`` and the Fermi level's
    motion are already inside ``dpsi``: the ``(f_i - f_j)/(eps_i - eps_j)``
    structure of ``orthogonalize``'s ``wwg`` is what puts the valence-valence
    block there -- it vanishes identically for an insulator, where every
    occupied ``f`` is 1 -- and ``ef_shift_wfc`` puts the rest. The check that
    it is complete is one this code already makes: ``wk 2 Re[psi* dpsi]`` equals
    the corrected response density to 1e-10, and for a local perturbation
    contracting the tangent *is* ``int drho dV_bare``, which is ``drhodvloc``.

    **The split is unconditional and an insulator is the proof of it.** There
    ``wk = wg`` on every occupied band and ``dpsi`` is zero on the rest, so the
    two ``jvp`` sum to exactly the one they replace -- silicon's optical mode is
    unchanged to round-off, which is the regression that guards this refactor.
    The density tangent goes with the frozen Hessian rather than with the
    states because the terms it reaches -- ``int vltot(tau) rho``,
    ``E_xc[rho + rho_core(tau)]`` -- carry no state weight at all, so it makes
    no difference which half it is put in.

    **The density is a separate argument rather than a function of the states,
    and that is not a convenience.** ``L`` builds its density with the SCF's own
    *scalar* symmetrisation, which is how a wedge sum is completed and is right
    for the ground state. A response must not go through it: displacing one atom
    breaks the crystal's symmetry, and averaging the result over the full group
    of the undisplaced crystal projects the perturbation away. Left to the chain
    rule the state tangent goes straight through that average, and the symptom
    is a matrix that looks perfectly cubic and violates the acoustic sum rule by
    0.72 Ry/bohr^2 -- 580 cm^-1 where the answer is 2. So ``drho`` comes from
    :func:`self_consistent_response`, already symmetrised the way ``symdvscf``
    symmetrises a response, and is handed in.

    **Which bands the tangent carries differs between the two regimes**, and
    the padding is exact in both for different reasons. For an insulator
    ``dpsi`` holds the ``nocc`` bands the Sternheimer equation solves for and
    the rest is padded with zero: the empty bands have ``wg = 0`` in every term
    of ``L``, so no derivative of it could see them anyway. For a metal
    ``solver.nocc`` is ``nbnd`` -- ``orthogonalize``'s smearing branch keeps
    every band, since "occupied" is not a count there -- and nothing is padded.
    The bands above the smearing carry ``wg1 = 0`` inside ``dpsi`` instead,
    which is what makes them harmless against a ``wk`` that is *not* zero there.
    """
    nat = positions.shape[0]
    ultrasoft = calculation.is_ultrasoft

    # **An ultrasoft or PAW metal: the band set twice** (``notes/incompat/I86.md``).
    # The state tangent has to be contracted at ``wk`` and everything else at
    # ``wg``, as the norm-conserving split below does with two ``jvp``. Here
    # that split is not available: the density is a function of the states
    # *and* the weights, so evaluating ``L[wk]`` puts the Hartree and XC
    # kernels at the wrong density, and the cross term ``d^2 rho / du dpsi``
    # pairs the state tangent (at ``wk``) with the augmentation charge's own
    # motion (at ``wg``) in one product. So every band is carried twice: a copy
    # at ``wg - wk`` with no tangent and a copy at ``wk`` that carries it. Every
    # term of ``L`` is linear in the weights, so the primal is the ground state
    # at ``wg`` exactly (to round-off), a derivative in the coordinate sees
    # ``wg``, and the state tangent sees ``wk`` wherever it enters. The ground
    # multipliers split the same way, ``diag((wg - wk) eps)`` and
    # ``diag(wk eps)``, and ``dLambda`` goes on the second copy's block. For an
    # insulator ``state_weights is weights`` and nothing is doubled.
    doubled = ultrasoft and state_weights is not weights
    nbnd = psi.shape[2]
    if doubled:
        psi = jnp.concatenate([psi, psi], axis=2)
        weights = jnp.concatenate(
            [weights - state_weights, state_weights], axis=2)
        eigenvalues = jnp.concatenate([eigenvalues, eigenvalues], axis=2)

    ground = (
        _ground_state_multipliers(weights, eigenvalues, psi.dtype)
        if ultrasoft else None
    )
    if ultrasoft:
        from defuqumat.response.born import _raw_mixed_state

        raw_density, raw_becsum = _raw_mixed_state(
            calculation, positions, psi, weights, density, becsum
        )

        def raw_mixed(pos, states):
            """The unsymmetrised mixed state, as a function of both."""
            moved = calculation.at_positions(pos)
            parts = raw_becsum(moved, states, weights)
            return raw_density(moved, states, weights, parts), parts

    def energy(pos, states, rho, parts, lambdas, w):
        """``L(u, psi, Lambda)`` with the mixed state where it belongs.

        For an ultrasoft or PAW dataset the mixed state stays a **function of
        both the positions and the states**, and ``rho``/``parts`` carry only
        the correction from the wedge sum to the symmetrised response. Freezing
        it as an array -- right for a norm-conserving run, and P25's choice --
        costs two different things: the gradient loses ``addusforce``, so what
        is differentiated is not the force; and the second derivative loses the
        cross term ``d^2 rho / du dpsi . dpsi``, the augmentation charge's own
        position dependence applied to the state response, which is what
        ``addusdynmat`` and ``drhodvus`` are.
        """
        if not ultrasoft:
            return frozen_energy(
                calculation, pos,
                FrozenState(wavefunctions=states, weights=w,
                            eigenvalues=eigenvalues),
                density=rho, becsum=parts, multipliers=lambdas,
                spinors=True,
            )

        def becsum_builder(moved, built_states, occupations):
            return tuple(
                None if raw is None else raw + extra
                for raw, extra in zip(
                    raw_becsum(moved, built_states, occupations), parts
                )
            )

        def density_builder(moved, built_states, occupations, built):
            return raw_density(moved, built_states, occupations, built) + rho

        return frozen_energy(
            calculation, pos,
            FrozenState(wavefunctions=states, weights=w, eigenvalues=eigenvalues),
            density=density_builder, becsum=becsum_builder, multipliers=lambdas,
            spinors=True,
        )

    gradient = jax.grad(energy, argnums=0)

    def frozen(pos, rho, parts, lambdas):
        """The coordinate, the mixed state and the multipliers, at ``wg``."""
        return gradient(pos, psi, rho, parts, lambdas, weights)

    def electronic(states):
        """The states, at ``wk`` -- the weight ``drhodvnl`` contracts with."""
        return gradient(positions, states, zero_density, zero_becsum, ground,
                        state_weights)

    def energy_gradient(pos, states, rho, parts, lambdas):
        """Everything in one call, which an ultrasoft cross term needs."""
        return gradient(pos, states, rho, parts, lambdas, weights)

    # For a norm-conserving dataset there is no ``becsum`` and no multiplier
    # term with any position dependence, so the extra primals are dropped
    # rather than carried at zero -- which keeps that path the two-argument
    # ``jvp`` it was, bit for bit.
    zero_becsum = tuple(None if b is None else jnp.zeros_like(b) for b in becsum)
    zero_density = jnp.zeros_like(jnp.asarray(density))
    if not ultrasoft:
        zero_density = jnp.asarray(density)

        def frozen(pos, rho):  # noqa: F811 -- the norm-conserving signature
            return gradient(pos, psi, rho, becsum, None, weights)

    # ``chosen`` are the atoms perturbed and ``matrix`` has one row per
    # perturbation. The *columns* stay the whole cell: a row is a gradient with
    # respect to every position, and restricting the perturbations does not
    # restrict what they push against.
    chosen = tuple(range(nat)) if atoms is None else tuple(atoms)
    matrix = np.zeros((len(chosen), 3, nat, 3))
    for row, atom in enumerate(chosen):
        for cart in range(3):
            tangent = jnp.zeros_like(positions).at[atom, cart].set(1.0)
            # On the second copy of the bands when they are doubled.
            first = nbnd if doubled else 0
            states = jnp.zeros_like(psi).at[:, :, first:first + nocc].set(
                dpsi[row, cart])
            if ort is not None:
                # The occupied block, which the Sternheimer solve does not
                # produce and the identity this module rests on needs: ``dpsi``
                # in ``d^2E = ... + (d_psi d_j L).dpsi_i`` is the *whole*
                # first-order state, and for an ultrasoft or PAW dataset its
                # occupied part is fixed by the constraint rather than free.
                states = states.at[:, :, first:first + nbnd].add(ort[row, cart])
            if not ultrasoft:
                _, hessian = compiled_jvp(
                    frozen, (positions, density), (tangent, drho[row, cart])
                )
                _, response = compiled_jvp(electronic, (psi,), (states,))
                matrix[row, cart] = np.asarray(hessian + response)
                if on_row is not None:
                    # Streamed as it is solved, so a job that is killed keeps
                    # the rows it paid for. The row is the whole gradient
                    # ``d^2E/du_(a i) du_(b j)`` over ``b``.
                    on_row(atom, cart, matrix[row, cart])
                continue

            # **One ``jvp`` and not two, and that is what an ultrasoft dataset
            # forces.** The metal's weight split (``PLAN.md`` P28) puts the
            # coordinate in one call and the states in another, which is exact
            # only while the two do not meet -- and here they do: ``rho``
            # depends on the positions *and* on the states, so the cross term
            # ``d^2 rho / du dpsi`` exists and a split jvp cannot see it. It is
            # ``addusdynmat``, it is 0.49 Ry/bohr^2 on two-atom ultrasoft
            # silicon against force constants of 0.37, and the acoustic sum
            # rule is what says so. A metal keeps the one ``jvp`` and doubles
            # the bands instead (``doubled`` above, ``notes/incompat/I86.md``).
            raw = compiled_jvp(raw_mixed, (positions, psi), (tangent, states))[1]
            parts = (
                tuple(None if b is None else jnp.zeros_like(b) for b in becsum)
                if dbecsum is None else
                tuple(
                    None if a is None else a - b
                    for a, b in zip(dbecsum[row, cart], raw[1])
                )
            )
            # **The augmentation charge's share of ``parts`` is taken out of the
            # density's correction**, because the density is built *from*
            # ``becsum``: ``parts`` reaches the density through ``augmented`` as
            # well as the one-centre energy, and ``drho`` is already the
            # symmetrised total, augmentation included. Without this the
            # density's tangent is ``Sym(raw) + A(Sym_b db - db)`` and the
            # second term counts the becsum average twice -- zero on a closed
            # ``nosym`` grid and on an ultrasoft dataset (``parts = 0``), so
            # only a PAW wedge sees it: 4.0e-4 Ry/bohr^2 between
            # ``si-epsilon-paw-unshifted`` and the same sample whole, 5.4e-14
            # with it. It is :func:`~defuqumat.response.born.
            # _full_zone_field_response`'s ``through_becsum``, one order up.
            correction = (drho[row, cart] - raw[0]
                          - calculation.augmented(jnp.zeros_like(raw[0]), parts))
            _, whole = compiled_jvp(
                energy_gradient,
                (positions, psi, zero_density, zero_becsum, ground),
                (tangent, states, correction, parts,
                 jnp.zeros_like(ground) if multipliers is None
                 else jnp.zeros_like(ground).at[
                     ..., first:first + nbnd, first:first + nbnd].set(
                         multipliers[row, cart])),
            )
            matrix[row, cart] = np.asarray(whole)
            if on_row is not None:
                on_row(atom, cart, matrix[row, cart])
    return matrix



def _impose_acoustic_sum_rule(matrix: np.ndarray) -> np.ndarray:
    """``sum_b D_(a i)(b j) = 0``: translating the crystal costs nothing.

    QE's ``asr = 'simple'`` in ``q2r``/``matdyn``, and **not** what ``ph.x``
    does by default, which is why this is off unless asked for. The correction
    goes on the diagonal block of each atom, as QE puts it.
    """
    corrected = np.array(matrix, dtype=float)
    for atom in range(matrix.shape[0]):
        corrected[atom, :, atom, :] -= matrix[atom].sum(axis=1)
    return corrected


def _diagonalize(matrix: np.ndarray, masses: np.ndarray):
    """``dyndia``: mass-weight, diagonalise, and report in cm^-1.

    ``amass`` is in amu and the force constants are in Ry/bohr^2, so the
    conversion is :data:`~defuqumat.units.AMU_TO_RY` -- QE's ``amu_ry``, applied
    in exactly the same place. A negative eigenvalue is an imaginary frequency
    and is reported as a negative number rather than as a complex one, which is
    ``dyndia``'s convention and the reason the acoustic modes of a PAW silicon
    run print as ``-6.1``.
    """
    nat = matrix.shape[0]
    flat = np.asarray(matrix, dtype=float).reshape(3 * nat, 3 * nat)
    scale = 1.0 / np.sqrt(np.repeat(masses, 3) * AMU_TO_RY)
    weighted = flat * scale[:, None] * scale[None, :]
    omega2, vectors = np.linalg.eigh(0.5 * (weighted + weighted.T))
    frequencies = np.sign(omega2) * np.sqrt(np.abs(omega2)) * RY_TO_CMM1
    return frequencies, vectors


#: The refusal of a closed subset with symmetry on either route that walks the
#: k axis a chunk at a time (a streamed solve, or a walked response handed in).
_SYMMETRISED_SUBSET_WALKED = (
    "a partial dynamical matrix with symmetry is validated on the "
    "whole-k route only, and this run walks the k axis a chunk at a "
    "time: switch symmetry off (nosym = .true.), or use a memory mode "
    "that holds the whole mesh"
)


def _require_a_subsettable_regime(calculation, atoms, nat):
    """Which atoms to perturb, and what a subset is refused for.

    Returns ``None`` for a whole-cell run -- the path every committed case
    takes, left bit for bit as it was -- and a validated tuple of indices
    otherwise.

    What is refused is a symmetrised run whose subset is **not a union of
    orbits**. The wedge was reduced by the whole group, so completing the wedge
    sum for ``drho_{a,i}`` reads ``drho_{S^-1(a),j}`` for every operation ``S``
    of it (``symdvscf``); that is available exactly when every operation maps
    the subset into itself, and then every symmetriser here acts on the partial
    stack through :func:`~defuqumat.system.symmetry.subset_atom_mapping`.
    Averaging over the subset's own stabiliser instead would be an average over
    a group the k-set was not reduced by, and wrong rather than slow.
    """
    if atoms is None:
        return None
    chosen = tuple(int(a) for a in atoms)
    if not chosen:
        raise ValueError("atoms is empty: there is nothing to displace")
    if len(set(chosen)) != len(chosen):
        raise ValueError(f"atoms repeats an index: {chosen}")
    if any(a < 0 or a >= nat for a in chosen):
        raise ValueError(
            f"atoms indexes outside the structure's {nat} atoms: {chosen}"
        )
    if calculation.use_symmetry:
        mapping = atom_mapping(calculation.system.cell,
                               calculation.system.structure,
                               calculation.symmetries)
        closure = orbit_closure(mapping, chosen)
        if set(closure) != set(chosen):
            raise NotImplementedError(
                f"a partial dynamical matrix with symmetry needs a subset that "
                f"is a union of orbits, and atoms={chosen} is not: symdvscf "
                f"symmetrises the responses together, and an operation rotates "
                f"the cartesian direction and carries the atom to atoms "
                f"{tuple(sorted(set(closure) - set(chosen)))}, outside the "
                f"subset. Ask for atoms={closure}, the smallest closed subset "
                f"containing it, or switch symmetry off (nosym = .true.)"
            )
    return chosen


def _require_one_spin_channel(calculation) -> None:
    """``nspin = 2`` on the route that walks the k axis a chunk at a time.

    **The whole-k route is spin-polarized now, and it is measured against
    ``ph.x``.** On triplet O2 (``tests/data/qe/o2-fixed-lsda.in``, ultrasoft,
    fixed occupations at ``tot_magnetization = 2``, counts ``(7, 5)``) every
    element of the force constants agrees with
    ``tests/data/qe/reference.dyn.o2-fixed-lsda-phonon`` to 2.6e-4 Ry/bohr^2,
    and the transverse block to 4.0e-5, at ``ph.x``'s ``tr2 = 1e-14``
    (``notes/incompat/I2.md``). Two earlier measurements stand beside it: a
    nonmagnetic cell run as ``nspin = 1`` and as ``nspin = 2`` gives the same
    matrix to 1.6e-14 Ry/bohr^2 (unshifted ``nosym`` silicon), and one column of
    the antiferromagnetic hydrogen chain, a smeared metal, agrees with a finite
    difference of the forces to 3.6e-7 on a column of 6.0e-3.

    **What was missing was not a spin axis of the assembly** but each
    channel's own occupied block. The solver keeps ``max(occupied_counts)``
    bands in every channel, so the down channel of O2 carried its two empty
    pi*_g bands inside ``S'_mn``, and they entered :func:`orthogonality_states`
    and :func:`multiplier_response` as occupied partners: ``D_xx(1,1)`` read
    -0.1288 against ``ph.x``'s -0.0102. :func:`occupied_pairs` is the fix. The
    earlier reading of this docstring, that the ultrasoft terms could not be
    what blocked an LSDA run, was argued for a norm-conserving cell while the
    cell measured was ultrasoft.

    **What stays refused is the walked route** -- a streamed store, memory mode
    with a chunk smaller than the mesh, and memory mode on a card whatever the
    chunk (:func:`~defuqumat.response.efield._streams`).
    :mod:`~defuqumat.response.chunked_phonon` carries the same mask, but it has
    not been run at ``nspin = 2``; the check it needs is no new reference, only
    the walked matrix against the whole-k one on this same cell.
    """
    if calculation.nspin == 2:
        raise NotImplementedError(
            "the dynamical matrix of a spin-polarized calculation is implemented "
            "on the whole-k route only (checked against ph.x on triplet O2); "
            "this run walks the k axis a chunk at a time -- a streamed store, "
            "memory mode with a chunk smaller than the mesh, or memory mode on a "
            "card -- and that route has not been run at nspin = 2. The whole-k "
            "route is memory_mode='speed' with the states held whole rather "
            "than streamed"
        )


def _require_no_spinor(calculation) -> None:
    """A spinor run: the assembly is written and measured, and still refused.

    **What a nonmagnetic spinor needed was three terms of this module's own
    assembly, not a new response**, and they are in (``notes/incompat/I3.md``).
    The Sternheimer solve has had a spinor form since P81 and the augmented
    spinor since P98; what this caller added is the ``spinors=True`` opt-in on
    :func:`~defuqumat.forces.energy.frozen_energy` in :func:`_force_constants`
    (the functional has had its spinor branch since P46), ``qq_so`` in
    :func:`overlap_derivatives`, the one contraction here written for an
    ``npwx``-long band, and the generic guard's ``noncollinear`` opt-in. The
    rest -- the bare perturbation through ``at_positions``, the multipliers'
    response, ``_raw_mixed_state`` and the screening, whose stop is already
    ``tr2 / npol`` as ``dfpt_kernels.f90:435`` has it -- is shape-agnostic or
    was made spinor-aware by the Born charges.

    **The measurements, with this guard bypassed** (D22, 2026-10-16, one run
    each, ``tests/regression/test_spinor_phonons.py``):

    * the same silicon as a scalar and as a spinor with no moment gives the same
      force constants to **2.7e-10** (norm-conserving), **1.2e-9** (ultrasoft)
      and **1.7e-9** Ry/bohr^2 (PAW), on force constants of 0.28;
    * fully-relativistic ultrasoft AlAs: the optical triplet at **376.717753
      cm^-1 against ``ph.x``'s 376.714126** (376.715282 at ``ph.x``'s own
      ``tr2 = 1e-14``, 1.2e-3 like-for-like). On this cell spin-orbit coupling
      is worth only 0.073 cm^-1 in ``ph.x``, so the ``fcoef`` dressing is below
      what it resolves, and the identities above cannot see it at all
      (``fcoef = 1`` for a scalar-relativistic dataset);
    * bismuthene (``bismuthene-epsilon-us-soc.in``), the one cell where the
      dressing is resolvable, **does not agree**: at ``tr2 = 1e-14`` the
      ``B_1g`` mode is 69.308519 against ``ph.x``'s 66.727386 and ``E_2g``
      117.145263 against 117.891803 cm^-1, the acoustic modes +16.2, +16.2,
      +18.0 against ``ph.x``'s +12.3, +12.3, -15.3. The loop itself is not
      converged there: at ``tr2 = 1e-16`` ``E_2g`` moves to 118.212126, by
      1.07, where AlAs moves by 2.5e-3. A central difference of this code's own
      forces (``h = 0.01`` bohr, atom 0 along ``z``) gives the inter-atomic
      constant ``D(0z, 1z)`` to **0.5 per cent** (-0.03625 against -0.03644)
      and the on-site ``D(0z, 0z)`` **11 per cent** off (0.04438 against
      0.03982). So the error sits in the self (acoustic) block, as in
      ``OPEN.md`` Part XXVII, on a cell with vacuum, a GGA and a core charge on
      a heavy atom -- which does not point at the spin axis and does not clear
      it either, since no scalar cell of that kind has been put through the
      same comparison.

    That is why the refusal stays, in the form :func:`_require_one_spin_channel`
    keeps for ``nspin = 2``. What would lift it, in order: bismuthene's DFPT at
    ``tr2 = 1e-18`` and below against the same finite-difference row (if
    ``D(0z, 0z)`` walks to 0.0444 it is the loop); the same comparison on a
    scalar ultrasoft insulator with vacuum, a GGA and a core charge (whether
    the 11 per cent is the spinor's at all); ``ph.x`` on bismuthene at a tighter
    ``tr2_ph``.

    **A magnetized spinor is refused for a separate reason**: the induced
    potential is then ``dv_0 I + dm . sigma`` through a 4x4 kernel, and the
    dielectric tensor of the one textured cell measured (``i-atom-soc.in``) sits
    5.3 per cent from ``ph.x`` along the moment for an unlocated reason
    (:func:`~defuqumat.response.efield.require_a_measured_spinor_response`).

    **What I79 measured for it** (D22, 2026-10-16, one run each, this guard
    bypassed, ``notes/incompat/I79.md``). Nothing in the assembly is new for a
    moment: the second solve ``ph.x`` runs under ``domag`` (``nsolv = 2``,
    ``dfpt_kernels.f90:212-213``) supplies the ``dpsi^* psi`` half that its
    density formula leaves to the time-reversed partner, and the density
    response here is a ``jvp``, ``2 Re`` of both halves already. The
    displacement average is axial on the grid and in ``becsum``
    (:meth:`~defuqumat.scf.driver.Calculation.symmetrize_atom_displacement`).
    The collinear limit is then an identity, and it is close but not exact:

    * triplet O2 (ultrasoft, the moment along ``z``, no spin-orbit coupling)
      against its ``nspin = 2`` run: **8.3e-6** Ry/bohr^2 on force constants of
      1.68, all of it in the transverse block (entries 0.010 to 0.019); the
      symmetrised spinor against the ``nosym`` one 2.3e-8;
    * the antiferromagnetic hydrogen chain (a norm-conserving metal):
      **1.9e-6** on 0.038;
    * the same, solved on the collinear ground state written as a spinor, so
      that the two runs share a fixed point: **2.3e-6** and **1.4e-6**.

    So about 1e-6 belongs to the spinor response route itself, four orders
    above the 2.7e-10 a nonmagnetic spinor reaches against the scalar run, and
    it is unlocated. It is below ``ph.x``'s own distance on O2 (4.0e-5), and the
    identity cannot see the transverse kernel or spin-orbit coupling at all,
    which is where the 5.3 per cent above sits.
    """
    if not calculation.noncolin:
        return
    if calculation.nspin_mag == 4:
        raise NotImplementedError(
            "the dynamical matrix of a noncollinear run with a magnetization "
            "(nspin_mag = 4, a textured spinor) is not implemented: the induced "
            "potential carries dm . sigma through the 4x4 exchange-correlation "
            "kernel, which the dielectric tensor of a textured spin-orbit "
            "insulator puts 5.3 per cent from ph.x for a reason not yet found; "
            "in the collinear limit, triplet O2 as a spinor against its "
            "nspin = 2 run, the force constants agree to 8.3e-6 Ry/bohr^2 and "
            "not to round-off, and 2.3e-6 of that remains on a shared ground "
            "state, which is unlocated (notes/incompat/I79.md)"
        )
    raise NotImplementedError(
        "the dynamical matrix of a noncollinear or spin-orbit run is not "
        "implemented, although its assembly is written and measured: a spinor "
        "without a moment reproduces the scalar run's force constants to "
        "1.7e-9 Ry/bohr^2 and ph.x's optical triplet of spin-orbit AlAs to "
        "1.2e-3 cm^-1, but on spin-orbit bismuthene the B_1g and E_2g modes are "
        "2.6 and 0.75 cm^-1 from ph.x and the on-site force constant 11 per "
        "cent from a finite difference of this code's own forces, which is "
        "unlocated (notes/incompat/I3.md)"
    )


def _require_a_moving_overlap_regime(calculation) -> None:
    """What is still refused once ``S`` is allowed to move with the atoms.

    ``PLAN.md`` P39. The norm-conserving restriction P25 introduced is gone --
    the two terms it named are implemented
    (:func:`orthogonality_states` for the multipliers' own motion,
    :func:`non_variational_response` for ``Q_ij(r - tau)`` at frozen
    ``becsum``) -- and what is left is one combination rather than one dataset.

    **An ultrasoft or PAW metal is refused.** The weight split
    :func:`_state_weights` makes is between a state tangent at ``wk`` and
    everything else at ``wg``, and it was derived for a response whose whole
    ``becsum`` dependence goes through ``dpsi``. With ``S`` moving there are
    three further tangents -- ``dpsi^ort``, ``becsumort`` and ``dLambda`` --
    and which weight each belongs with is a question the insulating case cannot
    answer, because there the two weights are equal. Refused rather than
    guessed: the symptom would be an acoustic sum rule that is violated by a
    plausible amount, and P28 is the record of how long that took to find once.

    **The dynamical matrix no longer calls this** (``notes/incompat/I86.md``):
    the weights are :func:`pair_weights` and the band doubling in
    :func:`_force_constants`, measured against ``ph.x``, and what it still
    refuses is :func:`_require_a_measured_overlap_metal`. This stays for the
    third derivative (:func:`~defuqumat.response.nonlinear.raman_tensors`),
    whose own Sternheimer guard refuses every metal before it is reached.
    """
    if calculation.is_ultrasoft and calculation.system.occupations != "fixed":
        raise NotImplementedError(
            "the third derivative (Raman tensor) of a *metal* with an ultrasoft "
            "or PAW pseudopotential is not implemented: the Gamma dynamical "
            "matrix of such a metal is (notes/incompat/I86.md), and the third "
            "derivative of a metal is not, on any pseudopotential kind"
        )


def _overlap_metal(calculation) -> bool:
    return bool(calculation.is_ultrasoft
                and calculation.system.occupations != "fixed")


def _require_a_measured_overlap_metal(calculation, atoms=None) -> None:
    """What is still refused of an ultrasoft or PAW metal's ``Gamma`` phonon.

    ``notes/incompat/I86.md``. The three tangents a moving ``S`` adds carry
    ``compute_weight``'s pair weight in a metal (:func:`pair_weights`), and the
    assembly doubles the bands so that the state tangent sees ``wk`` while the
    coordinate sees ``wg`` (:func:`_force_constants`). Measured against
    ``ph.x`` on ``al2-metal-us.in`` and ``al2-metal-paw.in``: one spin
    channel, ``nosym``, the whole cell. The regimes beside that are refused
    because they are not measured, not because a term is known to be missing.
    """
    if not _overlap_metal(calculation):
        return
    if calculation.nspin != 1:
        raise NotImplementedError(
            "the dynamical matrix of a spin-polarized metal with an ultrasoft "
            "or PAW pseudopotential is not implemented: the pair weight of the "
            "orthogonality block and of the multipliers (notes/incompat/I86.md) "
            "is measured against ph.x at nspin = 1 only. A norm-conserving "
            "spin-polarized metal, and an unpolarized ultrasoft or PAW metal, "
            "are implemented"
        )
    if calculation.use_symmetry:
        raise NotImplementedError(
            "the dynamical matrix of a metal with an ultrasoft or PAW "
            "pseudopotential is implemented on a nosym run only: it is measured "
            "against ph.x on al2-metal-us.in (nosym, notes/incompat/I86.md), "
            "and the symmetrisation of its pair-weighted orthogonality block on "
            "a wedge is not. Run with nosym = .true. on an unshifted grid"
        )
    if atoms is not None:
        raise NotImplementedError(
            "a partial dynamical matrix (atoms=) of a metal with an ultrasoft or "
            "PAW pseudopotential is not implemented: the whole cell is measured "
            "against ph.x (notes/incompat/I86.md) and the subset is not"
        )


def _require_no_walked_overlap_metal(calculation) -> None:
    """The k-chunked route has its own copies of the tangents, without the
    metal's pair weight (:mod:`defuqumat.response.chunked_phonon`)."""
    if _overlap_metal(calculation):
        raise NotImplementedError(
            "the dynamical matrix of a metal with an ultrasoft or PAW "
            "pseudopotential walked a k-chunk at a time is not implemented: "
            "the chunked route (response/chunked_phonon.py) builds the "
            "orthogonality block and the multipliers without the metal's pair "
            "weight, which the whole-k route carries (notes/incompat/I86.md). "
            "Run with memory_mode='speed' or a k_batch that holds the whole mesh"
        )
