"""Topological invariants of a converged calculation.

The shape of the other fixed-density workflows (:mod:`defuqumat.workflows.bands`,
:mod:`defuqumat.workflows.dos`): an SCF run produces a density, the potential
built from it is frozen, and the Hamiltonian is diagonalised wherever the
quantity asks for it. What is different here is *where* it asks, and how often:
a Chern number wants a closed mesh over a plane, a Wilson loop wants a whole
half-zone one loop at a time, and the Fu-Kane parity criterion wants four or
eight points and nothing else.

    ``run_berry_curvature``   Omega(k) and the Chern number on a plane
    ``run_z2``                the 2D Z2 of one plane
    ``run_z2_3d``             the four indices (nu0; nu1 nu2 nu3)

**Spin-orbit coupling is a precondition, not an option.** Without it every band
is doubly degenerate in spin, the two copies carry opposite Chern numbers, the
Wannier centres cross in pairs and every Z2 is trivially zero -- a number that
is correct, uninformative, and indistinguishable from a real answer. So a Z2 run
on a calculation with ``noncolin = .false.`` is refused. A Chern number is a
different matter: it needs *broken* time reversal, which this code can only
reach through a magnetization, so a nonmagnetic run's Chern number is zero by
symmetry and computing it is a check of the machinery rather than physics.

**Memory** (the standing rule, and this is the workflow where it bites). The
states of one loop are ``nloop * nocc * npol * npwx * 16`` bytes and are dropped
before the next loop is built, so ``npump`` costs time and not space. On the
committed bismuthene reference -- ``npwx = 2705``, ``npol = 2``, ``nocc = 30`` --
that is 2.6 MB per k-point and 63 MB for a 24-point loop. What dominates instead
is the setup each call rebuilds: the augmentation charge ``Q_ij(G)`` on the
dense grid is ``nh^2 * ngm`` complex, 1.1 GB for that cell, which is why the
source holds **one** :class:`~defuqumat.scf.driver.Calculation` at a time and why
a plane mesh is walked rather than materialised.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from defuqumat.pseudo.upf import Pseudopotential
from defuqumat.scf.driver import Calculation, default_nbnd
from defuqumat.workflows.nscf import fixed_density_ethr
from defuqumat.system.builder import System
from defuqumat.system.kpoints import KPoints, for_spin
from defuqumat.topology.berry import BerryCurvature
from defuqumat.topology.invariants import chern_number as _chern_number
from defuqumat.topology.invariants import z2_invariant, z2_invariant_3d
from defuqumat.topology.parity import inversion_centre
from defuqumat.topology.states import build_plane_wave_states

__all__ = ["DFTSource", "run_berry_curvature", "run_z2", "run_z2_3d"]


@dataclass
class DFTSource:
    """A state source: Kohn-Sham states anywhere in the zone, density fixed.

    Each call diagonalises at the k-points asked for, in the potential frozen
    from ``density``. The setup those k-points do *not* affect -- both G-vector
    sets, the local potential, the augmentation charge, the Ewald sum, the
    symmetry group, the radial tables -- is built once and shared between calls
    through :meth:`~defuqumat.scf.driver.Calculation.at_kpoints`; only the
    plane-wave spheres, ``|k+G|^2``, the stick layout and ``vkb(k)`` are rebuilt.
    The frozen potential itself, ``newd``'s ``D_ij`` and the smooth-grid copy
    of the potential (:meth:`~defuqumat.scf.driver.Calculation.local_terms`) are
    built on the first call and kept, as PAW's one-centre coefficients are, and
    so is the augmentation charge between two k-points, ``q_ij(b)``, which is
    cached on the shared setup rather than on each call's copy.

    That is what makes streaming the cheap option rather than the expensive one.
    Rebuilding a whole ``Calculation`` per call cost ~1 GB and seconds each time,
    which on any mesh worth taking is more than the states of the entire mesh
    cost to hold -- so the caller was forced to choose between a streaming loop
    that was ruinous in time and a resident mesh that was ruinous in memory.
    Sharing the k-independent setup removes the choice: streaming now costs one
    diagonalisation per call and nothing else.

    ``nocc`` is the number of occupied bands and is **explicit**: every quantity
    here is a property of a gapped manifold, so which bands are in it has to be
    stated rather than inferred from an occupation that a smearing has blurred.
    ``nbnd`` adds empty bands for the eigensolver's benefit only.
    """

    system: System
    pseudos: tuple[Pseudopotential, ...]
    density: jnp.ndarray
    nocc: int
    #: The converged ``becsum`` (``SCFResult.becsum``), needed by a PAW dataset
    #: and ignored by any other. A PAW Hamiltonian's nonlocal coefficients are
    #: ``D^(0) + int V Q + ddd_paw``, and only the first two can be rebuilt from
    #: the density: ``ddd_paw`` comes from ``becsum``, which is a property of the
    #: *wavefunctions*. Leaving it out is the same tenths-of-an-eV error
    #: :func:`~defuqumat.workflows.nscf.fixed_density_states` refuses, and here it
    #: would move the invariant only by moving a band across the manifold's gap
    #: -- silently, since the integer would still be an integer.
    becsum: tuple = ()
    #: The converged Hubbard occupation matrix (``SCFResult.ns``), needed for
    #: the same reason ``becsum`` is: ``v_ns`` is built from it and it is a
    #: property of the *wavefunctions*, so it cannot be rebuilt from the
    #: density. Without it the manifold is diagonalised on the **un-Hubbard**
    #: Hamiltonian -- and then :meth:`_check_gap` certifies it isolated using
    #: those eigenvalues, so the invariant is a confident integer taken off the
    #: wrong bands.
    ns: jnp.ndarray | None = None
    #: The converged kinetic energy density (``SCFResult.tau``), needed under a
    #: potential-only meta-GGA (``tb09``, ``bj06``) and ignored otherwise. The
    #: Becke-Roussel potential and the Tran-Blaha ``c`` are functions of
    #: ``tau``, which is a property of the occupied states over the whole zone
    #: and cannot be rebuilt from the density -- the argument
    #: :func:`~defuqumat.workflows.nscf.fixed_density_states` makes for a band
    #: path. With it the Hamiltonian is an ordinary local one, so nothing past
    #: the potential knows the functional is a meta-GGA.
    tau: jnp.ndarray | None = None
    #: ``SCFResult.magnetic_field`` and ``SCFResult.field_scale``, needed for
    #: the same reason ``becsum`` and ``ns`` are: what the run *converged*
    #: under is not what the input asked for. Elk's ``reducebf`` scales a
    #: symmetry-breaking field down as the SCF runs, and the fixed-spin-moment
    #: scheme drives its field from the moment's error, so rebuilding the field
    #: from the input re-applies one the SCF had already changed -- a rigid
    #: Zeeman shift of every eigenvalue, which is invisible in an invariant
    #: because the answer is still an integer.
    #: :func:`~defuqumat.workflows.nscf.fixed_density_states` has refused this
    #: since the 2026-08-29 sweep and this class did not, which made it the same
    #: defect one layer over.
    field: object = None
    field_scale: float | None = None
    nbnd: int | None = None
    conv_thr: float = 1.0e-8
    k_batch: int | None | str = "default"
    #: Smallest direct gap above the manifold, in Ry, that is still trusted.
    #: A Wilson loop or a parity product taken across a gap the mesh cannot see
    #: returns a confident integer that is simply wrong -- the failure mode the
    #: reference implementation records twice (``elkpy``: graphene with a narrow
    #: anticrossing, bulk Bi2Se3 on a coarse loop). Checking it costs nothing:
    #: the eigenvalues are already computed, since the eigensolver is asked for
    #: empty bands anyway. It is also what makes the Kubo route's own
    #: degeneracy step inert on the manifold total: no occupied/empty pair
    #: below it reaches the sum (``PLAN.md`` P149).
    gap_tol: float = 1.0e-4
    #: The caller's own :class:`~defuqumat.scf.driver.Calculation` of this
    #: system, used as :meth:`_base` rather than a second one built beside it
    #: -- it shares the augmentation and PAW tables and keeps the caller's
    #: memory mode (``notes/GPU-MEMORY-NEXT.md`` item 20). ``None`` builds one.
    calculation: object = None
    #: The collinear channel, 0 (up) or 1 (down), at ``nspin = 2`` and
    #: nowhere else. A collinear spin-polarized run has one Hamiltonian per
    #: channel, so one occupied manifold per channel and one set of invariants
    #: per channel; this source serves one of them, and ``nocc`` is that
    #: channel's count. Without spin-orbit coupling each channel's Hamiltonian
    #: is real in real space, so ``H_s(-k) = H_s(k)*`` and the channel's own
    #: Chern number is zero by symmetry in every collinear case, magnetic or
    #: not; what a channel carries is its pointwise curvature, odd in ``k``.
    spin: int | None = None

    def __post_init__(self):
        if self.system.nspin == 2:
            if self.spin is None:
                raise NotImplementedError(
                    "a collinear spin-polarized calculation has two independent "
                    "Hamiltonians and therefore two independent sets of "
                    "invariants, one per channel: pass spin = 0 or spin = 1 "
                    "(with that channel's nocc), or run the noncollinear "
                    "spinor path, where there is one Hamiltonian"
                )
            if self.spin not in (0, 1):
                raise ValueError(
                    f"spin = {self.spin}: a collinear channel is 0 or 1 (up or "
                    "down)"
                )
        elif self.spin is not None:
            raise ValueError(
                f"spin = {self.spin} names a collinear channel, which exists "
                f"only at nspin = 2; this calculation has nspin = "
                f"{self.system.nspin}"
            )

    @property
    def channel(self) -> int:
        """The index into the channel axis :meth:`states` diagonalises: ``spin``
        at ``nspin = 2``, and the one channel there is otherwise."""
        return 0 if self.spin is None else int(self.spin)

    def _base(self) -> Calculation:
        """The k-independent setup, built once and kept.

        Cached on the instance rather than in ``__post_init__`` so that
        constructing a source stays free, and so that a source that is never
        asked for states never pays for one. It is built at the system's own
        k-points -- whatever they are, they are replaced by
        :meth:`~defuqumat.scf.driver.Calculation.at_kpoints` before anything
        k-dependent is used.
        """
        if getattr(self, "_calculation", None) is None:
            object.__setattr__(
                self, "_calculation",
                self.calculation if self.calculation is not None
                else Calculation(self.system, self.pseudos, k_batch=self.k_batch),
            )
            from defuqumat.forces.energy import reject_single_band_side

            # Every invariant, the Berry-phase polarization, the Kubo
            # conductivities and the orbital magnetization come through here;
            # each is a phase or a sum over a mesh that the mode is not
            # checked for (I119).
            reject_single_band_side(
                self._calculation,
                "a quantity built from fixed-density states (a Berry phase, a Chern "
                "or Z2 invariant, a Kubo conductivity, an orbital magnetization)")
        return self._calculation

    @property
    def streams(self) -> bool:
        """Whether this source's calculation keeps its states streamed.

        True where ``wfc_store`` resolves to ``stream`` through the
        calculation's memory mode -- memory mode on a card -- and the default
        for :func:`~defuqumat.topology.invariants.chern_number` and the orbital
        magnetization to walk their meshes a column or a plane at a time
        (``notes/GPU-MEMORY-NEXT.md`` item 5). False on a CPU, where the store is on
        the one memory there is and a column's own compiled solve is the cost.
        """
        from defuqumat.batching import resolve_wfc_store

        return resolve_wfc_store("default", self._base().memory_mode) == "stream"

    def _ddd_paw(self, meta_c=None):
        """PAW's one-centre coefficients, built once and kept.

        A function of ``becsum`` alone, so it is k-independent and belongs with
        the rest of the shared setup rather than inside the per-call
        diagonalisation: a streaming Wilson loop calls :meth:`states` once per
        k-point, and rebuilding the radial one-centre terms there would put a
        Gauss-Legendre quadrature over every sphere in the inner loop.

        ``meta_c`` is the potential's cell-averaged Tran-Blaha coefficient,
        which the spheres cannot compute for themselves and which
        :meth:`~defuqumat.scf.driver.Calculation.onecenter` refuses to default
        under a meta-GGA; it is passed on the first call, which is the one that
        builds the frozen potential, and the cached value is what every later
        call returns.
        """
        if not self.becsum:
            return None
        if getattr(self, "_ddd", None) is None:
            _, ddd = self._base().onecenter(self.becsum, meta_c)
            object.__setattr__(self, "_ddd", ddd)
        return self._ddd

    def states(self, points, keep_projectors: bool = False,
               keep_velocity: bool = False, keep_hamiltonian: bool = False):
        """Occupied states at the given crystal k-points, in the given order.

        ``keep_hamiltonian`` keeps the fixed-density Hamiltonian itself, which
        an orbital magnetization applies between two k-derivatives. It costs
        nothing -- the object was built to diagonalise with and is otherwise
        dropped.

        ``keep_velocity`` additionally builds the
        :class:`~defuqumat.response.velocity.VelocityOperator` of *this*
        fixed-density Hamiltonian and keeps the whole diagonalised band set --
        what the ``kubo`` Berry curvature is a sum over. It costs the empty
        bands' coefficients (the manifold alone is what everything else here
        needs) and nothing else: the operator holds the frozen potential and
        the shared setup by reference and does no work until it is applied.
        """
        if keep_velocity and self.system.nspin == 2:
            # ``VelocityOperator`` builds every channel's ``dH/dk`` and the
            # Kubo sum (``topology/kubo.py``) hands it one channel's states as
            # ``psi[None]`` and reads channel 0 back, so at ``nspin = 2`` it
            # would pair the down states with the up Hamiltonian's velocity.
            raise NotImplementedError(
                "the kubo Berry curvature of one collinear channel is not "
                "implemented: the velocity operator is built for every channel "
                "and the sum reads the first. Use method = 'fhs', which is "
                "per channel"
            )
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        # Not passed through ``for_spin``: the weights of this hand-built set
        # are never summed over (fixed density, no occupations, every quantity
        # here a per-k overlap), so the ``nspin = 2`` weight convention does
        # not reach anything.
        kpoints = KPoints.from_crystal(
            points,
            np.full(len(points), 1.0 / len(points)),
            self.system.cell,
            precision=self.system.kpoints.precision,
        )
        if self.spin is not None:
            # ``for_spin`` at the boundary of a polarized run's caller-built
            # k-set, as the trap list asks. Nothing a channel source returns
            # reads the weights -- an overlap, a phase and a gap do not -- so
            # this is for whatever is built on the set later. The other
            # regimes keep the set they had: changing a spinor's weights here
            # would move every invariant route that does not need it to.
            kpoints = for_spin(kpoints, self.system.nspin)
        calculation = self._base().at_kpoints(kpoints)
        system = calculation.system
        if calculation.is_paw and not self.becsum:
            # The same refusal ``workflows.nscf`` makes, and for the same
            # reason: ``ddd_paw`` is a function of ``becsum`` and ``becsum`` is
            # a property of the wavefunctions, so it cannot be rebuilt from the
            # density this source is handed. It is passed rather than derived,
            # and the refusal now only catches *not* passing it.
            raise NotImplementedError(
                "a fixed-density run with a PAW pseudopotential needs the "
                "converged becsum as well as the density: pass "
                "becsum = scf_result.becsum. It cannot be rebuilt from the "
                "density, and leaving it out is wrong by tenths of an eV"
            )
        hubbard_terms = None
        if calculation.is_hubbard:
            if self.ns is None:
                # ``workflows.nscf``'s refusal again, and it had no counterpart
                # here at all: ``hamiltonian`` was called with two positional
                # arguments into a signature whose third is ``hubbard=None``,
                # so a DFT+U crystal was diagonalised without its Hubbard term
                # and the invariant came off eigenvalues wrong by the whole
                # Hubbard shift.
                raise ValueError(
                    "a topological invariant of a DFT+U calculation needs the "
                    "converged occupation matrix as well as the density: pass "
                    "ns = scf_result.ns. It cannot be rebuilt from the density, "
                    "and leaving the term out gives eigenvalues that look "
                    "plausible, are wrong by the whole Hubbard shift, and still "
                    "produce an integer"
                )
            _, _, hubbard_terms = calculation.hubbard_terms(jnp.asarray(self.ns))
        if calculation.functional.is_meta and self.tau is None:
            # ``fixed_density_states``' refusal, for the same reason: ``tau``
            # is a property of the occupied states over the whole zone, so it
            # crosses with the density the way ``ns`` and ``becsum`` do. The
            # invariant itself needs no energy -- it is a property of the
            # manifold of a local Hamiltonian -- so once ``tau`` is here the
            # functional being a potential rather than a derivative changes
            # nothing downstream (``notes/incompat/I13.md``: the states
            # reproduce the SCF's eigenvalues).
            raise NotImplementedError(
                f"a topological invariant under {calculation.functional.name} "
                "needs the converged kinetic energy density as well as the "
                "density: pass tau = scf_result.tau. It is a property of the "
                "occupied states over the whole zone and cannot be rebuilt "
                "from the density"
            )
        nbnd = self.nbnd or max(
            self.nocc + 2,
            default_nbnd(
                calculation.nelec, system.occupations, None, None,
                noncolin=system.noncolin,
            ),
        )
        if calculation.magnetic_field is not None and self.field is None:
            # ``fixed_density_states``' refusal, word for word, because it is
            # the same mistake: the converged field is state, not input.
            raise ValueError(
                "a fixed-density run of a calculation with a magnetic field or "
                "a constrained moment needs the field the SCF ended with: pass "
                "field = scf_result.magnetic_field and field_scale = "
                "scf_result.field_scale. Rebuilding it from the input "
                "re-applies a field that reducebf or the fixed-spin-moment "
                "scheme had already changed, which shifts every eigenvalue and "
                "still leaves an invariant looking like an integer"
            )
        # The frozen potential and the k-independent half of the Hamiltonian
        # (``newd``'s ``D_ij`` and the smooth-grid potential), built on the
        # first call and kept, as ``_ddd_paw`` is: a polarization calls this
        # once per string and a streamed Wilson loop once per pumping step, and
        # none of the three depends on the k-points (``OPEN.md`` Part XXIII
        # item 12). Only what carries a ``k`` index is built per call. They are
        # built on the first call's copy rather than on ``_base()``, because
        # the band side's dtype is read off the projector core and
        # ``at_kpoints`` rebuilds that in the cell's precision: every copy
        # agrees with every other, and a single-precision base would not.
        if getattr(self, "_frozen", None) is None:
            potential = calculation.potential(
                self.density,
                1.0 if self.field_scale is None else float(self.field_scale),
                self.field,
                tau=self.tau,
            )
            meta_c = None if self.tau is None else potential.meta_c
            object.__setattr__(self, "_frozen", (
                potential.v_scf,
                calculation.local_terms(potential.v_scf, self._ddd_paw(meta_c)),
            ))
        v_scf, terms = self._frozen
        # One channel's Hamiltonian, diagonalised alone: at ``nspin = 2``
        # ``hamiltonian_from`` builds one per channel from ``terms.potentials
        # [spin]`` and ``deeq[spin]``, and ``diagonalize`` walks whatever tuple
        # it is given, so the result's channel axis has length one either way.
        hamiltonians = calculation.hamiltonian_from(terms, hubbard_terms)
        if self.spin is not None and keep_velocity:
            raise NotImplementedError(
                "a velocity operator of one channel of an nspin = 2 run is "
                "not implemented: VelocityOperator carries the whole "
                "calculation's channels, and no consumer of a channel "
                "source needs it"
            )
        hamiltonians = (hamiltonians[self.channel],)
        ethr = fixed_density_ethr(self.conv_thr, calculation.nelec)
        eigenvalues, wavefunctions = calculation.diagonalize(
            hamiltonians, nbnd, None, ethr
        )
        self._check_gap(np.asarray(eigenvalues[0]), points)
        velocity = None
        if keep_velocity:
            from defuqumat.response.velocity import VelocityOperator

            # The same three things ``calculation.hamiltonian`` was just given,
            # because ``dH/dk`` is the derivative of *that* Hamiltonian and of
            # no other: the frozen potential, PAW's one-centre coefficients and
            # the Hubbard occupation matrix.
            velocity = VelocityOperator(
                calculation, v_scf, self._ddd_paw(), self.ns
            )
        return build_plane_wave_states(
            calculation,
            wavefunctions[0],
            nbnd=self.nocc,
            keep_projectors=keep_projectors,
            ethr=ethr,
            energies=eigenvalues[0],
            velocity=velocity,
            hamiltonian=hamiltonians[0] if keep_hamiltonian else None,
            setup=self._base(),
        )

    def _check_gap(self, eigenvalues: np.ndarray, points: np.ndarray) -> None:
        """Refuse a manifold that is not separated from the bands above it.

        ``(nk, nbnd)`` eigenvalues; the manifold is the lowest ``nocc``. Every
        invariant here is a property of an *isolated* set of bands, and the one
        way to get a wrong integer that still looks like an integer is to take
        it across a touching. Since the eigensolver has already produced the
        first empty band, this costs a subtraction.
        """
        if eigenvalues.shape[1] <= self.nocc:
            return
        gaps = eigenvalues[:, self.nocc] - eigenvalues[:, self.nocc - 1]
        worst = int(np.argmin(gaps))
        if gaps[worst] < self.gap_tol:
            raise ValueError(
                f"the occupied manifold is not isolated: bands {self.nocc} and "
                f"{self.nocc + 1} are {gaps[worst]:.3e} Ry apart at k = "
                f"{np.round(points[worst], 4).tolist()} (crystal), below the "
                f"gap_tol of {self.gap_tol:.1e} Ry. A topological invariant of a "
                "manifold that touches the one above it is not defined; choose "
                "a different band count, or raise gap_tol if the touching is "
                "known to be avoided"
            )


def _source(system, pseudos, density, nocc, nbnd, conv_thr, k_batch,
            becsum=(), ns=None, field=None, field_scale=None,
            calculation=None, spin=None, tau=None) -> DFTSource:
    if nocc is None and not (system.nspin == 2 and spin is None):
        # (with no channel named the source's own refusal is the one to see)
        nocc = _occupied_bands(system, pseudos, density, spin)
    return DFTSource(
        system=system,
        pseudos=tuple(pseudos),
        density=density,
        nocc=0 if nocc is None else int(nocc),
        becsum=tuple(becsum or ()),
        ns=ns,
        tau=tau,
        field=field,
        field_scale=field_scale,
        nbnd=nbnd,
        conv_thr=conv_thr,
        k_batch=k_batch,
        calculation=calculation,
        spin=spin,
    )


def _refuse_potential_only_meta(source: DFTSource, what: str) -> None:
    """The refusal :meth:`DFTSource.states` made for every caller until I13.

    The invariants now take ``tau`` and are checked against the SCF's own
    eigenvalues (``notes/incompat/I13.md``); the Berry-phase polarization and
    the orbital magnetization share the source but not that check, so they keep
    the old refusal, word for word, until each has a number of its own.
    """
    functional = source._base().functional
    if functional.is_meta:
        raise NotImplementedError(
            f"{what} under {functional.name} is not implemented: the "
            "functional is a potential rather than the derivative of an "
            "energy, and the combination is unvalidated. The band structure "
            "and the density of states are unaffected"
        )


def _occupied_bands(system: System, pseudos, density=None, spin=None) -> int:
    """How many bands a band insulator fills: one electron each, or two.

    A spinor band holds one electron and a spin-degenerate band two, which is
    the same rule :meth:`defuqumat.workflows.bands.BandStructure.gap` uses.
    Getting it wrong halves or doubles the manifold, and every invariant here is
    a property of the manifold.

    At ``nspin = 2`` the count is the channel's own, ``spin``'s, read off the
    density it integrates to -- ``nelup`` or ``neldw`` -- because neither the
    electron count nor ``starting_magnetization`` says how a converged moment
    split them. A channel holding a fractional count (a smeared magnet) has no
    gapped manifold to name, and is refused rather than rounded.
    """
    if system.nspin == 2:
        rho = np.asarray(density[spin], dtype=float)
        count = float(rho.mean()) * float(system.cell.volume)
        if abs(count - round(count)) > 1.0e-6:
            raise ValueError(
                f"channel {spin} holds {count:.8f} electrons, not a whole "
                "number of bands; a topological invariant needs a gapped "
                "manifold, so the channel's occupied band count must be given "
                "explicitly"
            )
        return int(round(count))
    nelec = sum(
        float(pseudos[t].z_valence) for t in np.asarray(system.structure.types)
    )
    per_band = 1 if system.noncolin else 2
    count = nelec / per_band
    if abs(count - round(count)) > 1e-8:
        raise ValueError(
            f"{nelec} electrons do not fill a whole number of bands; a "
            "topological invariant needs a gapped manifold, so the occupied "
            "band count must be given explicitly"
        )
    return int(round(count))


def _require_spinors(system: System, what: str) -> None:
    if not system.noncolin:
        raise ValueError(
            f"{what} needs spin-orbit coupling: without it the bands are "
            "spin-degenerate, the two copies contribute opposite windings and "
            "the invariant is zero for a reason that has nothing to do with the "
            "band structure. Run with noncolin = .true. and lspinorb = .true. "
            "and a fully-relativistic pseudopotential"
        )


def run_berry_curvature(
    system: System,
    pseudos,
    density: jnp.ndarray,
    shape=(12, 12),
    axis: int = 2,
    offset: float = 0.0,
    nocc: int | None = None,
    nbnd: int | None = None,
    method: str | None = None,
    conv_thr: float = 1.0e-8,
    k_batch: int | None | str = "default",
    becsum: tuple = (),
    ns: jnp.ndarray | None = None,
    tau: jnp.ndarray | None = None,
    field=None,
    field_scale: float | None = None,
    stream: bool | None = None,
    degeneracy_tol: float | None = None,
    calculation=None,
    spin: int | None = None,
    **kwargs,
) -> BerryCurvature:
    """Berry curvature and the Chern number on one plane of the zone.

    Args:
        shape: the plaquette mesh, ``(n1, n2)`` points along the two crystal
            directions that span the plane.
        stream: diagonalise the plane a column at a time rather than whole,
            holding at most three columns of states. ``None`` streams where the
            calculation does (memory mode on a card); see
            :func:`~defuqumat.topology.invariants.chern_number`.
        axis: the crystal direction held fixed at ``offset``.
        nocc: how many bands are occupied. Defaults to the electron count
            divided by one or two according to whether a band is a spinor.
        method: ``"fhs"`` (default) or ``"kubo"``; see
            :mod:`defuqumat.topology.berry`. ``"kubo"`` gives the pointwise
            curvature and reports the truncation of its sum over empty states;
            ``"fhs"`` is the only one a Chern *number* should be read from.
        degeneracy_tol: for ``"kubo"``, the gap in Ry below which two bands
            are one level in the per-band curvature; ``None`` resolves to the
            solver's own promise on these states
            (:func:`~defuqumat.solvers.davidson.degeneracy_tolerance`), a
            number is honoured as asked. Named here so that the facade
            forwards it.
        spin: the collinear channel, 0 or 1, at ``nspin = 2`` and nowhere
            else; required there, since the two channels are two
            Hamiltonians and two answers. ``nocc`` is then the channel's count
            and defaults to what the density holds in it. A channel's Chern
            number is zero by time reversal within the channel; its pointwise
            curvature is not. Only ``method = "fhs"``.
        kwargs: forwarded to the curvature method.
    """
    # Only the Kubo route has a degeneracy tolerance; the plaquette route is a
    # determinant of overlaps and takes none, so a shared numeric option must
    # not reach its signature (the facade forwards it to every method here).
    if degeneracy_tol is not None and method == "kubo":
        kwargs["degeneracy_tol"] = degeneracy_tol
    source = _source(system, pseudos, density, nocc, nbnd, conv_thr, k_batch,
                     becsum=becsum, ns=ns, tau=tau,
                     field=field, field_scale=field_scale,
                     calculation=calculation, spin=spin)
    return _chern_number(
        source, shape=shape, axis=axis, offset=offset, method=method,
        k_batch=k_batch, stream=stream, **kwargs,
    )


def run_z2(
    system: System,
    pseudos,
    density: jnp.ndarray,
    axis: int = 2,
    offset: float = 0.0,
    nocc: int | None = None,
    nbnd: int | None = None,
    method: str | None = None,
    nloop: int = 24,
    npump: int = 13,
    stream: bool = True,
    conv_thr: float = 1.0e-8,
    k_batch: int | None | str = "default",
    becsum: tuple = (),
    ns: jnp.ndarray | None = None,
    tau: jnp.ndarray | None = None,
    field=None,
    field_scale: float | None = None,
    calculation=None,
):
    """The 2D Z2 invariant of one plane of the zone.

    ``method="wilson"`` (the default) sweeps the Wannier charge centres over the
    half zone and needs only time-reversal symmetry. ``method="parity"`` takes
    the Fu-Kane product over the four TRIM of the plane, which costs four
    diagonalisations instead of ``nloop * npump`` and needs an inversion centre
    -- found here from the space group, and refused if there is none.

    Running both on a crystal that has an inversion centre is the check worth
    making: they share no machinery beyond the state set.
    """
    _require_spinors(system, "the Z2 invariant")
    source = _source(system, pseudos, density, nocc, nbnd, conv_thr, k_batch,
                     becsum=becsum, ns=ns, tau=tau,
                     field=field, field_scale=field_scale,
                     calculation=calculation)
    kwargs = dict(axis=axis, offset=offset)
    if (method or "wilson").lower() == "parity":
        kwargs.update(dimension=2, centre=_centre(system))
    else:
        kwargs.update(nloop=nloop, npump=npump, k_batch=k_batch, stream=stream)
    return z2_invariant(source, method=method, **kwargs)


def run_z2_3d(
    system: System,
    pseudos,
    density: jnp.ndarray,
    nocc: int | None = None,
    nbnd: int | None = None,
    method: str | None = None,
    nloop: int = 24,
    npump: int = 13,
    stream: bool = True,
    conv_thr: float = 1.0e-8,
    k_batch: int | None | str = "default",
    becsum: tuple = (),
    ns: jnp.ndarray | None = None,
    tau: jnp.ndarray | None = None,
    field=None,
    field_scale: float | None = None,
    calculation=None,
):
    """The four three-dimensional indices ``(nu0; nu1 nu2 nu3)``.

    By ``wilson``, six planes; by ``parity``, eight k-points. The weak indices
    are components on the *primitive* reciprocal basis of this cell and mean
    nothing without it -- they are not invariant under a change of cell.
    """
    _require_spinors(system, "the Z2 invariants")
    source = _source(system, pseudos, density, nocc, nbnd, conv_thr, k_batch,
                     becsum=becsum, ns=ns, tau=tau,
                     field=field, field_scale=field_scale,
                     calculation=calculation)
    kwargs = {}
    if (method or "wilson").lower() == "parity":
        kwargs["centre"] = _centre(system)
    else:
        kwargs["stream"] = stream
    return z2_invariant_3d(
        source, method=method, nloop=nloop, npump=npump, k_batch=k_batch, **kwargs
    )


def _centre(system: System) -> np.ndarray:
    """The crystal's inversion centre, from its space group."""
    from defuqumat.system.symmetry import find_symmetries

    return inversion_centre(find_symmetries(system.cell, system.structure))
