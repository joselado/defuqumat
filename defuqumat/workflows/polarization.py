"""The Berry-phase polarization, end to end from a converged density.

One entry point, :func:`run_polarization`, which is ``pw.x``'s ``lberry`` run:
a fixed-density diagonalisation on strings of k-points along one reciprocal
lattice vector, the occupied manifold's Berry phase accumulated along each
string, and the ions' phase added to it.

The strings are walked **one at a time**. That is the memory decision
:func:`~defuqumat.topology.wilson.wannier_centers` already makes and it matters
more here, because a polarization mesh is the whole transverse plane rather
than half a zone: the resident set is one string's occupied manifold,
``npoints * nbnd * npwx`` complex, and the number of strings never enters it.

**What does enter it is the process's count of virtual-memory mappings, and
that is what a long mesh runs out of rather than memory.** Each string is
diagonalised at its own sphere, so XLA compiles for it and maps the code
anonymously, and the mappings accumulate: measured on norm-conserving AlAs at
``ecutwfc = 10`` over sixteen six-point strings, ``/proc/self/maps`` goes from
2172 lines to 9873, roughly 480 per string, against a ``vm.max_map_count`` of
65530 on an ordinary Linux box. The growth is per *string* and not per distinct
shape -- thirteen of the sixteen add mappings, including strings whose phase
takes 0.04 s and which therefore compiled nothing -- so it is not cured by
compiling less, and a 396-string mesh dies with
``INTERNAL: Failed to materialize symbols`` a third of the way in.

``clear_caches`` is the dial for it. One ``jax.clear_caches()`` released 8924 of
those mappings in the measurement above and ``gc.collect()`` released none on
top of it, so the cure is to clear inside the loop rather than after it. On the
36-string mesh (``nppstr = 11`` over ``6x6``, ten points a string) the count is
**19036 without it and 3705 with it**, and every period from 8 to 32 gives the
same 3705, because a clear drops whatever has accumulated since the last one.
The phase is bit-identical in all of them, checked at periods of 1, 4, 8, 16
and 32 against the unclamped run.

The default is therefore the *longest* period that bounds the count, 32, since
clearing more often costs recompilation and buys no mappings.
``clear_caches=None`` turns it off for a short mesh that does not need it.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from defuqumat.system.builder import System
from defuqumat.topology.mesh import string_mesh
from defuqumat.topology.polarization import (
    Polarization,
    StringPhases,
    combine_string_phases,
    ionic_phase,
    polarization_quantum,
    string_phase,
)
from defuqumat.workflows.topology import _refuse_potential_only_meta, _source

__all__ = ["run_polarization"]


def run_polarization(
    system: System,
    pseudos,
    density: jnp.ndarray,
    gdir: int = 2,
    nppstr: int = 7,
    transverse: tuple[int, int] = (4, 4),
    shift: tuple[int, int, int] = (0, 0, 0),
    nocc: int | None = None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-8,
    k_batch: int | None | str = "default",
    becsum: tuple = (),
    ns: jnp.ndarray | None = None,
    field=None,
    field_scale: float | None = None,
    clear_caches: int | None = 32,
) -> Polarization:
    """The Berry-phase polarization along reciprocal lattice vector ``gdir``.

    Args:
        gdir: which reciprocal lattice vector the strings run along, **0-based**
            here where ``pw.x``'s input variable of the same name is 1-based.
        nppstr: k-points per string in QE's counting, which repeats the
            endpoint; the mesh built here holds ``nppstr - 1`` distinct points
            and closes the string with the reciprocal lattice vector instead.
            Passing the number from a ``pw.x`` input therefore gives the same
            calculation.
        transverse: how many strings along the two crystal directions other
            than ``gdir``.
        shift: QE's ``k1, k2, k3`` half-step offsets of the transverse grid.
        clear_caches: drop the compiled code every this many strings, which is
            what keeps a long mesh inside the process's mapping limit; ``None``
            never drops it. The module docstring carries the measurement.
        nocc: the occupied band count, defaulting to the electron count over
            one or two as a band is a spinor or not. At ``nspin = 2`` it is a
            pair ``(n_up, n_down)``, defaulting to ``(NINT(nelup),
            NINT(neldw))`` from ``tot_magnetization``.

    At ``nspin = 2`` each channel is its own string set on its own
    Hamiltonian, its phase placed and folded onto ``[-1/2, 1/2)`` alone, and
    the electronic phase is the sum of the two with no factor of two and no
    second fold, which is ``bp_c_phase.f90:775-782``; the quantum is 1.
    :attr:`~defuqumat.topology.polarization.Polarization.strings` then holds
    the up strings followed by the down ones, pw.x's table order, and
    ``strings_by_spin`` the two channels apart.

    The polarization is defined **modulo a quantum** and the result carries it.
    A single value is not a physical statement on its own: what is meaningful is
    a *difference* between two geometries taken on the same branch, which is
    what a Born effective charge and a piezoelectric constant are made of.
    """
    _refuse_ungapped(system)
    nspin = int(system.nspin)
    if nspin == 2 and str(getattr(system, "occupations", "fixed")).lower() != "fixed":
        raise NotImplementedError(
            "a Berry-phase polarization with nspin = 2 and occupations = "
            "'from_input' is not implemented: each channel's occupied count is "
            "read from tot_magnetization (pw.x reads it off wg per channel), "
            "and an input occupation list can leave a channel with a partly "
            "filled band, which is not a manifold a string phase is defined on"
        )
    if getattr(system, "spiral_q", None) is not None:
        raise NotImplementedError(
            "a Berry-phase polarization of a spin spiral is not implemented: "
            "the two spinor components live on spheres centred at k +- q/2, so "
            "each string's states need an NSCF on the doubled list at new "
            "k-points (at_kpoints, which a spiral refuses), and the overlap "
            "between neighbouring k-points is two gathers by Miller index, one "
            "per component on its own sphere, where the string phase does one"
        )

    axis = int(gdir) % 3
    mesh = string_mesh(transverse, int(nppstr) - 1, gdir=axis, shift=shift)
    nstring, npoints = mesh.shape
    weights = np.full(nstring, 1.0 / nstring)
    options = dict(becsum=becsum, ns=ns, field=field, field_scale=field_scale)

    if nspin == 2:
        # Two channels, two string sets, one Hamiltonian each, sharing one
        # k-independent setup. Each channel's manifold is its own occupied
        # count, ``(NINT(nelup), NINT(neldw))`` -- which is what
        # ``bp_c_phase.f90:439`` reads off ``wg`` per channel -- and the two
        # sources share the calculation so the setup is built once.
        from defuqumat.scf.driver import Calculation
        from defuqumat.scf.occupations import occupied_counts

        calculation = Calculation(system, pseudos, k_batch=k_batch)
        counts = occupied_counts(calculation)
        if nocc is not None:
            if np.ndim(nocc) != 1 or len(nocc) != 2:
                raise ValueError(
                    "nocc for an nspin = 2 run is one count per channel, "
                    f"(n_up, n_down); got {nocc!r}"
                )
            counts = tuple(int(n) for n in nocc)
        sources = [
            _source(system, pseudos, density, counts[spin], nbnd, conv_thr,
                    k_batch, calculation=calculation, spin=spin, **options)
            for spin in (0, 1)
        ]
    else:
        sources = [_source(system, pseudos, density, nocc, nbnd, conv_thr,
                           k_batch, **options)]
    _refuse_potential_only_meta(sources[0], "a Berry-phase polarization")

    # One string at a time: the states of the whole mesh are never resident.
    # The mappings are the other resource and they are not freed by ``del``,
    # so the compiled code goes with them every ``clear_caches`` strings --
    # see the module docstring for the count and what it costs. The count runs
    # over (channel, string), so a second channel does not halve the bound.
    every = None if clear_caches is None else max(1, int(clear_caches))
    walked = 0
    by_spin = []
    for source in sources:
        phases = np.empty(nstring)
        for index in range(nstring):
            states = source.states(mesh.points[index])
            phases[index] = string_phase(states, k_batch=k_batch,
                                         closing_shift=mesh.span2)
            del states
            walked += 1
            if every is not None and walked % every == 0:
                jax.clear_caches()
        # Each channel is placed and folded on its own, and the two folded
        # channel phases add with no factor (``bp_c_phase.f90:775-782``); the
        # doubling is the ``nspin = 1`` branch alone, inside this call.
        by_spin.append(combine_string_phases(phases, weights, nspin=nspin))

    if nspin == 2:
        # pw.x's table order, up strings then down strings (``:874-889``), and
        # the electronic phase the sum of the two folded channels, not folded
        # again (``pdl_elec_tot = pdl_elec_up + pdl_elec_dw``).
        strings = StringPhases(
            phases=np.concatenate([one.phases for one in by_spin]),
            weights=np.concatenate([one.weights for one in by_spin]),
            total=float(by_spin[0].total + by_spin[1].total),
        )
    else:
        strings = by_spin[0]

    valences = np.array(
        [float(pseudos[t].z_valence) for t in np.asarray(system.structure.types)]
    )
    positions = np.asarray(system.structure.positions_crystal(system.cell))
    ion_phases, ionic, _ = ionic_phase(positions, valences, axis)

    quantum = polarization_quantum(valences, nspin=nspin)
    # Not reduced again. ``bp_c_phase.f90`` adds the two contributions and
    # reports the sum against ``mod_tot`` without folding it, and each half has
    # already been reduced by *its own* quantum -- which is not always the
    # total's. Folding here would put a value on a different branch from the one
    # pw.x prints, for no gain: everything physical is a difference.
    total = float(strings.total + ionic)

    cell = system.cell
    at = np.asarray(cell.to_cartesian(np.eye(3)))
    lattice_vector = at[axis]
    length = float(np.linalg.norm(lattice_vector))

    return Polarization(
        gdir=axis,
        ionic_phase=float(ionic),
        electronic_phase=float(strings.total),
        total_phase=total,
        quantum=float(quantum),
        ion_phases=ion_phases,
        strings=strings,
        lattice_length=length,
        volume=float(cell.volume),
        direction=lattice_vector / length,
        points_per_string=int(npoints),
        strings_by_spin=tuple(by_spin),
    )


def _refuse_ungapped(system: System) -> None:
    """A Berry phase is a property of a gapped manifold; a metal has none.

    The same refusal every invariant in :mod:`defuqumat.topology` makes, and it
    is worth making at the workflow rather than letting the gap check inside
    :class:`~defuqumat.workflows.topology.DFTSource` report it per k-point: a
    smeared occupation does not say which bands are in the manifold, and a
    string phase taken over an arbitrary count is a confident number with no
    meaning.
    """
    occupations = str(getattr(system, "occupations", "fixed")).lower()
    if occupations not in ("fixed", "from_input"):
        raise NotImplementedError(
            "a Berry-phase polarization of a metal is not defined: the phase is "
            f"a property of an isolated occupied manifold and {occupations} "
            "smears the occupation across the Fermi level, so which bands the "
            "string carries is not determined. pw.x refuses the same "
            "combination (bp_c_phase.f90 needs a fixed band count)"
        )
