"""A converged state on disk, so a wall clock does not cost a calculation.

Nothing here was serialisable before: ``run_scf(starting_from=result)`` needs
the object in memory, so a job that hits its time limit loses the run, and two
processes cannot share a ground state at all. On a cell whose SCF is measured in
hours that is the difference between a calculation and a calculation that has to
be restarted from the beginning.

**What is saved is the state, not the system.** A :class:`System` carries
pseudopotential file names, a precision policy and a symmetry group, and writing
those to a file makes the file a second, weaker input parser. The resume path a
cluster job actually takes already has the input: it rebuilds the
:class:`~defuqumat.calculator.Calculator` from ``scf.in`` and hands the system in.
So what is stored beside the arrays is a **fingerprint** -- the shapes and the
counts a continuation would have to agree on anyway -- and loading against a
system that does not match is refused rather than discovered three iterations
later as a wrong answer.

**Every field of** :class:`~defuqumat.scf.driver.SCFResult` **is accounted for**,
and that is enforced rather than intended: :data:`_ARRAYS`, :data:`_SCALARS` and
:data:`_REFUSED` must together cover ``dataclasses.fields(SCFResult)``, and
``tests/unit/test_checkpoint.py`` asserts it. A field added to the result and
not to one of those three lists fails the test instead of being silently
dropped -- which is the failure mode that matters here, since a checkpoint that
quietly loses ``becsum`` reloads as a *different* state that converges to
something plausible.

**A magnetic field the run drove away from the input's is written as the
controller's state**, not refused. Under the fixed-spin-moment schemes
(:data:`~defuqumat.scf.fields.FEEDBACK`, Elk's ``fsmtype``) the field *is* the
controller's state and is replaced after every iteration, so it cannot be
rebuilt from the input file, and reloading without it applies the input's field
to a density mixed beside another one, which a later invariant still returns an
integer for (``PLAN.md`` P56 is the record of that bug found the hard way). So
the leaves the controller changes, :data:`~defuqumat.scf.fields.DRIVEN_LEAVES`
(the field, and the secant's previous field and moment), are written beside the
density, with the scheme and its update rule in the metadata, and
:func:`load_state` pours them back onto the field a ``calculation`` built from
the same input -- the regions, the targets and the gain are the input's and come
back from it, as ``system`` does. A file holding a driven field is refused
without a ``calculation`` to pour it onto, and refused against a calculation
whose scheme, update rule or array shapes differ, since either would apply a
field to the wrong problem without a word. This used to be the one field
refused by name (``notes/incompat/I65.md``).

**A Hubbard setup used to be the second, and it was the same mistake as the
blanket field refusal one paragraph down.** "``ns`` without its setup is an
array of numbers about nothing" is true of the *file* and the file is not what
a resume reads it against: a state is loaded against a ``system``, and
``run_scf`` rebuilds the manifold from the ``HUBBARD`` card through
``build_hubbard_setup`` before it looks at ``ns`` at all. Nothing in the loop
can make the two disagree -- ``Calculation.hubbard`` is assigned in exactly one
place (``driver.py``'s ``__init__``), the loop mixes ``ns`` and never the setup,
and the only attribute written on a ``HubbardSetup`` anywhere is
``constraints``, inside ``build_hubbard_setup`` before it returns. ``ns_adj``
does not reach a resume either: it is gated on ``iteration == 1``
(``PW/src/init_ns.f90``'s ``IF (first .AND. starting_pot == 'atomic')``) and a
resume re-enters at ``resumed_at + 1``. The mid-SCF path was always the
evidence -- ``_InProgressState`` has never carried a setup, so every DFT+U run
with ``checkpoint_dir`` has been writing and reloading one of these correctly.
So the setup is :data:`_FROM_CALLER` now, beside ``system``, and what a load
gives back without a ``calculation`` is exactly what a mid-SCF checkpoint has
always given back (``OPEN.md`` Part VII item 3).

**A field is not refused for being a field**, and the difference is worth
stating because the blanket version of this refusal made checkpointing useless
for exactly the runs it exists for -- long, magnetic, and unable to restart. An
applied ``LOCAL_MAGNETIC_FIELDS`` card, and a penalty constraint of any flavour,
leave :class:`~defuqumat.scf.fields.MagneticField` exactly as the input built it:
the resume rebuilds the calculator from ``scf.in`` and gets the same object
back. Elk's ``reducebf`` does not change it either -- it multiplies a *scalar*,
``field_scale``, which is saved here and restored by the driver on resume. So
what has to be carried is the feedback set's driven leaves
(:data:`~defuqumat.scf.fields.FEEDBACK`) and nothing wider.

**The scale and the driven field are one mechanism, not two.** ``field_scale``
riding in :data:`_SCALARS` while the loop reset it to 1.0 on resume was a
``reducebf`` run silently coming back at full field, which is the same silent
loss the driven leaves are carried to prevent, taken one level up; ``run_scf``
restores both in the same place on resume.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from defuqumat.parallel import PoolStore
from defuqumat.scf.fields import DRIVEN_LEAVES, FEEDBACK

__all__ = ["save_state", "load_state", "state_fingerprint",
           "save_mixer", "load_mixer", "save_pool_rows", "remove_stale_pool_rows"]

#: Array-valued fields, stored as their own entries in the ``.npz``. ``None`` is
#: representable: the key is simply absent.
_ARRAYS = (
    "eigenvalues", "occupations", "wavefunctions", "density", "potential",
    "potential_change", "ns",
    # ``tau`` is *state*, not a diagnostic: under a meta-GGA the potential is
    # rebuilt from the density **and** this, and a fixed density alone does not
    # determine it. It was the field the coverage check above was written for --
    # it was missing from the first draft of this list, and a reloaded meta-GGA
    # run would have silently rebuilt a different potential.
    "tau",
)

#: Everything JSON can hold. ``energy_terms`` is a dict of floats and
#: ``magnetization_vector`` a tuple of them.
_SCALARS = (
    "converged", "iterations", "total_energy", "energy_terms", "fermi_energy",
    # ``ethr`` is loop state rather than a report: ``save_in_electrons.f90``
    # writes ``iter, dr2, ethr`` because QE's schedule is indexed on the
    # iteration number, and a resume that re-enters at 1 without it takes the
    # reset and converges on a different schedule than the one it left.
    "homo", "lumo", "accuracy", "ethr", "nspin", "magnetization",
    # The gap across a fixed filling's cut, one float per channel and NaN where
    # a channel has no band above its cut; a report beside ``homo`` and ``lumo``,
    # carried the same way (JSON writes a NaN and reads it back).
    "cut_gap",
    "absolute_magnetization", "fermi_energy_up", "fermi_energy_down",
    "magnetization_vector", "field_energy", "constraint_energy", "field_scale",
    "efield_energy", "efield_dipole", "berry_dipoles",
    "nspin_mag", "meta_c",
)

#: Handled specially: a tuple of arrays, one per ultrasoft/PAW species.
_TUPLES = ("becsum",)

#: Not saved, and refused rather than dropped. Empty since I65: the one entry
#: it had, ``magnetic_field``, is :data:`_DRIVEN` now.
_REFUSED = ()

#: Half saved and half rebuilt. A feedback scheme's driven leaves
#: (:data:`~defuqumat.scf.fields.DRIVEN_LEAVES`) are written as ``field_<leaf>``
#: arrays; the rest of the field (the integration spheres, the targets, the
#: gain) is the input's and comes back from the ``calculation`` the state is
#: loaded against, as ``system`` does. A field no controller drove is the
#: calculation's own, unchanged.
_DRIVEN = ("magnetic_field",)

#: Saved by nothing and dropped on purpose: these are what the run *reported*,
#: not what it converged to. A checkpoint is a state to continue from -- the
#: stress is recomputed from the state in one strain gradient, the solver record
#: and the iteration history describe a run that is over. A reloaded result is
#: therefore a state and not a report, which :func:`load_state` says in its
#: docstring rather than leaving to be noticed.
#:
#: The site charges and moments are reports of the same kind and are dropped for
#: a second reason as well: they are a contraction of the *density* against the
#: integration spheres, costing about 2 ms, and the spheres are a function of the
#: geometry. Carrying them would let a pair computed for one geometry ride along
#: beside a density reloaded into another -- the ``at_cell`` defect one layer up.
#: :meth:`Calculation.site_moments` recomputes them from the loaded density.
#: ``constraint_residual`` is dropped for the same pair of reasons as the site
#: moments: it is a report rather than state, and it is a contraction of the
#: *density* against the constraint's target, so
#: ``MagneticField.cell_residual`` recomputes it from the loaded density. The
#: target itself rides on ``magnetic_field``, which is carried.
#: ``orientation_torque`` is dropped for the same pair of reasons again: it is
#: the last iteration's ``integral of B[rho_in] x m_out``, a report of the run,
#: and a contraction of the output density against the input potential, which a
#: resumed run recomputes at its first iteration.
_DROPPED = ("stress", "solver", "history", "site_charges", "site_moments",
             "site_residuals", "constraint_residual", "orientation_torque")

#: Reconstructed from what the caller supplies on load. ``system`` comes back
#: directly; ``hubbard_setup`` is ``Calculation.hubbard``, which
#: :func:`load_state` takes off a ``calculation`` when one is given and leaves
#: ``None`` when it is not -- ``build_hubbard_setup`` needs the datasets as well
#: as the card, and a bare system does not carry them.
_FROM_CALLER = ("system", "hubbard_setup")

#: Bumped when the layout changes in a way an older file cannot be read as.
FORMAT_VERSION = 1


def state_fingerprint(result) -> dict:
    """The shapes and counts a continuation has to agree on.

    Not a hash of the input file: two runs that differ in ``conv_thr`` or in
    their mixing share a state perfectly well, and refusing that would make
    checkpoints useless for exactly the restart they exist for. What must match
    is the basis, the k-set, the spin regime and the electron count -- which is
    what :func:`~defuqumat.scf.continuation._check_grid` checks, one layer down.
    """
    density = np.shape(result.density)
    store = result.wavefunctions
    # A k-point pool's store is fingerprinted as the whole set it is a share
    # of, so that a checkpoint written by pools and one written by a single
    # process describe the same state the same way.
    wavefunctions = ((store.array.shape[0], store.nk) + store.array.shape[2:]
                     if isinstance(store, PoolStore) else np.shape(store))
    system = getattr(result, "system", None)
    return {
        "density_shape": list(density),
        "wavefunction_shape": list(wavefunctions),
        "nspin": int(result.nspin),
        "nspin_mag": int(result.nspin_mag),
        "nbecsum": len(result.becsum or ()),
        "natoms": (None if system is None
                   else len(np.asarray(system.structure.types))),
        "types": (None if system is None
                  else [int(t) for t in np.asarray(system.structure.types)]),
    }


def _describe(fingerprint) -> str:
    return (f"density {tuple(fingerprint['density_shape'])}, "
            f"wavefunctions {tuple(fingerprint['wavefunction_shape'])}, "
            f"nspin {fingerprint['nspin']}")


def _refusal(result) -> str | None:
    """Why this state cannot be written, or ``None`` if it can.

    One place rather than two: the mid-SCF checkpoint, the pooled one and the
    converged one go through the same test. It refused a fixed-spin-moment
    run's driven field until the controller's leaves were written
    (:func:`_driven_payload`), and nothing else has needed refusing since; it
    is kept as the one hook a future refusal belongs in, which the pooled
    writer calls before any pool writes its rows.
    """
    return None


def _driven_payload(result) -> tuple[dict, dict | None]:
    """The feedback controller's leaves as ``field_<leaf>`` arrays, and its metadata.

    ``({}, None)`` for a run with no field, or with a field no controller
    drives: an applied field and a penalty constraint are the input's to the
    end, and a resume rebuilds them from ``scf.in``.
    """
    field = getattr(result, "magnetic_field", None)
    if field is None or getattr(field, "constraint", "none") not in FEEDBACK:
        return {}, None
    arrays = {}
    for leaf in DRIVEN_LEAVES:
        value = getattr(field, leaf)
        if value is not None:
            arrays[f"field_{leaf}"] = np.asarray(value)
    meta = {"constraint": field.constraint, "fsm_update": field.fsm_update,
            "leaves": sorted(name[len("field_"):] for name in arrays)}
    return arrays, meta


def _restore_driven_field(driven, saved, calculation, path, strict):
    """The field the run held: the input's, with the controller's leaves poured back.

    ``driven`` is the file's ``driven_field`` metadata or ``None``; ``saved``
    the ``field_<leaf>`` arrays. Returns ``None`` where there is nothing to
    rebuild against, which is what a load without a ``calculation`` has always
    given back for a field.
    """
    import equinox as eqx

    template = None if calculation is None else calculation.magnetic_field
    if driven is None:
        if template is not None and template.constraint in FEEDBACK:
            # A file from a run with no controller, loaded against one that
            # has: the resume would drive the field from the input's value
            # while the density was converged beside another field or none.
            _refuse(f"{path} carries no driven field and this calculation's "
                    f"field is driven by the {template.constraint!r} "
                    "fixed-spin-moment scheme: the state was written by a "
                    "different run", strict)
        return template
    if template is None:
        _refuse(f"{path} holds a field driven by the {driven['constraint']!r} "
                "fixed-spin-moment scheme, which is only its controller's leaves: "
                "the integration spheres, the targets and the gain are the "
                "input's. Pass calculation= built from the same input to load it"
                if calculation is None else
                f"{path} holds a field driven by the {driven['constraint']!r} "
                "fixed-spin-moment scheme and this calculation has no field at "
                "all", strict)
        return None
    if (template.constraint != driven["constraint"]
            or template.fsm_update != driven["fsm_update"]):
        _refuse(f"{path} holds a field driven by {driven['constraint']!r} with the "
                f"{driven['fsm_update']!r} update, and this calculation's field "
                f"is {template.constraint!r} with {template.fsm_update!r}: the "
                "controller's state belongs to another problem", strict)
        return template
    dtype = jnp.asarray(template.uniform).dtype
    field = template
    for leaf in DRIVEN_LEAVES:
        value = saved.get(f"field_{leaf}")
        if value is None:
            # Absent in the file is ``None`` in the run that wrote it: the
            # secant's previous pair before its first step. The input's field
            # has ``None`` there as well, except ``uniform``, which is never
            # ``None`` and so is always written.
            continue
        current = getattr(template, leaf)
        if current is not None and np.shape(current) != np.shape(value):
            _refuse(f"{path} holds the driven {leaf} with shape {np.shape(value)} "
                    f"and this calculation's is {np.shape(current)}", strict)
            return template
        field = eqx.tree_at(lambda f, leaf=leaf: getattr(f, leaf), field,
                            jnp.asarray(value, dtype=dtype),
                            is_leaf=lambda x: x is None)
    return field


def _refuse(message, strict) -> None:
    if strict:
        raise ValueError(message)
    warnings.warn(message, stacklevel=4)


def save_state(result, path) -> Path:
    """Write a converged :class:`SCFResult` to ``path`` as a ``.npz``.

    The wavefunctions dominate the file -- ``nspin nk nbnd npwx`` complex, which
    is tens of gigabytes on a large cell -- so this is a scratch-directory
    operation, not something to do every iteration without meaning to.
    """
    path = Path(path)
    refused = _refusal(result)
    if refused is not None:
        raise NotImplementedError(refused)

    payload, meta = {}, {"format": FORMAT_VERSION}
    for name in _ARRAYS:
        value = getattr(result, name, None)
        if isinstance(value, PoolStore):
            # Each pool writes its own rows (:func:`save_pool_rows`), before
            # this file; what this file records is which set of rows files
            # belongs to it, by the iteration they were written at.
            meta["pooled_wavefunctions"] = {
                "iterations": int(getattr(result, "iterations", 0) or 0),
                "nk": int(value.nk)}
            continue
        if value is not None:
            payload[name] = np.asarray(value)
    for name in _SCALARS:
        meta[name] = getattr(result, name, None)
    driven_arrays, driven_meta = _driven_payload(result)
    payload.update(driven_arrays)
    if driven_meta is not None:
        meta["driven_field"] = driven_meta
    becsum = result.becsum or ()
    meta["nbecsum"] = len(becsum)
    for index, block in enumerate(becsum):
        payload[f"becsum_{index}"] = np.asarray(block)
    meta["fingerprint"] = state_fingerprint(result)

    payload["__meta__"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the target and moved into place, so an interrupted write
    # never leaves a half file that a resume would read as a state.
    scratch = path.with_suffix(path.suffix + ".partial")
    np.savez(scratch, **payload)
    written = scratch.with_suffix(scratch.suffix + ".npz") if not scratch.suffix.endswith(".npz") else scratch
    written.replace(path)
    return path


def load_state(path, system=None, calculation=None, strict: bool = True, rows=None):
    """Read back what :func:`save_state` wrote.

    Args:
        path: the ``.npz``.
        system: the :class:`System` the state belongs to. Supplied rather than
            stored -- see the module docstring -- and checked against the
            fingerprint.
        calculation: optional; when given, the basis is checked too, which
            catches a cutoff change that the fingerprint alone would not.
        strict: whether a fingerprint mismatch raises. ``False`` warns instead,
            which is for inspecting a file rather than for running from one.
        rows: the global k rows a k-point pool holds, which makes the
            wavefunctions come back as that pool's
            :class:`~defuqumat.parallel.PoolStore`; ``None`` returns the whole
            set. Either works on a file written by pools or by one process,
            so a restart may change the pool count.

    **What comes back is a state, not a report.** ``stress``, ``solver`` and
    ``history`` describe the run that produced the file and are not stored, so
    they are at their defaults here. Everything a continuation consumes --
    the density, ``becsum``, ``ns``, ``tau`` and the wavefunctions -- is.

    ``hubbard_setup`` is the one field rebuilt rather than read: it is
    ``Calculation.hubbard``, so it comes back when a ``calculation`` is given
    and is ``None`` otherwise. A DFT+U state loaded without one carries ``ns``
    and nothing that says which atom each slot belongs to, which is what
    ``SCFResult.hubbard_occupations`` needs; the *resume* does not care, since
    ``run_scf`` rebuilds the manifold from the ``HUBBARD`` card either way.
    """
    from defuqumat.scf.driver import SCFResult

    path = Path(path)
    with np.load(path, allow_pickle=False) as handle:
        meta = json.loads(bytes(handle["__meta__"]).decode("utf-8"))
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"{path} is checkpoint format {meta.get('format')}, and this is "
                f"version {FORMAT_VERSION}"
            )
        fields = {name: meta[name] for name in _SCALARS if name in meta}
        for name in ("magnetization_vector", "cut_gap"):
            if fields.get(name) is not None:
                fields[name] = tuple(fields[name])
        host = ("eigenvalues", "occupations", "wavefunctions")
        arrays = {
            name: (np.asarray(handle[name]) if name in host
                   else jnp.asarray(handle[name]))
            for name in _ARRAYS if name in handle
        }
        # ``eigenvalues`` and ``occupations`` are numpy on the result, not JAX:
        # they are indexed and sliced host-side throughout. **So are the
        # wavefunctions**, which are the largest array a run holds: a streamed
        # result's are a host array already, and a resume that streams slices
        # them a chunk at a time (``stream_start``), so putting the whole set on
        # the device here was a transient of the whole store at iteration 1 --
        # 12.10 GB on the NiBr2 slab (``notes/GPU-MEMORY-NEXT.md`` item 8). Whatever
        # reads them whole brings them across itself. (Chosen when each array
        # is read, not converted afterwards: a device copy made first and
        # replaced is the same transient, and was, on the first try.)
        becsum = tuple(
            jnp.asarray(handle[f"becsum_{index}"])
            for index in range(int(meta.get("nbecsum", 0)))
        )
        saved_field = {name: np.asarray(handle[name]) for name in handle.files
                       if name.startswith("field_")}
    pooled = meta.get("pooled_wavefunctions")
    # A state may be written without its states (a pooled relaxation's rank 0
    # does, since its resume reads the density and ``becsum`` alone); it comes
    # back with ``None`` there, which a continuation reads as "start the states
    # from the atomic orbitals".
    arrays.setdefault("wavefunctions", None)
    if pooled is not None:
        arrays["wavefunctions"] = _gather_pool_rows(
            path, int(pooled["iterations"]), int(pooled["nk"]), rows)
    elif rows is not None and "wavefunctions" in arrays:
        whole = arrays["wavefunctions"]
        arrays["wavefunctions"] = PoolStore(
            np.ascontiguousarray(whole[:, np.asarray(rows)]), rows, whole.shape[1])

    result = SCFResult(
        **fields, **arrays, becsum=becsum, system=system,
        # Rebuilt from the caller, like ``system``. ``build_hubbard_setup``
        # resolves the ``HUBBARD`` card against the structure *and* the
        # datasets, so a bare system cannot do it and the field stays ``None``
        # -- which is what ``_InProgressState`` has always written, and what
        # every mid-SCF DFT+U resume has always reloaded.
        hubbard_setup=None if calculation is None else calculation.hubbard,
        # The input's field, with a feedback controller's leaves poured back
        # (:data:`_DRIVEN`); refused rather than guessed where the file and the
        # calculation do not describe the same controller.
        magnetic_field=_restore_driven_field(
            meta.get("driven_field"), saved_field, calculation, path, strict),
    )
    if system is not None:
        _check_fingerprint(meta["fingerprint"], state_fingerprint(result),
                           path, strict)
    if calculation is not None:
        grid = tuple(calculation.basis.dense.grid)
        shape = tuple(np.shape(result.density))[1:]
        if shape != grid:
            raise ValueError(
                f"{path} holds a density on a {shape} grid and this calculation "
                f"uses {grid}: the cell or a cutoff has changed"
            )
    return result


def _rows_file(path, iterations: int, rows) -> Path:
    """``<stem>.it<N>.rows<first>-<last>.npz`` beside the state file ``path``."""
    path = Path(path)
    stem = path.name[: -len(".npz")] if path.name.endswith(".npz") else path.name
    return path.with_name(f"{stem}.it{iterations:06d}.rows{int(rows[0]):06d}-"
                          f"{int(rows[-1]):06d}.npz")


def _rows_files(path, iterations: int | None = None) -> list[Path]:
    path = Path(path)
    stem = path.name[: -len(".npz")] if path.name.endswith(".npz") else path.name
    stamp = "*" if iterations is None else f"{iterations:06d}"
    return sorted(path.parent.glob(f"{stem}.it{stamp}.rows*.npz"))


def save_pool_rows(store: PoolStore, path, iterations: int) -> Path:
    """One k-point pool's rows of the store, stamped with the iteration they belong to.

    Written **before** the state file of the same iteration and never over a
    file of another iteration, since the stamp is in the name: a job killed
    between the two writes leaves the previous iteration's state file and its
    rows files whole, and the new rows files as orphans that
    :func:`remove_stale_pool_rows` clears once a state file names them. The
    file holds the rows' global indices, so a load with any pool count finds
    the rows it needs by reading them.
    """
    target = _rows_file(path, iterations, store.rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    scratch = target.with_name(target.name + ".partial.npz")
    np.savez(scratch, rows=np.asarray(store.rows), nk=np.asarray(store.nk),
             iterations=np.asarray(iterations), array=store.array)
    scratch.replace(target)
    return target


def remove_stale_pool_rows(path, iterations: int) -> None:
    """Delete the rows files beside ``path`` that belong to any other iteration."""
    for stale in _rows_files(path):
        if f".it{iterations:06d}." not in stale.name:
            stale.unlink(missing_ok=True)


def _gather_pool_rows(path, iterations: int, nk: int, rows):
    """The rows a process needs, read from the rows files of ``iterations``.

    ``rows = None`` assembles the whole set. A row no file holds is refused
    rather than filled: a store with a hole would start that k-point from
    whatever the array held, which converges to something plausible.
    """
    wanted = np.arange(nk) if rows is None else np.asarray(rows)
    found: dict[int, tuple[Path, int]] = {}
    for file in _rows_files(path, iterations):
        with np.load(file, allow_pickle=False) as handle:
            if int(handle["iterations"]) != iterations or int(handle["nk"]) != nk:
                continue
            for position, row in enumerate(np.asarray(handle["rows"])):
                found.setdefault(int(row), (file, position))
    missing = [int(row) for row in wanted if int(row) not in found]
    if missing:
        raise ValueError(
            f"{path} was written by k-point pools at iteration {iterations} and no "
            f"rows file of that iteration holds k rows {missing}; the rows files "
            f"are {[f.name for f in _rows_files(path)]}")
    store = None
    by_file: dict[Path, list[tuple[int, int]]] = {}
    for index, row in enumerate(wanted):
        file, position = found[int(row)]
        by_file.setdefault(file, []).append((index, position))
    for file, pairs in by_file.items():
        with np.load(file, allow_pickle=False) as handle:
            array = handle["array"]
            if store is None:
                store = np.empty((array.shape[0], len(wanted)) + array.shape[2:],
                                 array.dtype)
            for index, position in pairs:
                store[:, index] = array[:, position]
    return store if rows is None else PoolStore(store, wanted, nk)


def _check_fingerprint(stored, rebuilt, path, strict) -> None:
    """The stored state and the supplied system must be the same calculation."""
    differences = [
        key for key in ("density_shape", "wavefunction_shape", "nspin",
                        "nspin_mag", "nbecsum", "types")
        # ``types`` is ``None`` in the rebuilt copy when no system was given,
        # and in the stored one when the result that was saved had none.
        if stored.get(key) is not None and rebuilt.get(key) is not None
        and stored[key] != rebuilt[key]
    ]
    if not differences:
        return
    message = (
        f"{path} does not describe this system: {', '.join(differences)} "
        f"differ. Stored {_describe(stored)}; this run {_describe(rebuilt)}"
    )
    if strict:
        raise ValueError(message)
    warnings.warn(message, stacklevel=3)


def _covered_fields() -> set:
    """Every field name the lists above claim to handle."""
    return (set(_ARRAYS) | set(_SCALARS) | set(_TUPLES) | set(_REFUSED)
            | set(_DRIVEN) | set(_FROM_CALLER) | set(_DROPPED))


def unhandled_fields() -> set:
    """Fields of ``SCFResult`` no list mentions -- empty, and tested to be.

    A checkpoint that silently omits a field reloads as a different state, so
    the coverage is asserted rather than maintained by hand.
    """
    from defuqumat.scf.driver import SCFResult

    return {f.name for f in dataclasses.fields(SCFResult)} - _covered_fields()


# --- the optimizer's own state ---------------------------------------------
#
# A relaxation that resumes from positions alone throws away the inverse Hessian
# and the trust radius, so it takes its first step as if it were step one -- and
# BFGS earns its convergence rate entirely from that history. Restarting a
# half-finished relaxation without it is close to restarting it.

#: Rebuilt by constructing a fresh :class:`~defuqumat.relax.bfgs.BFGS` with the
#: same cell and settings, so they are not stored. ``at``, the metric and the
#: volume are functions of the cell; ``settings`` and the thresholds come from
#: the caller; ``h`` and the metric blocks are derived in ``_setup``.
_OPTIMIZER_DERIVED = frozenset({
    "at", "h", "metric", "inverse_metric", "metric_blocks",
    "inverse_metric_blocks", "omega", "settings", "energy_thr", "grad_thr",
    "cell_thr", "variable_cell", "pressure", "cell_mask",
})


def save_optimizer(optimizer, path) -> Path:
    """Write a :class:`~defuqumat.relax.bfgs.BFGS`'s evolving state.

    Everything in its ``__dict__`` that is not derivable from the cell and the
    settings -- the inverse Hessian above all, but also the trust radius and the
    previous step, which together decide the next one. What is *not* stored is
    listed in :data:`_OPTIMIZER_DERIVED` and is reconstructed by building the
    optimizer normally before the state is poured back in.
    """
    path = Path(path)
    payload, meta = {}, {"format": FORMAT_VERSION}
    for name, value in vars(optimizer).items():
        if name in _OPTIMIZER_DERIVED:
            continue
        if isinstance(value, np.ndarray):
            payload[f"array_{name}"] = value
        elif value is None or isinstance(value, (bool, int, float, str)):
            # ``np.inf`` survives JSON as ``Infinity``, which ``json`` reads
            # back; the errors start at infinity and a resume before the first
            # step has to see that rather than a zero.
            meta[name] = value
        else:
            raise NotImplementedError(
                f"the optimizer holds {name!r} of type {type(value).__name__}, "
                "which this checkpoint cannot represent. Add it to "
                "_OPTIMIZER_DERIVED if it is rebuilt from the cell and the "
                "settings, or extend this function"
            )
    payload["__meta__"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".partial")
    np.savez(scratch, **payload)
    written = (scratch if scratch.suffix.endswith(".npz")
               else scratch.with_suffix(scratch.suffix + ".npz"))
    written.replace(path)
    return path


def load_optimizer(optimizer, path):
    """Pour a saved state back into a freshly constructed optimizer.

    ``optimizer`` is built by the caller with the cell and settings the run
    uses, which is what supplies everything :data:`_OPTIMIZER_DERIVED` names;
    this then restores the history on top of it. Returns the same object.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as handle:
        meta = json.loads(bytes(handle["__meta__"]).decode("utf-8"))
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"{path} is checkpoint format {meta.get('format')}, and this is "
                f"version {FORMAT_VERSION}"
            )
        for key in handle.files:
            if key.startswith("array_"):
                setattr(optimizer, key[len("array_"):], np.asarray(handle[key]))
    for name, value in meta.items():
        if name != "format":
            setattr(optimizer, name, value)
    return optimizer


#: Rebuilt by ``get_mixer`` from ``mixing_mode``/``mixing_beta``, or installed by
#: the driver because building it needs the G-vectors the mixer does not have.
#: Everything else on a mixer is evolving state and is written.
_MIXER_DERIVED = frozenset({
    # ``beta_max`` is here for the same reason ``beta`` is: it is a setting
    # ``get_mixer`` rebuilds from the input, and storing it would mean a resume
    # silently ignored a caller who changed it. What the adaptive mixer *does*
    # carry across a resume is its evolved per-component state (``_betas``,
    # ``_previous``), which is stored like any other array.
    "beta", "beta_max", "history", "condition_limit", "precondition", "metric",
    # Installed by the driver from the G-vectors and the density's shape, as the
    # preconditioner is. What the stored vectors' layout *was* is evolving state
    # and is written: ``_history_space``, a string, which a resume compares with
    # the layout it installs and drops the history on a mismatch.
    "layout", "shape",
    # ``mixing_beta_mag``'s value, set by ``run_scf`` from its own argument as
    # ``beta`` is from ``mixing_beta``; restoring it from the file would keep a
    # step a resumed run's caller had dropped.
    "beta_mag",
})


def save_mixer(mixer, path) -> Path:
    """Write a :class:`~defuqumat.scf.mixing.Mixer`'s evolving history.

    **This is the half of a restart that is easy to leave out and expensive to
    get wrong.** A resume that restores only the density hands Anderson an empty
    history, so the first iterations back are plain mixing and the run pays back
    the iterations the checkpoint saved -- the same trap P67 solved on the BFGS
    side, where the assertion is "2 + 4 steps, not 2 + 6".

    ``_densities`` and ``_residuals`` are lists of equal-length one-dimensional
    arrays, so they stack; a :class:`~defuqumat.scf.mixing.LinearMixer` has no
    state at all and writes a file with nothing in it but the format, which is
    correct rather than a special case. What is *not* written is
    :data:`_MIXER_DERIVED` -- the settings, which ``get_mixer`` rebuilds, and
    the preconditioner, which the driver installs because it needs the
    G-vectors.
    """
    path = Path(path)
    payload, meta = {}, {"format": FORMAT_VERSION}
    for name, value in vars(mixer).items():
        if name in _MIXER_DERIVED:
            continue
        if isinstance(value, np.ndarray):
            payload[f"array_{name}"] = value
        elif isinstance(value, list):
            # The history. Stacked with its length in the metadata, because an
            # empty list and a list of empty arrays are different states and
            # ``np.stack`` cannot tell them apart on the way back.
            meta[f"len_{name}"] = len(value)
            for index, entry in enumerate(value):
                payload[f"list_{name}_{index}"] = np.asarray(entry)
        elif value is None or isinstance(value, (bool, int, float, str)):
            meta[name] = value
        else:
            raise NotImplementedError(
                f"the mixer holds {name!r} of type {type(value).__name__}, "
                "which this checkpoint cannot represent. Add it to "
                "_MIXER_DERIVED if ``get_mixer`` rebuilds it, or extend this "
                "function -- dropping it would restart the history silently"
            )
    payload["__meta__"] = np.frombuffer(
        json.dumps(meta).encode("utf-8"), dtype=np.uint8
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".partial")
    np.savez(scratch, **payload)
    written = (scratch if scratch.suffix.endswith(".npz")
               else scratch.with_suffix(scratch.suffix + ".npz"))
    written.replace(path)
    return path


def load_mixer(mixer, path):
    """Pour a saved history back into a freshly built mixer. Returns it."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as handle:
        meta = json.loads(bytes(handle["__meta__"]).decode("utf-8"))
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"{path} is checkpoint format {meta.get('format')}, and this is "
                f"version {FORMAT_VERSION}"
            )
        for key in handle.files:
            if key.startswith("array_"):
                setattr(mixer, key[len("array_"):], np.asarray(handle[key]))
        for key, count in meta.items():
            if not key.startswith("len_"):
                continue
            name = key[len("len_"):]
            setattr(mixer, name, [
                np.asarray(handle[f"list_{name}_{index}"])
                for index in range(int(count))
            ])
    for name, value in meta.items():
        if name != "format" and not name.startswith("len_"):
            setattr(mixer, name, value)
    return mixer


def unhandled_mixer_fields(mixer) -> set:
    """Attributes neither stored nor declared derived -- empty, and tested."""
    stored = set()
    for name, value in vars(mixer).items():
        if name in _MIXER_DERIVED:
            continue
        if isinstance(value, (np.ndarray, list)) or value is None or isinstance(
            value, (bool, int, float, str)
        ):
            stored.add(name)
    return set(vars(mixer)) - stored - set(_MIXER_DERIVED)


def unhandled_optimizer_fields(optimizer) -> set:
    """Attributes neither stored nor declared derived -- empty, and tested."""
    stored = set()
    for name, value in vars(optimizer).items():
        if name in _OPTIMIZER_DERIVED:
            continue
        if isinstance(value, np.ndarray) or value is None or isinstance(
            value, (bool, int, float, str)
        ):
            stored.add(name)
    return set(vars(optimizer)) - stored - set(_OPTIMIZER_DERIVED)


# ------------------------------------------------- the robust mode's supervisor

#: Rebuilt by ``run_scf`` from the cell and the input before the loop starts
#: (the electron count, whether the run is smeared, whether the LDOS
#: preconditioner is already the inner mixer's, whether the smearing is
#: variational), so a resume reads them from the run and not from the file.
_SUPERVISOR_DERIVED = frozenset({"nelec", "smeared", "ldos", "variational"})


def _encode(value, payload: dict):
    """A JSON-able image of ``value``, its arrays moved into ``payload``.

    The supervisor's states are ``(rho, becsum, ns, tau)`` tuples with ``None``
    for what a run does not have, and its ``_last`` record is a dict of
    scalars, so this walks tuples, lists and string-keyed dicts. Whether an
    array was a JAX array is kept, so the resumed run combines the same kinds
    of array the uninterrupted one did.
    """
    import jax

    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.ndarray, jax.Array)):
        key = f"array_{len(payload)}"
        payload[key] = np.asarray(value)
        return {"__array__": key, "jax": isinstance(value, jax.Array)}
    if isinstance(value, tuple):
        return {"__tuple__": [_encode(entry, payload) for entry in value]}
    if isinstance(value, list):
        return [_encode(entry, payload) for entry in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {"__dict__": {key: _encode(entry, payload)
                             for key, entry in value.items()}}
    raise NotImplementedError(
        f"the robust mode's state holds a {type(value).__name__}, which this "
        "checkpoint cannot represent; extend checkpoint._encode rather than drop it")


def _decode(value, handle):
    if isinstance(value, list):
        return [_decode(entry, handle) for entry in value]
    if isinstance(value, dict):
        if "__array__" in value:
            array = np.array(handle[value["__array__"]])
            return jnp.asarray(array) if value["jax"] else array
        if "__tuple__" in value:
            return tuple(_decode(entry, handle) for entry in value["__tuple__"])
        return {key: _decode(entry, handle) for key, entry in value["__dict__"].items()}
    return value


def save_supervisor(path, supervisor, damping, hamiltonian_in, mixer,
                    iteration: int) -> Path:
    """Write ``mixing_mode = 'robust'``'s loop state beside the mixer's history.

    What a resume needs and neither the state file nor the mixer file holds:
    every evolving field of the :class:`~defuqumat.scf.robust.Supervisor` (the
    kept input and the inner mixer's proposal from it, of which the next trial
    is a fraction; the windows the acceptance test and the stall are judged
    over; the rung; the step scale); the
    :class:`~defuqumat.scf.robust.HamiltonianDamping` of the bottom rung with
    its kept Hamiltonian and residual; ``hamiltonian_in``, the Hamiltonian the
    next iteration diagonalises on that rung, which is not a function of the
    saved density; and the mixer's ``beta`` and ``condition_limit``, which
    ``save_mixer`` treats as settings and the supervisor changes (an LDOS rung
    bakes the ``beta`` of its moment into the preconditioner). Every field of
    both dataclasses is written except :data:`_SUPERVISOR_DERIVED`, so a field
    added to either is carried, or raises here, and is never dropped.
    """
    path = Path(path)
    payload = {}
    meta = {
        "format": FORMAT_VERSION,
        "iteration": int(iteration),
        "supervisor": {f.name: _encode(getattr(supervisor, f.name), payload)
                       for f in dataclasses.fields(supervisor)
                       if f.name not in _SUPERVISOR_DERIVED},
        "damping": (None if damping is None else
                    {f.name: _encode(getattr(damping, f.name), payload)
                     for f in dataclasses.fields(damping)}),
        "hamiltonian_in": _encode(hamiltonian_in, payload),
        "mixer_beta": _encode(getattr(mixer, "beta", None), payload),
        "condition_limit": _encode(getattr(mixer, "condition_limit", None), payload),
    }
    payload["__meta__"] = np.frombuffer(json.dumps(meta).encode("utf-8"),
                                        dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    scratch = path.with_suffix(path.suffix + ".partial")
    np.savez(scratch, **payload)
    written = (scratch if scratch.suffix.endswith(".npz")
               else scratch.with_suffix(scratch.suffix + ".npz"))
    written.replace(path)
    return path


def load_supervisor(path) -> dict:
    """Read what :func:`save_supervisor` wrote.

    Returns a dict: ``supervisor`` (field name to value, to be set on a
    supervisor built the normal way), ``damping`` (a rebuilt
    :class:`~defuqumat.scf.robust.HamiltonianDamping`, or ``None``),
    ``hamiltonian_in``, ``mixer_beta``, ``condition_limit`` and ``iteration``.
    """
    from defuqumat.scf.robust import HamiltonianDamping

    path = Path(path)
    with np.load(path, allow_pickle=False) as handle:
        meta = json.loads(bytes(handle["__meta__"]).decode("utf-8"))
        if meta.get("format") != FORMAT_VERSION:
            raise ValueError(
                f"{path} is checkpoint format {meta.get('format')}, and this is "
                f"version {FORMAT_VERSION}")
        fields = {name: _decode(value, handle)
                  for name, value in meta["supervisor"].items()}
        damping = None
        if meta["damping"] is not None:
            damping = HamiltonianDamping()
            for name, value in meta["damping"].items():
                setattr(damping, name, _decode(value, handle))
        return {"supervisor": fields, "damping": damping,
                "hamiltonian_in": _decode(meta["hamiltonian_in"], handle),
                "mixer_beta": meta["mixer_beta"],
                "condition_limit": meta["condition_limit"],
                "iteration": int(meta["iteration"])}
