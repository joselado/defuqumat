"""A crystal stated as Python keywords, in angstrom and eV, as a ``pw.x`` input.

The other way into the code than a ``pw.x`` file: the lattice, the atoms, the
pseudopotentials, the cutoff and the k-mesh given as keywords, with lengths in
angstrom and energies in eV, which is how a structure is read off a paper or a
database. :meth:`defuqumat.Calculator.from_keywords` is the entry point.

**This module writes a ``pw.x`` input and nothing else.** Everything downstream
of the parser -- :func:`~defuqumat.system.builder.build_system`, its refusals and
its defaults, and :func:`~defuqumat.calculator.electrons_defaults` -- reads a
:class:`~defuqumat.io.pwin.PwInput`, so the keywords are translated into the
text of one and parsed by the same parser a file goes through. A keyword run and
a file run therefore cannot differ in a default or in a refusal: there is one
path after this module, not two. The text is also what :func:`pw_input_text`
returns, which is how a keyword run is taken to ``pw.x`` for an A/B.

Only what is typed for every crystal has a keyword of its own. Everything else
reaches its namelist unchanged through ``qe={'system': {...}, ...}``, in
``pw.x``'s own names and units (Ry, bohr), so no part of the code is out of
reach of this form. A ``qe`` key that one of the keywords already sets is
refused rather than resolved, since which of the two was meant is not
recoverable.

The conversions are the ones in :mod:`defuqumat.units`. Lengths are not
converted at all: they are written as ``pw.x``'s own ``A``, ``B``, ``C``,
``CELL_PARAMETERS angstrom`` and ``ATOMIC_POSITIONS angstrom``, which are in
angstrom already, so the bohr value is the one ``pw.x`` computes from the same
number.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from defuqumat.system.elements import atomic_weight
from defuqumat.units import RY_TO_EV

__all__ = ["KEYWORDS", "LATTICES", "pw_input_text"]

#: Named Bravais lattices and their ``ibrav``. ``a`` is the conventional cubic
#: constant for the three cubic ones, as in ``pw.x``; the hexagonal and
#: tetragonal ones take ``c`` as well, the orthorhombic one ``b`` and ``c``.
LATTICES = {
    "cubic": 1,
    "sc": 1,
    "fcc": 2,
    "bcc": 3,
    "hexagonal": 4,
    "tetragonal": 6,
    "orthorhombic": 8,
}

#: The lengths a named lattice needs beyond ``a``.
_NEEDS = {1: (), 2: (), 3: (), 4: ("c",), 6: ("c",), 8: ("b", "c")}

#: Smearing names a physicist types, as ``pw.x`` spells them.
_SMEARING = {
    "gaussian": "gaussian",
    "methfessel-paxton": "methfessel-paxton",
    "marzari-vanderbilt": "marzari-vanderbilt",
    "cold": "marzari-vanderbilt",
    "fermi-dirac": "fermi-dirac",
    "mp": "methfessel-paxton",
    "mv": "marzari-vanderbilt",
    "fd": "fermi-dirac",
}

#: Every keyword :func:`pw_input_text` takes. ``Calculator.from_keywords`` uses
#: this to tell an input keyword from a calculator option.
KEYWORDS = frozenset({
    "lattice", "a", "b", "c", "atoms", "coordinates", "pseudos", "masses",
    "ecut", "ecutrho", "kmesh", "kshift", "nbands", "functional", "smearing",
    "smearing_type", "spin", "spin_orbit", "qe",
})


def pw_input_text(
    *,
    lattice,
    atoms,
    pseudos: dict,
    ecut: float,
    kmesh,
    a: float | None = None,
    b: float | None = None,
    c: float | None = None,
    coordinates: str = "crystal",
    masses: dict | None = None,
    ecutrho: float | None = None,
    kshift=(0, 0, 0),
    nbands: int | None = None,
    functional: str | None = None,
    smearing: float | None = None,
    smearing_type: str = "gaussian",
    spin: str | None = None,
    spin_orbit: bool = False,
    qe: dict | None = None,
) -> str:
    """The ``pw.x`` input that the keywords describe, as text.

    Args:
        lattice: a name from :data:`LATTICES` (``'fcc'``, ``'hexagonal'``, ...),
            or three lattice vectors in angstrom, one per row.
        atoms: one entry per atom, ``(species, (x, y, z))``, or
            ``(species, (x, y, z), moment)`` for a magnetic run. The moment is
            the starting one, in Bohr magnetons: a number for ``spin='collinear'``
            and a cartesian vector for ``spin='noncollinear'``.
        pseudos: the pseudopotential file of each species, ``{'Si': 'Si.UPF'}``.
        ecut: the wavefunction cutoff, in eV.
        a, b, c: the lattice constants of a named lattice, in angstrom.
        coordinates: ``'crystal'`` (fractions of the lattice vectors, the
            default) or ``'angstrom'`` (cartesian).
        masses: atomic masses in amu by species. Defaults to the standard
            atomic weight of the element, ``pw.x``'s table.
        ecutrho: the density cutoff, in eV. Defaults to four times ``ecut``, as
            in ``pw.x``; an ultrasoft or PAW dataset wants eight to twelve.
        kmesh: ``(n1, n2, n3)`` for a Monkhorst-Pack mesh, or ``'gamma'`` for
            the single point, stored as a half sphere (P68). Required, as the
            cutoff is: neither has a default that is right for a crystal.
        kshift: ``(0 or 1, ...)``, a half-step shift of the mesh along each axis.
        nbands: the number of bands. Defaults to ``pw.x``'s choice.
        functional: overrides the functional the pseudopotentials were made with
            (``'pbe'``, ``'pbesol'``, ...).
        smearing: the smearing width, in eV, for a metal. ``None`` is an
            insulator with fixed occupations.
        smearing_type: ``'gaussian'``, ``'methfessel-paxton'``,
            ``'marzari-vanderbilt'`` (or ``'cold'``) or ``'fermi-dirac'``,
            and ``pw.x``'s ``'mp'``, ``'mv'`` and ``'fd'``.
        spin: ``None``, ``'collinear'`` or ``'noncollinear'``.
        spin_orbit: include spin-orbit coupling; needs fully relativistic
            pseudopotentials, and implies ``spin='noncollinear'``.
        qe: anything else, as ``pw.x`` namelist entries in ``pw.x``'s own units:
            ``{'electrons': {'conv_thr': 1e-10}, 'system': {'nosym': True}}``.
    """
    namelists: dict[str, dict[str, Any]] = {"control": {}, "system": {}, "electrons": {}}
    system = namelists["system"]
    cards: list[str] = []

    # --- the lattice -------------------------------------------------------
    if isinstance(lattice, str):
        name = lattice.lower()
        if name not in LATTICES:
            raise ValueError(
                f"unknown lattice {lattice!r}; the named ones are "
                f"{sorted(LATTICES)}, and any other is given as three vectors "
                "in angstrom"
            )
        ibrav = LATTICES[name]
        lengths = {"a": a, "b": b, "c": c}
        for needed in ("a",) + _NEEDS[ibrav]:
            if lengths[needed] is None:
                raise ValueError(f"lattice {lattice!r} needs {needed} (in angstrom)")
        for extra in {"b", "c"} - set(_NEEDS[ibrav]):
            if lengths[extra] is not None:
                raise ValueError(f"lattice {lattice!r} takes no {extra}")
        system["ibrav"] = ibrav
        system["A"] = float(a)
        if b is not None:
            system["B"] = float(b)
        if c is not None:
            system["C"] = float(c)
    else:
        vectors = np.asarray(lattice, dtype=float)
        if vectors.shape != (3, 3):
            raise ValueError("lattice vectors must be three rows of three components, in angstrom")
        if any(v is not None for v in (a, b, c)):
            raise ValueError(
                "a, b and c belong to a named lattice; with lattice vectors the "
                "lengths are in the vectors already"
            )
        system["ibrav"] = 0
        cards.append("CELL_PARAMETERS angstrom")
        cards.extend("  " + "  ".join(_num(x) for x in row) for row in vectors)

    # --- the atoms ---------------------------------------------------------
    if coordinates not in ("crystal", "angstrom"):
        raise ValueError(f"coordinates = {coordinates!r}: expected 'crystal' or 'angstrom'")
    species, rows, moments = [], [], []
    for entry in atoms:
        if len(entry) not in (2, 3):
            raise ValueError(
                f"atom {entry!r}: expected (species, (x, y, z)) or "
                "(species, (x, y, z), moment)"
            )
        label, position = entry[0], np.asarray(entry[1], dtype=float)
        if position.shape != (3,):
            raise ValueError(f"atom {entry!r}: the position needs three components")
        if label not in species:
            species.append(label)
        rows.append(f"  {label}  " + "  ".join(_num(x) for x in position))
        moments.append(entry[2] if len(entry) == 3 else None)

    missing = [s for s in species if s not in pseudos]
    if missing:
        raise ValueError(f"no pseudopotential given for {missing}")
    unused = sorted(set(pseudos) - set(species))
    if unused:
        raise ValueError(f"pseudopotentials given for species with no atom: {unused}")
    masses = dict(masses or {})
    system["nat"] = len(rows)
    system["ntyp"] = len(species)
    cards.append("ATOMIC_SPECIES")
    for label in species:
        mass = masses.pop(label) if label in masses else atomic_weight(label)
        cards.append(f"  {label}  {_num(mass)}  {pseudos[label]}")
    if masses:
        raise ValueError(f"masses given for species with no atom: {sorted(masses)}")
    cards.append(f"ATOMIC_POSITIONS {coordinates}")
    cards.extend(rows)

    # --- cutoffs, bands, functional -----------------------------------------
    system["ecutwfc"] = float(ecut) / RY_TO_EV
    if ecutrho is not None:
        system["ecutrho"] = float(ecutrho) / RY_TO_EV
    if nbands is not None:
        system["nbnd"] = int(nbands)
    if functional is not None:
        system["input_dft"] = str(functional)

    # --- occupations -------------------------------------------------------
    if smearing is not None:
        kind = _SMEARING.get(str(smearing_type).lower())
        if kind is None:
            raise ValueError(
                f"smearing_type = {smearing_type!r}; expected one of {sorted(_SMEARING)}"
            )
        system["occupations"] = "smearing"
        system["smearing"] = kind
        system["degauss"] = float(smearing) / RY_TO_EV

    # --- spin --------------------------------------------------------------
    if spin_orbit:
        if spin not in (None, "noncollinear"):
            raise ValueError("spin_orbit needs spin = 'noncollinear' (or no spin keyword)")
        spin = "noncollinear"
    if spin not in (None, "collinear", "noncollinear"):
        raise ValueError(f"spin = {spin!r}: expected None, 'collinear' or 'noncollinear'")
    given = [m for m in moments if m is not None]
    if given and spin is None:
        raise ValueError("an atom carries a moment, so give spin = 'collinear' or 'noncollinear'")
    if spin == "collinear":
        system["nspin"] = 2
    elif spin == "noncollinear":
        system["noncolin"] = True
    if spin_orbit:
        system["lspinorb"] = True
    if given:
        cards.append("STARTING_MOMENTS")
        for moment in moments:
            cards.append("  " + "  ".join(_num(x) for x in _moment(moment, spin)))

    # --- k-points ----------------------------------------------------------
    if isinstance(kmesh, str):
        if kmesh.lower() != "gamma":
            raise ValueError(f"kmesh = {kmesh!r}: expected (n1, n2, n3) or 'gamma'")
        cards.append("K_POINTS gamma")
    else:
        mesh = [int(n) for n in kmesh]
        shift = [int(s) for s in kshift]
        if len(mesh) != 3 or len(shift) != 3 or any(s not in (0, 1) for s in shift):
            raise ValueError("kmesh takes three integers and kshift three 0 or 1")
        cards.append("K_POINTS automatic")
        cards.append("  " + " ".join(str(n) for n in mesh + shift))

    # --- everything else, in pw.x's names ----------------------------------
    for namelist, entries in (qe or {}).items():
        target = namelists.setdefault(namelist.lower(), {})
        for key, value in entries.items():
            if key.lower() in {k.lower() for k in target}:
                raise ValueError(
                    f"qe['{namelist}']['{key}'] is already set by a keyword "
                    f"({target[_find(target, key)]!r}); give it in one place"
                )
            target[key] = value

    lines = []
    for namelist, entries in namelists.items():
        lines.append(f"&{namelist}")
        lines.extend(f"  {key} = {_value(value)}" for key, value in entries.items())
        lines.append("/")
    return "\n".join(lines + cards) + "\n"


def _moment(moment, spin):
    """A starting moment as a cartesian row, in Bohr magnetons."""
    if spin == "collinear":
        if np.ndim(moment) != 0:
            raise ValueError(
                f"moment {moment!r}: a collinear moment is one number, along z"
            )
        return (0.0, 0.0, float(moment))
    vector = np.asarray(moment, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"moment {moment!r}: a noncollinear moment is a cartesian vector")
    return tuple(vector)


def _find(entries, key):
    return next(k for k in entries if k.lower() == key.lower())


def _num(x) -> str:
    """A float as text that reads back to the same float."""
    return repr(float(x))


def _value(value) -> str:
    """A namelist value in Fortran's spelling."""
    if isinstance(value, bool):
        return ".true." if value else ".false."
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return _num(value)
    if isinstance(value, dict):
        raise ValueError(
            "an indexed pw.x variable is given in qe as its own keys, e.g. "
            "{'starting_magnetization(1)': 0.5}"
        )
    return f"'{value}'"
