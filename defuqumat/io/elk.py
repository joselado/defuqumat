"""Reading a converged Elk ground state, so a defuqumat SCF can start from it.

Elk is an all-electron LAPW code and defuqumat is a pseudopotential plane-wave
one, so the two converged densities are **different objects**: Elk's carries the
core electrons and the nuclear cusp, defuqumat's is the smooth pseudo valence
density normalised to ``nelec``. Nothing here changes that. What is transferred
is a **seed** -- a starting guess that is closer to the answer than a
superposition of isolated atomic charges -- and defuqumat's own SCF still runs on
top of it. Using an Elk density *fixed*, with no SCF above it, is refused by
name in :meth:`ElkState.density_on`.

``STATE.OUT`` is unformatted sequential Fortran binary written by Elk's
``writestate.f90``, and it is **not self-contained**: it holds no lattice
vectors, no atomic positions and no species names. The unit that can be read is
therefore an Elk *run directory* -- ``GEOMETRY.OUT`` for the geometry and
``STATE.OUT`` for the fields.

The layout, and the conventions that are easy to get wrong, are all from Elk
11.0.2 (``vendor/elk/src`` in the user's elkpy checkout; ``modmain.f90:1276``
carries the version):

* **One Fortran record holds two arrays.** ``write(100) rhomt, rhoir`` is a
  *single* record with both concatenated, and likewise for every magnetic pair.
  A reader that expects one array per record desyncs at the very first density
  record, and the symptom looks like a byte-order problem rather than a framing
  one.
* ``rhomt`` stores :math:`f_{lm}(r)`, **not** :math:`r^2 f_{lm}(r)`. The
  :math:`r^2` lives in the integration weight (``genrmesh.f90:66-68`` builds
  ``wr2mt`` as spline weights and *then* multiplies by :math:`r^2`).
* The muffin-tin block is written **unpacked**, ``(lmmaxo, nrmtmax, natmtot)``,
  with the ``lm > lmmaxi`` entries zeroed for ``ir <= nri``
  (``writestate.f90:58-60`` calls ``rfmtpack(.false., ...)``). Neither ``nrmti``
  nor ``lmmaxi`` is written, so the inner region needs no handling on the way
  in -- but see :mod:`defuqumat.io.elk_density` for where it does matter.
* **Padding past ``nrmt(is)`` is uninitialised buffer, not zeros.** Every slice
  here is ``[:, :nrmt(is), ias]``. Invisible with one species; silently wrong
  with two.
* ``rmt(is) = rsp(nrmt(is), is)`` exactly (``genrmesh.f90:56-59``), and that is
  the post-``autormt`` radius, which the species file's own ``rmt`` is not.
* ``ias`` runs species-outer, atom-inner (``init0.f90:78-91``).
* ``rhoir`` is raw: not multiplied by ``cfunir``, not by ``omega``, in
  e/bohr^3.
* **Elk is Hartree and defuqumat is Rydberg**, and only the *potentials* carry
  the factor of two -- ``vclmt``, ``vxcmt``, ``vsmt``, ``efermi``, ``dlefe``.
  Densities are e/bohr^3 in both and take no factor at all. The conversion
  happens here, at the ``io`` boundary, as the project rule requires.
* **An external field is not a potential and does not take the factor of two
  alone.** ``bfcmt`` and ``bfieldc`` are fields in Elk's units, coupled as
  ``cb sigma.B`` with ``cb = gfacte/(4 solsc)`` (``genbs.f90:15``, and
  ``energy.f90:77`` for the Zeeman energy), where this code's field is a
  Rydberg energy conjugate to the magnetization with energy ``-B.m``. So
  ``B_defuqumat = -2 cb B_Elk`` (:data:`ELK_FIELD_TO_RY`): the magnitude is
  ``PLAN.md`` P86's ``2 cb``, and the sign is the one
  :meth:`defuqumat.scf.fields.MagneticField.feedback` records against Elk's
  ``+(g_e/4c) sigma.B``. Reading a field as a Hartree energy is 273.75 times
  too large, which is P86's defect.
* ``tshift`` defaults to ``.true.``, which moves the origin onto the inversion
  centre. ``STATE.OUT`` and ``GEOMETRY.OUT`` are in the shifted frame and
  ``elk.in``'s ``atoms`` block is not, which is why the positions are read from
  ``GEOMETRY.OUT``.

**This is a gfortran build's file**: 4-byte record markers and a 4-byte logical.
An Intel-built ``STATE.OUT`` is not generally byte-compatible and nothing here
has been tested against one.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

__all__ = ["ElkState", "read_elk_state", "read_elk_geometry", "ElkGeometry"]

#: Hartree -> Rydberg. Potentials and ``efermi`` carry it; densities and
#: magnetizations do not, and an external magnetic field takes
#: :data:`ELK_FIELD_TO_RY` instead.
HARTREE_TO_RY = 2.0

#: Elk's electron g-factor and speed of light, ``modmain.f90:1264`` and
#: ``:1238``, both declared ``parameter``. ``solsc = sol * solscf`` and
#: ``solscf`` defaults to 1 (``readinput.f90``); a run that scales the speed of
#: light changes ``cb`` and is not what :data:`ELK_CB` describes.
ELK_GFACTE = 2.00231930436256
ELK_SOL = 137.035999084

#: ``cb = gfacte/(4 solsc) = 3.6529e-3``, the coupling of an external field in
#: Elk: ``genbs.f90`` adds ``cb * (bfcmt + bfieldc)`` to the Kohn-Sham magnetic
#: field and ``energy.f90`` writes the Zeeman energy with the same constant.
ELK_CB = ELK_GFACTE / (4.0 * ELK_SOL)

#: An Elk field (``bfcmt``, ``bfieldc``) -> this code's field in Ry.
#:
#: Elk's term is ``+cb sigma.B`` in Hartree, so a positive field *raises* the
#: majority channel; this code's is ``-B.m`` in Rydberg, with ``m`` the same
#: spin density. Matching the up-down splitting, ``2 cb |B_Elk|`` Ry, gives
#: ``B_defuqumat = -2 cb B_Elk``: the magnitude is ``PLAN.md`` P86's, and the sign
#: is the flip ``MagneticField.feedback`` records for the fixed-spin-moment
#: update (``B <- B + tau (M - M_fix)`` in Elk, a minus here).
ELK_FIELD_TO_RY = -HARTREE_TO_RY * ELK_CB


@dataclasses.dataclass(frozen=True)
class ElkGeometry:
    """What ``GEOMETRY.OUT`` carries: the cell and where the atoms are.

    ``avec`` is stored the way defuqumat stores a cell -- **row** ``i`` is the
    ``i``-th lattice vector in bohr -- which is the transpose of Elk's own
    ``avec(:,i)`` column layout. ``positions`` are in lattice (fractional)
    coordinates, ``natoms`` is per species and ``species`` are the species file
    names, in the order that fixes the ``ias`` index.
    """

    avec: np.ndarray            # (3, 3), row i is lattice vector i, bohr
    species: tuple[str, ...]
    natoms: tuple[int, ...]
    positions: np.ndarray       # (natmtot, 3), lattice coordinates
    #: ``(natmtot, 3)``: Elk's per-atom ``bfcmt``, **converted to this code's
    #: field in Ry** by :data:`ELK_FIELD_TO_RY`, so that it is already in the
    #: unit ``System.atomic_b_field`` and the ``LOCAL_MAGNETIC_FIELDS`` card are
    #: documented in. ``bfcmt`` is a field coupled through ``cb`` exactly as
    #: ``bfieldc`` is (``genbs.f90:37``, ``t1 = cb*(bfcmt + bfieldc)``), so the
    #: conversion is ``-2 cb`` and not the Hartree factor of two: the latter
    #: would hand a consumer a field 273.75 times too large, and with the sign
    #: that favours the moment Elk's field disfavours.
    magnetic_fields: np.ndarray

    @property
    def natmtot(self) -> int:
        return int(sum(self.natoms))

    @property
    def omega(self) -> float:
        """The cell volume in bohr^3."""
        return abs(float(np.linalg.det(self.avec)))

    def species_of(self) -> np.ndarray:
        """``idxis``: the species index of each ``ias``, species-outer."""
        return np.concatenate([np.full(n, i) for i, n in enumerate(self.natoms)])


def read_elk_geometry(path) -> ElkGeometry:
    """Parse ``GEOMETRY.OUT``.

    The format is Elk's own ``writegeom.f90``: labelled blocks ``scale``,
    ``scale1..3``, ``avec``, ``atoms``. The scale factors multiply the lattice
    vectors -- ``scale`` all three and ``scale1..3`` one each -- and Elk applies
    them in ``readinput.f90`` before anything else sees ``avec``, so they are
    applied here too.
    """
    path = Path(path)
    tokens = _blocks(path.read_text())

    scale = float(tokens.get("scale", ["1.0"])[0])
    scales = np.array(
        [float(tokens.get(f"scale{i}", ["1.0"])[0]) for i in (1, 2, 3)]
    )
    if "avec" not in tokens:
        raise ValueError(f"{path} has no 'avec' block; is it a GEOMETRY.OUT?")
    avec = np.array([float(x) for x in tokens["avec"][:9]]).reshape(3, 3)
    avec = avec * (scale * scales)[:, None]

    atoms = tokens.get("atoms")
    if atoms is None:
        raise ValueError(f"{path} has no 'atoms' block; is it a GEOMETRY.OUT?")
    cursor = 0
    nspecies = int(float(atoms[cursor])); cursor += 1
    species, natoms, positions, fields = [], [], [], []
    for _ in range(nspecies):
        species.append(atoms[cursor].strip().strip("'").strip('"')); cursor += 1
        n = int(float(atoms[cursor])); cursor += 1
        natoms.append(n)
        for _ in range(n):
            row = [float(x) for x in atoms[cursor:cursor + 6]]; cursor += 6
            positions.append(row[:3])
            fields.append(row[3:6])

    return ElkGeometry(
        avec=avec,
        species=tuple(species),
        natoms=tuple(natoms),
        positions=np.array(positions, dtype=float).reshape(-1, 3),
        magnetic_fields=(
            np.array(fields, dtype=float).reshape(-1, 3) * ELK_FIELD_TO_RY
        ),
    )


def _blocks(text: str) -> dict[str, list[str]]:
    """Elk's block format: a bare keyword line, then its values until the next.

    Everything after a ``:`` on a line is Elk's own annotation
    (``: nspecies``, ``: natoms; atpos, bfcmt below``) and is dropped, as is a
    ``!`` comment.
    """
    out: dict[str, list[str]] = {}
    key = None
    for line in text.splitlines():
        line = line.split("!")[0].split(":")[0].strip()
        if not line:
            continue
        first = line.split()[0]
        if len(line.split()) == 1 and not _numeric(first) and not first.startswith("'"):
            key = first
            out.setdefault(key, [])
            continue
        if key is not None:
            out[key].extend(line.split())
    return out


def _numeric(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


@dataclasses.dataclass(frozen=True)
class ElkState:
    """A converged Elk ground state, read from an Elk run directory.

    The density is in e/bohr^3 and the potentials are in **Rydberg**, converted
    on the way in. Everything else is Elk's own, unchanged.

    Reconstructing the density at a point, integrating it, and turning it into a
    seed for a defuqumat SCF all live in :mod:`defuqumat.io.elk_density` and are
    reached through the methods here.
    """

    version: tuple[int, int, int]
    geometry: ElkGeometry
    lmmaxo: int
    nrmtmax: int
    nrcmtmax: int
    nrmt: tuple[int, ...]
    rsp: tuple[np.ndarray, ...]
    nrcmt: tuple[int, ...]
    rcmt: tuple[np.ndarray, ...]
    ngridg: tuple[int, int, int]
    ngvec: int
    ndmag: int
    nspinor: int
    fsmtype: int
    ftmtype: int
    dftu: int
    lmmaxdm: int
    xcgrad: int
    efermi: float          # Ry
    dlefe: float           # Ry
    rhomt: np.ndarray      # (lmmaxo, nrmtmax, natmtot), e/bohr^3
    rhoir: np.ndarray      # ngridg, e/bohr^3
    vclmt: np.ndarray      # Ry
    vclir: np.ndarray
    vxcmt: np.ndarray
    vxcir: np.ndarray
    vsmt: np.ndarray
    vsir: np.ndarray
    directory: Path
    # The magnetic records, present only for a spin-polarized state
    # (``writestate.f90:84-108``). ``magmt`` is ``(lmmaxo, nrmtmax, natmtot,
    # ndmag)`` and ``magir`` is ``ngridg + (ndmag,)``, in e/bohr^3 like the
    # density -- ``m = n_up - n_down``, no factor. ``bxc`` and ``bs`` are the
    # exchange-correlation and Kohn-Sham magnetic fields, energies conjugate to
    # ``m`` and so in **Rydberg** like the potentials; ``bsmt`` is on the
    # **coarse** radial mesh ``nrcmt``. Nothing downstream consumes the two
    # fields; they are read so the file is read whole and its framing checked.
    magmt: np.ndarray | None = None
    magir: np.ndarray | None = None
    bxcmt: np.ndarray | None = None
    bxcir: np.ndarray | None = None
    bsmt: np.ndarray | None = None
    bsir: np.ndarray | None = None
    # ``vmatmt``, Elk's DFT+U potential matrix in each muffin tin, in Ry, as
    # ``(lmmaxdm, nspinor, lmmaxdm, nspinor, natmtot)`` (``readstate.f90:251``),
    # or ``None`` when the state is not DFT+U. It is in Elk's own muffin-tin
    # ``lm`` basis, built on Elk's radial functions, and is read so the file is
    # consumed to its end and its framing checked; nothing here transfers it,
    # since a pseudopotential run's Hubbard projectors are different orbitals.
    vmatmt: np.ndarray | None = None

    # -- what the file says about itself --------------------------------------

    @property
    def spinpol(self) -> bool:
        """Whether the state carries a magnetization (Elk's ``spinpol``)."""
        return self.magmt is not None

    def magnetization_component(self, idm: int = 0) -> "ElkState":
        """This state with component ``idm`` of the magnetization in the density's place.

        Every reconstruction routine in :mod:`defuqumat.io.elk_density` reads
        ``rhomt``/``rhoir``, and a magnetization component is reconstructed by
        exactly the same ``rfpts`` route (Elk's own ``vecplot`` hands
        ``magmt``/``magir`` to ``plot3d`` as it hands ``rhomt``/``rhoir`` to
        it in ``rhoplot``), so the view is the whole of the transfer rule's
        reconstruction: nothing is reimplemented for the second field.
        """
        if self.magmt is None:
            raise ValueError("this Elk state is not spin-polarized")
        return dataclasses.replace(
            self, rhomt=self.magmt[..., idm], rhoir=self.magir[..., idm]
        )

    @property
    def natmtot(self) -> int:
        return self.geometry.natmtot

    @property
    def nspecies(self) -> int:
        return len(self.geometry.natoms)

    @property
    def rmt(self) -> np.ndarray:
        """``rmt(is) = rsp(nrmt(is), is)``, the post-``autormt`` radius."""
        return np.array([self.rsp[i][self.nrmt[i] - 1] for i in range(self.nspecies)])

    @property
    def omega(self) -> float:
        return self.geometry.omega

    @property
    def lmaxo(self) -> int:
        return int(round(np.sqrt(self.lmmaxo))) - 1

    @property
    def nri(self) -> np.ndarray:
        """``nrmti`` per species, recovered from the zeroed inner block.

        ``rfmtpack(.false., ...)`` zeroes the ``lm > lmmaxi`` entries for
        ``ir <= nri`` on the way out, and neither ``nrmti`` nor ``lmmaxi`` is
        written to the file. The zeros are exact, so ``nri`` is the number of
        leading radial points whose high-``lm`` tail vanishes -- which is what
        this returns. Where the whole sphere is spherically symmetric there is
        nothing to find and it returns ``nrmt``, which is harmless: the terms it
        would then include are zero anyway.
        """
        out = []
        species = self.geometry.species_of()
        top = self.lmaxo ** 2   # the first ``lm`` of the highest ``l`` shell
        for i, nr in enumerate(self.nrmt):
            atoms = np.nonzero(species == i)[0]
            # The highest ``l`` shell is certainly above ``lmaxi``, whatever
            # ``lmaxi`` was. Channels that vanish on the whole mesh carry no
            # information about the boundary and are skipped.
            counts = []
            for lm in range(top, self.lmmaxo):
                column = np.abs(self.rhomt[lm, :nr, atoms]).max(axis=0)
                nonzero = np.nonzero(column)[0]
                if nonzero.size:
                    counts.append(int(nonzero[0]))
            out.append(min(counts) if counts else nr)
        return np.array(out)

    # -- what it can do -------------------------------------------------------

    def evaluate_at(self, points, coordinates: str = "lattice") -> np.ndarray:
        """The density at arbitrary points, exactly as Elk's ``rfpts`` does it.

        See :func:`defuqumat.io.elk_density.evaluate_at`.
        """
        from defuqumat.io.elk_density import evaluate_at

        return evaluate_at(self, points, coordinates=coordinates)

    def muffin_tin_charges(self) -> np.ndarray:
        """``chgmt`` per atom: :math:`4\\pi Y_{00}\\int f_{00}(r) r^2 dr`."""
        from defuqumat.io.elk_density import muffin_tin_charges

        return muffin_tin_charges(self)

    def characteristic_function(self) -> np.ndarray:
        """Elk's ``cfunir``: the Fourier-truncated step, 0 inside, 1 outside."""
        from defuqumat.io.elk_density import characteristic_function

        return characteristic_function(self)

    def interstitial_charge(self) -> float:
        """``chgir``, Elk's own way: ``rhoir`` against the *truncated* step."""
        from defuqumat.io.elk_density import interstitial_charge

        return interstitial_charge(self)

    def density_on(self, calculation, renormalise: bool = True, report=None):
        """The density reconstructed on a defuqumat calculation's dense grid.

        See :func:`defuqumat.io.elk_density.density_on`.
        """
        from defuqumat.io.elk_density import density_on

        return density_on(self, calculation, renormalise=renormalise, report=report)

    # -- reading --------------------------------------------------------------

    @classmethod
    def read(cls, directory) -> "ElkState":
        """Read ``STATE.OUT`` and ``GEOMETRY.OUT`` from an Elk run directory."""
        return read_elk_state(directory)


def read_elk_state(directory) -> ElkState:
    """Read an Elk run directory: ``STATE.OUT`` beside ``GEOMETRY.OUT``.

    **What this refuses, and why.** Each of these is a whole further piece of
    physics rather than a format detail, and a seed built from a state whose
    extra content is silently dropped is a wrong answer that starts:

    * a **noncollinear** spin-polarized state (``ndmag = 3``), and spin
      spirals with it: ``STATE.OUT`` records neither ``spinsprl`` nor
      ``vqlss``, so a spiral's magnetization, which Elk stores in the frame
      that turns with ``q``, cannot be told from a noncollinear texture's at
      read time. A **collinear** one (``ndmag = 1``) is read, magnetization
      and fields included;
    * a **fixed spin moment** state (``fsmtype``), whose constraint fields are
      trailing records and whose moment is the constrained one rather than the
      ground state's;
    * a **spin-polarized DFT+U** state (``spinpol`` with ``dftu``): each piece
      is read on its own, and the pair, with ``vmatmt`` at ``nspinor = 2`` after
      the magnetic records, has been measured by neither (refused at the
      integration of the two);
    * a **fixed tensor moment** state (``ftmtype``). Elk refuses tensor
      moments without spin-orbit coupling (``init0.f90:239-247``), and
      spin-orbit coupling switches ``spinpol`` on (``init0.f90:126``), so a
      genuine file of this kind is spin-polarized; the guard is kept by name
      because the moment it fixes is a constraint on Elk's functional that
      nothing here implements;
    * anything written by Elk **older than 2.0.0**, which ``readstate.f90``
      refuses itself.

    An unpolarized **DFT+U** state (``dftu``) is read. Its one extra record is ``vmatmt``,
    the Hubbard *potential* matrix in Elk's muffin-tin basis, written after
    the four density and potential pairs (``writestate.f90:122-125``); the
    occupation matrix is not in the file at all, since Elk rebuilds it from
    its states (``gndstate.f90:194-201``). What a seed carries is the density
    alone, see :func:`defuqumat.io.elk_density.density_on`.
    """
    from scipy.io import FortranFile

    directory = Path(directory)
    state_path = directory / "STATE.OUT"
    geometry_path = directory / "GEOMETRY.OUT"
    if not state_path.exists():
        raise FileNotFoundError(
            f"no STATE.OUT in {directory}. An Elk state is a run *directory*: "
            "STATE.OUT holds no lattice vectors, no positions and no species "
            "names, so GEOMETRY.OUT has to be beside it."
        )
    if not geometry_path.exists():
        raise FileNotFoundError(
            f"no GEOMETRY.OUT in {directory}. STATE.OUT is not self-contained "
            "-- it carries neither the cell nor the atomic positions -- and "
            "elk.in is not a substitute, because Elk's default tshift moves "
            "the origin onto the inversion centre and only GEOMETRY.OUT is "
            "written in that shifted frame."
        )

    geometry = read_elk_geometry(geometry_path)

    with FortranFile(state_path, "r") as f:
        version = tuple(int(x) for x in f.read_ints(np.int32))
        if version[0] < 2:
            raise NotImplementedError(
                f"STATE.OUT from Elk {version[0]}.{version[1]}.{version[2]}: "
                "readstate.f90 refuses anything earlier than 2.0.0 and so does "
                "this reader"
            )
        spinpol = bool(f.read_record(np.int32)[0])
        nspecies = int(f.read_ints(np.int32)[0])
        if nspecies != len(geometry.natoms):
            raise ValueError(
                f"STATE.OUT has {nspecies} species and GEOMETRY.OUT has "
                f"{len(geometry.natoms)}; these are not the same run"
            )
        lmmaxo = int(f.read_ints(np.int32)[0])
        nrmtmax = int(f.read_ints(np.int32)[0])
        nrcmtmax = int(f.read_ints(np.int32)[0])

        nrmt, rsp, nrcmt, rcmt = [], [], [], []
        for i in range(nspecies):
            natoms_i = int(f.read_ints(np.int32)[0])
            if natoms_i != geometry.natoms[i]:
                raise ValueError(
                    f"species {i + 1}: STATE.OUT has {natoms_i} atoms and "
                    f"GEOMETRY.OUT has {geometry.natoms[i]}"
                )
            nrmt.append(int(f.read_ints(np.int32)[0]))
            rsp.append(f.read_reals(np.float64))
            nrcmt.append(int(f.read_ints(np.int32)[0]))
            rcmt.append(f.read_reals(np.float64))

        ngridg = tuple(int(x) for x in f.read_ints(np.int32))
        ngvec = int(f.read_ints(np.int32)[0])
        ndmag = int(f.read_ints(np.int32)[0])
        nspinor = int(f.read_ints(np.int32)[0])
        fsmtype = int(f.read_ints(np.int32)[0])
        # ``ftmtype`` only exists from version 3 (readstate.f90:107-111).
        ftmtype = int(f.read_ints(np.int32)[0]) if version[0] > 2 else 0
        dftu = int(f.read_ints(np.int32)[0])
        lmmaxdm = int(f.read_ints(np.int32)[0])
        xcgrad = (
            int(f.read_ints(np.int32)[0])
            if (version[0] > 5 or (version[0] == 5 and version[1] > 0))
            else 0
        )
        # ``efermi`` from 9.6 and ``dlefe`` from 10.7. Below that Elk falls back
        # on a separate EFERMI.OUT, which is not read here -- the seed does not
        # use either number, and a NaN says so rather than a plausible zero.
        if version[0] > 9 or (version[0] == 9 and version[1] > 5):
            efermi = float(f.read_reals(np.float64)[0])
        else:
            efermi = float("nan")
        if version[0] > 10 or (version[0] == 10 and version[1] > 6):
            dlefe = float(f.read_reals(np.float64)[0])
        else:
            dlefe = float("nan")

        if spinpol and ndmag != 1:
            raise NotImplementedError(
                f"this STATE.OUT is noncollinear spin-polarized (ndmag = "
                f"{ndmag}), and only an unpolarized or a collinear (ndmag = 1) "
                "Elk state can be transferred. STATE.OUT records neither "
                "spinsprl nor vqlss, so a spin spiral, whose magnetization Elk "
                "stores in the frame that turns with q, cannot be told apart "
                "from a noncollinear texture here, and seeding a noncollinear "
                "run from it would start from the wrong magnetization without "
                "saying so."
            )
        if spinpol and fsmtype != 0:
            raise NotImplementedError(
                f"this STATE.OUT is a fixed spin moment state (fsmtype = "
                f"{fsmtype}). Its moment is the constrained one and its "
                "constraint fields are trailing records nothing here consumes; "
                "seeding from it is not implemented."
            )
        if spinpol and dftu != 0:
            raise NotImplementedError(
                f"this STATE.OUT is a spin-polarized DFT+U state (spinpol, "
                f"dftu = {dftu}). A collinear spin-polarized state and an "
                "unpolarized DFT+U state are each read, but the pair, with the "
                "DFT+U record at nspinor = 2 after the magnetic records, has "
                "not been measured against Elk, so seeding from it is not "
                "implemented."
            )
        if ftmtype != 0:
            raise NotImplementedError(
                f"this STATE.OUT carries a fixed tensor moment (ftmtype = "
                f"{ftmtype}). The moment is a constraint on Elk's functional, "
                "with its own potential matrix (vmftm), and nothing here "
                "implements it, so a seed taken from it would start a run "
                "whose functional is not the one the density was converged "
                "for. Elk itself only writes this for a spin-orbit, and so "
                "spin-polarized, run (init0.f90:126 and :239-247)."
            )

        # ``natmtot`` is the sum of ``natoms(is)`` over **all** species, not
        # ``natoms(1)``: ``writestate.f90:57-62`` dimensions the muffin-tin
        # block ``(lmmaxo, nrmtmax, natmtot)``, so a reader that confuses the
        # two reads the wrong number of bytes the moment a second species
        # exists. **Untested by every fixture here -- they have one species
        # each** -- which is why the record length is asserted as an equality
        # below rather than left to be noticed.
        natmtot = geometry.natmtot
        ngtot = int(np.prod(ngridg))
        nmt = lmmaxo * nrmtmax * natmtot

        def _pair():
            """One record holding ``(muffin-tin block, interstitial block)``."""
            record = f.read_reals(np.float64)
            if record.size != nmt + ngtot:
                raise ValueError(
                    f"record of {record.size} reals where {nmt + ngtot} were "
                    "expected. Elk writes the muffin-tin and interstitial "
                    "arrays as ONE record; a reader that takes them as two "
                    "desyncs here."
                )
            mt = record[:nmt].reshape((lmmaxo, nrmtmax, natmtot), order="F")
            return mt, record[nmt:].reshape(ngridg, order="F")

        # The muffin-tin block is dimensioned to ``nrmtmax``, and for any
        # species with ``nrmt(is) < nrmtmax`` the rows past ``nrmt(is)`` are
        # **uninitialised buffer, not zeros**. Every consumer slices
        # ``[:, :nrmt(is), ias]``; nothing here integrates the padding.
        # **Untested by every fixture here -- one species each, so
        # ``nrmt(1) == nrmtmax`` identically and the bug cannot show.**
        rhomt, rhoir = _pair()
        vclmt, vclir = _pair()
        vxcmt, vxcir = _pair()
        vsmt, vsir = _pair()

        magnetic = {}
        if spinpol:
            # ``writestate.f90:84-108``: three more one-record pairs, each with
            # ``ndmag`` as its slowest axis on both halves. ``bsmt`` is packed
            # on the **coarse** radial mesh, ``nrcmtmax``, so its record is
            # shorter than the other two and a reader that sized it like
            # ``magmt`` reads past it.
            def _vector_pair(nradial, name):
                nblock = lmmaxo * nradial * natmtot
                record = f.read_reals(np.float64)
                if record.size != ndmag * (nblock + ngtot):
                    raise ValueError(
                        f"{name} record of {record.size} reals where "
                        f"{ndmag * (nblock + ngtot)} were expected "
                        f"(ndmag = {ndmag}, radial mesh {nradial}). Elk writes "
                        "the muffin-tin and interstitial arrays as ONE record, "
                        "and bsmt is on the coarse mesh nrcmtmax rather than "
                        "nrmtmax."
                    )
                mt = record[:ndmag * nblock].reshape(
                    (lmmaxo, nradial, natmtot, ndmag), order="F"
                )
                ir = record[ndmag * nblock:].reshape((*ngridg, ndmag), order="F")
                return mt, ir

            magmt, magir = _vector_pair(nrmtmax, "magnetization")
            bxcmt, bxcir = _vector_pair(nrmtmax, "B_xc")
            bsmt, bsir = _vector_pair(nrcmtmax, "B_s")
            magnetic = dict(
                magmt=magmt, magir=magir,
                bxcmt=bxcmt * HARTREE_TO_RY, bxcir=bxcir * HARTREE_TO_RY,
                bsmt=bsmt * HARTREE_TO_RY, bsir=bsir * HARTREE_TO_RY,
            )

        # ``writestate.f90:112-119``: a meta-GGA state carries one more pair,
        # ``wxcmt, wxcir``, *before* the DFT+U record. Nothing here uses it,
        # but it has to be consumed or ``vmatmt`` below is read from the
        # wrong record.
        if xcgrad in (3, 4, 5, 6):
            _pair()

        # ``writestate.f90:122-125`` and ``readstate.f90:251``: one complex
        # record ``vmatmt(lmmaxdm, nspinor, lmmaxdm, nspinor, natmtot)``.
        vmatmt = None
        if dftu != 0:
            nvmat = lmmaxdm * nspinor * lmmaxdm * nspinor * natmtot
            record = f.read_record(np.complex128)
            if record.size != nvmat:
                raise ValueError(
                    f"DFT+U record of {record.size} complex numbers where "
                    f"{nvmat} = lmmaxdm^2 nspinor^2 natmtot were expected "
                    f"(lmmaxdm = {lmmaxdm}, nspinor = {nspinor}, natmtot = "
                    f"{natmtot}). Either the file was written by an Elk whose "
                    "layout this reader does not know, or a record before it "
                    "was not consumed."
                )
            vmatmt = record.reshape(
                (lmmaxdm, nspinor, lmmaxdm, nspinor, natmtot), order="F"
            ) * HARTREE_TO_RY

    return ElkState(
        version=version,
        geometry=geometry,
        lmmaxo=lmmaxo,
        nrmtmax=nrmtmax,
        nrcmtmax=nrcmtmax,
        nrmt=tuple(nrmt),
        rsp=tuple(rsp),
        nrcmt=tuple(nrcmt),
        rcmt=tuple(rcmt),
        ngridg=ngridg,
        ngvec=ngvec,
        ndmag=ndmag,
        nspinor=nspinor,
        fsmtype=fsmtype,
        ftmtype=ftmtype,
        dftu=dftu,
        lmmaxdm=lmmaxdm,
        xcgrad=xcgrad,
        efermi=efermi * HARTREE_TO_RY,
        dlefe=dlefe * HARTREE_TO_RY,
        rhomt=rhomt,
        rhoir=rhoir,
        vclmt=vclmt * HARTREE_TO_RY,
        vclir=vclir * HARTREE_TO_RY,
        vxcmt=vxcmt * HARTREE_TO_RY,
        vxcir=vxcir * HARTREE_TO_RY,
        vsmt=vsmt * HARTREE_TO_RY,
        vsir=vsir * HARTREE_TO_RY,
        directory=directory,
        **magnetic,
        vmatmt=vmatmt,
    )
