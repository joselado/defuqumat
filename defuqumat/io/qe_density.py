"""Reading and writing a converged ``pw.x`` state: its save directory.

``pw.x`` leaves the self-consistent state in ``<outdir>/<prefix>.save/``, and
this reads and writes the part of it a continuation needs: the density, and
beside it the PAW one-centre occupations ``becsum`` (``paw.txt``), the Hubbard
occupation matrix (``occup.txt``) and the meta-GGA kinetic energy density
(``ekin-density``). The routines transcribed are ``PW/src/io_rho_xml.f90``
(``write_scf``/``read_scf``), ``Modules/io_base.f90``
(``write_rhog``/``read_rhog``) and, for the order of the G-vectors a file for
``pw.x`` must have, ``Modules/recvec_subs.f90`` (``ggen``,
:mod:`defuqumat.io.qe_gvectors`).

**The density file.** ``charge-density.dat`` is Fortran unformatted, one record
each for ``(gamma_only, ngm_g, nspin)``, the three reciprocal vectors
``bg * tpiba`` in 1/bohr, the Miller indices ``(3, ngm_g)``, and then one
``complex128[ngm_g]`` record per component. ``charge-density.hdf5`` (a QE built
with HDF5) carries the same arrays as attributes and datasets; it is read, never
written. The coefficients are those of ``rho(r) = sum_G rho(G) exp(iGr)`` in
e/bohr^3, so ``rho(G = 0) * omega`` is the electron count. ``read_rhog``
scatters the coefficients in the order they are stored and reads the Miller
indices only when the ``gamma_only`` flags differ, so a written file lists
them in ``pw.x``'s order, and ``write_rhog`` here reproduces ``pw.x``'s own
files byte for byte.

**The spin layout is not defuqumat's in one case.** Since QE 6.4 ``rho%of_g``
is ``(n, m_z)`` for ``nspin = 2`` and ``(n, m_x, m_y, m_z)`` for ``nspin = 4``,
and it is written as held (``read_rhog`` converts nothing); a noncollinear run
without a magnetization writes the charge alone (``write_scf``, ``nspin_ = 1``
when ``noncolin .and. .not. domag``). defuqumat's collinear density is
``(up, down)``, so that pair is split on reading and joined on writing, and the
noncollinear one is taken as is. Measured on an O2 triplet: the second
component's ``G = 0`` times ``omega`` is 2.00, the "total magnetization" line
of ``pw.x``'s output, where an ``(up, down)`` file would read 5; written as
``(up, down)`` on purpose, ``pw.x`` renormalises it and needs 12 iterations
where the right file needs 1.

**``paw.txt``** is ``rho%bec`` written list-directed, shape
``(nhm (nhm + 1) / 2, nat, nspin)``: the upper triangle of each atom's
matrix in ``ijtoh`` order (``ih`` outer, ``jh >= ih`` inner, ``init_us_1.f90``)
and the off-diagonal entries doubled, ``sum_bec`` storing
``2 Re <psi|beta_i><beta_j|psi>`` once for the pair. ``nhm`` is the largest
``nh`` over all species. Unlike the density it stays ``(up, down)`` for
``nspin = 2`` (``PAW_atomic_becsum`` builds it so and ``rhoz_or_updw`` touches
only ``of_r``/``of_g``), and for a noncollinear run it is the four real Pauli
components ``(n, m_x, m_y, m_z)`` of ``add_becsum_nc``/``add_becsum_so``,
defuqumat's own layout. **Its ``nspin`` is ``lsda_mod``'s, 4 for every
noncollinear run**, so a spin-orbit run without a magnetization writes three
zero components: the number of components is read off the file's size.

**``occup.txt``** is ``rho%ns(m1, m2, is, na)`` list-directed, or the complex
``ns_nc`` of a spinor run (``is = npol (is1 - 1) + is2``, written ``(re,im)``),
over every atom at ``2 Hubbard_lmax + 1``. Measured on a tilted Ni moment and
on relativistic BN, where the spin-flip blocks and (under spin-orbit coupling)
the orbital part are complex: the conjugate, the ``m``-transpose and the
swapped spin pair are each 5e-4 off defuqumat's converged ``ns`` where the
layout above is 7e-7 off.

**``ekin-density``** is ``rho%kin_g`` in the density file's format, and it is
``(up, down)`` at ``nspin = 2`` (``potinit.f90``: "for LSDA rho is (tot,magn),
rho_kin is (up,down)"), defuqumat's layout for ``tau``, in Ry.

**What is refused, each by name:** a mismatch of cell, atomic positions,
electron count, density cutoff or spin regime between the save directory and
the calculation (a density belongs to one G-sphere on one lattice); a Hubbard
term on one side only, with other projectors or another ``lda_plus_u_kind``;
and what ``occup.txt`` can hold that defuqumat has no counterpart for (a
background manifold, ``Hubbard_V``, the orbital-resolved form). Writing refuses
a spin spiral and the HDF5 form.

**What the written XML is.** ``pw.x``'s ``scf`` restart reads the density files
only; the XML written beside them holds what :func:`read_qe_save` checks and
nothing more, so ``pp.x``, ``projwfc.x`` and an ``nscf`` run, which go through
``read_file``, do not read it (the user's choice of 2026-11-17: the restart
only, after which ``pw.x`` writes its own full save).

Measured in ``PLAN.md`` P160 (reading) and P180 (writing, and the spinor PAW,
DFT+U and meta-GGA reading).

Since both codes hold the same pseudo valence density, a state from a run with
the same input is a continuation rather than a guess in both directions: the
first iteration's ``dr2`` reads at the level of the file's own ``conv_thr``.
"""

from __future__ import annotations

import dataclasses
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["QESave", "read_qe_save", "read_rhog", "write_qe_save", "write_rhog"]


def read_rhog(path):
    """``read_rhog``: ``(gamma_only, b, miller, rho_g)`` from one density file.

    ``b`` is ``(3, 3)`` with the reciprocal vectors as rows in 1/bohr,
    ``miller`` is ``(ngm, 3)`` and ``rho_g`` is ``(nspin, ngm)``.
    """
    path = Path(path)
    if path.suffix == ".hdf5":
        import h5py

        with h5py.File(path, "r") as f:
            flag = f.attrs["gamma_only"]
            flag = flag.decode() if isinstance(flag, bytes) else str(flag)
            gamma_only = flag.strip().upper() in (".TRUE.", "T", "TRUE")
            nspin = int(f.attrs["nspin"])
            mill = f["MillerIndices"]
            b = np.array([mill.attrs[f"bg{i}"] for i in (1, 2, 3)], dtype=float)
            miller = np.asarray(mill[...], dtype=np.int64).reshape(-1, 3)
            names = (["rhotot_g", "rhodiff_g"] if nspin <= 2
                     else ["rhotot_g", "m_x", "m_y", "m_z"])[:nspin]
            rho_g = np.stack([_complex(f[name][...]) for name in names])
        return gamma_only, b, miller, rho_g

    from scipy.io import FortranFile

    with FortranFile(path, "r") as f:
        header = f.read_record(np.int32)
        gamma_only, ngm, nspin = bool(header[0]), int(header[1]), int(header[2])
        b = f.read_record(np.float64).reshape(3, 3)
        miller = f.read_record(np.int32).reshape(ngm, 3).astype(np.int64)
        rho_g = np.stack([f.read_record(np.complex128) for _ in range(nspin)])
    return gamma_only, b, miller, rho_g


def _complex(array):
    """An HDF5 dataset as complex: QE writes it as interleaved pairs of doubles.

    ``qeh5_wplan_complex`` doubles the first dimension (``qeh5_module.f90:380``).
    This branch has not been run against a ``pw.x`` built with HDF5.
    """
    array = np.asarray(array)
    if np.iscomplexobj(array):
        return array.reshape(-1)
    return array.reshape(-1, 2) @ np.array([1.0, 1.0j])


def _floats(element) -> np.ndarray:
    return np.array(element.text.split(), dtype=float)


def _read_list_directed(text: str) -> np.ndarray:
    """A list-directed Fortran file of reals, or of ``(re,im)`` complex pairs."""
    if "(" not in text:
        return np.array(text.replace("D", "E").replace("d", "e").split(), dtype=float)
    pairs = text.replace("D", "E").replace("d", "e").replace(")", " ").split("(")[1:]
    values = [complex(*map(float, pair.split(",")[:2])) for pair in pairs]
    return np.array(values, dtype=complex)


@dataclass(frozen=True)
class QESave:
    """A ``pw.x`` save directory: the density and what it was computed for.

    ``rho_g`` is ``(nspin, ngm)`` in QE's own layout (see the module
    docstring); ``becsum`` is the raw ``paw.txt`` array,
    ``(nhm (nhm + 1) / 2, nat, nspin_mag)``, or ``None``.
    """

    directory: Path
    gamma_only: bool
    b: np.ndarray
    miller: np.ndarray
    rho_g: np.ndarray
    at: np.ndarray            # (3, 3) lattice vectors as rows, bohr
    positions: np.ndarray     # (nat, 3) cartesian, bohr
    labels: tuple
    nelec: float
    ecutrho: float            # Ry
    lsda: bool
    noncolin: bool
    functional: str
    hubbard: bool
    converged: bool | None
    becsum: np.ndarray | None
    #: ``occup.txt`` as read, flat: real for a collinear run, complex for a
    #: spinor one (``ns_nc``). ``None`` without DFT+U.
    occupations: np.ndarray | None = None
    #: ``ekin-density``, ``(nspin, ngm)`` coefficients of the kinetic energy
    #: density, ``(up, down)`` at ``nspin = 2``. ``None`` without a meta-GGA.
    ekin_g: np.ndarray | None = None
    #: From ``<dftU>``: ``lda_plus_u_kind``, ``U_projection_type``, and whether
    #: the run had what ``occup.txt`` adds after ``ns`` or holds instead of it
    #: (a background manifold, ``Hubbard_V``) or the orbital-resolved form.
    hubbard_kind: int | None = None
    hubbard_projectors: str | None = None
    hubbard_extra: tuple = ()

    @classmethod
    def read(cls, directory) -> "QESave":
        return read_qe_save(directory)

    @property
    def omega(self) -> float:
        return float(abs(np.linalg.det(self.at)))

    @property
    def nspin_mag(self) -> int:
        """How many density components the file carries: 1, 2 or 4."""
        return self.rho_g.shape[0]

    def seed_for(self, calculation) -> dict:
        """See :func:`seed_for`."""
        return seed_for(self, calculation)


def read_qe_save(directory) -> QESave:
    """Read ``data-file-schema.xml``, the density and ``paw.txt`` if present.

    ``directory`` is the ``<prefix>.save`` directory itself.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} is not a pw.x save directory")
    density = next((directory / f"charge-density.{ext}" for ext in ("dat", "hdf5")
                    if (directory / f"charge-density.{ext}").exists()), None)
    if density is None:
        raise FileNotFoundError(
            f"{directory} has no charge-density.dat or charge-density.hdf5; "
            "point at the <prefix>.save directory of a pw.x run"
        )
    gamma_only, b, miller, rho_g = read_rhog(density)

    out = ET.parse(directory / "data-file-schema.xml").getroot().find("output")
    structure = out.find("atomic_structure")
    alat = float(structure.get("alat"))
    del alat  # the cell and positions below are already in bohr
    at = np.stack([_floats(structure.find(f"cell/a{i}")) for i in (1, 2, 3)])
    atoms = structure.findall("atomic_positions/atom")
    positions = np.stack([_floats(a) for a in atoms])
    labels = tuple(a.get("name") for a in atoms)
    magnetization = out.find("magnetization")
    converged = out.find("convergence_info/scf_conv/convergence_achieved")

    becsum = None
    if (directory / "paw.txt").exists():
        becsum = np.array((directory / "paw.txt").read_text().split(), dtype=float)
    occupations = None
    if (directory / "occup.txt").exists():
        occupations = _read_list_directed((directory / "occup.txt").read_text())
    ekin_g = None
    kinetic = next((directory / f"ekin-density.{ext}" for ext in ("dat", "hdf5")
                    if (directory / f"ekin-density.{ext}").exists()), None)
    if kinetic is not None:
        ekin_gamma, _, ekin_miller, ekin_g = read_rhog(kinetic)
        if ekin_gamma != gamma_only or not np.array_equal(ekin_miller, miller):
            raise ValueError("ekin-density and charge-density are on different G sets")

    dftu = out.find("dft/dftU")
    hubbard_kind = hubbard_projectors = None
    extra = []
    if dftu is not None:
        kind = dftu.find("lda_plus_u_kind")
        hubbard_kind = int(kind.text) if kind is not None else 0
        projection = dftu.find("U_projection_type")
        hubbard_projectors = (projection.text.strip() if projection is not None
                              else "atomic")
        names = {child.tag for child in dftu}
        # The tag names are qexsd_init.f90's.
        if names & {"Hubbard_back", "Hubbard_alpha_back"}:
            extra.append("a background Hubbard manifold (hub_back, nsb)")
        if "Hubbard_V" in names or hubbard_kind == 2:
            extra.append("Hubbard_V (lda_plus_u_kind = 2, nsg)")
        if (names & {"Hubbard_Um", "Hub_m_order"}
                or (directory / "Hubbard_m_order.txt").exists()):
            extra.append("the orbital-resolved form (Hubbard_Um, Hubbard_m_order.txt)")

    return QESave(
        directory=directory,
        gamma_only=gamma_only,
        b=b,
        miller=miller,
        rho_g=rho_g,
        at=at,
        positions=positions,
        labels=labels,
        nelec=float(out.find("band_structure/nelec").text),
        ecutrho=float(out.find("basis_set/ecutrho").text) * 2.0,  # Ha -> Ry
        lsda=magnetization.find("lsda").text.strip() == "true",
        noncolin=magnetization.find("noncolin").text.strip() == "true",
        functional=out.find("dft/functional").text.strip(),
        hubbard=out.find("dft/dftU") is not None,
        converged=(None if converged is None
                   else converged.text.strip() == "true"),
        becsum=becsum,
        occupations=occupations,
        ekin_g=ekin_g,
        hubbard_kind=hubbard_kind,
        hubbard_projectors=hubbard_projectors,
        hubbard_extra=tuple(extra),
    )


def _check_compatible(save: QESave, calculation) -> None:
    """Refuse, by name, a save directory that is not this calculation's."""
    system = calculation.system
    at = np.asarray(system.cell.at, dtype=float)
    if not np.allclose(at, save.at, atol=1e-6, rtol=1e-6):
        raise ValueError(
            "the pw.x run and this calculation are not the same cell (bohr):\n"
            f"  pw.x       {save.at.tolist()}\n"
            f"  defuqumat  {at.tolist()}\n"
            "A density is a set of Fourier coefficients on one lattice and does "
            "not transfer to another."
        )
    positions = np.asarray(system.structure.positions, dtype=float)
    if positions.shape != save.positions.shape:
        raise ValueError(
            f"the pw.x run has {len(save.positions)} atoms and this calculation "
            f"{len(positions)}"
        )
    # Folded into the cell, so a position written one lattice vector away is
    # the same atom.
    shift = (positions - save.positions) @ np.linalg.inv(at)
    shift -= np.round(shift)
    if np.abs(shift @ at).max() > 1e-5:
        raise ValueError(
            "the pw.x run and this calculation do not have the atoms in the same "
            "place, in the same order (largest displacement "
            f"{np.abs(shift @ at).max():.3e} bohr). The density of one geometry "
            "is not the density of another."
        )
    if abs(float(calculation.nelec) - save.nelec) > 1e-6:
        raise ValueError(
            f"the pw.x run has {save.nelec:g} electrons and this calculation "
            f"{float(calculation.nelec):g}: a different pseudopotential, or a "
            "charged cell, on one side only."
        )
    ecut = float(calculation.basis.dense.ecut)
    if abs(ecut - save.ecutrho) > 1e-6 * max(ecut, 1.0):
        raise NotImplementedError(
            f"the pw.x run has ecutrho = {save.ecutrho:g} Ry and this calculation "
            f"{ecut:g} Ry. A different density cutoff is a different G-sphere, "
            "and carrying a density between two of them is not implemented "
            "(notes/CONTINUATION-NEXT.md); run with the same ecutrho."
        )
    if save.hubbard != bool(calculation.is_hubbard):
        raise ValueError(
            ("the pw.x run is DFT+U and this calculation has no Hubbard term"
             if save.hubbard else
             "this calculation is DFT+U and the pw.x run had no Hubbard term")
            + ": the state is the density together with the occupation matrix "
            "ns, so a seed from the other functional starts from two states."
        )
    if save.hubbard:
        _check_hubbard(save, calculation)
    if calculation.functional.is_meta and save.ekin_g is None:
        raise ValueError(
            "this calculation is a meta-GGA and the save directory has no "
            "ekin-density file: the state carries the kinetic energy density "
            "tau as well as the density."
        )

    nspin_mag = calculation.nspin_mag
    if save.nspin_mag != nspin_mag:
        regime = {1: "unpolarized", 2: "collinear (nspin = 2)",
                  4: "noncollinear with a magnetization"}
        raise ValueError(
            f"the pw.x density has {save.nspin_mag} components, "
            f"{regime.get(save.nspin_mag, '?')}, and this calculation takes "
            f"{nspin_mag}, {regime.get(nspin_mag, '?')}. Converge the source "
            "in the same spin regime, or seed a defuqumat run and change the "
            "regime with starting_from=."
        )
    if calculation.is_paw and save.becsum is None:
        raise ValueError(
            "this calculation is PAW and the save directory has no paw.txt, "
            "so the one-centre occupations becsum are missing: a PAW density "
            "without them is two states at once."
        )


def _check_hubbard(save: QESave, calculation) -> None:
    """Refuse, by name, an ``occup.txt`` that is not this run's ``ns``."""
    setup = calculation.hubbard
    if save.occupations is None:
        raise ValueError(
            "the pw.x run is DFT+U and its save directory has no occup.txt, "
            "the occupation matrix ns"
        )
    if save.hubbard_extra:
        raise NotImplementedError(
            "the pw.x run has " + " and ".join(save.hubbard_extra) + ", which "
            "occup.txt carries beside or instead of ns and which defuqumat does "
            "not have, so its state cannot be continued here."
        )
    if save.hubbard_kind != int(setup.kind):
        raise ValueError(
            f"the pw.x run has lda_plus_u_kind = {save.hubbard_kind} and this "
            f"calculation {setup.kind}: a different functional of ns."
        )
    if save.hubbard_projectors != setup.projectors:
        raise ValueError(
            f"the pw.x run projects on '{save.hubbard_projectors}' orbitals and "
            f"this calculation on '{setup.projectors}': ns is an occupation of "
            "those orbitals, so it does not carry over to others."
        )


def _on_grid(save: QESave, grid) -> np.ndarray:
    """``rho(G)`` scattered onto the FFT box and transformed, ``(nspin, n1, n2, n3)``."""
    miller, rho_g = save.miller, save.rho_g
    if save.gamma_only:
        # The half sphere: rebuild -G from rho(-G) = conj(rho(G)), skipping
        # G = 0 so it is not counted twice.
        other = np.any(miller != 0, axis=1)
        miller = np.concatenate([miller, -miller[other]])
        rho_g = np.concatenate([rho_g, rho_g[:, other].conj()], axis=1)
    grid = np.asarray(grid)
    if np.any(np.abs(miller) > grid // 2):
        raise ValueError(
            f"the pw.x density has Miller indices up to {np.abs(miller).max(0)} "
            f"and this calculation's FFT grid is {tuple(grid)}"
        )
    box = np.zeros((rho_g.shape[0], *grid), dtype=complex)
    box[:, miller[:, 0] % grid[0], miller[:, 1] % grid[1], miller[:, 2] % grid[2]] = rho_g
    return np.real(np.fft.ifftn(box, axes=(1, 2, 3)) * np.prod(grid))


def _becsum(save: QESave, calculation) -> tuple:
    """``paw.txt`` as defuqumat's per-species ``(nspin_mag, nat_t, nh, nh)`` matrices.

    The number of components is read off the file's size, not taken from the
    run: ``scf_mod.f90`` allocates ``bec`` with ``lsda_mod``'s ``nspin``, which
    is 4 for **every** noncollinear run, so a spin-orbit run without a
    magnetization (``nspin_mag = 1``) writes its charge component and three
    zeros. Reshaping that file with the run's one component put atom 2's rows
    in the first zero block, silently (P180, on two-atom Pt with ``lspinorb``).
    """
    import jax.numpy as jnp

    nat = len(save.positions)
    nhm = max(int(p.nh) for p in calculation.pseudos)
    npack = nhm * (nhm + 1) // 2
    ncomp, rest = divmod(save.becsum.size, nat * npack)
    if rest or ncomp not in (1, 2, 4):
        raise ValueError(
            f"paw.txt holds {save.becsum.size} numbers, which is not "
            f"nhm (nhm + 1) / 2 = {npack} times {nat} atoms times 1, 2 or 4 "
            "components: the datasets are not the ones the pw.x run used."
        )
    nspin = calculation.nspin_mag
    raw = save.becsum.reshape(ncomp, nat, npack)  # Fortran (ijh, na, is), row-major
    if ncomp == 4 and nspin == 1:
        # A noncollinear run without a magnetization: add_becsum_so/_nc fill
        # only the charge component when .not. domag.
        if np.abs(raw[1:]).max() > 1e-12:
            raise ValueError(
                "paw.txt carries a magnetization and this calculation has none; "
                "converge the source without one, or run this one with it."
            )
        raw = raw[:1]
    elif ncomp != nspin:
        raise ValueError(
            f"paw.txt has {ncomp} spin components and this calculation "
            f"nspin_mag = {nspin}"
        )
    becsum = []
    for t, atoms in enumerate(calculation.augmentation.species_atoms):
        if calculation.species_channels[t] is None or not atoms:
            becsum.append(None)
            continue
        nh = calculation.pseudos[t].nh
        upper = np.triu_indices(nh)  # ih outer, jh >= ih inner: ijtoh's order
        packed = raw[:, list(atoms), : len(upper[0])]
        matrix = np.zeros((nspin, len(atoms), nh, nh))
        matrix[:, :, upper[0], upper[1]] = packed
        # The off-diagonal entries are stored doubled, once for the pair.
        off = upper[0] != upper[1]
        matrix[:, :, upper[0][off], upper[1][off]] /= 2.0
        matrix[:, :, upper[1][off], upper[0][off]] = matrix[:, :, upper[0][off], upper[1][off]]
        becsum.append(jnp.asarray(matrix))
    return tuple(becsum)


def _ns(save: QESave, calculation):
    """``occup.txt`` as defuqumat's ``(nspin, nslot, ldmx, ldmx)``.

    ``pw.x`` writes ``ns(m1, m2, is, na)`` (``ns_nc`` for a spinor run, its
    spin index ``is = npol (is1 - 1) + is2``, which is defuqumat's packing too)
    over every atom at ``2 Hubbard_lmax + 1``; the Hubbard atoms are gathered
    into slots. The array is read C-ordered as ``[na, is, m2, m1]``.
    """
    import jax.numpy as jnp

    setup = calculation.hubbard
    nat = len(save.positions)
    nspin = 4 if calculation.system.noncolin else (2 if save.lsda else 1)
    ldmx = int(setup.ldmx)
    raw = save.occupations
    if raw.size != ldmx * ldmx * nspin * nat:
        raise ValueError(
            f"occup.txt holds {raw.size} numbers and this calculation expects "
            f"(2 Hubbard_lmax + 1)^2 = {ldmx * ldmx} times {nspin} spin blocks "
            f"times {nat} atoms: a different Hubbard manifold."
        )
    if np.iscomplexobj(raw) != bool(calculation.system.noncolin):
        raise ValueError("occup.txt is complex for a spinor run and real otherwise")
    blocks = np.swapaxes(raw.reshape(nat, nspin, ldmx, ldmx), -1, -2)
    ns = np.stack([blocks[atom] for atom in setup.atoms], axis=1)
    return jnp.asarray(ns)


def seed_for(save: QESave, calculation) -> dict:
    """The pw.x state as ``run_scf`` keyword arguments for ``calculation``.

    Returns ``{"starting_density": (nspin_mag, n1, n2, n3)}`` in e/bohr^3,
    plus ``"starting_becsum"`` for a PAW run, ``"starting_ns"`` for DFT+U and
    ``"starting_tau"`` for a meta-GGA, so that
    ``calc.get_scf(**seed)`` continues the pw.x run. It also carries
    ``diago_thr_init = 1e-5``, ``pw.x``'s first threshold for a start from a
    density (``setup.f90:437``, ``startingpot = 'file'``): at the 1e-2 of a
    start from atomic orbitals the first diagonalisation throws the converged
    density away, and Si2 PAW took 9 iterations from it against 7 from scratch,
    where at 1e-5 it takes 1. Nothing is renormalised:
    the file's ``rho(G = 0)`` is already the electron count, and a mismatch is
    refused rather than scaled away.
    """
    import jax.numpy as jnp

    _check_compatible(save, calculation)
    dense = calculation.basis.dense
    rho = _on_grid(save, tuple(int(n) for n in dense.grid))
    if rho.shape[0] == 2:
        # (n, m_z) -> defuqumat's (up, down).
        rho = np.stack([rho[0] + rho[1], rho[0] - rho[1]]) / 2.0
    seed = {"starting_density": jnp.asarray(rho), "diago_thr_init": 1.0e-5}
    if calculation.is_paw:
        seed["starting_becsum"] = _becsum(save, calculation)
    if save.hubbard:
        seed["starting_ns"] = _ns(save, calculation)
    if calculation.functional.is_meta:
        # rho%kin is (up, down) at nspin = 2 (potinit.f90), defuqumat's own
        # layout for tau, so nothing is split.
        kinetic = dataclasses.replace(save, rho_g=save.ekin_g)
        seed["starting_tau"] = jnp.asarray(_on_grid(kinetic, tuple(int(n) for n in dense.grid)))
    return seed


# ---------------------------------------------------------------- writing

def write_rhog(path, b, miller, rho_g, gamma_only: bool) -> Path:
    """``write_rhog``: one density file, ``.dat`` (Fortran unformatted).

    ``b`` is ``(3, 3)`` with the reciprocal vectors as rows in 1/bohr
    (``bg * tpiba``), ``miller`` is ``(ngm, 3)`` in ``pw.x``'s order
    (:func:`defuqumat.io.qe_gvectors.qe_gvector_order`) and ``rho_g`` is
    ``(nspin, ngm)``. The records are ``write_rhog``'s, so the file is the one
    ``pw.x`` writes for the same arrays, byte for byte (gfortran's logical is
    a four-byte 1 or 0).
    """
    from scipy.io import FortranFile

    path = Path(path)
    if path.suffix == ".hdf5":
        raise NotImplementedError(
            "writing charge-density.hdf5 is not implemented: the pw.x this was "
            "validated against has no HDF5, so a file in that form could not "
            "be checked; write the .dat form, which every pw.x reads"
        )
    miller = np.asarray(miller, dtype=np.int32)
    rho_g = np.asarray(rho_g, dtype=np.complex128)
    if rho_g.ndim != 2 or rho_g.shape[1] != len(miller):
        raise ValueError(f"rho_g {rho_g.shape} does not match {len(miller)} G-vectors")
    with FortranFile(path, "w") as f:
        f.write_record(np.array([1 if gamma_only else 0, len(miller), rho_g.shape[0]],
                                dtype=np.int32))
        f.write_record(np.asarray(b, dtype=np.float64).reshape(9))
        f.write_record(miller.reshape(-1))
        for component in rho_g:
            f.write_record(component)
    return path


def _coefficients(field, miller) -> np.ndarray:
    """``(n, n1, n2, n3)`` real fields -> ``(n, ngm)`` coefficients of ``e^{iGr}``."""
    field = np.asarray(field, dtype=float)
    grid = np.array(field.shape[1:])
    if np.any(2 * np.abs(miller).max(axis=0) >= grid):
        raise ValueError(
            f"pw.x's G sphere reaches Miller indices {np.abs(miller).max(axis=0)} "
            f"and this calculation's dense FFT grid is {tuple(grid)}: the grid "
            "does not hold the sphere, so the coefficients would alias"
        )
    box = np.fft.fftn(field, axes=(1, 2, 3)) / np.prod(grid)
    return box[:, miller[:, 0] % grid[0], miller[:, 1] % grid[1], miller[:, 2] % grid[2]]


def _to_qe_layout(rho: np.ndarray) -> np.ndarray:
    """defuqumat's density components -> ``rho%of_g``'s.

    ``(up, down)`` becomes ``(n, m_z)``; one component and the noncollinear
    ``(n, m_x, m_y, m_z)`` are QE's already.
    """
    if rho.shape[0] == 2:
        return np.stack([rho[0] + rho[1], rho[0] - rho[1]])
    return rho


def _pack_becsum(becsum, calculation) -> np.ndarray:
    """defuqumat's per-species matrices -> ``rho%bec``, ``(nspin, nat, nhm(nhm+1)/2)``.

    ``nspin`` is ``lsda_mod``'s: 4 for any noncollinear run, so a run without
    a magnetization (``nspin_mag = 1``) writes its one component and three
    zeros, as ``pw.x`` does (``scf_mod.f90`` allocates ``bec`` with ``nspin``).
    ``nhm`` is the largest ``nh`` over **all** species, norm-conserving ones
    included (``uspp_param``), and the rows of a shorter species are zero.
    """
    system = calculation.system
    nat = len(system.structure.types)
    nhm = max(int(p.nh) for p in calculation.pseudos)
    nspin = 4 if system.noncolin else int(calculation.nspin_mag)
    packed = np.zeros((nspin, nat, nhm * (nhm + 1) // 2))
    for t, atoms in enumerate(calculation.augmentation.species_atoms):
        block = becsum[t] if t < len(becsum) else None
        if block is None or not atoms:
            continue
        block = np.asarray(block, dtype=float)  # (nspin_mag, nat_t, nh, nh)
        nh = block.shape[-1]
        upper = np.triu_indices(nh)  # ijtoh's order: ih outer, jh >= ih inner
        values = block[:, :, upper[0], upper[1]]
        # The off-diagonal entries are stored doubled, once for the pair.
        values = values * np.where(upper[0] == upper[1], 1.0, 2.0)
        packed[: block.shape[0], list(atoms), : len(upper[0])] = values
    return packed


def _list_directed(values) -> str:
    """Numbers for a list-directed Fortran ``READ``, exact to the last bit.

    Complex values are written ``(re,im)``, gfortran's own form for them.
    """
    values = np.asarray(values).reshape(-1)
    if np.iscomplexobj(values):
        items = [f"({v.real!r},{v.imag!r})" for v in values.tolist()]
    else:
        items = [repr(float(v)) for v in values]
    return "\n".join(" ".join(items[i:i + 4]) for i in range(0, len(items), 4)) + "\n"


def _pack_ns(ns, calculation) -> np.ndarray:
    """defuqumat's ``ns`` -> ``rho%ns`` (or ``ns_nc``), ``(nat, nspin, ldmx, ldmx)``.

    ``pw.x`` allocates it over **every** atom at ``2 Hubbard_lmax + 1``
    (``scf_mod.f90:159``) and fills only the Hubbard ones, so the other atoms
    are written as zeros. Written C-ordered as ``[na, is, m2, m1]``, which is
    the Fortran order of ``ns(m1, m2, is, na)``.
    """
    setup = calculation.hubbard
    ns = np.asarray(ns)
    nat = len(calculation.system.structure.types)
    out = np.zeros((nat, ns.shape[0], setup.ldmx, setup.ldmx), dtype=ns.dtype)
    for slot, atom in enumerate(setup.atoms):
        out[atom] = ns[:, slot]
    return np.swapaxes(out, -1, -2)


def _warn_another_functional(calculation) -> None:
    """Warn when the state written is the fixed point of a functional ``pw.x`` does not run.

    The file is read correctly either way; what changes is that ``pw.x``
    takes it as a nearby start rather than as its own converged state (P180,
    2026-11-18). ``'grad_m'`` and ``pw.x``'s ``'kubler'`` are the collinear GGA
    for a magnetization of one direction (P143), so they differ where that
    direction varies, which spin-orbit coupling always makes it do:
    ``fe-kind1-noncol`` restarts in 15 iterations from 1.0e-4 Ry under
    ``'grad_m'`` and in 1 from 6.8e-11 under ``'kubler'`` (``pw.x`` 7.2), the
    As atom with spin-orbit coupling in 3 from 1e-8 and in 1 from 1.3e-11
    (``pw.x`` 7.5), while O2 without it restarts alike under both. TB09:
    ``si2-tb09`` in 7 and ``si2-tb09-lsda`` in 13, because ``pw.x``'s ``tb09``
    calls libxc with a zero Laplacian and ``c = 1`` (P30).
    """
    import warnings

    if getattr(calculation, "noncollinear_gga", "kubler") == "grad_m":
        warnings.warn(
            "this state was converged under noncollinear_gga = 'grad_m', and pw.x "
            "runs 'kubler'. The two differ where the direction of the "
            "magnetization varies in space, as spin-orbit coupling makes it do; "
            "there pw.x reads the file but starts from it as from a nearby "
            "density (15 iterations against 1 on Fe with DFT+U, 3 against 1 on "
            "an As atom). Converge under noncollinear_gga = 'kubler' for a "
            "state pw.x takes as its own.",
            stacklevel=3,
        )
    name = str(getattr(calculation.functional, "meta", "") or "").upper()
    if name in ("TB09", "BJ06"):
        warnings.warn(
            f"this is a {name} state, and pw.x's input_dft = 'tb09' calls libxc "
            "with a zero Laplacian and c = 1 and adds TPSS correlation, which is "
            "neither this code's TB09 "
            "nor its BJ06: pw.x reads ekin-density correctly but converges to "
            "another state, 7 and 13 iterations from this one on Si2 at nspin = "
            "1 and 2.",
            stacklevel=3,
        )


def write_qe_save(directory, result, calculation, *, gamma_only: bool | None = None) -> Path:
    """Write a converged defuqumat state as a ``pw.x`` save directory.

    ``directory`` is the ``<outdir>/<prefix>.save`` that a ``pw.x`` run with
    ``startingpot = 'file'`` reads (``potinit.f90:86-97`` -> ``read_scf``). It
    receives ``charge-density.dat`` in ``pw.x``'s G order, ``paw.txt`` for a
    PAW run, ``occup.txt`` for DFT+U and ``ekin-density.dat`` for a meta-GGA,
    each in the layout ``write_scf`` gives it, and a
    ``data-file-schema.xml`` holding what :func:`read_qe_save` checks.

    ``gamma_only`` is the flag of the ``pw.x`` run that will read it; by
    default the input's own ``K_POINTS gamma`` (``Calculation.kpoints_gamma``,
    since the substitution of P68 erases it from the system). A full-sphere
    file is read by a ``K_POINTS gamma`` run too, through ``charge_k_to_g``,
    whose merge loop exits (``IF (ig .GE. ngm_g) EXIT``) before it fills the
    last G-vector; on Si2 both forms start at 2e-8 Ry, as ``pw.x`` does from
    its own gamma file, so that loss was not visible there (P180).

    Refused: a spin spiral (``pw.x`` has none), and a state ``pw.x`` cannot
    hold in ``occup.txt`` (a background Hubbard manifold or a ``Hubbard_V``,
    which defuqumat does not have either).
    """
    from defuqumat.io.qe_gvectors import qe_gvector_order

    system = calculation.system
    if getattr(system, "spiral_q", None) is not None:
        raise NotImplementedError(
            "a spin spiral has no pw.x counterpart, so there is no pw.x run that "
            "could read its density"
        )
    _warn_another_functional(calculation)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # The K_POINTS gamma card, which is what pw.x follows; not
    # calculation.gamma_only, defuqumat's storage, which P68 substitutes to the
    # full sphere for symmetry, ultrasoft, PAW and spinors.
    gamma = (bool(getattr(calculation, "kpoints_gamma", system.kpoints.gamma_only))
             if gamma_only is None else bool(gamma_only))

    cell = system.cell
    at = np.asarray(cell.at, dtype=float)
    alat = float(cell.alat)
    ecutrho = float(calculation.basis.dense.ecut)
    miller = qe_gvector_order(at, alat, ecutrho, gamma)
    b = np.linalg.inv(at).T * 2.0 * np.pi  # rows b_a in 1/bohr

    rho = np.asarray(result.density, dtype=float)
    write_rhog(directory / "charge-density.dat", b, miller,
               _to_qe_layout(_coefficients(rho, miller)), gamma)

    tau = getattr(result, "tau", None)
    if calculation.functional.is_meta:
        if tau is None:
            raise ValueError("a meta-GGA result without its kinetic energy density")
        # rho%kin_g is (up, down) at nspin = 2 (potinit.f90's comment), unlike
        # the charge, so it is written as held.
        write_rhog(directory / "ekin-density.dat", b, miller,
                   _coefficients(np.asarray(tau, dtype=float), miller), gamma)

    if calculation.is_paw:
        (directory / "paw.txt").write_text(
            _list_directed(_pack_becsum(result.becsum, calculation)))

    if calculation.is_hubbard:
        setup = calculation.hubbard
        if result.ns is None:
            raise ValueError("a DFT+U result without its occupation matrix")
        (directory / "occup.txt").write_text(
            _list_directed(_pack_ns(result.ns, calculation)))
        del setup

    _write_schema(directory / "data-file-schema.xml", result, calculation, alat, ecutrho)
    return directory


def _write_schema(path, result, calculation, alat: float, ecutrho: float) -> None:
    """The part of ``data-file-schema.xml`` that :func:`read_qe_save` checks.

    Element names are the ``qes`` schema's, and the producer says it is
    defuqumat. It is **not** the whole schema: ``pw.x``'s ``scf`` restart with
    ``startingpot = 'file'`` does not read the XML at all, while ``read_file``
    (an ``nscf`` or ``bands`` run, ``pp.x``, ``projwfc.x``) needs the whole
    ``output`` element and is not served by this file.
    """
    system = calculation.system
    structure = system.structure
    at = np.asarray(system.cell.at, dtype=float)
    positions = np.asarray(structure.positions, dtype=float)

    def vector(values) -> str:
        return " ".join(f"{v!r}" for v in np.asarray(values, dtype=float).tolist())

    root = ET.Element("qes:espresso", {
        "xmlns:qes": "http://www.quantum-espresso.org/ns/qes/qes-1.0",
        "Units": "Hartree atomic units",
    })
    info = ET.SubElement(root, "general_info")
    ET.SubElement(info, "creator", {"NAME": "defuqumat", "VERSION": "P180"}).text = (
        "a density for pw.x's startingpot = 'file'; not a full read_file schema")
    out = ET.SubElement(root, "output")
    scf = ET.SubElement(ET.SubElement(out, "convergence_info"), "scf_conv")
    ET.SubElement(scf, "convergence_achieved").text = (
        "true" if bool(result.converged) else "false")
    ET.SubElement(scf, "n_scf_steps").text = str(int(result.iterations))
    st = ET.SubElement(out, "atomic_structure", {"nat": str(len(positions)), "alat": repr(alat)})
    pos = ET.SubElement(st, "atomic_positions")
    for i, t in enumerate(structure.types):
        ET.SubElement(pos, "atom", {"name": structure.species[t].name,
                                    "index": str(i + 1)}).text = vector(positions[i])
    cell = ET.SubElement(st, "cell")
    for i in range(3):
        ET.SubElement(cell, f"a{i + 1}").text = vector(at[i])
    dft = ET.SubElement(out, "dft")
    ET.SubElement(dft, "functional").text = str(calculation.functional.name)
    if calculation.is_hubbard:
        hub = ET.SubElement(dft, "dftU")
        ET.SubElement(hub, "lda_plus_u_kind").text = str(int(calculation.hubbard.kind))
        ET.SubElement(hub, "U_projection_type").text = str(calculation.hubbard.projectors)
    mag = ET.SubElement(out, "magnetization")
    ET.SubElement(mag, "lsda").text = "true" if int(system.nspin) == 2 else "false"
    ET.SubElement(mag, "noncolin").text = "true" if system.noncolin else "false"
    ET.SubElement(mag, "spinorbit").text = "true" if system.lspinorb else "false"
    basis = ET.SubElement(out, "basis_set")
    ET.SubElement(basis, "gamma_only").text = (
        "true" if getattr(calculation, "kpoints_gamma", False) else "false")
    ET.SubElement(basis, "ecutrho").text = repr(ecutrho / 2.0)  # Ry -> Ha
    grid = calculation.basis.dense.grid
    ET.SubElement(basis, "fft_grid", {"nr1": str(grid[0]), "nr2": str(grid[1]),
                                      "nr3": str(grid[2])}).text = "true"
    bands = ET.SubElement(out, "band_structure")
    ET.SubElement(bands, "nelec").text = repr(float(calculation.nelec))
    ET.indent(root)
    Path(path).write_text('<?xml version="1.0" encoding="UTF-8"?>\n'
                          + ET.tostring(root, encoding="unicode") + "\n")
