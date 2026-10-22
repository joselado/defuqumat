"""Reading a converged ``pw.x`` density from its save directory.

``pw.x`` leaves the self-consistent state in ``<outdir>/<prefix>.save/``, and
this reads the part of it a continuation needs: the density, and for a PAW run
the one-centre occupations ``becsum``. The routines transcribed are
``PW/src/io_rho_xml.f90`` (``write_scf``/``read_scf``) and
``Modules/io_base.f90`` (``write_rhog``/``read_rhog``).

**The density file.** ``charge-density.dat`` is Fortran unformatted, one record
each for ``(gamma_only, ngm_g, nspin)``, the three reciprocal vectors
``bg * tpiba`` in 1/bohr, the Miller indices ``(3, ngm_g)``, and then one
``complex128[ngm_g]`` record per component. ``charge-density.hdf5`` (a QE built
with HDF5) carries the same arrays as attributes and datasets. The coefficients
are those of ``rho(r) = sum_G rho(G) exp(iGr)`` in e/bohr^3, so
``rho(G = 0) * omega`` is the electron count.

**The spin layout is not defuqumat's in one case.** Since QE 6.4 ``rho%of_g``
is ``(n, m_z)`` for ``nspin = 2`` and ``(n, m_x, m_y, m_z)`` for ``nspin = 4``,
and it is written as held (``read_rhog`` converts nothing); a noncollinear run
without a magnetization writes the charge alone (``write_scf``, ``nspin_ = 1``
when ``noncolin .and. .not. domag``). defuqumat's collinear density is
``(up, down)``, so that pair is split, and the noncollinear one is taken as is.
Measured on an O2 triplet: the second component's ``G = 0`` times ``omega`` is
2.00, the "total magnetization" line of ``pw.x``'s output, where an
``(up, down)`` file would read 5.

**``paw.txt``** is ``rho%bec`` written list-directed, shape
``(nhm (nhm + 1) / 2, nat, nspin_mag)``: the upper triangle of each atom's
matrix in ``ijtoh`` order (``ih`` outer, ``jh >= ih`` inner, ``init_us_1.f90``)
and the off-diagonal entries doubled, ``sum_bec`` storing
``2 Re <psi|beta_i><beta_j|psi>`` once for the pair. Unlike the density it
stays ``(up, down)`` for ``nspin = 2`` (``PAW_atomic_becsum`` builds it so and
``rhoz_or_updw`` touches only ``of_r``/``of_g``).

**What is refused, each by name, and what it would need:** a DFT+U run
(``occup.txt``, the occupation matrix, is not read), a meta-GGA run
(``ekin-density``, the kinetic energy density, is not read), a noncollinear PAW
run (its ``becsum`` is the spin-matrix form, not mapped), and any mismatch of
cell, atomic positions, electron count, density cutoff or spin regime between
the save directory and the calculation it is handed to: a density belongs to
one G-sphere on one lattice.

Measured on nine ``pw.x`` runs, NC, US and PAW, ``nspin = 1, 2, 4``, with and
without spin-orbit coupling, in ``PLAN.md`` P160.

Since both codes hold the same pseudo valence density, a seed from a run with
the same input is a continuation rather than a guess: the first iteration's
``dr2`` reads at the level of the file's own ``conv_thr``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["QESave", "read_qe_save", "read_rhog"]


def read_rhog(path):
    """``read_rhog``: ``(gamma_only, b, miller, rho_g)`` from one density file.

    ``b`` is ``(3, 3)`` with the reciprocal vectors as rows in 1/bohr,
    ``miller`` is ``(ngm, 3)`` and ``rho_g`` is ``(nspin, ngm)``.
    """
    path = Path(path)
    if path.suffix == ".hdf5":
        import h5py

        with h5py.File(path, "r") as f:
            gamma_only = str(f.attrs["gamma_only"]).strip().upper() in (".TRUE.", "T", "TRUE")
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
    """An HDF5 dataset as complex: QE writes it as pairs of doubles."""
    array = np.asarray(array)
    if np.iscomplexobj(array):
        return array.reshape(-1)
    return array.reshape(-1, 2) @ np.array([1.0, 1.0j])


def _floats(element) -> np.ndarray:
    return np.array(element.text.split(), dtype=float)


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
    if save.hubbard or calculation.is_hubbard:
        raise NotImplementedError(
            "a DFT+U state is the density together with the occupation matrix "
            "ns, and occup.txt is not read; seeding the density alone would "
            "start the run from two different states."
        )
    if calculation.functional.is_meta:
        raise NotImplementedError(
            "a meta-GGA state carries the kinetic energy density tau as well, "
            "and ekin-density is not read."
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
    if calculation.is_paw:
        if save.becsum is None:
            raise ValueError(
                "this calculation is PAW and the save directory has no paw.txt, "
                "so the one-centre occupations becsum are missing: a PAW density "
                "without them is two states at once."
            )
        if nspin_mag == 4:
            raise NotImplementedError(
                "a noncollinear PAW becsum is the spin-matrix form, and its "
                "mapping from paw.txt is not implemented."
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
    """``paw.txt`` as defuqumat's per-species ``(nspin, nat_t, nh, nh)`` matrices."""
    import jax.numpy as jnp

    nat = len(save.positions)
    nspin = calculation.nspin_mag
    raw = save.becsum.reshape(nspin, nat, -1)  # Fortran (ijh, na, is), read row-major
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


def seed_for(save: QESave, calculation) -> dict:
    """The pw.x state as ``run_scf`` keyword arguments for ``calculation``.

    Returns ``{"starting_density": (nspin_mag, n1, n2, n3)}`` in e/bohr^3,
    plus ``"starting_becsum"`` for a PAW run, so that
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
    return seed
