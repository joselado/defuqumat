"""Band structure: diagonalise at a k-path with the density held fixed.

A band structure is not a self-consistent calculation. The density comes from a
converged SCF run, the potential built from it is frozen, and the Hamiltonian is
then diagonalised at whatever k-points the band path asks for -- which is why
those k-points may be anywhere in the zone and carry no integration weights.

Following ``PW/src/non_scf.f90``. The diagonalisation itself lives in
:mod:`defuqumat.workflows.nscf`, because the same routine serves an NSCF run on a
denser grid, which is what a density of states needs; what is left here is the
band-path presentation of it -- the path length, the gap, the plotting abscissa.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from defuqumat.pseudo.upf import Pseudopotential
from defuqumat.system.builder import System
from defuqumat.system.kpoints import KPoints
from defuqumat.units import RY_TO_EV
from defuqumat.system.kpoints import for_spin as kpoints_for_spin
from defuqumat.workflows.nscf import (
    fixed_density_bands, fixed_density_ethr, kset_blocks, threaded_calculation,
)

__all__ = ["BandStructure", "run_bands"]


@dataclass
class BandStructure:
    """Eigenvalues along a path, with everything needed to plot them."""

    kpoints: KPoints
    #: ``(nk, nbnd)`` unpolarized, ``(2, nk, nbnd)`` for LSDA -- the same
    #: squeeze-when-there-is-one-channel convention as :class:`SCFResult`.
    eigenvalues: np.ndarray
    fermi_energy: float | None = None
    homo: float | None = None
    nspin: int = 1
    #: The scalar threshold every band was converged to, in Ry
    #: (:func:`~defuqumat.workflows.nscf.fixed_density_ethr`).
    ethr: float | None = None

    @property
    def eigenvalues_by_spin(self) -> np.ndarray:
        """``(nspin, nk, nbnd)`` whatever ``nspin`` is."""
        return self.eigenvalues if self.nspin == 2 else self.eigenvalues[None]

    @property
    def eigenvalues_ev(self) -> np.ndarray:
        return self.eigenvalues * RY_TO_EV

    @property
    def path_length(self) -> np.ndarray:
        """Cumulative distance along the path -- the x-axis of a band plot."""
        if self.kpoints.path_length is not None:
            return np.asarray(self.kpoints.path_length)
        coords = np.asarray(self.kpoints.coords)
        steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        return np.concatenate([[0.0], np.cumsum(steps)])

    def plot(self, ax=None, ev: bool = True, zero: bool = True, **kwargs):
        """Draw the bands, and return the axes.

        A band structure is looked at far more often than it is tabulated, and
        the five lines of matplotlib that follow every call to this workflow
        are five lines of the script that are not about physics. ``ev`` plots
        in electronvolts, ``zero`` puts the Fermi level (or the HOMO, for an
        insulator) at zero where the run knows one -- which it does, since
        :meth:`~defuqumat.calculator.Calculator.get_bands` carries both across
        from the SCF.

        matplotlib is imported here rather than at module scope: it is not a
        dependency of any calculation, and a headless run should not need it.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        reference, zero_label = 0.0, None
        if zero:
            if self.fermi_energy is not None:
                reference, zero_label = self.fermi_energy, "E$_F$"
            elif self.homo is not None:
                # An insulator has no Fermi level and the top of the valence
                # band is the zero every published band plot uses; saying so
                # keeps the axis honest about which one it is.
                reference, zero_label = self.homo, "E$_{HOMO}$"
        scale = RY_TO_EV if ev else 1.0
        x = self.path_length
        kwargs.setdefault("lw", 1.0)
        for spin, channel in enumerate(self.eigenvalues_by_spin):
            style = dict(kwargs)
            if self.nspin == 2:
                style.setdefault("color", "C0" if spin == 0 else "C3")
                style.setdefault("label", "up" if spin == 0 else "down")
            for band in range(channel.shape[1]):
                ax.plot(x, (channel[:, band] - reference) * scale, **style)
                style.pop("label", None)
        if zero and reference:
            ax.axhline(0.0, color="0.6", lw=0.8, ls="--")
        ax.set_xlim(float(x[0]), float(x[-1]))
        ax.set_xlabel("k-path")
        unit = "eV" if ev else "Ry"
        ax.set_ylabel(f"E - {zero_label} ({unit})" if zero_label
                      else f"Energy ({unit})")
        if self.nspin == 2:
            ax.legend()
        return ax

    def band_edges(
        self,
        nelec: float | None = None,
        tot_magnetization: float | None = None,
        *,
        occupied=None,
    ) -> tuple[float, float]:
        """Highest occupied and lowest empty level on the path, in eV.

        ``pw.x``'s "highest occupied, lowest unoccupied level" for a filling
        that is fixed band by band (``get_homo_lumo`` in
        ``PW/src/print_ks_energies.f90``): the HOMO is the largest occupied
        eigenvalue over every k-point *and* every spin channel, and the LUMO the
        smallest empty one, so at ``nspin = 2`` the two edges may sit in
        different channels.

        The filling is given in one of two ways.

        * ``nelec`` (and, at ``nspin = 2``, ``tot_magnetization``): a band holds
          two electrons when spin-degenerate, one when it is a spinor
          (``nspin = 4``), and one in each channel at ``nspin = 2``, where the
          channels fill ``(nelec + m)/2`` and ``(nelec - m)/2`` bands -- QE's
          ``nelup`` and ``neldw`` for fixed occupations. Both must be whole
          numbers, which is ``input.f90:784-800``'s own condition.
        * ``occupied``: the number of occupied bands directly, an integer, or
          ``(n_up, n_down)`` at ``nspin = 2``. This is the form for a filling
          given band by band (``occupations = 'from_input'``), where
          ``get_homo_lumo`` counts a band as occupied once its weight passes
          1e-3, so a channel with one electron spread over a p triplet has
          four occupied bands, not two.

        Where the filling cuts a degenerate multiplet the two edges are members
        of the same multiplet and their difference is the eigensolver's residue,
        not a gap.
        """
        levels = np.asarray(self.eigenvalues_by_spin) * RY_TO_EV
        nbnd = levels.shape[-1]
        if self.nspin == 2:
            counts = self._channel_counts(nelec, tot_magnetization, occupied)
        else:
            if occupied is None:
                if nelec is None:
                    raise ValueError("give nelec or occupied")
                occupied = int(round(nelec / (1 if self.nspin == 4 else 2)))
            counts = (int(occupied),)
        homo, lumo = -np.inf, np.inf
        for channel, n in zip(levels, counts):
            if not 0 <= n <= nbnd:
                raise ValueError(
                    f"{n} occupied bands in a channel with {nbnd} computed"
                )
            if n > 0:
                homo = max(homo, float(channel[:, n - 1].max()))
            if n < nbnd:
                lumo = min(lumo, float(channel[:, n].min()))
        if not np.isfinite(homo):
            raise ValueError("no occupied band: the filling is empty")
        if not np.isfinite(lumo):
            raise ValueError(
                "no empty band was computed above the filling; raise nbnd"
            )
        return homo, lumo

    def _channel_counts(self, nelec, tot_magnetization, occupied):
        """The occupied band count of each channel at ``nspin = 2``."""
        if occupied is not None:
            counts = tuple(int(n) for n in np.atleast_1d(occupied))
            if len(counts) != 2:
                raise ValueError(
                    "at nspin = 2, occupied is (n_up, n_down), one count per channel"
                )
            return counts
        if nelec is None or tot_magnetization is None:
            raise NotImplementedError(
                "a spin-polarized gap needs the filling of each channel, which "
                "this band structure does not carry: give tot_magnetization "
                "beside nelec (a fixed filling, nelup = (nelec + m)/2 and "
                "neldw = (nelec - m)/2) or occupied = (n_up, n_down)"
            )
        up = (nelec + tot_magnetization) / 2
        down = (nelec - tot_magnetization) / 2
        if abs(up - round(up)) > 1e-8 or abs(down - round(down)) > 1e-8:
            raise ValueError(
                f"nelec = {nelec} and tot_magnetization = {tot_magnetization} do "
                "not fill a whole number of bands in each channel, so there is no "
                "fixed-filling gap; give occupied = (n_up, n_down)"
            )
        return int(round(up)), int(round(down))

    def gap(
        self,
        nelec: float | None = None,
        tot_magnetization: float | None = None,
        *,
        occupied=None,
    ) -> float:
        """Fundamental band gap in eV, for a system with fixed filling.

        The lowest conduction level anywhere on the path minus the highest
        valence level anywhere on it -- the *indirect* gap, which is the smaller
        of the two and equals the direct one when both extrema sit at the same
        k-point. At ``nspin = 2`` "anywhere" includes both channels, so the gap
        may run from one channel's valence top to the other's conduction bottom,
        as ``pw.x`` reports it.

        How many bands are filled depends on how many electrons a band holds:
        two for a spin-degenerate calculation, one for a noncollinear one, where
        each band is a spinor, and one in each channel at ``nspin = 2``, which
        therefore needs ``tot_magnetization`` (or ``occupied``) to say how the
        electrons split. Getting that wrong halves or doubles the count of
        occupied bands and reports the gap between the wrong pair. The arguments
        are those of :meth:`band_edges`.
        """
        homo, lumo = self.band_edges(nelec, tot_magnetization, occupied=occupied)
        return lumo - homo


def run_bands(
    system: System,
    pseudos: tuple[Pseudopotential, ...],
    density: jnp.ndarray,
    kpoints: KPoints | None = None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-6,
    fermi_energy: float | None = None,
    homo: float | None = None,
    k_batch: int | None | str = "default",
    ns: jnp.ndarray | None = None,
    tau: jnp.ndarray | None = None,
    becsum: tuple = (),
    field=None,
    field_scale: float | None = None,
    calculation=None,
) -> BandStructure:
    """Diagonalise at a k-path with the density fixed.

    Args:
        system: the system the density was converged for.
        pseudos: its pseudopotentials.
        density: the converged real-space density from an SCF run, shaped
            ``(nspin, n1, n2, n3)``.
        kpoints: the path to evaluate on. Defaults to ``system.kpoints``, which
            is what an input file with ``calculation='bands'`` already carries.
        nbnd: number of bands; a band structure normally wants more than the
            occupied ones.
        conv_thr: the accuracy the density was converged to, which is what sets
            how accurately the bands are worth computing.
        ns: the converged Hubbard occupation matrix (``SCFResult.ns``), required
            when the run has a Hubbard U -- the term is built from it and it
            cannot be recovered from the density.
        tau: the converged kinetic energy density (``SCFResult.tau``), required
            under a meta-GGA for the same reason and a stronger one: a band path
            has no occupations, so there is nothing here to build one from at
            all. It is the SCF's zone-wide ``tau`` that is held fixed, exactly
            as the density is.
        becsum: the converged projector occupations (``SCFResult.becsum``),
            required for a PAW dataset -- ``ddd_paw`` is built from it and, like
            ``tau``, it is a property of the states rather than of the density.
        field: the field the SCF ended with (``SCFResult.magnetic_field``),
            with ``field_scale`` (``SCFResult.field_scale``) beside it. Required
            when the input carries one: ``reducebf`` and the fixed-spin-moment
            scheme both change the field as the loop runs, so rebuilding it from
            the input applies a field the ground state does not have.
        calculation: the SCF's own :class:`~defuqumat.scf.driver.Calculation`,
            moved to the path with ``at_kpoints`` rather than a second one
            built beside it -- it shares the augmentation and PAW tables and
            keeps the SCF's memory mode (:func:`~defuqumat.workflows.nscf.
            threaded_calculation`). ``Calculator.get_bands`` passes its own.

    The potential is built once from the given density and never updated -- that
    is the whole content of "non self-consistent", and it is why this is a thin
    wrapper around :func:`defuqumat.workflows.nscf.fixed_density_bands`. A band
    path carries no integration weights, so unlike an NSCF grid run it stops at
    the eigenvalues: any Fermi level or HOMO must come from the SCF that
    produced the density, which is what the two arguments are for.
    """
    plan = kset_blocks(calculation, kpoints)
    if plan is not None:
        # A long path in memory mode, a block of points at a time: each block is
        # its own ``at_kpoints``, padded to the whole path's widths, so only one
        # block's per-k tables are on the device (``kset_blocks``).
        from defuqumat.scf.driver import _kpoints_rows

        blocks, widths = plan
        path = kpoints_for_spin(kpoints, system.nspin)
        parts = []
        for rows, live in blocks:
            moved = calculation.at_kpoints(_kpoints_rows(path, rows), widths=widths)
            moved, _, block = fixed_density_bands(
                moved.system, pseudos, density, None, nbnd, conv_thr,
                moved.k_batch, ns, tau, becsum, field, field_scale,
                calculation=moved,
            )
            parts.append(np.asarray(block)[:, :live])
            del moved
        eigenvalues = np.concatenate(parts, axis=1)
        nspin = calculation.nspin
        return BandStructure(
            kpoints=path,
            eigenvalues=eigenvalues if nspin == 2 else eigenvalues[0],
            fermi_energy=fermi_energy,
            homo=homo,
            nspin=nspin,
            ethr=fixed_density_ethr(conv_thr, calculation.nelec),
        )
    calculation, system, kpoints, k_batch = threaded_calculation(
        calculation, system, kpoints, k_batch)
    calculation, system, eigenvalues = fixed_density_bands(
        system, pseudos, density, kpoints, nbnd, conv_thr, k_batch, ns, tau,
        becsum, field, field_scale, calculation=calculation,
    )
    nspin = calculation.nspin
    return BandStructure(
        kpoints=system.kpoints,
        eigenvalues=np.asarray(eigenvalues if nspin == 2 else eigenvalues[0]),
        fermi_energy=fermi_energy,
        homo=homo,
        nspin=nspin,
        ethr=fixed_density_ethr(conv_thr, calculation.nelec),
    )
