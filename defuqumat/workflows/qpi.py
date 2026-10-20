"""``run_qpi``: quasiparticle interference around one impurity on a surface.

``PLAN.md`` P157. The physics is :mod:`defuqumat.stm.qpi`; what is here is the
part specific to a plane-wave run -- which bands, on which k-set, read where.

**The k-set is the supercell.** The impurity sits in a Born-von Karman
supercell of ``N1 x N2`` surface cells, one per in-plane k-point, and the maps
are periodic on it: a scattered wave that has not decayed by the supercell's
edge comes back from the neighbouring impurity. So the k-set must be a
**whole** regular grid with one division along the surface normal
(:func:`~defuqumat.workflows.transport.whole_grid` builds it from ``grid=``),
and the broadening must be large enough that the Friedel tail,
``exp(-eta r / v)``, has died within half the supercell; how much of
``|G0(r, phi)|``'s peak is left on the far edge is returned as
:attr:`~defuqumat.stm.qpi.QPI.wraparound`, a number rather than a warning.

**The scatterer's matrix element is the pseudo-wavefunction's**, which is the
true one only outside the augmentation spheres, so an ultrasoft or PAW dataset
is refused rather than corrected; the two planes are refused inside a sphere
exactly as :func:`~defuqumat.workflows.transport.run_vertical_transport` does.

Peak working set, beside the states themselves: ``nk nbnd npoints`` complex for
the tip amplitudes, ``nk nbnd^2`` for the exit Gram matrices, and per energy an
``(N, nk)`` phase matrix and ``(N, npoints)`` maps, ``N = nk`` -- on a 24x24
grid with 24 bands and 16x16 tip points, 57 MB, 53 MB and 2.4 MB.
"""

from __future__ import annotations

import numpy as np

from defuqumat.basis.sample import sample_wavefunctions
from defuqumat.scf.driver import Calculation
from defuqumat.scf.whole_sphere import unfold_gamma_states
from defuqumat.stm.qpi import (
    QPI,
    READERS,
    QPIStates,
    fourier_map,
    impurity_maps,
    impurity_overlaps,
    lattice_translations,
)
from defuqumat.system.kpoints import is_reduced
from defuqumat.transport.green import amplitude_weights
from defuqumat.transport.substrate import exit_overlap
from defuqumat.workflows.nscf import fixed_density_states, threaded_calculation
from defuqumat.workflows.stm import (
    _refuse_what_has_no_fermi_level,
    refuse_an_ultracell_result,
)

__all__ = ["run_qpi"]


def run_qpi(
    system,
    pseudos,
    result,
    *,
    potential: float,
    impurity=0,
    sigma: float = 1.0,
    energies=None,
    height: float,
    axis: int = 2,
    shape: tuple[int, int] = (8, 8),
    readers=("ldos",),
    exit_height: float | None = None,
    broadening: float = 0.02,
    exchange: float = 0.0,
    born: bool = False,
    exit_method: str = "spectral",
    smearing: str = "gaussian",
    bands=None,
    grid=None,
    shift=None,
    kpoints=None,
    nbnd: int | None = None,
    conv_thr: float = 1.0e-8,
    k_batch="default",
    calculation=None,
) -> QPI:
    """``delta rho(q, w)``: the QPI pattern of one impurity, by the T-matrix.

    Args:
        system, pseudos, result: the converged surface; its wavefunctions are
            needed, or a ``grid`` to re-solve them on.
        potential: ``V`` in Ry, the scatterer's strength: the energy shift of a
            state lying entirely inside the Gaussian. Negative attracts.
        impurity: where it sits -- an atom index (a substituted atom, the
            default being atom 0) or ``(3,)`` crystal coordinates.
        sigma: the Gaussian's width in bohr. It is the scatterer's size, and it
            sets the ``q`` range it scatters into (``exp(-sigma^2 q^2 / 2)``).
        energies: ``w`` in Ry, a scalar or a sequence; defaults to the Fermi
            level of the k-set used.
        height: the tip plane's crystal coordinate along ``axis``, in vacuum.
        axis: the surface normal, a lattice vector; the other two span the
            surface.
        shape: ``(m1, m2)`` tip points per surface cell. The transform shows
            ``m_i`` zones along each direction, so 2 already shows the first
            zone whole, and more resolve the decay into the vacuum.
        readers: ``"ldos"``, the Tersoff-Hamann local density of states at the
            tip, and/or ``"landauer"``, the tunnelling transmission from the
            tip point to the exit plane, ``int_exit |G(r, r')|^2``, which is
            what the STM's own matrix element weights the scattered waves by.
        exit_height: the exit (substrate) plane's crystal coordinate along
            ``axis``, on the other side of the slab; needed for ``landauer``.
        broadening: ``eta`` in Ry. It is the inverse lifetime the Green's
            function carries and must exceed the k-grid's level spacing near
            ``w``; see :attr:`~defuqumat.stm.qpi.QPI.wraparound`.
        exchange: for a collinear ``nspin = 2`` run, a magnetic scatterer:
            ``V + J`` for spin up and ``V - J`` for spin down, in Ry.
        born: the first Born approximation, ``T = V``.
        exit_method: the ``landauer`` reader's exit leg: ``"spectral"``, the
            on-shell amplitude, which converges with the window, or
            ``"resolvent"``, the literal ``1/D``, which does not (the module
            docstring of :mod:`defuqumat.stm.qpi` has the numbers).
        smearing: the on-shell delta's shape, a positive one.
        bands: a slice or index array, the band window the Green's function
            and the impurity live in. Defaults to every band computed; its
            dependence is the model's, as for a Wannier model.
        grid, shift, kpoints, nbnd, conv_thr, k_batch: re-solve the bands at
            fixed density on the whole grid, which a supercell of useful size
            always needs.
        calculation: the SCF's own :class:`~defuqumat.scf.driver.Calculation`.

    Returns a :class:`~defuqumat.stm.qpi.QPI`.
    """
    refuse_an_ultracell_result(result, "a QPI map", "the ordinary run")
    _refuse_what_has_no_fermi_level(system, result)
    readers = tuple(readers)
    for reader in readers:
        if reader not in READERS:
            raise ValueError(f"unknown reader {reader!r}; expected one of {READERS}")
    if "landauer" in readers and exit_height is None:
        raise ValueError(
            "the landauer reader collects the electron on an exit plane: pass "
            "exit_height, the plane's crystal coordinate on the far side of the "
            "slab")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
    if getattr(system, "noncolin", False) or getattr(system, "spiral", False):
        raise NotImplementedError(
            "QPI is implemented for nspin = 1 and collinear nspin = 2: a spinor "
            "run makes T a 2x2 matrix in spin and the Landauer reader a spin "
            "trace, neither of which has been checked against the dense "
            "supercell inverse")
    if any(p.augmentation is not None or p.paw is not None for p in pseudos):
        raise NotImplementedError(
            "an ultrasoft or PAW dataset is refused: the scatterer's matrix "
            "element <psi|phi> is taken with the pseudo-wavefunction, which is "
            "the true one only outside the augmentation spheres, and a "
            "scatterer on an atom is inside one")
    if not broadening > 0.0:
        raise ValueError(f"broadening must be positive, got {broadening}")

    from defuqumat.workflows.transport import (
        _refuse_an_augmented_plane, whole_grid,
    )

    if kpoints is None and grid is not None:
        kpoints = whole_grid(system, grid, shift)
    resolve = kpoints is not None
    calculation, system, kpoints, k_batch = threaded_calculation(
        calculation, system, kpoints, k_batch)
    if not resolve:
        if calculation is None:
            calculation = Calculation(system, pseudos, k_batch=k_batch)
        eigenvalues = np.asarray(result.eigenvalues_by_spin)
        wavefunctions = result.wavefunctions
        levels = {"fermi_energy": result.fermi_energy}
    else:
        calculation, system, eigenvalues, wavefunctions = fixed_density_states(
            system, pseudos, result.density, kpoints, nbnd, conv_thr, k_batch,
            ns=getattr(result, "ns", None),
            tau=getattr(result, "tau", None),
            calculation=calculation,
        )
        eigenvalues = np.asarray(eigenvalues)
        _, levels = calculation.occupations(eigenvalues)
    if wavefunctions is None:
        raise ValueError(
            "QPI is built from the wavefunctions and this result carries none: "
            "run the SCF without discarding them, or pass a grid")
    calculation, wavefunctions = unfold_gamma_states(calculation, wavefunctions)
    used = calculation.system
    if is_reduced(used.kpoints):
        raise NotImplementedError(
            "a symmetry-reduced k-set is refused: the k-grid is the impurity's "
            "supercell and every k-point of it is a state the impurity "
            "scatters into. Pass grid= for the whole grid, or run with nosym")

    cell = used.cell
    inplane = tuple(i for i in range(3) if i != axis)
    kcrystal = np.asarray(used.kpoints.crystal(cell), dtype=float)
    divisions = _divisions(kcrystal, axis, inplane)
    translations = lattice_translations(divisions)

    if isinstance(impurity, (int, np.integer)):
        positions = np.asarray(used.structure.positions_crystal(cell))
        position = positions[int(impurity)]
    else:
        position = np.asarray(impurity, dtype=float).reshape(3)
    planes = (float(height),) + (() if exit_height is None
                                 else (float(exit_height),))
    _refuse_an_augmented_plane(used, pseudos, axis, planes, "a QPI plane")

    m1, m2 = (int(m) for m in shape)
    tip = np.zeros((m1 * m2, 3))
    tip[:, inplane[0]] = np.repeat(np.arange(m1) / m1, m2)
    tip[:, inplane[1]] = np.tile(np.arange(m2) / m2, m1)
    tip[:, axis] = float(height)

    from defuqumat.workflows.transport import _geometry
    geometry = _geometry(calculation)
    miller = np.asarray(geometry.miller)
    mask = np.asarray(geometry.mask)
    bg = np.asarray(cell.bg)
    volume = float(cell.volume)
    nspin, nk, nall = eigenvalues.shape
    window = np.arange(nall)[bands] if bands is not None else np.arange(nall)

    energies_axis = np.atleast_1d(
        np.asarray(levels["fermi_energy"] if energies is None else energies,
                   dtype=float))
    potentials = ([float(potential)] if nspin == 1 else
                  [float(potential) + float(exchange),
                   float(potential) - float(exchange)])
    if nspin == 1 and exchange != 0.0:
        raise ValueError("a magnetic scatterer (exchange) needs an nspin = 2 run")

    channels = []
    for ispin in range(nspin):
        tip_values, overlaps, gram = [], [], []
        for ik in range(nk):
            block = np.asarray(wavefunctions[ispin, ik])[window]
            tip_values.append(sample_wavefunctions(
                block, miller[ik], kcrystal[ik], tip, volume, mask=mask[ik]))
            overlaps.append(impurity_overlaps(
                block, miller[ik], kcrystal[ik], position, bg, volume, sigma,
                mask=mask[ik]))
            if "landauer" in readers:
                gram.append(exit_overlap(block, miller[ik], float(exit_height),
                                         axis, cell, mask=mask[ik]))
        channels.append(QPIStates(
            eigenvalues=eigenvalues[ispin][:, window],
            tip=np.array(tip_values), overlaps=np.array(overlaps),
            kcrystal=kcrystal[:, list(inplane) + [axis]],
            exit=np.array(gram) if gram else None))

    ne, nr = energies_axis.size, translations.shape[0]
    by_spin = {r: np.zeros((nspin, ne, nr, m1 * m2)) for r in readers}
    background = {r: np.zeros((ne, m1 * m2)) for r in readers}
    tvalues = np.zeros((ne, nspin), dtype=complex)
    g0values = np.zeros((ne, nspin), dtype=complex)
    wrap = np.zeros((ne, nspin))
    contour = np.full((ne, nspin), np.nan)
    far = _far_cells(translations, divisions)
    # A spin-degenerate run carries both spins in one channel.
    degeneracy = 2.0 if nspin == 1 else 1.0
    for ie, energy in enumerate(energies_axis):
        for ispin, states in enumerate(channels):
            weights = (amplitude_weights(states.eigenvalues, float(energy),
                                         float(broadening), method=exit_method,
                                         smearing=smearing)
                       if "landauer" in readers else None)
            maps, backs, extras = impurity_maps(
                states, float(energy), float(broadening), potentials[ispin],
                translations, inplane=(0, 1), readers=readers, born=born,
                exit_weights=weights)
            for r in readers:
                by_spin[r][ispin, ie] = maps[r]
                background[r][ie] += degeneracy * backs[r]
            tvalues[ie, ispin] = extras["t_matrix"]
            g0values[ie, ispin] = extras["g0"]
            if weights is not None:
                per_k = np.sum(np.abs(weights) ** 2, axis=1)
                contour[ie, ispin] = per_k.sum() ** 2 / np.sum(per_k ** 2)
            amplitude = np.abs(extras["g_to"])
            wrap[ie, ispin] = amplitude[far].max() / amplitude.max()

    maps_out, transforms, spin_maps = {}, {}, {}
    for r in readers:
        total = degeneracy * by_spin[r].sum(axis=0)
        grid_map, transform = fourier_map(total, divisions, (m1, m2))
        # The impurity's cell is put in the middle of the map; the transform
        # was taken before, so its phase is relative to the cell origin.
        centre = (divisions[0] // 2 * m1, divisions[1] // 2 * m2)
        maps_out[r] = np.roll(grid_map, centre, axis=(-2, -1))
        transforms[r] = transform
        if nspin == 2:
            spin_maps[r] = np.roll(fourier_map(by_spin[r], divisions, (m1, m2))[0],
                                   centre, axis=(-2, -1))
    return QPI(
        energies=energies_axis,
        maps=maps_out,
        transforms=transforms,
        backgrounds={r: v.reshape(ne, m1, m2) for r, v in background.items()},
        divisions=divisions,
        shape=(m1, m2),
        impurity={"position": tuple(float(x) for x in position),
                  "potential": float(potential), "exchange": float(exchange),
                  "sigma": float(sigma)},
        t_matrix=tvalues,
        g0=g0values,
        height=float(height),
        axis=int(axis),
        broadening=float(broadening),
        wraparound=wrap,
        exit_height=None if exit_height is None else float(exit_height),
        fermi_energy=levels.get("fermi_energy"),
        born=bool(born),
        maps_by_spin=spin_maps or None,
        contour_points=contour,
        notes={"exit_method": exit_method, "bands": (int(window[0]), int(window[-1]) + 1),
               "nk": int(nk), "grid": None if grid is None else tuple(grid)},
    )


def _divisions(kcrystal, axis, inplane):
    """``(N1, N2)`` of a whole regular in-plane grid, or a refusal.

    The test is the one the lattice sum needs, not a reading of the input:
    the ``(N, nk)`` matrix ``e^{2 pi i k.R}`` over the supercell's translations
    must be ``sqrt(N)`` times a unitary, which is the statement that the
    k-points are exactly the supercell's Bloch vectors, once each.
    """
    along = np.unique(np.round(kcrystal[:, axis], 8))
    if along.size > 1:
        raise NotImplementedError(
            f"the k-set has {along.size} divisions along the surface normal and "
            "QPI needs one: the impurity is in a slab, periodic only in the "
            "plane")
    divisions = []
    for i in inplane:
        values = np.mod(np.round(kcrystal[:, i], 10), 1.0)
        divisions.append(int(np.unique(np.round(values, 8)).size))
    divisions = tuple(divisions)
    nk = kcrystal.shape[0]
    translations = lattice_translations(divisions)
    phases = np.exp(2.0j * np.pi * translations @ kcrystal[:, list(inplane)].T)
    if phases.shape[0] != nk or not np.allclose(
            phases.conj().T @ phases / nk, np.eye(nk), atol=1e-8):
        raise NotImplementedError(
            f"the {nk} k-points are not a whole {divisions[0]}x{divisions[1]} "
            "grid, so they are not the Bloch vectors of one supercell. Pass "
            "grid= for a Monkhorst-Pack grid")
    return divisions


def _far_cells(translations, divisions):
    """The translations on the supercell's far edge, as a boolean row mask."""
    n1, n2 = divisions
    a = np.minimum(translations[:, 0], n1 - translations[:, 0])
    b = np.minimum(translations[:, 1], n2 - translations[:, 1])
    return (a == n1 // 2) | (b == n2 // 2)
