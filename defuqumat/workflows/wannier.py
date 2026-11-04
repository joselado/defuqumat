"""``run_wannier``: maximally localized Wannier functions from a converged density.

This is the ``pw.x`` nscf, ``pw2wannier90.x`` and ``wannier90.x`` sequence in one
call. The states are solved at fixed density on the whole uniform mesh
(Wannier90 needs every k-point, as ``pw2wannier90`` needs an nscf with no
symmetry), the overlaps ``M`` and projections ``A`` are built from them
(:mod:`defuqumat.wannier.matrices`), and the engine
(:mod:`defuqumat.wannier.engine`) disentangles a subspace when there are more
bands than functions and then minimises the spread
(Marzari and Vanderbilt, PRB 56, 12847 (1997); Souza, Marzari and Vanderbilt,
PRB 65, 035109 (2001); Mostofi et al., Comput. Phys. Commun. 185, 2309 (2014)).

**Units.** The engine runs in Wannier90's own units, angstrom and eV, so that
its tolerances (``conv_tol`` in A^2, ``kmesh_tol`` in 1/A) and the windows mean
what they mean in a ``.win`` file; the result converts back to bohr and Ry, and
gives each length in angstrom beside it. ``zona`` is in 1/bohr, as
``pw2wannier90`` reads it.

**Refused**, by name: a spinor (``noncolin``) calculation and a spin spiral,
whose projections onto spinor trial orbitals were not checked against
``pw2wannier90.x``; and ``gamma_only``. At ``nspin = 2`` one channel is
wannierised at a time, ``spin = 0`` or ``1``, as ``pw2wannier90``'s
``spin_component``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from defuqumat.units import BOHR_TO_ANGSTROM, RY_TO_EV

__all__ = ["run_wannier", "WannierFunctions", "uniform_mesh"]


def uniform_mesh(divisions) -> np.ndarray:
    """The Gamma-centred mesh in crystal coordinates, the last index fastest
    (Wannier90's ``kmesh.pl`` order)."""
    n1, n2, n3 = (int(n) for n in divisions)
    grid = np.stack(np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3), indexing="ij"),
                    axis=-1).reshape(-1, 3)
    return grid / np.array([n1, n2, n3], dtype=float)


@dataclass(frozen=True)
class WannierFunctions:
    """Maximally localized Wannier functions and the model they define.

    Lengths are in bohr and energies in Ry; the ``*_angstrom`` and ``*_ev``
    properties convert. ``centres`` are cartesian; ``spreads`` are
    ``<r^2> - <r>^2`` of each function, and ``omega_i + omega_d + omega_od``
    is their sum ``omega_total``. ``omega_i`` is the gauge-invariant part,
    fixed by the subspace; the minimisation lowers ``omega_d + omega_od``.

    ``u_matrix`` is ``(nk, num_bands, num_wann)``: ``|w_nk> = sum_m
    |psi_mk> u[k, m, n]``, the disentanglement and the rotation together.
    ``hoppings`` is ``H_mn(R)`` in Ry on the lattice vectors ``rvectors``
    (crystal units, ``degeneracy`` each).
    """

    num_wann: int
    num_bands: int
    centres: np.ndarray
    spreads: np.ndarray
    omega_i: float
    omega_d: float
    omega_od: float
    kpoints: np.ndarray
    divisions: tuple
    eigenvalues: np.ndarray
    u_matrix: np.ndarray
    rvectors: np.ndarray
    degeneracy: np.ndarray
    hoppings: np.ndarray
    lattice: np.ndarray
    disentangled: bool
    converged: bool
    iterations: int
    spin: int | None = None
    #: Wannier90's layout, kept so that :meth:`write_w90` can hand them over:
    #: ``M[m, n, nn, k]``, ``A[m, n, k]``, and the neighbour tables.
    mmn: np.ndarray = field(default=None, repr=False)
    amn: np.ndarray = field(default=None, repr=False)
    nnlist: np.ndarray = field(default=None, repr=False)
    nncell: np.ndarray = field(default=None, repr=False)
    ws_shifts: tuple | None = field(default=None, repr=False)

    @property
    def omega_total(self) -> float:
        return self.omega_i + self.omega_d + self.omega_od

    @property
    def centres_angstrom(self) -> np.ndarray:
        return self.centres * BOHR_TO_ANGSTROM

    @property
    def spreads_angstrom2(self) -> np.ndarray:
        return self.spreads * BOHR_TO_ANGSTROM ** 2

    def spread_terms_angstrom2(self) -> dict:
        """``omega_i``, ``omega_d``, ``omega_od`` and the total in A^2, as
        ``wannier90.x`` prints them."""
        a2 = BOHR_TO_ANGSTROM ** 2
        return {"omega_i": self.omega_i * a2, "omega_d": self.omega_d * a2,
                "omega_od": self.omega_od * a2, "omega_total": self.omega_total * a2}

    def bands(self, kpoints) -> np.ndarray:
        """``(nk, num_wann)`` interpolated band energies in Ry at crystal ``kpoints``."""
        from defuqumat.wannier.interpolation import interpolate

        shifted, count = (None, None) if self.ws_shifts is None else self.ws_shifts
        values, _ = interpolate(np.asarray(kpoints, dtype=float), self.rvectors,
                                self.degeneracy, self.hoppings, shifted, count)
        return values

    def bands_ev(self, kpoints) -> np.ndarray:
        return self.bands(kpoints) * RY_TO_EV

    def write_w90(self, prefix, projections=(), atoms=(), extra: dict | None = None) -> None:
        """Write ``prefix.mmn``, ``.amn``, ``.eig`` and a ``.win`` for ``wannier90.x``.

        The ``.win`` carries the cell in bohr to twelve digits, the mesh in the
        order of :attr:`kpoints`, ``num_bands``, ``num_wann``, the
        ``projections`` lines and ``atoms`` (``(symbol, (f1, f2, f3))`` pairs),
        and ``extra`` keywords verbatim (the windows, ``num_iter``, ...).
        """
        from defuqumat.wannier import files

        files.write_mmn(f"{prefix}.mmn", self.mmn, self.nnlist, self.nncell)
        files.write_amn(f"{prefix}.amn", self.amn)
        files.write_eig(f"{prefix}.eig", self.eigenvalues.T * RY_TO_EV)
        lines = [f"num_bands = {self.num_bands}", f"num_wann = {self.num_wann}"]
        lines += [f"{k} = {v}" for k, v in (extra or {}).items()]
        lines += ["begin unit_cell_cart", "bohr"]
        lines += [" ".join(f"{x:.12f}" for x in row) for row in self.lattice]
        lines += ["end unit_cell_cart", "begin atoms_frac"]
        lines += [f"{sym} " + " ".join(f"{x:.12f}" for x in pos) for sym, pos in atoms]
        lines += ["end atoms_frac", "begin projections", *projections, "end projections"]
        lines += ["mp_grid = " + " ".join(str(n) for n in self.divisions), "begin kpoints"]
        lines += [" ".join(f"{x:.12f}" for x in k) for k in self.kpoints]
        lines += ["end kpoints"]
        with open(f"{prefix}.win", "w") as f:
            f.write("\n".join(lines) + "\n")


def run_wannier(
    system,
    pseudos,
    density,
    *,
    num_wann: int,
    projections,
    divisions=None,
    num_bands: int | None = None,
    spin: int | None = None,
    dis_win=None,
    dis_froz=None,
    use_ws_distance: bool = True,
    engine_options: dict | None = None,
    becsum: tuple = (),
    ns=None,
    tau=None,
    field=None,
    field_scale=None,
    conv_thr: float = 1.0e-10,
    k_batch="default",
    calculation=None,
) -> WannierFunctions:
    """Maximally localized Wannier functions of ``num_bands`` bands.

    Args:
        num_wann: the number of Wannier functions.
        projections: Wannier90 projection lines, as a list of strings
            (``"Si:sp3"``, ``"f=0,0,0.5:s"``, ``"C:l=1,mr=1:z=0,0,1"``); the
            first ``num_wann`` orbitals they define are the trial orbitals.
        divisions: the uniform Gamma-centred mesh ``(n1, n2, n3)``; defaults to
            the grid the k-points came from.
        num_bands: the lowest bands taken in; defaults to ``num_wann``, which
            needs no disentanglement.
        spin: at ``nspin = 2``, the channel, 0 or 1.
        dis_win, dis_froz: ``(min, max)`` of the outer and the frozen window in
            eV, either end ``None`` for open; Wannier90's ``dis_win_min/max``
            and ``dis_froz_min/max``.
        use_ws_distance: Wannier90's ``use_ws_distance`` (its default) for the
            interpolation.
        engine_options: any other field of
            :class:`~defuqumat.wannier.engine.EngineOptions`, in Wannier90's
            units (``num_iter``, ``conv_tol`` in A^2, ``conv_window``,
            ``dis_num_iter``, ``dis_mix_ratio``, ``guiding_centres``, ...).
        becsum, ns, tau, field, field_scale: the converged state that cannot be
            rebuilt from the density.
        conv_thr: the fixed-density solve's threshold.
    """
    from defuqumat.wannier.engine import EngineOptions, kmesh_get, run_engine
    from defuqumat.wannier.interpolation import wannier_hamiltonian, ws_distance_shifts
    from defuqumat.wannier.matrices import overlap_matrices, projection_matrices
    from defuqumat.wannier.projections import apply_projection_selection, parse_projections
    from defuqumat.workflows.topology import _source

    if getattr(system, "spiral", False) or system.npol == 2:
        raise NotImplementedError(
            "Wannier functions of a spinor calculation are not implemented: the "
            "projections onto spinor trial orbitals were not checked against "
            "pw2wannier90.x")
    if system.nspin == 2 and spin not in (0, 1):
        raise ValueError("at nspin = 2 the channel is wannierised alone: pass spin = 0 or 1")
    if system.nspin != 2 and spin is not None:
        raise ValueError("spin is only for nspin = 2")
    if divisions is None:
        divisions = system.kpoints.grid
    if divisions is None:
        raise ValueError("Wannier functions need a uniform mesh: pass divisions=(n1, n2, n3)")
    divisions = tuple(int(n) for n in divisions)
    num_bands = int(num_wann if num_bands is None else num_bands)
    if num_bands < num_wann:
        raise ValueError(f"num_bands = {num_bands} is fewer than num_wann = {num_wann}")

    points = uniform_mesh(divisions)
    at = np.asarray(system.cell.at, dtype=float)
    bg = np.asarray(system.cell.bg, dtype=float)
    volume = float(system.cell.volume)
    crystal = np.asarray(system.structure.positions_crystal(system.cell), dtype=float)
    symbols = list(system.structure.symbols)
    parsed = parse_projections(list(projections), symbols, crystal.T, bg)
    trial = apply_projection_selection(parsed, num_wann, None)

    # The engine's reciprocal lattice in 1/A: its tolerances are Wannier90's.
    kmesh = kmesh_get(points.T, bg / BOHR_TO_ANGSTROM)

    source = _source(system, pseudos, density, num_bands, num_bands + 4, conv_thr, k_batch,
                     becsum=becsum, ns=ns, field=field, field_scale=field_scale,
                     calculation=calculation, spin=spin, tau=tau)
    # The gap test guards a topological invariant; a disentangled set is
    # entangled with the bands above it by construction.
    source = dataclasses.replace(source, gap_tol=0.0)
    states = source.states(points, keep_projectors=True)
    energies = np.asarray(states.energies, dtype=float)[:, :num_bands]  # (nk, nb) Ry

    mmn = overlap_matrices(states, kmesh, num_bands, k_batch=None if k_batch == "default" else k_batch)
    amn = projection_matrices(states, trial, num_bands, volume)
    del states

    window = dict(engine_options or {})
    if dis_win is not None:
        window["dis_win_min"], window["dis_win_max"] = dis_win
    if dis_froz is not None:
        window["dis_froz_min"], window["dis_froz_max"] = dis_froz
    options = EngineOptions(**window)
    site_cart = np.array([p.site_frac for p in trial]) @ at * BOHR_TO_ANGSTROM
    result = run_engine(mmn, amn, energies.T * RY_TO_EV, kmesh, num_wann, options,
                        kpt_frac=points.T, real_lattice=at * BOHR_TO_ANGSTROM,
                        mp_grid=divisions, site_cart=site_cart)

    v = np.einsum("bwk,wnk->kbn", result.u_matrix_opt, result.u_matrix)  # (nk, nb, nw)
    irvec, ndegen, hr = wannier_hamiltonian(v, energies, points, divisions, at)
    a2 = BOHR_TO_ANGSTROM ** 2
    centres = result.centres / BOHR_TO_ANGSTROM
    shifts = ws_distance_shifts(irvec, centres, divisions, at) if use_ws_distance else None
    terms = result.terms
    return WannierFunctions(
        num_wann=num_wann, num_bands=num_bands, centres=centres,
        spreads=result.spreads / a2, omega_i=terms.om_i / a2, omega_d=terms.om_d / a2,
        omega_od=terms.om_od / a2, kpoints=points, divisions=divisions,
        eigenvalues=energies, u_matrix=v, rvectors=irvec, degeneracy=ndegen, hoppings=hr,
        lattice=at, disentangled=result.disentangled,
        converged=result.converged and result.dis_converged, iterations=result.iterations,
        spin=spin, mmn=mmn, amn=amn, nnlist=kmesh.nnlist, nncell=kmesh.nncell,
        ws_shifts=shifts)
