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

**Symmetry** (``symmetry=True``, P164): the Wannier functions are made to
transform under every operation of the run's space group exactly as the trial
orbitals do (Sakuma, PRB 87, 235109 (2013); Wannier90's ``site_symmetry``),
with ``D`` and ``d`` from :mod:`defuqumat.wannier.dmatrix` and the constraint
imposed by :mod:`defuqumat.wannier.sitesym`. The group is the one the SCF used,
``system.symmetry_group(nosym=system.nosym)``; with ``nosym`` it is the identity
and the option changes nothing. Time reversal is not imposed. A frozen set is
carried (P164b, which Wannier90 refuses): an energy window ``dis_froz``, or the
frozen states of ``projections="auto"``, whose selection is then made at one
point of each star and copied to the rest. Refused with it: a magnetic group
(an operation that needs time reversal, a collinear spin flip included),
projections given as a name other than ``"auto"``, a frozen set that carries an
irrep the trial orbitals do not, and ``exclude_bands`` given by hand.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from defuqumat.units import BOHR_TO_ANGSTROM, RY_TO_EV

__all__ = ["run_wannier", "WannierFunctions", "uniform_mesh"]

#: The gap below which two bands are one multiplet for the projectability
#: decisions, Ry, before :func:`~defuqumat.solvers.davidson.resolve_degeneracy_tol`
#: raises it to what the solve promises.
PAO_DEGENERACY_TOL = 1.0e-6


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
    #: ``M[m, n, nn, k]``, ``A[m, n, k]``, and the neighbour tables; with
    #: ``symmetry=True`` also the :class:`~defuqumat.wannier.sitesym.SymmetryData`
    #: of the ``.dmn`` file.
    mmn: np.ndarray = field(default=None, repr=False)
    amn: np.ndarray = field(default=None, repr=False)
    nnlist: np.ndarray = field(default=None, repr=False)
    nncell: np.ndarray = field(default=None, repr=False)
    ws_shifts: tuple | None = field(default=None, repr=False)
    #: ``exclude_bands`` in force, Wannier90's 1-based indices of the solved bands.
    excluded_bands: tuple = ()
    #: How many bands the fixed-density solve kept (``num_bands`` plus the excluded).
    solved_bands: int | None = None
    #: ``projections = "auto"`` only. ``(nk, num_bands)``: the projectability
    #: ``p_mk`` onto the orthonormalised pseudo-atomic orbitals, averaged over each
    #: degenerate multiplet (rule D4: the per-band value is not invariant under
    #: the eigensolver's rotation inside a multiplet), and the outer and
    #: frozen sets the disentanglement used; ``edge`` the Fermi level of a metal or
    #: the conduction band minimum of an insulator, Ry; ``projection_labels`` one
    #: name per function, ``"Si1 3S:s"``; ``semicore`` what the semicore test found.
    projectability: np.ndarray | None = field(default=None, repr=False)
    outer: np.ndarray | None = field(default=None, repr=False)
    frozen: np.ndarray | None = field(default=None, repr=False)
    edge: float | None = None
    projection_labels: tuple = ()
    semicore: str | None = None
    #: ``projections = "auto"``: the Wannier90 projection line of each function's
    #: angular part, for :meth:`write_w90`.
    projection_lines: tuple = field(default=(), repr=False)
    #: ``symmetry=True``: the site-symmetry data imposed, for :meth:`write_w90`.
    site_symmetry: object = field(default=None, repr=False)
    #: The number of space-group operations imposed (1: none).
    nsym: int = 1

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

        With site symmetry also ``prefix.dmn`` and ``site_symmetry = .true.``
        with its ``symmetrize_eps`` in the ``.win``. The ``.win`` carries the cell in bohr to twelve digits, the mesh in the
        order of :attr:`kpoints`, ``num_bands``, ``num_wann``, the
        ``projections`` lines and ``atoms`` (``(symbol, (f1, f2, f3))`` pairs),
        and ``extra`` keywords verbatim (the windows, ``num_iter``, ...).
        """
        from defuqumat.wannier import files

        files.write_mmn(f"{prefix}.mmn", self.mmn, self.nnlist, self.nncell)
        files.write_amn(f"{prefix}.amn", self.amn)
        files.write_eig(f"{prefix}.eig", self.eigenvalues.T * RY_TO_EV)
        lines = [f"num_bands = {self.num_bands}", f"num_wann = {self.num_wann}"]
        sym = self.site_symmetry
        if sym is not None:
            files.write_dmn(f"{prefix}.dmn", sym.ik2ir, sym.ir2ik, sym.kptsym,
                            sym.d_matrix_wann, sym.d_matrix_band)
            lines += ["site_symmetry = .true.", f"symmetrize_eps = {sym.symmetrize_eps:.1e}"]
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
    num_wann: int | None = None,
    projections="auto",
    divisions=None,
    num_bands: int | None = None,
    spin: int | None = None,
    dis_win=None,
    dis_froz=None,
    exclude_bands=None,
    proj_min: float = 0.01,
    proj_max: float = 0.95,
    froz_above_edge: float | None = 2.0,
    semicore="auto",
    fermi_energy: float | None = None,
    use_ws_distance: bool = True,
    symmetry: bool = False,
    symmetrize_eps: float = 1.0e-8,
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
        num_wann: the number of Wannier functions. With ``projections="auto"``
            it is the number of pseudo-atomic orbitals left after the semicore
            cut, and another value is refused.
        projections: ``"auto"`` (projectability disentanglement on the
            pseudo-atomic orbitals, :mod:`defuqumat.wannier.pao`), or Wannier90
            projection lines, as a list of strings (``"Si:sp3"``,
            ``"f=0,0,0.5:s"``, ``"C:l=1,mr=1:z=0,0,1"``); the first
            ``num_wann`` orbitals they define are the trial orbitals.
        divisions: the uniform Gamma-centred mesh ``(n1, n2, n3)``; defaults to
            the grid the k-points came from.
        num_bands: the bands taken in after ``exclude_bands``, Wannier90's
            meaning; defaults to ``num_wann`` for given projections (no
            disentanglement) and, for ``"auto"``, to twice the number of
            pseudo-atomic orbitals less the cut semicore bands.
        spin: at ``nspin = 2``, the channel, 0 or 1.
        dis_win, dis_froz: ``(min, max)`` of the outer and the frozen window in
            eV, either end ``None`` for open; Wannier90's ``dis_win_min/max``
            and ``dis_froz_min/max``. Not with ``"auto"``.
        exclude_bands: Wannier90's ``exclude_bands``, 1-based indices of the
            solved bands left out of everything. With ``"auto"`` it is set by
            the semicore cut and cannot be given.
        proj_min, proj_max: ``"auto"``: a multiplet whose mean projectability
            is below ``proj_min`` is dropped, at or above ``proj_max`` frozen.
        froz_above_edge: ``"auto"``: every multiplet up to this many eV above
            the edge (``fermi_energy`` for a metal, the conduction band minimum
            of an insulator) is frozen too; ``None`` for projectability alone.
        semicore: ``"auto"`` cuts the semicore shells whose bands lie below the
            valence with a gap, ``True`` cuts them whenever their bands can be
            named, ``False`` keeps them.
        fermi_energy: the Fermi level in Ry of a smeared ground state, which
            makes the run a metal for the edge; ``None`` for an insulator.
        use_ws_distance: Wannier90's ``use_ws_distance`` (its default) for the
            interpolation.
        symmetry: impose the space group of the run on the Wannier functions
            (Wannier90's ``site_symmetry``): ``U(gk) = d(g,k) U(k) D(g,k)^+``
            for every operation the SCF used. Needs a band set, an outer
            window and a frozen window that cut no degenerate multiplet; with
            ``"auto"`` the selection keeps whole multiplets and whole stars.
        symmetrize_eps: Wannier90's ``symmetrize_eps``, the tolerance of the
            little-group average at each irreducible k (its default is 1e-3;
            1e-8 here, so that the symmetry holds to that order).
        engine_options: any other field of
            :class:`~defuqumat.wannier.engine.EngineOptions`, in Wannier90's
            units (``num_iter``, ``conv_tol`` in A^2, ``conv_window``,
            ``dis_num_iter``, ``dis_mix_ratio``, ``guiding_centres``, ...).
        becsum, ns, tau, field, field_scale: the converged state that cannot be
            rebuilt from the density.
        conv_thr: the fixed-density solve's threshold.
    """
    from defuqumat.solvers.davidson import resolve_degeneracy_tol
    from defuqumat.wannier.engine import EngineOptions, absolute_u_opt, kmesh_get, run_engine
    from defuqumat.wannier.interpolation import wannier_hamiltonian, ws_distance_shifts
    from defuqumat.wannier.matrices import overlap_matrices, projection_matrices
    from defuqumat.wannier.projections import apply_projection_selection, parse_projections
    from defuqumat.workflows.topology import _occupied_bands, _source

    if getattr(system, "spiral", False) or system.npol == 2:
        raise NotImplementedError(
            "Wannier functions of a spinor calculation are not implemented: the "
            "projections onto spinor trial orbitals were not checked against "
            "pw2wannier90.x")
    if getattr(system.kpoints, "gamma_only", False):
        raise NotImplementedError(
            "Wannier functions of a K_POINTS gamma run are not implemented: the "
            "engine's gamma-point branch (dis_extract_gamma, overlap_project_gamma) "
            "is not ported; run with a k-point grid")
    if system.nspin == 2 and spin not in (0, 1):
        raise ValueError("at nspin = 2 the channel is wannierised alone: pass spin = 0 or 1")
    if system.nspin != 2 and spin is not None:
        raise ValueError("spin is only for nspin = 2")
    if symmetry:
        _refuse_for_symmetry(system, projections, dis_froz, engine_options)
        if exclude_bands:
            raise NotImplementedError("exclude_bands together with symmetry was not validated")
    if divisions is None:
        divisions = system.kpoints.grid
    if divisions is None:
        raise ValueError("Wannier functions need a uniform mesh: pass divisions=(n1, n2, n3)")
    divisions = tuple(int(n) for n in divisions)
    auto = isinstance(projections, str)
    if auto and projections != "auto":
        raise ValueError(f"projections must be 'auto' or a list of Wannier90 lines, not "
                         f"{projections!r}")
    if auto and (dis_win is not None or dis_froz is not None):
        raise ValueError("projections='auto' chooses the outer and frozen sets itself; "
                         "dis_win and dis_froz are for given projections")
    if auto and exclude_bands is not None:
        raise ValueError("projections='auto' sets exclude_bands by the semicore cut; "
                         "give the projections to exclude bands by hand")
    if auto and any(k in (engine_options or {}) for k in
                    ("dis_win_min", "dis_win_max", "dis_froz_min", "dis_froz_max")):
        raise ValueError("projections='auto' chooses the outer and frozen sets itself; "
                         "the windows in engine_options would not be read")

    points = uniform_mesh(divisions)
    at = np.asarray(system.cell.at, dtype=float)
    bg = np.asarray(system.cell.bg, dtype=float)
    volume = float(system.cell.volume)
    crystal = np.asarray(system.structure.positions_crystal(system.cell), dtype=float)
    symbols = list(system.structure.symbols)

    if auto:
        from defuqumat.wannier import pao as paomod

        paos = paomod.pao_projections(pseudos, system.structure, crystal)
        candidates = [i for i, p in enumerate(paos)
                      if p.orbital in paomod.semicore_candidates(pseudos[p.species])]
        n_full = len(paos)
        nsolve = (2 * n_full if num_bands is None
                  else int(num_bands) + (len(candidates) if semicore is not False else 0))
        excluded = ()
    else:
        if num_wann is None:
            raise ValueError("num_wann is needed with given projections")
        parsed = parse_projections(list(projections), symbols, crystal.T, bg)
        trial = apply_projection_selection(parsed, num_wann, None)
        num_bands = int(num_wann if num_bands is None else num_bands)
        if num_bands < num_wann:
            raise ValueError(f"num_bands = {num_bands} is fewer than num_wann = {num_wann}")
        excluded = tuple(sorted({int(b) for b in (exclude_bands or ())}))
        nsolve = num_bands + len(excluded)
        if excluded and (excluded[0] < 1 or excluded[-1] > nsolve):
            raise ValueError(f"exclude_bands must lie in 1..{nsolve} (num_bands plus the "
                             "excluded ones)")

    # The engine's reciprocal lattice in 1/A: its tolerances are Wannier90's.
    kmesh = kmesh_get(points.T, bg / BOHR_TO_ANGSTROM)

    source = _source(system, pseudos, density, nsolve, nsolve + 4, conv_thr, k_batch,
                     becsum=becsum, ns=ns, field=field, field_scale=field_scale,
                     calculation=calculation, spin=spin, tau=tau)
    # The gap test guards a topological invariant; a disentangled set is
    # entangled with the bands above it by construction.
    source = dataclasses.replace(source, gap_tol=0.0)
    states = source.states(points, keep_projectors=True)
    all_energies = np.asarray(states.energies, dtype=float)  # (nk, nsolve + 4) Ry
    k_batch_m = None if k_batch == "default" else k_batch

    extra = {}
    outer = frozen = None
    sym, nsym = None, 1
    if auto:
        tol_ev = resolve_degeneracy_tol(PAO_DEGENERACY_TOL, states.ethr) * RY_TO_EV
        energies_ev = all_energies[:, :nsolve] * RY_TO_EV
        amn_raw, overlap = paomod.pao_matrices(states, paos, pseudos, nsolve, volume)
        mmn_all = overlap_matrices(states, kmesh, nsolve, k_batch=k_batch_m)
        ortho_all, _ = paomod.orthonormal_projections(amn_raw, overlap)
        n_cut, semicore_note = paomod.semicore_cut(energies_ev, ortho_all, candidates, tol_ev,
                                                   mode=semicore)
        cut = set(candidates) if n_cut else set()
        keep_orbitals = [i for i in range(n_full) if i not in cut]
        n_kept = len(keep_orbitals)
        if num_wann is None:
            num_wann = n_kept
        elif int(num_wann) != n_kept:
            raise ValueError(f"projections='auto' makes {n_kept} functions, one per "
                             f"pseudo-atomic orbital kept; num_wann = {num_wann} was asked")
        num_wann = int(num_wann)
        num_bands = (nsolve - n_cut) if num_bands is None else int(num_bands)
        if n_cut + num_bands > nsolve:
            raise ValueError(f"num_bands = {num_bands} with {n_cut} excluded needs more than "
                             f"the {nsolve} bands solved")
        bands = np.arange(n_cut, n_cut + num_bands)
        excluded = tuple(range(1, n_cut + 1))
        amn_ortho, p_all = paomod.orthonormal_projections(amn_raw, overlap, keep_orbitals)
        amn = amn_ortho[bands]
        projectability = p_all[:, bands]
        energies = all_energies[:, bands]
        if fermi_energy is not None:
            edge = float(fermi_energy)
        else:
            nocc = _occupied_bands(system, pseudos, density, spin)
            if nocc >= all_energies.shape[1]:
                raise ValueError("the conduction band minimum needs more bands than were solved")
            edge = float(all_energies[:, nocc].min())
        froz_max = None if froz_above_edge is None else edge * RY_TO_EV + float(froz_above_edge)
        averaged = paomod.block_average(projectability, energies * RY_TO_EV, tol_ev)
        top = n_cut + num_bands
        above = all_energies[:, top] * RY_TO_EV if top < all_energies.shape[1] else None
        trial = [paos[i] for i in keep_orbitals]
        if symmetry:  # P164b: the representations first, so the selection keeps the stars
            sym, nsym = _symmetry_data(system, states, trial, num_bands, points,
                                       symmetrize_eps, bands=bands)
        del states
        outer, frozen = paomod.select_states(
            energies * RY_TO_EV, averaged, num_wann=num_wann, proj_min=proj_min,
            proj_max=proj_max, froz_max=froz_max, tol=tol_ev, energy_above=above,
            kpoints=points, sym=sym)
        mmn = mmn_all[np.ix_(bands, bands)]
        labels = tuple(f"{symbols[p.atom]}{p.atom + 1} {p.name}" for p in trial)
        extra = dict(projectability=averaged, outer=outer.T.copy(), frozen=frozen.T.copy(),
                     edge=edge, projection_labels=labels, semicore=semicore_note,
                     projection_lines=tuple(p.w90_line() for p in trial))
    else:
        keep = np.array([b for b in range(nsolve) if b + 1 not in set(excluded)])
        energies = all_energies[:, keep]
        if excluded:
            mmn = overlap_matrices(states, kmesh, nsolve, k_batch=k_batch_m)[np.ix_(keep, keep)]
            amn = projection_matrices(states, trial, nsolve, volume)[keep]
        else:
            mmn = overlap_matrices(states, kmesh, num_bands, k_batch=k_batch_m)
            amn = projection_matrices(states, trial, num_bands, volume)
        if symmetry:
            sym, nsym = _symmetry_data(system, states, trial, num_bands, points,
                                       symmetrize_eps)
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
                        mp_grid=divisions, site_cart=site_cart, outer=outer, frozen=frozen,
                        sym=sym)

    v = np.einsum("bwk,wnk->kbn", absolute_u_opt(result), result.u_matrix)  # (nk, nb, nw)
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
        ws_shifts=shifts, excluded_bands=excluded, solved_bands=nsolve, site_symmetry=sym,
        nsym=nsym, **extra)


def _symmetry_data(system, states, trial, num_bands, points, symmetrize_eps, bands=None):
    """``(sym, nsym)`` of the run's group; ``sym`` is ``None`` for the identity alone,
    which is then a no-op bit for bit (not routed through the symmetric engine)."""
    from defuqumat.wannier.dmatrix import operations_of, site_symmetry_data

    operations = operations_of(system.cell, system.symmetry_group(nosym=system.nosym))
    if operations.nsym == 1:
        return None, 1
    return site_symmetry_data(states, trial, num_bands, operations, points,
                              symmetrize_eps=symmetrize_eps, bands=bands), operations.nsym


def _refuse_for_symmetry(system, projections, dis_froz, engine_options) -> None:
    """What ``symmetry=True`` does not treat, refused before any state is solved.

    ``dis_froz`` is accepted since P164b; the argument stays for the callers."""
    if isinstance(projections, str) and projections != "auto":
        raise NotImplementedError(
            f"projections = {projections!r} together with symmetry is not implemented: "
            "the site symmetry needs trial orbitals that form a representation of the "
            "group, given as projection lines or as 'auto'")
    if (engine_options or {}).get("precond"):
        raise NotImplementedError("precond together with symmetry was not validated")
    group = system.symmetry_group(nosym=system.nosym)
    if group.magnetic:
        raise NotImplementedError(
            "symmetry with a magnetic group is not implemented: some operations of this "
            "run are symmetries only with time reversal (a spin flip), which the unitary "
            "site symmetry does not carry")
