"""Pseudo-atomic orbitals as guiding functions, and projectability disentanglement.

Qiao, Pizzi and Marzari, "Projectability disentanglement for accurate and
automated electronic-structure Hamiltonians", arXiv:2303.07877 (PDWF). The
trial orbitals are the pseudo-atomic orbitals (``PP_PSWFC``) of every atom,
Loewdin-orthonormalised in the ``S`` metric, and each Bloch state is sorted by
its projectability onto them,

    p_mk = sum_n |<g~_n|S|psi_mk>|^2 = (A O^{-1} A^H)_mm,
    A_mn = <psi_mk|S|g_n>,  O_nn' = <g_n|S|g_n'>,

which is at most 1 because ``g~ = g O^{-1/2}`` spans a subspace: a state with
``p >= proj_max`` is frozen, one with ``p < proj_min`` is dropped, the rest are
disentangled. The paper's recommended form (its "PD+ED") also freezes every
state below an energy edge plus ``froz_above_edge`` (2 eV): the Fermi level of
a metal, the conduction band minimum of an insulator.

**The guiding functions are written in Wannier90's real harmonics**, not QE's:
``g(k+G) = (-i)^l Y^{W90}_{l,mr}(k+G) chi_l(|k+G|) e^{-i(k+G).tau}``, the
angular table and phase of :mod:`defuqumat.wannier.trial` with the radial part
of ``atomic_wfc`` (:func:`~defuqumat.pseudo.formfactors.atomic_form_factors`,
``4 pi / sqrt(Omega) int r chi(r) j_l(qr) dr`` on QE's 10-bohr mesh). So each
shell is the Wannier90 projection ``Atom:l=L`` with a pseudo-atomic radial
part, in ``mr`` order, and a symmetry representation built on
``trial.angular_parts`` applies to it unchanged. QE's own ``atomic_wfc`` carries
``i^l`` and QE's ``m`` order; the two sets differ inside each shell by a real
orthogonal matrix times ``(-1)^l``, which changes neither ``p_mk`` nor the
spread.

**The engine gets the orthonormalised projections** ``A O^{-1/2}``, as
``pw2wannier90.x`` with ``atom_proj_ortho = .true.`` (its default) writes them.
The engine's own Loewdin step would turn the raw ``A`` into the same initial
gauge only when ``O`` is the identity.

**Rule D4.** ``p_mk`` of one member of a degenerate multiplet depends on the
basis the eigensolver left inside it; the frozen/dropped decision is made on
its average over the multiplet, with the degeneracy tolerance resolved on the
solve's ``ethr``, so a multiplet is frozen, dropped or disentangled whole.

**Semicore shells.** A shell is a semicore candidate when the dataset has
another shell of the same ``l`` with a higher principal number, read from the
``PP_CHI`` label (``3S`` beside ``4S``; the UPF ``n`` attribute is a counter,
not the principal number). The paper excludes semicore orbitals; it is cut
here only when it can be done as Wannier90's ``exclude_bands`` does it: the
candidates' bands are the lowest ``N_sc`` at every k, carry most of their
weight (block-averaged projectability onto the semicore orbitals above 0.5
while every band above stays below), and lie below the rest with a gap at
every k, i.e. the semicore bands do not overlap the valence in energy.

Host-side NumPy, like :mod:`defuqumat.wannier.matrices`: ``A`` and ``O`` are
``(num_bands, npao, nk)`` and ``(npao, npao, nk)`` complex.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np

from defuqumat.wannier.sitesym import SiteSymmetryError

__all__ = [
    "PAOProjection",
    "pao_projections",
    "semicore_candidates",
    "pao_radial_tables",
    "pao_matrices",
    "orthonormal_projections",
    "degenerate_blocks",
    "block_average",
    "select_states",
    "symmetry_blocks",
    "semicore_cut",
]

_W90_NAMES = {0: ("s",), 1: ("pz", "px", "py"), 2: ("dz2", "dxz", "dyz", "dx2-y2", "dxy"),
              3: ("fz3", "fxz2", "fyz2", "fz(x2-y2)", "fxyz", "fx(x2-3y2)", "fy(3x2-y2)")}


@dataclass(eq=False)
class PAOProjection:
    """One pseudo-atomic orbital on one atom, as a Wannier90 ``(l, mr)`` function.

    It has the attributes :func:`~defuqumat.wannier.trial.guiding_functions`
    reads, with ``radial_key`` naming its radial table in place of the
    hydrogenic ``(zona, r)``.
    """

    atom: int
    species: int
    orbital: int          # index among the species' kept orbitals (occupation >= 0)
    label: str            # the PP_CHI label, "3S"
    shell: int | None     # the principal number read from the label
    l: int
    m: int                # Wannier90's mr, 1-based
    site_frac: np.ndarray
    z_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    x_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    zona: float = 1.0
    radial: int = 1
    spin: int | None = None
    s_qaxis: np.ndarray | None = None

    @property
    def radial_key(self) -> tuple:
        return ("pao", self.species, self.orbital)

    @property
    def name(self) -> str:
        return f"{self.label}:{_W90_NAMES[self.l][self.m - 1]}"

    def w90_line(self) -> str:
        """The Wannier90 projection line of this orbital's angular part."""
        f = ",".join(f"{x:.12f}" for x in self.site_frac)
        return f"f={f}:l={self.l},mr={self.m}"


def _principal(label: str) -> int | None:
    match = re.match(r"\s*(\d+)", label or "")
    return int(match.group(1)) if match else None


def _kept_orbitals(pseudo):
    """The orbitals ``atomic_wfc`` uses, with their radial index."""
    kept = []
    for orbital in pseudo.orbitals:
        if orbital.occupation < 0.0:
            continue
        kept.append(orbital)
    return kept


def pao_projections(pseudos, structure, crystal: np.ndarray) -> list[PAOProjection]:
    """Every kept orbital of every atom, atoms in order, then orbitals, then ``mr``.

    ``crystal`` is ``(nat, 3)``. A species with no ``PP_PSWFC`` is refused by
    name: its atoms would carry no guiding function.
    """
    out = []
    for atom, t in enumerate(np.asarray(structure.types)):
        pseudo = pseudos[int(t)]
        orbitals = _kept_orbitals(pseudo)
        if not orbitals:
            name = getattr(pseudo, "path", None) or pseudo.element
            raise ValueError(
                f"the pseudopotential of {structure.species[int(t)].name} ({name}) carries no "
                "pseudo-atomic orbital (PP_PSWFC), so projections='auto' has nothing to "
                "project its atoms on; use a dataset with orbitals or give the projections")
        for index, orbital in enumerate(orbitals):
            if orbital.l > 3:
                raise NotImplementedError(f"a pseudo-atomic orbital with l = {orbital.l} > 3")
            for mr in range(1, 2 * orbital.l + 2):
                out.append(PAOProjection(atom=atom, species=int(t), orbital=index,
                                         label=orbital.label, shell=_principal(orbital.label),
                                         l=int(orbital.l), m=mr,
                                         site_frac=np.asarray(crystal[atom], dtype=float)))
    return out


def semicore_candidates(pseudo) -> set[int]:
    """Kept-orbital indices of the shells with a same-``l`` partner of higher ``n``."""
    orbitals = _kept_orbitals(pseudo)
    shells = [(_principal(o.label), int(o.l)) for o in orbitals]
    out = set()
    for i, (n, l) in enumerate(shells):
        if n is None:
            continue
        if any(m is not None and ll == l and m > n for m, ll in shells):
            out.add(i)
    return out


def pao_radial_tables(pseudos, used_species, volume: float):
    """``q -> {("pao", t, i): (4, nq)}``, the radial part in row ``l``."""
    from defuqumat.pseudo.formfactors import atomic_form_factors

    def tables(q):
        out = {}
        for t in sorted(used_species):
            pseudo = pseudos[t]
            radial = np.asarray(atomic_form_factors(pseudo, np.asarray(q, dtype=float), volume),
                                dtype=float)
            for i, orbital in enumerate(_kept_orbitals(pseudo)):
                table = np.zeros((4, len(q)))
                table[orbital.l] = radial[i]
                out[("pao", t, i)] = table
        return out

    return tables


def pao_matrices(states, paos, pseudos, num_bands: int, volume: float):
    """``A`` ``(num_bands, npao, nk)`` and ``O`` ``(npao, npao, nk)`` in the ``S`` metric."""
    from defuqumat.topology.states import _augmentation_term, _cached_augmentation, _project
    from defuqumat.wannier.matrices import guiding_on_mesh

    if states.npol != 1:
        raise NotImplementedError("pseudo-atomic guiding functions for spinor states")
    factors = None
    if states.becp is not None:
        if states.vkb is None:
            raise ValueError("ultrasoft or PAW states need their projectors for S|g>")
        factors = _cached_augmentation(states._owner, np.zeros(3))
    nk = states.nk
    npao = len(paos)
    amn = np.zeros((num_bands, npao, nk), dtype=complex)
    overlap = np.zeros((npao, npao, nk), dtype=complex)
    extra = pao_radial_tables(pseudos, {p.species for p in paos}, volume)
    for k, g in guiding_on_mesh(states, paos, volume, extra_tables=extra):
        psi = np.asarray(states.coefficients[k])[:num_bands]
        a = psi.conj() @ g.T
        o = g.conj() @ g.T
        if factors is not None:
            becp_g = _project(g.astype(psi.dtype), states.vkb[k], 1)
            a = a + np.asarray(_augmentation_term(states.becp[k][:num_bands], becp_g, factors))
            o = o + np.asarray(_augmentation_term(becp_g, becp_g, factors))
        amn[:, :, k] = a
        overlap[:, :, k] = 0.5 * (o + o.conj().T)
    return amn, overlap


def orthonormal_projections(amn: np.ndarray, overlap: np.ndarray, keep=None):
    """``A O^{-1/2}`` on the kept orbitals and ``p_mk``, ``(nk, num_bands)``.

    The Loewdin step runs over the kept orbitals only, as ``pw2wannier90``
    excludes ``atom_proj_exclude`` before it builds ``O``.
    """
    if keep is None:
        keep = np.arange(amn.shape[1])
    keep = np.asarray(keep)
    a = amn[:, keep, :]
    o = overlap[np.ix_(keep, keep)]
    out = np.zeros_like(a)
    nk = a.shape[2]
    for k in range(nk):
        w, v = np.linalg.eigh(o[:, :, k])
        if w.min() <= 1.0e-10 * w.max():
            raise ValueError(f"the pseudo-atomic orbitals are linearly dependent at k-point {k} "
                             f"(smallest overlap eigenvalue {w.min():.3e})")
        out[:, :, k] = a[:, :, k] @ (v * w ** -0.5) @ v.conj().T
    p = np.einsum("mnk,mnk->km", out.conj(), out).real
    return out, p


def degenerate_blocks(energies: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """``[(start, stop), ...]``: runs of ascending ``energies`` closer than ``tol``."""
    blocks, start = [], 0
    for i in range(1, len(energies) + 1):
        if i == len(energies) or energies[i] - energies[i - 1] > tol:
            blocks.append((start, i))
            start = i
    return blocks


def block_average(values: np.ndarray, energies: np.ndarray, tol: float) -> np.ndarray:
    """``(nk, nb)``: each value replaced by its multiplet's mean (rule D4)."""
    out = np.empty_like(values, dtype=float)
    for k in range(values.shape[0]):
        for a, b in degenerate_blocks(energies[k], tol):
            out[k, a:b] = values[k, a:b].mean()
    return out


def select_states(energies: np.ndarray, p: np.ndarray, *, num_wann: int, proj_min: float,
                  proj_max: float, froz_max: float | None, tol: float,
                  energy_above: np.ndarray | None = None, kpoints=None, sym=None):
    """The outer and frozen sets, ``(nb, nk)`` bool each, in Wannier90's layout.

    ``energies`` and ``p`` are ``(nk, nb)``; ``froz_max`` freezes every
    multiplet at or below it (``None`` for none); ``energy_above`` ``(nk,)`` is
    the next band above ``nb``, when it was solved, so that a multiplet cut by
    the top of the set is seen and dropped whole. Decisions are on the
    block average of ``p`` over each multiplet. A frozen count above
    ``num_wann`` at some k is refused, naming the k.

    With ``sym`` (a :class:`~defuqumat.wannier.sitesym.SymmetryData` on these
    ``nb`` bands) the sets are the same at every point of a star by
    construction (:func:`_select_symmetric`).
    """
    if sym is not None:
        return _select_symmetric(energies, p, num_wann=num_wann, proj_min=proj_min,
                                 proj_max=proj_max, froz_max=froz_max, tol=tol,
                                 energy_above=energy_above, kpoints=kpoints, sym=sym)
    nk, nb = energies.shape
    outer = np.zeros((nb, nk), dtype=bool)
    frozen = np.zeros((nb, nk), dtype=bool)
    for k in range(nk):
        e = energies[k]
        blocks = degenerate_blocks(e, tol)
        for a, b in blocks:
            pbar = float(p[k, a:b].mean())
            ebar = float(e[a:b].mean())
            cut_at_top = (b == nb and energy_above is not None
                          and energy_above[k] - e[nb - 1] <= tol)
            froz = pbar >= proj_max or (froz_max is not None and ebar <= froz_max)
            if cut_at_top:
                if froz:
                    raise ValueError(
                        f"a frozen multiplet is cut by the top of the {nb} bands at k-point "
                        f"{_name(k, kpoints)}; raise num_bands")
                continue
            if froz:
                frozen[a:b, k] = True
                outer[a:b, k] = True
            elif pbar >= proj_min:
                outer[a:b, k] = True
        nfroz = int(frozen[:, k].sum())
        if nfroz > num_wann:
            raise ValueError(
                f"{nfroz} states are frozen at k-point {_name(k, kpoints)}, more than "
                f"num_wann = {num_wann}; lower froz_above_edge or raise proj_max")
        nout = int(outer[:, k].sum())
        if nout < num_wann:
            raise ValueError(
                f"only {nout} states are kept at k-point {_name(k, kpoints)}, fewer than "
                f"num_wann = {num_wann}; lower proj_min or raise num_bands")
    return outer, frozen


#: ``|d_mn|`` above this inside the little group puts bands ``m`` and ``n`` in one block.
MIXING_TOLERANCE = 1.0e-6


def symmetry_blocks(energies_k: np.ndarray, d: np.ndarray, little_group, tol: float):
    """The blocks of one k-point, lists of band indices: the bands the little group
    mixes (``|d_mn(g)| > MIXING_TOLERANCE``, the irreps), joined with the runs closer
    than ``tol`` in energy (rule D4: two irreps accidentally degenerate are not
    mixed by ``d``, but the eigensolver may mix them). ``d`` is ``(nb, nb, nsym)``."""
    nb = len(energies_k)
    parent = list(range(nb))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def join(i, j):
        a, b = root(i), root(j)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for a, b in degenerate_blocks(energies_k, tol):
        for i in range(a + 1, b):
            join(a, i)
    mixing = np.zeros((nb, nb), dtype=bool)
    for s in little_group:
        mixing |= np.abs(d[:, :, s]) > MIXING_TOLERANCE
    for i, j in zip(*np.nonzero(mixing)):
        join(int(i), int(j))
    groups = {}
    for i in range(nb):
        groups.setdefault(root(i), []).append(i)
    return [np.array(g) for g in sorted(groups.values(), key=lambda g: g[0])]


def _select_symmetric(energies, p, *, num_wann, proj_min, proj_max, froz_max, tol,
                      energy_above, kpoints, sym):
    """:func:`select_states` decided at the representatives of the stars only.

    The blocks are :func:`symmetry_blocks`; a block is frozen, kept or dropped
    whole on its mean ``p`` and mean energy, a block with the top band and the
    band above within ``tol``, or one that ``d`` maps partly outside the ``nb``
    bands (``sum_m |d_mn|^2 < 1``), is cut by the top and dropped (refused if it
    would be frozen), and the decision is copied to every point of the star
    by band index: the bands at ``g k`` are those at ``k``, in the same order of
    energy, so a block of whole multiplets at ``k`` is one at ``g k``.
    """
    nk, nb = energies.shape
    outer = np.zeros((nb, nk), dtype=bool)
    frozen = np.zeros((nb, nk), dtype=bool)
    for ir in range(sym.nkptirr):
        k = int(sym.ir2ik[ir])
        e = energies[k]
        little_group = np.nonzero(sym.kptsym[:, ir] == k)[0]
        d = sym.d_matrix_band[:, :, :, ir]
        for block in symmetry_blocks(e, d, little_group, tol):
            pbar = float(p[k, block].mean())
            ebar = float(e[block].mean())
            weight = min(float(np.sum(np.abs(d[:, block, s]) ** 2, axis=0).min())
                         for s in little_group)
            cut_at_top = ((nb - 1 in block and energy_above is not None
                           and energy_above[k] - e[nb - 1] <= tol) or weight < 1.0 - 1.0e-4)
            froz = pbar >= proj_max or (froz_max is not None and ebar <= froz_max)
            if cut_at_top:
                if froz:
                    raise SiteSymmetryError(
                        f"a frozen multiplet is cut by the top of the {nb} bands at k-point "
                        f"{_name(k, kpoints)}; raise num_bands")
                continue
            if froz:
                frozen[block, k] = True
                outer[block, k] = True
            elif pbar >= proj_min:
                outer[block, k] = True
        nfroz = int(frozen[:, k].sum())
        if nfroz > num_wann:
            raise SiteSymmetryError(
                f"{nfroz} states are frozen at k-point {_name(k, kpoints)}, more than "
                f"num_wann = {num_wann}; lower froz_above_edge or raise proj_max")
        nout = int(outer[:, k].sum())
        if nout < num_wann:
            raise SiteSymmetryError(
                f"only {nout} states are kept at k-point {_name(k, kpoints)}, fewer than "
                f"num_wann = {num_wann}; lower proj_min or raise num_bands")
        for s in range(sym.nsymmetry):
            jk = int(sym.kptsym[s, ir])
            outer[:, jk] = outer[:, k]
            frozen[:, jk] = frozen[:, k]
    return outer, frozen


def _name(k: int, kpoints) -> str:
    if kpoints is None:
        return str(k)
    return f"{k} ({', '.join(f'{x:.4f}' for x in kpoints[k])})"


def semicore_cut(energies: np.ndarray, amn_ortho: np.ndarray, semicore: np.ndarray, tol: float,
                 mode="auto"):
    """How many of the lowest bands are semicore and cut, and why.

    ``energies`` ``(nk, nb)``, ``amn_ortho`` ``(nb, npao, nk)`` orthonormalised
    over every orbital, ``semicore`` the candidate columns. Returns
    ``(n_cut, reason)``; ``n_cut`` is 0 when nothing is cut. ``mode`` is
    ``"auto"`` (cut when the bands are separated from the valence), ``True``
    (cut whenever the bands can be named) or ``False`` (never).
    """
    semicore = np.asarray(sorted(semicore), dtype=int)
    n_sc = len(semicore)
    if n_sc == 0 or mode is False:
        return 0, "no semicore shell" if n_sc == 0 else "semicore=False"
    psc = np.einsum("mnk,mnk->km", amn_ortho[:, semicore, :].conj(),
                    amn_ortho[:, semicore, :]).real
    psc = block_average(psc, energies, tol)
    low = psc[:, :n_sc]
    high = psc[:, n_sc:]
    named = low.min() > 0.5 and (high.size == 0 or high.max() < 0.5)
    if not named:
        message = (f"the {n_sc} semicore orbitals are not the lowest {n_sc} bands at every "
                   f"k-point (their weight there is {low.min():.3f} at least, and "
                   f"{high.max() if high.size else 0.0:.3f} at most above)")
        if mode is True:
            raise ValueError(message + ", so semicore=True cannot exclude them")
        warnings.warn(message + "; they are kept", stacklevel=3)
        return 0, "not the lowest bands"
    gap = float(energies[:, n_sc].min() - energies[:, n_sc - 1].max())
    if mode == "auto" and gap <= tol:
        return 0, f"overlaps the valence (gap {gap:.4g})"
    return n_sc, f"cut (gap {gap:.4g})"
