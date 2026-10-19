"""A finite homogeneous electric field by the Berry phase: ``pw.x``'s ``lelfield``.

A uniform field ``E`` cannot enter a periodic Hamiltonian as ``-e E . r``,
because ``r`` is not a periodic operator and the energy is not bounded from
below. Souza, Iniguez and Vanderbilt (PRL 89, 117602 (2002),
arXiv:cond-mat/0205633) and Umari and Pasquarello (PRL 89, 157602 (2002)) take
instead the **electric enthalpy**

    F[psi] = E_KS[psi] - E . P[psi],

with the polarization ``P`` the Berry phase of the occupied manifold on a
discrete string of k-points (King-Smith and Vanderbilt), and minimise it. The
minimum exists below a critical field set by the k-mesh (Zener breakdown on
the mesh), and the state it gives is the insulator polarized by the field.

**What is transcribed, and from where.** The derivative of the Berry term
with respect to ``<u_nk|`` is a Hermitian operator that couples ``k`` to its
two neighbours on the string. QE builds it in ``h_epsi_her_set.f90`` from the
**dual** vectors

    e+_m = sum_n |u_{k+dk,n}> (M+^-1)_{nm},   M+_{nm} = <u_{k,n}|u_{k+dk,m}>,
    e-_m = sum_n |u_{k-dk,n}> (M-^-1)_{nm},   M-_{nm} = <u_{k,n}|u_{k-dk,m}>,

each mapped onto the sphere of ``k``, and applies it in
``h_epsi_her_apply.f90`` as

    H_E psi = sum_n a_n <u_n|psi> + u_n <a_n|psi>,   a = fact (e- - e+),
    fact = -i E sqrt(2) |a_gdir| nppstr / (4 pi).

Since ``<u_j|e+-_m> = delta_jm``, ``<a|u> = 0`` and ``H_E u_n = a_n``, which
is ``(1/w_k) dE_P/d conj(u_nk)`` with ``w_k = 2/nk`` (checked in
``tests/unit/test_berryfield.py`` against ``jax.grad`` of :func:`phase_energy`).
The operator is built from the states **before** a diagonalisation and held
fixed during it, ``nberrycyc`` times per SCF iteration (``c_bands_efield``).

The energy is QE's ``calc_pol``: ``en_el = -E (el_pol + ion_pol + acc)``, with

    el_pol = 2 sqrt(2) |a_gdir|/(2 pi) (1/nkort) sum_s Im ln prod_k det M+(k),
    ion_pol = sqrt(2) sum_a Z_a tau_a(gdir),

per string on the **principal** branch (``c_phase_field.f90``), ``tau`` the
Cartesian component ``gdir`` of the unwrapped positions in bohr, and ``acc``
the ``+-2 pi fc_pol`` that ``calc_pol`` adds when the phase jumps by more than
``pi fc_pol`` between iterations. ``sqrt(2)`` is the electron's charge in
Rydberg units; ``efield`` is in Ry a.u. (36.3609e10 V/m).

**Spin.** At ``nspin = 2`` each channel has its own string set, its own
occupied count (``NINT(nelup)``, ``NINT(neldw)``) and its own operator, and
``el_pol`` is the sum of the two channels' string averages with no factor 2
(``c_phase_field.f90:612`` doubles only at ``nspin = 1``). A spinor (``npol =
2``) is one channel of ``NINT(nelec)`` occupied two-component states, again with
no factor 2 (``l_cal``, ``h_epsi_her_set.f90:225``), and the Miller gather is
applied to each component, as ``c_phase_field`` maps ``psi(ig+npwx)``.
``fact`` does not change in either: the factor 2 the energy loses is the
factor 2 the k weight ``w_k = 1/nk`` loses. A magnetized spinor is the same
code (measured against ``pw.x`` on the 90-degree hydrogen chain,
``reference.out.h-chain-90deg-lelfield``). A **spin spiral** keeps its two
components on the spheres of ``k + q/2`` and ``k - q/2`` (plane-wave rows
``ik`` and ``nk + ik``), so each component is gathered by its own index; both
spheres move by the same ``dk`` along a string, so the Miller shift at a
string end is ``e_gdir`` for both, and the spiral is the commensurate
supercell exactly (I85). Refused: ``nelup /= neldw``, where ``pw.x`` itself
builds the phase on ``nelec/2`` bands per channel (``l_cal`` reads ``f_inp``
only for ``occupations = 'from_input'``) whatever the channel holds, and a
spinor with spin-orbit coupling, for which no identity is measured here.

**Ultrasoft and PAW** (I103). Every product of two states becomes an ``S``
product, and the one between neighbours on a string is ``S`` between two
k-points (:mod:`defuqumat.topology.augmentation`):

    M+_{nm} = <u_{k,n}|u_{k+dk,m}>
              + sum_a sum_ij <psi_kn|beta_i^a> q^a_ij(dk) <beta_j^a|psi_{k+dk,m}>,
    q^a_ij(b) = Omega Q_ij(b) e^{-i b . tau_a},

``b`` the unwrapped step (``q_dk`` with ``struc`` in ``c_phase_field``, and
``q_dkp`` with ``CONJG(struc)`` for the backward link in ``h_epsi_her_set``).
The duals carry the same term: ``e+ = sum_n S(dk)|u_{k+dk,n}> (M+^-1)_{nm}``
with ``S(dk)|u'> = |u'> + sum |beta_i^k> q_ij(dk) <beta_j^{k+dk}|u'>``
("add US terms into evcm"), and ``h_epsi_her_apply`` puts ``S|u>`` where the
norm-conserving operator has ``|u>``: ``H_E = sum_n |a_n><S u_n| + |S u_n><a_n|``.
So ``<u_j|e_m> = delta_jm`` with the plain product, ``<S u_j|u_n> = delta_jn``,
and ``H_E u_n = a_n`` again. A neighbour's projections need no Miller gather,
since ``<beta^{k+G}|psi_{k+G}> = <beta^k|psi_k>``. The force
``forces_us_efield`` is not transcribed: :meth:`BerryFieldSetup.enthalpy`
builds ``M+`` from the projectors and structure phases of the moved
calculation, so ``jax.grad`` in the positions carries it. Measured against
``pw.x`` at ``nspin = 1`` only; ``nspin = 2``, a spinor and a spiral on an
augmented dataset are refused, as are the streamed store and the transcribed
force route, neither of which carries the term.

**What is not here**, refused by name at input: an augmented dataset outside
the regime just named, the two regimes named under **Spin**, a metal, ``efield_cart`` (three directions,
``l3dstring``), an automatic k-grid (QE turns that into ``l3dstring``), a cell
whose ``a_gdir`` is not along Cartesian axis ``gdir`` (QE's ionic dipole takes
the Cartesian component while the electronic one is along ``a_gdir``), the
stress (QE's field stress terms are commented out, ``stress.f90``), and
k-point pools.

**A streamed store** (``wfc_store = 'stream'``) keeps the states in host memory.
The operator is built from it a chunk at a time into host arrays before each
walk (:meth:`BerryFieldSetup.host_term`), and each chunk's solve is handed its
rows of it (:meth:`BerryFieldSetup.chunk_term`); the string phases are taken a
chunk at a time too. The device then holds one chunk's states, its
neighbours, and its rows of ``u`` and ``a``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["BerryFieldSetup", "BerryTerm", "string_layout", "phase_energy"]

SQRT2 = float(np.sqrt(2.0))
#: QE's switch-off: a field below this is no field (``h_epsi_her_set``,
#: ``h_epsi_her_apply``, ``c_bands``).
NO_FIELD = 1.0e-6


def string_layout(kcrystal: np.ndarray, gdir: int, nppstr: int) -> np.ndarray:
    """``(nkort, nppstr)`` k indices: consecutive blocks along ``gdir``, as ``setup.f90`` reads them.

    QE's manual k-list for ``lelfield`` is taken as strings of ``nppstr``
    consecutive points (``nx_el(ik, gdir) = ik``), the step being ``b_gdir /
    nppstr``. That is checked here rather than assumed: a list in another
    order would give a phase of nothing in particular.
    """
    nk = kcrystal.shape[0]
    if nk % nppstr:
        raise ValueError(f"lelfield: {nk} k-points are not strings of nppstr = {nppstr}")
    strings = np.arange(nk).reshape(nk // nppstr, nppstr)
    step = np.zeros(3)
    step[gdir] = 1.0 / nppstr
    for row in strings:
        differences = np.diff(kcrystal[row], axis=0) - step
        if not np.allclose(differences, 0.0, atol=1e-6):
            raise ValueError(
                "lelfield: the k-list is not strings of nppstr consecutive points "
                f"spaced by b_{gdir + 1}/nppstr (setup.f90 reads it that way); "
                f"points {row.tolist()} are not"
            )
    return strings


class BerryTerm(eqx.Module):
    """``H_E`` for one spin channel: ``sum_n |a_n><u_n| + |u_n><a_n|`` at every k.

    ``u`` and ``a`` are ``(nk, nocc, npol * npwx)``. ``u`` is a copy of the states the
    term was built from, never the block the eigensolver donates.

    ``rows`` is ``None`` for the whole k-set. For one chunk of a streamed store
    it is ``(nk,)`` and maps the global k index the solve passes (``indices``
    in :func:`~defuqumat.scf.streaming.stream_diagonalize`) to the row of ``u``
    and ``a``, which then hold that chunk's k-points only.
    """

    u: jnp.ndarray
    a: jnp.ndarray
    rows: jnp.ndarray | None = None

    def _at(self, ik):
        row = ik if self.rows is None else self.rows[ik]
        return self.u[row], self.a[row]

    def apply(self, psi: jnp.ndarray, ik) -> jnp.ndarray:
        u, a = self._at(ik)
        return (jnp.einsum("ng,...n->...g", a, jnp.einsum("ng,...g->...n", jnp.conj(u), psi))
                + jnp.einsum("ng,...n->...g", u, jnp.einsum("ng,...g->...n", jnp.conj(a), psi)))

    def matrix(self, ik) -> jnp.ndarray:
        u, a = self._at(ik)
        return a.T @ jnp.conj(u) + u.T @ jnp.conj(a)

    def diagonal(self, ik) -> jnp.ndarray:
        """Zero, as QE's ``h_diag``: ``Re sum_n a_n conj(u_n)`` is not sign-definite."""
        return jnp.zeros(self.u.shape[-1], dtype=jnp.real(self.u).dtype)


def _per_component(gather, found):
    """``(nk, ncomp, npwx)`` views of a gather: ``ncomp`` is 1 when every component shares it."""
    if gather.ndim == 2:
        return gather[:, None, :], found[:, None, :]
    return gather, found


def _gathered(coefficients, gather, found, npol: int = 1):
    """``u_{k'}`` on the sphere of ``k``: ``(nk, nb, npol * npwx)`` gathered by Miller index.

    A spinor's components are stored end to end (``evc(npwx*npol, nbnd)``).
    A ``(nk, npwx)`` gather is shared by every component, as ``c_phase_field``
    maps ``psi(ig+npwx)`` with the index of ``psi(ig)``. A spin spiral's
    components sit on two spheres, ``k + q/2`` and ``k - q/2``, and its gather
    is ``(nk, npol, npwx)``, one per component.
    """
    nk, nb, ndim = coefficients.shape
    gather, found = _per_component(gather, found)
    split = coefficients.reshape(nk, nb, npol, ndim // npol)
    picked = jnp.take_along_axis(split, gather[:, None, :, :], axis=3)
    return jnp.where(found[:, None, :, :], picked, 0.0).reshape(nk, nb, ndim)


def _overlaps(u, neighbour):
    """``M_{nm}(k) = <u_{k,n}|neighbour_{k,m}>``, ``(nk, nocc, nocc)``."""
    return jnp.einsum("kng,kmg->knm", jnp.conj(u), neighbour)


@jax.jit
def _duals(u, plus, minus):
    """``(e+, e-)`` on the sphere of each k, from the gathered neighbours."""
    m_plus = _overlaps(u, plus)
    m_minus = _overlaps(u, minus)
    # e_m = sum_n |nb_n> (M^-1)_{nm}: rows of M^-1 indexed by the neighbour band.
    e_plus = jnp.einsum("knm,kng->kmg", jnp.linalg.inv(m_plus), plus)
    e_minus = jnp.einsum("knm,kng->kmg", jnp.linalg.inv(m_minus), minus)
    return e_plus, e_minus


@eqx.filter_jit
def _augment(u, plus, minus, plus_index, minus_index, projectors, mask, qq, q_plus,
             q_minus, batch):
    """``(S u, S(+dk) plus, S(-dk) minus)`` on the sphere of each k, ``(nk, nocc, npwx)`` each.

    ``S(b)|u'> = |u'> + sum_ij |beta_i^k> q_ij(b) <beta_j^{k'}|u'>``, with the
    projections of the neighbour taken at the neighbour's own k-point (no
    gather: ``<beta^{k+G}|psi_{k+G}> = <beta^k|psi_k>``), and ``b = 0`` giving
    ``S`` itself through ``qq``. One k-point's projectors at a time
    (:meth:`~defuqumat.pseudo.projectors.Projectors.at_k`), walked by ``map_k``;
    the states are arguments, never closed over by an eager loop.
    """
    from defuqumat.batching import map_k

    rows = jnp.arange(u.shape[0])

    def projectors_at(ik):
        return jnp.where(mask[ik][:, None], projectors.at_k(ik), 0.0)

    becp = map_k(lambda ik: jnp.einsum("gi,ng->ni", jnp.conj(projectors_at(ik)), u[ik]),
                 rows, batch=batch)

    def apply(ik):
        vkb = projectors_at(ik)
        return (u[ik] + jnp.einsum("gi,ij,nj->ng", vkb, qq, becp[ik]),
                plus[ik] + jnp.einsum("gi,ij,nj->ng", vkb, q_plus, becp[plus_index[ik]]),
                minus[ik] + jnp.einsum("gi,ij,nj->ng", vkb, q_minus,
                                       becp[minus_index[ik]]))

    return map_k(apply, rows, batch=batch)


def phase_energy(u, plus, strings, prefactor):
    """``-E el_pol`` as a function of the states: the Berry term of the enthalpy.

    ``plus`` is the forward neighbour of each k gathered onto its sphere,
    ``strings`` the ``(nkort, nppstr)`` layout and ``prefactor`` ``-E 2
    sqrt(2) |a_gdir|/(2 pi)/nkort``. The phase of each string is the principal
    value of the product of its links' determinants, through ``slogdet`` so
    that a large manifold cannot underflow. Differentiable in ``u`` (and in
    ``plus`` when it is gathered from the same ``u``).
    """
    sign, _ = jnp.linalg.slogdet(_overlaps(u, plus))
    per_string = jnp.prod(sign[strings], axis=1)
    return prefactor * jnp.sum(jnp.angle(per_string))


@dataclass
class BerryFieldSetup:
    """Everything about ``lelfield`` that is fixed for a run, built once.

    ``plus``/``minus`` gathers are ``(nk, npwx)`` indices into the neighbour's
    plane-wave list with a ``found`` mask, the Miller shift ``+-e_gdir``
    applied at the two ends of each string (``u_{k+G}(G') = u_k(G'+G)``, a pure
    relabelling with no phase, ``gtr = g -+ gpar`` in ``h_epsi_her_set``).
    """

    gdir: int
    nppstr: int
    efield: float
    nberrycyc: int
    #: Occupied bands: one count, or one per channel at ``nspin = 2``.
    nocc: int | tuple
    strings: np.ndarray
    plus_index: np.ndarray  # (nk,) neighbour k
    minus_index: np.ndarray
    plus_gather: jnp.ndarray
    plus_found: jnp.ndarray
    minus_gather: jnp.ndarray
    minus_found: jnp.ndarray
    length: float  # |a_gdir| in bohr
    charges: np.ndarray
    #: Spinor components of a state, and how many electrons one band holds:
    #: the factor ``c_phase_field`` puts on ``el_pol`` at ``nspin = 1`` only.
    npol: int = 1
    spin_factor: float = 2.0
    #: The chunk a streamed store is walked in (the calculation's ``k_batch``).
    k_batch: int | None = None
    #: ``el_pol_acc`` and ``el_pol_old`` of ``calc_pol``: host state carried
    #: across the iterations of one run.
    accumulated: float = 0.0
    previous: float | None = field(default=None)
    #: An ultrasoft or PAW dataset (I103): ``Omega Q_ij(+dk)`` and
    #: ``Omega Q_ij(-dk)`` of each species, ``(nh, nh)`` or ``None``, with no
    #: structure phase (that is applied with the positions, traced), and the
    #: forward step ``dk`` in cartesian 1/bohr. ``None`` on a norm-conserving
    #: dataset, where ``S`` is the identity.
    cross: tuple | None = None
    dk: np.ndarray | None = None
    #: Drops ``q_ij(+-dk)`` from the overlaps and the duals and keeps ``S u``:
    #: the missing term of I103 switched off, for the record's A/B only.
    cross_off: bool = False

    @classmethod
    def build(cls, calculation, settings):
        from defuqumat.topology.states import build_plane_wave_states

        gdir, nppstr, efield, nberrycyc = settings
        system = calculation.system
        cell = system.cell
        kcrystal = np.asarray(system.kpoints.crystal(cell))
        strings = string_layout(kcrystal, gdir, nppstr)
        nk = kcrystal.shape[0]
        npwx = calculation.basis.planewaves.mask.shape[1]
        dummy = jnp.zeros((nk, 1, npwx), dtype=cell.precision.complex)
        states = build_plane_wave_states(calculation, dummy)
        unit = np.zeros(3, dtype=int)
        unit[gdir] = 1
        plus_index = np.empty(nk, dtype=int)
        minus_index = np.empty(nk, dtype=int)
        spiral = bool(getattr(calculation, "spiral", False))

        def aligned(ik, jk, shift):
            """One gather shared by every component, or a spiral's two.

            A spiral's plane-wave rows are ``k + q/2`` then ``k - q/2``
            (:func:`~defuqumat.system.spiral.spiral_kpoints`), so the down
            component of ``ik`` lives on row ``nk + ik`` and its neighbour's on
            ``nk + jk``. The Miller shift at a string end is the same ``e_gdir``
            for both, since both spheres move by the same ``dk``.
            """
            if not spiral:
                return states._alignment(ik, jk, shift)
            pairs = [states._alignment(ik, jk, shift),
                     states._alignment(nk + ik, nk + jk, shift)]
            return jnp.stack([g for g, _ in pairs]), jnp.stack([f for _, f in pairs])
        gathers = {"plus": [], "minus": []}
        for row in strings:
            for position, ik in enumerate(row):
                last = position == nppstr - 1
                first = position == 0
                jp = row[0] if last else row[position + 1]
                jm = row[-1] if first else row[position - 1]
                plus_index[ik], minus_index[ik] = jp, jm
                gathers["plus"].append((ik, aligned(ik, jp, unit if last else None)))
                gathers["minus"].append((ik, aligned(ik, jm, -unit if first else None)))
        def stack(entries):
            entries = sorted(entries, key=lambda e: e[0])
            return (jnp.stack([g for _, (g, f) in entries]),
                    jnp.stack([f for _, (g, f) in entries]))
        plus_gather, plus_found = stack(gathers["plus"])
        minus_gather, minus_found = stack(gathers["minus"])
        at = np.asarray(cell.at)
        nint = lambda x: int(np.floor(float(x) + 0.5))
        if system.nspin == 2:
            nocc = (nint(calculation.nelup), nint(calculation.neldw))
        elif system.nspin == 4:
            nocc = nint(calculation.nelec)
        else:
            nocc = nint(calculation.nelec / 2)
        cross, dk = None, None
        if calculation.augmentation is not None:
            from defuqumat.topology.augmentation import augmentation_species_at_q

            # The unwrapped step, read where the string wraps: the last point's
            # neighbour is the first plus e_gdir, and the difference is the same
            # b_gdir/nppstr as everywhere else on the string.
            row = strings[0]
            dk = np.asarray(states._difference(int(row[-1]), int(row[0]), unit))
            cross = (augmentation_species_at_q(calculation, dk),
                     augmentation_species_at_q(calculation, -dk))
        return cls(
            cross=cross, dk=dk,
            gdir=gdir, nppstr=nppstr, efield=float(efield), nberrycyc=int(nberrycyc),
            nocc=nocc, strings=strings, npol=int(system.npol),
            spin_factor=2.0 if system.nspin == 1 else 1.0, k_batch=calculation.k_batch,
            plus_index=plus_index, minus_index=minus_index,
            plus_gather=plus_gather, plus_found=plus_found,
            minus_gather=minus_gather, minus_found=minus_found,
            length=float(np.linalg.norm(at[gdir])),
            charges=np.asarray(calculation.charges, dtype=float),
        )

    # --- the operator ---------------------------------------------------------
    @property
    def active(self) -> bool:
        return abs(self.efield) >= NO_FIELD

    def occupied(self, spin: int = 0) -> int:
        """The occupied count of channel ``spin``."""
        return self.nocc[spin] if isinstance(self.nocc, tuple) else self.nocc

    def neighbours(self, u):
        """The forward and backward neighbours of each k, gathered onto its sphere."""
        plus = _gathered(u[self.plus_index], self.plus_gather, self.plus_found, self.npol)
        minus = _gathered(u[self.minus_index], self.minus_gather, self.minus_found,
                          self.npol)
        return plus, minus

    @property
    def augmented(self) -> bool:
        """Whether the dataset carries an augmentation charge, so ``S`` is not the identity."""
        return self.cross is not None

    def _cross_matrix(self, calculation, side: int, dtype) -> jnp.ndarray:
        """``q^a_ij(+-dk) = Omega Q_ij(+-dk) e^{-i (+-dk) . tau_a}``, ``(nkb, nkb)``.

        ``side`` 0 is the forward link, 1 the backward one. The phase is taken
        from the calculation's positions with ``jnp``, so that under
        ``jax.grad`` in the positions it is part of the force.
        """
        augmentation = calculation.augmentation
        sign = 1.0 if side == 0 else -1.0
        real = jnp.finfo(dtype).dtype
        positions = jnp.asarray(calculation.system.structure.positions, dtype=real)
        phases = jnp.exp(-1j * (positions @ jnp.asarray(sign * self.dk, dtype=real)))
        phases = phases.astype(dtype)
        blocks = tuple(
            None if values is None or not atoms
            else jnp.asarray(values, dtype=dtype)[None]
            * phases[jnp.asarray(atoms)][:, None, None]
            for values, atoms in zip(self.cross[side], augmentation.species_atoms))
        return augmentation.block_matrix(blocks).astype(dtype)

    def augmented_neighbours(self, u, calculation=None):
        """``(S u, S(+dk) u_{k+dk}, S(-dk) u_{k-dk})`` on the sphere of each k.

        On a norm-conserving dataset this is ``(u, plus, minus)`` of
        :meth:`neighbours`. On an augmented one it needs the calculation, for
        its projectors, ``qq`` and positions: the projectors and positions of a
        moved copy are what put ``forces_us_efield`` in the gradient.
        """
        from defuqumat.batching import resolve_k_batch

        plus, minus = self.neighbours(u)
        if not self.augmented:
            return u, plus, minus
        if calculation is None:
            raise ValueError(
                "lelfield on an ultrasoft or PAW dataset needs the calculation: "
                "the string overlap is an S product between two k-points")
        dtype = u.dtype
        q_plus = self._cross_matrix(calculation, 0, dtype)
        q_minus = self._cross_matrix(calculation, 1, dtype)
        if self.cross_off:
            q_plus, q_minus = jnp.zeros_like(q_plus), jnp.zeros_like(q_minus)
        projectors = calculation.projectors
        return _augment(u, plus, minus, jnp.asarray(self.plus_index),
                        jnp.asarray(self.minus_index), projectors,
                        jnp.asarray(calculation.basis.planewaves.mask),
                        projectors.qq.astype(dtype), q_plus, q_minus,
                        resolve_k_batch(self.k_batch))

    def _no_augmented_host_store(self):
        if self.augmented:
            raise NotImplementedError(
                "lelfield on an ultrasoft or PAW dataset with a streamed store "
                "(wfc_store = 'stream') is not implemented: the cross augmentation "
                "is built only on the in-memory store")

    @property
    def fact(self) -> complex:
        """``fact_hepsi``: ``-i E sqrt(2)/(2 dkfact)``, ``dkfact = 2 pi/(|a_gdir| nppstr)``."""
        return -1j * self.efield * SQRT2 * self.length * self.nppstr / (4.0 * np.pi)

    # --- a streamed store: the same quantities a chunk of k at a time --------
    def _host_chunks(self, store, spin: int):
        """``(rows, live, u, plus, minus)`` for each chunk of a host store, on the device.

        Only the chunk's own states and its neighbours' cross: the neighbours
        are gathered on the host from the rows ``plus_index``/``minus_index``
        name, by the same Miller index the whole-set path gathers by.
        """
        from defuqumat.batching import k_chunks

        self._no_augmented_host_store()
        nocc = self.occupied(spin)
        sides = [(self.plus_index, np.asarray(self.plus_gather), np.asarray(self.plus_found)),
                 (self.minus_index, np.asarray(self.minus_gather),
                  np.asarray(self.minus_found))]

        def gathered(index, gather, found, rows):
            source = store[spin, index[rows], :nocc]
            nk, nb, ndim = source.shape
            split = source.reshape(nk, nb, self.npol, ndim // self.npol)
            gather_r, found_r = _per_component(gather[rows], found[rows])
            picked = np.take_along_axis(split, gather_r[:, None, :, :], axis=3)
            picked = np.where(found_r[:, None, :, :], picked, 0.0)
            return jnp.asarray(picked.reshape(nk, nb, ndim))

        for rows, live in k_chunks(store.shape[1], self.k_batch):
            yield (rows, live, jnp.asarray(store[spin, rows, :nocc]),
                   gathered(*sides[0], rows), gathered(*sides[1], rows))

    def host_term(self, store, spin: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """``(u, a)`` of channel ``spin`` for the whole k-set, in host memory.

        Built from the states **before** the walk that diagonalises, as
        ``h_epsi_her_set`` builds every dual before ``c_bands``: the walk
        overwrites the store chunk by chunk, so a term built as it goes would
        see a neighbour already solved. ``(nk, nocc, npol * npwx)`` twice, on
        the host beside the store, where QE writes the duals to disk.
        """
        nocc = self.occupied(spin)
        u_host = np.array(store[spin, :, :nocc])
        a_host = np.empty_like(u_host)
        for rows, live, u, plus, minus in self._host_chunks(store, spin):
            e_plus, e_minus = _duals(u, plus, minus)
            a = (self.fact * (e_minus - e_plus)).astype(u.dtype)
            a_host[rows[:live]] = np.asarray(a)[:live]
        return u_host, a_host

    @staticmethod
    def chunk_term(host, rows) -> BerryTerm:
        """One chunk's :class:`BerryTerm` from :meth:`host_term`'s arrays, indexed globally."""
        u_host, a_host = host
        lookup = np.zeros(u_host.shape[0], dtype=np.int32)
        lookup[np.asarray(rows)] = np.arange(len(rows), dtype=np.int32)
        return BerryTerm(u=jnp.asarray(u_host[rows]), a=jnp.asarray(a_host[rows]),
                         rows=jnp.asarray(lookup))

    def _link_signs(self, wavefunctions, spin: int, calculation=None) -> np.ndarray:
        """``det M+(k) / |det M+(k)|`` at every k, ``(nk,)``, a chunk at a time for a host store."""
        if isinstance(wavefunctions, np.ndarray):
            signs = np.empty(wavefunctions.shape[1], dtype=wavefunctions.dtype)
            for rows, live, u, plus, _ in self._host_chunks(wavefunctions, spin):
                sign, _ = jnp.linalg.slogdet(_overlaps(u, plus))
                signs[rows[:live]] = np.asarray(sign)[:live]
            return signs
        u = wavefunctions[spin, :, : self.occupied(spin)]
        _, plus, _ = self.augmented_neighbours(u, calculation)
        sign, _ = jnp.linalg.slogdet(_overlaps(u, plus))
        return np.asarray(sign)

    def term(self, wavefunctions, spin: int = 0, calculation=None) -> BerryTerm:
        """``H_E`` of channel ``spin`` from ``wavefunctions`` ``(nspin, nk, nbnd, npol * npwx)``.

        On an augmented dataset the term holds ``S u`` in place of ``u``
        (``h_epsi_her_apply``'s ``evct``), so it applies
        ``|a><S u| + |S u><a|``.
        """
        u = jnp.array(wavefunctions[spin, :, : self.occupied(spin)], copy=True)
        su, plus, minus = self.augmented_neighbours(u, calculation)
        e_plus, e_minus = _duals(u, plus, minus)
        a = (self.fact * (e_minus - e_plus)).astype(u.dtype)
        return BerryTerm(u=su if self.augmented else u, a=a)

    # --- the polarization and the energy -------------------------------------
    def enthalpy(self, wavefunctions, positions, calculation=None):
        """``-E (el_pol + ion_pol + acc)`` as a traced function, for the frozen-energy force.

        The same number :meth:`energy` returns at the converged state, written
        in ``jnp`` so that ``jax.grad`` in the positions gives
        ``forces_ion_efield``, ``sqrt(2) E Z`` along Cartesian ``gdir``. The
        electronic part has no position dependence for a norm-conserving
        dataset and contributes nothing to the force. On an augmented one it
        does, through the projections and the structure phase in ``M+``, and
        ``calculation`` (the moved copy, traced) is what carries them: that
        gradient is ``forces_us_efield``.
        """
        if not self.active:
            return jnp.zeros((), dtype=jnp.real(wavefunctions).dtype)
        prefactor = (-self.efield * self.spin_factor * self.fc_pol
                     / self.strings.shape[0])
        electronic = 0.0
        for spin in range(wavefunctions.shape[0]):
            u = wavefunctions[spin, :, : self.occupied(spin)]
            _, plus, _ = self.augmented_neighbours(u, calculation)
            electronic = electronic + phase_energy(u, plus, self.strings, prefactor)
        charges = jnp.asarray(self.charges, dtype=positions.dtype)
        ionic = SQRT2 * jnp.sum(charges * positions[:, self.gdir])
        return electronic - self.efield * (ionic + self.accumulated)

    def string_phases(self, wavefunctions, calculation=None) -> np.ndarray:
        """Each string's principal phase, ``Im ln prod det M+``, radians, ``(nchannel, nkort)``."""
        phases = []
        for spin in range(wavefunctions.shape[0]):
            sign = self._link_signs(wavefunctions, spin, calculation)
            phases.append(np.angle(np.prod(sign[self.strings], axis=1)))
        return np.array(phases)

    @property
    def fc_pol(self) -> float:
        """``fact_pola``: ``sqrt(2)/tpiba * |a_gdir|/alat``, i.e. ``sqrt(2) |a_gdir|/(2 pi)``."""
        return SQRT2 * self.length / (2.0 * np.pi)

    def electronic_dipole(self, wavefunctions, calculation=None) -> float:
        """``el_pol`` in Ry a.u.: each channel's string average, summed, times 2 at ``nspin = 1``."""
        phases = self.string_phases(wavefunctions, calculation)
        return float(self.spin_factor * self.fc_pol * np.sum(np.mean(phases, axis=1)))

    def ionic_enthalpy(self, positions):
        """``-E (ion_pol + acc)`` as a traced function of the positions.

        The part of :meth:`enthalpy` that moves with the atoms on a
        norm-conserving dataset, so the whole of the field's term in a force
        that walks a streamed store a chunk at a time
        (:mod:`defuqumat.forces.chunked`); the electronic part is a constant
        there and is added to the energy on the host.
        """
        if not self.active:
            return jnp.zeros((), dtype=positions.dtype)
        charges = jnp.asarray(self.charges, dtype=positions.dtype)
        ionic = SQRT2 * jnp.sum(charges * positions[:, self.gdir])
        return -self.efield * (ionic + self.accumulated)

    def ionic_dipole(self, positions) -> float:
        """``ion_pol``: ``sqrt(2) sum Z tau(gdir)``, unwrapped positions in bohr."""
        return float(SQRT2 * np.sum(self.charges * np.asarray(positions)[:, self.gdir]))

    def energy(self, wavefunctions, positions,
               calculation=None) -> tuple[float, float, float]:
        """``calc_pol``: ``(en_el, el_pol, ion_pol)``, updating the branch on the way.

        Called once per SCF iteration, as QE does. The jump rule is QE's
        literally, including its scale at ``nspin = 1``: ``el_pol`` carries the
        spin factor and ``fc_pol`` does not, so it fires at half a turn of the
        average phase and corrects by half a quantum.
        """
        el = self.electronic_dipole(wavefunctions, calculation)
        ion = self.ionic_dipole(positions)
        if self.previous is not None:
            sca = (el - self.previous) / self.fc_pol
            if sca < -np.pi:
                self.accumulated += 2.0 * np.pi * self.fc_pol
            elif sca > np.pi:
                self.accumulated -= 2.0 * np.pi * self.fc_pol
        self.previous = el
        if not self.active:
            return 0.0, el, ion
        return -self.efield * (el + ion + self.accumulated), el, ion


def require_a_berry_field_regime(calculation) -> None:
    """Refuse, by name, every regime ``lelfield`` has not been built or measured in."""
    system = calculation.system
    reasons = []
    spiral = bool(getattr(calculation, "spiral", False))
    if calculation.augmentation is not None and (system.nspin != 1 or spiral):
        reasons.append(
            f"an ultrasoft or PAW dataset at nspin = {system.nspin}"
            + (" on a spin spiral" if spiral else "")
            + ": the cross augmentation q_ij(dk) e^{-i dk.tau} in the string "
            "overlap is measured against pw.x at nspin = 1 only")
    if system.nspin == 2:
        nint = lambda x: int(np.floor(float(x) + 0.5))
        up, down = nint(calculation.nelup), nint(calculation.neldw)
        if up != down:
            reasons.append(
                f"nspin = 2 with {up} and {down} occupied bands in the two channels: "
                "pw.x builds the Berry phase on NINT(nelec/2) bands in each channel "
                "whatever the channel holds (l_cal in c_phase_field and "
                "h_epsi_her_set reads f_inp only for occupations = 'from_input'), "
                "so there is no pw.x number for a magnet, and none is measured here")
    if system.nspin == 4 and system.lspinorb:
        reasons.append(
            "nspin = 4 with spin-orbit coupling: the spinor field is measured only "
            "without it (against nspin = 1, and against pw.x on a magnetized chain)")
    if system.occupations != "fixed":
        reasons.append(
            f"occupations = '{system.occupations}': a Berry phase belongs to a "
            "gapped manifold (pw.x refuses smearing with lelfield too)")
    if calculation.is_hubbard:
        reasons.append("DFT+U, never run with the field here")
    if getattr(calculation, "sawtooth", None) is not None:
        reasons.append("tefield at the same time")
    if getattr(calculation, "gamma_only", False):
        reasons.append("gamma-only storage (pw.x refuses it, input.f90)")
    gdir = system.berry_field[0]
    at = np.asarray(system.cell.at)
    if not np.allclose(np.delete(at[gdir], gdir), 0.0, atol=1e-8 * np.abs(at).max()):
        reasons.append(
            f"a_{gdir + 1} not along Cartesian axis {gdir + 1}: pw.x's ionic dipole "
            "takes tau's Cartesian component gdir and its electronic one is along "
            "a_gdir, and the two agree only when those are the same direction")
    if reasons:
        raise NotImplementedError(
            "lelfield is not implemented for " + "; ".join(reasons))
