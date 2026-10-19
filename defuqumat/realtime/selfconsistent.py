"""The propagation with the Hartree and exchange-correlation potentials updated in time.

The time-dependent Kohn-Sham equation of :mod:`defuqumat.realtime.propagate`,
with the local potential rebuilt from the density at every step:

    i d|u_nk>/dt = [T(k + kappa) + V_NL(k + kappa) + vltot + v(t)] |u_nk>,
    v(t) = v_scf + U[rho(t)] - U[rho0],     rho(t) = sum_nk w |u_nk(r)|^2,

with ``U`` the Hartree potential (``potential = 'hartree'``, local fields with
the exchange-correlation potential frozen, the Dyson route's ``rpa``) or the
Hartree plus exchange-correlation potential of the ground state's functional
(``'hxc'``, the adiabatic functional: the ALDA for an LDA run, the Dyson
route's ``alda``). ``rho0`` is the density of the propagated states at the
start (``notes/HARMONICS-NEXT.md``, "Updating the potential in time").

**The difference form is required, not a convenience.** The states come from a
fixed-density solve at the ground state's density on the k-set the run
propagates, and their own density is that density only on the ground state's
grid and to its ``conv_thr``: measured in review on two-atom silicon at
``conv_thr = 1e-10``, ``max |V[rho0] - v_scf|`` is 1.2e-6 Ry on the SCF's own
4x4x4 grid and 1.0e-2 Ry on a 6x6x6 one. Written as ``v(t) = V[rho(t)]`` the
start would not be a fixed point and a run with no field would evolve; as the
difference it is one by construction, and the energy the dynamics conserves is

    E = E_kinNL(kappa) + int (vltot + v_scf - U[rho0]) rho + E_U[rho],

whose derivative in ``rho`` is ``vltot + v(t)``, with ``E_U`` the Hartree
energy or the Hartree and exchange-correlation energies (``E_xc`` of
``rho + rho_core``). With ``U = 0`` it is the band energy of the frozen route.

**The applied field is the whole macroscopic field.** The Hartree potential
has no ``G = 0`` term (``potential.hartree``), so no induced macroscopic field
is fed back, the current over the applied field is the macroscopic
conductivity, and ``1 + 4 pi i sigma / z`` from a kick is ``eps_M`` with local
fields, ``1/[eps^-1]_00``, Elk's ``tddft`` with ``tafindt`` off. The ``G = 0``
part of the exchange-correlation difference is uniform, since ``rho(t)`` keeps
its charge, and is a global phase.

**The loop is inverted.** At a frozen potential the k-points are independent
and the driver walks k outside and time inside. Here every step needs the
density of every k-point, so time is outside and the whole state set is
resident; the k-chunks of the calculation's ``k_batch`` are a ``lax.map``
inside each step, so what is in flight is one chunk's Taylor terms and its
projectors at ``k + kappa``, which are built inside that map (built for the
whole mesh at once they are 16 nk npwx nkb bytes a step). The chunks' arrays
with a k index are stacked on a leading axis and passed to the kept block as
an argument, never closed over (``propagate.py``, ``_Chunk``); the
calculation's grids and the potential's constants are the same for the whole
run and are closed over.

**The step is the midpoint rule with a predictor-corrector for the potential.**
The field is already at the midpoint; the potential there is extrapolated,
``3/2 v(t) - 1/2 v(t - dt)``, the states stepped, the density and potential
rebuilt at ``t + dt``, and with ``corrector = 1`` (the default) the step taken
again from ``t`` at ``(v(t) + v(t + dt))/2``. Both are second order in ``dt``;
the extrapolation misses ``v(t + dt/2)`` by ``-(3/8) dt^2 v''`` and the corrector
by ``+(1/8) dt^2 v''``, so the corrector buys a factor of three in the constant
for twice the cost of a step. Elk's ``tddft.f90`` steps with the potential of
``t``, which is first order.

**The density is symmetrised with the field's little group and nothing else.**
On the wedge of that group (:func:`~defuqumat.workflows.realtime.field_little_group`)
each k-point stands for its star, since ``S kappa = kappa`` makes
``u_Sk(t) = S u_k(t)``, and the whole mesh's density is the group average of
the wedge's. The calculation's own symmetrisation is the crystal's full group,
and the first-order density of a field along ``x`` is odd under silicon's
inversion, so averaging it over that group removes the whole local-field
effect and ``'hxc'`` would read exactly as ``'frozen'``; which is why
:meth:`~defuqumat.scf.driver.Calculation.density` and ``finish_density`` are not
called here.

**Memory.** The states of the whole mesh are resident, ``16 nk nbnd npwx``
bytes, twice inside a step (the states at ``t`` and the stepped ones), and
``2^n`` times that at order ``n`` of :func:`propagate_orders_self_consistent`;
beside them, every chunk's k-dependent arrays (kinetic energies, FFT indices
and masks; the projectors at ``kappa = 0`` are dropped, since every step builds
its own), three dense potentials and the density.
"""

from __future__ import annotations

import dataclasses
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.basis.fft import g_to_r, r_to_g
from defuqumat.basis.interpolate import to_dense
from defuqumat.batching import map_k
from defuqumat.eager import compiled_function
from defuqumat.hamiltonian.noncollinear import SpinorHamiltonian
from defuqumat.scf.density import _becsum
from defuqumat.scf.fields import FEEDBACK

__all__ = ["POTENTIALS", "require_a_potential_mode", "propagate_self_consistent",
           "propagate_orders_self_consistent"]

#: The potential modes, by name: frozen at the ground state's, the Hartree
#: potential updated, the Hartree and exchange-correlation potentials updated.
POTENTIALS = ("frozen", "hartree", "hxc")


#: The callers of :func:`require_a_potential_mode`: the propagation of
#: :func:`propagate_self_consistent`, the perturbative orders of
#: :func:`propagate_orders_self_consistent`, and the frequency-domain hierarchy's
#: first order (:func:`~defuqumat.realtime.hierarchy.hierarchy_linear_self_consistent`).
ROUTES = ("propagation", "orders", "hierarchy")


def require_a_potential_mode(calculation, potential: str, symmetrise, density_symmetry,
                             route: str = "propagation"):
    """Refuse, by name, a mode or a regime the update does not cover.

    ``route`` names the caller (:data:`ROUTES`): an ultrasoft or PAW dataset
    runs in the propagation alone (``notes/incompat/I98.md``), and there only
    for a collinear ``nspin = 1`` run.
    """
    if potential not in POTENTIALS:
        raise ValueError(f"potential must be one of {POTENTIALS}, not {potential!r}")
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, not {route!r}")
    if potential == "frozen":
        return
    if getattr(calculation, "augmentation", None) is not None:
        if route == "orders":
            raise NotImplementedError(
                "the perturbative orders with the potential updated in time are not "
                "implemented for an ultrasoft or PAW dataset: the nested jvp would "
                "have to carry the kick's jump of the augmented states and the "
                "augmentation charge at k + kappa(t) through every level. The "
                "propagation (propagate with a small finite kick) runs, and "
                "potential = 'frozen' runs here")
        if route == "hierarchy":
            raise NotImplementedError(
                "the frequency-domain hierarchy with the potential updated is not "
                "implemented for an ultrasoft or PAW dataset: its induced density has "
                "no augmentation charge from the response of becsum and its kernel "
                "no response of D through newd or the one-centre terms. The "
                "real-time propagation with the potential updated runs")
        if calculation.noncolin:
            raise NotImplementedError(
                "the potential updated in time is not implemented for an ultrasoft or "
                "PAW dataset on spinors: becsum(t) would need the spinor projections "
                "at k + kappa(t) and D(t) the spin-orbit sandwich of newd at every "
                "step. potential = 'frozen' runs")
        if calculation.nspin != 1:
            raise NotImplementedError(
                "the potential updated in time is not implemented for an ultrasoft or "
                "PAW dataset at nspin = 2: the per-channel becsum(t) and D(t) are "
                "written but not measured. potential = 'frozen' runs")
        if potential == "hartree" and calculation.is_paw:
            raise NotImplementedError(
                "potential = 'hartree' is not implemented for a PAW dataset: the "
                "one-centre energy is Hartree and exchange-correlation in one, and "
                "keeping its Hartree part alone in time is not written. "
                "potential = 'hxc' runs")
    field = getattr(calculation, "magnetic_field", None)
    if field is not None and field.constraint != "none" and field.constraint not in FEEDBACK:
        # A field put in by hand, a reducebf-scaled one and a feedback (fsm)
        # field all have an energy linear in m, so their potential does not move
        # with rho(t): it sits frozen inside v_scf and ``U`` leaves it out
        # (``_Run``, ``field_scale = 0``). A penalty ``lambda (m - m0)^2`` is
        # quadratic, and updating it in time would put its curvature, a
        # restoring force on the moment whose stiffness is a convergence
        # parameter, into the response. Whether the dynamics should carry it is
        # a choice not taken here (notes/incompat/I64.md).
        raise NotImplementedError(
            f"the potential updated in time with constrained_magnetization = "
            f"{field.constraint!r} is not implemented: the penalty lambda (m - m0)^2 "
            "is quadratic in the moment, so updating it would put its stiffness, a "
            "convergence parameter, into the response as a restoring force. "
            "potential = 'frozen' runs, with the converged constraint field frozen")
    if potential == "hxc" and calculation.functional.is_meta:
        raise NotImplementedError(
            "the exchange-correlation potential updated in time is not implemented "
            "for a meta-GGA: a potential-only one (tb09, bj06) has no energy for the "
            "dynamics to conserve, and either kind needs tau(t) propagated beside "
            "rho(t). potential = 'hartree' keeps the ground state's v_xc and runs")
    if symmetrise is not None and density_symmetry is None:
        raise ValueError(
            "a k-set reduced by symmetry needs the group its density is completed "
            "with when the potential is updated: pass density_symmetry, the field's "
            "little group (workflows.realtime.field_symmetries). The crystal's own "
            "group would average the response away")


class _KPart(eqx.Module):
    """The arrays of one k-chunk that carry a k index, stacked over chunks.

    A :class:`~defuqumat.realtime.propagate._Chunk` without the table, the
    positions and the local potential, which are the same for every chunk and
    are supplied at the step; the template's potential is a scalar placeholder
    so that the stack does not hold one copy of the grid per chunk.
    """

    k0: jnp.ndarray
    gcart: jnp.ndarray
    mask: jnp.ndarray
    core: object
    template: object
    nk: int = eqx.field(static=True)
    channel: int = eqx.field(static=True, default=0)
    #: the augmentation's dipole, for the position term of an augmented
    #: dataset's equation of motion and current; ``None`` without one
    dipole: jnp.ndarray | None = None

    @classmethod
    def of(cls, chunk):
        """The chunk's k-indexed arrays, with what every step rebuilds left out.

        The step builds the projectors and their columns at ``k + kappa`` from
        the table (:meth:`~defuqumat.realtime.propagate._Chunk.moved`), so the ones
        at ``kappa = 0`` the chunk was built with, ``vkb`` and the real columns
        with their ``k + G``, are never read again; kept for the whole mesh they
        were ``16 nk npwx nkb + 8 nk npwx (ncs + 3)`` bytes resident beside states
        that are ``16 nk npwx nbnd``, several times their size.
        """
        placeholder = jnp.zeros((), dtype=chunk.template.potential.dtype)
        projectors = chunk.template.projectors
        projectors = dataclasses.replace(
            projectors, stored=jnp.zeros((chunk.nk, 1, projectors.dij.shape[0]),
                                         dtype=projectors.dtype))
        template = dataclasses.replace(chunk.template, potential=placeholder,
                                       potential_wave=placeholder, projectors=projectors)
        core = chunk.core
        core = eqx.tree_at(lambda c: (c.columns, c.kg), core,
                           (jnp.zeros((chunk.nk, 1, core.columns.shape[-1]),
                                      dtype=core.columns.dtype),
                            jnp.zeros((chunk.nk, 1, 3), dtype=core.kg.dtype)))
        return cls(k0=chunk.k0, gcart=chunk.gcart, mask=chunk.mask, core=core,
                   template=template, nk=chunk.nk, channel=chunk.channel,
                   dipole=chunk.dipole)

    def chunk(self, table, positions, terms=None, deeq=None):
        """The chunk at the local terms ``terms`` (``LocalTerms``), or at the template's.

        For an augmented dataset ``D`` is taken from ``terms`` with the
        potential, ``newd``'s ``D(t)`` from ``v(t)``; ``deeq`` overrides it,
        ``(nspin, nkb, nkb)``, which is how the conserved energy's band part
        is taken with the bare ``D``. A norm-conserving template keeps the
        file's ``D`` whatever is passed.
        """
        from defuqumat.realtime.propagate import _Chunk

        template = self.template
        if terms is not None:
            # a collinear channel's own potential; a spinor's one, whose
            # components are the charge's and the magnetization's
            index = self.channel if len(terms.potentials) > 1 else 0
            template = dataclasses.replace(template, potential=terms.potentials[index],
                                           potential_wave=terms.waves[index])
            if deeq is None:
                deeq = terms.deeq
        if deeq is not None and template.has_overlap:
            template = dataclasses.replace(
                template, deeq=deeq if isinstance(template, SpinorHamiltonian)
                else deeq[self.channel])
        return _Chunk(k0=self.k0, gcart=self.gcart, mask=self.mask, core=self.core,
                      template=template, table=table, positions=positions, nk=self.nk,
                      channel=self.channel, dipole=self.dipole)




def _plain_norms(states):
    """``<u|u>`` of host states ``(nk, nbnd, ndim)``, the norm of a norm-conserving run's states."""
    states = np.asarray(states)
    return np.real(np.einsum("kng,kng->kn", np.conj(states), states))


def _stack(trees):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *trees)


def _extrapolated(now, previous):
    """``3/2 now - 1/2 previous`` for every array of the potential, ``None`` left alone."""
    return jax.tree_util.tree_map(lambda a, b: 1.5 * a - 0.5 * b, now, previous)


def _midpoint(now, later):
    """``(now + later)/2`` for every array of the potential, ``None`` left alone."""
    return jax.tree_util.tree_map(lambda a, b: 0.5 * (a + b), now, later)


class _Run:
    """What one self-consistent run needs, built once: the stacked chunks and the update.

    One entry per channel in ``parts``, ``w`` and ``psi``: a collinear
    ``nspin = 2`` run has two, each with its own number of carried bands, and
    every other run one. ``density(parts, table, positions, w, psi, kappa)``
    takes the stacked states, one ``(nchunks, kb, nbnd, ndim)`` per channel,
    and returns ``(rho, becsum)``: the density on the dense grid,
    ``(nspin_mag, n1, n2, n3)`` -- the two channels' ``(up, down)``, or a
    spinor's ``(n, m_x, m_y, m_z)`` -- symmetrised with the little group's maps
    when there are any, and for an augmented dataset the augmentation charge
    from ``becsum`` at ``k + kappa`` in it (``()`` otherwise). The local
    potential is the pair ``(v, ddd)``, ``ddd`` PAW's one-centre ``D`` (``None``
    without PAW); ``potential(rho, becsum)`` returns ``((v(t), ddd(t)),
    E_U)``; ``local_energy(rho, becsum)`` is the part of the conserved energy
    that is not the band's kinetic and nonlocal energy at :attr:`band_deeq`.

    **An augmented dataset** (``notes/incompat/I98.md``). The density is

        rho(t) = sum_nk w |u_nk|^2 + sum_ij Q_ij(r - R) becsum_ij(t),
        becsum_ij(t) = sum_nk w <u_nk|beta_i(k + kappa)> <beta_j(k + kappa)|u_nk>,

    the projectors at ``k + kappa`` because the gauge factor that takes every
    operator to ``k + kappa`` cancels between the two projectors of one atom
    as it does in ``V_NL``. ``D(t)`` is ``newd`` of ``vltot + v(t)``, which
    :meth:`~defuqumat.scf.driver.Calculation.local_terms` builds at every step,
    and PAW's one-centre ``D`` is in the difference form too,
    ``ddd(t) = ddd_scf + ddd[becsum(t)] - ddd[becsum0]``, for the reason the
    grid potential is: the one-centre ``D`` of the propagated states'
    ``becsum`` is the ground state's only to its ``conv_thr`` and on its own
    k-set. The conserved energy is

        E = sum w <u|T + V_NL[D_bare + ddd_scf - ddd0]|u>
            + int (vltot + v_scf - U[rho0]) rho + E_U[rho] + E_1[becsum],

    whose derivative in ``becsum`` is ``newd(vltot + v(t)) + ddd(t)``, with
    ``E_1`` the one-centre energy; the band part is taken at
    :attr:`band_deeq`, the bare ``D`` (plus ``ddd_scf - ddd0`` for PAW), since
    at ``D(t)`` it would count the augmentation's share twice.
    """

    def __init__(self, calculation, setups, pairs, v_scf, potential: str,
                 density_symmetry, ddd_paw=None, kappa0=None):
        from defuqumat.realtime.propagate import _chunk_weights
        from defuqumat.scf.driver import _symmetrize, _symmetrize_noncollinear
        from defuqumat.scf.potential import as_potential_components, hartree
        from defuqumat.system.symmetry import cartesian_rotations, symmetry_maps

        real = setups[0].real
        self.calculation = calculation
        self.mode = potential
        self.chunks = setups[0].chunks
        self.table = setups[0].table
        self.positions = jnp.asarray(calculation.system.structure.positions)
        self.augmented = getattr(calculation, "augmentation", None) is not None
        self.paw = self.augmented and bool(calculation.is_paw)
        parts, ws, psis, flats = [], [], [], []
        for setup, (states, weights) in zip(setups, pairs):
            built = [setup.chunk(i, rows) for i, (rows, _) in enumerate(self.chunks)]
            parts.append(_stack([_KPart.of(c) for c in built]))
            ws.append(_stack([_chunk_weights(weights, rows, live, real)
                              for rows, live in self.chunks]))
            psis.append(_stack([jnp.asarray(states[rows]) for rows, _ in self.chunks]))
            flats.append(jnp.asarray(weights, dtype=real))
        self.parts, self.w, self.psi = tuple(parts), tuple(ws), tuple(psis)
        kb = len(self.chunks[0][0])
        self.order = np.asarray([i * kb + j for i, (_, live) in enumerate(self.chunks)
                                 for j in range(live)])
        self.nk = pairs[0][0].shape[0]
        self.weights = tuple(flats)

        dense, smooth = calculation.basis.dense, calculation.basis.smooth
        cell = calculation.system.cell
        nspin_mag = calculation.nspin_mag
        maps = (None if density_symmetry is None
                else symmetry_maps(dense, density_symmetry))
        rotations = None
        if maps is not None and nspin_mag == 4:
            # an axial vector's rotation, ``det(R) R``; the field's little group
            # has no operation that needs time reversal (``field_symmetries``)
            cartesian = np.asarray(cartesian_rotations(cell, density_symmetry))
            rotations = jnp.asarray(np.sign(np.linalg.det(cartesian))[:, None, None] * cartesian)
        order, flat_weights = self.order, self.weights
        augmented = self.augmented
        species_channels = calculation.species_channels if augmented else ()
        becsum_symmetry = None
        if augmented and density_symmetry is not None:
            # the little group's average of becsum, which PAW's one-centre terms
            # read per atom; the crystal's own (``Calculation._becsum_symmetry``)
            # would average the response away as the crystal's sym_rho would
            from defuqumat.paw.symmetry import build_becsum_symmetry

            becsum_symmetry = build_becsum_symmetry(
                calculation.pseudos, calculation.system.structure, cell, density_symmetry,
                nspin_mag)

        def becsum_of(parts, table, positions, w, psi, kappa):
            """``becsum`` at ``k + kappa``, per species ``(nspin, nat, nh, nh)``."""
            channels = []
            for part_set, wc, states in zip(parts, w, psi):
                def one(args):
                    part, wk, p = args
                    _, projectors = part.chunk(table, positions).moved(kappa)
                    return _becsum(p[None], projectors.vkb, wk[None], species_channels, None)
                values = jax.lax.map(one, (part_set, wc, states))
                channels.append(jax.tree_util.tree_map(lambda a: jnp.sum(a, axis=0), values))
            values = tuple(
                None if channels[0][s] is None
                else jnp.concatenate([c[s] for c in channels])
                for s in range(len(species_channels)))
            if becsum_symmetry is not None:
                values = becsum_symmetry.apply(values)
            return values

        def density(parts, table, positions, w, psi, kappa):
            channels = []
            for states, weights in zip(psi, flat_weights):
                whole = states.reshape((-1,) + states.shape[2:])[order]
                channels.append(calculation.smooth_density(whole[None], weights[None]))
            rho = to_dense(jnp.concatenate(channels), smooth, dense)
            becsum_ = ()
            if augmented:
                becsum_ = becsum_of(parts, table, positions, w, psi, kappa)
                rho = calculation.augmented(rho, becsum_)
            if maps is not None:
                if rotations is not None:
                    rho = _symmetrize_noncollinear(rho, dense.fft_index, dense.grid, maps,
                                                   rotations)
                else:
                    rho = _symmetrize(rho, dense.fft_index, dense.grid, maps)
            return rho, becsum_

        if potential == "hxc":
            # ``field_scale = 0``: no applied field in ``U``. Its energy is
            # linear in m, so its potential is the same at rho(t) as at rho0 and
            # stays in ``v_scf`` at the converged strength; left in ``U`` at
            # the input's it would cancel from ``v`` but not from the energy,
            # whose ``E_U`` is ``ehart + etxc``. A penalty constraint, the one
            # term that does move, is refused by ``require_a_potential_mode``.
            def field(rho):
                result = calculation.potential(rho, 0.0)
                return result.v_scf, result.ehart + result.etxc
        else:
            def field(rho):
                total = rho[0] if calculation.noncolin or rho.shape[0] == 1 else rho.sum(axis=0)
                vg, energy = hartree(r_to_g(total, dense.fft_index), dense, cell)
                v = jnp.real(g_to_r(vg, dense.fft_index, dense.grid))
                return as_potential_components(v, nspin_mag), energy

        self.density = density
        fixed = (self.parts, self.table, self.positions, self.w)
        if kappa0 is None:
            kappa0 = jnp.zeros(3, dtype=real)
        kappa0 = jnp.asarray(kappa0, dtype=real)
        if augmented:
            # the step in kappa the run starts with, from the ground state's
            # zero, crossed as the frozen route crosses it (``_Chunk.jump``)
            zero = jnp.zeros(3, dtype=real)
            self.psi = compiled_function(_jumped_all, *fixed[:3], self.psi, zero, kappa0)(
                *fixed[:3], self.psi, zero, kappa0)
        self.rho0, self.becsum0 = compiled_function(density, *fixed, self.psi, kappa0)(
            *fixed, self.psi, kappa0)
        u0, _ = field(self.rho0)
        v_scf = jnp.asarray(v_scf)
        self.v_scf = v_scf
        paw = self.paw
        ddd_scf = d0 = None
        if paw:
            if ddd_paw is None:
                raise ValueError(
                    "a PAW ground state propagated with the potential updated needs its "
                    "one-centre D, ddd_paw = Calculation.onecenter(becsum)[1]")
            ddd_scf = jnp.asarray(ddd_paw)
            _, d0 = calculation.onecenter(self.becsum0)
        self.ddd_scf = ddd_scf
        #: the potential at the start, ``(v_scf, ddd_scf)``
        self.p0 = (v_scf, ddd_scf)
        constant = as_potential_components(jnp.asarray(calculation.vltot), nspin_mag) + (v_scf - u0)
        volume = float(cell.volume)

        def potential_of(rho, becsum_):
            u, energy = field(rho)
            v = v_scf + u - u0
            if not paw:
                return (v, None), energy
            e1, d1 = calculation.onecenter(becsum_)
            return (v, ddd_scf + d1 - d0), energy + e1

        def local_energy(rho, becsum_):
            _, energy = potential_of(rho, becsum_)
            return volume / rho[0].size * jnp.sum(constant * rho) + energy

        def local_terms(p):
            return calculation.local_terms(p[0], p[1])

        self.potential = potential_of
        self.local_energy = local_energy
        self.local_terms = local_terms
        #: ``D`` for the band part of the conserved energy: the bare one, and
        #: for PAW ``ddd_scf - ddd0`` beside it; ``None`` (the file's) without
        #: an augmentation
        self.band_deeq = None
        if augmented:
            self.band_deeq = calculation.coefficients(
                jnp.zeros_like(v_scf), None if not paw else ddd_scf - d0)

    def flat(self, psi):
        """The stacked states of each channel as ``(nk, nbnd, ndim)`` on the host, the padding dropped."""
        out = []
        for states in psi:
            states = np.asarray(states)
            out.append(states.reshape((-1,) + states.shape[2:])[self.order])
        return out


def _advance(step_fn, centres, parts, table, positions, psi, terms, kappa, rate, step):
    """Every channel's and chunk's states one step on, at the local terms ``terms`` and ``kappa``.

    An augmented chunk steps with its generalised generator,
    ``S^-1 [H - kappadot . X]`` (:meth:`~defuqumat.realtime.propagate._Chunk.generator`),
    which reads ``rate``; a norm-conserving one with ``H``.
    """
    moved = []
    for part_set, states, centre in zip(parts, psi, centres):
        def one(args, centre=centre):
            part, p = args
            chunk = part.chunk(table, positions, terms)
            if chunk.augmented:
                return step_fn(chunk.generator(kappa, rate), p, step, centre)
            ham = chunk.hamiltonian(kappa)
            return map_k(lambda ik: step_fn(lambda v: ham.apply(v, ik), p[ik], step, centre),
                         jnp.arange(part.nk), batch=None)
        moved.append(jax.lax.map(one, (part_set, states)))
    return tuple(moved)


def _slope(parts, table, positions, w, psi, kappa, terms=None):
    """The current up to ``-1/Omega``, summed over channels and chunks, Ry bohr.

    ``terms`` are the local terms the current is taken at, which an augmented
    dataset's reads (its ``D`` and the ``H`` of its position term); ``None``
    keeps the template's, all a norm-conserving current needs.
    """
    total = 0.0
    for part_set, wc, states in zip(parts, w, psi):
        def one(args):
            part, wk, p = args
            return part.chunk(table, positions, terms).current(kappa, p, wk)
        total = total + jnp.sum(jax.lax.map(one, (part_set, wc, states)), axis=0)
    return total


def _band(parts, table, positions, w, psi, kappa, deeq=None):
    total = 0.0
    for part_set, wc, states in zip(parts, w, psi):
        def one(args):
            part, wk, p = args
            return part.chunk(table, positions, deeq=deeq).kappa_energy(kappa, p, wk)
        total = total + jnp.sum(jax.lax.map(one, (part_set, wc, states)))
    return total


def _chunkwise(method):
    """``f(parts, table, positions, psi, kappa)``: ``method`` of every chunk, one array per channel."""
    def apply(parts, table, positions, psi, kappa):
        out = []
        for part_set, states in zip(parts, psi):
            def one(args):
                part, p = args
                return getattr(part.chunk(table, positions), method)(kappa, p)
            out.append(jax.lax.map(one, (part_set, states)))
        return tuple(out)
    return apply


_norms_all = _chunkwise("norms")
_overlapped_all = _chunkwise("overlap")


def _jumped_all(parts, table, positions, psi, kappa_from, kappa_to):
    """Every chunk's states across a step of ``kappa`` (:meth:`~defuqumat.realtime.propagate._Chunk.jump`)."""
    out = []
    for part_set, states in zip(parts, psi):
        def one(args):
            part, p = args
            return part.chunk(table, positions).jump(kappa_from, kappa_to, p)
        out.append(jax.lax.map(one, (part_set, states)))
    return tuple(out)


def _block_function(run: _Run, step_fn, centres, corrector: int):
    """``block(parts, table, positions, w, carry, kmid, kend, rates, steps) -> (carry, gradients)``.

    ``carry`` is ``(psi, p_prev, p_now)``: the stacked states at ``t`` and the
    potential ``(v, ddd)`` at ``t - dt`` and ``t``. ``rates`` is ``dkappa/dt``
    at the midpoints, read by an augmented dataset's step alone. A step of
    length zero leaves the states and ``p_now`` where they are, which is how a
    block is padded.
    """
    density, potential_of, local_terms = run.density, run.potential, run.local_terms
    augmented = run.augmented
    # constants of the program, not literals in it, one per channel
    centres = tuple(jnp.asarray(c) for c in centres)

    def block(parts, table, positions, w, carry, kmid, kend, rates, steps):
        def body(state, x):
            psi, p_prev, p_now = state
            k_mid, k_end, k_rate, step = x

            def advance(p_mid):
                return _advance(step_fn, centres, parts, table, positions, psi,
                                local_terms(p_mid), k_mid, k_rate, step)

            def updated(moved):
                rho, becsum_ = density(parts, table, positions, w, moved, k_end)
                return potential_of(rho, becsum_)[0]

            moved = advance(_extrapolated(p_now, p_prev))
            p_next = updated(moved)
            for _ in range(corrector):
                moved = advance(_midpoint(p_now, p_next))
                p_next = updated(moved)
            gradient = _slope(parts, table, positions, w, moved, k_end,
                              local_terms(p_next) if augmented else None)
            return (moved, p_now, p_next), gradient
        return jax.lax.scan(body, carry, (kmid, kend, rates, steps))
    return block


def _energy_function(run: _Run):
    density, local_energy, band_deeq = run.density, run.local_energy, run.band_deeq

    def energy(parts, table, positions, w, psi, kappa):
        rho, becsum_ = density(parts, table, positions, w, psi, kappa)
        return (_band(parts, table, positions, w, psi, kappa, band_deeq)
                + local_energy(rho, becsum_))
    return energy


def _slope_function(run: _Run):
    """The current at the start's potential, ``(v_scf, ddd_scf)``."""
    terms = run.local_terms(run.p0) if run.augmented else None

    def slope(parts, table, positions, w, psi, kappa):
        return _slope(parts, table, positions, w, psi, kappa, terms)
    return slope


def _setups(calculation, states, weights, v_scf, kappa_max, dt, propagator, k_batch, kcart,
            ddd_paw=None):
    """One :func:`~defuqumat.realtime.propagate._prepare` per channel, and the channels' states."""
    from defuqumat.realtime.propagate import _prepare, channel_arrays

    pairs = channel_arrays(calculation, states, weights)
    setups = [_prepare(calculation, own, carried, v_scf, kappa_max, dt, propagator,
                       k_batch, kcart, channel=channel, ddd_paw=ddd_paw)
              for channel, (own, carried) in enumerate(pairs)]
    return pairs, setups


def propagate_self_consistent(calculation, states, weights, v_scf, pulse, *, dt: float,
                              duration=None, start=None, propagator: str = "taylor4",
                              k_batch="default", block_steps: int = 400, kcart=None,
                              symmetrise=None, density_symmetry=None,
                              potential: str = "hxc", corrector: int = 1, ddd_paw=None):
    """:func:`~defuqumat.realtime.propagate.propagate` with the potential updated in time.

    ``potential`` is ``'hartree'`` or ``'hxc'`` (module docstring);
    ``density_symmetry`` the :class:`~defuqumat.system.symmetry.Symmetries` of the
    field's little group when the k-set is its wedge, which ``symmetrise`` (its
    cartesian rotations, for the current) then says; ``corrector`` how many
    times the midpoint potential is corrected after its extrapolation;
    ``ddd_paw`` PAW's one-centre ``D`` of the ground state, required for a PAW
    dataset (:class:`_Run` has the augmented case). The energy on the result
    is the conserved functional of the module docstring. The states are
    :func:`~defuqumat.realtime.propagate.propagate`'s, one block per channel of
    a collinear run, which the density couples.
    """
    from defuqumat.realtime.propagate import (
        RealTimeResult, _check_growth, _padded, _padded_grid, _reach, _warn_damping,
        kappa_rate, kick_jumps, time_grid)

    require_a_potential_mode(calculation, potential, symmetrise, density_symmetry)
    times, kappa_t, kappa_mid, efield = time_grid(pulse, dt, duration, start)
    nsteps = len(times) - 1
    augmented = getattr(calculation, "augmentation", None) is not None
    if augmented and kick_jumps(calculation, pulse, times) is not None:
        raise NotImplementedError(
            "a kick inside the run is not implemented for an ultrasoft or PAW dataset "
            "with the potential updated in time: the jump of the augmented states would "
            "have to be crossed inside the step with the density and D(t) rebuilt after "
            "it. Start the run at the kick (start = the kick's time)")
    kappa_max = float(np.max(np.linalg.norm(np.concatenate([kappa_t, kappa_mid]), axis=-1)))
    pairs, setups = _setups(calculation, states, weights, v_scf, kappa_max, dt, propagator,
                            k_batch, kcart, ddd_paw)
    for setup in setups:
        _warn_damping(setup, nsteps)
    real = setups[0].real
    kappa0 = jnp.asarray(kappa_t[0], dtype=real)
    run = _Run(calculation, setups, pairs, v_scf, potential, density_symmetry,
               ddd_paw=ddd_paw, kappa0=kappa0)
    kappa_mid_p, kappa_end_p, dts, nblocks = _padded_grid(
        kappa_mid, kappa_t[1:], nsteps, block_steps, setups[0].dt_ry)
    rate_p = _padded(kappa_rate(kappa_t, setups[0].dt_ry), nsteps, block_steps)

    fixed = (run.parts, run.table, run.positions, run.w)
    carry = (run.psi, run.p0, run.p0)
    block = compiled_function(
        _block_function(run, setups[0].step_fn, [s.centre for s in setups], int(corrector)),
        *fixed, carry,
        jnp.asarray(kappa_mid_p[:block_steps], dtype=real),
        jnp.asarray(kappa_end_p[:block_steps], dtype=real),
        jnp.asarray(rate_p[:block_steps], dtype=real),
        jnp.asarray(dts[:block_steps], dtype=real))
    measure = compiled_function(_energy_function(run), *fixed, run.psi, kappa0)
    slope = compiled_function(_slope_function(run), *fixed, run.psi, kappa0)
    norms_of = overlapped = None
    if augmented:
        norms_of = compiled_function(_norms_all, *fixed[:3], run.psi, kappa0)
        overlapped = compiled_function(_overlapped_all, *fixed[:3], run.psi, kappa0)

    current = np.zeros((nsteps + 1, 3))
    energy_index = [0] + [min((b + 1) * block_steps, nsteps) for b in range(nblocks)]
    energy = np.zeros(len(energy_index))
    current[0] = np.asarray(slope(*fixed, run.psi, kappa0))
    energy[0] = float(measure(*fixed, run.psi, kappa0))
    for b in range(nblocks):
        sl = slice(b * block_steps, (b + 1) * block_steps)
        carry, gradients = block(*fixed, carry, jnp.asarray(kappa_mid_p[sl], dtype=real),
                                 jnp.asarray(kappa_end_p[sl], dtype=real),
                                 jnp.asarray(rate_p[sl], dtype=real),
                                 jnp.asarray(dts[sl], dtype=real))
        stop = min((b + 1) * block_steps, nsteps)
        current[b * block_steps + 1:stop + 1] = np.asarray(gradients)[:stop - b * block_steps]
        kappa_stop = jnp.asarray(kappa_t[stop], dtype=real)
        energy[b + 1] = float(measure(*fixed, carry[0], kappa_stop))
        if augmented:
            # ``<u|S(k + kappa)|u>``, the norm the generalised equation keeps
            norms = [_check_growth(n, run.nk, f"after step {stop}")
                     for n in run.flat(norms_of(*fixed[:3], carry[0], kappa_stop))]
        else:
            norms = [_check_growth(_plain_norms(f), run.nk, f"after step {stop}")
                     for f in run.flat(carry[0])]

    final = run.flat(carry[0] if overlapped is None else overlapped(
        *fixed[:3], carry[0], jnp.asarray(kappa_t[-1], dtype=real)))
    excited = 0.0
    for (own, carried), f in zip(pairs, final):
        overlap = np.einsum("kmg,kng->kmn", np.conj(own), f)
        kept = np.sum(np.abs(overlap) ** 2, axis=1)
        excited += float(np.sum(carried * (1.0 - kept)))
    volume = setups[0].volume
    current = -current / (2.0 * volume)
    if symmetrise is not None:
        rotations = np.asarray(symmetrise, dtype=float)
        current = np.einsum("sab,tb->ta", rotations, current) / len(rotations)
    lower = min(s.lower for s in setups)
    upper = max(s.upper for s in setups)
    return RealTimeResult(
        times=times, kappa=kappa_t, efield=efield, current=current,
        energy_times=times[np.asarray(energy_index)], energy=0.5 * energy,
        norm_drift=max(float(np.abs(n - 1.0).max()) for n in norms), excited=excited,
        volume=volume, nelec=float(sum(np.sum(c) for _, c in pairs)), dt=float(dt),
        effective_cutoff=min(s.effective_cutoff for s in setups),
        propagator=propagator, pulse=pulse, spectrum=(lower, upper),
        step_radius=max(s.dt_ry * _reach(s.lower, s.upper, s.centre) for s in setups),
        symmetry_operations=1 if symmetrise is None else len(symmetrise),
        extras={"potential": potential, "corrector": int(corrector),
                "potential_change": float(np.abs(np.asarray(carry[2][0] - run.v_scf)).max()),
                "augmented": bool(augmented)},
    )


def propagate_orders_self_consistent(calculation, states, weights, v_scf, shape, *,
                                     dt: float, order: int = 3, duration=None, start=None,
                                     propagator: str = "taylor4", k_batch="default",
                                     block_steps: int = 400, kcart=None, symmetrise=None,
                                     density_symmetry=None, potential: str = "hxc",
                                     corrector: int = 1):
    """:func:`~defuqumat.realtime.orders.propagate_orders` with the potential updated in time.

    The tower of nested ``jvp`` carries the states and the two potentials of
    the step, so the derivative of the density goes through the potential as
    it does in the Sternheimer stack; order ``n`` carries ``2^n`` copies of the
    whole mesh's states. Norm-conserving datasets only
    (:func:`require_a_potential_mode`, ``route = 'orders'``).
    """
    from defuqumat.realtime.orders import OrdersResult, _derivative, _lift, _tower
    from defuqumat.realtime.propagate import (
        _check_growth, _padded, _padded_grid, _warn_damping, kappa_rate, time_grid)

    require_a_potential_mode(calculation, potential, symmetrise, density_symmetry,
                             route="orders")
    times, a_t, a_mid, _ = time_grid(shape, dt, duration, start)
    nsteps = len(times) - 1
    pairs, setups = _setups(calculation, states, weights, v_scf, 0.0, dt, propagator,
                            k_batch, kcart)
    for setup in setups:
        _warn_damping(setup, nsteps)
    real = setups[0].real
    run = _Run(calculation, setups, pairs, v_scf, potential, density_symmetry)
    a_mid_p, a_end_p, dts, nblocks = _padded_grid(a_mid, a_t[1:], nsteps, block_steps,
                                                  setups[0].dt_ry)
    rate_p = _padded(kappa_rate(a_t, setups[0].dt_ry), nsteps, block_steps)
    base = _block_function(run, setups[0].step_fn, [s.centre for s in setups], int(corrector))
    depth = int(order)

    def lifted(parts, table, positions, w, tower, lam, amid, aend, rates, steps):
        def f(carry, l):
            return base(parts, table, positions, w, carry, l * amid, l * aend, l * rates,
                        steps)
        return _lift(f, depth)(tower, lam)

    def initial(parts, table, positions, w, tower, lam, a0):
        def f(carry, l):
            return carry, _slope(parts, table, positions, w, carry[0], l * a0)
        return _lift(f, depth)(tower, lam)[1]

    fixed = (run.parts, run.table, run.positions, run.w)
    lam = jnp.zeros((), dtype=real)
    a0 = jnp.asarray(a_t[0], dtype=real)
    tower = _tower((run.psi, run.p0, run.p0), depth)
    run_block = compiled_function(
        lifted, *fixed, tower, lam, jnp.asarray(a_mid_p[:block_steps], dtype=real),
        jnp.asarray(a_end_p[:block_steps], dtype=real),
        jnp.asarray(rate_p[:block_steps], dtype=real),
        jnp.asarray(dts[:block_steps], dtype=real))
    start_run = compiled_function(initial, *fixed, tower, lam, a0)

    raw = np.zeros((depth + 1, nsteps + 1, 3))
    out = start_run(*fixed, tower, lam, a0)
    for k in range(depth + 1):
        raw[k, 0] = np.asarray(_derivative(out, k, depth))
    for b in range(nblocks):
        sl = slice(b * block_steps, (b + 1) * block_steps)
        tower, out = run_block(*fixed, tower, lam, jnp.asarray(a_mid_p[sl], dtype=real),
                               jnp.asarray(a_end_p[sl], dtype=real),
                               jnp.asarray(rate_p[sl], dtype=real),
                               jnp.asarray(dts[sl], dtype=real))
        stop = min((b + 1) * block_steps, nsteps)
        for k in range(depth + 1):
            raw[k, b * block_steps + 1:stop + 1] = np.asarray(
                _derivative(out, k, depth))[:stop - b * block_steps]
        norms = [_check_growth(_plain_norms(f), run.nk, f"after step {stop}")
                 for f in run.flat(_derivative(tower, 0, depth)[0])]

    factorials = np.asarray([math.factorial(k) for k in range(depth + 1)], dtype=float)
    currents = -raw / (2.0 * setups[0].volume) / factorials[:, None, None]
    if symmetrise is not None:
        rotations = np.asarray(symmetrise, dtype=float)
        currents = np.einsum("sab,ntb->nta", rotations, currents) / len(rotations)
    return OrdersResult(times=times, currents=currents, shape=shape, dt=float(dt),
                        volume=setups[0].volume,
                        norm_drift=max(float(np.abs(n - 1.0).max()) for n in norms))
