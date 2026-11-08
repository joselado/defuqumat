"""Zero-field hybrid Wannier states along ``a1``: the basis of the magnetic ultracell.

Wang and Vafek, *Revisiting Bloch electrons in magnetic field: Hofstadter physics via
hybrid Wannier states*, PRB 108, 245109 (2023), arXiv:2303.16347, build the magnetic Bloch
problem on zero-field states that are Bloch along ``a2`` and localized along ``a1``. This
module makes those states from plane-wave Kohn-Sham states; the field matrix elements and
the magnetic Bloch sum are :mod:`defuqumat.magnetic.ultracell_field`. The conventions (gauge,
sign, k labels) are :mod:`defuqumat.magnetic.conventions`, imported and never redefined.

**The gauge along kappa1.** At a fixed ``kappa' `` (the crystal coordinate along the
chemical ``b2``) the ``N1`` states ``psi_{m,i}`` at ``k_i = (i/N1, kappa', 0)`` are made
smooth by parallel transport on the closed loop: ``M_i = <u_i|u_{i+1}>`` (P16's overlaps,
:meth:`~defuqumat.topology.states.PlaneWaveStates.overlaps`, the ``(1, 0, 0)`` Miller shift
on the closing pair), ``W_i`` its unitary part (SVD), ``V_0 = 1``, ``V_{i+1} = W_i^dagger
V_i``; the closing Wilson loop ``Lambda = V_{N1-1}^dagger W_wrap = Z diag(e^{i phi_n})
Z^dagger`` (complex Schur form, exact for a unitary matrix); and the gauge

    psi~_{n,i} = sum_m psi_{m,i} [V_i Z]_{mn} exp(+i phi_n i / N1),   phi_n in (-pi, pi].

**The sign of the phase is the opposite of the design's** (``exp(-i phi_n i/N1)``, centre
``+phi_n / 2 pi``) and was derived and measured here, not taken: parallel transport brings
the frame back as ``psi'_N = psi_0 Lambda^dagger``, so ``psi~_N = psi~_0`` needs ``f_N =
diag(e^{+i phi})``. With that sign every consecutive overlap, the closing one included, has
the diagonal phase ``phi_n / N1`` and the centre along ``a1`` is ``-phi_n / 2 pi`` cells
(the Berry phase of the loop is ``-phi``); with the other sign the closing overlap differs
from the others by ``e^{2 i phi}``, a jump that delocalizes the function
(``test_the_other_sign_of_the_phase_is_not_periodic``). The branch ``(-pi, pi]`` rather than ``[0, 2 pi)``
puts every centre within half a cell of the origin, which keeps the half width ``w`` of the
band in ``Delta`` smallest; it relabels ``X`` and changes nothing else.

**The hybrid state of cell X** is ``h_{n,X} = (1/N1) sum_i exp(-2 pi i (i/N1) X)
psi~_{n,i}``, normalized to one over the ``N1 x 1 x 1`` box, periodic over ``N1`` cells.
On the box of :class:`~defuqumat.ultracell.grid.Ultracell` (P88) ``psi~_{n,i}`` sits at
``box_index(G, q_index = (i, 0, 0))`` and the sum over ``i`` is a scatter: the box
coefficients of ``h_{n,0}`` are ``c~_{n,i}(G) / sqrt(N1)``, relative to the common
wavevector ``K = 2 pi kappa' b2`` (the supercell k-point ``(0, kappa', 0)``).

**Step 1 is local only** (:func:`local_only`): the bare nonlocal term does not commute with
the magnetic translations ``T(R)``, which the magnetic basis rests on, so the zero-field
states are eigenstates of ``T + V_loc``; :func:`free_electron` zeroes ``V_loc`` too, for the
free Landau problem. Both edit a :class:`~defuqumat.hamiltonian.operator.Hamiltonian` with
``eqx.tree_at`` and leave everything else as the zero-field code built it.

**Localization is a property of the band set, and it is measured, not assumed.** The
lowest ``nbnd`` bands along a ``k1`` line form a smooth subspace only when a gap separates
band ``nbnd`` from band ``nbnd + 1`` on the whole line. Where they cross, the subspace jumps
and the hybrid states have power-law tails; :func:`outside_weight` is the number that says
so, and the half width of :mod:`~defuqumat.magnetic.ultracell_field` is read from it.
Measured on D22 (``plans/orbital-field/u1.md``): free electrons on the graphene cell at
``q = 10``, ``nbnd = 2``, ``N1 = 10``, the weight outside ``|xi1| > w`` is 0.27, 0.11, 0.06,
0.03 at ``w = 1 .. 4``, the ``1/(pi^2 w)`` of a sinc (the lowest band switches plane wave at
``k1 = 1/2``), and graphene local-only at ``nbnd = 8`` is the same (0.02 at ``w = 5``,
``N1 = 12``). An isolated composite (graphene local-only, ``nbnd = 2``, a 6 Ry gap) reads
1.4e-4 at ``w = 1`` and 7e-6 at ``w = 5``, a slower tail than its gap implies: the plane-wave
sphere of each ``k`` is a step function of ``k``.

**The projected trial basis (route U step 3, P176; fable's decision,
``plans/orbital-field/route-u-basis/decision.md``).** The transport gauge above refuses on every
plane-wave crystal, because the lowest ``nbnd`` bands are entangled and a hybrid state of an
entangled set is a sinc along ``xi1``, whose ``<x^2>`` diverges: its Landau-gauge matrix elements
do not exist. :func:`projected_hybrid` replaces the gauge. The hybrid state of trial ``j`` is the
trial projected on the lowest ``N`` bands at each ``k`` of the loop,

    psi~_{j,i} = sum_{m < N} psi_{m,i} A_{mj}(k_i),   A_{mj} = <psi_{m,i}|g_j(k_i)>,

with no Loewdin step and no transport gauge, so :func:`hybrid_set` takes the rotation ``A``
``(N1, N, J)`` unchanged and the overlap ``S`` of the ultracell carries the non-orthogonality
(the canonical orthogonalisation of ``solve`` handles it). ``g_j(k)`` is the Bloch sum of a
localized function, a function of ``k + G`` alone, so ``h_{j,0}`` is localized at cell 0 and
periodic in ``kappa'``, which the Miller relabelling of the reduced states needs. Adding a trial
adds one function and changes none of the others, so the spans are nested in ``J`` and the
levels of a Rayleigh-Ritz evaluation fall monotonically with ``J``. ``N`` is not a knob: it is
fixed by the **projectability** ``p_j(k) = sum_{m<N} |A_{mj}|^2 / |g_j|^2`` (Qiao, Pizzi and
Marzari, arXiv:2303.07877), one for a trial inside the ``N``-band space. Measured by the probe
on the free problem at 1/10 (c = 5 A): ``1 - p`` of 1e-6 passes, 2.7e-3 reads Hermiticity
2.5e-4, and 0.12 ruins the spectrum, so the workflow refuses ``1 - p > 1e-4``.

The trials are atom-centred (:func:`atomic_trials`, the hydrogenic and pseudo-atomic functions
of :mod:`defuqumat.wannier.trial` and :mod:`defuqumat.wannier.pao` as they are) or, for the
free problem, :class:`StripeTrials`: a Hermite-Gaussian along ``xi1`` on the stripe of the
lowest free band, ``G2 = -round(kappa')``, ``G3 = 0`` (:func:`stripe_trials`). The stripe must
be periodic in ``kappa'``: a ``G2 = 0`` stripe at every reduced point cuts the chain of
Gaussians at the magnetic cell's boundary and reads levels 0.3 to 1.2 ``B_ph`` off
(``fable-basis/blocks-s05.log``).
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.linalg

__all__ = ["local_only", "free_electron", "TransportGauge", "transport_gauge",
           "HybridSet", "hybrid_set", "outside_weight", "half_width",
           "ProjectedGauge", "projected_hybrid", "StripeTrials", "stripe_trials", "atomic_trials"]


# --- the zero-field operator of step 1 --------------------------------------------------

def local_only(hamiltonian):
    """``hamiltonian`` with its nonlocal coefficients set to zero: ``T + V_loc`` only.

    The projectors stay (so every shape and the Davidson path are those of the full
    operator); ``D_ij = 0`` removes their term from ``apply``, ``matrix`` and
    ``diagonal`` alike, through :attr:`Hamiltonian.coefficients`.
    """
    zeros = jnp.zeros_like(hamiltonian.projectors.dij if hamiltonian.deeq is None
                           else hamiltonian.deeq)
    return eqx.tree_at(lambda h: h.deeq, hamiltonian, zeros, is_leaf=lambda x: x is None)


def free_electron(hamiltonian):
    """``-grad^2`` alone: :func:`local_only` with the local potential set to zero as well."""
    h = local_only(hamiltonian)
    h = eqx.tree_at(lambda m: m.potential, h, jnp.zeros_like(h.potential))
    if h.potential_wave is not None:
        h = eqx.tree_at(lambda m: m.potential_wave, h, jnp.zeros_like(h.potential_wave))
    return h


# --- the parallel-transport gauge along kappa1 -----------------------------------------

@dataclasses.dataclass(frozen=True)
class TransportGauge:
    """The smooth gauge of one closed ``k1`` loop, host NumPy.

    ``rotation`` ``(N1, nbnd, nbnd)``: ``psi~_{.,i} = psi_{.,i} @ rotation[i]``;
    ``phases`` ``(nbnd,)`` the Wilson-loop phases ``phi_n`` in ``(-pi, pi]``;
    ``centres`` ``-phi_n / 2 pi`` in cells along ``a1``; ``singular`` the smallest singular
    value of the ``N1`` link overlaps, which is one for a subspace that does not change
    between neighbours and falls where the lowest ``nbnd`` bands exchange a member.
    """

    rotation: np.ndarray
    phases: np.ndarray
    centres: np.ndarray
    singular: float


def transport_gauge(links: np.ndarray) -> TransportGauge:
    """The gauge of the module docstring from the ``N1`` link overlaps of one loop.

    ``links[i] = <u_i | u_{i+1}>`` for ``i < N1 - 1`` and ``links[N1 - 1]`` the closing
    ``<u_{N1-1} | u_N>``, ``u_N(G) = u_0(G + b1)``, each ``(nbnd, nbnd)``.
    """
    links = np.asarray(links, dtype=complex)
    n1, nbnd = links.shape[0], links.shape[1]
    smallest = 1.0
    unitary = []
    for m in links:
        u, s, vh = np.linalg.svd(m)
        smallest = min(smallest, float(s.min()))
        unitary.append(u @ vh)
    frames = [np.eye(nbnd, dtype=complex)]
    for i in range(n1 - 1):
        frames.append(unitary[i].conj().T @ frames[i])
    wilson = frames[-1].conj().T @ unitary[-1]
    # a unitary matrix is normal, so its complex Schur form is diagonal and Z unitary
    t, z = scipy.linalg.schur(wilson, output="complex")
    phases = np.angle(np.diag(t))                       # (-pi, pi]
    order = np.argsort(-phases / (2.0 * np.pi), kind="stable")
    z, phases = z[:, order], phases[order]
    rotation = np.stack([frames[i] @ z * np.exp(1j * phases * i / n1)[None, :]
                         for i in range(n1)])
    return TransportGauge(rotation=rotation, phases=phases,
                          centres=-phases / (2.0 * np.pi), singular=smallest)


# --- the hybrid states on the box -------------------------------------------------------

class HybridSet(eqx.Module):
    """``h_{n,0}(kappa')`` for every band of one ``kappa'``, as box plane-wave coefficients.

    ``values`` ``(nbnd, N1 npwx)``: ``c~_{n,i}(G) / sqrt(N1)`` (zero on padding);
    ``h0_values`` the same weighted by the zero-field eigenvalues, the box coefficients of
    ``H_0 h_{n,0}`` (exact, ``H_0 psi = eps psi``); ``miller`` ``(N1 npwx, 3)`` the chemical
    Miller index of each entry and ``qindex`` ``(N1 npwx,)`` its ``i``. ``kappa`` is the
    reduced ``kappa'`` in ``[0, 1)``; ``centres`` in cells.
    """

    values: jnp.ndarray
    h0_values: jnp.ndarray
    miller: np.ndarray = eqx.field(static=False)
    qindex: np.ndarray = eqx.field(static=False)
    centres: np.ndarray = eqx.field(static=False)
    singular: float = eqx.field(static=True)
    kappa: float = eqx.field(static=True)


def hybrid_set(coefficients, energies, miller, valid, gauge: TransportGauge, kappa: float):
    """The :class:`HybridSet` of one ``kappa'`` from its ``N1`` solved k-points.

    ``coefficients`` ``(N1, nbnd, npwx)`` (the band dtype), ``energies`` ``(N1, nbnd)`` Ry,
    ``miller`` ``(N1, npwx, 3)``, ``valid`` ``(N1, npwx)``, in the order ``i = 0 .. N1 - 1``.
    """
    n1 = coefficients.shape[0]
    rotation = jnp.asarray(gauge.rotation, dtype=coefficients.dtype)
    mask = jnp.asarray(valid)[:, None, :]
    coefficients = jnp.where(mask, coefficients, 0.0)
    scale = 1.0 / np.sqrt(n1)
    # psi~_{n,i}(G) = sum_m c_{m,i}(G) R_i[m, n]
    smooth = jnp.einsum("img,imn->ing", coefficients, rotation) * scale
    weighted = jnp.einsum("img,im,imn->ing", coefficients,
                          jnp.asarray(energies, dtype=coefficients.real.dtype), rotation) * scale
    nbnd, npwx = smooth.shape[1], smooth.shape[2]
    values = jnp.transpose(smooth, (1, 0, 2)).reshape(nbnd, n1 * npwx)
    h0_values = jnp.transpose(weighted, (1, 0, 2)).reshape(nbnd, n1 * npwx)
    qindex = np.repeat(np.arange(n1), npwx)
    return HybridSet(values=values, h0_values=h0_values,
                     miller=np.asarray(miller).reshape(n1 * npwx, 3).astype(np.int64),
                     qindex=qindex, centres=np.asarray(gauge.centres),
                     singular=gauge.singular, kappa=float(kappa))


def outside_weight(profile: np.ndarray, xi: np.ndarray, widths) -> np.ndarray:
    """``(nbnd, len(widths))``: each function's weight at ``|xi1| > w`` for each ``w``.

    ``profile`` ``(nbnd, n1_box)`` is ``|h|^2`` summed over the two other box axes and
    normalized to one per function; ``xi`` ``(n1_box,)`` the sawtooth ``xi1`` in cells.
    """
    profile = np.asarray(profile, dtype=float)
    profile = profile / profile.sum(axis=1, keepdims=True)
    return np.stack([profile[:, np.abs(xi) > w].sum(axis=1) for w in widths], axis=1)


def half_width(weights: np.ndarray, widths, tol: float = 1.0e-8):
    """The smallest ``w`` whose largest outside weight is below ``tol``, or ``None``."""
    worst = np.max(weights, axis=0)
    for w, value in zip(widths, worst):
        if value < tol:
            return int(w)
    return None


# --- the projected trial basis (step 3) ------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ProjectedGauge:
    """The projected trials of one closed ``k1`` loop, host NumPy, for :func:`hybrid_set`.

    ``rotation`` ``(N1, N, J)``: ``A_{mj}(k_i) = <psi_{m,i}|g_j(k_i)>``, so ``psi~_{.,i} =
    psi_{.,i}[:N] @ rotation[i]`` is the projection of each trial; ``projectability`` ``(N1, J)``
    ``sum_m |A_{mj}|^2 / |g_j|^2``, at most one; ``centres`` zeros (the trials sit at cell 0 by
    construction); ``singular`` the smallest singular value over the loop of ``A`` with its
    columns scaled by ``1/|g_j|``, which falls where two projected trials become dependent.
    """

    rotation: np.ndarray
    projectability: np.ndarray
    centres: np.ndarray
    singular: float


def projected_hybrid(coefficients, trials, N: int) -> ProjectedGauge:
    """The :class:`ProjectedGauge` of one loop: ``A = <psi|g>`` over the lowest ``N`` bands.

    ``coefficients`` ``(N1, nbnd, npwx)`` with ``nbnd >= N``, ``trials`` ``(N1, J, npwx)``, both on
    the same sphere of each ``k_i`` (zero on padding). No Loewdin step and no transport gauge:
    the overlap of the ultracell carries the non-orthogonality.
    """
    psi = np.asarray(coefficients)
    g = np.asarray(trials)
    N = int(N)
    if psi.shape[1] < N:
        raise ValueError(f"the loop holds {psi.shape[1]} bands, fewer than the window N = {N}")
    if g.shape[0] != psi.shape[0] or g.shape[2] != psi.shape[2]:
        raise ValueError(f"trials {g.shape} do not match the states {psi.shape} (N1, ., npwx)")
    rotation = np.einsum("img,ijg->imj", psi[:, :N].conj(), g)
    norm = np.sum(np.abs(g) ** 2, axis=2)                                 # (N1, J)
    if np.any(norm <= 0.0):
        i, j = np.argwhere(norm <= 0.0)[0]
        raise ValueError(f"trial {j} has no weight on the sphere of k-point {i} of the loop")
    projectability = np.sum(np.abs(rotation) ** 2, axis=1) / norm
    scaled = rotation / np.sqrt(norm)[:, None, :]
    smallest = min(float(np.linalg.svd(a, compute_uv=False).min()) for a in scaled)
    return ProjectedGauge(rotation=rotation, projectability=projectability,
                          centres=np.zeros(g.shape[1]), singular=smallest)


@dataclasses.dataclass(frozen=True)
class StripeTrials:
    """The free problem's trials: ``J`` Hermite-Gaussians along ``xi1`` on the lowest stripe.

    ``g_j = H_j(xi1/sigma) exp(-xi1^2 / 2 sigma^2)``, ``xi1 = b1 . r`` in cells, ``sigma`` in
    cells, times the plane wave of the lowest free band along ``b2`` (``G2 = -round(kappa')``,
    ``G3 = 0``). ``chirp`` dresses the Gaussian (``J = 1`` only) with the phase ``exp(-i f
    xi1^2)`` of the analytic Landau state on a cell with ``b1 . b2 != 0`` (u1's item 0b), ``f``
    given by the workflow from the flux p/q (never from ``scale``): with ``sigma = l_B |b1|`` the exact ``n = 0`` state is
    then in the span, the plumbing line of the gate.
    """

    J: int
    sigma: float = 0.5
    chirp: bool = False

    def __post_init__(self):
        if int(self.J) < 1:
            raise ValueError(f"a stripe trial set needs J >= 1, got {self.J}")
        if not self.sigma > 0.0:
            raise ValueError(f"the stripe trial's sigma must be positive, got {self.sigma}")
        if self.chirp and int(self.J) != 1:
            raise ValueError("the chirped stripe is the analytic n = 0 state, J = 1 only")

    @property
    def names(self) -> tuple:
        tag = ", chirped" if self.chirp else ""
        return tuple(f"stripe H{j} sigma {self.sigma:g}{tag}" for j in range(int(self.J)))


def stripe_trials(spec: StripeTrials, kpoints, miller, valid, chirp_coefficient: float = 0.0):
    """``(nk, J, npwx)``: :class:`StripeTrials` on the sphere of every k-point.

    ``kpoints`` ``(nk, 3)`` crystal, ``miller`` ``(nk, npwx, 3)``, ``valid`` ``(nk, npwx)``.
    The momentum along ``xi1`` is ``kap = 2 pi (k1 + G1)`` in 1/cells, so the trial is a function
    of ``k + G`` alone; the stripe keeps ``G2 = -floor(k2 + 1/2)`` (the plane wave whose
    ``k2 + G2`` lies in ``[-1/2, 1/2)``) and ``G3 = 0``. ``chirp_coefficient`` is ``f`` of the
    class docstring, per cell squared.
    """
    from numpy.polynomial.hermite import hermval

    kpoints = np.asarray(kpoints, dtype=float)
    miller = np.asarray(miller)
    valid = np.asarray(valid, dtype=bool)
    J, sigma = int(spec.J), float(spec.sigma)
    out = np.zeros((len(kpoints), J, miller.shape[1]), dtype=complex)
    for k, kpt in enumerate(kpoints):
        kap = 2.0 * np.pi * (kpt[0] + miller[k, :, 0])
        g2 = -int(np.floor(kpt[1] + 0.5))
        stripe = valid[k] & (miller[k, :, 1] == g2) & (miller[k, :, 2] == 0)
        for j in range(J):
            if spec.chirp:
                # exp(-a xi^2), a = 1/(2 sigma^2) + i f: the Gaussian times exp(-i f xi^2)
                a = 1.0 / (2.0 * sigma ** 2) + 1j * float(chirp_coefficient)
                g = np.exp(-kap ** 2 / (4.0 * a))
            else:
                c = np.zeros(j + 1)
                c[j] = 1.0
                g = (-1j) ** j * hermval(sigma * kap, c) * np.exp(-0.5 * sigma ** 2 * kap ** 2)
            out[k, j] = np.where(stripe, g, 0.0)
    return out


def atomic_trials(states, projections, pseudos, volume: float):
    """``(nk, J, npwx)``: atom-centred trials on the sphere of every k-point of ``states``.

    ``projections`` are Wannier90 projections (:func:`~defuqumat.wannier.projections.parse_projections`)
    or pseudo-atomic orbitals (:func:`~defuqumat.wannier.pao.pao_projections`), in any mix,
    evaluated by :func:`~defuqumat.wannier.matrices.guiding_on_mesh` exactly as the Wannier
    engine evaluates them. Norm-conserving and collinear only (the ultracell refuses the rest).
    """
    from defuqumat.wannier.matrices import guiding_on_mesh
    from defuqumat.wannier.pao import pao_radial_tables

    if states.npol != 1:
        raise NotImplementedError("projected trials for spinor states")
    species = {p.species for p in projections if getattr(p, "radial_key", None) is not None}
    extra = pao_radial_tables(pseudos, species, volume) if species else None
    out = np.zeros((states.nk, len(projections), states.npwx), dtype=complex)
    for k, g in guiding_on_mesh(states, projections, volume, extra_tables=extra):
        out[k] = g
    return out
