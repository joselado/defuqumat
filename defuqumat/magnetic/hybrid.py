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
"""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import scipy.linalg

__all__ = ["local_only", "free_electron", "TransportGauge", "transport_gauge",
           "HybridSet", "hybrid_set", "outside_weight", "half_width"]


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
