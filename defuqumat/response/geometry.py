"""The quantum metric of the occupied manifold, and the localization tensor.

The occupied Bloch states ``|u_nk>`` of an insulator span a subspace that turns
as ``k`` moves. How fast it turns is measured by the **quantum geometric
tensor**,

    Q_ab(k) = sum_{n occ} <d_a u_n| (1 - P_k) |d_b u_n>,   P_k = sum_{n occ} |u_n><u_n|,

whose imaginary part is the Berry curvature (``Omega_ab = -2 Im Q_ab``,
:mod:`defuqumat.topology.kubo`) and whose real part is the **quantum metric**
``g_ab(k) = Re Q_ab(k)``, the distance ``ds^2 = g_ab dk_a dk_b`` between the
manifolds at ``k`` and ``k + dk``. With the projector written out over the
empty states,

    g_ab(k) = Re sum_{n occ, m empty} v^a_nm v^b_mn / (e_m - e_n)^2,

where ``v^a_nm = <n|dH_a - e_m dS_a|m> + (e_m - e_n) K^a_nm`` is the
generalised velocity :mod:`defuqumat.response.conductivity` already contracts.
**The metric is the real part of the very product whose imaginary part is the
curvature**, with the same weight ``W_n (1 - f_m) / e_mn^2``: per pair
``v^a_nm v^b_mn`` is identically the product ``kubo_from_matrices`` forms
(checked on a two-band model and on a generalised one with a moving ``S``, both
to 6e-15; without the augmentation connection the generalised model is off by
2.8 of 4.6). So no third assembly of the matrix elements is written here: the
elements are the conductivity's, and the ``e_n``/``e_m`` and ``K``/``K^dag``
placements stay where they are pinned.

**What the zone integral is, and what measures it.** The k-average of the
manifold metric divided by the number of occupied bands is the *localization
tensor* of Resta and of Souza, Wilkens and Martin (SWM, PRB 62, 1666 (2000);
Sgiarovello, Peressi and Resta, PRB 64, 115202 (2001), arXiv:cond-mat/0101440,
Eq. 17), the second cumulant moment of the electrons' positions,

    <r_a r_b>_c = (1/N_e) sum_k sum_{n,m} W_n (1 - f_m) Re(v^a_nm v^b_mn) / e_mn^2,

in bohr^2 per electron, with ``W_n`` QE's ``wg`` (k weight times occupation,
summing to ``N_e``). It is finite in an insulator and infinite in a metal, which
is Kohn's distinction between the two made into a number. Its trace times the
number of Wannier functions is the gauge-invariant part ``Omega_I`` of the
Marzari-Vanderbilt spread, the floor no choice of Wannier gauge can go below
(``wann_omega`` in Wannier90's ``wannierise.F90``; QE itself has no counterpart).
And it is a measured quantity through the SWM sum rule,

    <r_a r_b>_c = hbar V_c / (pi e^2 N_e) int_0^inf dw Re sigma_ab(w) / w,

the inverse-frequency moment of the optical absorption (the constant is the one
this code's own resolvent gives: ``Re[i t z / (w - e + i eta)] -> pi t Re z
delta(w - e)``).

**What is refused, and why.** A **metal**: the sum diverges as the mesh grows,
because occupied/empty pairs at the Fermi surface have ``e_mn -> 0``, and a
smeared k-sum gives a finite number that is a property of the mesh and the
smearing, not of the crystal. The test is on the fillings, not on the
occupation scheme, so a smeared insulator (every ``f`` at 0 or 1 to round-off)
runs. A **spin spiral**, for the conductivity's reason. A wedge of a magnetic
group, for :func:`~defuqumat.system.wedge.wedge_group`'s.

**The band count is the convergence parameter**, as for the conductivity: the
sum over empty states stops where the eigensolver stopped. It is reported as
:attr:`QuantumMetric.truncation`, the shift of the localization tensor when the
highest band is dropped. The overlap route
(:func:`~defuqumat.workflows.geometry.run_quantum_metric` with ``method =
"overlap"``) has no sum over empty states at all, and is what that number is
read against.

**Cost and peak** are the conductivity's static route: three ``jvp`` calls per
k-point for ``(3, chunk, nbnd, nbnd)`` matrix elements, walked a k-chunk at a
time (:mod:`defuqumat.response.walk`), and an ``(nspin, nk, 3, 3)`` map.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from defuqumat.response.conductivity import DEGENERACY_TOL, PAIR_WEIGHT_FLOOR
from defuqumat.response.velocity import VelocityOperator, with_connections
from defuqumat.response.walk import (
    chunk_size, padded, row_leaves, store_rows, walk, with_rows,
)
from defuqumat.system.wedge import wedge_group
from defuqumat.units import BOHR_TO_ANGSTROM

__all__ = [
    "QuantumMetric",
    "quantum_metric",
    "require_an_insulator",
    "METAL_FILLING_TOL",
]

#: Above this ``max f (1 - f)`` over the weight-carrying bands the run is a
#: metal and refused. A smeared insulator's fillings are 0 and 1 to round-off,
#: far below it; one band at the Fermi level of a metal is of order 0.1.
METAL_FILLING_TOL = 1.0e-8


@dataclass(frozen=True)
class QuantumMetric:
    """The quantum metric on a k-set and its zone integral."""

    #: ``(nk, 3, 3)`` in bohr^2: ``g_ab(k)`` of the occupied manifold, summed
    #: over the occupied bands and over the collinear channels, cartesian, on
    #: the run's own k-points (on a wedge, at its representatives; the image at
    #: ``R k`` is ``R g R^T``). Symmetric, and positive semi-definite.
    metric: np.ndarray
    #: ``(nspin, nk, 3, 3)``, the same channel by channel.
    metric_by_spin: np.ndarray
    #: ``(nk, 3)`` crystal coordinates of the k-points the map is on.
    kpoints: np.ndarray
    #: ``(3, 3)`` in bohr^2: ``<r_a r_b>_c``, per electron, completed by the
    #: point group on a wedge.
    localization_tensor: np.ndarray
    #: How many electrons the tensor is divided by.
    nelec: float
    #: How many Wannier functions the occupied manifold makes: ``nelec / 2``
    #: spatial ones at ``nspin = 1``, ``nelec`` spin-orbitals otherwise.
    nwann: float
    #: Bands the sum over empty states ran over, and the route that gave it.
    nbnd: int | None = None
    method: str = "kubo"
    #: The largest change of :attr:`localization_tensor` when the highest band
    #: is dropped from the sum, over its largest element. ``None`` for the
    #: overlap route, which has no such sum.
    truncation: float | None = None
    #: The same change unnormalised, in bohr^2.
    truncation_abs: float | None = None
    #: ``min_k (e_(nbnd+1) - e_nbnd)`` in Ry; below the degeneracy tolerance the
    #: band set was cut inside a multiplet and is warned about.
    band_cut_gap: float = float("nan")
    #: Occupied/empty pairs the degeneracy guard dropped. Nonzero means the gap
    #: closes on the mesh and the number is not the insulator's.
    singular_pairs: int = 0
    #: For the overlap route: the mesh it was taken on.
    divisions: tuple | None = None

    @property
    def spread_invariant(self) -> float:
        """``Omega_I = nwann Tr <r r>_c`` in bohr^2: Marzari and Vanderbilt's
        gauge-invariant spread, the part of the total no Wannier gauge can
        lower (Wannier90's ``Omega I``, which it prints in angstrom^2)."""
        return float(self.nwann * np.trace(self.localization_tensor))

    @property
    def spread_invariant_angstrom2(self) -> float:
        """:attr:`spread_invariant` in angstrom^2, Wannier90's unit."""
        return self.spread_invariant * BOHR_TO_ANGSTROM ** 2

    @property
    def localization_length(self) -> np.ndarray:
        """``(3,)`` in bohr: the square roots of the eigenvalues of
        :attr:`localization_tensor`, the spread of an electron along its
        principal axes."""
        values = np.linalg.eigvalsh(0.5 * (self.localization_tensor
                                           + self.localization_tensor.T))
        return np.sqrt(np.clip(values, 0.0, None))


def wannier_count(nelec: float, nspin: int) -> float:
    """How many Wannier functions ``nelec`` electrons fill: two per spatial
    function at ``nspin = 1``, one per spin-orbital otherwise."""
    return float(nelec) / (2.0 if int(nspin) == 1 else 1.0)


# -- what it refuses ------------------------------------------------------------


def require_an_insulator(filling, wg) -> None:
    """Refuse a run whose occupied manifold has a Fermi surface.

    ``filling`` and ``wg`` are ``(nspin, nk, nbnd)``. Judged on the fillings
    themselves, ``max f (1 - f)`` over the bands that carry weight, so that a
    smeared insulator passes and a metal does not whatever its scheme.
    """
    filling = np.asarray(filling, dtype=float)
    partial = filling * (1.0 - filling)
    worst = float(np.max(partial)) if partial.size else 0.0
    if worst > METAL_FILLING_TOL:
        raise ValueError(
            "the quantum metric of the occupied manifold is refused on a metal: "
            f"a band is partially filled here (max f(1-f) = {worst:.3g}). The "
            "localization tensor of a metal is infinite (Souza, Wilkens and "
            "Martin): occupied/empty pairs at the Fermi surface have a vanishing "
            "gap and 1/e_mn^2 diverges, so a smeared k-sum gives a number set by "
            "the mesh and the smearing. What a metal has instead is the "
            "Fermi-sea metric, weighted (f_n - f_m)/e_mn^2, which is not "
            "implemented"
        )


def _require_a_regime(calculation) -> None:
    if getattr(calculation, "spiral", False):
        raise NotImplementedError(
            "the quantum metric of a spin spiral is not implemented: the two "
            "spinor components live on spheres centred at k + q/2 and k - q/2, "
            "and the walk over k takes one list of nk rows where a spiral's are "
            "rows ik and nk + ik of a list of 2 nk (the optical conductivity's "
            "refusal, for the same matrix elements)"
        )


# -- the assembly ---------------------------------------------------------------


def _metric_sum(element, energies, filling, tol):
    """One k-point's ``(g_ab, g_ab without the top band, dropped pairs)``.

    ``element`` is ``(3, nbnd, nbnd)``, the generalised velocity. The weight is
    ``f_n (1 - f_m) / e_mn^2``, the filling and not the k-weighted ``wg``, so
    that the map is the metric itself; the k weight is applied on the host.
    """
    gap = energies[None, :] - energies[:, None]          # [n, m] = e_m - e_n
    finite = jnp.abs(gap) > tol
    safe = jnp.where(finite, gap, 1.0)
    weight = filling[:, None] * (1.0 - filling[None, :])
    pair = jnp.where(finite, weight / safe ** 2, 0.0)
    nbnd = energies.shape[0]
    off = ~jnp.eye(nbnd, dtype=bool)
    dropped = jnp.sum(jnp.where(off & ~finite & (weight > PAIR_WEIGHT_FLOOR), 1, 0))
    z = jnp.real(jnp.einsum("inm,jmn->nmij", element, element))
    full = jnp.einsum("nm,nmij->ij", pair, z)
    kept = jnp.arange(nbnd) < nbnd - 1
    trimmed = jnp.einsum("nm,nmij->ij", jnp.where(kept[None, :] & kept[:, None], pair, 0.0), z)
    return full, trimmed, dropped


def quantum_metric(
    calculation,
    wavefunctions,
    eigenvalues,
    v_scf,
    *,
    ddd_paw=None,
    ns=None,
    band_cut_gap: float = float("nan"),
    degeneracy_tol: float | None = None,
    k_batch="default",
) -> QuantumMetric:
    """``g_ab(k)`` and ``<r_a r_b>_c`` from a fixed-density run with empty states.

    The pure assembly, taking the states rather than a result object, in the
    shape of :func:`~defuqumat.response.conductivity.optical_conductivity`;
    :func:`~defuqumat.workflows.geometry.run_quantum_metric` is the entry point
    that produces the states.

    Args:
        calculation: the :class:`~defuqumat.scf.driver.Calculation` the states
            belong to, on the whole k-grid or a non-magnetic wedge.
        wavefunctions: ``(nspin, nk, nbnd, ndim)``.
        eigenvalues: ``(nspin, nk, nbnd)`` or ``(nk, nbnd)``, Ry.
        v_scf: the converged potential ``dH/dk`` is built at.
        ddd_paw, ns: PAW's one-centre coefficients and the Hubbard
            occupations, for the Hamiltonian ``dH/dk`` differentiates.
        band_cut_gap: ``min_k (e_(nbnd+1) - e_nbnd)``, measured by the caller.
        degeneracy_tol: in Ry, the gap below which an occupied/empty pair is
            dropped and counted; ``None`` is the conductivity's floor.
        k_batch: the chunk of the walk over k.
    """
    _require_a_regime(calculation)
    system = calculation.system
    wedge = wedge_group(system, quantity="the quantum metric")

    eigenvalues = jnp.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    if wavefunctions.ndim == 3:
        wavefunctions = wavefunctions[None]
    nspin = int(eigenvalues.shape[0])
    nk = int(eigenvalues.shape[1])
    nbnd = int(eigenvalues.shape[-1])

    wg, _ = calculation.occupations(eigenvalues)
    wg = np.asarray(wg)                                   # (nspin, nk, nbnd)
    wk = np.asarray(system.kpoints.weights)               # (nk,)
    filling = wg / wk[None, :, None]
    require_an_insulator(filling, wg)
    if not np.any(filling[..., -1] < 0.5):
        raise ValueError("the quantum metric is a sum over empty states and none "
                         "was diagonalised: raise nbnd")
    # The fillings of an insulator, rounded: what is left of a smearing in them
    # is round-off, and the metric weight multiplies it by 1/e^2.
    filling = np.round(filling)

    tol = DEGENERACY_TOL if degeneracy_tol is None else float(degeneracy_tol)
    from defuqumat.response.photocurrent import _warn_about_the_cut

    _warn_about_the_cut(band_cut_gap, "the quantum metric", _CUT_RESIDUE_METRIC, tol=tol)

    kcart = np.asarray(VelocityOperator(calculation, v_scf, ddd_paw, ns).kcart)
    from defuqumat.response.efield import _augmentation_dipole

    dipole = _augmentation_dipole(calculation)
    terms = calculation.local_terms(v_scf, ddd_paw)
    batch = chunk_size(calculation, k_batch)
    augmented = getattr(calculation, "augmentation", None) is not None
    bare = np.asarray(eigenvalues)

    def velocity_on(rowset, kc):
        return VelocityOperator(with_rows(calculation, rowset), v_scf, ddd_paw, ns,
                                kcart=kc, dipole=dipole, local_terms=terms)

    def tangents_of(rowset, psi, energies, kc):
        return velocity_on(rowset, kc).tangent_elements(psi, energies, sequential=True)

    def connections_of(tangents, rowset, psi, kc):
        return velocity_on(rowset, kc).connections(psi, sequential=True)

    def contract(tangents, *rest):
        connections = rest[0] if augmented else None
        energies, fill, rows, live = rest[1:] if augmented else rest
        elements = with_connections(tangents, energies, connections)
        elements = jnp.moveaxis(elements, 0, 2)           # (nspin, chunk, 3, nb, nb)
        per_k = jnp.vectorize(
            lambda e, en, f: _metric_sum(e, en, f, tol),
            signature="(i,n,n),(n),(n)->(i,i),(i,i),()",
        )
        full, trimmed, dropped = per_k(elements, energies, fill)
        live = live[None, :, None, None]
        # Scattered into the whole axis, so that the walk's sum over chunks is
        # the map; padded rows repeat a real one and carry ``live = 0``.
        place = jnp.zeros((nspin, nk, 3, 3), full.dtype)
        full_map = place.at[:, rows].add(full * live)
        trimmed_map = place.at[:, rows].add(trimmed * live)
        return full_map, trimmed_map, jnp.sum(dropped * live[..., 0, 0])

    def arguments(rows, live_count):
        rowset, psi = row_leaves(calculation, rows), store_rows(wavefunctions, rows)
        energies, kc = jnp.asarray(bare[:, rows]), jnp.asarray(kcart[rows])
        mask = padded(np.ones(nk), rows, live_count)
        last = (energies, jnp.asarray(filling[:, rows]), jnp.asarray(rows),
                jnp.asarray(mask))
        if augmented:
            return (rowset, psi, energies, kc), (rowset, psi, kc), last
        return (rowset, psi, energies, kc), last

    stages = ((tangents_of, connections_of, contract) if augmented
              else (tangents_of, contract))
    full_map, trimmed_map, dropped = walk(nk, batch, arguments, *stages)
    full_map = np.asarray(full_map)
    trimmed_map = np.asarray(trimmed_map)
    dropped = int(np.asarray(dropped))
    if dropped:
        warnings.warn(
            f"{dropped} occupied/empty pairs closer than {tol:.1e} Ry were "
            "dropped from the quantum metric: the gap closes on this k-set and "
            "1/e_mn^2 diverges there, so the number is not an insulator's",
            stacklevel=2,
        )

    # The k weight carries the spin degeneracy at nspin = 1 (it sums to 2) and
    # not otherwise, so this is the electron-weighted average of the channels.
    nelec = float(calculation.nelec)
    localization = np.einsum("k,skab->ab", wk, full_map) / nelec
    trimmed_loc = np.einsum("k,skab->ab", wk, trimmed_map) / nelec
    if wedge is not None:
        # g is a polar rank-2 tensor, symmetric, and even under time reversal.
        localization = wedge.average(localization, (-2, -1))
        trimmed_loc = wedge.average(trimmed_loc, (-2, -1))
    shift = float(np.max(np.abs(localization - trimmed_loc)))
    scale = float(np.max(np.abs(localization)))

    return QuantumMetric(
        metric=full_map.sum(axis=0),
        metric_by_spin=full_map,
        kpoints=np.asarray(system.kpoints.crystal(system.cell)),
        localization_tensor=localization,
        nelec=nelec,
        nwann=wannier_count(nelec, system.nspin),
        nbnd=nbnd,
        method="kubo",
        truncation=shift / scale if scale > 0.0 else float("nan"),
        truncation_abs=shift,
        band_cut_gap=float(band_cut_gap),
        singular_pairs=dropped,
    )


#: On a mesh holding Gamma and X every cut of silicon from 20 to 43 bands falls
#: inside a multiplet somewhere (the largest cut gap over 12^3 is 2.7e-15 Ry), so
#: the warning cannot be avoided by choosing ``nbnd``. What the rotation can move
#: is the share of the members kept at the top of the set, which is the scale
#: :attr:`QuantumMetric.truncation` measures (1.1e-4 there).
_CUT_RESIDUE_METRIC = (
    "for the metric the rotation moves only the share of the bands at the top of "
    "the set, the scale QuantumMetric.truncation reports (1.1e-4 of the tensor on "
    "silicon at 30 bands on a 12^3 mesh, where every cut from 20 to 43 bands is "
    "inside a multiplet somewhere)"
)
