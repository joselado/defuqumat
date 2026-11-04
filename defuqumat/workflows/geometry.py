"""``run_quantum_metric``: the quantum metric and the localization tensor of an insulator.

Two routes to one quantity, and they share the converged density and nothing
after it.

**``method = "kubo"``** (the default) is one fixed-density run with empty
states and the sum over occupied/empty pairs of
:mod:`defuqumat.response.geometry`. It gives ``g_ab(k)`` at every k-point of
the set, and its convergence parameter is the band count, reported as
:attr:`~defuqumat.response.geometry.QuantumMetric.truncation`.

**``method = "overlap"``** needs the occupied manifold only. The trace of the
product of the projectors at two neighbouring k-points is

    Tr P(k) P(k + b) = N - b_a b_b g_ab(k) + O(b^3),

so ``N - sum_mn |<u_m(k)|S|u_n(k + b)>|^2``, averaged over a uniform mesh, is the
mesh's own estimate of ``b . g . b``. It is the gauge-invariant spread
``Omega_I`` of Marzari and Vanderbilt (PRB 56, 12847 (1997), Eq. 34), which
Wannier90 computes in ``wann_omega``; here it is taken along the three mesh
steps and the three face diagonals, which gives the whole cartesian tensor
(``g_ab = (q(e_a + e_b) - q(e_a) - q(e_b)) / 2`` in mesh steps). Its error is
the finite difference's, ``O(1/N^2)`` in the mesh divisions, and there is no
sum over empty states to truncate, so it is what the Kubo route's truncation is
read against. The overlap is the primitive every invariant here is built from
(:meth:`~defuqumat.topology.states.PlaneWaveStates.overlaps`, the
Miller-index alignment and the ultrasoft ``q_ij(b)``).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from defuqumat.response.geometry import (
    QuantumMetric, quantum_metric, wannier_count, DEGENERACY_TOL,
)
from defuqumat.solvers.davidson import degeneracy_tolerance
from defuqumat.workflows.nscf import (
    fixed_density_ethr, fixed_density_states, threaded_calculation,
)
from defuqumat.scf.occupations import cut_gaps

__all__ = ["run_quantum_metric"]

#: The six mesh displacements of the overlap route, in mesh steps.
_STEPS = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1))


def run_quantum_metric(
    system,
    pseudos,
    density,
    *,
    method: str = "kubo",
    kpoints=None,
    nbnd: int | None = None,
    divisions=None,
    becsum: tuple = (),
    ns=None,
    tau=None,
    field=None,
    field_scale=None,
    conv_thr: float = 1.0e-10,
    degeneracy_tol: float | None = None,
    k_batch="default",
    calculation=None,
) -> QuantumMetric:
    """The quantum metric ``g_ab(k)`` and the localization tensor ``<r_a r_b>_c``.

    Args:
        system: the converged :class:`~defuqumat.system.builder.System`.
        pseudos: its pseudopotentials.
        density: the converged density.
        method: ``"kubo"`` (the sum over empty states, a map on the k-set) or
            ``"overlap"`` (neighbour overlaps on a uniform mesh, the occupied
            manifold alone, the zone integral).
        kpoints: for ``"kubo"``, the k-set to evaluate on, if not the one the
            density converged on: the whole grid, or a non-magnetic wedge.
        nbnd: for ``"kubo"``, the bands the fixed-density run resolves.
            Defaults to three times the occupied count, as the conductivity's.
            It is the truncation of the sum; read
            :attr:`~defuqumat.response.geometry.QuantumMetric.truncation`.
        divisions: for ``"overlap"``, the uniform Gamma-centred mesh
            ``(n1, n2, n3)``; defaults to the grid the k-points came from.
        becsum, ns, tau, field, field_scale: the converged state that cannot be
            rebuilt from the density (PAW, DFT+U, a meta-GGA, a field).
        conv_thr: the fixed-density solve's threshold.
        degeneracy_tol: for ``"kubo"``, the gap in Ry below which an
            occupied/empty pair is dropped and counted.
        k_batch: the chunk of the walk over k.
        calculation: the SCF's own, reused rather than rebuilt.
    """
    if method == "overlap":
        return _by_overlaps(system, pseudos, density, divisions=divisions,
                            becsum=becsum, ns=ns, tau=tau, field=field,
                            field_scale=field_scale, conv_thr=conv_thr,
                            k_batch=k_batch, calculation=calculation)
    if method != "kubo":
        raise ValueError(f"method must be 'kubo' or 'overlap', not {method!r}")

    from defuqumat.scf.driver import Calculation
    from defuqumat.workflows.conductivity import _default_nbnd

    calculation, system, kpoints, k_batch = threaded_calculation(
        calculation, system, kpoints, k_batch)
    if kpoints is not None:
        import equinox as eqx

        from defuqumat.system.kpoints import for_spin

        # The for_spin boundary of a caller-built k-set (P51).
        system = eqx.tree_at(lambda s: s.kpoints, system,
                             for_spin(kpoints, system.nspin))
    if calculation is None:
        calculation = Calculation(system, pseudos, k_batch=k_batch)
    # One band more than the sum uses, so the gap across the cut is measured.
    nbnd = int(nbnd or _default_nbnd(calculation))
    calculation, system, eigenvalues, wavefunctions = fixed_density_states(
        system, pseudos, density, nbnd=nbnd + 1,
        conv_thr=conv_thr, k_batch=k_batch, ns=ns, becsum=becsum, tau=tau,
        field=field, field_scale=field_scale, calculation=calculation,
    )
    k_batch = calculation.k_batch
    eigenvalues = jnp.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    band_cut_gap = float(np.min(cut_gaps(eigenvalues, (nbnd,) * eigenvalues.shape[0])))
    eigenvalues = eigenvalues[..., :nbnd]
    wavefunctions = wavefunctions[..., :nbnd, :]
    potential = calculation.potential(
        jnp.asarray(density),
        1.0 if field_scale is None else float(field_scale),
        field, tau=tau,
    )
    _, ddd_paw = calculation.onecenter(becsum)
    degeneracy_tol = degeneracy_tolerance(
        degeneracy_tol, DEGENERACY_TOL,
        ethr=fixed_density_ethr(conv_thr, calculation.nelec),
        site="the quantum metric")
    return quantum_metric(
        calculation, wavefunctions, eigenvalues, potential.v_scf,
        ddd_paw=ddd_paw, ns=ns, band_cut_gap=band_cut_gap,
        degeneracy_tol=degeneracy_tol, k_batch=k_batch,
    )


def _by_overlaps(system, pseudos, density, *, divisions, becsum, ns, tau, field,
                 field_scale, conv_thr, k_batch, calculation) -> QuantumMetric:
    """The overlap route: ``N - ||M(k, k + b)||^2`` on a uniform mesh."""
    from defuqumat.workflows.topology import _occupied_bands, _source

    if getattr(system, "spiral", False) or getattr(calculation, "spiral", False):
        raise NotImplementedError(
            "the quantum metric of a spin spiral is not implemented: its two "
            "spinor components live on two spheres and the neighbour overlaps "
            "of the mesh are written for one"
        )
    if divisions is None:
        divisions = system.kpoints.grid
    if divisions is None:
        raise ValueError("the overlap route needs a uniform mesh: pass divisions=(n1, n2, n3)")
    divisions = tuple(int(n) for n in divisions)
    n1, n2, n3 = divisions
    grid = np.stack(np.meshgrid(np.arange(n1), np.arange(n2), np.arange(n3),
                                indexing="ij"), axis=-1).reshape(-1, 3)
    points = grid / np.asarray(divisions, dtype=float)

    def index(cell):
        return int((cell[0] * n2 + cell[1]) * n3 + cell[2])

    bg = np.asarray(system.cell.bg)                       # rows b_i, 1/bohr
    steps = bg / np.asarray(divisions, dtype=float)[:, None]
    inverse = np.linalg.inv(steps)

    channels = (None,) if system.nspin != 2 else (0, 1)
    metric = np.zeros((len(channels), len(points), 3, 3))
    total = np.zeros((3, 3))
    for c, spin in enumerate(channels):
        nocc = _occupied_bands(system, pseudos, density, spin)
        source = _source(system, pseudos, density, nocc, nocc + 1, conv_thr,
                         k_batch, becsum=becsum, ns=ns, field=field,
                         field_scale=field_scale, calculation=calculation,
                         spin=spin, tau=tau)
        states = source.states(points)
        energies = np.asarray(states.energies)
        if float(np.min(energies[:, nocc])) <= float(np.max(energies[:, nocc - 1])):
            raise ValueError(
                "the quantum metric of the occupied manifold is refused on a "
                "metal: the lowest empty band dips below the highest occupied "
                "one somewhere on the mesh, so the manifold of the lowest "
                f"{nocc} bands is not the occupied one. The localization tensor "
                "of a metal is infinite (Souza, Wilkens and Martin)")
        quadratic = np.zeros((len(_STEPS), len(points)))
        for s, step in enumerate(_STEPS):
            pairs = []
            for cell in grid:
                moved = cell + np.asarray(step)
                wrapped = moved % np.asarray(divisions)
                shift = tuple(int(x) for x in moved // np.asarray(divisions))
                pairs.append((index(cell), index(wrapped), shift))
            matrices = np.asarray(states.overlaps(pairs, k_batch=k_batch))
            quadratic[s] = nocc - np.sum(np.abs(matrices) ** 2, axis=(1, 2))
        # q(x) = x^T C x in mesh steps, C = steps g steps^T.
        mesh = np.zeros((len(points), 3, 3))
        for a in range(3):
            mesh[:, a, a] = quadratic[a]
        for s, (a, b) in zip(range(3, 6), ((0, 1), (0, 2), (1, 2))):
            off = 0.5 * (quadratic[s] - quadratic[a] - quadratic[b])
            mesh[:, a, b] = mesh[:, b, a] = off
        metric[c] = np.einsum("ia,kab,jb->kij", inverse, mesh, inverse)
        total += metric[c].mean(axis=0)

    nelec = float(_electrons(system, pseudos))
    # nspin = 1: the mean manifold metric over nelec/2 spatial bands; a spinor
    # over nelec bands; nspin = 2 the two channels' sum over nelec.
    localization = total / wannier_count(nelec, system.nspin)
    return QuantumMetric(
        metric=metric.sum(axis=0),
        metric_by_spin=metric,
        kpoints=points,
        localization_tensor=localization,
        nelec=nelec,
        nwann=wannier_count(nelec, system.nspin),
        nbnd=None,
        method="overlap",
        divisions=divisions,
    )


def _electrons(system, pseudos) -> float:
    from defuqumat.scf.driver import Calculation

    nelec = getattr(system, "nelec", None)
    if nelec is not None:
        return float(nelec)
    return float(Calculation(system, pseudos).nelec)
