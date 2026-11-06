"""Deterministic KPM traces, the spin-resolved trace and the Fermi level (P168).

Each check is against something the recursion does not touch: the Loewdin
charges from the occupied SCF states (``atomic_projections``), the SCF density
on its grid, the exact scalar traces under a known uniform exchange field, and
the SCF's own Fermi level.
"""

from __future__ import annotations

import functools
from pathlib import Path

import jax
import numpy as np
import pytest

import defuqumat  # noqa: F401
from defuqumat import Calculator
from defuqumat.kpm import kpm_moments
from defuqumat.kpm.bounds import spectral_interval
from defuqumat.kpm.moments import live_mask, spin_moments
from tests.unit.test_kpm import SILICON, _dense_spectrum, _silicon

pytestmark = pytest.mark.unit

PSEUDO = Path(__file__).parents[1] / "data" / "pseudo"


def _gap(hamiltonians):
    spectra = _dense_spectrum(hamiltonians[0])
    return 0.5 * (max(s[3] for s in spectra) + min(s[4] for s in spectra))


def test_the_orbital_moments_are_the_dense_projections():
    """``<phi_a|T_n(H')|phi_a> = sum_n |<phi_a|psi_n>|^2 T_n(e_n')`` over every eigenvector of the dense H.

    On the whole grid, where the per-orbital trace runs; exact to round-off at
    any number of moments, and the eigensolve shares nothing with the recursion.
    """
    from defuqumat.kpm.traces import whole_grid_calculation
    from defuqumat.projwfc.projections import atomic_orbitals
    from defuqumat.workflows.nscf import fixed_density_hamiltonians

    calc, result, _ = _silicon()
    pdos = calc.get_kpm_pdos(n_moments=40)
    whole = whole_grid_calculation(calc.calculation)
    hamiltonian = fixed_density_hamiltonians(whole, result.density)[0]
    orbitals = np.asarray(atomic_orbitals(whole))  # (nk, npwx, nwfc)
    weights = np.asarray(whole.system.kpoints.weights)
    expected = np.zeros_like(pdos.moments[0])
    for ik, w in enumerate(weights):
        live = np.asarray(hamiltonian.mask[ik])
        values, vectors = np.linalg.eigh(np.asarray(hamiltonian.matrix(ik))[np.ix_(live, live)])
        weight = np.abs(orbitals[ik][live].conj().T @ vectors) ** 2  # (nwfc, nstates)
        x = (values - pdos.centre) / pdos.half_width
        expected += w * weight @ np.cos(np.arange(40)[:, None] * np.arccos(x)[None, :]).T
    np.testing.assert_allclose(pdos.moments[0], expected, rtol=0, atol=1e-10)


def test_the_orbital_charges_are_the_loewdin_charges():
    """Below the gap, the KPM PDOS of each orbital integrates to its Loewdin charge from the bands.

    The difference is the kernel's tail across the 0.033 Ry gap of this mesh,
    and it is the s orbitals' (the lowest conduction band has s weight):
    measured 3.7e-4 at a broadening of 0.005 Ry and 4.3e-5 at 0.0025, the same
    at conv_thr 1e-10 and 1e-13.
    """
    from defuqumat.projwfc.projections import atomic_projections

    calc, result, hamiltonians = _silicon()
    pdos = calc.get_kpm_pdos(broadening=0.0025)
    charges = pdos.charges_at(_gap(hamiltonians))[0]
    projections = atomic_projections(calc.calculation, result.wavefunctions)[0]
    reference = np.einsum("kb,kab->a", np.asarray(result.occupations), projections)
    np.testing.assert_allclose(charges, reference, rtol=0, atol=1e-4)
    assert pdos.select(l="p").shape == pdos.energies.shape


def test_the_local_dos_integrates_to_the_density():
    """``int^gap LDOS(r0, E) dE`` is the SCF density at ``r0``, at three grid points.

    The difference is the kernel's tail across the gap, as for the charges:
    about 3e-4 relative at 0.005 Ry and 2e-5 to 4e-5 at 0.0025 when measured.
    """
    calc, result, hamiltonians = _silicon()
    density = np.asarray(result.density)[0]
    n = np.array(density.shape)
    volume = float(calc.system.cell.volume)
    assert abs(density.sum() * volume / density.size - 8.0) < 1e-8  # electrons per bohr^3
    index = np.array([[0, 0, 0], n // 8, [n[0] // 2, 0, 0]])
    ldos = calc.get_kpm_ldos(index / n, broadening=0.0025)
    measured = ldos.density_at(_gap(hamiltonians))[0]
    expected = density[tuple(index.T)]
    np.testing.assert_allclose(measured, expected, rtol=1e-4, atol=0)


@functools.lru_cache(maxsize=1)
def _spinor():
    calc = Calculator.from_keywords(pseudo_dir=PSEUDO, announce=False, conv_thr=1e-10,
                                    **{**SILICON, "qe": {"system": {"noncolin": True}}})
    result = calc.get_scf()
    from defuqumat.workflows.nscf import fixed_density_hamiltonians

    return calc, fixed_density_hamiltonians(calc.calculation, result.density)[0]


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_the_spin_trace_under_a_uniform_field(axis):
    """``H0 + b sigma_axis``: ``Tr[sigma_axis T_n]`` is the scalar trace at ``c - b`` minus at ``c + b``.

    The field shifts the two spin states of every level by ``+b`` and ``-b``
    along ``axis``, so the exact answer comes from two scalar traces with shifted
    centres; the other two components are zero. The stochastic spin moments
    (R = 64) must give it to well within their noise: at b = 0.05 Ry the signal's
    rms was 4.4 against a noise of 1.2, so the field is 0.5 Ry.
    """
    import equinox as eqx
    import jax.numpy as jnp

    calc, hamiltonians = _silicon()[0], _silicon()[2]
    spinor_calc, spinor = _spinor()
    b = 0.5
    potential = spinor.potential
    field = [jnp.zeros_like(potential[0])] * 3
    field[axis] = jnp.full_like(potential[0], b)
    magnetic = eqx.tree_at(lambda h: (h.potential, h.sticks, h.potential_wave), spinor,
                           (jnp.stack([potential[0]] + field), None, None),
                           is_leaf=lambda x: x is None)
    centre, half, _ = spectral_interval(hamiltonians)
    half = half + 2 * b
    weights = np.asarray(calc.calculation.system.kpoints.weights)
    shifted = [kpm_moments(hamiltonians, weights, c, half, 30, trace="exact")[0, 0]
               for c in (centre - b, centre + b)]
    expected = 0.5 * (shifted[0] - shifted[1])
    raw = np.asarray(spin_moments(magnetic, centre, half, jax.random.PRNGKey(4), 30, 64))
    live = np.asarray(jnp.sum(live_mask(magnetic), axis=-1), dtype=float)
    spinor_weights = np.asarray(spinor_calc.calculation.system.kpoints.weights)
    measured = np.einsum("k,kirn->in", spinor_weights * live, raw) / 64
    scale = np.sqrt(np.mean(expected[1:] ** 2))
    assert np.sqrt(np.mean((measured[axis] - expected) ** 2)) < 0.1 * scale
    for other in {0, 1, 2} - {axis}:
        assert np.sqrt(np.mean(measured[other] ** 2)) < 0.1 * scale


def test_the_fermi_level_of_a_metal_from_the_count():
    """Aluminium: the level where the KPM count reaches three electrons is the SCF's, within the noise."""
    calc = Calculator.from_keywords(
        lattice="fcc", a=4.05, atoms=[("Al", (0.0, 0.0, 0.0))], pseudos={"Al": "Al.pz-vbc.UPF"},
        pseudo_dir=PSEUDO, ecut=204.1, kmesh=(4, 4, 4), kshift=(1, 1, 1),
        smearing=0.5, announce=False, conv_thr=1e-10)
    scf = calc.get_scf()
    dos = calc.get_kpm_dos(broadening=0.01, n_random=32)
    level = dos.fermi_level()
    noise = dos.count_error_at(level) / max(float(np.interp(level, dos.energies, dos.dos)), 1e-3)
    assert abs(level - scf.fermi_energy) < 3.0 * noise + 0.01, (level, scf.fermi_energy, noise)


SILICON8 = dict(
    lattice="cubic", a=5.398, pseudos={"Si": "Si.pz-vbc.UPF"}, ecut=163.3, kmesh=(1, 1, 1),
    atoms=[("Si", p) for p in [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0),
                               (.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)]],
)


def test_unfolding_a_pristine_supercell_is_the_primitive_block():
    """Cubic Si8 unfolded to fcc k-points: the primitive plane waves are an invariant block of H.

    Three checks that do not use the recursion: the subset holds as many plane
    waves as the two-atom cell's sphere at the same k and cutoff; the dense
    supercell matrix couples the subset to the rest only at round-off; and the
    unfolded moments are the exact trace of the dense block.
    """
    from defuqumat.kpm.traces import primitive_subset, supercell_matrix
    from defuqumat.system.kpoints import KPoints, for_spin
    from defuqumat.workflows.nscf import fixed_density_hamiltonians

    primitive, _, _ = _silicon()
    supercell = Calculator.from_keywords(pseudo_dir=PSEUDO, announce=False, conv_thr=1e-10,
                                         **SILICON8)
    scf = supercell.get_scf()
    prim_at = np.asarray(primitive.system.cell.at)
    kp = np.array([[0.5, 0.0, 0.5], [0.1, 0.2, 0.3]])
    spectrum = supercell.get_kpm_unfolded(prim_at, kp, n_moments=40)

    def at(calc, cell_of):
        bg = 2.0 * np.pi * np.linalg.inv(prim_at).T
        path = KPoints.from_cartesian((kp @ bg) / float(cell_of.tpiba), np.ones(len(kp)))
        return calc.calculation.at_kpoints(for_spin(path, 1))

    moved = at(supercell, supercell.system.cell)
    hamiltonian = fixed_density_hamiltonians(moved, scf.density)[0]
    subset = primitive_subset(hamiltonian,
                              supercell_matrix(np.asarray(supercell.system.cell.at), prim_at))
    small = fixed_density_hamiltonians(at(primitive, primitive.system.cell),
                                       _silicon()[1].density)[0]
    counts = np.asarray(small.mask).sum(axis=-1)
    assert counts.min() > 50
    np.testing.assert_array_equal(subset.sum(axis=-1), counts)
    exact = np.zeros((len(kp), 40))
    for ik in range(len(kp)):
        matrix = np.asarray(hamiltonian.matrix(ik))
        inside, outside = subset[ik], np.asarray(hamiltonian.mask[ik]) & ~subset[ik]
        assert np.abs(matrix[np.ix_(inside, outside)]).max() < 1e-8
        values = np.linalg.eigvalsh(matrix[np.ix_(inside, inside)])
        x = (values - spectrum.centre) / spectrum.half_width
        exact[ik] = np.cos(np.arange(40)[:, None] * np.arccos(x)[None, :]).sum(axis=1)
    np.testing.assert_allclose(spectrum.moments[0], exact, rtol=0, atol=1e-9 * exact[:, :1])
