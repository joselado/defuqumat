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


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Free the compiled programs after each test (``.claude/rules/memory-design.md``).

    The KPM files compile a recursion per cell and per moment count, and in the
    one-process fast gate they sat in memory beside the files that follow; the
    gate was killed at its 6 GB cap two files later (2026-11-07).
    """
    yield
    import jax

    jax.clear_caches()

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
    (R = 64) must give it to within their noise. At b = 0.5 Ry the signal's rms
    is about 10 and the noise about 1.1 on every axis (measured: 0.11 of the
    signal on y), so the bound is 0.2 of the signal: a sign error would read 2
    and a wrong axis 1, which is what the test tells apart.
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
    assert np.sqrt(np.mean((measured[axis] - expected) ** 2)) < 0.2 * scale
    for other in {0, 1, 2} - {axis}:
        assert np.sqrt(np.mean(measured[other] ** 2)) < 0.2 * scale


@functools.lru_cache(maxsize=1)
def _zincblende_selenium():
    """A nonmagnetic spinor with spin-orbit coupling and no inversion, on a shifted mesh.

    Two Se atoms on the zincblende sites (Td), fully relativistic, at 5 Ry,
    since the zero the tests read is one of symmetry and holds at any cutoff.
    The shifted 2x2x2 mesh has no time-reversal-invariant point, where Kramers
    pairs would make the spin trace vanish at each k on their own.
    """
    calc = Calculator.from_keywords(
        lattice="fcc", a=5.6, atoms=[("Se", (0.0, 0.0, 0.0)), ("Se", (0.25, 0.25, 0.25))],
        pseudos={"Se": "Se.rel-pbe-nc-sg15.UPF"}, pseudo_dir=PSEUDO, ecut=68.0,
        kmesh=(2, 2, 2), kshift=(1, 1, 1), spin_orbit=True, smearing=0.5,
        announce=False, conv_thr=1e-6)
    calc.get_scf()
    return calc


@pytest.mark.slow
def test_the_spin_dos_of_a_spinor_without_inversion_is_zero():
    """Time reversal makes ``sum_k Tr[sigma_i delta(E - H_k)]`` zero; the wedge's weights do not.

    ``Tr[sigma_i T_n(H_k)]`` is odd under time reversal and turns as an axial
    vector, so the zone sum vanishes for a nonmagnetic crystal, while the
    weighted sum over the wedge is the trace of one representative per star
    and does not (audit 2026-11-15, ``kpm`` F1). The exact trace makes the zero
    a round-off zero, and the scale it is read against is the scalar DOS.
    Before the fix the spin DOS read 2.3 states/Ry against a DOS peak of 19.9.
    Slow: 25 s, most of it the spin-orbit SCF.
    """
    calc = _zincblende_selenium()
    assert calc.calculation.system.kpoints.reduced
    dos = calc.get_kpm_dos(n_moments=20, trace="exact", spin_resolved=True,
                           control_variate=False)
    assert dos.n_random == 0
    assert np.max(np.abs(dos.spin_dos)) < 1e-8 * np.max(np.abs(dos.dos))
    # the moments themselves, which the kernel only damps
    assert np.max(np.abs(dos.spin_moments)) < 1e-8 * np.max(np.abs(dos.moments))


def test_the_reconstructed_spin_dos_follows_the_new_moments():
    """``reconstructed`` rebuilds ``spin_dos`` on the new grid and moment count (``kpm`` F2).

    Host-side only: a result built by hand from known moments, so no recursion runs.
    """
    from defuqumat.kpm.dos import KPMDensityOfStates, _evaluate

    rng = np.random.default_rng(0)
    n = 40
    decay = np.exp(-0.1 * np.arange(n))
    samples = (rng.normal(size=(1, 3, n)) * decay)
    samples[..., 0] = 1.0
    spin = rng.normal(size=(3, n)) * decay
    energies = np.linspace(-0.8, 0.8, 161)
    dos, _ = _evaluate(samples.mean(axis=1), "jackson", n, 0.0, 1.0, energies)
    full, _ = _evaluate(spin, "jackson", n, 0.0, 1.0, energies)
    result = KPMDensityOfStates(
        energies=energies, dos=dos[0], integrated=dos[0], dos_error=0 * dos[0],
        samples=samples, centre=0.0, half_width=1.0, kernel="jackson", n_moments=n,
        broadening=0.0, reference_energy=0.0, trace="stochastic", n_random=3, seed=0,
        nspin=1, nelec=2.0, spin_dos=full, spin_moments=spin)
    coarse = result.reconstructed(energies=energies[::3], n_moments=20)
    expected, _ = _evaluate(spin, "jackson", 20, 0.0, 1.0, energies[::3])
    assert coarse.spin_dos.shape == (3, energies[::3].size)
    np.testing.assert_allclose(coarse.spin_dos, expected, rtol=0, atol=1e-12)
    assert not np.allclose(coarse.spin_dos, full[:, ::3], atol=1e-3)


def test_the_fermi_level_of_a_metal_from_the_count():
    """Aluminium: the level where the KPM count reaches three electrons is the SCF's, within the noise."""
    calc = Calculator.from_keywords(
        lattice="fcc", a=4.05, atoms=[("Al", (0.0, 0.0, 0.0))], pseudos={"Al": "Al.pz-vbc.UPF"},
        pseudo_dir=PSEUDO, ecut=204.1, kmesh=(4, 4, 4), kshift=(1, 1, 1),
        smearing=0.5, announce=False, conv_thr=1e-10)
    scf = calc.get_scf()
    dos = calc.get_kpm_dos(broadening=0.01, n_random=32, control_variate=False)
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
    np.testing.assert_allclose(spectrum.moments[0], exact, rtol=0, atol=1e-9 * float(exact[:, 0].max()))


def test_an_odd_moment_count_is_refused_before_the_unfolding():
    """The doubling gives an even count only; an odd one was a shape error deep inside (``kpm`` F4)."""
    from defuqumat.kpm.traces import run_kpm_unfolded

    with pytest.raises(ValueError, match="n_moments must be even"):
        run_kpm_unfolded(None, None, None, np.eye(3), np.zeros((1, 3)), n_moments=41)


def _fcc_cell():
    from defuqumat.system.cell import Cell

    half = 5.6 / 2.0
    return Cell.from_vectors(np.array([[-half, 0.0, half], [0.0, half, half], [-half, half, 0.0]]))


def test_an_explicit_wedge_is_refused_and_an_explicit_whole_grid_is_kept():
    """A wedge written as a K_POINTS list is not a whole zone, with unequal or equal weights (audit 2026-11-20 #11).

    ``whole_grid_calculation`` read only ``KPoints.reduced``, which an explicit
    list never sets, so a written-out wedge passed unchanged and its weights
    summed the spin, per-orbital, per-point and velocity traces over part of
    the zone. Two wedges cut by time reversal alone: 3x3x3 unshifted (14
    points, weights 1 and 2, which ``is_reduced``'s weight spread sees) and
    2x2x2 shifted (4 points of weight 2, which it does not). Both are now
    refused; the whole shifted grid written as a list is kept as it is, and
    an automatic wedge still moves to its grid.
    """
    from types import SimpleNamespace

    from defuqumat.kpm.traces import whole_grid_calculation
    from defuqumat.system.kpoints import KPoints, covers_a_whole_grid, is_reduced

    cell = _fcc_cell()
    identity = np.eye(3, dtype=int)[None]

    def calculation(kpoints):
        moved = []
        return SimpleNamespace(system=SimpleNamespace(kpoints=kpoints, cell=cell, nspin=4),
                               at_kpoints=lambda k: moved.append(k) or moved), moved

    def written_out(kpoints):
        return KPoints.from_cartesian(np.asarray(kpoints.coords), np.asarray(kpoints.weights))

    unequal = written_out(KPoints.automatic((3, 3, 3), (0, 0, 0), cell, rotations=identity))
    equal = written_out(KPoints.automatic((2, 2, 2), (1, 1, 1), cell, rotations=identity))
    assert (unequal.nk, equal.nk) == (14, 4)
    assert is_reduced(unequal) and not is_reduced(equal)
    assert not (unequal.reduced or equal.reduced) and unequal.grid is equal.grid is None
    for wedge in (unequal, equal):
        assert not covers_a_whole_grid(wedge, cell)
        with pytest.raises(ValueError, match="not a whole grid"):
            whole_grid_calculation(calculation(wedge)[0])
    for mesh, shift in (((2, 2, 2), (1, 1, 1)), ((3, 3, 1), (0, 0, 1)), ((1, 1, 1), (0, 0, 0))):
        whole = written_out(KPoints.automatic(mesh, shift, cell))
        assert covers_a_whole_grid(whole, cell), (mesh, shift)
        stand_in, moved = calculation(whole)
        assert whole_grid_calculation(stand_in) is stand_in and not moved
    automatic = KPoints.automatic((2, 2, 2), (1, 1, 1), cell, rotations=identity)
    assert automatic.reduced
    stand_in, moved = calculation(automatic)
    whole_grid_calculation(stand_in)
    assert len(moved) == 1 and moved[0].nk == 8


def test_spin_resolved_on_a_collinear_run_is_refused_before_the_whole_grid(monkeypatch):
    """The spinor test sits at the door of ``run_kpm_dos`` (audit 2026-11-20 #27).

    It used to sit in ``spin_resolved_dos``, after the move to the whole grid,
    the Lanczos interval, the scalar moments and the control variate, so a
    collinear call paid for the whole scalar DOS on the whole grid and then
    raised. A spinor passes the door and reaches the move.
    """
    from types import SimpleNamespace

    import defuqumat.kpm.dos as dos

    def reached(*args, **kwargs):
        raise LookupError("the whole-grid move was reached")

    monkeypatch.setattr(dos, "_on_the_whole_grid", reached)
    for nspin in (1, 2):
        with pytest.raises(ValueError, match="needs a spinor"):
            dos.run_kpm_dos(SimpleNamespace(noncolin=False, nspin=nspin), (), None,
                            spin_resolved=True)
    with pytest.raises(LookupError, match="reached"):
        dos.run_kpm_dos(SimpleNamespace(noncolin=True, nspin=4), (), None, spin_resolved=True)
