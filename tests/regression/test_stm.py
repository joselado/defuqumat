"""STM images against ``pp.x``, against a sum rule, and against the physics.

``PLAN.md`` P65. Elk's task 162 (``wfplot.f90``) and QE's ``PP/src/stm.f90``.
The machinery -- the sampler, the plane, the weights, the spin projection -- is
checked on objects in ``tests/unit/test_stm_machinery.py``; what is here needs
a converged cell.

Four kinds of check, in the order they are worth reading:

* ``pp.x`` on the same cell and the same pseudopotential, whose ``plot_num = 5``
  at ``sample_bias = 0`` is exactly the zero-bias image times ``degauss``;
* the sum rule ``int rho_STM d3r = D(E_F)``, which shares nothing with the
  reference and closes inside the package;
* an antiferromagnet whose two spin projections are each other's mirror while
  its charge image is flat, and a noncollinear chain where turning the tip
  selects a different sublattice -- the spin-polarized image, which neither
  ``pp.x`` nor Elk's task 162 computes;
* graphite, whose STM image famously shows one of its two sublattices, and the
  exponential decay of the tunnelling density into the vacuum.
"""

from functools import lru_cache
from pathlib import Path

import jax
import numpy as np
import pytest

from defuqumat import Calculator

pytestmark = pytest.mark.slow

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Every cell here compiles its own SCF stack and XLA keeps them all.

    ``CLAUDE.md``'s rule for a file that sweeps cells: the results stay cached
    below, only the executables are dropped.
    """
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(case: str, conv_thr: float | None = None):
    """The ground state, with ``conv_thr`` overriding the input where it is given.

    **A number read off an STM plane converges more slowly than the residual
    ``conv_thr`` measures**, and by enough to matter. Measured on
    ``h-chain-afm``, whose input asks for 1e-11, as the *relative* gap between
    the two antiferromagnetic sublattices, which symmetry makes exactly zero:

    | ``conv_thr`` | iterations | accuracy | sublattice gap |
    |---|---|---|---|
    | 1e-09 | 11 | 1.50e-10 | 1.60e-05 |
    | 1e-11 | 12 | 7.70e-13 | 1.39e-05 |
    | 1e-13 | 14 | 3.24e-15 | 3.43e-06 |
    | 1e-15 | 15 | 2.78e-16 | 4.28e-07 |

    The density residual is already at 7.7e-13 in the second row while the
    quantity read off the plane is still moving in its sixth digit, so a test
    asserting the sublattice symmetry to 1e-5 has to say which convergence it
    is asserting it at rather than inherit the input's. Three more iterations
    buy two orders, which is why the tight row is affordable here.
    """
    calculator = Calculator.from_file(CASES / f"{case}.in", pseudo_dir=PSEUDO)
    calculator.get_scf(**({} if conv_thr is None else {"conv_thr": conv_thr}))
    return calculator


def _read_filplot(path: Path) -> np.ndarray:
    """``Modules/plot_io.f90``'s text dump of the dense grid, as ``(n1, n2, n3)``.

    The header is fixed-shape apart from the species and atom tables, whose
    lengths the second line gives, and the data is Fortran-ordered.
    """
    lines = path.read_text().splitlines()
    n1x, n2x, _, _, _, n3, nat, ntyp = (int(x) for x in lines[1].split())
    values = np.array(" ".join(lines[4 + ntyp + nat:]).split(), dtype=float)
    return np.transpose(values[:n1x * n2x * n3].reshape((n3, n2x, n1x)), (2, 1, 0))


# --------------------------------------------------------------------------
# against pp.x
# --------------------------------------------------------------------------


def test_the_tunnelling_density_matches_pp_x():
    """``plot_num = 5``, ``sample_bias = 0``, on QE's own fcc aluminium.

    Three things make this an exact comparison rather than an approximate one.
    ``stm.f90`` divides by the cell volume and *not* by ``degauss``, so its
    image is the zero-bias one times the width. It uses the run's own smearing,
    which here is Marzari-Vanderbilt, not the Gaussian this defaults to. And it
    truncates the band sum at three widths either side of the window
    (``first_band``/``last_band``), which is worth 0.4 per cent of the image on
    this cell -- so ``band_cutoff = 3`` is what makes the two the same sum, and
    leaving it off is what makes ours the complete one.
    """
    reference = _read_filplot(CASES / "reference.stm.al-metal")
    calculator = _converged("al-metal")
    image = calculator.get_stm(height=0.0, width=0.05,
                               smearing="marzari-vanderbilt", band_cutoff=3.0)
    ours = np.asarray(image.density) * image.width
    assert ours.shape == reference.shape
    assert np.abs(ours - reference).max() < 1.0e-8


def test_the_band_truncation_is_the_whole_of_the_difference():
    """Without QE's cutoff the two differ, and by the states it drops.

    The pair to the test above: it is the truncation that separates them and
    not anything in the assembly. **The complete sum is the smaller one**,
    which is the direction that says what QE's three widths are really doing
    here -- Marzari-Vanderbilt's delta is negative for ``x > sqrt(2)``, so the
    states past the cutoff carry *negative* weight and dropping them raises the
    image. It is the same objection ``PLAN.md`` P52 makes to using a
    non-positive delta for a quantity that has a sign, one order out: a
    tunnelling density built from cold smearing is not positive by
    construction, and that is why the default here is a Gaussian.
    """
    reference = _read_filplot(CASES / "reference.stm.al-metal")
    calculator = _converged("al-metal")
    whole = calculator.get_stm(height=0.0, width=0.05,
                               smearing="marzari-vanderbilt")
    ours = np.asarray(whole.density) * whole.width
    gap = np.abs(ours - reference).max() / reference.max()
    assert 1.0e-3 < gap < 1.0e-2
    assert ours.mean() < reference.mean()

    # and a Gaussian, which is positive everywhere, loses nothing to the cutoff
    positive = calculator.get_stm(height=0.0, width=0.05, smearing="gaussian")
    assert np.asarray(positive.density).min() > 0.0


def test_the_image_integrates_to_the_density_of_states():
    """``int rho_STM d3r = D(E_F)``: the sum rule, against ``compute_dos``.

    Nothing of the reference is involved. It is the check that would catch a
    factor of two in the spin degeneracy, a missing ``1/Omega``, or a delta
    that is not normalised.
    """
    from defuqumat.workflows.dos import compute_dos

    calculator = _converged("al-metal")
    scf = calculator.get_scf()
    image = calculator.get_stm(height=0.0, width=0.05,
                               smearing="marzari-vanderbilt")
    dos = compute_dos(scf.eigenvalues_by_spin,
                      calculator.system.kpoints.weights,
                      np.array([scf.fermi_energy]),
                      "marzari-vanderbilt", degauss=0.05)
    assert image.integral == pytest.approx(float(np.asarray(dos.dos).ravel()[0]),
                                           rel=1e-10)


def test_a_denser_grid_re_solves_the_bands_and_moves_the_fermi_level():
    """The knob every docstring here leads with, and it had no test.

    A delta at the Fermi level on the SCF's own handful of k-points is a sum
    over the few bands that happen to be near it, so ``grid=`` is the
    convergence parameter of the whole quantity. Two things have to be true and
    only the first is obvious: the bands are re-solved *there*, and the Fermi
    level is recomputed *there* -- it is the level of that k-set that makes the
    cell neutral, and the delta has to sit on it for the sum rule to hold. The
    check is the sum rule against the new k-set's own density of states, which
    fails if either half is stale.
    """
    from defuqumat.workflows.dos import compute_dos
    from defuqumat.workflows.nscf import denser_grid

    calculator = _converged("al-metal")
    scf = calculator.get_scf()
    image = calculator.get_stm(height=0.0, width=0.05,
                               smearing="marzari-vanderbilt", grid=(6, 6, 6))
    assert image.grid == (6, 6, 6)
    assert image.energy != scf.fermi_energy      # recomputed, not carried over
    assert abs(image.energy - scf.fermi_energy) < 0.05

    kpoints = denser_grid(calculator.system, (6, 6, 6))
    states = calculator.get_nscf(kpoints=kpoints)
    dos = compute_dos(states.eigenvalues, states.kpoints.weights,
                      np.array([image.energy]), "marzari-vanderbilt", degauss=0.05)
    assert image.integral == pytest.approx(float(np.asarray(dos.dos).ravel()[0]),
                                           rel=1e-8)


def test_the_plane_is_read_out_of_the_grid_exactly():
    """Sampling on the grid's own points returns the grid, on a real cell."""
    from defuqumat.basis.builder import build_basis
    from defuqumat.basis.sample import sample_field

    calculator = _converged("al-metal")
    image = calculator.get_stm(height=0.0, width=0.05)
    dense = build_basis(calculator.system).dense
    axes = [np.arange(n) / n for n in dense.grid]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape((-1, 3))
    back = sample_field(np.asarray(image.density), dense, points).reshape(dense.grid)
    assert np.abs(back - np.asarray(image.density)).max() < 1.0e-12


# --------------------------------------------------------------------------
# spin-polarized STM
# --------------------------------------------------------------------------


#: The plane both hydrogen chains are imaged on: it contains the chain axis and
#: stands off it by 0.15 of the cell's ``a``, which is 1.8 bohr.
_CHAIN_PLANE = ((0.15, 0.0, 0.0), (0.15, 0.30, 0.0), (0.15, 0.0, 1.0))


def test_an_antiferromagnet_is_flat_in_charge_and_alternates_in_spin():
    """The point of a spin-polarized tip, on the cheapest cell that has one.

    Both atoms of the antiferromagnetic hydrogen chain carry the same charge,
    so an ordinary STM image cannot tell them apart -- and it does not, to six
    figures. A tip magnetized along ``+z`` sees one of them and a tip along
    ``-z`` sees the other, by equal and opposite amounts, which is the
    antiferromagnetic symmetry reproducing itself rather than being imposed.
    """
    # 1e-15 rather than the input's 1e-11: the sublattice gap this asserts is a
    # convergence artifact all the way down, and at the input's value it is
    # 1.39e-5 against the 1e-5 below. See :func:`_converged` for the table.
    calculator = _converged("h-chain-afm", conv_thr=1.0e-15)
    common = dict(plane=_CHAIN_PLANE, shape=(4, 32), width=0.10)
    charge = calculator.get_stm(**common)
    up = calculator.get_stm(spin="up", **common)
    down = calculator.get_stm(spin="down", **common)

    # the two channels are a partition of the charge, exactly
    assert np.abs(up.values + down.values - charge.values).max() < 1.0e-14

    first, second = 0, 16          # the atoms sit at z = 0 and z = 1/2
    assert charge.values[0, first] == pytest.approx(charge.values[0, second],
                                                    rel=1e-5)
    asymmetry = ((up.values - down.values) / charge.values)[0]
    assert asymmetry[first] > 0.02
    assert asymmetry[second] == pytest.approx(-asymmetry[first], rel=1e-3)


def test_a_noncollinear_tip_selects_the_sublattice_it_points_at():
    """Four moments, each 90 degrees from the last, and a tip that turns.

    The strongest statement here has no reference in it: a tip along ``x``
    sees atoms 1 and 3 with opposite sign and atoms 2 and 4 **not at all**,
    because their moments are perpendicular to it -- and a tip along ``z`` sees
    none of the four, every moment lying in the plane. The contrast has the
    same magnitude in every direction, which says the four moments are equal
    without anything having imposed it.
    """
    calculator = _converged("h-chain-90deg")
    common = dict(plane=_CHAIN_PLANE, shape=(4, 64), width=0.10)
    charge = calculator.get_stm(**common)
    atoms = [0, 16, 32, 48]        # z = 0, 1/4, 1/2, 3/4

    def contrast(direction):
        image = calculator.get_stm(spin=direction, **common)
        return ((2.0 * image.values - charge.values) / charge.values)[0, atoms]

    flat = charge.values[0, atoms]
    assert np.ptp(flat) / flat.mean() < 1.0e-4

    along_x = contrast((1, 0, 0))
    assert along_x[0] > 0.05
    assert along_x[2] == pytest.approx(-along_x[0], rel=1e-3)
    assert abs(along_x[1]) < 1.0e-4 and abs(along_x[3]) < 1.0e-4

    along_y = contrast((0, 1, 0))
    assert along_y[1] == pytest.approx(along_x[0], rel=1e-3)
    assert along_y[3] == pytest.approx(-along_x[0], rel=1e-3)
    assert abs(along_y[0]) < 1.0e-4 and abs(along_y[2]) < 1.0e-4

    assert np.abs(contrast((-1, 0, 0)) + along_x).max() < 1.0e-8
    assert np.abs(contrast((0, 0, 1))).max() < 1.0e-4


def test_a_spinor_image_obeys_the_same_sum_rule():
    """The check that catches a factor of two on the noncollinear path.

    A spinor band holds one electron where a scalar one holds two, and every
    ``KPoints`` constructor applies the unpolarized degeneracy unconditionally
    (``PLAN.md`` P51's ``for_spin`` trap). Neither the contrast test above nor
    ``up + down = charge`` can see that factor: both are ratios. This is the
    same ``int rho_STM = D(E_F)`` the aluminium test makes, on the branch where
    the degeneracy is different.
    """
    from defuqumat.workflows.dos import compute_dos

    calculator = _converged("h-chain-90deg")
    scf = calculator.get_scf()
    image = calculator.get_stm(plane=_CHAIN_PLANE, shape=(2, 2), width=0.10)
    dos = compute_dos(scf.eigenvalues_by_spin,
                      calculator.system.kpoints.weights,
                      np.array([scf.fermi_energy]), "gaussian", degauss=0.10)
    assert image.integral == pytest.approx(float(np.asarray(dos.dos).ravel()[0]),
                                           rel=1e-10)


@pytest.mark.parametrize("case", ["si2-us", "si2-paw"])
def test_a_soft_dataset_image_carries_its_augmentation_charge(case):
    """``becsum`` is rebuilt from the tunnelling weights, not left behind.

    A window wide enough to hold whole states integrates to the number of
    states in it, because ``<psi|S|psi> = 1`` -- and the ``S`` is where the
    augmentation charge is. Dropping it, or rebuilding ``becsum`` from the
    ground state's occupations instead of the tip's, moves this number by the
    augmentation's share of the norm, which for silicon is percent-sized.

    ``pp.x`` cannot be the reference here: ``stm.f90`` sums ``|psi|^2`` and
    never calls ``addusdens``, so it is norm-conserving only.
    """
    from defuqumat.stm.image import tunnelling_weights

    calculator = _converged(case)
    scf = calculator.get_scf()
    energy = scf.homo if scf.fermi_energy is None else scf.fermi_energy
    image = calculator.get_stm(height=0.0, shape=(4, 4), bias=-0.5,
                               energy=energy, width=1.0e-4)
    expected = tunnelling_weights(
        np.asarray(scf.eigenvalues_by_spin),
        np.asarray(calculator.system.kpoints.weights),
        energy=energy, width=1.0e-4, bias=-0.5,
    ).sum()
    assert image.integral == pytest.approx(float(expected), rel=2e-6)


# --------------------------------------------------------------------------
# the physics, on a surface
# --------------------------------------------------------------------------


def test_graphite_shows_one_of_its_two_sublattices():
    """The classic STM result, and it is a statement about stacking.

    In AB-stacked graphite one sublattice of the surface layer sits above an
    atom of the layer below and the other above a hollow, so the two are
    inequivalent and only one is bright at the Fermi level. Both are carbon and
    both carry the same charge; what differs is the density of states *at the
    Fermi level*, which is what an STM measures and a total-charge plot does
    not.
    """
    calculator = _converged("graphene-bilayer")
    image = calculator.get_stm(height=0.85, shape=(48, 48))
    values = image.values

    def at(fx, fy):
        return values[int(round(fx * 48)) % 48, int(round(fy * 48)) % 48]

    over_a, over_b = at(0.0, 0.0), at(2 / 3, 1 / 3)
    hollow = at(1 / 3, 2 / 3)
    assert over_b / over_a > 1.3          # the two sublattices are not alike
    assert over_a / hollow > 100.0        # and both are far above the hollow


def test_the_tunnelling_density_decays_exponentially_into_the_vacuum():
    """Tersoff-Hamann's own statement, and it involves none of the assembly.

    A state at energy ``E`` under a vacuum level ``V`` decays as
    ``exp(-sqrt(V - E + k_par^2) z)`` and its square twice as fast, so the
    tunnelling density falls off log-linearly at a rate of **at least**
    ``2 sqrt(V - E_F)``. The bound is one-sided because the states at graphene's
    Fermi level sit at ``K`` rather than at ``Gamma``, and their in-plane
    momentum steepens the decay -- 1.86 measured here against 0.98 for
    ``k_par = 0`` and 2.92 for ``|K|``.
    """
    calculator = _converged("graphene-bilayer")
    scf = calculator.get_scf()
    image = calculator.get_stm(height=0.85, shape=(8, 8))

    density = np.asarray(image.density)
    potential = np.asarray(scf.potential)
    if potential.ndim == 4:
        potential = potential[0]
    length = float(np.asarray(calculator.system.cell.at)[2, 2])
    z = np.arange(density.shape[-1]) / density.shape[-1] * length

    planar = density.mean(axis=(0, 1))
    vacuum_level = float(potential.mean(axis=(0, 1))[(z > 18.5) | (z < 1.5)].max())
    kappa = np.sqrt(vacuum_level - scf.fermi_energy)

    band = (z > 15.5) & (z < 18.5)
    logarithm = np.log(planar[band])
    fit = np.polyfit(z[band], logarithm, 1)
    assert -fit[0] > 2.0 * kappa                     # no slower than the bound
    # and it really is a line: two decay constants superpose here (the layer
    # below shows through, and the states at E_F carry a spread of in-plane
    # momenta), so what is claimed is a straight log and not a single exponent.
    residual = logarithm - np.polyval(fit, z[band])
    assert 1.0 - residual.var() / logarithm.var() > 0.999   # measured 0.9992 over five orders of decay


# --------------------------------------------------------------------------
# a sample in an applied field (I118)
# --------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _in_a_field(regime: str, b: float):
    """``h-sheet-field.in`` under a uniform field ``b`` along z, in Ry.

    A paramagnetic metal: the moment is the field's doing (0.0079 per 0.001 Ry
    of field, linear to 2.5 per cent up to 0.01 Ry), so the field-free run is
    the unpolarized one and the weak-field limit is analytic.
    """
    calculator = Calculator.from_file(CASES / "h-sheet-field.in",
                                      pseudo_dir=PSEUDO)
    system = calculator.system
    if regime == "noncollinear":
        system = system.with_spin(4)
    # A calculator built from a System does not read the input's &electrons,
    # so the input's conv_thr is given again, as the calculator's default so
    # that the image and the spectrum find the same SCF in the cache: at the
    # package default the stored states and a re-solve on the same k-set
    # differ by 5.7e-3 at B = 0.01.
    calculator = Calculator(system.with_b_field((0.0, 0.0, b)),
                            calculator.pseudos, conv_thr=1e-12)
    calculator.get_scf()
    return calculator


def _field_tip(regime):
    return "up" if regime == "collinear" else (0.0, 0.0, 1.0)


def _field_readings(calculator, regime, energies):
    """The charge and the spin-up image, and the same pair of spectra at two
    tip points. A run with no magnetization refuses a spin tip, and its spin
    channel is half the charge exactly (``project_spin`` at ``m = 0``)."""
    shared = dict(height=0.72, shape=(12, 12))
    tip = np.array([[0.0, 0.0, 0.72], [0.5, 0.5, 0.72]])
    charge = calculator.get_stm(**shared)
    spectrum = calculator.get_sts(energies=energies, tip=tip)
    try:
        spin = calculator.get_stm(spin=_field_tip(regime), **shared).values
        spin_spectrum = calculator.get_sts(
            energies=energies, tip=tip, spin=_field_tip(regime)).values
    except NotImplementedError:
        spin = 0.5 * np.asarray(charge.values)
        spin_spectrum = 0.5 * np.asarray(spectrum.values)
    return charge, (np.asarray(charge.values), np.asarray(spin),
                    np.asarray(spectrum.values), np.asarray(spin_spectrum))


def _relative(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("regime", ["collinear", "noncollinear"])
def test_the_image_in_a_weak_field_tends_to_the_field_free_one(regime):
    """I118. The image and the spectrum of a paramagnet in a field ``B``.

    The charge channel is even in ``B`` and the spin channel odd, so their
    distance from the field-free reading falls as ``B^2`` and as ``B``. This
    checks the weak-field limit and nothing about which states or which level
    are used: a tip at ``E_F(0)`` or states without the field give the same
    exponents (1.957 to 1.986 and 1.005 to 1.009; 1.994 to 2.000 and 1.006 to
    1.014), so that is what :func:`test_the_states_in_a_field_are_the_run_s_own`
    checks. The tolerance is for this unshifted 4x4x1 mesh: on a shifted one
    (``4 4 1 1 1 0``) the spin exponent reads 0.658 at this pair and tends to 1
    only as ``B`` falls (0.906 at 1.25e-4). Measured (notes/incompat/I118.md), between
    ``B = 5e-4`` and ``1e-3`` Ry, the exponents are 1.994 and 1.023 for the
    image and 1.996 and 1.008 for the spectrum, in both spin regimes; between
    ``1e-3`` and ``2e-3`` they are 1.976, 1.039, 1.985 and 1.008, so the
    deviation from 2 and 1 is the next order and shrinks with ``B``. The floor
    (the image at ``conv_thr = 1e-10`` against ``1e-12``) is 2e-6 at ``B = 0``,
    far under the smallest distance read, 3.7e-4.
    """
    ef0 = float(_in_a_field(regime, 0.0).get_scf().fermi_energy)
    energies = ef0 + np.linspace(-0.1, 0.1, 9)
    zero_image, zero = _field_readings(_in_a_field(regime, 0.0), regime, energies)
    assert zero_image.field_scale is None

    distances = []
    for b in (5e-4, 1e-3):
        image, readings = _field_readings(_in_a_field(regime, b), regime,
                                          energies)
        assert image.field_scale == 1.0
        distances.append([_relative(r, r0) for r, r0 in zip(readings, zero)])
    exponents = np.log2(np.array(distances[1]) / np.array(distances[0]))
    np.testing.assert_allclose(exponents, [2.0, 1.0, 2.0, 1.0], atol=0.05)
    assert distances[0][0] < 1e-3          # 3.7e-4: the image tends to B = 0


def test_the_states_in_a_field_are_the_run_s_own():
    """I118. The stored states' image against a re-solve on the same k-set.

    With the field the SCF ended with, the two agree to the band solve: 9.0e-7
    in the charge and 4.0e-6 in the spin-up image at ``B = 0.01`` Ry, and the
    re-solve's Fermi level is the SCF's to 1.2e-9 Ry. The same re-solve with the
    field dropped from the Hamiltonian, at the same density, is off by 7.1e-2
    and 3.3e-1, and the field states read at the field-free Fermi level are
    off by 4.06e-2 in the charge, which is the entry's own worry and does not
    happen.
    """
    from types import SimpleNamespace

    calculator = _in_a_field("collinear", 0.01)
    scf = calculator.get_scf()
    shared = dict(height=0.72, shape=(12, 12))
    resolve = dict(grid=(4, 4, 1), conv_thr=1e-11)
    stored = calculator.get_stm(**shared)
    stored_spin = calculator.get_stm(spin="up", **shared)
    again = calculator.get_stm(**shared, **resolve)
    again_spin = calculator.get_stm(spin="up", **shared, **resolve)
    assert _relative(again.values, stored.values) < 1e-5
    assert _relative(again_spin.values, stored_spin.values) < 2e-5
    assert abs(again.energy - scf.fermi_energy) < 1e-8

    off = Calculator(calculator.system.with_b_field((0.0, 0.0, 0.0)),
                     calculator.pseudos)
    fieldless = SimpleNamespace(
        density=scf.density, eigenvalues_by_spin=scf.eigenvalues_by_spin,
        wavefunctions=scf.wavefunctions, fermi_energy=scf.fermi_energy,
        homo=scf.homo, lumo=scf.lumo, magnetic_field=None, field_scale=1.0)
    from defuqumat.workflows.stm import run_stm

    dropped_spin = run_stm(off.system, off.pseudos, fieldless, spin="up",
                           **shared, **resolve)
    assert _relative(dropped_spin.values, stored_spin.values) > 0.1

    # the entry's own worry: the field states read at the field-free level
    level = float(_in_a_field("collinear", 0.0).get_scf().fermi_energy)
    misplaced = calculator.get_stm(energy=level, **shared)
    assert _relative(misplaced.values, stored.values) > 1e-2   # 4.06e-2
