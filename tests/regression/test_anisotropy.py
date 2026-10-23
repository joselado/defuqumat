"""The magnetocrystalline anisotropy by the force theorem (P58).

Four rungs, and only the last one reaches Fortran -- which is why the first
three are here at all. Each is a statement the assembly cannot satisfy by
accident:

1. **Switch the coupling off and every direction must agree.** Without
   spin-orbit coupling the Hamiltonian is invariant under a *global* spin
   rotation, so a band energy cannot depend on where the moment points -- not
   approximately, exactly. This is the check that found the phase's one real
   bug: the density was being rotated and the *quantization axis* QE's
   ``compute_ux`` builds was not, so the gradient correction differentiated
   ``|m|`` through its own nodes. It was worth **36.8 meV** on a cell whose
   answer is zero, it survived switching spin-orbit coupling off entirely
   (which is what identified it), and no spin-orbit test could have seen it.

2. **A cubic crystal has no anisotropy between its cubic axes.** Nothing in
   the code imposes that; it comes out of the k-sum.

3. **The first-order term is small**, a fraction of a per cent of the
   anisotropy, which is the reason this phase is a diagonalisation and not an
   expectation value.

4. **QE's own committed force-theorem example**, ``PP/examples/
   ForceTheorem_example`` -- a 3-layer Co(0001) slab, PRB 90, 205409 (2014),
   whose reference output carries ``eband`` to thirteen digits for two
   directions. That case is the phase's external anchor and the reason the
   pseudopotential pair ``Co.pbe-nd-rrkjus`` / ``Co.rel-pbe-nd-rrkjus`` is
   committed.
"""

from functools import lru_cache

import jax
import numpy as np
import pytest

from defuqumat.calculator import Calculator
from defuqumat.scf.continuation import (
    direction_from_angles,
    nc_magnetization_from_lsda,
)
from defuqumat.units import RY_TO_EV
from defuqumat.workflows.anisotropy import (
    angles_from_direction,
    frozen_expectation,
    run_torque,
    cardinal_directions,
    run_anisotropy,
    run_force_theorem,
    sphere_cover,
)
from tests.conftest import GENERATED

pytestmark = [pytest.mark.regression]

#: ``pw.x`` with ``lforcet``, from the committed reference outputs of
#: ``PP/examples/ForceTheorem_example`` (``eband, Ef (eV)`` in ``par.out`` and
#: ``per.out``). ``par`` is ``angle1 = 90``, in the slab plane; ``per`` is
#: ``angle1 = 0``, along the surface normal.
#: The force-theorem anisotropy of tetragonal cobalt on each direction's own
#: wedge against the whole grid, in meV: what D22 measured and the bound asserted.
WEDGE_FT_MEASURED = 5.4e-11
WEDGE_FT_BOUND = 1e-4
QE_EBAND = {"par": -75.5059287216436, "per": -75.5062821245659}
QE_FERMI = {"par": -0.454721315571854, "per": -0.457778043909281}


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """Every cell here compiles the whole SCF stack afresh and XLA keeps it.

    The rule of ``CLAUDE.md``'s memory section, applied because this file
    sweeps several cells: the results stay cached below, only the executables
    are dropped.
    """
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _smoke_pair():
    """A one-atom cubic Co cell, and the two datasets of the same generation.

    Cubic and one atom on purpose: every direction is then related to every
    other by a symmetry, so the answer to rungs 1 and 2 is *zero* and is known
    without computing anything.
    """
    scalar = Calculator.from_text(
        _SMOKE_SR, pseudo_dir=GENERATED.parent / "pseudo", conv_thr=1.0e-10
    )
    return scalar, scalar.get_scf()


@lru_cache(maxsize=2)
def _tetragonal():
    """The one-atom tetragonal cobalt cell: cheap, and *not* cubic.

    A cubic cell's anisotropy vanishes by symmetry, which is the right check
    for the assembly and useless for the torque -- a zero derivative of a zero
    curve says nothing. Stretching ``c/a`` to 1.30 gives a genuine uniaxial
    ``K1`` in one atom and eighteen k-points.
    """
    directory = GENERATED.parent / "pseudo"
    return (
        Calculator.from_file(GENERATED / "co-tetragonal-anisotropy-sr.in",
                             pseudo_dir=directory, announce=False),
        Calculator.from_file(GENERATED / "co-tetragonal-anisotropy-soc.in",
                             pseudo_dir=directory, announce=False),
    )


_SMOKE_SR = """
&control
   calculation='scf'
/
&system
   ibrav = 1, celldm(1) = 5.0, nat = 1, ntyp = 1,
   nspin = 2, ecutwfc = 20.0, ecutrho = 160.0,
   occupations = 'smearing', smearing = 'mv', degauss = 0.02,
   starting_magnetization(1) = 0.5,
/
&electrons
   conv_thr = 1.0e-10
/
ATOMIC_SPECIES
Co 58.933 Co.pbe-nd-rrkjus.UPF
ATOMIC_POSITIONS crystal
Co 0.0 0.0 0.0
K_POINTS automatic
2 2 2 0 0 0
"""

_SMOKE_SOC = _SMOKE_SR.replace(
    "calculation='scf'", "calculation='nscf'"
).replace(
    "   nspin = 2,",
    "   noncolin = .true., lspinorb = .true., lforcet = .true., nosym = .true.,",
).replace("Co.pbe-nd-rrkjus.UPF", "Co.rel-pbe-nd-rrkjus.UPF")

#: The same one-shot leg with the coupling *off*: a scalar-relativistic dataset
#: in a noncollinear run, which is legal and is the control for rung 1.
_SMOKE_NOSOC = _SMOKE_SOC.replace("lspinorb = .true., ", "").replace(
    "Co.rel-pbe-nd-rrkjus.UPF", "Co.pbe-nd-rrkjus.UPF"
)


# ----------------------------------------------------------------------
# the rotation itself
# ----------------------------------------------------------------------

def test_the_rotation_keeps_the_charge_and_the_moment():
    """``nc_magnetization_from_lsda`` moves the moment and nothing else."""
    rng = np.random.default_rng(0)
    density = rng.random((2, 6, 6, 6)) + 0.5
    charge = density[0] + density[1]
    moment = np.abs(density[0] - density[1])

    for angle1, angle2 in [(0, 0), (90, 0), (90, 90), (45, 30), (137, 201)]:
        direction = direction_from_angles(angle1, angle2)
        rotated = np.asarray(nc_magnetization_from_lsda(density, direction))
        assert rotated.shape == (4, 6, 6, 6)
        np.testing.assert_allclose(rotated[0], charge, atol=1e-14)
        np.testing.assert_allclose(
            np.linalg.norm(rotated[1:4], axis=0), moment, atol=1e-14
        )


def test_the_angles_and_the_direction_are_inverses():
    for direction in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (0, -1, 0),
                      (-1, 2, -3)]:
        wanted = np.asarray(direction, dtype=float)
        wanted = wanted / np.linalg.norm(wanted)
        angle1, angle2 = angles_from_direction(wanted)
        np.testing.assert_allclose(
            direction_from_angles(angle1, angle2), wanted, atol=1e-14
        )


def test_sphere_cover_is_unit_vectors_spread_over_the_sphere():
    """Elk's ``sphcover``: unit length, and an even spread in ``z``."""
    for n in (1, 2, 7, 40):
        points = np.asarray(sphere_cover(n))
        assert points.shape == (n, 3)
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-14)
    # ``dz = 2/n`` between consecutive points is the formula's own statement.
    z = np.asarray(sphere_cover(20))[:, 2]
    np.testing.assert_allclose(np.diff(z), -2.0 / 20, atol=1e-12)


@pytest.mark.slow
def test_cardinal_directions_are_reduced_by_the_crystal_group():
    """``gentpmae``'s ``npmae < 0``: a cube keeps far fewer than its 26 rays."""
    scalar, _ = _smoke_pair()
    directions = cardinal_directions(scalar.system, 1)
    points = np.asarray(directions)
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-12)
    # 26 non-zero integer rays with |n_i| <= 1; a cubic group leaves the three
    # inequivalent families <100>, <110>, <111>.
    assert 1 <= len(directions) <= 26
    assert len(directions) < 26


def test_the_direction_orbit_is_the_point_group_and_not_its_transpose():
    """Which action reduces the candidates, settled without reading a docstring.

    ``Symmetries`` stores ``rotations[s] = M`` with ``S a_i = sum_j M_ij a_j``,
    so a *direct*-lattice coordinate vector goes to ``M^T n`` -- a Miller index
    is the one that goes to ``M m``. ``cardinal_directions`` built its orbit as
    ``M n`` on a vector it converts two lines later as ``at.T @ lattice``, which
    is a direct-lattice vector.

    The check here needs neither convention: **exactly one of the two actions is
    the point group**, because only one of ``basis M inverse`` and
    ``basis M^T inverse`` is orthogonal, and orthogonality is a property of the
    matrix rather than of anybody's index order. Measured as
    ``max|R R^T - I|``: hcp cobalt **5.33 against 2.2e-16**, zincblende AlAs
    5.00 against 1.6e-17, noncollinear nickel 5.00 against 2.1e-18.

    **Tetragonal cobalt cannot tell them apart** (2.2e-16 both ways), which is
    the control -- and it is also why this went unnoticed. The audit entry put
    the safe set as "cubic, tetragonal and orthorhombic"; that is true of a
    *simple* lattice, and false for fcc and hcp, whose rotation matrices in the
    primitive crystal basis are not signed permutations however cubic the
    crystal is.
    """
    from pathlib import Path

    from defuqumat.system.builder import system_from_file

    cases = Path(__file__).resolve().parents[1] / "data" / "qe"
    discriminating = {"co-hcp-anisotropy-sr": 5.0, "alas-piezo": 4.0,
                      "ni-noncol-111": 4.0}
    for case, floor in discriminating.items():
        system = system_from_file(cases / f"{case}.in")
        basis = np.asarray(system.cell.at, dtype=float).T
        inverse = np.linalg.inv(basis)
        straight = transposed = 0.0
        for matrix in np.asarray(system.symmetry_group().rotation_array(),
                                 dtype=float):
            for value, label in ((matrix, "straight"), (matrix.T, "transposed")):
                cartesian = basis @ value @ inverse
                residue = float(np.abs(cartesian @ cartesian.T - np.eye(3)).max())
                if label == "straight":
                    straight = max(straight, residue)
                else:
                    transposed = max(transposed, residue)
        assert transposed < 1e-12, f"{case}: M^T has to be the point group"
        assert straight > floor, f"{case}: M has to be visibly not one"

    # ...and the cell where the two coincide, which is the control that says the
    # discriminating cells are discriminating and not merely different.
    system = system_from_file(cases / "co-tetragonal-anisotropy-sr.in")
    basis = np.asarray(system.cell.at, dtype=float).T
    inverse = np.linalg.inv(basis)
    for matrix in np.asarray(system.symmetry_group().rotation_array(), dtype=float):
        for value in (matrix, matrix.T):
            cartesian = basis @ value @ inverse
            assert np.abs(cartesian @ cartesian.T - np.eye(3)).max() < 1e-12


def test_hexagonal_cobalt_keeps_both_basal_families():
    """The count was right either way, so only the membership shows it.

    A hexagonal crystal has two inequivalent in-plane direction families,
    ``[100]``-type at ``phi = 0, 60, 120, ...`` and ``[210]``-type at
    ``30, 90, 150, ...``, and the basal-plane anisotropy is the difference
    between them. Reduced with the wrong action, hcp cobalt's two in-plane
    representatives came out at ``phi = 180`` and ``240`` -- **both** of the
    first family, with the second missing entirely, so
    ``MagneticAnisotropy.anisotropy`` in the basal plane was zero by omission
    rather than by physics. After, they are ``240`` and ``150``, one of each.

    **Five directions both ways**, which is why a length check could not have
    caught this and the test asserts the families instead.
    """
    from pathlib import Path

    from defuqumat.system.builder import system_from_file

    cases = Path(__file__).resolve().parents[1] / "data" / "qe"
    system = system_from_file(cases / "co-hcp-anisotropy-sr.in")
    directions = cardinal_directions(system, 1)
    assert len(directions) == 5

    in_plane = [d for d in directions if abs(d[2]) < 1e-9]
    # Round before the modulo: an angle a last bit under 240 degrees is 59.999...
    # modulo 60, which rounds to 60.0 and not to the 0.0 of its family (Triton csl).
    families = {abs(round(float(np.degrees(np.arctan2(d[1], d[0]))), 1) % 60.0)
                for d in in_plane}
    assert len(in_plane) == 2
    assert families == {0.0, 30.0}, (
        f"both basal families have to be represented, got {families}")


def test_turning_the_axis_needs_nosym_at_every_entry_point():
    """The argument was in a docstring and enforced at two of four call sites.

    ``_with_quantization_axis`` rebuilds the k-points through
    ``System.with_spin``, which for a magnetic noncollinear run takes the
    *magnetic* group of the **new** angles and reduces the grid with it. Its own
    docstring argued that this is safe "because it only runs when the direction
    differs from the system's own, which ``run_force_theorem`` already requires
    ``nosym`` for". That is true of ``run_force_theorem`` and of the relaxed
    path, and false of ``run_torque`` and ``frozen_expectation``, which reach
    the same helper with no such clause -- and ``run_torque``'s direction is
    ``cos(angle) first + sin(angle) second`` at a default angle of ``pi/4``, so
    the rebuild fires essentially always.

    **The measurement is a null on both cells the tree has**, and the record
    says so rather than borrowing the entry's forecast. Driving ``run_torque``
    with the refusal monkeypatched away, tetragonal cobalt gives
    ``K1 = +0.552275`` meV on 18 k-points against ``+0.552274`` on the 6-point
    wedge, and hexagonal cobalt ``-0.927715`` against ``-0.927716``: agreement
    to 1e-6 meV, so on these two the reduced wedge is a valid sampling. What
    would show it is a magnetic group the torque's axial perturbation is not
    invariant under, and there is no such cell here. The check is therefore a
    **consistency** fix -- the same argument the other two entry points already
    enforce, moved to where the rebuild is so a fifth call site cannot miss it.
    """
    import re

    scalar, spinor = _tetragonal()
    text = re.sub(r"nosym\s*=\s*\.true\.,?", "",
                  (GENERATED / "co-tetragonal-anisotropy-soc.in").read_text())
    reduced = Calculator.from_text(text, GENERATED.parent / "pseudo",
                                   announce=False)
    assert not reduced.system.nosym, "this cell has to reach the guarded branch"

    scf = scalar.get_scf()
    with pytest.raises(ValueError, match="needs nosym"):
        run_torque(reduced.system, reduced.pseudos, scf.density)
    with pytest.raises(ValueError, match="needs nosym"):
        frozen_expectation(reduced.system, reduced.pseudos, scf.density,
                           direction=(1.0, 0.0, 0.0))

    # ...and the committed cell, which carries nosym, still runs.
    torque = run_torque(spinor.system, spinor.pseudos, scf.density)
    assert np.isfinite(float(torque.anisotropy_constant_mev))


# ----------------------------------------------------------------------
# rung 1: without the coupling there is no anisotropy at all
# ----------------------------------------------------------------------

@pytest.mark.slow
def test_without_spin_orbit_the_band_energy_does_not_know_the_direction():
    """The identity that caught the quantization-axis bug.

    The Hamiltonian without spin-orbit coupling commutes with a global spin
    rotation, so this is exact rather than approximate -- and it is checked
    with the *coupling off* precisely so that a failure cannot be blamed on the
    spin-orbit term.
    """
    scalar, scf = _smoke_pair()
    nosoc = Calculator.from_text(_SMOKE_NOSOC, pseudo_dir=GENERATED.parent / "pseudo")
    assert not nosoc.system.lspinorb and nosoc.system.noncolin

    energies = [
        run_force_theorem(
            nosoc.system, nosoc.pseudos, scf.density, direction=direction,
            require_spin_orbit=False,
        ).band_energy
        for direction in [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1), (0, 1, 1)]
    ]
    spread = (max(energies) - min(energies)) * RY_TO_EV
    assert spread == pytest.approx(0.0, abs=1.0e-10), (
        f"the band energy moved by {spread:.3e} eV under a rotation the "
        "Hamiltonian is invariant under"
    )


@pytest.mark.slow
def test_soc_scale_zero_gives_exactly_no_anisotropy():
    """The same identity as above, on **one** dataset instead of a matched pair.

    ``soc_scale = 0`` switches the coupling off inside the fully-relativistic
    file, so this is the control the two-file route cannot run: same
    projectors, same overlap, same everything but the coupling. It is the
    check that caught the phase's second bug -- ``newd_so``'s ``fcoef``
    sandwich left unscaled, which is invisible in a bulk cell and worth
    -6.7 meV on a slab, because ``dvan_so`` carries the coupling too and
    switching only *it* off still looks like it worked.
    """
    scalar, scf = _smoke_pair()
    soc = Calculator.from_text(_SMOKE_SOC, pseudo_dir=GENERATED.parent / "pseudo")
    result = run_anisotropy(
        soc.system, soc.pseudos, scf.density,
        directions=[(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)], soc_scale=0.0,
    )
    # 1e-7 meV is 1e-10 eV: the identity is exact and what is left is the
    # eigensolver's own residual over four directions.
    assert result.anisotropy_mev == pytest.approx(0.0, abs=1.0e-7)


def test_an_intermediate_soc_scale_is_admitted_on_an_augmented_dataset():
    """Between 0 and 1 the scale blends all four sandwiches on ultrasoft (I114).

    This cell's cobalt is ultrasoft, so 0.5 is admitted, and the calculation's
    spinor overlap is the blend of the two ends' (it was refused until I114,
    on a measurement taken with the reduction P115 replaced). Outside
    ``[0, 1]`` it is not a blend of anything and is refused by ``System``. The
    fully-relativistic PAW refusal is held in ``tests/unit/test_spiral_soc.py``
    and ``tests/unit/test_soc_scale.py``.
    """
    from defuqumat.scf.driver import Calculation

    soc = Calculator.from_text(_SMOKE_SOC, pseudo_dir=GENERATED.parent / "pseudo")
    ends = [np.asarray(Calculation(soc.system.with_soc_scale(s), soc.pseudos).qq_so)
            for s in (0.0, 1.0)]
    half = np.asarray(Calculation(soc.system.with_soc_scale(0.5), soc.pseudos).qq_so)
    np.testing.assert_allclose(half, 0.5 * (ends[0] + ends[1]), rtol=0.0,
                               atol=1e-12 * np.abs(ends[1]).max())
    assert np.abs(ends[1] - ends[0]).max() > 1e-3
    with pytest.raises(ValueError, match="not a blend"):
        soc.system.with_soc_scale(1.5)


def test_the_first_order_operator_is_the_coupled_hamiltonian_minus_the_reduced_one():
    """What ``frozen_expectation`` evaluates is ``H(1) - H(0)``, entry by entry.

    The operator used to be written as a list of the terms ``soc_scale``
    touches, and the list was one short: ``newd_so``'s sandwich of ``int V Q``
    against its spin trace was missing, worth 1.26e-2 meV on the cubic cell and
    1.79e-3 meV of anisotropy on tetragonal cobalt (``PLAN.md`` P119, P120).
    So the check is not a value but an identity against the two Hamiltonians
    themselves: at any magnetized potential, the operator's nonlocal part must
    equal a coupled ``Calculation``'s ``coefficients`` minus the reduced one's,
    and its overlap part the difference of their ``qq_so``. A term either
    Hamiltonian gains later and the operator does not fails it.

    The potential is the atomic superposition's, which is magnetized along
    ``z`` by ``starting_magnetization``, so the exchange components the missing
    term lives in are nonzero, and the last assertion is that the guard would
    fire: without the ``newd_so`` term the difference is far outside the
    tolerance.
    """
    import jax.numpy as jnp

    from defuqumat.scf.driver import Calculation
    from defuqumat.scf.potential import as_potential_components
    from defuqumat.workflows.anisotropy import _first_order_operator

    soc = Calculator.from_text(_SMOKE_SOC, pseudo_dir=GENERATED.parent / "pseudo",
                               announce=False)
    reduced = Calculation(soc.system.with_soc_scale(0.0), soc.pseudos)
    coupled = Calculation(soc.system, soc.pseudos)
    assert reduced.nspin_mag == 4

    total = reduced.potential(reduced.starting_density()).v_scf + (
        as_potential_components(reduced.vltot, reduced.nspin_mag)
    )
    assert float(jnp.abs(total[1:]).max()) > 1.0e-3, "the potential must be magnetized"

    delta_d, delta_qq = _first_order_operator(reduced, total)
    expected_d = np.asarray(coupled.coefficients(total) - reduced.coefficients(total))
    expected_qq = np.asarray(coupled.qq_so - reduced.qq_so)
    scale = np.abs(expected_d).max()
    np.testing.assert_allclose(np.asarray(delta_d), expected_d, rtol=0, atol=1.0e-13 * scale)
    np.testing.assert_allclose(np.asarray(delta_qq), expected_qq, rtol=0,
                               atol=1.0e-13 * np.abs(expected_qq).max())

    # The guard fires: the bare coefficients' difference alone, which is what
    # the operator was before P120, misses by more than the whole operator's
    # largest entry -- 1.42 against 0.149 on this potential, the bare and the
    # augmentation differences each being ten times their sum.
    bare_only = np.asarray(coupled.dvan_so - reduced.dvan_so)
    assert np.abs(bare_only - expected_d).max() > scale


@pytest.mark.slow
def test_the_first_order_term_is_isotropic_on_a_cubic_cell():
    """Freezing the states and taking the coupling's expectation value once.

    That is the calculation the force theorem is often assumed to be, and on a
    cubic cell its value must not depend on the direction, since every axis is
    related to every other by a symmetry. It does not vanish: +1.2605e-2 meV in
    each direction at ``conv_thr = 1e-10``, all of it from the exchange
    components of ``newd_so``'s sandwich, which a quenched orbital moment does
    not remove (``frozen_expectation``'s docstring has the argument).

    **The isotropy bound is the one that must be tight.** Taking the sandwich
    on ``v_scf`` without ``vltot`` gives a spread of 1.2e-5 meV where the right
    potential gives 1.1e-7, so a bound of 1e-6 separates the two and a looser
    one would pass either.
    """
    scalar, scf = _smoke_pair()
    soc = Calculator.from_text(_SMOKE_SOC, pseudo_dir=GENERATED.parent / "pseudo")
    first = [
        frozen_expectation(soc.system, soc.pseudos, scf.density, direction=d)
        * RY_TO_EV * 1000
        for d in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    ]
    spread = max(first) - min(first)
    assert spread < 1.0e-6, (
        f"the first-order term acquired a direction dependence of {spread:.3e} meV"
    )
    assert first[0] == pytest.approx(1.2605e-2, abs=5.0e-6)


@pytest.mark.slow
def test_the_first_order_anisotropy_is_a_fraction_of_a_per_cent_of_the_diagonalisation():
    """Why this is a diagonalisation and not an expectation value.

    On tetragonal cobalt the first-order term does carry an anisotropy, and it
    is 1.79e-3 meV (+1.1408e-2 along ``x``, +0.9623e-2 along ``z``) against
    0.552 meV of free-energy anisotropy from the force theorem on the same
    density: the anisotropy is second order in the coupling to 0.3 per cent,
    and what supplies it is the repulsion between levels a diagonalisation
    performs. The first-order term is compared with the **free** energy
    because it is ``dF/d(soc_scale)`` at frozen occupations; the band energy's
    anisotropy on this cell is 1.235 meV.

    Both at ``conv_thr = 1e-10`` on the scalar-relativistic leg, which is
    tighter than the committed input's 1e-9 and is where the numbers above
    were measured.
    """
    directory = GENERATED.parent / "pseudo"
    scalar = Calculator.from_file(GENERATED / "co-tetragonal-anisotropy-sr.in",
                                  pseudo_dir=directory, announce=False,
                                  conv_thr=1.0e-10)
    _, spinor = _tetragonal()
    scf = scalar.get_scf()
    along_x, along_z = (
        frozen_expectation(spinor.system, spinor.pseudos, scf.density, direction=d)
        * RY_TO_EV * 1000
        for d in [(1, 0, 0), (0, 0, 1)]
    )
    first_order = along_x - along_z
    assert first_order == pytest.approx(1.785e-3, abs=2.0e-5)

    energies = run_anisotropy(spinor.system, spinor.pseudos, scf.density,
                              directions="xz")
    free = energies.free_energies * RY_TO_EV * 1000
    second_order = free[0] - free[1]
    assert second_order == pytest.approx(0.5523, abs=1.0e-3)
    # the same sign, and three hundred times smaller
    assert 0.0 < first_order < 1.0e-2 * second_order


@pytest.mark.slow
def test_a_cubic_crystal_has_no_anisotropy_between_its_axes():
    """Rung 2, and nothing in the assembly imposes it."""
    scalar, scf = _smoke_pair()
    soc = Calculator.from_text(_SMOKE_SOC, pseudo_dir=GENERATED.parent / "pseudo")
    result = run_anisotropy(soc.system, soc.pseudos, scf.density, directions="xyz")
    assert result.anisotropy_mev == pytest.approx(0.0, abs=1.0e-6)


# ----------------------------------------------------------------------
# rung 4: QE's own force-theorem example
# ----------------------------------------------------------------------

@pytest.mark.slow
def test_co_slab_reproduces_pw_x_on_the_force_theorem():
    """``PP/examples/ForceTheorem_example``: a 3-layer Co(0001) slab.

    Two ``pw.x`` runs with ``lforcet``, whose reference outputs carry ``eband``
    to thirteen digits, and whose difference -- the MAE -- is 0.3534 meV for
    the three-atom cell. QE's own README warns that its k-grid is too coarse
    for the *physics*, which does not matter here: what is being reproduced is
    the number ``pw.x`` prints on that grid.

    **About 25 minutes**, nearly all of it the SCF, and it does not reach
    ``conv_thr``. QE converges this slab in 24 iterations with
    ``mixing_mode = 'local-TF'`` (``approx_screening2``), which is not
    implemented here and which exists for exactly this shape of system: plain
    Anderson at ``beta = 0.7`` *diverges* on it (to +335 Ry), and Kerker at
    ``beta = 0.3`` converges linearly and is still an order short of 1e-10 at
    250 iterations. So the assertions below are about what survives that.

    **What is asserted is the pair, not the two numbers.** Each leg's ``eband``
    sits about 0.25 eV above QE's, because the density is the one the mixer
    reached rather than the one QE reached, and a density error shifts every
    eigenvalue. The two statements that are the actual finding are that the
    *same* shift applies to both legs -- which is what a frozen-density theorem
    promises -- and that the MAE, where it cancels, is QE's.
    """
    scalar = Calculator.from_file(
        GENERATED / "co-slab-forcetheorem-sr.in",
        pseudo_dir=GENERATED.parent / "pseudo",
        mixing_mode="kerker", mixing_beta=0.3, max_iterations=250,
    )
    scf = scalar.get_scf()
    # Not ``scf.converged``: see the docstring. What *is* asserted is that it
    # found QE's magnetic solution, which is the thing a different mixer could
    # plausibly have got wrong -- ``pw.x`` prints 5.26 and 5.84 Bohr magnetons.
    assert scf.accuracy < 1.0e-5
    moment = float(np.sum(np.asarray(scf.density)[0] - np.asarray(scf.density)[1])
                   * scalar.system.cell.volume / np.asarray(scf.density)[0].size)
    assert moment == pytest.approx(5.26, abs=0.02)

    energies = {}
    for name in ("par", "per"):
        leg = Calculator.from_file(
            GENERATED / f"co-slab-forcetheorem-{name}.in",
            pseudo_dir=GENERATED.parent / "pseudo",
        )
        assert leg.system.lforcet and leg.system.lspinorb and leg.system.nosym
        result = run_force_theorem(leg.system, leg.pseudos, scf.density)
        energies[name] = result.band_energy * RY_TO_EV

    # The two legs are offset from QE by the same amount, which is the
    # frozen-density theorem's own promise and what makes the difference mean
    # anything. Measured identical to four significant figures (+0.2481 eV).
    offsets = {name: energies[name] - QE_EBAND[name] for name in energies}
    assert offsets["par"] == pytest.approx(offsets["per"], abs=1.0e-3), (
        f"the two legs are shifted differently: {offsets}"
    )

    mae = energies["par"] - energies["per"]
    reference = QE_EBAND["par"] - QE_EBAND["per"]
    assert mae * 1000 == pytest.approx(reference * 1000, abs=0.02), (
        f"MAE {mae * 1000:.4f} meV against pw.x's {reference * 1000:.4f} meV"
    )


# ----------------------------------------------------------------------
# the refusals
# ----------------------------------------------------------------------

@pytest.mark.slow
def test_a_direction_off_the_system_s_own_runs_on_its_own_wedge():
    """Each direction on the wedge of its own magnetic group, against ``nosym``.

    The refusal this replaces said two directions on two magnetic wedges would
    differ by the k-sampling. Not when each wedge is the direction's own: a band
    energy is a scalar, and a scalar's wedge sum is the whole grid's whenever
    the group maps the grid onto itself. Tetragonal cobalt, the same frozen
    density in both arms: 18 points with ``nosym``, and with symmetry 6 along
    ``z``, 8 along ``x`` and ``(1,1,0)``, 10 at a generic direction. Measured on
    D22 at ``WEDGE_FT_MEASURED`` meV in the anisotropy against the whole grid.

    A projected decomposition is still refused off the axis on a wedge: the
    group permutes its site and orbital labels.
    """
    import re

    scalar, spinor = _tetragonal()
    text = re.sub(r"nosym\s*=\s*\.true\.,?", "",
                  (GENERATED / "co-tetragonal-anisotropy-soc.in").read_text())
    reduced = Calculator.from_text(text, GENERATED.parent / "pseudo",
                                   announce=False)
    scf = scalar.get_scf()
    energies = {}
    for name, calc in (("whole", spinor), ("wedge", reduced)):
        for d in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.3, 0.5, 0.8)):
            energies[name, d] = run_force_theorem(
                calc.system, calc.pseudos, scf.density, direction=d).band_energy
    for d in ((1.0, 0.0, 0.0), (0.3, 0.5, 0.8)):
        whole = energies["whole", d] - energies["whole", (0.0, 0.0, 1.0)]
        wedge = energies["wedge", d] - energies["wedge", (0.0, 0.0, 1.0)]
        assert abs(wedge - whole) * RY_TO_EV * 1000 < WEDGE_FT_BOUND
    with pytest.raises(ValueError, match="projected"):
        run_force_theorem(reduced.system, reduced.pseudos, scf.density,
                          direction=(1.0, 0.0, 0.0), projected=True)


def test_a_relaxed_direction_with_symmetry_on_is_still_refused():
    """The relaxed route keeps its ``nosym`` refusal; the force theorem does not.

    Running each direction on its own magnetic wedge was tried on tetragonal
    cobalt and is not verified for a total energy: at ``(1,0,1)`` the wedge and
    the ``nosym`` run read 0.678 meV apart on D22 with neither SCF reaching
    ``conv_thr = 1e-10`` in 100 iterations (5.5e-10 and 7.0e-9 Ry). No SCF here.
    """
    import re

    from defuqumat.workflows.anisotropy import run_relaxed_direction

    text = re.sub(r"nosym\s*=\s*\.true\.,?", "",
                  (GENERATED / "co-tetragonal-anisotropy-soc.in").read_text())
    reduced = Calculator.from_text(text, GENERATED.parent / "pseudo",
                                   announce=False)
    with pytest.raises(ValueError, match="needs nosym"):
        run_relaxed_direction(reduced.system, reduced.pseudos,
                              direction=(1.0, 0.0, 1.0))


def test_a_scalar_relativistic_one_shot_leg_is_refused_by_default():
    """It is legal only as the control above, so it has to be asked for."""
    scalar, scf = _smoke_pair()
    nosoc = Calculator.from_text(_SMOKE_NOSOC, pseudo_dir=GENERATED.parent / "pseudo")
    with pytest.raises(ValueError, match="lspinorb"):
        run_force_theorem(nosoc.system, nosoc.pseudos, scf.density)


def test_a_collinear_one_shot_leg_is_refused():
    scalar, _ = _smoke_pair()
    with pytest.raises(ValueError, match="noncollinear"):
        run_force_theorem(scalar.system, scalar.pseudos,
                          np.zeros((2, 4, 4, 4)), direction=(0, 0, 1))


def test_a_paw_dataset_without_becsum_is_refused_with_qe_s_own_reason():
    """``potinit.f90:98``, and the refusal has to fire *before* the NSCF's.

    ``fixed_density_states`` refuses PAW too, and its advice -- pass
    ``becsum = scf_result.becsum`` -- cannot be followed on the **two-file**
    route this cell is written for: that ``becsum`` belongs to a run with a
    different pseudopotential file and a different number of projectors. So the
    message has to come from this workflow, which means the check has to run
    before any array is built. The one-file route, where it can be followed, is
    the test below.
    """
    scalar, _ = _smoke_pair()
    text = _SMOKE_SOC.replace(
        "Co.rel-pbe-nd-rrkjus.UPF", "Pt.rel-pbe-n-kjpaw_psl.0.1.UPF"
    ).replace("Co 58.933", "Pt 195.08").replace("Co 0.0 0.0 0.0", "Pt 0.0 0.0 0.0")
    paw = Calculator.from_text(text, pseudo_dir=GENERATED.parent / "pseudo")
    with pytest.raises(NotImplementedError, match="PAW"):
        run_force_theorem(paw.system, paw.pseudos, np.zeros((2, 24, 24, 24)))



#: One file and one projector set for both legs: a collinear PAW run to
#: converge the state and the same file, noncollinear, for the one-shot. It is
#: the route a PAW anisotropy needs, since ``becsum`` cannot cross between the
#: two files of the scalar/relativistic pair.
_PAW_SR = """
&control
   calculation = 'scf'
/
&system
   ibrav = 1, celldm(1) = 14.0, nat = 2, ntyp = 1,
   ecutwfc = 30, ecutrho = 240,
   occupations = 'smearing', smearing = 'gaussian', degauss = 0.02,
   nspin = 2, nosym = .true.,
   starting_magnetization(1) = 0.3,
/
&electrons
   conv_thr = 1.0d-9
/
ATOMIC_SPECIES
 O  15.999  O.pz-kjpaw.UPF
ATOMIC_POSITIONS crystal
 O  0.25 0.25 0.25
 O  0.75 0.75 0.75
K_POINTS gamma
STARTING_MOMENTS
  0.0 0.0  1.5
  0.0 0.0 -1.5
"""

_PAW_NC = _PAW_SR.replace("calculation = 'scf'", "calculation = 'nscf'").replace(
    "   nspin = 2, nosym = .true.,",
    "   noncolin = .true., lforcet = .true., nosym = .true.,",
)


def test_a_becsum_from_the_other_file_is_refused_and_a_matching_count_is_not_enough():
    """The two ways a wrong ``becsum`` can arrive, and only one has a shape.

    The route's own rule is one file for both legs, and the check has to hold
    it. A ``becsum`` from the scalar-relativistic partner is caught by its
    shape, ``nh`` being 18 there against 34 in the relativistic file for both
    committed pairs. A ``becsum`` from a *different* dataset with the same
    count is not caught by any shape at all -- ``Si.pbe-n-rrkjus_psl.0.1`` and
    ``Si.pbe-n-kjpaw_psl.0.1`` both have ``nh = 8``, and
    ``Ni.rel-pbe-spn-rrkjus_psl.1.0.0`` and its ``kjpaw`` partner both have 34
    -- so what separates those is the file each species names, which is what
    the front door asks. Both halves are asserted because only the first of
    them announces itself.
    """
    from defuqumat.pseudo import read_upf
    from defuqumat.workflows.anisotropy import _checked_becsum, becsum_fits

    directory = GENERATED.parent / "pseudo"
    scalar = read_upf(directory / "Co.pbe-nd-rrkjus.UPF")
    relativistic = read_upf(directory / "Co.rel-pbe-nd-rrkjus.UPF")

    wrong_shape = (np.zeros((2, 1, 18, 18)),)
    with pytest.raises(ValueError, match="projector channels"):
        _checked_becsum(wrong_shape, (relativistic,))
    assert not becsum_fits(wrong_shape, (relativistic,), source=(scalar,))

    # ... and the half no shape can see: the right count, the wrong file.
    right_shape = (np.zeros((2, 1, 34, 34)),)
    _checked_becsum(right_shape, (relativistic,))          # passes, as it must
    assert becsum_fits(right_shape, (relativistic,), source=(relativistic,))
    assert not becsum_fits(right_shape, (relativistic,), source=(scalar,)), (
        "a becsum from another file with the same projector count has to be "
        "refused by the file it names, since nothing about its shape differs"
    )


@pytest.mark.slow
def test_the_front_door_hands_becsum_over_on_one_file_and_not_on_two(monkeypatch):
    """``Calculator.get_anisotropy`` decides this, so the decision is asserted.

    The guide says the front door passes ``becsum`` when the first leg's
    dataset is the one the second leg has, and hands over nothing when it is
    not. Both branches are checked against what actually arrives at
    :func:`run_anisotropy`, since a facade that quietly dropped it would leave
    a PAW run refusing with a message about a route the caller had already
    taken.
    """
    import defuqumat.calculator as facade

    seen = {}

    def spy(system, pseudos, density, **options):
        seen["becsum"] = options.get("becsum", ())
        raise RuntimeError("stop here")

    monkeypatch.setattr(facade, "run_anisotropy", spy, raising=False)
    monkeypatch.setattr(
        "defuqumat.workflows.anisotropy.run_anisotropy", spy, raising=False
    )

    scalar, _ = _smoke_pair()
    two_file = Calculator.from_text(_SMOKE_SOC, pseudo_dir=GENERATED.parent / "pseudo")
    with pytest.raises(RuntimeError, match="stop here"):
        scalar.get_anisotropy(two_file, directions="xz")
    assert not seen["becsum"], (
        "the two legs are two files here, so nothing about the first leg's "
        "becsum fits the second leg's projectors"
    )

    one_file = Calculator.from_text(
        _SMOKE_SR.replace("calculation='scf'", "calculation='nscf'").replace(
            "   nspin = 2,",
            "   noncolin = .true., lforcet = .true., nosym = .true.,",
        ),
        pseudo_dir=GENERATED.parent / "pseudo",
    )
    with pytest.raises(RuntimeError, match="stop here"):
        scalar.get_anisotropy(one_file, directions="xz")
    assert seen["becsum"], (
        "one file for both legs, so the first leg's becsum is indexed by the "
        "projectors the second leg has and has to cross"
    )


@pytest.mark.slow
def test_a_paw_force_theorem_carries_becsum_and_the_rotation_identity_holds():
    """Rung 1 on a PAW dataset, which is what the handoff had to reach.

    A PAW Hamiltonian's one-centre coefficients are a functional of ``becsum``
    exactly as the grid potential is a functional of ``rho``, so the pair is
    what the theorem freezes. Both legs run the same file here -- the projector
    sets of a scalar-relativistic dataset and of its fully-relativistic partner
    are indexed differently, so a ``becsum`` cannot cross between them -- and
    with the coupling off the band energy cannot depend on where the moment
    points. Measured: **3.1e-10 meV** over five directions.

    **The guard is fed a case that must trip it**, because a clean zero here is
    also what a run that ignored ``becsum`` would give: handing the same
    ``becsum`` to every direction without rotating it with the density leaves
    the one-centre field pointing along ``z`` while the grid field points where
    it was asked to, and the spread is then **468 meV** on a cell whose answer
    is exactly zero.

    The cell is antiferromagnetic on purpose: the rotation is read off the
    *density's* axis, and one species' own ``becsum`` would answer a different
    question, the two sublattices pointing opposite ways while the cell has one
    frame to rotate.
    """
    import defuqumat.workflows.anisotropy as anisotropy

    directory = GENERATED.parent / "pseudo"
    collinear = Calculator.from_text(_PAW_SR, pseudo_dir=directory)
    assert any(pseudo.is_paw for pseudo in collinear.pseudos), "the cell must be PAW"
    scf = collinear.get_scf()
    assert scf.converged
    assert scf.becsum, "a PAW run has to carry becsum for this to test anything"
    spinor = Calculator.from_text(_PAW_NC, pseudo_dir=directory)
    directions = [(0, 0, 1), (1, 0, 0), (0, 1, 0), (1, 1, 1), (0, 1, 1)]

    def spread(becsum):
        energies = [
            run_force_theorem(
                spinor.system, spinor.pseudos, scf.density, direction=d,
                require_spin_orbit=False, becsum=becsum,
            ).band_energy
            for d in directions
        ]
        return (max(energies) - min(energies)) * RY_TO_EV * 1000.0

    assert spread(scf.becsum) < 1.0e-6

    # ... and the same run with the rotation removed, which must not pass.
    frozen = tuple(
        None if b is None else nc_magnetization_from_lsda(b, (0.0, 0.0, 1.0))
        for b in scf.becsum
    )
    rotate = anisotropy.nc_magnetization_from_lsda
    anisotropy.nc_magnetization_from_lsda = lambda values, direction, axis_from=None: (
        rotate(values, direction) if axis_from is None else values
    )
    try:
        stuck = spread(frozen)
    finally:
        anisotropy.nc_magnetization_from_lsda = rotate
    assert stuck > 1.0, (
        "an unrotated becsum has to break the identity; it gave "
        f"{stuck:.3e} meV, so the test cannot tell a carried becsum from an "
        "ignored one"
    )


@pytest.mark.slow
def test_the_decomposition_recovers_the_band_energy_up_to_the_spilling():
    """``force_theorem``'s two printed totals, and the gap between them.

    ``projwfc.f90:636`` writes ``eband_tot`` and ``eband_proj_tot`` on one
    line, which is this decomposition's own diagnostic: the atomic-orbital
    basis does not span the occupied manifold, and what it misses is the
    spilling. The electron count is asserted beside it because a projection
    that has lost a ``degspin`` factor still looks entirely reasonable
    (``PLAN.md`` P51).
    """
    scalar, scf = _smoke_pair()
    soc = Calculator.from_text(_SMOKE_SOC, pseudo_dir=GENERATED.parent / "pseudo")
    result = run_force_theorem(
        soc.system, soc.pseudos, scf.density, direction=(0, 0, 1), projected=True
    )
    # A spinor band holds one electron; the weights must still sum to nelec.
    assert result.occupations.sum() == pytest.approx(9.0, abs=1e-9)

    shifted = float(
        np.sum(result.occupations * (result.eigenvalues - result.projected.ef_0))
    )
    assert result.projected.total == pytest.approx(shifted, rel=2.0e-3)
    # Every orbital belongs to the one atom, so the per-atom sum is the total.
    assert result.projected.by_atom.sum() == pytest.approx(
        result.projected.total, abs=1e-12
    )
    # With the moment along ``z`` and the coupling on, a cubic site keeps
    # ``p_x`` and ``p_y`` degenerate and need **not** keep ``p_z`` with them --
    # which is the anisotropy itself, seen orbital by orbital. QE's real
    # harmonics for ``l = 1`` are ordered ``(z, x, y)``
    # (:data:`defuqumat.projwfc.channels.M_LABELS`), so it is the last two that
    # must agree.
    p_up = {
        channel.m: value for (channel, spin), value in
        zip(result.projected.labels, result.projected.by_orbital)
        if channel.l == 1 and spin == "up" and channel.wfc == 2
    }
    assert set(p_up) == {0, 1, 2}
    assert p_up[1] == pytest.approx(p_up[2], rel=1e-8)


# ----------------------------------------------------------------------
# the torque: the anisotropy as a derivative rather than a difference
# ----------------------------------------------------------------------

@pytest.mark.slow
def test_the_torque_reproduces_the_free_energy_difference():
    """One angle against two, and they share almost no machinery.

    ``run_anisotropy`` takes the anisotropy as ``E(n_1) - E(n_2)``: two
    independent diagonalisations differenced, with seven digits of
    cancellation. ``run_torque`` takes it as ``-dF/dtheta`` at a single angle,
    where nothing cancels. For ``E = K1 sin^2(theta)`` the torque at 45 degrees
    *is* ``-K1``, so the two must give the same constant -- and on tetragonal
    cobalt they agree to **2.4e-5 meV**.

    **The comparison is against the FREE energy and that is the whole point of
    this test.** A Hellmann-Feynman derivative at frozen occupations is the
    derivative of ``F = sum w eps - TS``, not of ``sum w eps``: the band energy
    carries an extra ``sum (dw/dtheta) eps`` that the entropy cancels. On this
    cell at ``degauss = 0.02`` Ry that term is **55 per cent** of the answer, so
    comparing against ``anisotropy_mev`` instead would look like a factor-of-two
    bug in the gradient and is not one.
    """
    scalar, spinor = _tetragonal()
    scf = scalar.get_scf()
    energies = run_anisotropy(spinor.system, spinor.pseudos, scf.density,
                              directions="xz")
    torque = run_torque(spinor.system, spinor.pseudos, scf.density)

    assert torque.anisotropy_constant_mev == pytest.approx(
        energies.free_anisotropy_mev, abs=2.0e-3
    )
    # ... and it is emphatically not the band-energy difference here.
    assert abs(energies.anisotropy_mev - energies.free_anisotropy_mev) > 0.5


@pytest.mark.slow
def test_the_torque_is_the_gradient_of_the_energy_it_claims_to_be():
    """Two checks that need no second method at all.

    ``sum w <psi|H|psi>`` must reproduce ``sum w eps`` at the angle the states
    came from -- one line that catches a wrong contraction, a lost weight or a
    mis-shaped spinor -- and the analytic gradient must reproduce a central
    difference of that same functional.
    """
    import jax.numpy as jnp
    from defuqumat.forces.torque import band_energy_at_angle, torque_at_angle

    scalar, spinor = _tetragonal()
    scf = scalar.get_scf()
    result = run_torque(spinor.system, spinor.pseudos, scf.density)
    assert result.residual * RY_TO_EV * 1000 < 1.0e-6, (
        "sum w <psi|H|psi> does not reproduce sum w eps"
    )

    # The gradient against a central difference of its own functional.
    from defuqumat.scf.continuation import nc_magnetization_from_lsda
    from defuqumat.workflows.anisotropy import _with_quantization_axis
    from defuqumat.workflows.nscf import fixed_density_states

    angle, plane = result.angle, result.plane
    direction = (np.cos(angle) * np.asarray(plane[0])
                 + np.sin(angle) * np.asarray(plane[1]))
    system = _with_quantization_axis(spinor.system, tuple(direction))
    rotated = nc_magnetization_from_lsda(scf.density, tuple(direction))
    calculation, system, eigenvalues, states = fixed_density_states(
        system, spinor.pseudos, rotated, conv_thr=1.0e-10)
    weights, _ = calculation.occupations(jnp.asarray(eigenvalues))

    step = 1.0e-3
    def energy(value):
        return float(band_energy_at_angle(
            calculation, states, weights, scf.density, plane, value))
    difference = (energy(angle + step) - energy(angle - step)) / (2 * step)
    analytic = -torque_at_angle(
        calculation, states, weights, scf.density, plane, angle)
    assert analytic == pytest.approx(difference, rel=1.0e-5)


@pytest.mark.slow
def test_the_forward_torque_is_the_reverse_gradient_at_every_chunk():
    """The dial must not be visible in the answer, and neither must the mode.

    ``E(theta) = sum_k w_k <psi_k|H(theta)|psi_k>`` has no term coupling two
    k-points -- the potential comes from the ``density`` argument rather than
    from the states -- so chunking the k axis is exact rather than an
    approximation. The torque is three (here one) ``jax.jvp`` since
    2026-10-10 (``OPEN.md`` Part XXII item 2), and it is held to ``jax.grad``
    of the same public energy taken here, at the whole axis and at chunks of
    one, three, and four, which does not divide the eighteen k-points, so the
    short tail is differentiated as well. Both modes differentiate one
    function, so they agree to round-off; what checks the rules they share is
    the central difference and the closed form elsewhere in this file.

    The states are held fixed across the calls, so what is compared is the
    derivative and not a second NSCF.
    """
    import jax.numpy as jnp
    from defuqumat.forces.torque import (
        band_energy_at_angle,
        band_energy_at_rotation,
        orientation_torque,
        torque_at_angle,
    )

    from defuqumat.scf.continuation import nc_magnetization_from_lsda
    from defuqumat.workflows.anisotropy import _with_quantization_axis
    from defuqumat.workflows.nscf import fixed_density_states

    scalar, spinor = _tetragonal()
    scf = scalar.get_scf()

    angle = np.pi / 4.0
    plane = ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    direction = (np.cos(angle) * np.asarray(plane[0])
                 + np.sin(angle) * np.asarray(plane[1]))
    system = _with_quantization_axis(spinor.system, tuple(direction))
    rotated = nc_magnetization_from_lsda(scf.density, tuple(direction))
    calculation, system, eigenvalues, states = fixed_density_states(
        system, spinor.pseudos, rotated, conv_thr=1.0e-10)
    weights, _ = calculation.occupations(jnp.asarray(eigenvalues))
    assert np.shape(states)[1] == 18

    reverse = -float(jax.grad(lambda value: band_energy_at_angle(
        calculation, states, weights, scf.density, plane, value))(
            jnp.asarray(angle)))
    for chunk in (None, 1, 3, 4):
        forward = torque_at_angle(calculation, states, weights, scf.density,
                                  plane, angle, k_batch=chunk)
        assert forward == pytest.approx(reverse, rel=1.0e-10, abs=1.0e-15), chunk

    # The vector, on the same states: the same rotation written as a turn of
    # the collinear texture about ``y`` from ``c``.
    texture = nc_magnetization_from_lsda(scf.density, (0.0, 0.0, 1.0))
    base = _rotation_about_y(angle)
    vector_reverse = -np.asarray(jax.grad(lambda omega: band_energy_at_rotation(
        calculation, states, weights, texture, base, omega))(jnp.zeros(3)))
    for chunk in (None, 4):
        vector = orientation_torque(calculation, states, weights, texture, base,
                                    k_batch=chunk)
        np.testing.assert_allclose(vector, vector_reverse, rtol=0,
                                   atol=1.0e-10 * np.linalg.norm(vector_reverse))
    assert vector_reverse[1] == pytest.approx(reverse, rel=1.0e-9)


def _tetragonal_states_at_45_degrees():
    """The one-shot states of tetragonal cobalt with the moment 45 degrees off ``c``.

    ``(calculation, states, weights, density, rotation, direction)``, the
    rotation being the one about ``y`` that takes ``z`` to the moment.
    """
    import jax.numpy as jnp

    from defuqumat.workflows.anisotropy import _rotation_taking, _with_rotation
    from defuqumat.workflows.nscf import fixed_density_states

    scalar, spinor = _tetragonal()
    scf = scalar.get_scf()
    direction = np.array([np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
    rotation = _rotation_taking((0.0, 0.0, 1.0), direction)
    system = _with_rotation(spinor.system, rotation)
    rotated = nc_magnetization_from_lsda(scf.density, tuple(direction))
    calculation, system, eigenvalues, states = fixed_density_states(
        system, spinor.pseudos, rotated, conv_thr=1.0e-10)
    weights, _ = calculation.occupations(jnp.asarray(eigenvalues))
    return calculation, states, weights, scf.density, rotation, direction


@pytest.mark.slow
def test_the_orientation_torque_contains_the_plane_torque():
    """Three components on the same states, and P60's number is one of them.

    ``notes/ORIENTATION-NEXT.md`` step 1. The moment turning in the ``(z, x)`` plane is
    a turn about ``y``, so ``torque_at_angle`` must be the ``y`` component of the
    vector, on the same states, to round-off: the two build the same potential
    through two different parameterisations of the same rotation. The
    component about the moment itself is zero because such a turn moves
    nothing, and it is compared against the ``y`` component so that the zero is
    seen to be one. The third, a tilt out of the ``(z, x)`` plane, is zero by
    symmetry (the mirror ``y -> -y`` combined with time reversal, which maps a
    moment in the ``(z, x)`` plane onto itself and which the unshifted grid
    keeps), so it is bounded by the diagonalisation rather than by round-off:
    measured at 5.8e-11 Ry per radian, 1.4e-6 of the in-plane torque, at the
    one-shot's ``conv_thr = 1e-10``.
    """
    from defuqumat.forces.torque import (
        band_energy_at_rotation,
        orientation_torque,
        torque_at_angle,
    )

    calculation, states, weights, density, rotation, direction = (
        _tetragonal_states_at_45_degrees()
    )
    texture = nc_magnetization_from_lsda(density, (0.0, 0.0, 1.0))
    torque = orientation_torque(calculation, states, weights, texture, rotation)
    plane = torque_at_angle(calculation, states, weights, density,
                            ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)), np.pi / 4)

    assert torque[1] == pytest.approx(plane, rel=1.0e-10)
    assert abs(float(torque @ direction)) < 1.0e-12 * abs(torque[1])
    out_of_plane = float(torque @ np.cross(direction, (0.0, 1.0, 0.0)))
    assert abs(out_of_plane) < 1.0e-5 * abs(torque[1])

    # A central difference of the functional it differentiates, about ``y``,
    # with the exact rotation on either side rather than ``rotation_near``.
    step = 1.0e-3

    def turned(sign):
        return float(band_energy_at_rotation(
            calculation, states, weights, texture,
            _rotation_about_y(sign * step) @ rotation, np.zeros(3)))

    central = (turned(+1) - turned(-1)) / (2 * step)
    assert -torque[1] == pytest.approx(central, rel=1.0e-5)


def _rotation_about_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


@pytest.mark.slow
def test_the_orientation_torque_is_the_plane_torque_end_to_end():
    """Through the front door, two separate one-shot runs, one number.

    ``get_orientation_torque`` at the rotation taking ``c`` 45 degrees towards
    ``a`` against ``get_torque`` at its default of 45 degrees in the ``(z, x)``
    plane: the same system turned by two routes (``_with_rotation`` and
    ``_with_quantization_axis``), the same density, so the same NSCF, and the
    ``y`` component must be P60's torque. It also checks the assembly's own
    identity, ``sum w <psi|H|psi> = sum w eps`` at the orientation the states
    came from.
    """
    from defuqumat.workflows.anisotropy import _rotation_taking

    scalar, spinor = _tetragonal()
    direction = np.array([np.sin(np.pi / 4), 0.0, np.cos(np.pi / 4)])
    rotation = _rotation_taking((0.0, 0.0, 1.0), direction)
    vector = scalar.get_orientation_torque(spinor, rotation=rotation)
    plane = scalar.get_torque(spinor)

    assert vector.torque[1] == pytest.approx(plane.torque, rel=1.0e-9)
    assert vector.residual * RY_TO_EV * 1000 < 1.0e-6
    np.testing.assert_allclose(vector.direction, direction, atol=1e-12)
    assert abs(vector.along_moment) < 1.0e-12 * abs(vector.torque[1])


@pytest.mark.slow
def test_turning_the_texture_leaves_the_hartree_and_xc_energies_alone():
    """The force theorem's premise, for a turn the quantization axis did not follow.

    Every term of the total energy but the band sum is a functional of the
    density that a global spin rotation leaves alone, which is what makes a
    difference of band energies a difference of total energies. Checked at two
    orientations 30 degrees apart built on **one** calculation, whose GGA axis
    sits at the first: for a collinear texture ``sign(m . u)`` is the correct
    signed projection for any axis ``u`` not perpendicular to the moment, so the
    energies must agree to round-off although the axis did not turn. (A turn to
    90 degrees from the axis is the 36.8 meV trap and is not what this checks.)
    """
    from defuqumat.forces.torque import rotate_texture

    calculation, _, _, density, rotation, _ = _tetragonal_states_at_45_degrees()
    texture = nc_magnetization_from_lsda(density, (0.0, 0.0, 1.0))
    first = calculation.potential(rotate_texture(texture, rotation), 1.0, None)
    second = calculation.potential(
        rotate_texture(texture, _rotation_about_y(np.pi / 6) @ rotation), 1.0, None)
    assert float(second.ehart) == pytest.approx(float(first.ehart), abs=1.0e-12)
    assert float(second.etxc) == pytest.approx(float(first.etxc), abs=1.0e-12)


#: An orientation off every symmetry element of tetragonal cobalt, so that all
#: three components of the torque are nonzero and none is a symmetry's zero.
OBLIQUE = (0.4, 0.9, -0.3)


def _oblique_states(conv_thr=1.0e-10, turn_the_lattice=False):
    """``(calculation, states, weights, eigenvalues, levels, texture, rotation)``.

    The one-shot states at :data:`OBLIQUE`. With ``turn_the_lattice`` the spins
    stay where the source had them and the **lattice** is turned by the inverse
    rotation instead, which is the same relative orientation reached by code
    that shares nothing with ``_with_rotation`` or ``rotate_texture``'s ``R``.
    """
    import jax.numpy as jnp

    from defuqumat.forces.torque import rotate_texture
    from defuqumat.workflows.anisotropy import (
        _reference_axis,
        _reference_texture,
        _with_rotation,
        rotation_from_euler,
    )
    from defuqumat.workflows.nscf import fixed_density_states

    scalar, spinor = _tetragonal()
    scf = scalar.get_scf()
    rotation = rotation_from_euler(*OBLIQUE)
    texture = _reference_texture(scf.density, _reference_axis(spinor.system))
    if turn_the_lattice:
        at = np.asarray(spinor.system.cell.at)
        system, applied = spinor.system.with_cell(at @ rotation), np.eye(3)
    else:
        system, applied = _with_rotation(spinor.system, rotation), rotation
    calculation, system, eigenvalues, states = fixed_density_states(
        system, spinor.pseudos, rotate_texture(texture, applied), conv_thr=conv_thr)
    weights, levels = calculation.occupations(jnp.asarray(eigenvalues))
    return calculation, states, weights, eigenvalues, levels, texture, applied


@pytest.mark.slow
def test_the_torque_is_the_exchange_field_acting_on_the_coupled_magnetization():
    """``tau = integral of m_out x B``, with no automatic differentiation in it.

    Only the exchange field carries the rotation, and ``B[R rho] = R B[rho]``,
    so ``dF/dw_a = integral of m_out . (e_a x B)``: the frozen field acting on
    the magnetization of the states solved with the coupling, which leans off
    it by exactly what the coupling does. ``exchange_torque`` integrates it from
    the output density and the potential alone. On an ultrasoft dataset the
    output density must carry its augmentation for this to hold, since
    ``D_ij`` enters the band energy as ``integral of V Q_ij``. Measured at
    1.7e-12 relative, all three components live.
    """
    from defuqumat.forces.torque import orientation_torque, rotate_texture
    from defuqumat.scf.spin_torque import exchange_torque

    calculation, states, weights, _, _, texture, rotation = _oblique_states()
    torque = orientation_torque(calculation, states, weights, texture, rotation)
    output = calculation.density(states, weights)
    potential = calculation.potential(rotate_texture(texture, rotation), 1.0, None)
    closed = np.asarray(exchange_torque(output, potential.v_scf,
                                        calculation.system.cell).total)

    assert np.min(np.abs(torque)) > 1.0e-7, "a component is a symmetry's zero"
    np.testing.assert_allclose(closed, torque, rtol=1.0e-9, atol=1.0e-15)


@pytest.mark.slow
def test_turning_the_lattice_is_turning_the_spins_the_other_way(monkeypatch):
    """The same relative orientation by two routes that share no rotation code.

    Turning every spin by ``R`` in a fixed lattice is the same calculation as
    turning the lattice (and the atoms, whose crystal coordinates stay) by
    ``R^-1`` under fixed spins, so the free energies are equal and the torques
    are related by ``tau = R tau'``. It is Elk's own way of changing the
    orientation (``mae.f90`` rotates ``avec``).

    **The torque's agreement is set by the eigensolver, not by the rotation.**
    The energies agree to 1.3e-15 Ry at any threshold, being second order in the
    eigenvectors' error, while a gradient at frozen states is first order in it:
    at the default floor ``ETHR_MIN = 1e-13`` the torques agree to 1.05e-6
    relative and stop improving (a one-shot ``conv_thr`` of 1e-12 and 1e-14
    both reach the floor), and with the floor lowered they agree to 1.5e-10 at
    ``ethr = 1.1e-16`` and 4.1e-11 at 1e-17. So the floor is lowered here, and
    the default floor's torque noise, about 7e-11 Ry per radian on this cell,
    is the resolution a relaxation's gradient threshold has to sit above.
    """
    import defuqumat.workflows.nscf as nscf
    from defuqumat.forces.torque import orientation_torque

    monkeypatch.setattr(nscf, "ETHR_MIN", 1.0e-17)
    energies, torques = [], []
    for turn_the_lattice in (False, True):
        calculation, states, weights, eigenvalues, levels, texture, applied = (
            _oblique_states(conv_thr=1.0e-16, turn_the_lattice=turn_the_lattice))
        energies.append(float(np.sum(np.asarray(weights) * np.asarray(eigenvalues)))
                        + float(levels.get("smearing", 0.0)))
        torques.append(orientation_torque(calculation, states, weights, texture,
                                          applied))
    from defuqumat.workflows.anisotropy import rotation_from_euler

    rotation = rotation_from_euler(*OBLIQUE)
    assert energies[1] == pytest.approx(energies[0], abs=1.0e-12)
    np.testing.assert_allclose(rotation @ torques[1], torques[0], rtol=0,
                               atol=1.0e-9 * np.linalg.norm(torques[0]))


@pytest.mark.slow
def test_without_the_coupling_the_torque_on_a_texture_vanishes():
    """``soc_scale = 0`` on the same file: every component zero, beside a live one.

    Without the coupling the Hamiltonian at ``R rho`` is the one at ``rho``
    conjugated by a spin rotation, so the band energy does not depend on ``R``
    and neither does anything derived from it. The zero is read against the
    same torque with the coupling on, so that it is seen to be a zero and not a
    silence: 1.3e-10 against 4.0e-5 Ry per radian, which is the default
    eigensolver floor's noise (see the lattice test above).
    """
    from defuqumat.workflows.anisotropy import rotation_from_euler, run_orientation_torque

    scalar, spinor = _tetragonal()
    scf = scalar.get_scf()
    rotation = rotation_from_euler(*OBLIQUE)
    live = run_orientation_torque(spinor.system, spinor.pseudos, scf.density,
                                  rotation=rotation)
    off = run_orientation_torque(spinor.system, spinor.pseudos, scf.density,
                                 rotation=rotation, soc_scale=0.0)
    assert np.linalg.norm(live.torque) > 1.0e-5
    assert np.linalg.norm(off.torque) < 2.0e-5 * np.linalg.norm(live.torque)


@pytest.mark.slow
def test_the_orientation_relaxes_onto_the_easy_axis():
    """Tetragonal cobalt from an oblique start lands on ``c``, the easy axis.

    ``notes/ORIENTATION-NEXT.md`` step 3. Started 51.6 degrees from ``c`` with every
    component of the torque live, the BFGS in the rotation vector converges on
    ``c``, which P60 and P87 both have as the easy axis. The curvature there is
    the second number: one eigenvalue is the turn about the moment, which moves
    nothing, and the other two are the tilts, equal by the four-fold axis, and
    ``2 K1`` for ``E = K1 sin^2 + K2 sin^4``, where the 45-degree torque reads
    ``K1 + K2``; their difference is ``K2``.
    """
    from defuqumat.workflows.anisotropy import rotation_from_euler

    scalar, spinor = _tetragonal()
    relaxed = scalar.get_relaxed_orientation(
        spinor, rotation=rotation_from_euler(*OBLIQUE), curvature=True)

    assert relaxed.converged
    assert abs(relaxed.direction[2]) > 1.0 - 1.0e-6
    eigenvalues = relaxed.curvature_eigenvalues
    tilts = eigenvalues[1:]
    assert abs(eigenvalues[0]) < 1.0e-3 * tilts.min()
    assert tilts[1] == pytest.approx(tilts[0], rel=1.0e-3)
    # 2 K1 against the 45-degree torque's K1 + K2 = 4.059e-5 Ry.
    assert 0.5 * tilts.mean() == pytest.approx(4.059e-5, rel=0.2)


def _oblique_coupled_cobalt():
    """Tetragonal cobalt with the coupling, self-consistent, seeded 45 degrees off ``c``."""
    text = (GENERATED / "co-tetragonal-relaxed-mae.in").read_text().replace(
        "angle1(1) = 0.0, angle2(1) = 0.0,", "angle1(1) = 45.0, angle2(1) = 0.0,")
    assert "angle1(1) = 45.0" in text
    return Calculator.from_text(text, pseudo_dir=GENERATED.parent / "pseudo",
                                announce=False)


@pytest.mark.slow
def test_turning_the_moments_inside_the_scf_converges_on_the_easy_axis():
    """Route C: the run with the coupling converges on ``c`` instead of wandering.

    ``notes/ORIENTATION-NEXT.md`` step 4. Seeded 45 degrees off ``c``, the plain SCF
    does not converge in 100 iterations: the angle wanders between 36 and 54
    degrees and ``dr2`` between 1e-8 and 2e-5 while the per-iteration torque
    sits at -3.3e-5 Ry per radian, which is ``-K1`` of the relaxed anisotropy
    (P87's 0.447 meV) read from one run. With the moments turned by the torque
    after every mix it converges in 44 iterations 0.002 degrees from ``c``, and
    its total energy is the SCF started exactly along ``c`` to 1.3e-8 Ry, sixty
    times inside the 8e-7 Ry floor P87 measured between two paths to one state.
    """
    turned = _oblique_coupled_cobalt().get_scf(
        rotate_moments=True, max_iterations=150, conv_thr=1.0e-12,
        torque_conv_thr=1.0e-9)
    along = Calculator.from_file(GENERATED / "co-tetragonal-relaxed-mae.in",
                                 pseudo_dir=GENERATED.parent / "pseudo",
                                 announce=False).get_scf(conv_thr=1.0e-12)
    moment = np.asarray(turned.magnetization_vector)
    angle = np.degrees(np.arccos(moment[2] / np.linalg.norm(moment)))

    assert turned.converged
    assert angle < 0.05
    assert np.linalg.norm(turned.orientation_torque) < 1.0e-9
    assert turned.total_energy == pytest.approx(along.total_energy, abs=1.0e-7)
    # The torque was recorded every iteration, and early on it was the size of
    # the anisotropy: the blind spot of dr2 that the option exists for.
    torques = [np.linalg.norm(e["orientation_torque"]) for e in turned.history]
    assert max(torques[5:15]) > 1.0e-5

    # **The drift pair**, which is what says the state is a stable fixed point of
    # the loop with nothing turning it, where the energy above only says the
    # run reached the symmetric point. A plain SCF from the converged density
    # stays (measured: 8 iterations, 0.0023 degrees, torque 9.8e-10), and from
    # the same density turned 5 degrees off ``c`` the torque is restoring and
    # the run ends nearer ``c`` than it started, so the guard fires. **How far
    # it gets is not asserted**: it is the soft mode Route C exists for, and it
    # measured 0.026 degrees after 73 iterations in one run and 4.4 degrees,
    # unconverged, after 150 in another, from end states of Route C a hair
    # apart. An iteration budget on a plain SCF along that mode is a bet.
    from defuqumat.forces.torque import rotate_texture
    from defuqumat.scf.driver import run_scf

    calculator = _oblique_coupled_cobalt()

    def angle_of(result):
        m = np.asarray(result.magnetization_vector)
        return np.degrees(np.arccos(abs(m[2]) / np.linalg.norm(m)))

    stays = run_scf(calculator.system, calculator.pseudos,
                    starting_density=turned.density, conv_thr=1.0e-12,
                    max_iterations=100)
    assert stays.converged and stays.iterations < 20
    assert angle_of(stays) < 0.01
    c, s_ = np.cos(np.radians(5.0)), np.sin(np.radians(5.0))
    tilted = rotate_texture(turned.density,
                            np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]]))
    back = run_scf(calculator.system, calculator.pseudos, starting_density=tilted,
                   conv_thr=1.0e-12, max_iterations=150)
    # Turned 5 degrees about +y, so a restoring torque is along -y.
    assert back.history[0]["orientation_torque"][1] < -1.0e-6
    assert angle_of(back) < 5.0


def _cobalt_in_a_field(b_x: float, angle: float = 30.0):
    """Tetragonal cobalt with the coupling, a uniform field ``B_field(1) = b_x`` Ry along ``x``."""
    text = (GENERATED / "co-tetragonal-relaxed-mae.in").read_text().replace(
        "angle1(1) = 0.0, angle2(1) = 0.0,",
        f"angle1(1) = {angle}, angle2(1) = 0.0,\n   B_field(1) = {b_x},")
    assert f"B_field(1) = {b_x}" in text
    return Calculator.from_text(text, pseudo_dir=GENERATED.parent / "pseudo",
                                announce=False)


@pytest.mark.slow
def test_turning_the_moments_in_a_field_balances_the_anisotropy_against_the_zeeman_term():
    """I113: ``rotate_moments`` under a field along ``x`` stops where ``dE/dtheta = B M cos(theta)``.

    The anisotropy of this cell is ``E = K1 sin^2 + K2 sin^4`` to the order the
    check needs, and the field's energy ``-B M sin(theta)``, so the minimum
    satisfies ``kappa = B M / (2 sin(theta)) = K1 + 2 K2 sin^2(theta)``. Two
    fields give ``K1`` and ``K2``, and ``K1 + K2`` must be the relaxed
    anisotropy ``E(x) - E(z)`` of two plain runs, which share no machinery with
    the step. Measured on D22 (``notes/incompat/I113.md``): 25.974 and 55.712
    degrees at 1.5e-5 and 2.9e-5 Ry, ``K1 + K2 = 3.1032e-5`` against
    ``3.1054e-5`` Ry, 0.07 per cent, where one field alone reads the relaxed
    anisotropy to -1.6 and +0.9 per cent, which is ``2 K2 (sin^2 - 1/2)``. With
    the field's share left in the step the run sat at 0.04 degrees, on the
    easy axis, unconverged after 150 iterations.
    """
    options = dict(conv_thr=1.0e-12, max_iterations=200)
    along_z = _cobalt_in_a_field(0.0, angle=0.0).get_scf(**options)
    along_x = _cobalt_in_a_field(0.0, angle=90.0).get_scf(**options)
    relaxed = along_x.total_energy - along_z.total_energy
    assert relaxed == pytest.approx(3.1054e-5, rel=1.0e-3)

    points = []
    for b in (1.5e-5, 2.9e-5):
        result = _cobalt_in_a_field(b).get_scf(rotate_moments=True,
                                               torque_conv_thr=1.0e-9, **options)
        assert result.converged
        m = np.asarray(result.magnetization_vector)
        sine = np.hypot(m[0], m[1]) / np.linalg.norm(m)
        assert abs(m[1]) < 1.0e-6
        points.append((sine ** 2, b * np.linalg.norm(m) / (2.0 * sine)))
    (s1, k1), (s2, k2) = points
    assert np.degrees(np.arcsin(np.sqrt(s1))) == pytest.approx(25.974, abs=0.01)
    assert np.degrees(np.arcsin(np.sqrt(s2))) == pytest.approx(55.712, abs=0.01)
    slope = (k2 - k1) / (s2 - s1)        # 2 K2
    first = k1 - slope * s1              # K1
    # -0.074 per cent measured. Not K3: a three-field fit gives K3 = -1.35e-8 Ry
    # and leaves the gap (verifier). Cause not identified; the candidate is the
    # 0.076 per cent difference of |M| between the plain z and x runs.
    assert first + 0.5 * slope == pytest.approx(relaxed, rel=5.0e-3)
    # One field alone is off by 2 K2 (sin^2 - 1/2), which is what the
    # extrapolation removes.
    assert abs(k1 - relaxed) > 1.0e-2 * relaxed


def _cobalt_with_u(angle: float):
    """Tetragonal cobalt with the coupling and ``U(Co-3d) = 1`` eV, ortho-atomic, ``nosym``."""
    text = (GENERATED / "co-tetragonal-relaxed-mae.in").read_text().replace(
        "angle1(1) = 0.0, angle2(1) = 0.0,", f"angle1(1) = {angle}, angle2(1) = 0.0,")
    assert f"angle1(1) = {angle}" in text
    text = text + "HUBBARD {ortho-atomic}\nU Co-3d 1.0\n"
    return Calculator.from_text(text, pseudo_dir=GENERATED.parent / "pseudo",
                                announce=False)


@pytest.mark.slow
def test_the_torque_with_a_hubbard_u_is_the_derivative_of_the_frozen_band_energy(monkeypatch):
    """I112: the torque ``rotate_moments`` steps on, with a U, is ``-dF/dtheta`` at a frozen input.

    The converged state along ``c`` is turned rigidly by ``theta`` about ``y``,
    the density by ``rotate_texture`` and ``ns`` by ``rotate_ns``, and
    diagonalised once at a tight threshold. The band free energy of that one
    diagonalisation has the derivative ``sum f <dH/dtheta>``, which is minus
    the recorded torque, the grid integral plus DFT+U's share; a central
    difference at ``h = 0.5`` degree shares no machinery with the torque.
    Measured on D22 (``notes/incompat/I112.md``): -1.5e-5 relative at 20 and
    45 degrees (-2.4e-4 at ``h = 2``, the ``h^2`` of the stencil), and +0.52
    with DFT+U's share left out.
    """
    import jax.numpy as jnp

    import defuqumat.scf.driver as driver
    from defuqumat.forces.torque import rotate_texture
    from defuqumat.scf.orientation import rotate_ns, rotation_matrix

    shares = []
    original = driver._hubbard_torque

    def logged(ns_out, v_ns):
        value = original(ns_out, v_ns)
        shares.append(np.asarray(value))
        return value

    monkeypatch.setattr(driver, "_hubbard_torque", logged)
    calc = _cobalt_with_u(0.0)
    reference = calc.get_scf(conv_thr=1.0e-12, max_iterations=150)
    assert reference.converged
    rho0, ns0 = np.asarray(reference.density), np.asarray(reference.ns)

    def frozen(theta):
        omega = np.array([0.0, np.radians(theta), 0.0])
        shares.clear()
        result = driver.run_scf(
            calc.system, calc.pseudos,
            starting_density=jnp.asarray(rotate_texture(rho0, rotation_matrix(omega))),
            starting_ns=jnp.asarray(rotate_ns(ns0, omega)),
            max_iterations=1, diago_thr_init=1.0e-12, conv_thr=1.0e-30)
        free = float(np.sum(np.asarray(result.occupations) * np.asarray(result.eigenvalues))
                     + result.energy_terms.get("smearing", 0.0))
        return free, np.asarray(result.history[0]["orientation_torque"]), shares[0]

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")    # the one-iteration runs do not converge, by design
        _, torque, hubbard = frozen(20.0)
        h = 0.5
        derivative = (frozen(20.0 + h)[0] - frozen(20.0 - h)[0]) / (2.0 * np.radians(h))
    assert (derivative + torque[1]) / abs(torque[1]) == pytest.approx(0.0, abs=1.0e-4)
    # Measured 0.5225 and 0.5227 at 20 degrees (h = 2, 0.5), 0.5248 and 0.5250 at 45.
    without = (derivative + torque[1] - hubbard[1]) / abs(torque[1])
    assert without == pytest.approx(0.5227, abs=0.01)


@pytest.mark.slow
def test_turning_the_moments_with_a_hubbard_u_stops_on_a_fixed_point_of_the_plain_loop():
    """I112: ``rotate_moments`` with a U ends where a plain SCF from its state does not move.

    From 45 degrees the run turns the density, the mixer history and ``ns``
    together and stops at 38.557 degrees from ``c`` (``torque_conv_thr =
    1e-9``). A plain SCF from its density and ``ns`` converges in about ten
    iterations and moves the moment by 3e-5 degree and the energy by 1e-12
    Ry. The end state is a branch of the occupation matrix that the atomic
    start does not reach on the axes, which ``notes/incompat/I112.md`` records.
    """
    from defuqumat.scf.driver import run_scf

    calc = _cobalt_with_u(45.0)
    turned = calc.get_scf(rotate_moments=True, torque_conv_thr=1.0e-9, conv_thr=1.0e-12,
                          max_iterations=300)
    assert turned.converged
    m = np.asarray(turned.magnetization_vector)
    angle = np.degrees(np.arctan2(np.hypot(m[0], m[1]), m[2]))
    assert angle == pytest.approx(38.557, abs=0.01)
    stays = run_scf(calc.system, calc.pseudos, starting_density=turned.density,
                    starting_ns=turned.ns, conv_thr=1.0e-12, max_iterations=100)
    assert stays.converged
    n = np.asarray(stays.magnetization_vector)
    assert np.degrees(np.arctan2(np.hypot(n[0], n[1]), n[2])) == pytest.approx(angle, abs=1e-3)
    assert stays.total_energy == pytest.approx(turned.total_energy, abs=1.0e-9)


def test_turning_the_moments_without_the_coupling_is_refused():
    """Without the coupling the orientation is not a coordinate, so no step is taken."""
    from defuqumat.scf.driver import run_scf

    calculator = _oblique_coupled_cobalt()
    with pytest.raises(ValueError, match="spin-orbit"):
        run_scf(calculator.system.with_soc_scale(0.0), calculator.pseudos,
                rotate_moments=True)


def _oblique_coupled_cobalt_with_symmetry():
    """The same cell with ``nosym`` removed: the 45-degree start's group, ``nsym = 4``, 8 points."""
    text = (GENERATED / "co-tetragonal-relaxed-mae.in").read_text().replace(
        "angle1(1) = 0.0, angle2(1) = 0.0,", "angle1(1) = 45.0, angle2(1) = 0.0,"
    ).replace("nosym = .true.,", "")
    assert "nosym" not in text.split("&system")[1].split("/")[0]
    return Calculator.from_text(text, pseudo_dir=GENERATED.parent / "pseudo",
                                announce=False)


def test_turning_the_moments_on_the_starting_orientation_s_group_is_refused():
    """I78: the k-set reduced with the 45-degree start's magnetic group is refused.

    The group of the start (``E``, ``I`` and the two-fold axis along ``y`` and
    its mirror with time reversal) is kept by every step of the stepper, so its
    8-point wedge stays complete (with the guard bypassed it ends 1.8e-13 Ry
    from ``nosym``, ``notes/incompat/I78.md``); it is refused because it holds
    the moment exactly in the ``xz`` plane, and accepting that is the user's
    open choice. The message must name the way out.
    """
    from defuqumat.scf.driver import run_scf

    calculator = _oblique_coupled_cobalt_with_symmetry()
    assert calculator.system.kpoints.nk == 8
    with pytest.raises(ValueError, match="with_turning_moments"):
        run_scf(calculator.system, calculator.pseudos, rotate_moments=True)


@pytest.mark.slow
def test_turning_the_moments_on_the_turning_group_is_the_nosym_run():
    """I78: ``with_turning_moments`` (``{E, I}``, 10 points) against ``nosym`` (18 points).

    Inversion acts on an axial vector as the identity, so it is a symmetry of
    every turned texture and ``E(k) = E(-k)`` holds along the whole run: the
    wedge sum is exact, and the run on 10 points must converge on ``c`` with the
    energy of the run on 18. Measured on D22 (``notes/incompat/I78.md``, one
    sample each): 52 iterations against 53, both converged, 0.0019 and 0.0011
    degrees from ``c``, torque 9.6e-10 and 9.2e-11 Ry/rad, and total energies
    5.7e-14 Ry apart. The 1e-9 asserted below is the stepper's freedom to stop
    at a different point inside ``torque_conv_thr``, not that spread.
    """
    options = dict(rotate_moments=True, max_iterations=150, conv_thr=1.0e-12,
                   torque_conv_thr=1.0e-9)
    reduced = _oblique_coupled_cobalt_with_symmetry().with_turning_moments()
    assert reduced.system.kpoints.nk == 10
    full = _oblique_coupled_cobalt()
    assert full.system.kpoints.nk == 18
    turned, reference = reduced.get_scf(**options), full.get_scf(**options)

    def angle_of(result):
        m = np.asarray(result.magnetization_vector)
        return np.degrees(np.arccos(abs(m[2]) / np.linalg.norm(m)))

    assert turned.converged and reference.converged
    assert angle_of(turned) < 0.05
    assert np.linalg.norm(turned.orientation_torque) < 1.0e-9
    assert turned.total_energy == pytest.approx(reference.total_energy, abs=1.0e-9)


def test_the_rotation_plane_must_be_orthogonal():
    scalar, spinor = _tetragonal()
    with pytest.raises(ValueError, match="orthogonal"):
        run_torque(spinor.system, spinor.pseudos, np.zeros((2, 4, 4, 4)),
                   plane=((0, 0, 1), (0, 0.3, 1)))
