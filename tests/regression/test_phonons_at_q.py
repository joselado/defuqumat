"""P71: the dynamical matrix at ``q != 0``, and the four things it is checked against.

The zone centre is a special case in every part of a phonon calculation, and
almost all of the ways it is special make a wrong finite-``q`` answer look
right. ``dpsi`` lives on the same sphere as ``psi``; the response density is
real; the ``G = 0`` Hartree term is dropped; the acoustic sum rule holds; the
matrix is real and symmetric. Each of those stops being true away from
``Gamma``, so the checks here are chosen to be ones ``Gamma`` cannot pass by
accident:

* **``q = 0`` through the two-sphere path.** The regression, and the reason it
  is first: everything new -- the second sphere, the shifted local potential,
  the cross-sphere nonlocal term, the transcribed ``d2ionq``, the contraction
  that replaces ``drhodv`` -- has to reproduce P25's matrix, which is validated
  against ``ph.x`` independently. What is left over is the deliberate Ewald
  swap and is measured rather than tolerated.
* **``ph.x`` at ``L`` and at ``X``**, from ``reference.out.ph-si-epsilon-unshifted-dispersion``
  -- one ``ldisp`` run on the 2x2x2 q-grid, which is commensurate with the
  4x4x4 k-grid both codes sample.
* **``D(q + G) = D(q)``.** The dynamical matrix is periodic in the reciprocal
  lattice, and the two sides of that identity are built on *different* plane-wave
  spheres from *different* diagonalisations -- so it is the check that the sphere
  at ``k + q`` and the phases that go with it are consistent, which is P16's
  zone-edge trap in a second place.
* **``D(-q) = conj(D(q))`` and ``D(q)`` hermitian.** Time reversal and the
  reality of the force constants, neither of which anything here imposes.

The acoustic sum rule is *not* among them: it holds at ``Gamma`` and nowhere
else, which is exactly why a finite-``q`` matrix has one fewer free diagnostic
than P25's does.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response.phonon import dynamical_matrix
from defuqumat.response.phononq import (
    dynamical_matrix_at_q,
    ewald_dynamical_matrix,
    frozen_force_constants,
    require_a_two_sphere_regime,
)
from defuqumat.response.sternheimer import make_sternheimer
from defuqumat.scf import Calculation, run_scf
from defuqumat.system import build_system

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: What the **vendored** ``ph.x`` prints for the 2x2x2 q-grid of
#: ``si-epsilon-unshifted.in``, committed as
#: ``reference.out.ph-si-epsilon-unshifted-dispersion``. Three wavevectors from
#: one run: the zone centre, ``L`` and ``X``, in cartesian units of 2 pi/alat.
#: The two zone-boundary sets are the quantity this phase exists for -- nothing
#: before it could compute a frequency anywhere but the first row.
QE_FREQUENCIES = {
    "Gamma": ((0.0, 0.0, 0.0),
              (3.273643, 3.273643, 3.273643,
               519.198154, 519.198154, 519.198154)),
    "L": ((0.5, -0.5, 0.5),
          (101.842827, 101.842827, 382.240553,
           405.120790, 488.019400, 488.019400)),
    "X": ((0.0, -1.0, 0.0),
          (132.805268, 132.805268, 405.358862,
           405.358862, 455.138711, 455.138711)),
}


@pytest.fixture(autouse=True)
def _bounded_compilation_cache():
    """``CLAUDE.md``'s bound for a file that runs several cells and several q.

    Each ``q`` builds a second plane-wave sphere and therefore a second set of
    shapes, so this file compiles the whole response stack once per wavevector
    and XLA would keep every one of them for the life of the process.
    """
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _ground_state(case: str):
    """The converged run. ``maxsize=2`` for the reason ``CLAUDE.md`` gives."""
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    result = run_scf(system, pseudos, calculation=calculation,
                     conv_thr=1e-12, max_iterations=100)
    return calculation, result


def _phonons(case: str, q, **options):
    calculation, result = _ground_state(case)
    return dynamical_matrix_at_q(
        calculation, result.wavefunctions, result.eigenvalues, result.density,
        result.becsum, q=q, **options,
    )


# ---------------------------------------------------------------------------
# Against ph.x.
# ---------------------------------------------------------------------------

#: How far a frequency may sit from ``ph.x``'s, in cm^-1. Measured: **0.054**
#: at ``L``, the worst of the twelve zone-boundary modes. The floor is the same
#: one P25 quotes at ``Gamma`` (0.05 on this cell) and has the same cause -- QE
#: interpolates every radial form factor from a ``dq = 0.01`` table where this
#: code integrates it directly -- so there is no reason for a zone-boundary
#: mode to hold it more tightly than a zone-centre one.
FREQUENCY_TOLERANCE = 0.2

#: The three **acoustic** modes at ``Gamma`` are exempt, and that is a
#: statement about the reference rather than about this code. Translating the
#: crystal costs no energy, so what those three report is the finite basis's own
#: residue -- and the two runs make it differently, because ``ph.x`` is on the
#: eight-point wedge with the response symmetrised and this is on the whole
#: 64-point grid. 4.32 against 3.27, both of them the same 1e-4 of the force
#: constants. Away from ``Gamma`` there is no acoustic triplet and nothing is
#: exempt, which is why the zone-boundary rows are the sharper comparison.
ACOUSTIC_CEILING = 8.0


@pytest.mark.parametrize("point", ["Gamma", "L", "X"])
def test_the_dispersion_matches_quantum_espresso(point):
    """Silicon at ``Gamma``, ``L`` and ``X`` against the vendored ``ph.x``.

    The zone-boundary rows are what P71 exists for. Nothing in the response
    stack before it could produce a frequency at a wavevector other than zero,
    because a displacement pattern ``u e^{iqR}`` is not a displacement of the
    unit cell and there is no energy in this cell to differentiate twice. What
    makes it reachable is that the *first-order* quantities are still lattice
    periodic once ``e^{iqr}`` is factored out, so the whole calculation stays
    in the unit cell with one extra plane-wave sphere.

    ``ph.x`` runs on the eight-point wedge with the response symmetrised over
    the small group of ``q``; this runs the whole 64-point grid with nothing
    symmetrised at all. So the comparison is also the only check the two routes
    get against each other, in the way P25's wedge-against-grid test is at
    ``Gamma``.
    """
    q, reference = QE_FREQUENCIES[point]
    phonons = _phonons("si-epsilon-unshifted-nosym", q, q_cartesian=True)
    assert phonons.converged
    frequencies = np.asarray(phonons.frequencies)
    if point == "Gamma":
        assert np.abs(frequencies[:3]).max() < ACOUSTIC_CEILING
        frequencies, reference = frequencies[3:], np.asarray(reference)[3:]
    assert np.allclose(frequencies, np.asarray(reference),
                       atol=FREQUENCY_TOLERANCE), (
        f"{point}: {frequencies} against ph.x's {reference}"
    )


# ---------------------------------------------------------------------------
# Against the Gamma machinery.
# ---------------------------------------------------------------------------

#: What ``q = 0`` through the two-sphere path may differ from P25's matrix by,
#: in Ry/bohr^2 on force constants of 0.287. Measured: **1.94e-6**, and it is
#: not noise -- it is the whole of the deliberate Ewald swap. P25's frozen
#: Hessian contains the ground state's own Ewald sum, whose real-space cutoff is
#: ``4/sqrt(alpha)``: ``erfc(4) ~ 2e-8`` bounds the error in the *energy* and
#: not in a second derivative, which carries ``1/r^5``. ``d2ionq.f90`` uses
#: ``5/sqrt(alpha)`` for exactly that reason and this follows it. Extending the
#: ground-state sum to the same cutoff brings the two into agreement at
#: **1.07e-9**, which is what says the residue is that and nothing else.
GAMMA_AGREEMENT = 5e-6


def test_q_zero_reproduces_the_gamma_machinery():
    """Everything new, run at the wavevector where the old code also works.

    This is the regression the whole phase rests on, because the four things
    that change away from ``Gamma`` -- the second sphere, the shifted local
    potential, the ``G = 0`` Hartree term, the complex response density -- all
    have a limit here that is already validated against ``ph.x`` through P25.
    A sign or an index order that is wrong at finite ``q`` is very often still
    wrong at zero, where there is something to compare it with.

    **The two loops are compared at one ``tr2``.** Since P149 the at-``q`` route
    runs its scheduled solves capped at 1e-8 with ``tr2 = 1e-16``
    (:data:`~defuqumat.response.phononq.TR2_AT_Q_CAPPED`), while ``Gamma``'s
    default was ``ph.x``'s 1e-14 until 2026-10-15 and is 1e-18 since, and the
    ``Gamma`` loop alone moved by 1.45e-5 between 1e-14 and 1e-16 (its
    acoustic residue 5.68 to 4.45 cm^-1, D22, 2026-10-14); like for like the routes agree to 2.2e-6 capped and 1.95e-6
    uncapped at either ``tr2``, which is the Ewald residue
    :data:`GAMMA_AGREEMENT` is about. So ``Gamma`` is asked for the at-``q``
    default's ``tr2`` here, and the bound keeps its meaning.
    """
    from defuqumat.response.phononq import TR2_AT_Q_CAPPED

    calculation, result = _ground_state("si-epsilon-unshifted-nosym")
    gamma = dynamical_matrix(
        calculation, result.wavefunctions, result.eigenvalues, result.density,
        result.becsum, tr2=TR2_AT_Q_CAPPED,
    )
    here = _phonons("si-epsilon-unshifted-nosym", (0.0, 0.0, 0.0))
    difference = np.abs(
        np.asarray(here.matrix).reshape(6, 6)
        - np.asarray(gamma.matrix).reshape(6, 6)
    ).max()
    assert difference < GAMMA_AGREEMENT, difference
    # The matrix is complex by construction and real by physics at q = 0.
    assert np.abs(np.imag(np.asarray(here.matrix))).max() < 1e-7


# ---------------------------------------------------------------------------
# A metal at q = 0: ph.x's ef_shift on this route (I4).
# ---------------------------------------------------------------------------

#: ``reference.out.ph-al2-metal:435-437``, the three optical modes ``ph.x``
#: prints for ``al2-metal.in`` at ``Gamma``, in cm^-1.
QE_AL2_OPTICAL = (146.710511, 146.714378, 311.035401)

#: In cm^-1. Measured 2026-10-16 on D22 (CPU, one core): **2.9e-3** on the
#: worst of the three, 146.709565, 146.713471 and 311.032515, where the
#: ``Gamma`` route on the same state gives 3.2e-3. The bound is
#: ``test_phonons.py``'s ``AL2_OPTICAL_TOLERANCE``, for its reason.
AL2_OPTICAL_TOLERANCE = 0.05

#: ``q = 0`` through this route against the ``Gamma`` route on the same metal,
#: in Ry/bohr^2. Measured 2026-10-16 on D22: **5.7e-7** on ``al2-metal.in``
#: (on-site 0.048) and **7.2e-7** on :data:`_OFF_CENTRE` (on-site 0.069), the
#: Ewald swap :data:`GAMMA_AGREEMENT` describes on silicon.
METAL_GAMMA_AGREEMENT = 5e-6

#: ``al2-metal.in`` with the second atom moved off its inversion centre, to
#: (-0.45, 0.50, 0.08) alat. On ``al2-metal.in`` every atom sits at an inversion
#: centre, so ``def`` is zero by symmetry for every mode (a scalar odd in the
#: displacement) and the route with the Fermi level held gives the same matrix
#: to 1e-10: a check there could not have read differently. Here ``def`` is
#: 1.13e-2 to 1.22e-2 per mode on both routes (agreeing to 1e-7), and holding
#: the level is worth **5.1e-2** Ry/bohr^2 on the matrix and an imaginary mode
#: at -197 cm^-1, measured 2026-10-16 on D22.
_OFF_CENTRE = """&control
  calculation = 'scf'
/
&system
  ibrav = 0, celldm(1) = 7.50, nat = 2, ntyp = 1, ecutwfc = 15.0,
  nosym = .true., noinv = .true.,
  occupations = 'smearing', smearing = 'marzari-vanderbilt', degauss = 0.05
/
&electrons
  conv_thr = 1.0d-12
/
CELL_PARAMETERS alat
  -0.50  0.00  0.50
   0.00  0.50  0.50
  -1.00  1.00  0.00
ATOMIC_SPECIES
 Al 26.98 Al.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Al  0.00  0.00  0.00
 Al -0.45  0.50  0.08
K_POINTS automatic
 4 4 2 0 0 0
"""


def _metal_pair(calculator):
    """``(Gamma route, q route at q = 0)`` on one converged state, at one ``tr2``."""
    from defuqumat.response.phononq import TR2_AT_Q_CAPPED

    gamma = calculator.get_phonons(tr2=TR2_AT_Q_CAPPED)
    here = calculator.get_phonons_at_q((0.0, 0.0, 0.0))
    assert gamma.converged and here.converged
    difference = np.abs(np.asarray(here.matrix) - np.asarray(gamma.matrix)).max()
    assert np.abs(np.imag(np.asarray(here.matrix))).max() < 1e-7
    return gamma, here, difference


def test_a_metal_at_q_zero_matches_quantum_espresso():
    """``al2-metal.in`` at ``q = 0`` through two spheres, against ``ph.x``.

    The route carries ``dfpt_kernels.f90``'s three ``lmetq0`` terms (``def
    ldos`` in the density, ``def`` off ``dV_scf``, ``ef_shift_wfc`` on
    ``dpsi``). On this cell ``def`` vanishes by symmetry, so this test is the
    route's metallic assembly at ``q = 0``;
    :func:`test_a_metal_off_an_inversion_centre_moves_the_fermi_level` is the
    one the shift itself is checked by.
    """
    from defuqumat import Calculator

    calculator = Calculator.from_file(CASES / "al2-metal.in", pseudo_dir=PSEUDO)
    calculator.get_scf()
    _, here, difference = _metal_pair(calculator)
    assert difference < METAL_GAMMA_AGREEMENT, difference
    optical = np.sort(np.asarray(here.frequencies))[3:]
    assert optical[1] - optical[0] == pytest.approx(
        QE_AL2_OPTICAL[1] - QE_AL2_OPTICAL[0], abs=1e-3)
    assert np.abs(optical - np.asarray(QE_AL2_OPTICAL)).max() < AL2_OPTICAL_TOLERANCE


def test_a_metal_off_an_inversion_centre_moves_the_fermi_level():
    """Where ``def != 0``, the route at ``q = 0`` is the ``Gamma`` route.

    The ``Gamma`` route's ``ef_shift`` is the one validated against ``ph.x``
    (``test_phonons.py``), and the two share the shift's arithmetic but not
    the sphere, the projector, the kernel or the assembly; without the shift
    this cell's matrices are 5.1e-2 apart (:data:`_OFF_CENTRE`).
    """
    import warnings

    from defuqumat import Calculator

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculator = Calculator.from_text(_OFF_CENTRE, str(PSEUDO), announce=False)
    calculator.get_scf()
    gamma, here, difference = _metal_pair(calculator)
    assert difference < METAL_GAMMA_AGREEMENT, difference
    # The optical modes, 127.06, 202.55 and 343.30 cm^-1, measured 1.4e-4 apart.
    assert np.abs(np.sort(here.frequencies)[3:]
                  - np.sort(gamma.frequencies)[3:]).max() < 1e-3


def test_the_ewald_term_reproduces_the_ground_state_hessian_at_q_zero():
    """``d2ionq(0)`` against ``jax.hessian`` of the Ewald energy already here.

    The transcription's own check, and it is worth having separately from the
    matrix: ``d2ionq`` is the one piece of the frozen Hessian that is *not*
    reused from ``Gamma``, so if it is wrong the error arrives inside a total
    that has three other terms in it.

    The comparison is made at the ground-state sum's own ``alpha`` and cutoff,
    which is what makes it an equality rather than a tolerance -- the two
    truncations otherwise differ by the 1.95e-6 that
    :data:`GAMMA_AGREEMENT` is about.
    """
    import jax.numpy as jnp

    from defuqumat.scf.ewald import EwaldSum
    from defuqumat.system.cell import lattice_translations, pair_separation_bound

    calculation, _ = _ground_state("si-epsilon-unshifted-nosym")
    cell = calculation.system.cell
    structure = calculation.system.structure
    positions = jnp.asarray(structure.positions)
    alpha = calculation.ewald_sum.alpha

    # The ground state's sum, with its real-space cutoff extended from
    # 4/sqrt(alpha) to d2ionq's 5/sqrt(alpha). Nothing else about it changes.
    rmax = 5.0 / np.sqrt(alpha)
    at = np.asarray(cell.at)
    radius = rmax + pair_separation_bound(at, np.asarray(structure.positions))
    extended = EwaldSum(
        charges=calculation.ewald_sum.charges,
        alpha=alpha,
        rmax=float(rmax),
        translations=jnp.asarray(lattice_translations(at, radius)),
        gamma_factor=calculation.ewald_sum.gamma_factor,
    )
    hessian = np.asarray(jax.hessian(
        lambda pos: extended.energy(cell, pos, calculation.basis.dense)
    )(positions)).reshape(6, 6)

    ours = np.asarray(ewald_dynamical_matrix(
        calculation, jnp.zeros(3), alpha=alpha
    ))
    assert np.abs(ours - hessian).max() < 5e-9
    # The ionic matrix obeys the acoustic sum rule *identically* at q = 0,
    # because the self term is constructed as the sum of the cross terms.
    assert np.abs(ours.reshape(2, 3, 2, 3).sum(axis=2)).max() < 1e-12
    assert np.abs(np.imag(ours)).max() < 1e-12


# ---------------------------------------------------------------------------
# Identities the zone centre cannot check.
# ---------------------------------------------------------------------------

def test_the_frozen_electronic_hessian_carries_no_q():
    """The fact the whole assembly rests on, measured instead of read.

    A displacement pattern ``u_s e^{iq.R}`` moves every cell, so almost
    everything in the second derivative depends on ``q``. The exception is the
    part taken at a **frozen** electronic state: it is diagonal in the atom
    index, so the phases ``e^{iq.R}`` of the two displacements it couples are
    attached to the same atom and cancel. That is why one ground state serves
    every wavevector and only the ion-ion sum has to be redone.

    It was established by reading ``dynmat_us.f90:105-124`` -- ``init_us_2`` is
    called at ``ikk`` and ``compute_nldyn`` uses only ``ikk`` and ``wk(ikk)``,
    never the ``k + q`` index. This is the same statement as a number: the
    frozen half's dependence on ``q`` is *entirely* its Ewald term, so removing
    that term must leave something constant.

    Measured: the frozen half differs by 1.515768e-6 between ``q`` and
    ``q + G``, and :func:`ewald_dynamical_matrix` differs by 1.515768e-6 -- the
    same number, which is the Ewald reciprocal sum's own truncation and is what
    sets the floor of the periodicity check below. The electronic remainder is
    constant to 1e-12, which is where this asserts.

    Cheap by construction: the frozen half needs no linear solve, so this runs
    in a ground state and two Ewald sums.
    """
    calculation, result = _ground_state("si-epsilon-unshifted-nosym")
    cell = calculation.system.cell
    solver = make_sternheimer(calculation, result, threshold=1.0e-14)
    positions = jnp.asarray(calculation.system.structure.positions)

    def frozen_minus_ewald(q_crystal):
        q_cart = np.asarray(
            cell.k_to_cartesian(np.asarray(q_crystal, dtype=float))
        ) * cell.tpiba
        whole = np.asarray(frozen_force_constants(
            calculation, solver, positions, result.density, q_cart
        ))
        return whole - np.asarray(ewald_dynamical_matrix(calculation, q_cart))

    reference = frozen_minus_ewald([0.0, 0.0, 0.0])
    for q in ([0.25, 0.25, 0.0], [0.5, -0.5, 0.5], [1.25, 0.25, -1.0]):
        difference = np.abs(frozen_minus_ewald(q) - reference).max()
        assert difference < 1e-12, (q, difference)


def test_the_matrix_is_periodic_in_the_reciprocal_lattice():
    """``D(q + G) = D(q)``, and the two sides share almost nothing.

    A phonon wavevector is defined modulo a reciprocal lattice vector, so this
    is exact physics -- and on this side of it the two calculations are
    genuinely different objects. ``k + q`` and ``k + q + G`` select **different
    plane-wave spheres**, are diagonalised separately, give first-order
    wavefunctions with different coefficients in different orders, and the
    shifted local potential is tabulated at ``|G' + q|`` against
    ``|G' + q + G|``. What has to cancel for them to agree is exactly the
    bookkeeping P16 records at the zone edge: a coefficient is labelled by its
    Miller index, and the wrap is a *shift* of that index.

    Checked on the matrix rather than the frequencies, since a frequency is
    blind to a phase convention that a matrix element is not.

    **Measured: 8.22e-7**, against a largest element of 0.3255 Ry/bohr^2, so
    2.5e-6 relative. The tolerance is deliberately loose against that, because
    the residue is not the electrons'. The *frozen* half alone differs by
    1.515768e-6 between the two wavevectors, and
    :func:`~defuqumat.response.phononq.ewald_dynamical_matrix` alone differs by
    1.515768e-6 -- the same number to every digit printed. So what bounds this
    check is the Ewald reciprocal sum's truncation, and the electronic response
    is clean well below it; the time-reversal identity, which has no such
    floor, lands at 3.18e-9.
    """
    here = _phonons("si-epsilon-unshifted-nosym", (0.25, 0.25, 0.0))
    shifted = _phonons("si-epsilon-unshifted-nosym", (1.25, 0.25, -1.0))
    difference = np.abs(
        np.asarray(here.matrix) - np.asarray(shifted.matrix)
    ).max()
    assert difference < 1e-5, difference


def test_time_reversal_conjugates_the_matrix():
    """``D(-q) = conj(D(q))``, and ``D(q)`` is hermitian.

    The force constants of a crystal are real, so their Fourier transform obeys
    both. Nothing here imposes either: the ``-q`` run builds its own second
    sphere and its own first-order states, and the hermitisation in the
    assembly is applied *after* :attr:`~defuqumat.response.phonon.Phonons.asymmetry`
    has recorded what it removed.

    Measured: **3.18e-9** on the matrix and 8.79e-10 on the asymmetry, both of
    them the linear solves' own residue at ``threshold = 1e-14``. Unlike the
    periodicity check above this one has no Ewald truncation floor -- the two
    runs truncate the ion-ion sum identically, since ``|-q| = |q|`` -- so it is
    the sharper of the two identities, and the tolerances are set to match
    rather than left at a round number the residue clears by four orders.
    """
    # Two runs whose agreement is the check, so both converged past ``ph.x``'s
    # level: at the defaults (``tr2`` in ``ph.x``'s units and a scheduled CG,
    # since 2026-10-03) the asymmetry reads 5.5e-7, both runs' residue where
    # ``ph.x`` stops. 1e-22 is about the old ``max |dV|^2 < 1e-14`` on this
    # 20^3 grid, and 1e-14 the fixed CG threshold this entry held until then.
    plus = _phonons("si-epsilon-unshifted-nosym", (0.25, 0.25, 0.0),
                    tr2=1.0e-22, threshold=1.0e-14)
    minus = _phonons("si-epsilon-unshifted-nosym", (-0.25, -0.25, 0.0),
                     tr2=1.0e-22, threshold=1.0e-14)
    assert plus.asymmetry < 1e-8, plus.asymmetry
    difference = np.abs(
        np.asarray(minus.matrix) - np.conj(np.asarray(plus.matrix))
    ).max()
    assert difference < 1e-7, difference


# ---------------------------------------------------------------------------
# The refusals.
# ---------------------------------------------------------------------------

def test_a_reduced_k_set_is_refused_by_name():
    """The small group of ``q`` is not written, so the wedge is not usable.

    A response on a reduced k-set has to be averaged over the operations that
    leave the *perturbation* invariant, and for a phonon at ``q`` those are the
    ones with ``S q = q + G`` rather than the crystal's whole group. Averaging
    over the wrong group projects the perturbation away, which is P24's trap
    and P25's, and here the right group does not exist yet.
    """
    calculation, _ = _ground_state("si-epsilon-unshifted")
    with pytest.raises(NotImplementedError, match="small group of q"):
        require_a_two_sphere_regime(calculation, np.array([0.25, 0.25, 0.0]))


#: Two-atom silicon at ``ecutwfc = 12`` on the whole unshifted 2x2x2 grid, with
#: ``ecutrho`` left open: at 48 the smooth and dense grids coincide, at 96 they
#: do not.
_TWO_GRIDS = """&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0,
  ecutrho = {ecutrho}, nosym = .true.
/
&electrons
  conv_thr = 1.0d-12
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


def test_two_grids_keep_the_imaginary_part():
    """The ``+q`` response is complex, and the grid interpolation is written for real fields.

    :func:`~defuqumat.basis.interpolate.to_dense` and ``to_smooth`` end in
    ``jnp.real``, and on a cell whose two grids coincide they return their
    argument untouched -- so every case above, all on one grid, passed through
    a resampling that never ran. With ``ecutrho = 8 ecutwfc`` the response
    density lost its imaginary part on the way to the dense grid and the induced
    potential on the way back, and every mode at ``q = (1/2, 0, 0)`` came out
    imaginary: -1899 to -120 cm^-1, against -94 to 482 on one grid. Taken
    through the interpolation in two parts the two grids agree to **8.7e-5**
    Ry/bohr^2 on force constants of 0.27, which is the finer density grid's own
    effect on the exchange-correlation integral.
    """
    import warnings

    from defuqumat import Calculator

    def phonons(ecutrho):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            calculator = Calculator.from_text(
                _TWO_GRIDS.format(ecutrho=ecutrho), str(PSEUDO), announce=False)
            return calculator, calculator.get_phonons_at_q(q=(0.5, 0.0, 0.0))

    one_calculator, one = phonons(48.0)
    two_calculator, two = phonons(96.0)
    basis = two_calculator.calculation.basis
    assert one_calculator.calculation.basis.smooth.grid == \
        one_calculator.calculation.basis.dense.grid
    assert basis.smooth.grid != basis.dense.grid
    assert np.abs(two.matrix - one.matrix).max() < 5e-4
    assert np.abs(two.frequencies - one.frequencies).max() < 1.0


#: I82, measured on D22 2026-10-16 (one run, CPU, one core): ``D(q = b3/2)`` of
#: the magnetic trimer chain against the doubled cell's ``S_aa - S_ab`` at
#: ``Gamma``, **1.52e-5** Ry/bohr^2 on force constants up to 0.46, the nine
#: frequencies at ``q`` 0.051 cm^-1 at worst from their partners among the
#: doubled cell's eighteen. The two loops stop at different ``tr2`` (1e-16 at
#: ``q``, 1e-18 at ``Gamma``), which is where the residue is.
MAGNETIC_FOLDING_MATRIX = 1.0e-4
MAGNETIC_FOLDING_FREQUENCY = 0.2


def test_a_magnetic_insulator_at_half_b3_is_the_doubled_cell_at_gamma():
    """``nspin = 2`` at ``q != 0``: the supercell identity (I82).

    ``h3-chain-lsda-fixed.in`` is a magnetic insulator, one Bohr magneton a
    trimer by fixed occupations, and ``h3-chain-lsda-fixed-x2.in`` is the same
    chain doubled along ``a_3`` with both trimers polarized the same way and
    the 1x1x8 grid folded to 1x1x4. A pattern ``u e^{i q R}`` at ``q = b_3/2``
    is the doubled cell's antiphase pattern, so with ``S`` the doubled cell's
    ``Gamma`` matrix and ``a``, ``b`` the two trimers,

        D(b_3/2) = S_aa - S_ab,     D(0) = S_aa + S_ab.

    The two sides share no response machinery: ``D(q)`` is the two-sphere
    route, whose response density keeps one sum per spin channel, and ``S`` is
    the ``Gamma`` route's ``jvp`` of the density builder, checked against
    ``ph.x`` on triplet O2. With the channels added before the LSDA kernel the
    two-sphere route did not run at all; a spin-blind kernel would differ by
    ``f_xc``'s spin part acting on a magnetization response that is not zero
    here.
    """
    calculation, result = _ground_state("h3-chain-lsda-fixed")
    assert calculation.nspin == 2
    at_q = dynamical_matrix_at_q(
        calculation, result.wavefunctions, result.eigenvalues, result.density,
        result.becsum, q=(0.0, 0.0, 0.5))
    assert at_q.converged
    doubled_calculation, doubled = _ground_state("h3-chain-lsda-fixed-x2")
    supercell = dynamical_matrix(
        doubled_calculation, doubled.wavefunctions, doubled.eigenvalues,
        doubled.density, doubled.becsum)
    assert supercell.converged

    s = np.asarray(supercell.matrix)
    a, b = slice(0, 3), slice(3, 6)
    folded = s[a, :, a, :] - s[a, :, b, :]
    assert np.abs(np.asarray(at_q.matrix) - folded).max() < MAGNETIC_FOLDING_MATRIX
    partners = np.asarray(supercell.frequencies)
    for frequency in np.asarray(at_q.frequencies):
        assert np.abs(partners - frequency).min() < MAGNETIC_FOLDING_FREQUENCY, (
            frequency, np.sort(partners))
