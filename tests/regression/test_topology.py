"""Topological quantities on real Kohn-Sham states.

The unit tests establish that the constructions are right by running them on
models whose answers are known in closed form. What they cannot check is the
part that only exists in a plane-wave code, and it is the part with the classic
silent failures:

* two neighbouring k-points store their states on **different spheres of
  G-vectors**, so the coefficients have to be aligned by Miller index;
* the neighbour of the last mesh point is the first one plus a **reciprocal
  lattice vector**, and the periodic gauge makes that a shift of the Miller
  index -- the omission that makes a Chern number non-integer;
* the overlap operator is not the identity for an ultrasoft dataset, and
  between two k-points it is not ``qq`` either but ``q_ij(b)``.

Each has a test here that fails loudly if it is wrong, on a cell small enough to
run in seconds. The physics -- a Z2 invariant of a real material -- is the
bismuthene case at the end, and it is slow.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo.upf import read_upf
from defuqumat.scf.driver import run_scf
from defuqumat.system.builder import build_system
from defuqumat.topology.augmentation import augmentation_at_q
from defuqumat.workflows.topology import DFTSource, _source, run_berry_curvature, run_z2

pytestmark = pytest.mark.regression

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@lru_cache(maxsize=4)
def converged(name: str, conv_thr: float = 1.0e-8):
    """An SCF run of a committed input, cached for the whole module."""
    system = build_system(read_pw_input(CASES / name))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, run_scf(system, pseudos, conv_thr=conv_thr).density


def source(name: str, nocc: int) -> DFTSource:
    system, pseudos, density = converged(name)
    return DFTSource(system=system, pseudos=pseudos, density=density, nocc=nocc)


# --- the overlap operator ---------------------------------------------------

@pytest.mark.parametrize(
    "case, nocc", [("si2-nc-pbe.in", 4), ("si2-us.in", 4)]
)
@pytest.mark.slow
def test_states_are_orthonormal_through_s(case, nocc):
    """``<u_m|S|u_n> = delta_mn`` at a single k-point.

    For a norm-conserving dataset this says the coefficients are normalised.
    For an ultrasoft one it says a great deal more: the augmentation term is
    what makes it true at all, so the identity here is a direct check that
    ``q_ij(0)`` is being built and contracted correctly.
    """
    states = source(case, nocc).states(np.array([[0.13, 0.21, 0.07]]))
    matrix = np.asarray(states.overlap(0, 0, None))
    assert np.allclose(matrix, np.eye(nocc), atol=1e-9)


@pytest.mark.slow
def test_augmentation_at_zero_wavevector_is_the_overlap_matrix():
    """``q_ij(b) -> qq`` as ``b -> 0``.

    The overlap between two different k-points needs the augmentation charge at
    the wavevector separating them, which is a different object from the
    ``qq_ij`` a single k-point uses -- computed by a different code path, from
    radial Bessel transforms evaluated at ``|b|`` rather than tabulated on the
    dense G-vector set. At ``b = 0`` the two must coincide exactly, and nothing
    downstream would notice if they did not.
    """
    states = source("si2-us.in", 4).states(np.zeros((1, 3)))
    got = np.asarray(augmentation_at_q(states.calculation, np.zeros(3)))
    expected = np.asarray(states.calculation.projectors.qq)
    assert np.allclose(got, expected, atol=1e-12)
    assert np.max(np.abs(expected)) > 0.1  # the case would be vacuous otherwise


# --- the periodic-gauge wrap ------------------------------------------------

@pytest.mark.parametrize("case", ["si2-nc-pbe.in", "si2-us.in"])
def test_the_wrap_reproduces_a_direct_overlap(case):
    """The check that the reciprocal-lattice shift is right.

    ``k = 0.4`` and ``k = 0.5`` are neighbours; so are ``k = 0.4`` and
    ``k = -0.5``, *through* the reciprocal lattice vector ``b_1``, because
    ``-0.5 + 1 = 0.5``. The two overlaps are taken between different
    diagonalisations on different spheres, so the matrices differ by a gauge --
    but ``|det M|`` does not, and that is what is compared.

    The third number is what the same calculation gives with the shift omitted:
    a hundred times smaller, which on a mesh becomes a Chern number that is
    smooth, plausible and not an integer.
    """
    states = source(case, 4)
    direct = states.states(np.array([[0.4, 0, 0], [0.5, 0, 0]]))
    through = states.states(np.array([[0.4, 0, 0], [-0.5, 0, 0]]))

    expected = abs(np.linalg.det(np.asarray(direct.overlap(0, 1, None))))
    wrapped = abs(np.linalg.det(np.asarray(through.overlap(0, 1, [1, 0, 0]))))
    unwrapped = abs(np.linalg.det(np.asarray(through.overlap(0, 1, None))))

    assert wrapped == pytest.approx(expected, rel=1e-6)
    assert expected > 0.9
    assert unwrapped < 0.05


# --- the Chern number of a nonmagnetic crystal ------------------------------

@pytest.mark.parametrize("case", ["si2-nc-pbe.in", "si2-us.in"])
def test_silicon_has_no_berry_curvature(case):
    """Time reversal *and* inversion force ``Omega(k) = 0`` pointwise.

    So this is not merely a Chern number that integrates to zero -- every
    plaquette phase must vanish separately, which is a far sharper statement
    and is exactly what a mishandled wrap would break, plaquette by plaquette,
    at the edge of the mesh.
    """
    system, pseudos, density = converged(case)
    result = run_berry_curvature(system, pseudos, density, shape=(4, 4), nocc=4)
    assert result.chern_number == pytest.approx(0.0, abs=1e-9)
    assert result.max_flux < 1e-5


@pytest.mark.slow
@pytest.mark.parametrize("case", ["alas-berry.in", "alas-piezo-tiny.in"])
def test_a_plane_walked_a_column_at_a_time_is_the_whole_plane(case):
    """``notes/GPU-MEMORY-NEXT.md`` item 5, on real states: the streamed Chern route.

    Zincblende has no inversion centre, so -- unlike silicon above -- its
    Berry flux is **nonzero** pointwise (odd in ``k``, which is why the Chern
    number is still zero), and the comparison is about something. Each column
    is its own diagonalisation on its own padded sphere from its own random
    start, so the two routes agree to what the eigensolver's threshold leaves
    in the states rather than to round-off: at ``conv_thr = 1e-10`` the
    largest flux difference measured 9.6e-9 on the norm-conserving cell (a
    6x6 plane) and 2.6e-8 on the ultrasoft one (4x4), against fluxes of 2.2e-2
    and 5.1e-2; on the norm-conserving 4x4 it falls from 1.5e-7 at
    ``conv_thr = 1e-8`` to 8.6e-9 at 1e-12, which is convergence and not a
    floor. The ultrasoft case takes its links through ``q_ij(b)`` between two
    state sets that were never one.
    """
    system, pseudos, density = converged(case)
    runs = {
        stream: run_berry_curvature(system, pseudos, density, shape=(4, 4),
                                    nocc=4, conv_thr=1e-10, stream=stream)
        for stream in (False, True)
    }
    whole, streamed = runs[False], runs[True]
    assert whole.max_flux > 1e-2  # a flux, not a null
    assert streamed.chern_number == pytest.approx(whole.chern_number, abs=1e-9)
    assert abs(whole.chern_number) < 1e-9
    np.testing.assert_allclose(streamed.flux, whole.flux, atol=1e-7)


def test_silicon_is_a_trivial_insulator_by_the_parity_criterion():
    """``nu0 = 0`` from the eight TRIM of diamond silicon.

    The cheap end-to-end check of the parity path on real states, and it pins
    three things at once. Inversion maps the plane-wave sphere onto itself, so
    ``P`` is exact up to the eigensolver's convergence: it comes out Hermitian
    to round-off, squares to the identity to ``1e-10``, and its eigenvalues are
    ``+-1`` and not merely near them. And the eight deltas multiply to ``+1``,
    which is the known answer -- silicon is not a topological insulator.

    Note that all four valence bands are **even** at Gamma. They are the bonding
    combinations of the two sublattices' orbitals, and inversion about the bond
    centre exchanges the sublattices, so every one of them has to be; the odd
    states are the antibonding conduction bands. A calculation that reported
    otherwise would have the inversion centre in the wrong place.

    Silicon has no spin-orbit coupling here, so its bands are spin-degenerate
    rather than Kramers doublets and the halving in
    :func:`~defuqumat.topology.parity.trim_delta` does not apply -- the delta is
    the plain product over the four orbital bands. That is why this test builds
    the deltas itself instead of going through ``run_z2``, which requires
    spinors and would be right to refuse.
    """
    from defuqumat.system.symmetry import find_symmetries
    from defuqumat.topology.mesh import trim_points
    from defuqumat.topology.parity import inversion_centre, parity_eigenvalues

    system, pseudos, density = converged("si2-nc-pbe.in")
    centre = inversion_centre(find_symmetries(system.cell, system.structure))
    points = trim_points(3)
    states = source("si2-nc-pbe.in", 4).states(points, keep_projectors=True)

    product = 1
    for index, point in enumerate(points):
        matrix = np.asarray(states.parity_matrix(index, centre))
        assert np.max(np.abs(matrix - matrix.conj().T)) < 1e-10
        assert np.max(np.abs(matrix @ matrix - np.eye(4))) < 1e-8
        values = parity_eigenvalues(matrix)
        assert set(values.astype(int)) <= {-1, 1}
        if np.allclose(point, 0.0):
            assert np.all(values > 0)
        product *= int(np.prod(values))
    assert product == 1  # nu0 = 0


def test_a_z2_run_without_spin_orbit_coupling_is_refused():
    """Zero for the wrong reason is worse than an error."""
    system, pseudos, density = converged("si2-nc-pbe.in")
    with pytest.raises(ValueError, match="spin-orbit"):
        run_z2(system, pseudos, density, nocc=4)


# --- bismuthene: a real quantum spin Hall insulator -------------------------

@pytest.mark.slow
def test_bismuthene_parity_invariant_is_well_formed_and_reproducible():
    """The Fu-Kane route on the hardest thing this subpackage handles.

    Two-component spinors, an ultrasoft *fully relativistic* dataset, PBE, and a
    cell that is two thirds vacuum. Four diagonalisations, which is the whole
    point of the parity route -- the Wilson-loop calculation on the same cell is
    a hundred times the work and, at a resolution that fits one core and five
    gigabytes, does not converge (its own ``gap_step`` diagnostic says so, and
    ``PLAN.md``'s P16 entry records the measurement).

    What is asserted is what this calculation actually establishes, and no more.
    The eigenvalue and Kramers checks are the sharp part: they say the manifold
    really is an inversion eigenspace and that the window has not cut a doublet.
    ``nu = 0`` is then the arithmetic, pinned so that a change in the sign
    convention or the inversion centre would show up.

    The system is *freestanding planar* bismuth at the test-suite cutoff, which
    is neither the substrate-supported layer the quantum-spin-Hall literature is
    about nor ``elkpy``'s buckled one; nothing here claims to reproduce either.
    """
    system, pseudos, density = converged("bismuthene-soc-small.in")
    result = run_z2(system, pseudos, density, method="parity", nocc=30, axis=2)

    for values in result.eigenvalues.values():
        assert len(values) == 30
        assert set(values.astype(int)) <= {-1, 1}
        # Both members of a Kramers doublet carry the same parity, so an odd
        # count would mean the window split one.
        assert int(np.sum(values < 0)) % 2 == 0
    assert set(result.deltas.values()) == {-1}
    assert result.nu0 == 0


# --- PAW: the one-centre coefficients have to cross with the density --------

@lru_cache(maxsize=2)
def _converged_result(name: str):
    """The whole :class:`SCFResult`, not just its density.

    ``becsum`` is the part of the mixed state a fixed-density run cannot
    rebuild, so the PAW tests below need the result object rather than
    :func:`converged`'s density alone.
    """
    system = build_system(read_pw_input(CASES / name))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, run_scf(system, pseudos, conv_thr=1.0e-12)


def test_a_paw_source_without_becsum_is_refused():
    """The refusal that replaced the blanket one, and it is the useful half.

    ``ddd_paw`` is a function of ``becsum``, which is a property of the
    *wavefunctions*: a source handed only the density would build a PAW
    Hamiltonian missing its one-centre coefficients, converge cleanly, and give
    eigenvalues wrong by tenths of an eV -- which for an invariant means an
    integer that is still an integer and no longer the crystal's.
    """
    system, pseudos, result = _converged_result("si2-paw.in")
    with pytest.raises(NotImplementedError, match="becsum"):
        DFTSource(system=system, pseudos=pseudos, density=result.density,
                  nocc=4).states(np.array([[0.0, 0.0, 0.0]]))


def test_a_paw_fixed_density_run_reproduces_the_scf_eigenvalues():
    """The sharp check, and the one the refusal above names the failure mode of.

    At the SCF's *own* k-points a fixed-density diagonalisation is the last SCF
    iteration over again, so its eigenvalues must be the converged ones. The
    error that removing ``ddd_paw`` makes is tenths of an eV -- four orders
    above this tolerance -- so the assertion is a direct measurement of whether
    the one-centre coefficients crossed with the density.
    """
    system, pseudos, result = _converged_result("si2-paw.in")
    points = np.asarray(system.kpoints.crystal(system.cell))
    states = DFTSource(system=system, pseudos=pseudos, density=result.density,
                       becsum=result.becsum, nocc=4).states(points)

    reference = np.asarray(result.eigenvalues)[:, :4]
    assert np.max(np.abs(np.asarray(states.energies)[:, :4] - reference)) < 1.0e-6


@pytest.mark.slow
def test_a_paw_chern_number_is_an_exact_integer():
    """Silicon is trivial, so the number is zero; that it is an *integer* is
    the statement about the machinery.

    The Fukui-Hatsugai-Suzuki sum is a lattice quantity and comes out exactly
    integral on any mesh where the manifold is gapped -- including one whose
    overlaps carry PAW's augmentation charge, which is what this adds to the
    ultrasoft case already covered above.
    """
    system, pseudos, result = _converged_result("si2-paw.in")
    curvature = run_berry_curvature(system, pseudos, result.density,
                                    becsum=result.becsum, nocc=4, shape=(6, 6))
    assert abs(curvature.chern_number) < 1.0e-8


# --- a collinear spin-polarized run, one channel at a time -------------------

@lru_cache(maxsize=2)
def _channels_of(name: str, nspin: int, conv_thr: float):
    """An SCF of a committed input, at ``nspin = 2`` with no moment if asked.

    ``with_spin(2, tot_magnetization=0)`` from an ``nspin = 1`` input starts
    spin-degenerate and stays there, since nothing in an SCF breaks spin
    symmetry on its own, so its two channels are the unpolarized run twice.
    """
    system = build_system(read_pw_input(CASES / name))
    if nspin != system.nspin:
        system = system.with_spin(nspin, tot_magnetization=0)
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos, run_scf(system, pseudos, conv_thr=conv_thr)


@pytest.mark.slow
def test_each_collinear_channel_reproduces_its_own_scf_eigenvalues():
    """The check that ``spin`` selects the channel it names, on a magnet.

    Triplet O2 at fixed occupations, ``tot_magnetization = 2``: seven up and
    five down electrons, each channel gapped at its own count. At the SCF's own
    k-point a fixed-density diagonalisation of channel ``s`` is the SCF's last
    one over again, so its eigenvalues are ``eigenvalues_by_spin[s]``; the two
    channels are an exchange splitting apart, so a source that read channel 0
    whatever ``spin`` said fails the down channel by that splitting. The
    occupied counts are not passed: they are what the density integrates to.
    """
    system, pseudos, result = _channels_of("o2-fixed-lsda.in", 2, 1.0e-12)
    points = np.asarray(system.kpoints.crystal(system.cell))
    reference = np.asarray(result.eigenvalues_by_spin)
    worst, counts = [], []
    for spin in (0, 1):
        source = _source(
            system, pseudos, result.density, None, None, 1.0e-10, "default",
            spin=spin,
        )
        counts.append(source.nocc)
        energies = np.asarray(source.states(points).energies)[:, :source.nocc]
        worst.append(np.max(np.abs(energies - reference[spin][:, :source.nocc])))
    assert counts == [7, 5]
    splitting = np.max(np.abs(reference[0][:, :5] - reference[1][:, :5]))
    print(f"\nO2 per-channel |e - e_SCF| max: up {worst[0]:.3e}, down "
          f"{worst[1]:.3e} Ry; exchange splitting {splitting:.3e} Ry")
    assert splitting > 1.0e-2  # the channels are distinguishable
    assert max(worst) < 1.0e-6


@pytest.mark.slow
def test_a_nonmagnetic_channel_has_the_unpolarized_curvature():
    """``nspin = 2`` with no moment, per channel, is ``nspin = 1``, pointwise.

    AlAs is zincblende, so its Berry flux is nonzero plaquette by plaquette
    (odd in ``k``, the Chern number zero), and the comparison is about a field
    rather than a null integer. Each channel of the unpolarized ``nspin = 2``
    run, at the count its own density holds, must give the ``nspin = 1`` flux.
    The two SCFs and every diagonalisation are separate runs, and they agreed
    to 9.7e-13 against a largest flux of 4.1e-2 (D22, 2026-10-16): at zero
    moment nothing in the ``nspin = 2`` run differs from the ``nspin = 1`` one.
    """
    shape = (4, 4)
    _, pseudos, one = _channels_of("alas-berry.in", 1, 1.0e-10)
    system1 = build_system(read_pw_input(CASES / "alas-berry.in"))
    reference = run_berry_curvature(system1, pseudos, one.density, shape=shape,
                                    nocc=4, conv_thr=1.0e-10)
    system2, pseudos, two = _channels_of("alas-berry.in", 2, 1.0e-10)
    assert system2.nspin == 2
    moment = float(np.max(np.abs(np.asarray(two.density[0] - two.density[1]))))
    assert moment < 1.0e-8
    channels = [
        run_berry_curvature(system2, pseudos, two.density, shape=shape,
                            conv_thr=1.0e-10, spin=spin)
        for spin in (0, 1)
    ]
    differences = [float(np.max(np.abs(np.asarray(c.flux)
                                       - np.asarray(reference.flux))))
                   for c in channels]
    print(f"\nAlAs 4x4 flux: max |flux| {reference.max_flux:.3e}; per-channel "
          f"max |flux_s - flux_nspin1| {differences[0]:.3e}, "
          f"{differences[1]:.3e}; Chern {[c.chern_number for c in channels]}")
    assert reference.max_flux > 1.0e-2  # a flux, not a null
    for curvature in channels:
        assert abs(curvature.chern_number) < 1.0e-9
    assert max(differences) < 1.0e-6
