"""The parts of the optical conductivity that need no self-consistent field.

Three kinds of thing live here, and each is silent when wrong. The **unit
conversion**, because a conductivity assembled in Rydberg quantities comes out
in *Hartree* atomic units and the cancellation that makes that true is a
coincidence of the expression's homogeneity rather than a rule (``PLAN.md``
P50's factor of two is the warning). The **post-processing** -- the dielectric
tensor and the Kerr angle -- which are closed-form functions of ``sigma`` and
can therefore be checked against the closed form. And the **refusals**, which
are the promise that a run which starts is a run whose physics is there.
"""

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.response.conductivity import (
    OpticalConductivity,
    _curvature_sum,
    _pair_weights,
    _resolvent_sum,
    require_a_conductivity_regime,
)
from defuqumat.solvers.davidson import EMPTY_ETHR_FLOOR
from defuqumat.system import build_system
from defuqumat.units import AU_TO_S_PER_CM, FPI

pytestmark = [pytest.mark.unit]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


def _tensor(sigma, frequencies, *, broadening=0.01, nelec=8.0, volume=270.0):
    """An :class:`OpticalConductivity` built by hand, to test what it derives."""
    sigma = np.asarray(sigma, dtype=complex)
    return OpticalConductivity(
        frequencies=np.asarray(frequencies, dtype=float),
        sigma=sigma,
        interband=sigma,
        intraband=np.zeros_like(sigma),
        plasma=np.zeros((3, 3)),
        volume=volume,
        broadening=broadening,
        relaxation=broadening,
        nbnd=16,
        nelec=nelec,
    )


# -- the unit that a conductivity is in ---------------------------------------


def test_the_atomic_unit_of_conductivity_is_e2_over_hbar_bohr():
    """4.6e6 S/m, and it is what takes an anomalous Hall constant to S/cm.

    The number is fixed by the assembly rather than chosen: the Kubo sum is
    ``1/Omega`` times a squared velocity over a squared energy, which in
    Rydberg units is ``bohr^2 / bohr^3`` and therefore ``1/bohr`` -- the
    atomic unit ``e^2/(hbar a_0)``. An anomalous Hall conductivity of a
    ferromagnet is of order 1000 S/cm, which is 0.02 of it.
    """
    assert AU_TO_S_PER_CM == pytest.approx(45998.48, rel=1.0e-6)


def test_the_hall_conductivity_is_the_antisymmetric_part():
    """A symmetric ``sigma`` has no Hall conductivity however large it is.

    The symmetric part of a metal's static conductivity is its Drude weight and
    is orders of magnitude larger than the Hall part, so reading ``sigma[0, 1]``
    rather than antisymmetrising is a leak that grows with the metal.
    """
    sigma = np.zeros((1, 3, 3), dtype=complex)
    sigma[0] = [[10.0, 3.0, 0.0], [3.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    assert np.max(np.abs(_tensor(sigma, [0.0]).hall_conductivity)) == 0.0

    sigma[0, 0, 1], sigma[0, 1, 0] = 3.0 + 1.0, 3.0 - 1.0
    hall = _tensor(sigma, [0.0]).hall_conductivity
    assert hall[0, 1] == pytest.approx(AU_TO_S_PER_CM, rel=1.0e-12)
    assert hall[1, 0] == pytest.approx(-AU_TO_S_PER_CM, rel=1.0e-12)


def test_the_dielectric_tensor_is_the_hartree_frequency_not_the_rydberg_one():
    """``eps = 1 + 4 pi i sigma / w`` with ``w`` in **Hartree**, twice over.

    ``sigma`` is in Hartree atomic units, so the ``w`` it is divided by has to
    be too -- and the frequency axis this module carries is in Ry, as every
    energy in the package is. Getting it wrong is a clean factor of two in
    every dielectric function, with nothing else changing.
    """
    w = np.array([0.4])  # Ry
    sigma = np.zeros((1, 3, 3), dtype=complex)
    sigma[0] = np.eye(3) * (0.0 + 0.5j)
    eps = _tensor(sigma, w, broadening=0.0).dielectric
    assert eps[0, 0, 0] == pytest.approx(1.0 + FPI * 1j * 0.5j / 0.2)


def test_the_kerr_angle_is_moke_f90s_expression():
    """``-sigma_xy / (sigma_xx sqrt(1 + 4 pi i sigma_xx / w))``, in degrees.

    ``moke.f90`` is 87 lines and this is the only one that computes anything;
    the rest reads two files. It is zero at ``w = 0`` by construction there,
    which is a definition rather than a limit.
    """
    w = np.array([0.0, 0.4])
    sigma = np.zeros((2, 3, 3), dtype=complex)
    for i in range(2):
        sigma[i, 0, 0] = 0.3 + 0.1j
        sigma[i, 0, 1], sigma[i, 1, 0] = 0.02 - 0.01j, -0.02 + 0.01j
    kerr = _tensor(sigma, w).kerr

    assert kerr[0] == 0.0
    sxx, sxy, omega = 0.3 + 0.1j, 0.02 - 0.01j, 0.2
    expected = -sxy / (sxx * np.sqrt(1.0 + FPI * 1j * sxx / omega))
    assert kerr[1] == pytest.approx(expected * 180.0 / np.pi)


def test_the_fsum_is_the_spectral_weight_over_pi_n_over_two():
    """A flat ``Re sigma`` of known area comes back as its own ratio.

    The f-sum rule's normalisation is ``pi n_e / 2`` with ``n_e`` the electron
    density, so the *volume* and the *electron count* both enter it -- which is
    what makes it a check on the assembly's prefactor and not only on its band
    truncation.
    """
    w = np.linspace(0.0, 2.0, 2001)  # Ry
    sigma = np.zeros((w.size, 3, 3), dtype=complex)
    sigma[:, 0, 0] = 1.0
    nelec, volume = 8.0, 270.0
    # The integral runs over w in Hartree, so a flat unit Re sigma over 2 Ry
    # has area 1.0.
    expected = 1.0 / (np.pi * (nelec / volume) / 2.0)
    assert _tensor(sigma, w, nelec=nelec, volume=volume).fsum == pytest.approx(
        expected, rel=1.0e-9
    )


# -- what it refuses -----------------------------------------------------------


def _calculation(case: str, **overrides):
    """A stand-in carrying only what the guard reads."""
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    fields = {"is_ultrasoft": False, "is_paw": False, "spiral": False,
              "system": system}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_an_augmented_dataset_is_no_longer_refused_for_the_overlaps_velocity():
    """P99 wrote the term this used to refuse, so the guard has to let it past.

    What was missing was never ``dS/dk`` -- that is one tangent of the same
    ``jvp`` as ``dH/dk`` and has been here since P24. It was
    ``<psi_n|T^dag dT/dk_a|psi_m>``, the augmentation dipole, which no
    arrangement of the two tangents can supply. The guard is checked from both
    ends: an augmented calculation passes, and the two refusals that remain
    still fire.
    """
    require_a_conductivity_regime(_calculation("si2-nosym", is_ultrasoft=True))
    require_a_conductivity_regime(_calculation("si2-nosym", is_paw=True))


def test_a_spin_spiral_is_refused_for_its_two_spheres():
    with pytest.raises(NotImplementedError, match="two spinor components"):
        require_a_conductivity_regime(_calculation("si2-nosym", spiral=True))


def test_a_symmetry_reduced_k_set_is_completed_and_what_it_cannot_complete_is_refused():
    """The wedge is completed by its group now; three k-sets still refuse.

    ``si-epsilon.in`` is silicon on a reduced set and passes, carrying the 48
    operations of ``Oh`` twice over (time reversal), which is what the average
    of :func:`~defuqumat.response.conductivity._wedge` runs over. What still
    refuses, each by name: a magnetic wedge (``ni-noncol-111-nosource.in``,
    no measured number), a set whose weights the group does not unfold
    uniformly (the wedge of ``si-epsilon.in`` with its weights permuted), and a
    reduced explicit set on a ``nosym`` run, which was cut by a group the run
    does not carry.
    """
    import dataclasses

    from defuqumat.response.conductivity import _wedge
    from defuqumat.system.kpoints import KPoints

    group = _wedge(_calculation("si-epsilon"))
    assert group.order == 96 and group.has_antiunitary
    require_a_conductivity_regime(_calculation("si-epsilon"))
    require_a_conductivity_regime(_calculation("si2-nosym"))
    assert _wedge(_calculation("si2-nosym")) is None

    with pytest.raises(NotImplementedError, match="magnetic"):
        require_a_conductivity_regime(_calculation("ni-noncol-111-nosource"))

    wedge = _calculation("si-epsilon").system
    weights = np.asarray(wedge.kpoints.weights)
    scrambled = KPoints.from_crystal(
        np.asarray(wedge.kpoints.crystal(wedge.cell)), weights[::-1].copy(),
        wedge.cell, precision=wedge.kpoints.precision)
    with pytest.raises(NotImplementedError, match="tile it uniformly"):
        require_a_conductivity_regime(SimpleNamespace(
            spiral=False, system=dataclasses.replace(wedge, kpoints=scrambled)))
    with pytest.raises(NotImplementedError, match="nosym run"):
        require_a_conductivity_regime(SimpleNamespace(
            spiral=False, system=dataclasses.replace(wedge, nosym=True)))


# -- the degeneracy guard, and which route actually needs it -------------------

def test_a_degenerate_pair_cancels_on_the_frequency_route_and_diverges_on_the_other():
    """``OPEN.md`` B1's mechanism, and the correction to it.

    The sum divides by ``e_mn``, so a pair degenerate by symmetry that the
    eigensolver returns split by round-off looks like a catastrophe. On the
    **frequency** route with a smeared occupation it is not one, and not by
    luck. The two orderings carry ``t_nm = -t_mn`` and ``z_mn = conj(z_nm)``,
    so what survives is ``w_k[f(e_n) - f(e_m)]``, itself linear in the gap
    whenever ``f`` is a smooth function of energy: the singularity cancels
    analytically and the answer is flat as the splitting goes to zero.

    The **curvature** route divides by ``e_mn`` twice. The same numerator
    difference kills one power and leaves ``1/g``, so that route -- which is
    the intrinsic anomalous Hall conductivity -- is what the guard is really
    protecting. Asserting both together is the point: a test of the guard on
    the frequency route alone would be measuring something that cannot move.
    """
    rng = np.random.default_rng(0)
    v = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    element = jnp.asarray(v + np.conj(np.swapaxes(v, 1, 2)))  # Hermitian, as dH/dk is
    zomega = jnp.asarray([0.0 + 0.01j])

    frequency, curvature = [], []
    for gap in (1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4):
        energies = jnp.asarray([0.0, gap])
        # One occupation *function* of energy, which is what a smearing is.
        filling = 1.0 / (1.0 + jnp.exp(energies / 0.02))
        # The guard off, so what is measured is the expression and not the mask.
        f, _ = _resolvent_sum(element, energies, filling, filling, zomega, 0.0)
        c, _ = _curvature_sum(element, energies, filling, filling, 0.0)
        frequency.append(float(jnp.max(jnp.abs(f))))
        curvature.append(float(jnp.max(jnp.abs(c))))

    # Flat over eight decades of splitting: 1.9237e4 at every one of them.
    assert max(frequency) / min(frequency) < 1.001

    # And exactly ``1/g`` on the other route -- each step of a hundred in the
    # gap is a step of a hundred in the answer.
    for coarse, fine in zip(curvature[:-1], curvature[1:]):
        assert coarse / fine == pytest.approx(100.0, rel=1e-3)


def test_fixed_occupations_across_a_degeneracy_diverge_on_both_routes():
    """The other place nothing cancels, and it has a name already.

    A band set cut *inside* a degenerate multiplet puts one member full and
    its partner empty at the same eigenvalue, so ``f`` is not a function of
    energy at all and the cancellation above has nothing to work with:
    ``1/g`` on the frequency route and ``1/g^2`` on the curvature one. That is
    :attr:`~defuqumat.response.conductivity.OpticalConductivity.band_cut_gap`'s
    pathology, measured here from the expression rather than from a run.
    """
    rng = np.random.default_rng(0)
    v = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    element = jnp.asarray(v + np.conj(np.swapaxes(v, 1, 2)))
    zomega = jnp.asarray([0.0 + 0.01j])
    filling = jnp.asarray([1.0, 0.0])   # full and empty, at the same energy

    frequency, curvature = [], []
    for gap in (1.0e-10, 1.0e-8, 1.0e-6, 1.0e-4):
        energies = jnp.asarray([0.0, gap])
        f, _ = _resolvent_sum(element, energies, filling, filling, zomega, 0.0)
        c, _ = _curvature_sum(element, energies, filling, filling, 0.0)
        frequency.append(float(jnp.max(jnp.abs(f))))
        curvature.append(float(jnp.max(jnp.abs(c))))

    for coarse, fine in zip(frequency[:-1], frequency[1:]):
        assert coarse / fine == pytest.approx(100.0, rel=1e-3)
    for coarse, fine in zip(curvature[:-1], curvature[1:]):
        assert coarse / fine == pytest.approx(1.0e4, rel=1e-3)


def test_the_guard_covers_the_round_off_an_scf_leaves_on_an_empty_band():
    """Why the constant is 1e-5 and not ``dielectric.f90``'s 1e-8.

    :func:`optical_conductivity` takes an array of eigenvalues and cannot see
    where they came from, and its own docstring's advice is not a guarantee:
    hand it an ``SCFResult``'s and the *empty* bands carry
    ``max(5 ethr, 1e-5)`` Ry, since that is all
    :func:`~defuqumat.scf.driver.band_thresholds` asks of them. A symmetry
    degeneracy split by 3e-7 of that residue is the case, and on the curvature
    route the old 1e-8 let it through into a ``1/g^2`` weight.
    """
    rng = np.random.default_rng(1)
    v = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    element = jnp.asarray(v + np.conj(np.swapaxes(v, 1, 2)))
    energies = jnp.asarray([0.0, 3.0e-7])
    # The occupation has to be the *function*, not two equal numbers. Two
    # states given exactly the same filling cancel on both routes -- the
    # numerator difference is identically zero -- and the divergence the guard
    # is for lives in ``f(e_n) - f(e_m)``, which a smearing makes small and
    # nonzero rather than absent.
    filling = 1.0 / (1.0 + jnp.exp(energies / 0.02))

    kept, survived = _curvature_sum(element, energies, filling, filling, 1.0e-8)
    dropped, removed = _curvature_sum(element, energies, filling, filling,
                                      EMPTY_ETHR_FLOOR)
    assert int(survived) == 0            # the old constant saw nothing
    assert int(removed) == 2             # the new one sees the pair
    # Measured 1.593e8 against a curvature whose honest value here is nothing
    # at all: the pair is degenerate and the number is its round-off inverted.
    assert float(jnp.max(jnp.abs(kept))) > 1.0e8
    assert float(jnp.max(jnp.abs(dropped))) == 0.0


def test_the_guard_does_not_count_the_diagonal_it_always_masks():
    """The count has to discriminate, and ``e_nn = 0`` is the way it would not.

    Every partially filled band of every metal has ``W_n (1 - f_n) != 0`` on
    the diagonal, where the gap is zero exactly. Counting those would report a
    large number on every metal ever run and the same number whether or not
    anything was wrong, which is ``CLAUDE.md``'s "a check whose null result
    cannot be told from a pass".
    """
    energies = jnp.asarray([0.0, 0.1, 0.2, 0.3])
    wg = jnp.asarray([1.0, 0.5, 0.5, 0.0])
    filling = jnp.asarray([1.0, 0.5, 0.5, 0.0])
    _, _, dropped = _pair_weights(energies, wg, filling, EMPTY_ETHR_FLOOR)
    assert int(dropped) == 0


def test_the_two_halves_of_the_tensor_partition_the_pairs_at_the_tolerance():
    """A gap of exactly ``tol`` belongs to one half, not to neither.

    The interband sum keeps ``|gap| > tol`` and the Drude multiplet block
    keeps ``|gap| <= tol``, so every pair is in exactly one of them. With both
    written ``>`` and ``<`` a pair sitting on the boundary contributed to no
    part of ``sigma`` at all.
    """
    tol = 1.0e-6
    energies = jnp.asarray([0.0, tol])
    wg = jnp.asarray([0.5, 0.5])
    filling = jnp.asarray([0.5, 0.5])
    t, gap, dropped = _pair_weights(energies, wg, filling, tol)
    # Excluded from the interband half ...
    assert float(jnp.max(jnp.abs(t))) == 0.0
    assert int(dropped) == 2
    # ... and therefore inside the multiplet block, which is ``<=``.
    off = ~np.eye(2, dtype=bool)
    assert bool(np.all(np.abs(np.asarray(gap))[off] <= tol))


def test_the_ordinary_shifted_grid_is_seen_as_the_wedge_it_is():
    """The behaviour, where ``test_kpoints.py`` checks the flag underneath it.

    ``require_a_conductivity_regime`` refuses a symmetry-reduced k-set because
    the antisymmetric part of ``sigma_ab`` is a Berry-curvature sum, odd under
    ``k -> -k``, so a half-zone sum at weight 2 returns a spurious anomalous
    Hall conductivity and Kerr angle on a crystal that forces both to zero.
    The guard never fired on the most ordinary grid QE writes: a shifted
    Monkhorst-Pack reduced by time reversal alone keeps 32 of 64 points with
    ``ptp(weights) = 0`` **exactly**, because a shift is what makes the group
    act freely, and every one of the six copies of that guard tested for a
    spread in the weights.

    ``nosym = .true.`` is the escape the message names, and it is a real one:
    it sets ``rotations = None``, which ``KPoints.automatic`` reads as "the
    whole grid" and which skips time reversal with it, so the same cell comes
    back at 64 of 64.
    """
    from defuqumat.io.pwin import parse_pw_input
    from defuqumat.response.conductivity import require_a_conductivity_regime
    from defuqumat.scf.driver import Calculation
    from defuqumat.system.builder import build_system
    from defuqumat.pseudo import read_upf

    text = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0{extra}
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.01 0.00 0.00
 Si 0.26 0.24 0.25
K_POINTS automatic
 4 4 4 1 1 1
"""
    pseudo = (read_upf("tests/data/pseudo/Si.pz-vbc.UPF"),)

    reduced = build_system(parse_pw_input(text.format(extra="")))
    assert reduced.kpoints.nk == 32
    assert float(np.ptp(np.asarray(reduced.kpoints.weights))) == 0.0
    # Seen as the wedge it is, and completed by the group it was cut with:
    # the identity and time reversal, so the average is (sigma + sigma^T)/2.
    from defuqumat.response.conductivity import _wedge

    group = _wedge(Calculation(reduced, pseudo))
    assert group is not None and group.order == 2 and group.has_antiunitary

    whole = build_system(parse_pw_input(text.format(extra=",\n  nosym = .true.")))
    assert whole.kpoints.nk == 64
    require_a_conductivity_regime(Calculation(whole, pseudo))



# -- the Drude term's delta on a tetrahedron run (I93) -------------------------


def _tetrahedron_run(occupations: str, *, nosym: bool, grid: int = 4):
    """fcc aluminium's cell with tetrahedra, and the table a Calculation builds.

    A stand-in carrying the system and ``tetrahedron_table``, built by the
    same call :meth:`~defuqumat.scf.driver.Calculation.tetrahedron_table`
    makes, so the delta is tested without an SCF.
    """
    from defuqumat.io.pwin import parse_pw_input
    from defuqumat.scf.tetrahedra import tetrahedra_for

    text = (CASES / "al-conductivity.in").read_text()
    text = text.replace(
        "occupations = 'smearing', smearing = 'marzari-vanderbilt', degauss = 0.05",
        f"occupations = '{occupations}'")
    text = text.replace(" 8 8 8 0 0 0", f" {grid} {grid} {grid} 0 0 0")
    if not nosym:
        text = text.replace(" nosym = .true.,", "")
    system = build_system(parse_pw_input(text))
    table = tetrahedra_for(occupations, system.kpoints, system.symmetry_group(),
                           system.cell, system.grid_symmetry())
    return SimpleNamespace(system=system, tetrahedron_table=lambda: table)


def _fcc_bands(system):
    """Two first-neighbour tight-binding bands on the cell's k-set, in Ry.

    ``-sum_R cos(k.R)`` over the twelve first neighbours of the fcc lattice,
    written in crystal coordinates so it is periodic and carries the whole
    cubic group, which a wedge needs.
    """
    neighbours = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                           [1, -1, 0], [0, 1, -1], [1, 0, -1]])
    k = np.asarray(system.kpoints.crystal(system.cell))
    tb = -0.1 * np.sum(np.cos(2 * np.pi * k @ neighbours.T), axis=1)
    return np.stack([tb, 0.5 * tb + 0.6], axis=1)  # (nk, 2)


@pytest.mark.parametrize("occupations", ["tetrahedra", "tetrahedra_lin",
                                         "tetrahedra_opt"])
def test_the_tetrahedron_drude_delta_integrates_to_the_tetrahedron_dos(occupations):
    """I93: the delta of the plasma sum on a tetrahedron run, with ``v_a v_b = 1``.

    The plasma sum is ``sum_k w_k sum_n delta_kn v_a v_b``; with the velocity
    replaced by one it must be the tetrahedron DOS at ``E_F``, computed by
    :func:`~defuqumat.scf.tetrahedra.tetrahedron_dos` through the derivative
    of the integrated count rather than through the corner weights. This is
    what tells ``w_k`` once from ``w_k`` twice (the weights carry the zone
    measure already), and a spin factor of two from none. The wedge of the
    same grid has to give the same number: its representatives collect the
    corners of their whole orbit.
    """
    from defuqumat.response.conductivity import _fermi_surface_delta
    from defuqumat.scf.tetrahedra import tetrahedron_dos

    fermi = 0.05
    totals = {}
    for nosym in (True, False):
        run = _tetrahedron_run(occupations, nosym=nosym)
        bands = _fcc_bands(run.system)
        wk = np.asarray(run.system.kpoints.weights)
        delta = _fermi_surface_delta(run, fermi, bands, enabled=True)
        assert delta.shape == bands.shape
        total = float(np.sum(wk[:, None] * delta))
        dos, _ = tetrahedron_dos(run.tetrahedron_table(), bands, wk,
                                 np.array([fermi]))
        assert total == pytest.approx(float(dos[0]), rel=1e-10)
        totals[nosym] = total
    assert totals[True] > 1.0  # states per Ry at a level inside the band
    assert totals[False] == pytest.approx(totals[True], rel=1e-10)


def test_the_tetrahedron_drude_delta_keeps_its_sibling_refusals():
    """What still gives no Drude term, or an error, next to I93's lift.

    ``intraband = False`` and a fixed-occupation run return no delta, and a
    tetrahedron run on an explicit k-list still raises ``tetrahedra_for``'s
    error, since the method needs the grid the points came from.
    """
    import dataclasses

    from defuqumat.response.conductivity import _fermi_surface_delta
    from defuqumat.scf.tetrahedra import tetrahedra_for
    from defuqumat.system.kpoints import KPoints

    run = _tetrahedron_run("tetrahedra_opt", nosym=True)
    bands = _fcc_bands(run.system)
    assert _fermi_surface_delta(run, 0.05, bands, enabled=False) is None
    fixed = SimpleNamespace(system=dataclasses.replace(run.system,
                                                       occupations="fixed"))
    assert _fermi_surface_delta(fixed, 0.05, bands, enabled=True) is None

    system = run.system
    explicit = KPoints.from_crystal(
        np.asarray(system.kpoints.crystal(system.cell)),
        np.asarray(system.kpoints.weights), system.cell,
        precision=system.kpoints.precision)
    listed = dataclasses.replace(system, kpoints=explicit)
    stand_in = SimpleNamespace(system=listed, tetrahedron_table=lambda: tetrahedra_for(
        listed.occupations, listed.kpoints, listed.symmetry_group(), listed.cell,
        listed.grid_symmetry()))
    with pytest.raises(ValueError, match="automatic k-point grid"):
        _fermi_surface_delta(stand_in, 0.05, bands, enabled=True)


def test_the_tetrahedron_drude_delta_is_averaged_over_a_multiplet():
    """Rule D4 for I93: a tetrahedron delta is not equal across a degenerate pair.

    ``_plasma_sum`` contracts ``delta_n v^nm v^mn`` over each multiplet, which
    is invariant under a rotation inside the multiplet only when ``delta_n`` is
    the same for every member. A smeared delta is, a tetrahedron weight is not,
    so the delta is block-averaged; the pair here is rotated by a unitary and
    the plasma sum must not move, where with the unequal delta it does.
    """
    from defuqumat.response.conductivity import _multiplet_average, _plasma_sum

    rng = np.random.default_rng(3)
    nb = 4
    energies = np.array([[0.1, 0.5, 0.5 + 1e-9, 0.9]])
    v = rng.normal(size=(3, nb, nb)) + 1j * rng.normal(size=(3, nb, nb))
    v = v + np.conj(np.swapaxes(v, -1, -2))
    theta, phase = 0.7, np.exp(0.4j)
    u = np.eye(nb, dtype=complex)
    u[1:3, 1:3] = [[np.cos(theta), -np.sin(theta) * phase],
                   [np.sin(theta) * np.conj(phase), np.cos(theta)]]
    rotated = np.einsum("ji,ajk,kl->ail", np.conj(u), v, u)
    raw = np.array([[0.0, 2.0, 6.0, 0.0]])
    averaged = _multiplet_average(raw, energies, 1e-5)
    assert np.allclose(averaged, [[0.0, 4.0, 4.0, 0.0]])

    def plasma(delta, elements):
        return np.asarray(_plasma_sum(jnp.ones(1), jnp.asarray(delta),
                                      jnp.asarray(energies),
                                      jnp.asarray(elements[None]), 1e-5))

    assert np.allclose(plasma(averaged, v), plasma(averaged, rotated), atol=1e-12)
    assert not np.allclose(plasma(raw, v), plasma(raw, rotated), atol=1e-3)
