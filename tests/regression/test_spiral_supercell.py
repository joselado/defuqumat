"""The force, the stress and the velocity operator of a spin spiral, against its supercell.

``pw.x`` has no spin spiral, so the check is the identity P19 rests on: at a
commensurate ``q`` the spiral *is* a supercell, the generalized Bloch theorem
being exact there. The cell is the nickel chain with an iodine beside each bond
(``tests/data/qe/nii-chain-spiral.in``) at ``q = b3/4``, against the four-cell
supercell ``nii-chain-4cell.in`` diagonalised at the spiral's density unfolded onto
it, exactly as ``test_spiral_soc.py`` builds it. Three things make it the right
cell and the hydrogen chain the wrong one:

* both datasets carry projectors, so the pair ``vkb(k + q/2)``, ``vkb(k - q/2)``
  in the nonlocal term -- the whole of what was missing -- is exercised
  (``H.pz-vbc`` has none);
* the iodine sits off the chain, so the force on each atom is not zero by
  symmetry: an identity between two zeros would test nothing;
* the supercell's two spinor components share one sphere, so it is computed by
  the ordinary noncollinear path that ``test_spinor_forces.py`` holds against
  ``pw.x``.

Both sides are frozen-state derivatives of states that describe the same physical
electrons, so they agree to the eigensolver's tolerance and not to the SCF's: the
supercell is never converged on its own, it is diagonalised once at the unfolded
density. The force is compared atom by atom (each of the four nickel and four
iodine atoms of the supercell against the spiral's one), the stress as it is,
since both are intensive.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.forces import compute_forces
from defuqumat.forces.energy import FrozenState, energy_at, state_from_result
from defuqumat.response.velocity import VelocityOperator
from defuqumat.scf.driver import Calculation
from defuqumat.stress import compute_stress
from defuqumat.workflows.nscf import fixed_density_states
from defuqumat.workflows.spiral import unfold_spiral_density
from tests.conftest import GENERATED

pytestmark = [pytest.mark.regression, pytest.mark.slow]

SPIRAL = GENERATED / "nii-chain-spiral.in"
SUPERCELL = GENERATED / "nii-chain-4cell.in"

#: The spiral against the supercell, Ry/bohr and Ry/bohr^3. Both from this
#: code at one density, so the floor is the eigensolver's (``conv_thr = 1e-10``
#: on the supercell's diagonalisation); the measured figures are in
#: ``notes/incompat/I20.md`` and ``I21.md``.
FORCE_IDENTITY = 1.0e-6
STRESS_IDENTITY = 1.0e-7


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


def _frozen(calculation, eigenvalues, states) -> FrozenState:
    eigenvalues = jnp.asarray(eigenvalues)
    weights, levels = calculation.occupations(eigenvalues)
    return FrozenState(
        wavefunctions=jnp.asarray(states), weights=jnp.asarray(weights),
        eigenvalues=eigenvalues, entropy=float(levels.get("smearing", 0.0)),
    )


@lru_cache(maxsize=1)
def _spiral(pseudo_dir: Path):
    """The spiral's SCF, and its states re-solved once at the SCF's density.

    **Both cells are compared at one input density**, not at the end of two
    different processes. The SCF's states were diagonalised in the potential
    of the last *input* density, and the density it reports is one mixing step
    from that, so the supercell diagonalised at the reported density is a
    slightly different state: measured on this cell, a Fermi level
    1.8e-6 Ry away and energy terms 1e-4 to 1e-3 Ry apart, which put 2e-5 to
    2e-4 Ry/bohr into the force identity. Re-solving the spiral at the same
    density as the supercell removes it, and leaves the eigensolver's floor.
    """
    calculator = Calculator.from_file(SPIRAL, pseudo_dir, announce=False)
    result = calculator.get_scf()
    assert result.converged
    calculation, _, eigenvalues, states = fixed_density_states(
        calculator.system, calculator.pseudos, result.density, conv_thr=1.0e-10,
    )
    return calculator, calculation, result, _frozen(calculation, eigenvalues, states)


@lru_cache(maxsize=1)
def _supercell(pseudo_dir: Path):
    """The supercell at the unfolded spiral density, as a frozen state."""
    spiral, _, result, _ = _spiral(pseudo_dir)
    supercell = Calculator.from_file(SUPERCELL, pseudo_dir, announce=False)
    system = supercell.system.with_soc_scale(0.0)
    calculation = Calculation(system, supercell.pseudos)
    unit = Calculation(spiral.system, spiral.pseudos).basis.dense.grid
    assert calculation.basis.dense.grid == (unit[0], unit[1], 4 * unit[2])
    lab = unfold_spiral_density(np.asarray(result.density), spiral.system.spiral_q,
                                (1, 1, 4), calculation.basis.dense.grid)
    calculation, _, eigenvalues, states = fixed_density_states(
        system, supercell.pseudos, lab, conv_thr=1.0e-10, calculation=calculation,
    )
    return calculation, _frozen(calculation, eigenvalues, states), lab


def test_the_functional_is_the_spiral_total_energy(pseudo_dir):
    """What is differentiated is the energy the SCF converged, pair term included."""
    calculator, _, result, _ = _spiral(pseudo_dir)
    calculation = Calculation(calculator.system, calculator.pseudos)
    frozen = float(energy_at(calculation, state_from_result(result), spinors=True))
    assert frozen == pytest.approx(result.total_energy, abs=1.0e-8)


def test_the_spiral_force_is_the_supercell_force(pseudo_dir):
    calculator, _, _, frozen = _spiral(pseudo_dir)
    # A fresh calculation, so that the ``'grad_m'`` spiral potential is first
    # built inside the force's trace (``spiral_box_frequencies`` on a cold
    # cache, which once read a traced ``cell.bg`` there and died).
    calculation = Calculation(calculator.system, calculator.pseudos)
    spiral = compute_forces(calculation, frozen)
    supercell_calculation, state, _ = _supercell(pseudo_dir)
    supercell = compute_forces(supercell_calculation, state)

    # The iodine is off the chain, so the force is not a symmetry's zero.
    assert np.abs(spiral.forces[1]).max() > 1.0e-3
    # Nickel then iodine, in both cells.
    unfolded = np.tile(spiral.forces, (4, 1))
    assert np.abs(supercell.forces - unfolded).max() < FORCE_IDENTITY


def test_the_spiral_stress_is_the_supercell_stress(pseudo_dir):
    _, calculation, _, frozen = _spiral(pseudo_dir)
    spiral = compute_stress(calculation, frozen)
    supercell_calculation, state, _ = _supercell(pseudo_dir)
    supercell = compute_stress(supercell_calculation, state)

    assert np.abs(spiral.tensor).max() > 1.0e-5
    assert np.abs(supercell.tensor - spiral.tensor).max() < STRESS_IDENTITY


#: ``Tr(f v_a f v_b)`` per supercell k-point against the sum over the four spiral
#: k-points that fold onto it, relative to its largest entry.
VELOCITY_IDENTITY = 1.0e-6


def _filled_velocity_traces(calculation, v_scf, psi, weights):
    """``T_ab(k) = sum_mn f_m f_n Re <m|v_a|n><n|v_b|m>``, ``(nk, 3, 3)``.

    A trace of ``f(H) v_a f(H) v_b`` at each k-point, so it is invariant under
    any unitary mixing of degenerate bands (rule D4) and needs no matching of
    bands between the two cells; ``f`` is the filling, ``wg / wk``, which also
    makes the bands above the spiral's window (``nbnd = 34`` against the
    supercell's 144) count for nothing.
    """
    elements = np.asarray(VelocityOperator(calculation, v_scf).matrix_elements(psi))[:, 0]
    fill = np.asarray(weights)[0] / np.asarray(calculation.system.kpoints.weights)[:, None]
    return np.real(np.einsum("akmn,bknm,km,kn->kab", elements, elements, fill, fill))


def test_the_spiral_velocity_is_the_supercell_velocity(pseudo_dir):
    """``dH/dk`` with both spheres moved together, against the folded supercell.

    A spiral state at ``k`` is the supercell state at ``k + q/2``, which for
    ``q = b3/4`` folds the spiral's eight k-points onto the supercell's two,
    four onto each. ``v`` commutes with the spiral's generalized translation,
    so it has no matrix element between two spiral k-points and the supercell
    trace at ``k'`` is the sum of the four spiral traces folded onto it.
    """
    _, calculation, result, frozen = _spiral(pseudo_dir)
    v_spiral = calculation.potential(result.density).v_scf
    spiral = _filled_velocity_traces(
        calculation, v_spiral, frozen.wavefunctions, frozen.weights)
    supercell_calculation, state, lab = _supercell(pseudo_dir)
    v_super = supercell_calculation.potential(jnp.asarray(lab)).v_scf
    supercell = _filled_velocity_traces(
        supercell_calculation, v_super, state.wavefunctions, state.weights)

    cell = calculation.system.cell
    kz = np.asarray(calculation.system.kpoints.crystal(cell))[:, 2]
    folded = np.round(np.mod(kz + 0.5 * calculation.system.spiral_q[2], 0.25) * 8) % 2
    super_cell = supercell_calculation.system.cell
    kz_super = np.asarray(supercell_calculation.system.kpoints.crystal(super_cell))[:, 2]
    landing = np.round(np.mod(kz_super / 4.0, 0.25) * 8) % 2
    assert sorted(landing) == [0.0, 1.0] and np.bincount(folded.astype(int)).tolist() == [4, 4]

    scale = np.abs(supercell).max()
    assert scale > 1.0e-3
    for index, where in enumerate(landing):
        summed = spiral[folded == where].sum(axis=0)
        assert np.abs(supercell[index] - summed).max() < VELOCITY_IDENTITY * scale



#: Site moments and charges, spiral against supercell, in hbar and electrons.
SITE_IDENTITY = 1.0e-6


def _about_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_the_spiral_site_moments_are_the_supercell_moments(pseudo_dir):
    """``<S>``, ``<L>`` and the Löwdin charge, atom by atom, laboratory frame.

    Each spinor component is projected on the orbitals of its own sphere, so
    the spiral's site matrix is the laboratory one on the atoms of the home
    cell; the supercell's atom in cell ``j`` carries the same moment turned
    about the spiral's axis by ``-q . R_j``, which is the turn
    ``unfold_spiral_density`` writes (and the ``angle2`` sequence 0, 270, 180,
    90 of ``nii-chain-4cell.in``). ``<L>`` is quenched without spin-orbit
    coupling, so its zero is part of the identity rather than the test of it.
    """
    from types import SimpleNamespace

    from defuqumat.projwfc.angular_momentum import angular_momenta

    _, calculation, _, frozen = _spiral(pseudo_dir)
    spiral = angular_momenta(calculation, SimpleNamespace(
        wavefunctions=frozen.wavefunctions, occupations=np.asarray(frozen.weights)[0]))
    supercell_calculation, state, _ = _supercell(pseudo_dir)
    frozen = SimpleNamespace(wavefunctions=state.wavefunctions,
                             occupations=np.asarray(state.weights)[0])
    supercell = angular_momenta(supercell_calculation, frozen)

    assert np.linalg.norm(spiral.spin[0]) > 1.0e-2  # nickel carries the moment
    q = calculation.system.spiral_q[2]
    for cell in range(4):
        turn = _about_z(-2.0 * np.pi * q * cell)
        for atom in range(2):
            here = supercell.atoms[2 * cell + atom]
            there = spiral.atoms[atom]
            assert np.abs(np.asarray(here.charge) - np.asarray(there.charge)).max() < SITE_IDENTITY
            assert np.abs(here.s - turn @ there.s).max() < SITE_IDENTITY
            assert np.abs(here.l - turn @ there.l).max() < SITE_IDENTITY



#: The charge image, spiral against supercell, relative to its largest value.
IMAGE_IDENTITY = 1.0e-6


def test_the_spiral_charge_image_is_the_supercell_image(pseudo_dir):
    """The constant-height charge image on one plane, at one tip energy.

    The plane is ``x = 0.6`` in the unit cell's crystal coordinates, between
    the nickel and the iodine, spanning one unit cell along the chain; in the
    supercell the same plane is its first quarter along ``a3``, sampled at the
    same points. The tip sits at the spiral's Fermi level in both, so the
    image does not depend on the supercell finding the same level.
    """
    from types import SimpleNamespace

    from defuqumat.workflows.stm import run_stm

    calculator, calculation, result, frozen = _spiral(pseudo_dir)
    energy = float(result.fermi_energy)
    shape = (12, 12)
    resolved = SimpleNamespace(
        eigenvalues_by_spin=np.asarray(frozen.eigenvalues),
        wavefunctions=frozen.wavefunctions, fermi_energy=energy, homo=None, lumo=None,
        density=result.density, fermi_energy_up=None, magnetic_field=None)
    spiral = run_stm(calculator.system, calculator.pseudos, resolved, energy=energy,
                     plane=((0.6, 0.0, 0.0), (0.6, 1.0, 0.0), (0.6, 0.0, 1.0)),
                     shape=shape, calculation=calculation)
    supercell_calculation, state, lab = _supercell(pseudo_dir)
    frozen = SimpleNamespace(
        eigenvalues_by_spin=np.asarray(state.eigenvalues), wavefunctions=state.wavefunctions,
        fermi_energy=energy, homo=None, lumo=None, density=lab,
        fermi_energy_up=None, magnetic_field=None)
    supercell = run_stm(supercell_calculation.system, supercell_calculation.pseudos,
                        frozen, energy=energy,
                        plane=((0.6, 0.0, 0.0), (0.6, 1.0, 0.0), (0.6, 0.0, 0.25)),
                        shape=shape, calculation=supercell_calculation)
    scale = np.abs(spiral.values).max()
    assert scale > 0.0
    assert np.abs(supercell.values - spiral.values).max() < IMAGE_IDENTITY * scale


# -- the meta-GGA on a spiral (the hydrogen chain, which needs no projectors) --

#: The folded spiral eigenvalues against the supercell's at the unfolded density
#: and ``tau``, Ry. The floor is the eigensolver's, as above.
META_IDENTITY = 1.0e-8


def _meta(text: str) -> str:
    """Tran-Blaha under ``'kubler'``, and the chain stretched to 5.2 bohr.

    At the inputs' 5.0 bohr the unit cell's dense grid is 15 points along the
    chain and the supercell's 64, not 60, so the two potentials would be
    evaluated on different points -- worth 4e-3 Ry in the levels here, since
    Tran-Blaha reads the Laplacian and ``tau / rho``. At 5.2 bohr the grids are
    18 and 72 (``o-chain-spiral-us.in`` makes the same choice).
    """
    text = text.replace("celldm(3) = 0.41666666666667", "celldm(3) = 0.43333333333333")
    text = text.replace("celldm(3) = 1.66666666666667", "celldm(3) = 1.73333333333333")
    return text.replace(
        "    nosym = .true.",
        "    nosym = .true.\n    input_dft = 'tb09'\n    noncollinear_gga = 'kubler'", 1)


def test_the_meta_gga_spiral_is_the_supercell(pseudo_dir):
    """``tau`` from each component's own sphere, against the 90-degree supercell.

    The hydrogen chain at ``q = b3/4`` under Tran and Blaha's potential: the
    spiral's ``(rho, tau)`` unfolded onto the four-cell supercell
    (``h-chain-90deg.in``), where the supercell is diagonalised at them. Its
    one k-point is the one all four spiral k-points fold onto (``k + q/2``),
    so its spectrum is the spiral's four spectra together. A ``tau`` built
    with one sphere for both components, or with the wrong sign of the
    ``q/2`` split, gives a different potential and moves every level.
    """
    spiral = Calculator.from_text(_meta((GENERATED / "h-chain-spiral.in").read_text()),
                                  pseudo_dir, announce=False)
    # Converged or not, the identity is taken with both cells diagonalised at
    # one ``(rho, tau)``, so it does not depend on where the SCF stopped.
    result = spiral.get_scf()
    assert result.tau is not None
    assert np.abs(np.asarray(result.tau)[1:3]).max() > 1.0e-4  # a transverse tau
    supercell = Calculator.from_text(_meta((GENERATED / "h-chain-90deg.in").read_text()),
                                     pseudo_dir, announce=False)
    calculation = Calculation(supercell.system, supercell.pseudos)
    unit = Calculation(spiral.system, spiral.pseudos).basis.dense.grid
    assert calculation.basis.dense.grid == (unit[0], unit[1], 4 * unit[2])
    grid = calculation.basis.dense.grid
    rho = unfold_spiral_density(np.asarray(result.density), spiral.system.spiral_q,
                                (1, 1, 4), grid)
    tau = unfold_spiral_density(np.asarray(result.tau), spiral.system.spiral_q,
                                (1, 1, 4), grid)
    _, _, eigenvalues, _ = fixed_density_states(
        supercell.system, supercell.pseudos, rho, tau=tau, conv_thr=1.0e-10,
        calculation=calculation,
    )
    # The spiral diagonalised once more at its own ``(rho, tau)``, so that the
    # two sides are at one input exactly rather than one SCF iteration apart.
    _, _, own, _ = fixed_density_states(
        spiral.system, spiral.pseudos, result.density, tau=result.tau,
        conv_thr=1.0e-10,
    )
    folded = np.sort(np.asarray(own).ravel())
    levels = np.sort(np.asarray(eigenvalues).ravel())
    nocc = 12
    assert np.abs(levels[:nocc] - folded[:nocc]).max() < META_IDENTITY


# -- DFT+U on a spiral (the NiI chain with a U on the nickel d shell) ---------

#: Levels and the occupation matrix, spiral against supercell, Ry and electrons.
HUBBARD_IDENTITY = 1.0e-7


def _hubbard(text: str, labels) -> str:
    return text + "HUBBARD {atomic}\n" + "".join(f"U {label}-3d 4.0\n" for label in labels)


def _turned(ns, q, cells):
    """The spiral's ``ns`` on the nickel of each supercell cell, ``(4, cells, l, l)``.

    A spiral state's up component picks up ``e^{i q.R/2}`` and its down
    component ``e^{-i q.R/2}`` from one cell to the next (beyond the common
    ``e^{i k.R}``), so with the blocks ``(uu, ud, du, dd)`` of
    ``<phi_s|psi>* <phi_t|psi>`` the ``ud`` block of cell ``j`` is
    ``e^{-i q.R_j}`` times the home cell's and ``du`` the conjugate.
    """
    ns = np.asarray(ns)[:, 0]
    out = np.empty((4, cells) + ns.shape[1:], dtype=complex)
    for j in range(cells):
        phase = np.exp(-2j * np.pi * q * j)
        out[0, j], out[3, j] = ns[0], ns[3]
        out[1, j], out[2, j] = phase * ns[1], np.conj(phase) * ns[2]
    return out


def test_the_hubbard_spiral_is_the_supercell(pseudo_dir):
    """A U on the nickel, the spiral's projectors on both spheres, against four cells.

    The supercell is diagonalised at the unfolded density and at the spiral's
    occupation matrix turned onto each of its four nickel atoms
    (:func:`_turned`), and two things are asserted: the occupation matrix the
    supercell's states measure is the one it was handed (so the turn, and with
    it the spiral's laboratory ``ns``, is self-consistent), and the supercell's
    levels at each of its two k-points are the spiral's at the four k-points
    folded onto it. A projector built on one sphere for both components fails
    both.
    """
    from defuqumat.workflows.nscf import fixed_density_states as solve

    spiral = Calculator.from_text(_hubbard(SPIRAL.read_text(), ["Ni"]), pseudo_dir,
                                  announce=False)
    result = spiral.get_scf()
    assert result.converged and result.ns is not None
    q = spiral.system.spiral_q[2]
    # The spiral diagonalised again at its own ``(rho, ns)``, and the ``ns``
    # its states measure: both sides at one input, whatever the SCF's last
    # step left between its input and output occupation matrices.
    unit, _, own, own_states = solve(spiral.system, spiral.pseudos, result.density,
                                     ns=result.ns, conv_thr=1.0e-10)
    own_weights, _ = unit.occupations(jnp.asarray(own))
    own_ns = unit.occupation_matrix(jnp.asarray(own_states), own_weights)

    supercell = Calculator.from_text(
        _hubbard(SUPERCELL.read_text(), ["Ni1", "Ni2", "Ni3", "Ni4"]), pseudo_dir,
        announce=False)
    system = supercell.system.with_soc_scale(0.0)
    calculation = Calculation(system, supercell.pseudos)
    lab = unfold_spiral_density(np.asarray(result.density), spiral.system.spiral_q,
                                (1, 1, 4), calculation.basis.dense.grid)
    ns = jnp.asarray(_turned(result.ns, q, 4))
    calculation, _, eigenvalues, states = solve(
        system, supercell.pseudos, lab, ns=ns, conv_thr=1.0e-10, calculation=calculation)
    weights, _ = calculation.occupations(jnp.asarray(eigenvalues))
    measured = np.asarray(calculation.occupation_matrix(jnp.asarray(states), weights))
    expected = _turned(own_ns, q, 4)
    assert np.abs(expected[1:3]).max() > 1.0e-3  # a transverse occupation to turn
    assert np.abs(measured - expected).max() < HUBBARD_IDENTITY

    kz = np.asarray(spiral.system.kpoints.crystal(spiral.system.cell))[:, 2]
    folded = np.round(np.mod(kz + 0.5 * q, 0.25) * 8) % 2
    kz_super = np.asarray(system.kpoints.crystal(system.cell))[:, 2]
    landing = np.round(np.mod(kz_super / 4.0, 0.25) * 8) % 2
    own = np.asarray(own).reshape(len(kz), -1)
    levels = np.asarray(eigenvalues).reshape(len(kz_super), -1)
    count = 80
    for index, where in enumerate(landing):
        mine = np.sort(own[folded == where].ravel())[:count]
        theirs = np.sort(levels[index])[:count]
        assert np.abs(theirs - mine).max() < HUBBARD_IDENTITY
