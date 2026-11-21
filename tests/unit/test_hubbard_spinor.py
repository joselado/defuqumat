"""P62b unit checks: the occupation matrix as a 2x2 matrix in spin space.

The comparison against Quantum ESPRESSO is in
``tests/regression/test_ldau_flavours.py``. What is here needs no SCF, and the
two statements it makes are the ones that cannot be made about a collinear
occupation matrix at all:

* **a spinor with its moment along z reproduces the collinear answer exactly**,
  in the energy and in the potential, for both functionals and both double
  countings -- the check that catches a factor of two in the spin sum;
* **the energy is invariant under a global rotation of the spin frame**, which
  is what the off-diagonal spin blocks are for. Nothing in this functional knows
  about spin-orbit coupling, so turning the moment must cost nothing, and a
  wrong index in the exchange term breaks that where the first check does not
  see it.
"""

from dataclasses import replace

import numpy as np
import pytest

import jax.numpy as jnp

from defuqumat.hubbard.energy import (
    coefficients_from_setup,
    hubbard_energy,
    hubbard_potential,
)
from defuqumat.hubbard.interaction import default_racah
from defuqumat.hubbard.manifold import HubbardSetup, HubbardSpecies
from defuqumat.units import RY_TO_EV

pytestmark = pytest.mark.unit

U = 4.3 / RY_TO_EV
J = 1.0 / RY_TO_EV


def _setup(kind, noncolin, double_counting="fll", l=2):
    species = HubbardSpecies(n=3, l=l, u=U, occupation=6.0)
    if kind == 1:
        species = replace(species, j=J, racah=default_racah(l, J, (0.0,) if l == 2 else ()))
    width = 2 * l + 1
    return HubbardSetup(
        species=(species,), atoms=(0,), ldims=(width,), ldmx=width, offsets=(0,),
        atomwfc_offsets=(0,), nwfcU=(2 if noncolin else 1) * width, types=(0,),
        kind=kind, noncolin=noncolin, double_counting=double_counting,
    )


def _collinear_pair(width=5, seed=5):
    rng = np.random.default_rng(seed)
    block = rng.normal(size=(2, 1, width, width)) * 0.2
    return jnp.asarray(block + np.swapaxes(block, -1, -2) + np.eye(width) * 0.6)


def _as_spinor_along_z(collinear):
    """The same state written as a spinor: the two channels on the diagonal."""
    spinor = np.zeros((4,) + collinear.shape[1:], dtype=complex)
    spinor[0] = np.asarray(collinear[0])
    spinor[3] = np.asarray(collinear[1])
    return jnp.asarray(spinor)


def _hermitian_spinor(width=5, seed=9):
    rng = np.random.default_rng(seed)
    block = rng.normal(size=(2, 2, 1, width, width)) + 1j * rng.normal(
        size=(2, 2, 1, width, width)
    )
    block = 0.1 * (block + np.conj(np.einsum("stnab->tsnba", block)))
    block = block + np.eye(width) * 0.6 * np.eye(2)[:, :, None, None, None]
    return jnp.asarray(block.reshape((4, 1, width, width)))


CASES = [(0, "fll"), (1, "fll"), (1, "amf")]
IDS = ["simplified", "full/fll", "full/amf"]


@pytest.mark.parametrize("kind,double_counting", CASES, ids=IDS)
def test_a_spinor_along_z_is_the_collinear_answer(kind, double_counting):
    """Two channels or one 2x2 matrix: the same energy and the same potential.

    The off-diagonal spin blocks of the potential must come out **exactly**
    zero, which is the half of the statement a total energy would not catch.
    """
    collinear = _collinear_pair()
    spinor = _as_spinor_along_z(collinear)
    c_col = coefficients_from_setup(_setup(kind, False, double_counting))
    c_spin = coefficients_from_setup(_setup(kind, True, double_counting))

    assert float(hubbard_energy(spinor, c_spin)) == pytest.approx(
        float(hubbard_energy(collinear, c_col)), abs=1e-13
    )
    v_col = np.asarray(hubbard_potential(collinear, c_col))
    v_spin = np.asarray(hubbard_potential(spinor, c_spin))
    assert v_spin[0] == pytest.approx(v_col[0], abs=1e-13)
    assert v_spin[3] == pytest.approx(v_col[1], abs=1e-13)
    assert np.abs(v_spin[1]).max() == pytest.approx(0.0, abs=1e-14)
    assert np.abs(v_spin[2]).max() == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("kind,double_counting", CASES, ids=IDS)
def test_turning_the_moment_costs_nothing(kind, double_counting):
    """A global ``SU(2)`` rotation of the spin frame leaves the energy alone.

    The functional knows nothing about spin-orbit coupling, so the direction of
    the shell's moment cannot enter it. The rotation mixes the four spin blocks
    into each other, so this is the statement that the *exchange* term's spin
    indices are paired the way ``v_hubbard_full_nc`` pairs them and not the
    other way round -- which the check above, on a spin-diagonal matrix, cannot
    distinguish.
    """
    theta, phi = 0.7, 1.1
    rotation = np.array([
        [np.cos(theta / 2), -np.exp(-1j * phi) * np.sin(theta / 2)],
        [np.exp(1j * phi) * np.sin(theta / 2), np.cos(theta / 2)],
    ])
    blocks = np.asarray(_hermitian_spinor()).reshape((2, 2, 1, 5, 5))
    turned = np.einsum("sa,tb,stnmp->abnmp", np.conj(rotation), rotation, blocks)

    coefficients = coefficients_from_setup(_setup(kind, True, double_counting))
    original = float(hubbard_energy(jnp.asarray(blocks.reshape(4, 1, 5, 5)), coefficients))
    rotated = float(hubbard_energy(jnp.asarray(turned.reshape(4, 1, 5, 5)), coefficients))
    assert rotated == pytest.approx(original, abs=1e-13)


@pytest.mark.parametrize("kind,double_counting", CASES, ids=IDS)
def test_the_potential_is_hermitian(kind, double_counting):
    """``v_hub`` is an operator on the ``2(2l+1)`` space and must be Hermitian.

    ``jax.grad`` of a real function of a complex argument returns the
    *conjugate* Wirtinger derivative, which is the transpose of the potential
    rather than the potential. On a Hermitian ``ns`` the two differ by exactly
    that transpose, and nothing about a symmetric test matrix would show it.
    """
    ns = _hermitian_spinor()
    coefficients = coefficients_from_setup(_setup(kind, True, double_counting))
    v = np.asarray(hubbard_potential(ns, coefficients)).reshape((2, 2, 1, 5, 5))
    assert v == pytest.approx(np.conj(np.einsum("stnab->tsnba", v)), abs=1e-13)


def test_the_projector_columns_are_spin_slowest():
    """``offsetU + m + ldim (is - 1)``: all the up columns, then all the down.

    ``atomic_wfc_nc`` emits a channel's ``2l+1`` up columns as a block and its
    down ones after them, and ``new_ns_nc`` reads the occupation matrix at that
    stride. Interleaving them per orbital instead transposes the spin and ``m``
    indices of every block, which is a converged run with a rotated moment.
    """
    setup = _setup(0, True)
    columns = setup.column_map()
    assert columns.shape == (1, 2, 5)
    assert list(columns[0, 0]) == [0, 1, 2, 3, 4]
    assert list(columns[0, 1]) == [5, 6, 7, 8, 9]

    spin, slot, row, column = setup.block_indices()
    # Four quadrants of one block, not four separate matrices.
    assert set(np.unique(spin)) == {0, 1, 2, 3}
    assert row.max() == column.max() == 9


# --- the preconditioner -------------------------------------------------------


def _hubbard_hamiltonian(case, pseudo_dir):
    """A DFT+U Hamiltonian at the starting density: no SCF, one build."""
    from pathlib import Path

    from defuqumat.io.pwin import read_pw_input
    from defuqumat.pseudo import read_upf
    from defuqumat.scf.driver import Calculation
    from defuqumat.system import build_system

    cases = Path(__file__).resolve().parents[1] / "data" / "qe"
    system = build_system(read_pw_input(cases / case))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    calculation = Calculation(system, pseudos)
    potential = calculation.potential(calculation.starting_density())
    _, _, terms = calculation.hubbard_terms(calculation.starting_ns())
    hamiltonian = calculation.hamiltonian(potential.v_scf, None, terms)[0]
    assert hamiltonian.hubbard is not None, f"{case} built no Hubbard term"
    return hamiltonian


def _hubbard_share(hamiltonian, ik=0):
    """How much of ``diagonal`` comes from the Hubbard term, in Ry."""
    import equinox as eqx

    bare = eqx.tree_at(lambda h: h.hubbard, hamiltonian, None,
                       is_leaf=lambda x: x is None)
    return (np.asarray(hamiltonian.diagonal(ik))
            - np.asarray(bare.diagonal(ik)))


def test_the_spinor_preconditioner_is_usnldiag_nc_and_nothing_more(pseudo_dir):
    """``diagonal`` carries no ``v_ns``, and that is ``pw.x``'s own term list.

    Read as an asymmetry -- the collinear twin adds ``hubbard.diagonal(ik)``
    and this does not -- it looks like a missing term. It is not.
    ``PW/src/usnldiag.f90`` imports nothing from ``ldaU``, contains no
    reference to ``vhpsi``, ``v_hub`` or ``ns``, and builds ``h_diag`` as
    ``g2kin + v_of_0`` plus the ``deeq``/``deeq_nc`` diagonal and nothing else;
    the only routine that touches ``h_diag`` afterwards is ``oscdft_h_diag``,
    which is a different feature. ``vhpsi`` is applied in ``h_psi.f90``, to the
    operator, never to the preconditioner. So this class matches the reference
    and the collinear one is the departure.

    Adding it was measured before it was reverted, on the one committed cell
    where the term is not negligible: no configuration where it helps.
    """
    hamiltonian = _hubbard_hamiltonian("ni-ldau-noncol.in", pseudo_dir)
    assert np.abs(_hubbard_share(hamiltonian)).max() == 0.0


def test_the_collinear_preconditioner_carries_a_term_pw_x_does_not(pseudo_dir):
    """The other half of the pair, so the asymmetry is pinned from both sides.

    ``Hamiltonian.diagonal`` adds the Hubbard term where ``usnldiag.f90`` does
    not. It is left that way because removing it is worth nothing either: on
    ``ni-ldau-j0.in`` one diagonalisation from the same starting vectors takes
    **208 steps against 209** at ``ethr = 1e-6`` and **332 against 331** at
    1e-10, one step in three hundred, the term being 0.214 eV there. What this
    test stops is the pair being made consistent in either direction without a
    number, which is how the question arose.
    """
    hamiltonian = _hubbard_hamiltonian("ni-ldau-j0.in", pseudo_dir)
    share = _hubbard_share(hamiltonian)
    assert np.abs(share).max() > 0.0
    # ... and it is a real term rather than round-off, so a reader can tell
    # this assertion from one that would pass on a numerically dead branch.
    assert np.abs(share).max() * RY_TO_EV > 0.1


#: Pt fcc with spin-orbit coupling and U = 1e-6 eV on 5d, so the state is the
#: spin-orbit one on which defuqumat and pw.x agree to 1e-8 Ry.
_PT_SOC = """&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=7.42, nat=1, ntyp=1, ecutwfc=25.0, ecutrho=200.0,
 occupations='smearing', smearing='mv', degauss=0.02,
 noncolin=.true., lspinorb=.true.
/
&electrons
 conv_thr=1.0d-10
/
ATOMIC_SPECIES
 Pt 195.08 Pt.rel-pbe-n-kjpaw_psl.0.1.UPF
ATOMIC_POSITIONS {crystal}
 Pt 0.0 0.0 0.0
K_POINTS {automatic}
3 3 3 0 0 0
HUBBARD {atomic}
U Pt-5d 1.0e-6
"""


def test_the_spinor_symmetriser_keeps_a_spin_orbit_occupation_matrix(pseudo_dir):
    """The whole grid's ``ns`` of spin-orbit Pt 5d is a fixed point of the average.

    ``tests/data/hubbard/pt-soc-ns-nosym.npy`` is defuqumat's ``ns`` of the cell
    above under ``nosym`` on the whole 3x3x3 grid, which is ``pw.x``'s matrix:
    eigenvalues 0.8227 (4), 0.8748 (2), 1.0028 (4), spin-orbit elements 0.063,
    0.073 and 0.037, as ``pw.x`` prints them. A symmetric matrix must come back
    unchanged. Until P180 (2026-11-18) the spin factor was ``S x conj(S)``,
    which is the law of rho where ``ns`` is rho^T; it halved those elements
    (0.035, 0.040, 0.020) and put the DFT+U energy 4.8e-3 Ry from ``pw.x``.
    The test asserts that the old law moves the matrix, so it can tell them
    apart; the regression test on Ni with the moment along z cannot, because
    an ``ns`` diagonal in spin is kept by both.
    """
    from pathlib import Path
    from defuqumat import Calculator
    from defuqumat.hubbard.occupations import NsSymmetry, build_ns_symmetry

    calculation = Calculator.from_text(_PT_SOC, pseudo_dir=pseudo_dir).calculation
    symmetry = build_ns_symmetry(calculation.hubbard, calculation.system.cell,
                                 calculation.system.structure, calculation.symmetries)
    assert symmetry is not None and symmetry.spin is not None and symmetry.nsym == 48
    ns = np.load(Path(__file__).resolve().parents[1] / "data" / "hubbard"
                 / "pt-soc-ns-nosym.npy")
    kept = np.asarray(symmetry.apply(jnp.asarray(ns)))
    assert np.abs(kept - ns).max() < 1e-7
    # the old law, rho's: S x conj(S) instead of conj(S) x S
    old_spin = jnp.asarray(np.einsum("sabcd->sbadc", np.asarray(symmetry.spin)
                                     .reshape(-1, 2, 2, 2, 2)).reshape(-1, 4, 4))
    old = NsSymmetry(symmetry.operators, symmetry.sources, symmetry.slots,
                     symmetry.ldims, symmetry.nsym, symmetry.shape,
                     spin=old_spin, time_reversed=symmetry.time_reversed)
    moved = np.asarray(old.apply(jnp.asarray(ns)))
    assert np.abs(moved - ns).max() > 2e-2


_KAGOME = """&control
 calculation = 'scf'
/
&system
 ibrav = 4, celldm(1) = 10.0, celldm(3) = 1.6, nat = 3, ntyp = 1,
 ecutwfc = 20, noncolin = .true., lspinorb = .true.,
 occupations = 'smearing', degauss = 0.02
/
&electrons
/
ATOMIC_SPECIES
 Co 58.9 Co.upf
ATOMIC_POSITIONS crystal
 Co 0.5 0.0 0.0
 Co 0.0 0.5 0.0
 Co 0.5 0.5 0.0
K_POINTS automatic
 2 2 2 0 0 0
"""


def test_the_spinor_average_is_a_projector_on_a_three_fold_orbit():
    """The spinor ``ns`` average on a kagome orbit, where 8 of 24 operations are 3-cycles.

    The code pairs ``D_s ns D_s^T`` (and ``conj(S_s) ns S_s^T``) with the
    **inverse** atom permutation, which is ``new_ns`` and ``PAW_symmetrize``
    with ``s`` relabelled. ``new_ns_nc`` writes ``D_s ns D_s^T`` at
    ``irt(isym, na)`` itself. On an orbit whose permutations are involutions
    (hcp Pt, OPEN.md Part XXXI) the two are one sum; on a three-fold orbit
    they are not (audit 2026-11-20 #44). This checks the property that does not
    need a reference: applied twice, the code's average changes nothing, while
    the other pairing, built from the same tables, does not even give a
    projector. It says nothing about ``pw.x``, which was never run on such a cell.
    """
    from types import SimpleNamespace

    from defuqumat.hubbard.occupations import build_ns_symmetry
    from defuqumat.io.pwin import parse_pw_input
    from defuqumat.paw.symmetry import harmonic_rotations
    from defuqumat.system.builder import build_system
    from defuqumat.system.symmetry import atom_mapping, spin_rotations

    system = build_system(parse_pw_input(_KAGOME))
    group = system.symmetry_group()
    mapping = np.asarray(atom_mapping(system.cell, system.structure, group))
    cycles = [not np.array_equal(m[m], np.arange(3)) for m in mapping]
    assert group.nsym == 24 and sum(cycles) == 8

    l, width = 2, 5
    setup = SimpleNamespace(types=[0, 0, 0], atoms=[0, 1, 2], noncolin=True,
                            species=[SimpleNamespace(l=l)], nslot=3, ldmx=width)
    average = build_ns_symmetry(setup, system.cell, system.structure, group)

    rng = np.random.default_rng(0)
    ns = rng.normal(size=(2, 2, 3, width, width)) + 1j * rng.normal(
        size=(2, 2, 3, width, width))
    ns = ns + np.conj(np.einsum("stnab->tsnba", ns))
    once = average.apply(jnp.asarray(ns.reshape(4, 3, width, width)))
    twice = average.apply(once)
    assert float(jnp.abs(once).max()) > 0.1
    assert float(jnp.abs(twice - once).max()) < 1e-13

    d = np.asarray(harmonic_rotations(system.cell, group, l)[l])
    u = np.asarray(spin_rotations(system.cell, group))

    def forward(x):
        x = np.asarray(x).reshape(2, 2, 3, width, width)
        out = np.zeros_like(x)
        for s in range(group.nsym):
            for a in range(3):
                out[:, :, a] += np.einsum("ik,ac,cdkl,bd,jl->abij", d[s], u[s].conj(),
                                          x[:, :, mapping[s, a]], u[s], d[s])
        return (out / group.nsym).reshape(4, 3, width, width)

    other = forward(ns)
    assert np.abs(forward(other) - other).max() > 0.1


_WURTZITE = """&control
 calculation = 'scf'
/
&system
 ibrav = 4, celldm(1) = 6.14, celldm(3) = 1.602, nat = 4, ntyp = 2,
 ecutwfc = 20, noncolin = .true., lspinorb = .true.,
 occupations = 'smearing', degauss = 0.02
/
&electrons
/
ATOMIC_SPECIES
 Zn 65.38 Zn.upf
 O  16.00 O.upf
ATOMIC_POSITIONS crystal
 Zn 0.3333333333 0.6666666667 0.0
 Zn 0.6666666667 0.3333333333 0.5
 O  0.3333333333 0.6666666667 0.382
 O  0.6666666667 0.3333333333 0.882
K_POINTS automatic
 2 2 2 0 0 0
"""


def _spinor_ns_of(states, width):
    """``occupation_matrix`` on fully occupied states given on the flat ``(spin, m)`` basis."""
    from defuqumat.hubbard.occupations import occupation_matrix

    nbnd = states.shape[0]
    ns = occupation_matrix(
        jnp.eye(2 * width, dtype=states.dtype)[None], jnp.asarray(states)[None, None],
        jnp.ones((1, 1, nbnd)), jnp.arange(2 * width).reshape(1, 2, width),
        jnp.ones((1, width), dtype=bool))
    return np.asarray(ns)


def test_the_kramers_image_is_time_reversal_of_the_states():
    """``NsSymmetry.kramers_image`` against ``ns`` built from a state and its Kramers partner.

    ``Theta psi = i sigma_y conj(psi)`` on real-harmonic orbitals. The pair
    ``{psi, Theta psi}`` is closed under time reversal, so its ``ns`` from
    :func:`occupation_matrix` must be its own image, and the image of
    ``psi`` alone must be the ``ns`` of ``Theta psi``, with its spin moment
    reversed.
    """
    from defuqumat.hubbard.occupations import NsSymmetry, spinor_ns_components

    width = 5
    rng = np.random.default_rng(1)
    psi = rng.normal(size=(2, width)) + 1j * rng.normal(size=(2, width))
    partner = np.stack([np.conj(psi[1]), -np.conj(psi[0])])        # i sigma_y psi*
    pair = _spinor_ns_of(np.stack([psi.ravel(), partner.ravel()]), width)
    assert np.abs(np.asarray(NsSymmetry.kramers_image(jnp.asarray(pair))) - pair).max() < 1e-14

    single = _spinor_ns_of(psi.ravel()[None], width)
    image = NsSymmetry.kramers_image(jnp.asarray(single))
    assert np.abs(np.asarray(image) - _spinor_ns_of(partner.ravel()[None], width)).max() < 1e-14
    charge, moment = spinor_ns_components(jnp.asarray(single))
    charge_t, moment_t = spinor_ns_components(image)
    assert np.abs(np.asarray(moment)).max() > 0.1
    # the orbital charge matrix comes back transposed, the trace of each moment reversed
    assert np.abs(np.asarray(charge_t - jnp.swapaxes(charge, -1, -2))).max() < 1e-14
    assert np.abs(np.trace(np.asarray(moment_t + moment), axis1=-2, axis2=-1)).max() < 1e-14


def test_a_nonmagnetic_spinor_ns_loses_its_time_reversal_odd_part_without_inversion():
    """Wurtzite Zn 3d: the odd part the spatial group allows (audit 2026-11-20 #14).

    The k-set of a nonmagnetic spinor run merges ``k`` and ``-k``. The group of
    wurtzite has no inversion, so no spatial operation completes that pair,
    and the spatial average keeps every time-reversal-odd part of ``ns`` the
    group allows. Built by hand here: the spatial average of a random matrix,
    minus its Kramers image. The old average keeps it; the new one removes it,
    and keeps the even part unchanged.
    """
    from types import SimpleNamespace

    from defuqumat.hubbard.occupations import NsSymmetry, build_ns_symmetry
    from defuqumat.io.pwin import parse_pw_input
    from defuqumat.system.builder import build_system

    system = build_system(parse_pw_input(_WURTZITE))
    group = system.symmetry_group()
    assert group.nsym == 12 and not np.any(np.asarray(group.t_rev_array()))
    width = 5
    setup = SimpleNamespace(types=[0, 0], atoms=[0, 1], noncolin=True,
                            species=[SimpleNamespace(l=2)], nslot=2, ldmx=width)
    spatial = build_ns_symmetry(setup, system.cell, system.structure, group)
    full = build_ns_symmetry(setup, system.cell, system.structure, group, kramers=True)
    assert not spatial.kramers and full.kramers

    rng = np.random.default_rng(2)
    x = rng.normal(size=(2, 2, 2, width, width)) + 1j * rng.normal(size=(2, 2, 2, width, width))
    x = x + np.conj(np.einsum("stnab->tsnba", x))
    allowed = spatial.apply(jnp.asarray(x.reshape(4, 2, width, width)))
    odd = 0.5 * (allowed - NsSymmetry.kramers_image(allowed))
    even = allowed - odd
    assert float(jnp.abs(odd).max()) > 0.05
    assert float(jnp.abs(spatial.apply(odd) - odd).max()) < 1e-13
    assert float(jnp.abs(full.apply(odd)).max()) < 1e-13
    assert float(jnp.abs(full.apply(even) - even).max()) < 1e-13

    # With no spatial group (a trivial one, or symmetry off) the partner is
    # the whole average; a collinear setup never gets it.
    alone = build_ns_symmetry(setup, system.cell, system.structure, None, kramers=True)
    assert alone.kramers and alone.nsym == 1
    assert float(jnp.abs(alone.apply(odd)).max()) < 1e-13
    collinear = SimpleNamespace(**{**vars(setup), "noncolin": False})
    assert build_ns_symmetry(collinear, system.cell, system.structure, None,
                             kramers=True) is None


def test_the_kramers_average_is_on_for_a_nonmagnetic_spinor_run_only(pseudo_dir):
    """The driver's choice, and fcc Pt, which has inversion, unchanged by it.

    ``pt-soc-ns-nosym.npy`` is the whole grid's ``ns`` of nonmagnetic spin-orbit
    Pt, so it is already its own Kramers image to the SCF's accuracy, and the
    average with and without the partner agree to that accuracy. A moment
    turns the partner off.
    """
    from pathlib import Path
    from defuqumat import Calculator

    calculation = Calculator.from_text(_PT_SOC, pseudo_dir=pseudo_dir).calculation
    symmetry = calculation.hubbard_symmetry
    assert symmetry is not None and symmetry.kramers and symmetry.nsym == 48
    ns = jnp.asarray(np.load(Path(__file__).resolve().parents[1] / "data" / "hubbard"
                             / "pt-soc-ns-nosym.npy"))
    spatial = symmetry._spatial_average(ns)
    assert float(jnp.abs(symmetry.apply(ns) - spatial).max()) < 1e-9

    magnetic = _PT_SOC.replace("lspinorb=.true.",
                               "lspinorb=.true., starting_magnetization(1)=0.3")
    calculation = Calculator.from_text(magnetic, pseudo_dir=pseudo_dir).calculation
    assert not calculation.hubbard_symmetry.kramers


def _as_crystal_list(text, kpoints, cell, decimals=10):
    """``text`` with its ``K_POINTS`` card replaced by ``kpoints`` written out in crystal units."""
    crystal = np.asarray(kpoints.crystal(cell))
    weights = np.asarray(kpoints.weights)
    rows = "\n".join(" ".join(f"{x:.{decimals}f}" for x in k) + f" {w:.{decimals}f}"
                     for k, w in zip(crystal, weights))
    head = text.split("K_POINTS")[0]
    tail = text.split("HUBBARD")[1]
    return f"{head}K_POINTS crystal\n{len(crystal)}\n{rows}\nHUBBARD{tail}"


def test_the_kramers_average_is_on_for_a_wedge_written_as_an_explicit_list(pseudo_dir):
    """The automatic wedge of spin-orbit Pt, written out as a K_POINTS list (audit 2026-11-20b #10).

    The flag of #14 read ``KPoints.reduced``, which only an automatic grid
    sets, so the same wedge written as a list summed ``ns`` without its
    Kramers partner. It is now on for the list as for the automatic set, and
    off for the whole 3x3x3 grid written as a list, where time reversal
    reduced nothing.
    """
    from defuqumat import Calculator
    from defuqumat.io.pwin import parse_pw_input
    from defuqumat.system.builder import build_system
    from defuqumat.system.kpoints import KPoints

    automatic = build_system(parse_pw_input(_PT_SOC))
    assert automatic.kpoints.reduced and automatic.kpoints.nk == 4
    wedge = _as_crystal_list(_PT_SOC, automatic.kpoints, automatic.cell)
    listed = Calculator.from_text(wedge, pseudo_dir=pseudo_dir).calculation
    assert listed.system.kpoints.grid is None and listed.system.kpoints.nk == 4
    symmetry = listed.hubbard_symmetry
    assert symmetry is not None and symmetry.kramers and symmetry.nsym == 48

    whole = KPoints.automatic((3, 3, 3), (0, 0, 0), automatic.cell)
    grid = _as_crystal_list(_PT_SOC, whole, automatic.cell, decimals=7)
    listed = Calculator.from_text(grid, pseudo_dir=pseudo_dir).calculation
    assert listed.system.kpoints.nk == 27
    assert not listed.hubbard_symmetry.kramers


# -- ns_adj on a spinor (audit 2026-11-20c #7) ---------------------------------

def _spinor_ns_adj_setup(starting_ns):
    """A d shell and a p shell on two species, so the p slot is padded to ldmx = 5."""
    d = HubbardSpecies(n=3, l=2, u=U, occupation=8.0)
    p = HubbardSpecies(n=2, l=1, u=U, occupation=3.0)
    return HubbardSetup(
        species=(d, p), atoms=(0, 1), ldims=(5, 3), ldmx=5, offsets=(0, 10),
        atomwfc_offsets=(0, 10), nwfcU=16, types=(0, 1), noncolin=True,
        starting_ns=dict(starting_ns),
    )


def _qe_ns_adj_noncolin(ns, slot, ldim, requested):
    """``ns_adj.f90``'s ``IF (noncolin)`` branch, line by line, on one slot.

    ``requested`` maps ``(is, i)``, zero-based, to ``starting_ns(i, is, nt)``.
    """
    f = np.zeros((2 * ldim, 2 * ldim), dtype=complex)
    for m1 in range(ldim):
        for m2 in range(ldim):
            f[m1, m2] = ns[0, slot, m1, m2]
            f[m1, ldim + m2] = ns[1, slot, m1, m2]
            f[ldim + m1, m2] = ns[2, slot, m1, m2]
            f[ldim + m1, ldim + m2] = ns[3, slot, m1, m2]
    lam, vet = np.linalg.eigh(f)
    j = 0
    for spin in range(2):
        for i in range(ldim):
            if requested.get((spin, i), -1.0) >= 0.0:
                lam[j] = requested[(spin, i)]
            j += 1
    for m1 in range(2 * ldim):
        for m2 in range(m1, 2 * ldim):
            temp = sum(vet[m1, i] * lam[i] * np.conj(vet[m2, i]) for i in range(2 * ldim))
            f[m1, m2] = temp
            f[m2, m1] = np.conj(temp)
    out = np.array(ns, dtype=complex)
    out[0, slot, :ldim, :ldim] = f[:ldim, :ldim]
    out[1, slot, :ldim, :ldim] = f[:ldim, ldim:]
    out[2, slot, :ldim, :ldim] = f[ldim:, :ldim]
    out[3, slot, :ldim, :ldim] = f[ldim:, ldim:]
    return out, lam


def _canted_spinor_ns(seed=11):
    """A Hermitian occupation matrix per slot with complex off-diagonal spin blocks."""
    rng = np.random.default_rng(seed)
    ns = np.zeros((4, 2, 5, 5), dtype=complex)
    for slot, ldim in enumerate((5, 3)):
        a = rng.normal(size=(2 * ldim, 2 * ldim)) + 1j * rng.normal(size=(2 * ldim, 2 * ldim))
        vectors = np.linalg.qr(a)[0]
        full = (vectors * rng.uniform(0.05, 0.95, size=2 * ldim)) @ vectors.conj().T
        for s1 in range(2):
            for s2 in range(2):
                ns[2 * s1 + s2, slot, :ldim, :ldim] = full[
                    s1 * ldim:(s1 + 1) * ldim, s2 * ldim:(s2 + 1) * ldim]
    return ns


def test_ns_adj_on_a_spinor_is_qe_noncollinear_branch():
    """``starting_ns_eigenvalue`` on a spinor sets an eigenvalue of the ``2 ldim`` matrix.

    ``pw.x`` builds ``f`` from the four packed blocks, diagonalises it whole and
    sets eigenvalue ``(is - 1) ldim + i`` of the ascending spectrum. The old code
    looped over the four packed blocks as if they were channels: a request with
    ``is = 2`` landed on the off-diagonal ``ud`` block and the result was not
    Hermitian. Checked three ways: against a line-by-line transcription of
    ``ns_adj.f90``, by the Hermiticity of the rebuilt matrix, and by its
    spectrum, which must be the old one with the named entries replaced.
    """
    from defuqumat.hubbard.occupations import adjust_ns

    # Species 0 (the d shell): eigenvalue i = 3 of is = 2, i = 1 of is = 1.
    # Species 1 (the p shell): eigenvalue i = 2 of is = 2. A negative value
    # means "leave it", as in pw.x, and an m beyond ldim is ignored.
    request = {(0, 1, 2): 0.3, (0, 0, 0): 0.97, (0, 0, 4): -1.0,
               (1, 1, 1): 0.6, (1, 0, 4): 0.5}
    setup = _spinor_ns_adj_setup(request)
    ns = _canted_spinor_ns()
    adjusted = np.asarray(adjust_ns(jnp.asarray(ns), setup))

    expected = np.array(ns)
    for slot, (ldim, kind) in enumerate(zip((5, 3), (0, 1))):
        wanted = {(spin, m): value for (t, spin, m), value in request.items()
                  if t == kind and m < ldim}
        expected, lam = _qe_ns_adj_noncolin(expected, slot, ldim, wanted)
        full = np.block([[adjusted[2 * s1 + s2, slot, :ldim, :ldim] for s2 in range(2)]
                         for s1 in range(2)])
        np.testing.assert_allclose(full, full.conj().T, atol=1e-14)
        np.testing.assert_allclose(np.linalg.eigvalsh(full), np.sort(lam), atol=1e-13)
    np.testing.assert_allclose(adjusted, expected, atol=1e-13)
    # The positions asked for, in the ascending spectrum of the whole matrix.
    d = np.block([[adjusted[2 * s1 + s2, 0] for s2 in range(2)] for s1 in range(2)])
    original = np.linalg.eigvalsh(np.block([[ns[2 * s1 + s2, 0] for s2 in range(2)]
                                            for s1 in range(2)]))
    replaced = original.copy()
    replaced[5 + 2], replaced[0] = 0.3, 0.97
    np.testing.assert_allclose(np.linalg.eigvalsh(d), np.sort(replaced), atol=1e-13)
    # The padding of the p slot is untouched.
    assert np.abs(adjusted[:, 1, 3:, :]).max() == 0.0
    assert np.abs(adjusted[:, 1, :, 3:]).max() == 0.0
