"""P48b: site-resolved ``<L>``, ``<S>`` and ``<J>`` (Elk tasks 15/16).

Three layers, and the first is where a mistake would actually be:

* the **operators**, which are ``L_x``, ``L_y``, ``L_z`` conjugated out of the
  complex spherical harmonics into the real ones ``ylmr2`` builds. They must be
  Hermitian, **purely imaginary** (a consequence of ``L`` being Hermitian and
  the real harmonics being real, so it is a test and not a convention), obey
  ``[L_x, L_y] = i L_z``, and have ``L^2 = l(l+1)``;
* the **quenching**, which is the headline: ``<L>`` vanishes identically
  without spin-orbit coupling, because nothing in a scalar-relativistic
  Hamiltonian locks the orbital moment to the lattice. Measured, not bounded --
  1.7e-16 on silicon;
* the **refusals**, which are the promise that a run which starts is a run
  whose physics is there.

The physics number -- nickel's orbital moment, and that ``<L>`` rotates with
the magnetization -- is expensive enough to be marked slow.
"""

import numpy as np
import pytest

from defuqumat.calculator import Calculator
from defuqumat.projwfc.angular_momentum import (
    PAULI,
    angular_momenta,
    orbital_matrices,
)

pytestmark = pytest.mark.unit


_SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0
  {extra}
/
&electrons
  conv_thr = 1e-10
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 4 4 4 0 0 0
"""


# --- the operators ----------------------------------------------------------


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_the_angular_momentum_operators_obey_their_own_algebra(l):
    matrices = orbital_matrices(l)
    size = 2 * l + 1
    assert matrices.shape == (3, size, size)
    for axis in range(3):
        assert np.allclose(matrices[axis], matrices[axis].conj().T, atol=1e-14)
    # Purely imaginary: <y_i|L|y_j> with real harmonics on both sides.
    assert np.abs(matrices.real).max() < 1.0e-14
    lx, ly, lz = matrices
    assert np.allclose(lx @ ly - ly @ lx, 1j * lz, atol=1e-13)
    assert np.allclose(ly @ lz - lz @ ly, 1j * lx, atol=1e-13)
    assert np.allclose(lz @ lx - lx @ lz, 1j * ly, atol=1e-13)
    casimir = sum(matrices[a] @ matrices[a] for a in range(3))
    assert np.allclose(casimir, l * (l + 1) * np.eye(size), atol=1e-13)


def test_the_pauli_matrices_are_the_ones_dmatls_uses():
    """``xs`` in ``dmatls.f90``, written out as the three sigma matrices."""
    sx, sy, sz = PAULI
    assert np.allclose(sx @ sy - sy @ sx, 2j * sz)
    for sigma in PAULI:
        assert np.allclose(sigma @ sigma, np.eye(2))


def test_an_l_beyond_the_table_is_refused():
    with pytest.raises(ValueError, match="l must be in"):
        orbital_matrices(4)


# --- the quenching ----------------------------------------------------------


@pytest.fixture(scope="module")
def silicon(pseudo_dir):
    text = _SILICON.format(extra=", nosym = .true.")
    calculator = Calculator.from_text(text, pseudo_dir, announce=False)
    return calculator, calculator.get_scf()


def test_the_orbital_moment_is_quenched_without_spin_orbit(silicon):
    """The headline: ``<L>`` is zero, and it is zero to machine precision.

    Nothing in a scalar-relativistic Hamiltonian locks the orbital moment to
    the lattice, so this is an identity rather than a tolerance -- the
    "vanishes pointwise" style of P47's silicon curvature. It is also the check
    that would catch an ``L`` matrix built in the wrong basis, because a wrong
    unitary gives a small non-zero answer and not an obviously wrong one.
    """
    calculator, scf = silicon
    momenta = angular_momenta(calculator.calculation, scf)
    assert np.abs(momenta.orbital).max() < 1.0e-12
    assert np.abs(momenta.spin).max() == 0.0  # nspin = 1 has no spin structure
    assert np.abs(momenta.total).max() < 1.0e-12


def test_the_projected_charge_is_the_valence_count(silicon):
    """Four electrons per silicon, up to what the projector set misses."""
    calculator, scf = silicon
    momenta = angular_momenta(calculator.calculation, scf)
    assert len(momenta.atoms) == 2
    for atom in momenta.atoms:
        assert 3.5 < atom.charge < 4.2
    assert momenta.table().startswith("Angular momenta on the ortho-atomic")


def test_j_is_l_plus_s(silicon):
    calculator, scf = silicon
    momenta = angular_momenta(calculator.calculation, scf)
    for atom in momenta.atoms:
        assert np.allclose(atom.j, atom.l + atom.s)


# --- the refusals -----------------------------------------------------------


def test_a_symmetry_reduced_kset_is_averaged_rather_than_refused(pseudo_dir):
    """``<L>`` is an axial vector, and a wedge sum is now completed as one.

    This test asserted the **refusal** until P82. A wedge sum really is a wedge
    sum -- that part was never wrong -- but the escape it pointed at (run the
    whole unshifted grid) was the only one available because nothing here
    averaged a per-atom axial vector over the group. P82 wrote that average,
    so the quantity is computed rather than declined.

    What is asserted here is **physics rather than the absence of an
    exception**: silicon is non-magnetic and centrosymmetric, so every site's
    ``<L>`` and ``<S>`` must vanish identically. A wedge sum that was *not*
    completed would leave a smooth, plausible, nonzero vector pointing along
    whichever direction the irreducible wedge happens to favour, which is
    exactly the failure the old refusal existed to prevent -- so this catches
    it without needing the whole grid to compare against.

    The number that pins the average against the closed grid on a *magnetic*
    cell is
    ``tests/regression/test_spinor_hubbard_symmetry.py::test_site_angular_momenta_agree_between_the_wedge_and_the_grid``.
    """
    calculator = Calculator.from_text(
        _SILICON.format(extra=""), pseudo_dir, announce=False
    )
    scf = calculator.get_scf()
    momenta = angular_momenta(calculator.calculation, scf)
    assert calculator.calculation.use_symmetry, "this cell must be the wedge"
    for name in ("orbital", "spin"):
        vectors = np.asarray(getattr(momenta, name))
        assert np.abs(vectors).max() < 1e-8, (
            f"non-magnetic silicon has a site <{name[0].upper()}> of "
            f"{np.abs(vectors).max():.3e}, which is an uncompleted wedge sum"
        )


def test_an_unknown_projector_set_is_refused(silicon):
    calculator, scf = silicon
    with pytest.raises(ValueError, match="unknown projector set"):
        angular_momenta(calculator.calculation, scf, kind="muffin-tin")


def test_the_calculator_reaches_it():
    """P38's rule: a new entry point gets a facade method in the same pass."""
    import ast
    import inspect
    import textwrap

    assert hasattr(Calculator, "get_angular_momenta")
    source = textwrap.dedent(inspect.getsource(Calculator.get_angular_momenta))
    body = ast.parse(source).body[0].body
    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    code = "\n".join(ast.unparse(node) for node in body)
    assert "angular_momenta(" in code
    assert "np." not in code and "einsum" not in code


# --- the physics, which costs a spinor SCF ----------------------------------


_NICKEL = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 6.65, nat = 1, ntyp = 1, ecutwfc = 60.0,
  occupations = 'smearing', smearing = 'mv', degauss = 0.02, nosym = .true.,
  noncolin = .true., lspinorb = .true., starting_magnetization(1) = 0.3,
  angle1(1) = {angle1}, angle2(1) = {angle2}
/
&electrons
  conv_thr = 1e-8, mixing_beta = 0.3
/
ATOMIC_SPECIES
 Ni 58.69 Ni.rel-pbe-nc-dojo.UPF
ATOMIC_POSITIONS crystal
 Ni 0.00 0.00 0.00
K_POINTS automatic
 4 4 4 0 0 0
"""


def _nickel(pseudo_dir, angle1, angle2=0.0, conv_thr=1.0e-12):
    text = _NICKEL.format(angle1=angle1, angle2=angle2)
    calculator = Calculator.from_text(text, pseudo_dir, announce=False)
    scf = calculator.get_scf(conv_thr=conv_thr)
    return angular_momenta(calculator.calculation, scf), scf


@pytest.mark.slow
def test_nickel_has_an_orbital_moment_when_spin_orbit_is_on(pseudo_dir):
    """The physics this feature exists for.

    ``<L_z>`` is **0.0365** hbar on a fully-relativistic norm-conserving nickel,
    against a measured orbital moment of about 0.05 mu_B -- the underestimate
    GGA is known for. What makes it a check rather than a number is the ratio:
    ``|L|/|S| = 0.11665`` against an experimental ``m_L/m_S`` of about 0.1, and
    the same quantity is identically zero without the coupling.
    """
    momenta, scf = _nickel(pseudo_dir, angle1=0.0)
    orbital, spin = momenta.orbital[0], momenta.spin[0]
    assert 0.02 < orbital[2] < 0.06
    assert abs(orbital[0]) < 1.0e-4 and abs(orbital[1]) < 1.0e-4
    assert 0.05 < np.linalg.norm(orbital) / np.linalg.norm(spin) < 0.20
    # <L> is parallel to <S>, which is the sign of the spin-orbit coupling in a
    # more-than-half-filled shell (Hund's third rule).
    assert orbital @ spin > 0.0


@pytest.mark.slow
def test_the_orbital_moment_rotates_with_the_magnetization(pseudo_dir):
    """``|<L>|`` is a scalar and nothing here imposes that it behaves like one.

    The moment is driven along ``z``, ``x`` and ``y`` in turn; the magnitude
    must be the same in all three -- the three cubic axes are equivalent by
    symmetry -- and ``<L>`` must follow ``<S>``. Measured at ``conv_thr =
    1e-12``: **0.03647659** in each, a spread of **7.4e-10** over the three
    axes, with ``L.S/|L||S| = 1.00000000``. All three take 20 iterations and
    their total energies agree to 3e-12 Ry.

    **The threshold is the whole test and 1e-10 is not tight enough**, which is
    a stronger statement than the one this docstring used to make. At 1e-10 the
    three runs stop at *different states*: the x one at
    -335.167135773432 Ry in 15 iterations and the other two at
    -335.167135784928 in 16, which is 1.15e-8 Ry apart, and the spread in
    ``|<L>|`` is 2.3e-7 -- three hundred times the symmetry it is meant to
    measure. The state the tight runs reach is the *x* one, so the other two
    were stopping short rather than the x one overshooting.

    That is worth more than a threshold change. ``dr2`` is 9e-11 at the stop and
    the energy is 1.15e-8 out, a hundredfold, and the reason is item 18's:
    ``rho_ddot`` weights the magnetization residual by a constant where it
    weights the charge by ``1/G^2``, so on a magnetic cell an ``accuracy`` below
    ``conv_thr`` bounds the moment much more weakly than it bounds the charge --
    and ``<L>`` is a moment. ``scf.history`` now carries the two halves
    separately, which is the instrument this would have needed.
    """
    magnitudes = []
    for angle1, angle2 in ((0.0, 0.0), (90.0, 0.0), (90.0, 90.0)):
        momenta, _ = _nickel(pseudo_dir, angle1, angle2)
        orbital, spin = momenta.orbital[0], momenta.spin[0]
        cosine = orbital @ spin / (np.linalg.norm(orbital) * np.linalg.norm(spin))
        assert abs(cosine - 1.0) < 1.0e-6, (angle1, angle2, cosine)
        magnitudes.append(np.linalg.norm(orbital))
    # 7.4e-10 measured; the bound is an order above it, because what is left at
    # 1e-12 is the residual scatter of three independent runs and not a
    # quantity anything here converges further.
    assert np.ptp(magnitudes) < 5.0e-9, magnitudes


@pytest.mark.slow
def test_a_negative_occupation_enters_with_its_sign(pseudo_dir):
    """``wg`` is a weight, not a magnitude, and Methfessel-Paxton makes it negative.

    The site matrix was once accumulated with ``sqrt(w_a w_b)``, which is
    ``|w|``: identical until a smearing pushes an occupation below zero, and
    then that band enters ``rho`` with the **wrong sign**. Silent, small, and
    invisible to the rotation check above, because it is systematic across
    orientations. Methfessel-Paxton on this nickel puts **486 of 1664**
    occupations below zero, which is what makes it the case that catches it.

    The assertion is the inequality a projection cannot violate: an orthonormal
    orbital set can only *lose* weight relative to the electron count, never
    gain it, so ``sum_a Tr rho_a <= nelec``. With ``|w|`` the projected charge
    is inflated by twice the negative weight and the inequality fails.
    """
    text = _NICKEL.format(angle1=0.0, angle2=0.0).replace(
        "smearing = 'mv'", "smearing = 'mp'"
    )
    calculator = Calculator.from_text(text, pseudo_dir, announce=False)
    scf = calculator.get_scf(conv_thr=1e-8)

    occupations = np.asarray(scf.occupations)
    assert (occupations < 0.0).sum() > 100, "this case is meant to have them"

    momenta = angular_momenta(calculator.calculation, scf)
    nelec = calculator.calculation.nelec
    charge = sum(atom.charge for atom in momenta.atoms)
    assert charge <= nelec + 1.0e-8, (charge, nelec)
    # ...and the spilling is small, so this is not passing by losing everything.
    assert charge > 0.95 * nelec


@pytest.mark.parametrize("l", [0, 1, 2, 3])
def test_the_spin_angle_operators_are_l_and_s_in_another_basis(l):
    """``L`` and ``S`` on ``|l j m_j>``, which is what an augmented
    fully-relativistic dataset is decomposed on.

    The columns are ``spinor_orbital_blocks``'s, so the checks are the ones
    that fail on a wrong conjugation, ordering or basis: ``J_z`` diagonal with
    ``m_j`` in the order the channels label, ``J^2 = j(j+1)``, ``L^2 =
    l(l+1)``, both algebras, and ``L.S = [j(j+1) - l(l+1) - 3/4]/2``, which
    fixes the relative sign of the two.
    """
    from defuqumat.projwfc.angular_momentum import spin_angle_operators
    from defuqumat.projwfc.channels import _compute_mj
    from defuqumat.pseudo.spinorbit import spinor

    js = [j for j in (l - 0.5, l + 0.5) if j > 0]
    orbital, spin = spin_angle_operators(l, js)
    total = orbital + spin
    mj, jj = [], []
    for j in js:
        for m in range(-l - 1, l + 1):
            if abs(spinor(l, j, m, 0)) > 1e-8 or abs(spinor(l, j, m, 1)) > 1e-8:
                mj.append(_compute_mj(j, l, m))
                jj.append(j)
    mj, jj = np.array(mj), np.array(jj)
    n = 2 * (2 * l + 1)
    assert total.shape == (3, n, n)
    np.testing.assert_allclose(total[2], np.diag(mj), atol=1e-13)
    np.testing.assert_allclose(sum(a @ a for a in total),
                               np.diag(jj * (jj + 1)), atol=1e-12)
    np.testing.assert_allclose(sum(a @ a for a in orbital),
                               l * (l + 1) * np.eye(n), atol=1e-12)
    for ops in (orbital, spin):
        np.testing.assert_allclose(ops[0] @ ops[1] - ops[1] @ ops[0],
                                   1j * ops[2], atol=1e-12)
    np.testing.assert_allclose(
        sum(orbital[i] @ spin[i] for i in range(3)),
        np.diag((jj * (jj + 1) - l * (l + 1) - 0.75) / 2), atol=1e-12)


_COBALT = """
&control
  calculation = 'scf'
/
&system
  ibrav = 6, celldm(1) = 4.60, celldm(3) = 1.30, nat = 1, ntyp = 1,
  noncolin = .true., lspinorb = .true., nosym = .true., ecutwfc = 20.0,
  ecutrho = 160.0, occupations = 'smearing', smearing = 'mv', degauss = 0.02,
  starting_magnetization(1) = 0.6, angle1(1) = {angle1}, angle2(1) = {angle2}
/
&electrons
  conv_thr = 1e-11
/
ATOMIC_SPECIES
 Co 58.933 Co.rel-pbe-nd-rrkjus.UPF
ATOMIC_POSITIONS crystal
 Co 0.0 0.0 0.0
K_POINTS automatic
 3 3 2 0 0 0
"""


def _cobalt(pseudo_dir, angle1, angle2, soc_scale=1.0):
    from defuqumat.io.pwin import parse_pw_input
    from defuqumat.system.builder import build_system

    text = _COBALT.format(angle1=angle1, angle2=angle2)
    system = build_system(parse_pw_input(text)).with_soc_scale(soc_scale)
    calculator = Calculator(system, pseudo_dir=pseudo_dir, announce=False)
    scf = calculator.get_scf()
    return angular_momenta(calculator.calculation, scf)


@pytest.mark.slow
def test_a_relativistic_ultrasoft_magnet_has_a_site_orbital_moment(pseudo_dir):
    """``<L>`` on a fully-relativistic ultrasoft dataset, tetragonal cobalt.

    Three runs. At ``soc_scale = 0`` the Hamiltonian has no spin-orbit term and
    ``<L>`` is quenched; on the spin-angle functions it is quenched only to
    the cross-``j`` approximation (the two ``j`` radial functions are taken as
    one), so the bound is the measured residue and not round-off. With the
    coupling on, the moment along ``x`` and along ``y`` are related by the
    ``C4`` of the cell, so ``|<L>|`` must agree and ``<L>`` must follow
    ``<S>``. The metric itself, ``qq_so``, is held to ``projwfc.x`` in
    ``tests/regression/test_spinor_pdos.py``.
    """
    quenched = _cobalt(pseudo_dir, 0.0, 0.0, soc_scale=0.0)
    along_x = _cobalt(pseudo_dir, 90.0, 0.0)
    along_y = _cobalt(pseudo_dir, 90.0, 90.0)
    print("quenched L", quenched.orbital[0], "S", quenched.spin[0])
    print("x L", along_x.orbital[0], "S", along_x.spin[0], "q", along_x.atoms[0].charge)
    print("y L", along_y.orbital[0], "S", along_y.spin[0], "q", along_y.atoms[0].charge)

    assert np.linalg.norm(quenched.spin[0]) > 0.5
    lx, ly = along_x.orbital[0], along_y.orbital[0]
    sx, sy = along_x.spin[0], along_y.spin[0]
    assert np.linalg.norm(lx) > 0.03
    assert abs(np.linalg.norm(lx) - np.linalg.norm(ly)) < 1.0e-6
    np.testing.assert_allclose([lx[0], lx[1]], [ly[1], -ly[0]], atol=1.0e-6)
    for orbital, spin in ((lx, sx), (ly, sy)):
        cosine = orbital @ spin / (np.linalg.norm(orbital) * np.linalg.norm(spin))
        assert cosine > 0.999, cosine
    assert np.linalg.norm(quenched.orbital[0]) < 0.05 * np.linalg.norm(lx)


# --- the block build ----------------------------------------------------------


@pytest.mark.parametrize("regime", ["", ", noncolin = .true."],
                         ids=["unpolarized", "spinor"])
def test_the_block_build_is_the_whole_build(pseudo_dir, monkeypatch, regime):
    """In memory mode the projectors are built a block of k-points at a time.

    ``OPEN.md`` Part XXIII item 17: each block's set on its own row-subset
    calculation, its states projected before the next is built, as the
    projected density of states does it. Each k-point's set is its own
    arithmetic, so the blocks must give the whole build's density matrix to
    round-off. The block is forced down to three k-points here, so that eight
    k-points take three blocks and the last one is padded. The density matrix
    rather than the moments, because silicon's ``<L>`` and ``<S>`` are zero by
    symmetry and both routes would pass on them; the starting states rather than
    converged ones, because the identity needs none.
    """
    import types
    import warnings

    import jax.numpy as jnp

    import defuqumat.projwfc.projections as projections
    from defuqumat.projwfc.angular_momentum import (
        _atomic_projectors, _projection_blocks, _site_density_matrix,
    )
    from defuqumat.projwfc.channels import projection_channels

    text = _SILICON.format(extra=", nosym = .true." + regime).replace(
        " 4 4 4 0 0 0", " 2 2 2 0 0 0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calculation = Calculator.from_text(text, pseudo_dir, announce=False,
                                           memory_mode="memory").calculation
    assert calculation.memory_mode == "memory"
    potential = calculation.potential(calculation.starting_density())
    states = calculation.starting_wavefunctions(
        calculation.hamiltonian(potential.v_scf), 8)
    nk = states.shape[1]
    weights = jnp.asarray(np.broadcast_to(
        np.asarray(calculation.system.kpoints.weights)[None, :, None], (1, nk, 8)))
    result = types.SimpleNamespace(wavefunctions=states, occupations=weights[0])
    channels = projection_channels(calculation.pseudos, calculation.system.structure)

    whole = _site_density_matrix(calculation, result,
                                 np.asarray(_atomic_projectors(calculation, "ortho-atomic")),
                                 channels)
    per_k = int(states.shape[-1]) * len(channels) * 16
    monkeypatch.setattr(projections, "PROJECTOR_BLOCK_BYTES", 3 * per_k)
    blocks = _projection_blocks(calculation, states, channels, False)
    assert [len(rows) for rows, _ in blocks] == [3, 3, 3] and blocks[-1][1] == 2
    walked = _site_density_matrix(calculation, result, None, channels)

    assert np.max(np.abs(whole)) > 0.1
    np.testing.assert_allclose(walked, whole, atol=1e-12)
