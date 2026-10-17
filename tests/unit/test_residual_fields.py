"""The residual solver under an applied field or a constrained moment (I73).

``ScfResidual.step`` builds its potential with ``Calculation.potential(rho)``,
which applies ``calculation.magnetic_field`` at full strength, so for a field
the loop never changes the map ``x -> F(x)`` is already a function of the
state alone. The mixing loop changes the field in exactly two places
(``run_scf``: ``field_scale *= field.reducebf`` and ``field.feedback``), and
``feedback`` is the identity outside :data:`~defuqumat.scf.fields.FEEDBACK`.
So a uniform ``B_field`` in a collinear run and the measured *penalty*
constraints in a noncollinear run (neither beside the other) run. A feedback
(fixed-spin-moment) constraint, whose field is a second unknown, runs with that
field in the residual's state and Elk's update as its row of ``F`` (the
I73 feedback round), in the regimes measured: ``'fsm'`` collinear, the two
per-atom schemes noncollinear. What stays refused as physics is
``reducebf < 1``, whose decay has the field-free problem as its only fixed
point while the loop stops at whatever field is left. A
``LOCAL_MAGNETIC_FIELDS`` card and the ``total direction``
and ``atomic texture`` penalties, a noncollinear uniform field, a collinear
penalty and ``jvp='autodiff'`` under any field are refused only because nothing
measured them; a field together with a penalty is refused at input.

These tests build a ``Calculation`` and nothing more; the fixed point itself,
Newton against the mixing loop, is ``tests/regression/test_scf_solvers.py``'s
``test_newton_krylov_finds_the_mixer_s_fixed_point_under_a_field`` and
``test_newton_krylov_finds_the_mixer_s_fixed_point_under_a_feedback_field``
(both slow).
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.scf.driver import Calculation
from defuqumat.scf.residual import make_residual
from defuqumat.system.builder import build_system

pytestmark = pytest.mark.unit


_SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.2, nat = 2, ntyp = 1, ecutwfc = 12.0
  nspin = 2, starting_magnetization(1) = 0.2
  occupations = 'smearing', smearing = 'gaussian', degauss = 0.02
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


#: Replaces ``nspin = 2`` in :data:`_SILICON` by a noncollinear run.
NONCOLLINEAR = "noncolin = .true., angle1(1) = 90.0, angle2(1) = 0.0"


#: Per-atom targets for the per-atom feedback schemes, in mu_B.
MOMENTS = "STARTING_MOMENTS\n 0.0 0.0 0.1\n 0.0 0.1 0.0\n"


def _calculation(pseudo_dir, extra, card=""):
    from defuqumat.pseudo import read_upf

    text = _SILICON.format(extra=extra) + card
    if "noncolin" in extra:
        text = text.replace("nspin = 2, ", "")
    system = build_system(parse_pw_input(text))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return Calculation(system, pseudos)


FSM = "constrained_magnetization = 'fsm', lambda = 0.02, fixed_magnetization(3) = 0.1"


@pytest.mark.parametrize(
    "extra, card, leaf, shape",
    [
        (FSM, "", "uniform", (1,)),
        (NONCOLLINEAR + ", constrained_magnetization = 'atomic fsm', lambda = 0.02",
         MOMENTS, "atomic", (2, 3)),
        (NONCOLLINEAR
         + ", constrained_magnetization = 'atomic fsm direction', lambda = 0.02",
         MOMENTS, "atomic", (2, 3)),
    ],
    ids=["fsm-collinear", "atomic-fsm", "atomic-fsm-direction"],
)
def test_a_feedback_field_joins_the_state(pseudo_dir, extra, card, leaf, shape):
    """The driven field is the last block of ``x``, as ``ns`` is for DFT+U."""
    calculation = _calculation(pseudo_dir, extra, card)
    field = calculation.magnetic_field
    assert field.driven_leaf == leaf
    residual = make_residual(calculation, 16, 1e-10)
    assert residual.field_shape == shape
    bare = int(np.prod(residual.shapes[0])) + sum(
        int(np.prod(s)) for s in residual.shapes[1:] if s is not None)
    assert residual.size == bare + int(np.prod(shape))

    rho = calculation.starting_density()
    becsum_ = calculation.starting_becsum()
    driven = np.arange(1, int(np.prod(shape)) + 1, dtype=float).reshape(shape) * 1e-3
    x = residual.pack(rho, becsum_, driven=driven)
    assert x.size == residual.size
    np.testing.assert_array_equal(np.asarray(residual.unpack_field(x)), driven)
    np.testing.assert_array_equal(np.asarray(residual.unpack(x)[0]), np.asarray(rho))
    np.testing.assert_array_equal(np.asarray(residual.field_at(x).driven()), driven)
    # Without ``driven`` the starting field's is packed, zero for a fresh run.
    start = residual.pack(rho, becsum_)
    np.testing.assert_array_equal(np.asarray(residual.unpack_field(start)),
                                  np.asarray(field.driven()))


def test_the_field_rows_count_towards_convergence(pseudo_dir):
    """The guard that has to fire: a residual zero in the density and not in the field.

    Such a state is self-consistent at the field it holds and is not the
    constrained one; a convergence measure that read the density alone would
    pass it. The share is ``|dm| (|B| + tau |dm|)`` with ``r_B = -tau dm``
    and ``B`` the field the state holds, so it fires at ``B = 0`` too, where
    ``|B| |dm|`` alone would read zero for any moment.
    """
    calculation = _calculation(pseudo_dir, FSM)
    residual = make_residual(calculation, 8, 1e-10)
    rho = calculation.starting_density()
    becsum_ = calculation.starting_becsum()
    held = residual.pack(rho, becsum_, driven=np.full(residual.field_shape, -0.011))
    bare = residual.pack(rho, becsum_, driven=np.zeros(residual.field_shape))
    r = np.zeros(residual.size)
    assert residual.field_accuracy(r, held) == 0.0
    r[-1] = -0.02 * 1e-3      # a moment 1e-3 mu_B off its target
    assert residual.field_accuracy(r, held) == pytest.approx(
        1e-3 * (0.011 + 0.02 * 1e-3), rel=1e-12)
    assert residual.field_accuracy(r, bare) == pytest.approx(0.02 * 1e-6, rel=1e-12)
    assert residual.field_accuracy(r, bare) > 1e-11
    r[-1] = -0.02 * 1e-9      # 1e-9 mu_B off: B dm = 1.1e-11 Ry
    assert residual.field_accuracy(r, held) > 1e-11
    assert residual.field_accuracy(r, bare) < 1e-11


def test_no_field_rows_without_a_feedback_scheme(pseudo_dir):
    calculation = _calculation(pseudo_dir, "b_field(3) = 0.01")
    residual = make_residual(calculation, 8, 1e-10)
    assert residual.field_shape is None and residual.feedback_field is None
    assert residual.unpack_field(np.zeros(residual.size)) is None
    assert residual.field_accuracy(np.ones(residual.size), np.ones(residual.size)) == 0.0


@pytest.mark.parametrize(
    "extra, card",
    [
        (FSM, ""),
        (NONCOLLINEAR + ", constrained_magnetization = 'atomic fsm', lambda = 0.02",
         MOMENTS),
        (NONCOLLINEAR
         + ", constrained_magnetization = 'atomic fsm direction', lambda = 0.02",
         MOMENTS),
    ],
    ids=["fsm", "atomic-fsm", "atomic-fsm-direction"],
)
def test_the_field_row_is_the_loop_s_elk_step(pseudo_dir, extra, card):
    """``elk_driven`` is the field ``feedback`` steps to under Elk's rule.

    The residual's row and the loop's update are written apart, so they are
    held to each other here, on a density whose moment is off every target.
    """
    calculation = _calculation(pseudo_dir, extra, card)
    field = calculation.magnetic_field
    assert field.fsm_update == "elk"
    rho = calculation.starting_density()
    cell = calculation.system.cell
    stepped = field.feedback(rho, cell).driven()
    row = field.elk_driven(rho, cell)
    np.testing.assert_array_equal(np.asarray(row), np.asarray(stepped))
    assert float(np.max(np.abs(np.asarray(row) - np.asarray(field.driven())))) > 1e-6


def test_a_secant_controller_is_refused(pseudo_dir):
    """The secant has memory and steps on converged pairs only, so it is no map."""
    calculation = _calculation(pseudo_dir, FSM + ", fsm_update = 'secant'")
    assert calculation.magnetic_field.fsm_update == "secant"
    with pytest.raises(NotImplementedError, match="secant"):
        make_residual(calculation, 8, 1e-10)


def test_an_unmeasured_feedback_regime_is_refused(pseudo_dir):
    """Only ``'fsm'`` collinear and the per-atom schemes noncollinear are measured.

    The per-atom schemes in a collinear run never get here: ``build_system``
    refuses them, so a noncollinear ``'fsm'`` is the one unmeasured case.
    """
    calculation = _calculation(
        pseudo_dir,
        NONCOLLINEAR
        + ", constrained_magnetization = 'fsm', lambda = 0.02, fixed_magnetization(3) = 0.1",
    )
    with pytest.raises(NotImplementedError, match="noncollinear run is not yet checked"):
        make_residual(calculation, 16, 1e-10)


def test_a_feedback_field_under_pools_is_refused(pseudo_dir):
    """The pooled ``F`` carries no field row, so the pair is refused before any pool work."""
    from defuqumat.scf.residual import PooledScfResidual

    residual = make_residual(_calculation(pseudo_dir, FSM), 8, 1e-10)
    with pytest.raises(NotImplementedError, match="k-point pools"):
        PooledScfResidual.of(residual, None, [0], None)


def test_a_decaying_field_is_refused(pseudo_dir):
    """``reducebf < 1``: the decay's only fixed point is the field-free problem.

    The loop stops at whatever field is left, so a residual solver carrying the
    scale would solve a different problem, by the field the loop had not shed.
    """
    calculation = _calculation(pseudo_dir, "b_field(3) = 0.01, reducebf = 0.9")
    assert calculation.magnetic_field.reducebf == 0.9
    with pytest.raises(NotImplementedError, match="reducebf"):
        make_residual(calculation, 8, 1e-10)


@pytest.mark.parametrize(
    "extra",
    [
        "b_field(3) = 0.01",
        "b_field(3) = 0.01, reducebf = 1.0",
    ],
    ids=["uniform-field", "reducebf-one"],
)
def test_a_field_the_loop_holds_fixed_builds_a_residual(pseudo_dir, extra):
    calculation = _calculation(pseudo_dir, extra)
    assert calculation.magnetic_field is not None
    residual = make_residual(calculation, 8, 1e-10)
    assert residual.size > 0


def test_a_noncollinear_penalty_builds_a_residual(pseudo_dir):
    """The measured penalty regime: noncollinear, no uniform field beside it."""
    calculation = _calculation(
        pseudo_dir, NONCOLLINEAR + ", constrained_magnetization = 'atomic', lambda = 0.5"
    )
    assert calculation.noncolin
    assert calculation.magnetic_field.constraint == "atomic"
    residual = make_residual(calculation, 16, 1e-10)
    assert residual.size > 0


def test_a_collinear_penalty_is_refused(pseudo_dir):
    """Measured against pw.x only on noncollinear inputs; collinear is not."""
    calculation = _calculation(pseudo_dir, "constrained_magnetization = 'atomic', lambda = 0.5")
    assert not calculation.noncolin
    with pytest.raises(NotImplementedError, match="collinear run is not yet checked"):
        make_residual(calculation, 8, 1e-10)


def test_a_field_together_with_a_penalty_never_reaches_the_residual(pseudo_dir):
    """Refused at input, as ``pw.x`` refuses it, so no residual guard is needed."""
    with pytest.raises(ValueError, match="together with constrained_magnetization"):
        _calculation(
            pseudo_dir,
            NONCOLLINEAR
            + ", b_field(3) = 0.01, constrained_magnetization = 'atomic', lambda = 0.5",
        )


def test_a_noncollinear_uniform_field_is_refused(pseudo_dir):
    """Only the collinear uniform field is measured against the mixing loop."""
    calculation = _calculation(pseudo_dir, NONCOLLINEAR + ", b_field(3) = 0.01")
    assert calculation.noncolin
    with pytest.raises(NotImplementedError, match="noncollinear run is not yet checked"):
        make_residual(calculation, 16, 1e-10)


def test_autodiff_jacobian_is_refused_under_a_field(pseudo_dir):
    """Every residual-solver number with a field used the central difference.

    The guard fires before the solver's first evaluation of ``F`` (the driver's
    warm-up mixing steps still run before it), so the state handed
    to the solver is never read.
    """
    from defuqumat.scf.solvers import newton_krylov

    calculation = _calculation(pseudo_dir, "b_field(3) = 0.01")
    residual = make_residual(calculation, 8, 1e-10)
    with pytest.raises(NotImplementedError, match="autodiff"):
        newton_krylov(residual, None, None, None, jvp="autodiff")


def test_the_sternheimer_jacobian_still_refuses_a_field(pseudo_dir):
    """The sibling refusal: ``jvp='sternheimer'`` has no field term.

    ``require_a_sternheimer_regime`` refuses any calculation with a magnetic
    field, and lifting the residual's own guard must not reach past it.
    """
    calculation = _calculation(pseudo_dir, "b_field(3) = 0.01")
    residual = make_residual(calculation, 8, 1e-10)
    with pytest.raises(NotImplementedError, match="magnetic field"):
        residual._refuse_sternheimer(None)


@pytest.mark.parametrize("scheme", ["total direction", "atomic texture"])
def test_an_unmeasured_penalty_is_refused(scheme):
    """No reference input carries these two, so they wait for a number.

    Both need a noncollinear cell, and the guard reads nothing but the field,
    so a stand-in field is enough to see it fire.
    """
    calculation = SimpleNamespace(
        system=None,
        magnetic_field=SimpleNamespace(constraint=scheme, reducebf=1.0, atomic=None),
    )
    with pytest.raises(NotImplementedError, match="not yet checked"):
        make_residual(calculation, 8, 1e-10)


def test_a_local_field_card_is_refused():
    """A per-atom field is held fixed like a uniform one, but is not measured."""
    calculation = SimpleNamespace(
        system=None,
        magnetic_field=SimpleNamespace(
            constraint="none", reducebf=1.0, atomic=np.zeros((2, 1))
        ),
    )
    with pytest.raises(NotImplementedError, match="LOCAL_MAGNETIC_FIELDS"):
        make_residual(calculation, 8, 1e-10)


_QE = Path(__file__).resolve().parents[1] / "data" / "qe"

#: A Hubbard U on the iron 3d shell, as in the verifier's probes.
_IRON_U = "HUBBARD {atomic}\nU Fe-3d 4.0\n"


def _from_text(pseudo_dir, text):
    from defuqumat.pseudo import read_upf

    system = build_system(parse_pw_input(text))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return Calculation(system, pseudos)


def _canted(scheme=None):
    """``fe2-canted-nosoc.in``, under a per-atom feedback scheme when given one."""
    text = (_QE / "fe2-canted-nosoc.in").read_text()
    if scheme is None:
        return text
    text = text.replace(
        "    nosym = .true.\n",
        f"    nosym = .true.\n    constrained_magnetization = '{scheme}'\n"
        "    lambda = 0.02\n")
    return text + "STARTING_MOMENTS\n  1.868 0 0\n  0 1.868 0\n"


def _fe_fsm_tetrahedra():
    return (_QE / "fe-fsm.in").read_text().replace(
        "occupations = 'smearing', smearing = 'marzari-vanderbilt', degauss = 0.05",
        "occupations = 'tetrahedra_opt'",
    ).replace(
        "K_POINTS\n 1\n   0.1875000  0.1875000  0.4375000   1.00",
        "K_POINTS automatic\n 2 2 2 0 0 0",
    )


@pytest.mark.parametrize(
    "case, match",
    [
        ("fsm-hubbard", "'fsm' with a Hubbard U is not yet checked"),
        ("fsm-tetrahedra", "occupations = 'tetrahedra_opt' is not yet checked"),
        ("fsm-gamma", "K_POINTS gamma"),
        ("fsm-meta-gga", "under the meta-GGA"),
        ("atomic-fsm-hubbard", "'atomic fsm' with a Hubbard U is not yet checked"),
    ],
)
def test_a_feedback_field_beside_an_unmeasured_term_is_refused(pseudo_dir, case, match):
    """``MEASURED_FEEDBACK`` keys on the scheme and ``noncolin`` only.

    Each of these pairs got through it before (the verifier's probes on
    ``fe-fsm.in`` and the canted pair): a Hubbard U, a tetrahedron
    occupation, half-sphere storage and a meta-GGA, none of which either
    measured cell carries. The gamma case is collinear and norm-conserving
    with ``nosym``, the one way to reach half-sphere storage: a spinor or an
    ultrasoft run at ``K_POINTS gamma`` is substituted to the full sphere at
    input, which is an ordinary one-point k-set.
    """
    if case == "fsm-hubbard":
        text = (_QE / "fe-fsm.in").read_text() + _IRON_U
        calculation = _from_text(pseudo_dir, text)
    elif case == "fsm-tetrahedra":
        calculation = _from_text(pseudo_dir, _fe_fsm_tetrahedra())
    elif case == "fsm-gamma":
        text = _SILICON.format(extra=FSM + ", nosym = .true.").replace(
            "K_POINTS automatic\n 2 2 2 0 0 0", "K_POINTS gamma")
        calculation = _from_text(pseudo_dir, text)
        assert calculation.system.kpoints.gamma_only
    elif case == "fsm-meta-gga":
        calculation = _calculation(pseudo_dir, FSM + ", input_dft = 'tb09'")
    else:
        calculation = _from_text(pseudo_dir, _canted("atomic fsm") + _IRON_U)
    with pytest.raises(NotImplementedError, match=match):
        make_residual(calculation, 16, 1e-10)


def test_the_unmeasured_terms_pass_without_a_feedback_field(pseudo_dir):
    """The converse: the same cells without the constraint still build a residual.

    So the refusals above are keyed on the pair, not on the term alone.
    """
    text = (_QE / "fe-fsm.in").read_text().replace(
        "    constrained_magnetization = 'fsm'\n", "")
    residual = make_residual(_from_text(pseudo_dir, text + _IRON_U), 16, 1e-10)
    assert residual.ns_shape is not None and residual.field_shape is None
    tetra = _fe_fsm_tetrahedra().replace("    constrained_magnetization = 'fsm'\n", "")
    assert make_residual(_from_text(pseudo_dir, tetra), 16, 1e-10).field_shape is None


def test_a_noncollinear_hubbard_u_is_refused_with_or_without_a_field(pseudo_dir):
    """A spinor ``ns`` is complex and ``ScfResidual.pack`` keeps its real part only.

    So ``F`` is not the mixing loop's map for any noncollinear Hubbard run,
    with or without a field (the spiral one was refused already, I72).
    """
    calculation = _from_text(pseudo_dir, _canted() + _IRON_U)
    assert calculation.magnetic_field is None
    assert np.iscomplexobj(np.asarray(calculation.starting_ns()))
    with pytest.raises(NotImplementedError, match="noncollinear run with a Hubbard U"):
        make_residual(calculation, 16, 1e-10)


def _unconverged(scheme, converged=False):
    return SimpleNamespace(
        field=SimpleNamespace(constraint=scheme), converged=converged,
        accuracy=6.5e-10, iterations=40, field_accuracy=6.48e-10,
    )


@pytest.mark.parametrize(
    "scheme, tolerance",
    [("fsm", "0.001 Bohr magnetons"), ("atomic fsm", "0.001 Bohr magnetons"),
     ("atomic fsm direction", "0.1 degrees")],
)
def test_an_unconverged_hand_over_under_a_feedback_field_warns(scheme, tolerance):
    """The guard that has to fire: Newton stopped short and the loop takes over.

    The loop accepts the field at its own moment tolerance, so
    ``SCFResult.converged`` would read True with the total off by the field
    share; the warning names that share and the tolerance.
    """
    from defuqumat.scf.driver import _warn_unconverged_hand_over

    with pytest.warns(RuntimeWarning, match="stopped without converging") as record:
        _warn_unconverged_hand_over(_unconverged(scheme), 1e-11)
    message = str(record[0].message)
    assert "6.480e-10 Ry" in message and tolerance in message
    assert repr(scheme) in message


@pytest.mark.parametrize("converged, field", [(True, True), (False, False)])
def test_the_hand_over_is_silent_when_newton_converged_or_has_no_field(converged, field):
    """The converse: a converged solve, or one with no feedback field, says nothing."""
    import warnings

    from defuqumat.scf.driver import _warn_unconverged_hand_over

    result = _unconverged("fsm", converged=converged)
    if not field:
        result.field = None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_unconverged_hand_over(result, 1e-11)


def test_run_scf_warns_when_newton_hands_over_short_under_a_feedback_field(pseudo_dir):
    """The same guard from inside ``run_scf``: one Newton step cannot reach 1e-11.

    The silicon cell under ``'fsm'``, Newton cut to one outer step and the
    loop after it to two iterations, so the run costs a few diagonalisations;
    what is asserted is only that the hand-over says so.
    """
    from defuqumat.pseudo import read_upf
    from defuqumat.scf.driver import run_scf

    calculation = _calculation(pseudo_dir, FSM)
    system = calculation.system
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    with pytest.warns(RuntimeWarning, match="Newton-Krylov solver stopped without converging"):
        result = run_scf(
            system, pseudos, calculation=calculation, conv_thr=1e-11, max_iterations=2,
            scf_solver="newton-krylov",
            scf_solver_options={"max_iterations": 1, "gmres_maxiter": 2, "warmup": 0},
        )
    assert result.solver is not None and not result.solver.converged
    assert result.solver.field_accuracy > 0.0
