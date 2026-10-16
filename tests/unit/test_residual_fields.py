"""The residual solver under an applied field or a constrained moment (I73).

``ScfResidual.step`` builds its potential with ``Calculation.potential(rho)``,
which applies ``calculation.magnetic_field`` at full strength, so for a field
the loop never changes the map ``x -> F(x)`` is already a function of the
state alone. The mixing loop changes the field in exactly two places
(``run_scf``: ``field_scale *= field.reducebf`` and ``field.feedback``), and
``feedback`` is the identity outside :data:`~defuqumat.scf.fields.FEEDBACK`.
So a uniform ``B_field`` in a collinear run and the measured *penalty*
constraints in a noncollinear run (neither beside the other) run, and what
stays refused as physics is the two cases where the loop moves the field: a
feedback (fixed-spin-moment) constraint, whose field is a second unknown solved
for beside the density, and ``reducebf < 1``, whose fixed point moves with the
accumulated scale. A ``LOCAL_MAGNETIC_FIELDS`` card and the ``total direction``
and ``atomic texture`` penalties, a noncollinear uniform field, a collinear
penalty and ``jvp='autodiff'`` under any field are refused only because nothing
measured them; a field together with a penalty is refused at input.

These tests build a ``Calculation`` and nothing more; the fixed point itself,
Newton against the mixing loop, is ``tests/regression/test_scf_solvers.py``'s
``test_newton_krylov_finds_the_mixer_s_fixed_point_under_a_field`` (slow).
"""

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


def _calculation(pseudo_dir, extra):
    from defuqumat.pseudo import read_upf

    text = _SILICON.format(extra=extra)
    if "noncolin" in extra:
        text = text.replace("nspin = 2, ", "")
    system = build_system(parse_pw_input(text))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return Calculation(system, pseudos)


def test_a_feedback_constraint_is_refused(pseudo_dir):
    """``fsm`` drives its field from the moment's error between iterations."""
    calculation = _calculation(
        pseudo_dir,
        "constrained_magnetization = 'fsm', lambda = 0.02, fixed_magnetization(3) = 0.1",
    )
    assert calculation.magnetic_field is not None
    with pytest.raises(NotImplementedError, match="feedback"):
        make_residual(calculation, 8, 1e-10)


def test_a_decaying_field_is_refused(pseudo_dir):
    """``reducebf < 1`` makes the loop's fixed point a function of the scale."""
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
