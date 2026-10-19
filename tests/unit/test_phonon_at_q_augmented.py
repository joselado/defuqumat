"""The phonon at ``q != 0`` refuses an ultrasoft or PAW dataset, and says why.

``notes/INCOMPATIBILITIES.md`` I81, ``notes/incompat/I81.md``. The refusal is
:func:`~defuqumat.response.phononq.require_a_two_sphere_regime`'s first, and
until this file nothing fired it: no test built an augmented ``Calculation``
and asked for a phonon at ``q``. The guard needs no SCF, so these are unit
tests; each builds the ``Calculation`` and stops at the door.

Three entry points reach it -- the guard itself, the public
:func:`~defuqumat.response.phononq.dynamical_matrix_at_q`, which calls it
before it touches a state, and the electron-phonon coupling on an ultrasoft
metal, which inherits it from the phonon it is built on. Each is asserted on
the message, so a sibling refusal firing first cannot pass for this one, and a
norm-conserving cell is shown to pass this guard and reach the next one.
"""

import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation
from defuqumat.system.builder import build_system

pytestmark = pytest.mark.unit

_CELL = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = {alat}, nat = 2, ntyp = 1,
  ecutwfc = 12.0, ecutrho = {ecutrho}, nosym = .true., noinv = .true.
  {extra}
/
&electrons
/
ATOMIC_SPECIES
 {element} 28.0 {dataset}
ATOMIC_POSITIONS crystal
 {element} 0.00 0.00 0.00
 {element} 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


def _calculation(pseudo_dir, dataset, element="Si", alat=10.2, ecutrho=96.0,
                 extra=""):
    system = build_system(parse_pw_input(_CELL.format(
        alat=alat, ecutrho=ecutrho, extra=extra, element=element,
        dataset=dataset)))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file)
                    for s in system.structure.species)
    return Calculation(system, pseudos)


_AUGMENTED = ["Si.pz-n-rrkjus_psl.0.1.UPF", "Si.pz-n-kjpaw_psl.0.1.UPF"]


@pytest.mark.parametrize("dataset", _AUGMENTED)
def test_an_augmented_dataset_is_refused_at_q(pseudo_dir, dataset):
    """The guard fires on ultrasoft and on PAW, by its own message."""
    from defuqumat.response.phononq import require_a_two_sphere_regime

    calculation = _calculation(pseudo_dir, dataset)
    assert calculation.is_ultrasoft
    with pytest.raises(NotImplementedError, match="ultrasoft or PAW"):
        require_a_two_sphere_regime(calculation, np.array([0.5, 0.0, 0.0]))


@pytest.mark.parametrize("dataset", _AUGMENTED)
def test_the_public_entry_refuses_before_it_reads_a_state(pseudo_dir, dataset):
    """``dynamical_matrix_at_q`` calls the guard before anything uses the states,
    so ``None`` in their place is never read."""
    from defuqumat.response.phononq import dynamical_matrix_at_q

    calculation = _calculation(pseudo_dir, dataset)
    with pytest.raises(NotImplementedError, match="ultrasoft or PAW"):
        dynamical_matrix_at_q(calculation, None, None, None, (),
                              q=(0.5, 0.0, 0.0))


def test_the_coupling_on_an_ultrasoft_metal_inherits_the_refusal(pseudo_dir):
    """The electron-phonon coupling's own guards are the insulator and ``q = G``;
    an ultrasoft metal at ``q != G`` passes both and is stopped by the phonon's."""
    from defuqumat.response.elph import electron_phonon_at_q

    calculation = _calculation(
        pseudo_dir, "Al.pbe-n-rrkjus_psl.1.0.0.UPF", element="Al", alat=7.5,
        ecutrho=96.0, extra="occupations = 'smearing', degauss = 0.05")
    assert calculation.is_ultrasoft
    with pytest.raises(NotImplementedError, match="ultrasoft or PAW"):
        electron_phonon_at_q(calculation, None, None, None, (),
                             q=(0.5, 0.0, 0.0))


def test_a_norm_conserving_cell_passes_this_guard_and_meets_the_next(pseudo_dir):
    """The sibling order is unchanged: without augmentation the guard is silent,
    and the same cell with symmetry on is stopped by the small group of ``q``."""
    from defuqumat.response.phononq import require_a_two_sphere_regime

    calculation = _calculation(pseudo_dir, "Si.pz-vbc.UPF", ecutrho=48.0)
    assert not calculation.is_ultrasoft
    require_a_two_sphere_regime(calculation, np.array([0.5, 0.0, 0.0]))

    text = _CELL.format(alat=10.2, ecutrho=48.0, extra="", element="Si",
                        dataset="Si.pz-vbc.UPF")
    text = text.replace("nosym = .true., noinv = .true.", "")
    system = build_system(parse_pw_input(text))
    symmetric = Calculation(system, (read_upf(pseudo_dir / "Si.pz-vbc.UPF"),))
    with pytest.raises(NotImplementedError, match="small group of q"):
        require_a_two_sphere_regime(symmetric, np.array([0.5, 0.0, 0.0]))


# The parts of the message the earlier one (126d6ec) did not have, each a
# substring of the text as it is raised. The earlier message named four terms;
# the one asserted here says the route is a second implementation, adds P39's
# orthogonality set at q, the q-dependent frozen Hessian (addusdynmat's int2
# and int5), the PAW one-centre response, and points an insulator to a
# commensurate supercell. Matching on "ultrasoft or PAW" alone cannot tell the
# two apart, which is why this test exists.
_NEW_MESSAGE_PARTS = (
    "a second implementation rather than four missing terms",
    "(4) P39's whole orthogonality set at q",
    "compute_drhous.f90 builds vkb at ikq",
    "(5) a frozen Hessian that is not atom-diagonal",
    "addusdynmat contracts dvanqq's int2 and int5",
    "frozen_force_constants' 'the same matrix at every q' is a "
    "norm-conserving fact",
    "(6) for PAW the one-centre response at q",
    "Use a norm-conserving dataset, or, for an insulator, the Gamma phonon "
    "of a commensurate supercell",
)


@pytest.mark.parametrize("dataset", _AUGMENTED)
def test_the_refusal_names_what_the_route_lacks(pseudo_dir, dataset):
    """The message is pinned (the user's decision, 2026-10-16): every part the
    earlier four-term message lacked is present, and the earlier opening is not."""
    from defuqumat.response.phononq import require_a_two_sphere_regime

    calculation = _calculation(pseudo_dir, dataset)
    with pytest.raises(NotImplementedError) as raised:
        require_a_two_sphere_regime(calculation, np.array([0.5, 0.0, 0.0]))
    message = " ".join(str(raised.value).split())
    missing = [part for part in _NEW_MESSAGE_PARTS if part not in message]
    assert not missing, missing
    assert "four absent terms rather than the one" not in message
