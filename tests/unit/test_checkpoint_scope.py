"""A checkpoint is read only by the run that wrote it (audit 2026-11-20c #24).

``checkpoint_dir`` and ``mixing_from`` name one run's state on disk. Given to
the constructor they are that calculator's, and they used to cross into every
other run the calculator started: a derived calculator (``with_moments`` and
its siblings, ``relaxed``) and each ground state of a workflow that runs
several (the spiral scan, the relaxed anisotropy, the six magnetoelectric
runs). The fingerprint of ``load_state`` is shapes and counts, so a state of
another texture, geometry or wavevector was resumed without a word wherever
the shapes agreed.

Every test here replaces the SCF by a recorder: what is asserted is which
options reach it, and no ground state is run.
"""

import functools
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from defuqumat.calculator import Calculator
from defuqumat.scf.driver import SCF_CHECKPOINT
from defuqumat.system.kpoints import KPoints

pytestmark = pytest.mark.unit


SILICON = """
&control
  calculation = 'scf'
/
&system
  ibrav = 2, celldm(1) = 10.20, nat = 2, ntyp = 1, ecutwfc = 12.0,
  occupations = 'smearing', smearing = 'gaussian', degauss = 0.02,
  nosym = .true.
/
&electrons
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""

# ``nosym`` keeps the k-set of ``with_positions`` the parent's: with the point
# group on, moving an atom changes the wedge, and the plane-wave setup the
# child freezes was enumerated for the parent's k-points.
RELAXED_MAE = "tests/data/qe/co-tetragonal-relaxed-mae.in"
MAGNETOELECTRIC = "tests/data/qe/alas-magnetoelectric.in"
XZ = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0))


def _parent(pseudo_dir, directory, **defaults):
    """A converged calculator whose constructor named ``directory``."""
    options = dict(checkpoint_dir=directory, mixing_from=directory,
                   checkpoint_every=3, mixing_beta=0.3)
    options.update(defaults)
    parent = Calculator.from_text(SILICON, pseudo_dir, announce=False, **options)
    # ``_derived`` reads ``converged`` and hands the object on as the seed; the
    # recorder below never looks inside it.
    parent._scf = SimpleNamespace(converged=True)
    return parent


def _record_run_scf(monkeypatch):
    """Replace ``run_scf`` where the calculator looks it up, and record calls.

    The stand-in keeps the signature of the real one. The calculator forwards
    its defaults by named parameter, read off the function it is about to call,
    so a bare ``**options`` would receive none of them and every assertion
    below that an option is absent would pass on any code.
    """
    import defuqumat.calculator as calculator

    calls = []

    @functools.wraps(calculator.run_scf)
    def fake_run_scf(system, pseudos, **options):
        calls.append(options)
        return SimpleNamespace(converged=True)

    monkeypatch.setattr(calculator, "run_scf", fake_run_scf)
    return calls


def _relaxed(parent):
    """``parent.relaxed()`` on a relaxation that is already in the slot."""
    from defuqumat.workflows.relax import run_relax

    parent._relax = SimpleNamespace(converged=True, system=parent.system, scf=None)
    parent._relax_variable_cell = False
    parent._relax_options = parent._defaults_for(run_relax, {})
    return parent.relaxed()


#: Every public route to ``Calculator._derived``, and whether the child is
#: seeded with the parent's converged state.
CHILDREN = {
    "with_positions": (lambda c: c.with_positions(
        np.asarray(c.system.structure.positions) + 0.01), True),
    "with_cell": (lambda c: c.with_cell(np.asarray(c.system.cell.at) * 1.01), False),
    "with_kpoints": (lambda c: c.with_kpoints(
        KPoints.automatic((3, 3, 3), (0, 0, 0), c.system.cell)), True),
    "with_spin": (lambda c: c.with_spin(2), True),
    "with_moments": (lambda c: c.with_moments(
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])), True),
    "with_turning_moments": (lambda c: c.with_turning_moments(), True),
    "relaxed": (_relaxed, False),
}


@pytest.mark.parametrize("route", sorted(CHILDREN))
def test_a_child_does_not_read_the_checkpoint_of_its_parent(
        route, pseudo_dir, monkeypatch, tmp_path):
    """The parent's state file is in the directory, and the child is another run.

    Before the fix the child inherited ``checkpoint_dir`` with the other
    defaults, ``_seeded`` withheld the parent's converged seed because a state
    file was there, and ``run_scf`` resumed from it: the old texture for
    ``with_moments`` on an unchanged k-set, a fingerprint error for
    ``with_spin`` and ``with_kpoints``, a mid-run state of the old geometry for
    ``with_positions``.
    """
    make, seeded = CHILDREN[route]
    parent = _parent(pseudo_dir, tmp_path)
    (tmp_path / SCF_CHECKPOINT).write_bytes(b"")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        child = make(parent)

    calls = _record_run_scf(monkeypatch)
    child.get_scf()
    assert "checkpoint_dir" not in calls[-1]
    assert "mixing_from" not in calls[-1]
    assert calls[-1]["checkpoint_every"] == 3, (
        "the cadence names no state, so it stays")
    assert calls[-1]["mixing_beta"] == 0.3
    if seeded:
        assert calls[-1]["starting_from"] is parent._scf, (
            "with no directory of its own the child starts from the seed")
    assert "checkpoint_dir" not in child.defaults
    assert "mixing_from" not in child.defaults
    said = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert len(said) == 1 and "checkpoint_dir" in said[0] and "mixing_from" in said[0]


def test_the_warning_names_the_cure(pseudo_dir, tmp_path):
    parent = _parent(pseudo_dir, tmp_path)
    with pytest.warns(UserWarning) as caught:
        parent.with_spin(2)
    text = str(caught[0].message)
    assert "get_scf(checkpoint_dir=" in text
    assert caught[0].filename == __file__, (
        "the warning points at the line that derived the calculator")


def test_a_calculator_without_a_directory_derives_in_silence(pseudo_dir):
    """The control of the warning: one that always fired would say nothing."""
    parent = Calculator.from_text(SILICON, pseudo_dir, announce=False,
                                  checkpoint_every=3, mixing_beta=0.3)
    parent._scf = SimpleNamespace(converged=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        child = parent.with_spin(2)
        options = parent._shared_scf_options()
        inner = parent._inner_scf_options(None)
    assert child.defaults == {"checkpoint_every": 3, "mixing_beta": 0.3}
    assert options["checkpoint_every"] == 3 and inner["checkpoint_every"] == 3


def test_the_parent_keeps_its_own_directory(pseudo_dir, monkeypatch, tmp_path):
    """The run that writes the file is the one that reads it."""
    parent = _parent(pseudo_dir, tmp_path)
    parent._scf = None
    calls = _record_run_scf(monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        parent.get_scf()
    assert calls[-1]["checkpoint_dir"] == tmp_path
    assert calls[-1]["mixing_from"] == tmp_path


def test_a_child_checkpoints_in_a_directory_named_at_its_own_call(
        pseudo_dir, monkeypatch, tmp_path):
    """The flow ``test_continuation_machinery`` pins, from a parent with a directory.

    It passes before and after the fix, which is the point: the child's own
    directory is read and written as it was.
    """
    parent = _parent(pseudo_dir, tmp_path / "parent")
    own = tmp_path / "child"
    own.mkdir()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        child = parent.with_spin(2)
    calls = _record_run_scf(monkeypatch)

    child.get_scf(checkpoint_dir=own)
    assert calls[-1]["checkpoint_dir"] == own
    assert calls[-1]["starting_from"] is parent._scf

    (own / SCF_CHECKPOINT).write_bytes(b"")
    child._scf = None
    child.get_scf(checkpoint_dir=own)
    assert calls[-1]["checkpoint_dir"] == own
    assert "starting_from" not in calls[-1], "its own checkpoint beats the seed"


# --- the workflows that run several ground states in one call ---------------


def test_the_shared_options_of_a_workflow_leave_the_directory_out(
        pseudo_dir, tmp_path):
    """``_shared_scf_options`` and ``_inner_scf_options``, the two routes.

    They feed the spiral scan, the relaxed anisotropy and the magnetoelectric
    difference, whose ground states would otherwise all read and write the one
    directory.
    """
    calc = _parent(pseudo_dir, tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        shared = calc._shared_scf_options()
        inner = calc._inner_scf_options(None)
    for options in (shared, inner):
        assert "checkpoint_dir" not in options
        assert "mixing_from" not in options
        assert options["mixing_beta"] == 0.3
        assert options["checkpoint_every"] == 3
    assert len(caught) == 2 and all(
        "checkpoint_dir" in str(w.message) for w in caught)

    # What the caller wrote at the call site is the caller's, and it stays.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        named = calc._inner_scf_options({"checkpoint_dir": "mine"})
    assert named["checkpoint_dir"] == "mine"


def _recorder(function, seen):
    """A stand-in with the signature of ``function`` that records its options."""

    @functools.wraps(function)
    def fake(*args, **options):
        seen.append(options)
        return "result"

    return fake


def test_the_spiral_scan_takes_the_directory_only_from_its_own_call(
        pseudo_dir, monkeypatch, tmp_path):
    import defuqumat.workflows.spiral as spiral

    seen = []
    monkeypatch.setattr(spiral, "run_spiral_scan",
                        _recorder(spiral.run_spiral_scan, seen))
    calc = _parent(pseudo_dir, tmp_path)
    wavevectors = [(0.0, 0.0, 0.1), (0.0, 0.0, 0.2)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc.get_spiral_scan(wavevectors)
    assert "checkpoint_dir" not in seen[-1] and "mixing_from" not in seen[-1]
    assert seen[-1]["mixing_beta"] == 0.3
    assert len(caught) == 1 and "get_spiral_scan" in str(caught[0].message)
    assert caught[0].filename == __file__, (
        "through the pools guard the warning still points at the caller")

    # Named in the call it is passed on: every wavevector then shares the
    # directory, which costs iterations and not the answer. Nothing is said
    # about an option the call replaced, only about the one it did not.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc.get_spiral_scan(wavevectors, checkpoint_dir=tmp_path)
    assert seen[-1]["checkpoint_dir"] == tmp_path
    assert "mixing_from" not in seen[-1]
    assert len(caught) == 1
    assert "not given mixing_from of" in str(caught[0].message)


def test_the_relaxed_anisotropy_does_not_take_the_directory_from_the_defaults(
        pseudo_dir, monkeypatch, tmp_path):
    import defuqumat.workflows.anisotropy as anisotropy

    seen = []
    monkeypatch.setattr(anisotropy, "run_relaxed_anisotropy",
                        _recorder(anisotropy.run_relaxed_anisotropy, seen))
    calc = _parent(pseudo_dir, tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calc.get_relaxed_anisotropy(directions="xz")
    assert "checkpoint_dir" not in seen[-1] and "mixing_from" not in seen[-1]
    assert seen[-1]["mixing_beta"] == 0.3
    assert len(caught) == 1 and "get_relaxed_anisotropy" in str(caught[0].message)


def test_the_magnetoelectric_ground_states_do_not_take_the_directory(
        monkeypatch, tmp_path):
    """Six ground states at six fields, which ``chain=False`` keeps apart."""
    import defuqumat.scf.driver as driver

    seen = {}

    class Stop(Exception):
        pass

    def recording(system, pseudos, **options):
        seen.clear()
        seen.update(options)
        raise Stop

    monkeypatch.setattr(driver, "run_scf", recording)
    calc = Calculator.from_file(
        MAGNETOELECTRIC, pseudo_dir="tests/data/pseudo", announce=False,
        checkpoint_dir=tmp_path, mixing_from=tmp_path, checkpoint_every=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(Stop):
            calc.get_magnetoelectric_tensor(directions=(2,))
    assert "checkpoint_dir" not in seen and "mixing_from" not in seen
    assert seen["mixing_beta"] == 0.7 and seen["checkpoint_every"] == 3
    assert [w for w in caught if "get_magnetoelectric_tensor" in str(w.message)]

    # Named in ``scf_options`` it is the caller's: the six runs then start from
    # each other's states, as ``chain=True`` makes them.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(Stop):
            calc.get_magnetoelectric_tensor(
                directions=(2,), scf_options={"checkpoint_dir": tmp_path / "me"})
    assert seen["checkpoint_dir"] == tmp_path / "me"


def test_a_relaxed_anisotropy_refuses_one_directory_for_several_directions(
        pseudo_dir, monkeypatch, tmp_path):
    """The case that gives a wrong number, refused before the first SCF.

    Direction 2 would resume the state direction 1 left, with its moment along
    direction 1. A cardinal axis is stationary, so the run converges back
    there and the two total energies come out nearly equal.
    """
    import defuqumat.workflows.anisotropy as anisotropy

    def never(*args, **options):
        raise AssertionError("the refusal comes before the first direction")

    monkeypatch.setattr(anisotropy, "run_relaxed_direction", never)
    with pytest.raises(ValueError, match="checkpoint_dir.*2 directions"):
        anisotropy.run_relaxed_anisotropy(None, None, directions=XZ,
                                          checkpoint_dir=tmp_path)

    calc = Calculator.from_text(SILICON, pseudo_dir, announce=False)
    with pytest.raises(ValueError, match="checkpoint_dir"):
        calc.get_relaxed_anisotropy(directions="xz", checkpoint_dir=tmp_path)


def test_one_relaxed_direction_still_checkpoints(monkeypatch, tmp_path):
    """The regime next to the refusal: one direction is one run."""
    import defuqumat.scf.driver as driver
    import defuqumat.workflows.anisotropy as anisotropy

    seen = {}

    class Stop(Exception):
        pass

    def recording(system, pseudos, **options):
        seen.update(options)
        raise Stop

    monkeypatch.setattr(driver, "run_scf", recording)
    calc = Calculator.from_file(RELAXED_MAE, pseudo_dir="tests/data/pseudo",
                                announce=False)
    with pytest.raises(Stop):
        anisotropy.run_relaxed_direction(
            calc.system, calc.pseudos, direction=(0.0, 0.0, 1.0),
            checkpoint_dir=tmp_path)
    assert seen["checkpoint_dir"] == tmp_path

    seen.clear()
    with pytest.raises(Stop):
        anisotropy.run_relaxed_anisotropy(
            calc.system, calc.pseudos, directions=((0.0, 0.0, 1.0),),
            checkpoint_dir=tmp_path)
    assert seen["checkpoint_dir"] == tmp_path
