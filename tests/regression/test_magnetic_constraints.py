"""P18: constrained moments and external fields, against QE.

QE's suite ships three constrained runs on bcc iron, one per scheme it
implements for a noncollinear calculation: the moment on each atom
(``i_cons = 1``), the *direction* of that moment (``i_cons = 2``), and the total
moment of the cell (``i_cons = 3``). All three are here, and with them the
identity that a constraint machinery which is switched off must change nothing.

**The references are regenerated rather than committed**, and for a reason
specific to this phase: ``noncolin-constrain_atomic.in`` carries a commented-out
``lambda = 1`` above the ``lambda = 0.005`` it actually sets, and the committed
2017 output prints a constraint energy of 8.022 Ry at the starting density,
which is the *unscaled* sum of squares -- what ``lambda = 1`` gives. The
committed output therefore does not belong to the committed input.
``tools/generate_reference.py`` re-runs all three with the vendored pw.x.

**A constrained total energy is a softer number than an unconstrained one.** The
penalty holds the moment away from where the functional wants it, so the energy
is first-order sensitive to exactly where the moment ends up, and QE's own last
two iterations of the atomic case still move the constraint energy by 1.3e-6 Ry.
That is why the tolerance here is looser than the 1e-8 an ordinary total energy
gets, and it is a property of the state rather than of either code.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from defuqumat.io import read_qe_output
from defuqumat.io.pwin import parse_pw_input, read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import run_scf
from defuqumat.system import build_system

pytestmark = [pytest.mark.regression, pytest.mark.slow]

#: A constrained run's total energy, for the reason in the module docstring. At
#: ``conv_thr = 1e-10`` -- what the unconstrained cases use, and what QE's
#: reference was generated at -- the same comparison is 1.5e-6: the penalty makes
#: the energy first-order sensitive to the moment, so both codes have to be run
#: further than usual before they are comparing the same state.
CONSTRAINED_RY = 3e-7


@pytest.fixture(autouse=True)
def _bound_compilation_cache():
    """Several cells of different shapes in one process; see ``CLAUDE.md`` on memory.

    The converged states stay cached and only the compiled code is dropped, which
    trades recompilation for a peak that does not grow with the number of cells.
    The cache below stays unbounded because the parametrized tests come back to
    every cell once per test function, so ``maxsize=2`` would rerun each SCF that
    many times (``OPEN.md`` X2).
    """
    yield
    import jax

    jax.clear_caches()


@lru_cache(maxsize=None)
def _run(path: Path, pseudo_dir: Path, conv_thr: float = 1e-13, max_iterations: int = 200):
    """A constrained noncollinear run, converged tightly enough to be compared.

    **1e-13 rather than 1e-11, and the four iterations it costs are the whole
    price.** A constraint on the total moment is a stiff global feedback, so the
    total energy trails the density residual by further than usual here:
    measured against `pw.x`'s -55.69055687 on
    ``noncolin-constrain_atomic.in``, the gap is **4.31e-07 Ry at 1e-11 in 43
    iterations and 1.91e-09 at 1e-13 in 47**, against a tolerance of 3e-07. The
    looser value failed it; the tighter one lands two orders inside `pw.x`'s own
    printed precision. 1e-15 buys nothing, reading 3.73e-08 in 64 iterations,
    which is that printed precision rather than a real move.
    """
    pwin = read_pw_input(path)
    system = build_system(pwin)
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    # The input's own ``mixing_beta``, which for these runs is not a detail: a
    # constraint on the *total* moment is a uniform field proportional to the
    # moment's error, which is a stiff global feedback. QE's input asks for 0.3
    # and oscillates by several Bohr magnetons per iteration even so; at the
    # 0.7 default neither code converges in 150 iterations.
    beta = float(pwin.get("electrons", "mixing_beta") or 0.7)
    return system, run_scf(
        system, pseudos, conv_thr=conv_thr, max_iterations=max_iterations,
        mixing_beta=beta,
    )


CASES = [
    ("noncolin-constrain_atomic.in", "atomic"),
    ("noncolin-constrain_angle.in", "atomic direction"),
    ("noncolin-constrain_total.in", "total"),
]


@pytest.mark.parametrize(("name", "scheme"), CASES)
def test_constrained_total_energy(name, scheme, qe_testsuite, pseudo_dir, benchmark):
    reference = read_qe_output(benchmark("pw_noncolin", name))
    system, result = _run(qe_testsuite / "pw_noncolin" / name, pseudo_dir)

    assert system.constrained_magnetization == scheme
    assert result.converged
    assert result.total_energy == pytest.approx(
        reference.total_energy, abs=CONSTRAINED_RY
    )

    # The constrained moment itself, to the two decimals QE prints.
    assert np.asarray(result.magnetization_vector) == pytest.approx(
        np.asarray(reference.magnetization_vector), abs=6e-3
    )


def test_constraint_energy_matches_qe(qe_testsuite, pseudo_dir, benchmark):
    """``etcon``, which QE prints at every iteration and never adds to the total.

    Checked at *both* ends: at the starting density, where it is decided by the
    penalty expression and the integration spheres alone and agrees to eight
    decimals, and at convergence, where it also depends on where each code's SCF
    stopped.
    """
    name = "noncolin-constrain_atomic.in"
    reference_text = Path(benchmark("pw_noncolin", name)).read_text()
    printed = [
        float(line.split("=")[1])
        for line in reference_text.splitlines()
        if "constraint energy" in line
    ]
    if not printed:
        pytest.skip("this reference does not print the constraint energy")

    system, result = _run(qe_testsuite / "pw_noncolin" / name, pseudo_dir)
    from defuqumat.scf.driver import Calculation

    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    starting = calculation.potential(calculation.starting_density())
    assert float(starting.e_constraint) == pytest.approx(printed[0], abs=1e-7)

    assert result.constraint_energy == pytest.approx(printed[-1], abs=1e-6)
    # And it is *not* in the total energy: QE prints it and never returns it.
    assert result.total_energy == pytest.approx(
        read_qe_output(benchmark("pw_noncolin", name)).total_energy, abs=CONSTRAINED_RY
    )


@pytest.mark.parametrize(("name", "scheme"), CASES)
def test_newton_krylov_reproduces_pw_x_under_a_constrained_moment(
    name, scheme, qe_testsuite, pseudo_dir, benchmark
):
    """I73: the residual solver on QE's three constrained iron inputs, against ``pw.x``.

    A penalty constraint's field is ``jax.grad`` of the penalty, so it is a
    function of the density and part of the residual's ``F``. Measured on D22
    at ``conv_thr = 1e-13``, one sample each (``review/incompat/I73/harnessA.log``,
    ``h3-angle.log``, ``h3-total.log``): the total energy is 7.8e-8 Ry from
    ``pw.x`` for ``atomic``, 2.7e-9 for ``atomic direction`` and 1.3e-8 for
    ``total``, against the 3e-7 the mixing route is held to; Newton converged
    in 13, 13 and 29 outer steps, and the mixing loop it hands over to stopped
    after one iteration each time, so the number is Newton's.

    **The loop's iteration count is what carries the claim**, not
    ``solver.converged``: a Newton whose ``F`` dropped the penalty converges on
    its own field-free residual, and the mixing loop after it (``max_iterations
    = 200``) then finishes the job and lands on ``pw.x`` anyway. Measured on D22
    with that mutation (``make_residual`` handed the calculation with
    ``magnetic_field = None``): every other assertion passes, and the loop takes
    14 iterations on ``atomic`` and 41 on ``total``, against 1 clean.
    ``atomic direction`` does not discriminate at all: its constraint energy at
    convergence is 5e-33, so the field-free root is the same root, and it is
    kept as a pw.x regression only. On ``atomic`` the constraint energy is
    0.00646111 against the 0.00646103 ``pw.x`` prints last. The ``total`` case
    took 370 s on one core, the other two under 90 s.
    """
    pwin = read_pw_input(qe_testsuite / "pw_noncolin" / name)
    system = build_system(pwin)
    assert system.constrained_magnetization == scheme
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    result = run_scf(
        system, pseudos, conv_thr=1e-13, max_iterations=200, mixing_beta=0.2,
        scf_solver="newton-krylov",
        scf_solver_options={"forcing": 0.5, "warmup": 6, "max_iterations": 40},
    )
    assert result.converged
    assert result.solver is not None and result.solver.converged
    # A Newton converged on the wrong problem is finished by the mixing loop,
    # which would then take more than a step or two (14 and 41 measured).
    assert result.iterations <= 2
    reference = read_qe_output(benchmark("pw_noncolin", name))
    assert result.total_energy == pytest.approx(
        reference.total_energy, abs=CONSTRAINED_RY
    )
    assert np.asarray(result.magnetization_vector) == pytest.approx(
        np.asarray(reference.magnetization_vector), abs=6e-3
    )
    if scheme == "atomic":
        printed = [
            float(line.split("=")[1])
            for line in Path(benchmark("pw_noncolin", name)).read_text().splitlines()
            if "constraint energy" in line
        ]
        assert result.constraint_energy == pytest.approx(printed[-1], abs=1e-6)


def test_no_constraint_is_no_change(qe_testsuite, pseudo_dir, tmp_path_factory):
    """``lambda`` at zero and no field: the same run as with no machinery at all.

    The field enters through ``v_scf``, so a zero field is a zero *added array*
    rather than a skipped code path -- which is exactly the kind of thing that
    is right to within round-off and worth pinning to it.
    """
    text = (qe_testsuite / "pw_noncolin" / "noncolin.in").read_text()
    directory = tmp_path_factory.mktemp("zero-field")
    with_field = directory / "zero-field.in"
    marker = text.lower().index("&system") + len("&system")
    with_field.write_text(
        text[:marker]
        + "\n    constrained_magnetization = 'atomic', lambda = 0.0\n"
        + text[marker:]
    )

    _, plain = _run(qe_testsuite / "pw_noncolin" / "noncolin.in", pseudo_dir)
    system, constrained = _run(with_field, pseudo_dir)

    assert system.constrained_magnetization == "atomic"
    assert constrained.constraint_energy == pytest.approx(0.0, abs=1e-14)
    assert constrained.total_energy == pytest.approx(plain.total_energy, abs=1e-9)


def test_a_local_field_breaks_the_symmetry_it_should(pseudo_dir, tmp_path_factory):
    """Elk's ``bfcmt`` and ``reducebf``: a field that starts a state and then leaves.

    The use case, exactly: **a run with no starting magnetization cannot become
    magnetic.** Nothing in a spin-symmetric functional breaks the symmetry, so
    the SCF sits in the nonmagnetic solution however far from the ground state
    it is -- for a hydrogen atom, 52 mRy above it. A field breaks the symmetry;
    ``reducebf`` then multiplies the field down to nothing, and what is left is
    the magnetic solution of the *field-free* problem.

    Three runs make that a test rather than an anecdote: the nonmagnetic one to
    show there is something to break out of, the properly-started magnetic one
    as the answer, and a field held fixed to show that the field itself distorts
    the state it is used to find.
    """
    from tests.conftest import GENERATED

    text = (GENERATED / "h-atom-lsda.in").read_text()
    directory = tmp_path_factory.mktemp("reducebf")
    marker = text.lower().index("&system") + len("&system")
    unmagnetised = text.replace(
        "starting_magnetization(1) = 0.6", "starting_magnetization(1) = 0.0"
    )
    card = "LOCAL_MAGNETIC_FIELDS\n 0.0 0.0 0.10\n"

    def variant(source: str, extra: str, name: str) -> Path:
        path = directory / name
        path.write_text(source[:marker] + extra + source[marker:] + card)
        return path

    nonmagnetic = directory / "nonmagnetic.in"
    nonmagnetic.write_text(unmagnetised)
    fading = variant(unmagnetised, "\n    reducebf = 0.5\n", "fading.in")
    held = variant(unmagnetised, "\n", "held.in")

    _, magnetic = _run(GENERATED / "h-atom-lsda.in", pseudo_dir)
    _, stuck = _run(nonmagnetic, pseudo_dir)
    system, reduced = _run(fading, pseudo_dir)
    _, fixed = _run(held, pseudo_dir)

    assert system.atomic_b_field == ((0.0, 0.0, 0.1),)
    # Without the field the run never finds the magnetic state at all.
    assert stuck.magnetization == pytest.approx(0.0, abs=1e-12)
    assert stuck.total_energy - magnetic.total_energy > 0.05

    # With it, and with reducebf, it finds it *and* ends up field-free.
    assert reduced.magnetization == pytest.approx(magnetic.magnetization, abs=1e-8)
    assert reduced.total_energy == pytest.approx(magnetic.total_energy, abs=1e-9)

    # A field left switched on finds the state too, and shifts it.
    assert fixed.magnetization == pytest.approx(magnetic.magnetization, abs=1e-8)
    assert abs(fixed.total_energy - magnetic.total_energy) > 1e-5


def _fsm(pseudo_dir, scheme, max_iterations):
    """``fe-fsm.in`` run with one of the two update rules."""
    from tests.conftest import GENERATED

    text = (GENERATED / "fe-fsm.in").read_text()
    marker = text.index("&system") + len("&system")
    source = text[:marker] + f"\n    fsm_update = '{scheme}'\n" + text[marker:]
    system = build_system(parse_pw_input(source))
    pseudos = tuple(
        read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species
    )
    return system, run_scf(
        system, pseudos, conv_thr=1e-8, max_iterations=max_iterations,
        mixing_beta=float(parse_pw_input(source).get("electrons", "mixing_beta") or 0.7),
    )


def test_fixed_spin_moment_holds_the_moment(pseudo_dir):
    """Elk's ``fsmtype``: a field driven by the moment's error, not a penalty.

    The difference from QE's schemes is why Elk has both. A penalty adds
    ``lambda (m - m_fix)^2`` to the energy and leaves a residual force on the
    moment -- the constrained state sits wherever the penalty balances the
    functional, which for iron's ``lambda = 0.005`` is 1.68 mu_B against a
    target of 0.5. The feedback scheme instead *searches* for the field at which
    the unconstrained functional puts the moment where it was asked, so what
    converges is a genuine stationary point under that field -- and the field it
    found is a result worth reading, which is why Elk prints it and
    :attr:`SCFResult.magnetic_field` carries it.

    bcc iron held at 2.0 mu_B, where it wants 3.18. The moment is part of the
    *convergence test*: the field is outside the density, so ``dr2`` falls below
    ``conv_thr`` long before the moment arrives.

    The budget is a **performance guard**, not a tolerance. The ``secant``
    update takes 74 iterations on this case, and if a change puts this back into
    the hundreds it is the controller that broke, not the physics. (It is no
    longer the default: Elk's rule, read off the output density, takes 20 --
    see the next test.)
    """
    system, result = _fsm(pseudo_dir, "secant", 300)

    assert system.constrained_magnetization == "fsm"
    assert system.fsm_update == "secant"
    assert result.converged
    assert result.iterations < 200
    assert result.magnetization == pytest.approx(2.0, abs=1e-3)
    # No penalty in the energy -- the constraint is entirely in the field.
    assert result.constraint_energy == pytest.approx(0.0, abs=1e-14)
    # ... and the field is one the run found for itself.
    assert abs(float(np.asarray(result.magnetic_field.uniform)[0])) > 1e-3


def test_the_two_fixed_spin_moment_rules_find_the_same_field(pseudo_dir):
    """``secant`` and ``elk`` are the same answer, and Elk's rule is now the faster.

    The transcription is kept and checked against the other scheme, the way
    every pluggable piece here is: what may differ is the path, never the fixed
    point. Both stop as soon as ``|m - 2|`` is inside 1e-3, so the residual
    difference in the field is that tolerance divided by the susceptibility --
    1.1e-3 mu_B over the 45 mu_B/Ry measured on this case, which is the 4e-5 Ry
    asserted below.

    **This test used to assert the opposite order, and the reason it was wrong
    is the finding.** Elk updates the field after every SCF iteration, and this
    code read the moment for that update off the *mixed* density, which lags the
    iteration's output by the mixer's damping. The controller then saw a
    susceptibility swinging between +2591 and -1252 mu_B/Ry from one iteration
    to the next and rang for 288, 1380 or 3380 iterations depending on
    round-off, and that was written up as the interleaved rule's nature. Elk
    reads ``mommt`` off the output density (``rhomag``, before ``mixerifc``), and
    read the same way this case converges in **20** iterations against the
    secant's 74 (``PLAN.md`` P108). The ordering is asserted because it is what
    the phase is about; the size of the gap is not, since both counts are
    properties of a controller and not of the physics.
    """
    _, secant = _fsm(pseudo_dir, "secant", 300)
    _, elk = _fsm(pseudo_dir, "elk", 300)

    assert secant.converged and elk.converged
    field_secant = float(np.asarray(secant.magnetic_field.uniform)[0])
    field_elk = float(np.asarray(elk.magnetic_field.uniform)[0])
    assert field_secant == pytest.approx(field_elk, abs=1e-4)
    assert secant.total_energy == pytest.approx(elk.total_energy, abs=1e-4)
    assert secant.magnetization == pytest.approx(2.0, abs=1e-3)
    assert elk.magnetization == pytest.approx(2.0, abs=1e-3)
    # Measured 20 against 74. Before P108 it was hundreds to thousands against
    # 74, so this is the assertion that fails if the read goes back to the
    # mixed density.
    assert elk.iterations < 60
    assert elk.iterations < secant.iterations


_FSM_SOC = """
 &control
    calculation = 'scf'
 /
 &system
    ibrav = 3, celldm(1) = 5.217, nat = 1, ntyp = 1,
    ecutwfc = 25.0, ecutrho = 200.0,
    occupations = 'smearing', smearing = 'marzari-vanderbilt', degauss = 0.05
    noncolin = .true.
    nosym = .true.
    noinv = .true.
    lspinorb = .true.
    soc_scale = %(soc)s
    starting_magnetization(1) = 0.5
    angle1(1) = 30.0
    angle2(1) = 0.0
%(constraint)s
 /
 &electrons
    conv_thr = 1.0d-10
 /
ATOMIC_SPECIES
 Fe 55.847 Fe.rel-pbe-spn-rrkjus_psl.0.2.1.UPF
ATOMIC_POSITIONS (alat)
 Fe 0.0 0.0 0.0
K_POINTS automatic
 2 2 2 1 1 1
"""

#: 1.0 mu_B, 30 degrees from z in the xz plane; the free moment there is 2.01.
_FSM_SOC_TARGET = np.array([0.5, 0.0, np.sqrt(3.0) / 2.0])


def _fsm_soc_run(pseudo_dir, soc, constraint, starting_from=None, max_iterations=1000):
    system = build_system(parse_pw_input(_FSM_SOC % dict(soc=soc, constraint=constraint)))
    pseudos = tuple(read_upf(pseudo_dir / s.pseudo_file) for s in system.structure.species)
    seed = {} if starting_from is None else dict(starting_from=starting_from)
    return system, run_scf(system, pseudos, conv_thr=1e-10, max_iterations=max_iterations,
                           mixing_beta=0.05, mixing_mode="adaptive", **seed)


def _fixed(field):
    return "\n".join(f"    b_field({i}) = {float(v):.15e}" for i, v in enumerate(field, 1))


@pytest.mark.parametrize("soc", ["1.0", "0.0"])
def test_fsm_under_spin_orbit_is_a_fixed_field_fixed_point(pseudo_dir, soc, monkeypatch):
    """I111: the fsm state under ``lspinorb`` is self-consistent under its own field held fixed.

    Elk's rule ends at a pair (density, ``B*``). The identity, which shares no
    code with the controller: the same ``B*`` as a plain ``b_field``, started
    from that density, is already self-consistent and keeps the moment on the
    target. Measured on D22 (``notes/incompat/I111.md``, ``r5-fsm.log``): first
    ``dr2`` 1.07e-10, total 2.2e-10 Ry from the fsm run, moment 2e-5 off.
    ``soc = 0`` is the term switched off: ``B*`` then has no transverse part
    (1.7e-7 Ry against 1.5e-4) and is 0.0119 Ry long against 0.0097. The
    alternative, the field without its transverse part, is
    ``test_fsm_under_spin_orbit_needs_its_transverse_field``. About 90 s (soc 1)
    and 170 s (soc 0) on one D22 core.
    """
    import defuqumat.scf.fields as fields

    # The harness's 'fsm-tight', the setting every number above was taken at.
    monkeypatch.setattr(fields, "FSM_TOLERANCE", 1.0e-5)
    constraint = "    constrained_magnetization = 'fsm'\n    lambda = 0.02\n" + "\n".join(
        f"    fixed_magnetization({i}) = {v:.12f}" for i, v in enumerate(_FSM_SOC_TARGET, 1))
    _, fsm = _fsm_soc_run(pseudo_dir, soc, constraint, max_iterations=1200)
    m = np.asarray(fsm.magnetization_vector, dtype=float)
    assert np.abs(m - _FSM_SOC_TARGET).max() < 1e-3
    field = np.asarray(fsm.magnetic_field.uniform, dtype=float)
    perp = field - (field @ _FSM_SOC_TARGET) * _FSM_SOC_TARGET
    if soc == "1.0":
        assert fsm.converged
        assert np.linalg.norm(perp) > 1e-4  # the coupling's transverse field
    else:
        assert np.linalg.norm(perp) < 1e-5
    # Measured at soc 1 only: two iterations, 2.2e-10 Ry. At soc 0 only the
    # first iteration was run (dr2 1.29e-10), so only that is asserted there.
    budget = 50 if soc == "1.0" else 1
    _, held = _fsm_soc_run(pseudo_dir, soc, _fixed(field), starting_from=fsm,
                           max_iterations=budget)
    assert held.history[0]["accuracy"] < 1e-9
    moment = np.asarray(held.history[0]["magnetization_vector"], dtype=float)
    assert np.abs(moment - _FSM_SOC_TARGET).max() < 1e-4
    if soc == "1.0":
        assert held.converged and held.iterations <= 3
        assert abs(held.total_energy - fsm.total_energy) < 1e-8


def test_fsm_under_spin_orbit_needs_its_transverse_field(pseudo_dir, monkeypatch):
    """The alternative to the identity above reads differently (I111).

    ``B*`` projected on the target direction, held fixed from the fsm state:
    without the transverse 1.5e-4 Ry the first iteration is no longer
    self-consistent: ``dr2`` 5.8e-7 against 1.07e-10 with the whole field
    (``notes/incompat/I111.md``, ``r6-par.log``). Only the first iteration is
    asserted: left to run, this field and the whole one both lose the state to
    the reversed moment once it is disturbed, because the constrained state
    under a fixed field opposing the moment is not a minimum (``bfrom`` in the
    record), so a longer run measures that instability and not the coupling.
    """
    import defuqumat.scf.fields as fields

    monkeypatch.setattr(fields, "FSM_TOLERANCE", 1.0e-5)
    constraint = "    constrained_magnetization = 'fsm'\n    lambda = 0.02\n" + "\n".join(
        f"    fixed_magnetization({i}) = {v:.12f}" for i, v in enumerate(_FSM_SOC_TARGET, 1))
    _, fsm = _fsm_soc_run(pseudo_dir, "1.0", constraint)
    field = np.asarray(fsm.magnetic_field.uniform, dtype=float)
    parallel = (field @ _FSM_SOC_TARGET) * _FSM_SOC_TARGET
    _, held = _fsm_soc_run(pseudo_dir, "1.0", _fixed(parallel), starting_from=fsm,
                           max_iterations=1)
    assert held.history[0]["accuracy"] > 1e-8
