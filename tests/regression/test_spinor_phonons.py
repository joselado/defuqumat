"""The ``Gamma`` dynamical matrix of a spinor run without a magnetization.

``noncolin = .true.`` with no moment, with or without ``lspinorb``: the
assembly is written (``notes/incompat/I3.md``) and **still refused** by
:func:`~defuqumat.response.phonon._require_no_spinor`, because spin-orbit
bismuthene does not agree with ``ph.x`` and the reason is unlocated. The
measurements below are taken **with that guard bypassed** (the
``spinor_guard_bypassed`` fixture), the way ``_require_one_spin_channel``'s LSDA
numbers were, so that the assembly stays measured while it is refused; the
gate test asserts that both refusals fire through the front door.

**The anchors are one identity and one external number.**

1. The same silicon run as a scalar and as a spinor with no magnetization is
   the same physics on a doubled space, so the force constants must be the same
   matrix. One cell per dataset kind, because each kind reaches a different
   term: the norm-conserving one only the ``spinors=True`` opt-in of the frozen
   energy, the ultrasoft and PAW ones also ``qq_so`` in
   :func:`~defuqumat.response.phonon.overlap_derivatives`, the multipliers'
   response and the moving augmentation charge. A scalar-relativistic dataset
   has ``fcoef = 1``, so the identity exercises the spin blocks and
   *distinguishes* none of them.
2. ``ph.x`` on the fully-relativistic ultrasoft AlAs
   (``reference.out.ph-alas-epsilon-us-soc``), where ``fcoef`` is not the
   identity, though the coupling is small there (see the test).

The second half of the file is the **magnetic** spinor (I79,
``notes/incompat/I79.md``), also refused and measured with its guard bypassed:
the collinear limit, a spinor with the moment along ``z`` against the
``nspin = 2`` run, on triplet O2 and on the antiferromagnetic hydrogen chain.

The file visits more than three cells, so it clears the compiled code between
tests and keeps two converged states at most (``CLAUDE.md``, memory).
"""

from functools import lru_cache
from pathlib import Path
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response.phonon import dynamical_matrix
from defuqumat.scf import run_scf
from defuqumat.system import build_system


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    """``jax.clear_caches()`` between tests: each cell compiles its own stack."""
    yield
    jax.clear_caches()


pytestmark = [pytest.mark.regression]


@pytest.fixture
def spinor_guard_bypassed(monkeypatch):
    """The spinor refusal switched off, for the measurements only."""
    from defuqumat.response import phonon

    monkeypatch.setattr(phonon, "_require_no_spinor", lambda calculation: None)

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


def _reference_frequencies(name: str) -> np.ndarray:
    """The ``freq (  n) = ... [cm-1]`` lines ``dyndia`` prints, in order."""
    text = (CASES / name).read_text()
    found = re.findall(
        r"freq \(\s*\d+\) =\s*\S+ \[THz\] =\s*(\S+) \[cm-1\]", text
    )
    return np.array([float(value) for value in found])


def _calculation(case: str, noncolin: bool):
    from defuqumat.scf import Calculation

    parsed = read_pw_input(CASES / f"{case}.in")
    if noncolin:
        parsed.namelists["system"]["noncolin"] = True
    system = build_system(parsed)
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    return system, pseudos, Calculation(system, pseudos)


@lru_cache(maxsize=2)
def _phonons(case: str, noncolin: bool):
    """One cell converged and its ``Gamma`` dynamical matrix solved.

    ``noncolin`` adds the one line to the committed input, so the scalar and
    the spinor side of the identity cannot drift apart. ``maxsize = 2``: the
    identity compares two, and each holds the wavefunctions and ``3 nat``
    first-order states.
    """
    system, pseudos, calculation = _calculation(case, noncolin)
    assert calculation.noncolin == (noncolin or system.noncolin)
    if calculation.noncolin:
        # The regime this file is about, or a pass means nothing.
        assert calculation.nspin_mag == 1
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=100)
    assert result.converged
    phonons = dynamical_matrix(
        calculation, result.wavefunctions, result.eigenvalues, result.density,
        result.becsum,
    )
    assert phonons.converged
    return calculation, result, phonons


#: ``max|D_spinor - D_scalar|`` in Ry/bohr^2, on force constants of 0.28.
#: Measured on D22 (2026-10-16, one run each, ``conv_thr = 1e-12``, the default
#: ``tr2 = 1e-18``): **2.7e-10** norm-conserving, **1.2e-9** ultrasoft and
#: **1.7e-9** PAW, with the optical triplets equal to 1e-6 cm^-1 and the
#: acoustic ones to 1e-4. Not a route difference: the scalar and the spinor
#: Davidson converge their ground states to different last digits, the scatter
#: ``test_spinor_dielectric.py`` records for ``epsilon``. The bounds are 36x,
#: 86x and 60x over it, and a dropped spin block or a factor of two in the spin
#: sum is of the order of the matrix itself.
IDENTITY = {
    "si-epsilon": 1e-8,
    "si-epsilon-us": 1e-7,
    "si-epsilon-paw": 1e-7,
}


@pytest.mark.slow
@pytest.mark.parametrize("case", sorted(IDENTITY))
def test_a_nonmagnetic_spinor_gives_the_scalar_run_s_force_constants(
        case, spinor_guard_bypassed):
    """The identity, one dataset kind at a time.

    Four bands of two electrons against eight of one: a factor of two lost in
    the spin sum, a dropped spin block of ``qq_so`` or a refused energy shows
    here at the size of the matrix itself.
    """
    scalar = _phonons(case, False)[2]
    spinor = _phonons(case, True)[2]
    difference = float(np.max(np.abs(spinor.matrix - scalar.matrix)))
    scale = float(np.max(np.abs(scalar.matrix)))
    print(f"\n{case}: max|D_spinor - D_scalar| = {difference:.3e} on {scale:.4f}; "
          f"frequencies scalar {np.array2string(scalar.frequencies, precision=6)} "
          f"spinor {np.array2string(spinor.frequencies, precision=6)}")
    assert difference < IDENTITY[case]


#: ``ph.x`` on the fully-relativistic ultrasoft AlAs, read at test time.
RELATIVISTIC_REFERENCE = "reference.out.ph-alas-epsilon-us-soc"

#: The bound on the optical triplet against ``ph.x``, cm^-1. Measured on D22
#: (2026-10-16, one run): **376.717753 against 376.714126**, 3.6e-3 cm^-1
#: (9.6e-6 relative), so the bound is 5.5x over it.
OPTICAL_TOLERANCE = 0.02


@pytest.mark.slow
def test_the_relativistic_ultrasoft_phonon_matches_quantum_espresso(
        spinor_guard_bypassed):
    """AlAs with spin-orbit coupling: the optical triplet against ``ph.x``.

    ``ph.x`` ran at ``tr2_ph = 1e-14`` and this at the default 1e-18; the
    optical triplet is converged at either, the acoustic one is not the check
    (``OPEN.md`` Part XXVII) and is only printed.

    **What this cannot see.** Spin-orbit coupling moves ``ph.x``'s triplet by
    only 0.073 cm^-1 (376.641198 with the scalar files), so a dropped ``fcoef``
    dressing would pass here, as it would in the dielectric constant of the
    same cell. And the **scalar** sibling is not this close: the same cell with
    the scalar files gives 376.732697 against ``ph.x``'s 376.641198, 0.0915
    cm^-1, and 376.730210 at ``ph.x``'s own ``tr2 = 1e-14``, so it is not the
    loop's stop. That number belongs to the scalar route, which had no AlAs
    phonon test, and it is recorded in ``notes/incompat/I3.md`` rather than
    explained; it is why the spin-orbit *shift* is not asserted here.
    """
    calculation, _, relativistic = _phonons("alas-epsilon-us-soc", False)
    assert calculation.noncolin and calculation.is_ultrasoft
    assert any(p.has_so for p in calculation.pseudos)
    assert calculation.nspin_mag == 1

    reference = _reference_frequencies(RELATIVISTIC_REFERENCE)
    assert reference.shape == (6,)
    print(f"\nrelativistic AlAs: {np.array2string(relativistic.frequencies, precision=6)}"
          f" against {reference}")
    np.testing.assert_allclose(relativistic.frequencies[3:], reference[3:],
                               atol=OPTICAL_TOLERANCE)


def test_a_spinor_phonon_is_refused_by_name():
    """Both halves of the spinor regime refuse, each with its own reason.

    Cheap: no SCF, the guard reads the calculation alone. Each cell has to *be*
    the regime its message names, or the test passes for the wrong reason.
    """
    _, _, textured = _calculation("i-atom-soc", False)
    assert textured.noncolin and textured.nspin_mag == 4
    with pytest.raises(NotImplementedError, match="with a magnetization"):
        dynamical_matrix(
            textured, jnp.zeros((1, 1, 1, 1)), jnp.zeros((1, 1, 1)),
            jnp.zeros((4, 1, 1, 1)),
        )

    _, _, plain = _calculation("alas-epsilon-us-soc", False)
    assert plain.noncolin and plain.nspin_mag == 1
    with pytest.raises(NotImplementedError, match="bismuthene"):
        dynamical_matrix(
            plain, jnp.zeros((1, 1, 1, 1)), jnp.zeros((1, 1, 1)),
            jnp.zeros((1, 1, 1, 1)),
        )


# --- I79: a spinor with a magnetization ------------------------------------


@pytest.fixture
def textured_guard_bypassed(monkeypatch, spinor_guard_bypassed):
    """The spinor phonon refusal and the textured dielectric one, both off."""
    from defuqumat.response import efield

    monkeypatch.setattr(efield, "require_a_measured_spinor_response",
                        lambda calculation: None)


def _magnetic(case: str, edit, origin_tangent: bool = True):
    """``case`` with ``edit`` applied to its ``&system``, converged and solved."""
    from defuqumat.scf import Calculation

    parsed = read_pw_input(CASES / f"{case}.in")
    if edit is not None:
        edit(parsed.namelists["system"])
    system = build_system(parsed)
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos, origin_tangent=origin_tangent)
    result = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                     max_iterations=200)
    assert result.converged
    phonons = dynamical_matrix(
        calculation, result.wavefunctions, result.eigenvalues, result.density,
        result.becsum, max_iterations=80,
    )
    assert phonons.converged
    return calculation, result, phonons


def _o2_spinor(nosym: bool):
    """Triplet O2 as a spinor, the moment along z and no spin-orbit coupling."""
    def edit(namelist):
        namelist.pop("nspin")
        namelist.pop("tot_magnetization")
        namelist["noncolin"] = True
        namelist["nbnd"] = 16
        if nosym:
            namelist["nosym"] = True
    return edit


def _afm_spinor(namelist):
    """The antiferromagnetic chain as a spinor, the moments along +z and -z."""
    namelist.pop("nspin")
    namelist["noncolin"] = True
    namelist["nbnd"] = 24


def test_the_walked_route_refuses_a_spinor_even_past_the_spinor_guard(
        spinor_guard_bypassed):
    """The sibling refusal behind ``_require_no_spinor``: the k-chunked route.

    ``chunked_phonon.py`` has the collinear ``qq`` and no spinor energy, so a
    spinor that reached it would be assembled with the wrong overlap. A numpy
    store is a streamed one, which takes that route with no SCF. Run with the
    front-door guard bypassed, or the test would pass on the wrong message.
    """
    parsed = read_pw_input(CASES / "h-chain-afm.in")
    _afm_spinor(parsed.namelists["system"])
    system = build_system(parsed)
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    from defuqumat.scf import Calculation

    calculation = Calculation(system, pseudos)
    assert calculation.noncolin and calculation.nspin_mag == 4
    with pytest.raises(NotImplementedError, match="walked a k-chunk at a time"):
        dynamical_matrix(calculation, np.zeros((1, 2, 24, 8)),
                         jnp.zeros((1, 2, 24)), None)


#: ``max|D_spinor - D_lsda|`` in Ry/bohr^2 for the collinear limit of a magnetic
#: spinor, measured on D22 (2026-10-16, one run each, ``conv_thr = 1e-12``, the
#: default ``tr2``): **8.3e-6** on triplet O2 (force constants of 1.68, all of
#: it in the transverse block), **1.9e-6** on the antiferromagnetic chain (0.038).
#: Not round-off: on the collinear ground state written as a spinor the two are
#: still 2.3e-6 and 1.4e-6 apart, against 2.7e-10 for a nonmagnetic spinor, and
#: the reason is unlocated (``notes/incompat/I79.md``). Each is one sample, it
#: sits in the transverse and acoustic blocks, which move with the loop's stop
#: (``OPEN.md`` Part XXVII), and the spinor loop stops at ``tr2 / npol`` over
#: four channels, so the bounds are 12x and 11x over it; a dropped spin block or
#: a factor of two is of the order of 1e-2 and is still caught.
MAGNETIC_IDENTITY = {"o2": 1e-4, "hafm": 2e-5}


@pytest.mark.slow
def test_a_magnetic_spinor_along_z_gives_the_lsda_force_constants(
        textured_guard_bypassed):
    """Triplet O2 as a spinor with the moment along ``z`` against ``nspin = 2``.

    Without spin-orbit coupling a displacement induces no transverse moment, so
    the 4x4 kernel's longitudinal block is the LSDA kernel and the two runs are
    one calculation. Both take ``origin_tangent=False``, the convention the
    LSDA run was checked against ``ph.x`` in (``test_lsda_response.py``). The
    dielectric tensor of the pair is printed beside it: 1.110915839 against
    1.11091585 across the axis and 1.198006816 against 1.198006981 along it.
    """
    from defuqumat.response.efield import dielectric_tensor

    _, _, lsda = _magnetic("o2-fixed-lsda", None, origin_tangent=False)
    calculation, result, spinor = _magnetic(
        "o2-fixed-lsda", _o2_spinor(nosym=True), origin_tangent=False)
    assert calculation.noncolin and calculation.nspin_mag == 4
    assert not any(p.has_so for p in calculation.pseudos)
    difference = float(np.abs(spinor.matrix - lsda.matrix).max())
    epsilon = dielectric_tensor(
        calculation, result.wavefunctions, result.eigenvalues, result.density,
        result.becsum, max_iterations=40).epsilon
    print(f"\nO2 spinor against LSDA: {difference:.3e}; "
          f"eps {np.array2string(np.diag(np.asarray(epsilon)), precision=9)}")
    assert difference < MAGNETIC_IDENTITY["o2"]


@pytest.mark.slow
def test_the_symmetrised_magnetic_spinor_is_the_nosym_one(spinor_guard_bypassed):
    """The same O2 spinor with its eight magnetic operations, against ``nosym``.

    Measured at **2.3e-8**. It exercises the axial displacement average on the
    grid and in ``becsum`` end to end, but **it cannot tell the axial rule from
    the scalar one**: the response of a collinear state has no transverse
    magnetization, and on ``m_z`` alone the magnetic group's signs are all +1.
    The rule itself is checked in
    ``tests/unit/test_magnetic_displacement_symmetry.py``.
    """
    _, _, free = _magnetic("o2-fixed-lsda", _o2_spinor(nosym=True),
                           origin_tangent=False)
    calculation, _, wedge = _magnetic("o2-fixed-lsda", _o2_spinor(nosym=False),
                                      origin_tangent=False)
    assert calculation.use_symmetry and calculation.symmetries.nsym > 1
    assert float(np.abs(wedge.matrix - free.matrix).max()) < 1e-6


@pytest.mark.slow
def test_an_antiferromagnetic_spinor_metal_gives_the_lsda_force_constants(
        spinor_guard_bypassed):
    """The hydrogen chain with moments along ``+z`` and ``-z``, a metal.

    The smeared branch of the solve and the Fermi level's shift at ``q = 0``,
    with the magnetization's own local density of states in it.
    """
    _, _, lsda = _magnetic("h-chain-afm", None)
    calculation, _, spinor = _magnetic("h-chain-afm", _afm_spinor)
    assert calculation.noncolin and calculation.nspin_mag == 4
    difference = float(np.abs(spinor.matrix - lsda.matrix).max())
    print(f"\nAFM chain spinor against LSDA: {difference:.3e}")
    assert difference < MAGNETIC_IDENTITY["hafm"]


@pytest.mark.slow
def test_the_textured_wedge_is_the_whole_zone(spinor_guard_bypassed):
    """The helix on its 4-operation wedge against the whole zone, the whole matrix.

    The end-to-end check of the axial displacement average, on a texture where
    the transverse magnetization is not zero and the axial rule is therefore
    not the scalar one. **Not measured yet**: queued on D22 on 2026-10-16
    (``/l/ladovj1/review/incompat/I79/helix.log``, which also runs the scalar
    rule and no symmetrisation), its ``nosym`` leg alone ran past 35 minutes on
    one core. The bound is the relative one the field-direction version of
    this average reached against the whole zone (2.6e-5 axial, 9.0e-4 with no
    average, 0.665 with three scalars, ``test_spinor_response.py``), times four.
    """
    _, _, free = _magnetic("h4-cycloid-90-nosym", None)
    calculation, _, wedge = _magnetic("h4-cycloid-90", None)
    assert calculation.use_symmetry and calculation.nspin_mag == 4
    scale = float(np.abs(free.matrix).max())
    relative = float(np.abs(wedge.matrix - free.matrix).max()) / scale
    print(f"\nhelix wedge against the whole zone: {relative:.3e} of {scale:.4e}")
    assert relative < 1e-4
