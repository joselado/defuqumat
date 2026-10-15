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
