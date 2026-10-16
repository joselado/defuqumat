"""Which regimes the transverse spin susceptibility takes, each fired by name.

``notes/INCOMPATIBILITIES.md`` I58 to I62. No SCF here: every guard reads the
calculation alone, so each test builds one from the text of a committed input
with one line changed. The cell is ``h-fcc-magnon.in``, the one the magnon
identities are measured on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import Calculation
from defuqumat.system import build_system
from defuqumat.tddft.spinchi0 import require_a_transverse_regime

pytestmark = [pytest.mark.unit]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

HYDROGEN = (CASES / "h-fcc-magnon.in").read_text()
SYSTEM_LINE = "    nspin = 2\n"


def _calculation(text: str) -> Calculation:
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return Calculation(system, pseudos)


def _with(extra: str, card: str = "") -> str:
    assert SYSTEM_LINE in HYDROGEN
    return HYDROGEN.replace(SYSTEM_LINE, SYSTEM_LINE + extra) + card


# --- I58: an applied field ----------------------------------------------------


def test_a_uniform_field_is_accepted():
    """``B_field`` along the moment: the Larmor case, which the states carry."""
    calculation = _calculation(_with("    B_field(3) = 0.002\n"))
    assert calculation.magnetic_field is not None
    require_a_transverse_regime(calculation)


def test_a_per_atom_field_is_refused():
    """A ``LOCAL_MAGNETIC_FIELDS`` card is not a ``G = 0`` field.

    Larmor's theorem puts the uniform mode at one frequency only when the
    applied field is the same everywhere; a field confined to spheres is a
    perturbation with structure in ``G``, and no single frequency closes the
    identity.
    """
    calculation = _calculation(_with("", "LOCAL_MAGNETIC_FIELDS\n 0.0 0.0 0.002\n"))
    assert calculation.magnetic_field.atomic is not None
    with pytest.raises(NotImplementedError, match="per-atom field"):
        require_a_transverse_regime(calculation)


def test_a_constrained_moment_is_refused():
    """``fe-fsm.in``: a constrained moment is a different functional."""
    text = (CASES / "fe-fsm.in").read_text()
    calculation = _calculation(text)
    with pytest.raises(NotImplementedError, match="constrained moment"):
        require_a_transverse_regime(calculation)


# --- I59 and I62: the noncollinear family -------------------------------------


def test_a_spiral_is_refused_for_its_own_reason():
    """``h-chain-spiral.in``: the spiral branch used to sit behind ``noncolin``.

    A spiral is noncollinear by construction, so the generic refusal fired
    first and the spiral's own reason (two spheres, the rotating frame) was
    never reached.
    """
    calculation = _calculation((CASES / "h-chain-spiral.in").read_text())
    with pytest.raises(NotImplementedError, match="spin spiral"):
        require_a_transverse_regime(calculation)


def test_nosource_is_refused_for_its_own_reason():
    """``ni-noncol-111-nosource.in``: the source-free branch, reached.

    ``nosource`` needs ``noncolin`` (``refuse_source_free``), so before this
    branch was moved ahead of the noncollinear one it could not fire at all.
    """
    calculation = _calculation((CASES / "ni-noncol-111-nosource.in").read_text())
    assert calculation.source_free
    with pytest.raises(NotImplementedError, match="under nosource"):
        require_a_transverse_regime(calculation)


def test_a_collinear_spinor_run_is_accepted():
    """``h-fcc-magnon.in`` as a spinor with its moment along z (I59)."""
    calculation = _calculation(HYDROGEN.replace(
        "    nspin = 2\n", "    noncolin = .true.\n    angle1(1) = 0.0\n"))
    assert calculation.nspin_mag == 4
    require_a_transverse_regime(calculation)


def test_a_texture_is_refused_where_the_density_is_read():
    """The moment's direction is a property of the density, not of the run.

    A density with an ``m_x`` is refused by the kernel; the same density with
    ``m_x`` zeroed passes and is the collinear pair ``(n + m_z)/2, (n - m_z)/2``.
    """
    import numpy as np

    from defuqumat.tddft.spinkernel import collinear_channels

    rng = np.random.default_rng(0)
    charge = 1.0 + rng.random((4, 4, 4))
    mz = 0.3 * rng.random((4, 4, 4))
    collinear = np.stack([charge, 0 * mz, 0 * mz, mz])
    pair = np.asarray(collinear_channels(None, collinear))
    np.testing.assert_allclose(pair[0] - pair[1], mz, atol=1e-15)
    np.testing.assert_allclose(pair[0] + pair[1], charge, atol=1e-15)
    texture = collinear.copy()
    texture[1] = 0.1 * mz
    with pytest.raises(NotImplementedError, match="not along z"):
        collinear_channels(None, texture)


@pytest.mark.parametrize("interacting", [False, True])
def test_the_workflow_refuses_a_texture_before_the_bands(interacting):
    """A moment along x reaches no ``X_0``, interacting or bare, at ``q != 0``.

    The bare response builds no kernel, so the kernel's own texture check was
    never reached there, and a moment along x came back as the response of
    the up and down components (D22, ``review/incompat/magnon/ver2/
    texture.log``). The density here is placeholder data: the refusal comes
    before the fixed-density run, which is what keeps this test free of an SCF.
    """
    import numpy as np

    from defuqumat.workflows.magnons import run_spin_susceptibility

    calculation = _calculation(HYDROGEN.replace(
        "    nspin = 2\n", "    noncolin = .true.\n    angle1(1) = 90.0\n"))
    density = np.zeros((4, 4, 4, 4))
    density[0] = 1.0
    density[1] = 0.3
    with pytest.raises(NotImplementedError, match="not along z"):
        run_spin_susceptibility(
            calculation.system, calculation.pseudos, density, (0.0, 0.0, 0.25),
            np.linspace(0.005, 0.04, 4), interacting=interacting,
            calculation=calculation)


def test_spin_orbit_coupling_is_refused():
    """``ni-soc-nosym.in``: a fully-relativistic spinor run."""
    calculation = _calculation((CASES / "ni-soc-nosym.in").read_text())
    with pytest.raises(NotImplementedError, match="spin-orbit coupling"):
        require_a_transverse_regime(calculation)


def test_an_ultrasoft_spinor_run_is_refused():
    """``ni-noncol-111-nosym.in``: the spinor augmentation is not carried."""
    calculation = _calculation((CASES / "ni-noncol-111-nosym.in").read_text())
    with pytest.raises(NotImplementedError, match="ultrasoft dataset"):
        require_a_transverse_regime(calculation)


# --- I61: a Hubbard U ---------------------------------------------------------

HUBBARD = "HUBBARD {atomic}\nU H-1s 2.0\n"


def test_a_simplified_hubbard_u_is_accepted_with_its_channel():
    """``h-fcc-magnon-ldau.in``: one s shell, so one Hubbard channel."""
    from defuqumat.tddft.spinchi0 import hubbard_pairs

    calculation = _calculation((CASES / "h-fcc-magnon-ldau.in").read_text())
    require_a_transverse_regime(calculation)
    triples, rows, columns = hubbard_pairs(calculation)
    assert triples == ((0, 0, 0),)
    assert list(rows) == [0] and list(columns) == [0]


def test_a_d_shell_has_both_orderings_of_each_pair():
    """A transverse ``ns^{+-}`` at finite ``q`` is not symmetric, so the
    channel carries ``(a, c)`` and ``(c, a)`` apart: 25 entries for a d shell."""
    from defuqumat.tddft.spinchi0 import hubbard_pairs

    calculation = _calculation((CASES / "ni-ldau-nospin.in").read_text())
    triples, _, _ = hubbard_pairs(calculation)
    assert len(triples) == 25
    assert (0, 1, 3) in triples and (0, 3, 1) in triples


def test_hubbard_beta_is_refused():
    calculation = _calculation(HYDROGEN + HUBBARD + "BETA H-1s 0.5\n")
    with pytest.raises(NotImplementedError, match="Hubbard_beta"):
        require_a_transverse_regime(calculation)


def test_the_full_functional_is_refused():
    """``feo-kind1-J.in``: a Hubbard J selects ``lda_plus_u_kind = 1``."""
    calculation = _calculation((CASES / "feo-kind1-J.in").read_text())
    assert calculation.hubbard_coefficients["kind"] == 1
    with pytest.raises(NotImplementedError, match="full DFT\\+U"):
        require_a_transverse_regime(calculation)


# --- I60: PAW -----------------------------------------------------------------


def test_paw_is_refused_for_its_one_centre_kernel():
    """``o2-paw-texture.in``: the refusal names the projector channel it lacks.

    The construction-only twin of the slow file's
    ``test_ultrasoft_magnon.py::test_paw_is_still_refused``.
    """
    calculation = _calculation((CASES / "o2-paw-texture.in").read_text())
    assert calculation.is_paw
    with pytest.raises(NotImplementedError, match="projector channel"):
        require_a_transverse_regime(calculation)
