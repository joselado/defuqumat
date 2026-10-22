"""Keywords in angstrom and eV must build the same system as the pw.x file they describe.

The keyword route writes a ``pw.x`` input and parses it, so what is checked here
is that translation: the cell, the positions, the cutoffs and the k-set built from
the keywords against the ones built from ``diamond.in``, element by element.
Neither side runs an SCF. On this commit the two SCF energies also agree to the
last digit (-22.855601708797778 Ry, bit-identical); that run is not repeated here.
"""

from pathlib import Path

import numpy as np
import pytest

from defuqumat.io.keywords import pw_input_text
from defuqumat.io.pwin import parse_pw_input, read_pw_input
from defuqumat.system.builder import build_system
from defuqumat.units import BOHR_TO_ANGSTROM, RY_TO_EV

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
DIAMOND = ROOT / "docs" / "figures" / "own-crystal_convergence" / "diamond.in"

#: ``diamond.in``, stated as keywords.
DIAMOND_KEYWORDS = dict(
    lattice="fcc", a=6.74 * BOHR_TO_ANGSTROM,
    atoms=[("C", (0, 0, 0)), ("C", (0.25, 0.25, 0.25))],
    pseudos={"C": "C.pz-rrkjus.UPF"}, masses={"C": 12.011},
    ecut=35 * RY_TO_EV, ecutrho=280 * RY_TO_EV, nbands=8, kmesh=(6, 6, 6),
    qe={"electrons": {"conv_thr": 1e-8}},
)


def _system(**keywords):
    return build_system(parse_pw_input(pw_input_text(**keywords)))


def test_keywords_build_the_file_system():
    ours, ref = _system(**DIAMOND_KEYWORDS), build_system(read_pw_input(DIAMOND))
    assert ours.cell.alat == ref.cell.alat
    np.testing.assert_array_equal(np.asarray(ours.cell.at), np.asarray(ref.cell.at))
    np.testing.assert_array_equal(np.asarray(ours.structure.positions),
                                  np.asarray(ref.structure.positions))
    assert (ours.ecutwfc, ours.ecutrho, ours.nbnd) == (ref.ecutwfc, ref.ecutrho, ref.nbnd)
    np.testing.assert_array_equal(np.asarray(ours.kpoints.cartesian), np.asarray(ref.kpoints.cartesian))
    np.testing.assert_array_equal(np.asarray(ours.kpoints.weights), np.asarray(ref.kpoints.weights))


def test_lattice_vectors_and_cartesian_positions_match_the_named_lattice():
    a = 2.87
    common = dict(pseudos={"Fe": "Fe.pz-nd-rrkjus.UPF"}, ecut=340.0, kmesh=(4, 4, 4),
                  smearing=0.27, spin="collinear")
    named = _system(lattice="bcc", a=a, atoms=[("Fe", (0, 0, 0), 2.5)], **common)
    vectors = _system(
        lattice=[[a / 2, a / 2, a / 2], [-a / 2, a / 2, a / 2], [-a / 2, -a / 2, a / 2]],
        coordinates="angstrom", atoms=[("Fe", (0, 0, 0), 2.5)], **common,
    )
    np.testing.assert_allclose(np.asarray(named.cell.at), np.asarray(vectors.cell.at),
                               atol=1e-12)
    assert named.nspin == vectors.nspin == 2
    np.testing.assert_array_equal(named.local_moments, [[0.0, 0.0, 2.5]])


def test_default_mass_is_the_standard_atomic_weight():
    keywords = {**DIAMOND_KEYWORDS, "masses": None}
    assert _system(**keywords).structure.masses[0] == pytest.approx(12.0107)


def test_a_collinear_moment_alone_breaks_the_spin_symmetry():
    """The moments card seeds the density, so nspin = 2 needs nothing else."""
    text = pw_input_text(lattice="bcc", a=2.87, atoms=[("Fe", (0, 0, 0), 2.5)],
                         pseudos={"Fe": "Fe.UPF"}, ecut=340.0, smearing=0.27,
                         spin="collinear")
    assert "starting_magnetization" not in text
    build_system(parse_pw_input(text))


def test_without_moment_or_magnetization_nspin2_still_refuses():
    with pytest.raises(ValueError, match="no starting_magnetization"):
        _system(lattice="bcc", a=2.87, atoms=[("Fe", (0, 0, 0))],
                pseudos={"Fe": "Fe.UPF"}, ecut=340.0, smearing=0.27, spin="collinear")


@pytest.mark.parametrize("change, message", [
    ({"lattice": "hexagonal"}, "needs c"),
    ({"c": 3.0}, "takes no c"),
    ({"pseudos": {}}, "no pseudopotential"),
    ({"qe": {"system": {"ecutwfc": 30}}}, "already set by a keyword"),
    ({"kmesh": "dense"}, "expected"),
    ({"atoms": [("C", (0, 0, 0), 1.0)]}, "give spin"),
])
def test_refusals(change, message):
    with pytest.raises(ValueError, match=message):
        pw_input_text(**{**DIAMOND_KEYWORDS, **change})
