"""Keywords in angstrom and eV must build the same system as the pw.x file they describe.

The keyword route writes a ``pw.x`` input and parses it, so what is checked here
is that translation: the cell, the positions, the cutoffs and the k-set built from
the keywords against the ones built from ``diamond.in``, element by element.
Neither side runs an SCF. On this commit the two SCF energies also agree to the
last digit (-22.855601708797778 Ry, bit-identical); that run is not repeated here.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.io.keywords import KEYWORDS, pw_input_text
from defuqumat.io.pwin import parse_pw_input, read_pw_input
from defuqumat.system.builder import build_system
from defuqumat.units import ANGSTROM_TO_BOHR, BOHR_TO_ANGSTROM, RY_TO_EV

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
    np.testing.assert_array_equal(np.asarray(ours.kpoints.coords), np.asarray(ref.kpoints.coords))
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
                         pseudos={"Fe": "Fe.UPF"}, ecut=340.0, kmesh=(2, 2, 2), smearing=0.27,
                         spin="collinear")
    assert "starting_magnetization" not in text
    build_system(parse_pw_input(text))


def test_without_moment_or_magnetization_nspin2_still_refuses():
    with pytest.raises(ValueError, match="no starting_magnetization"):
        _system(lattice="bcc", a=2.87, atoms=[("Fe", (0, 0, 0))],
                pseudos={"Fe": "Fe.UPF"}, ecut=340.0, kmesh=(2, 2, 2), smearing=0.27, spin="collinear")


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


def test_keywords_set_is_the_signature():
    """A name missing from KEYWORDS would be routed to Calculator and refused."""
    assert KEYWORDS == set(inspect.signature(pw_input_text).parameters)


def test_a_misspelt_keyword_names_both_sets():
    with pytest.raises(TypeError, match="input keywords are"):
        Calculator.from_keywords(**DIAMOND_KEYWORDS, kmsh=(2, 2, 2))


def test_constructor_keywords_pass_through():
    calc = Calculator.from_keywords(**DIAMOND_KEYWORDS, announce=False, conv_thr=1e-9,
                                    pseudo_dir=ROOT / "tests" / "data" / "pseudo")
    assert calc.announce is False and calc.defaults["conv_thr"] == 1e-9


def test_kmesh_is_required():
    keywords = {k: v for k, v in DIAMOND_KEYWORDS.items() if k != "kmesh"}
    with pytest.raises(TypeError, match="kmesh"):
        pw_input_text(**keywords)


@pytest.mark.parametrize("lattice, lengths, rows", [
    ("hexagonal", dict(a=2.46, c=6.7),
     [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [0, 0, 6.7 / 2.46]]),
    ("tetragonal", dict(a=3.0, c=5.0), [[1, 0, 0], [0, 1, 0], [0, 0, 5 / 3]]),
    ("orthorhombic", dict(a=3.0, b=4.0, c=5.0), [[1, 0, 0], [0, 4 / 3, 0], [0, 0, 5 / 3]]),
])
def test_non_cubic_lattices(lattice, lengths, rows):
    """The b and c of a named lattice reach the cell as b/a and c/a."""
    system = _system(lattice=lattice, atoms=[("C", (0, 0, 0))], pseudos={"C": "C.UPF"},
                     ecut=300.0, kmesh=(2, 2, 2), **lengths)
    np.testing.assert_allclose(np.asarray(system.cell.at),
                               np.array(rows) * lengths["a"] * ANGSTROM_TO_BOHR, atol=1e-12)


# --- every card, as the committed input that uses it ---------------------------
#
# Each case states a committed pw.x input as keywords and asks for the two built
# systems to be equal as whole trees, every array and every static field, which
# is what ``eqx.tree_equal`` compares. Lengths and energies are converted from
# the file's bohr and Ry exactly as a user would, so a unit slip in a keyword
# shows here as a False rather than as a plausible crystal.

QE = ROOT / "tests" / "data" / "qe"
B = BOHR_TO_ANGSTROM

CARD_CASES = {
    # HUBBARD, in eV in the card and so not converted by the keyword
    "alas-ldau.in": dict(
        lattice="fcc", a=10.575 * B,
        atoms=[("Al", (0, 0, 0)), ("As", (0.25 * 10.575 * B,) * 3)],
        coordinates="angstrom",
        pseudos={"Al": "Al.pz-vbc.UPF", "As": "As.pz-bhs.UPF"},
        masses={"Al": 26.98, "As": 74.92}, ecut=10 * RY_TO_EV, kmesh=(4, 4, 4),
        hubbard={"U": {"As-4p": 3.0}},
        qe={"control": {"prefix": "alas"}, "system": {"nosym": True, "noinv": True},
            "electrons": {"conv_thr": 1e-12, "mixing_beta": 0.7}},
    ),
    # B_field, a uniform Zeeman field in Ry in the file and in eV here
    "h-sheet-field.in": dict(
        lattice="tetragonal", a=4.0 * B, c=12.0 * B,
        atoms=[("H", (0, 0, 0.5))], pseudos={"H": "H.pz-vbc.UPF"}, masses={"H": 1.008},
        ecut=15 * RY_TO_EV, kmesh=(4, 4, 1), nbands=8,
        smearing=0.05 * RY_TO_EV, spin="collinear", field=(0, 0, 0.01 * RY_TO_EV),
        qe={"system": {"nosym": True, "starting_magnetization(1)": 0.0},
            "electrons": {"conv_thr": 1e-12}},
    ),
    # an explicit K_POINTS crystal list, and indexed variables through qe
    "h-chain-90deg-fixed.in": dict(
        lattice="tetragonal", a=12.0 * B, c=12.0 * 1.66666666666667 * B,
        atoms=[(f"H{i}", (0, 0, 0.25 * (i - 1))) for i in range(1, 5)],
        pseudos={f"H{i}": "H.pz-vbc.UPF" for i in range(1, 5)},
        masses={f"H{i}": 1.008 for i in range(1, 5)},
        ecut=25 * RY_TO_EV, nbands=8, spin="noncollinear",
        kmesh=[((0.0, 0.0, 0.5), 1), ((0.333333333333, 0.0, 0.5), 1),
               ((0.666666666667, 0.0, 0.5), 1)],
        qe={"system": {"occupations": "fixed", "nosym": True,
                       **{f"starting_magnetization({i})": 0.6 for i in range(1, 5)},
                       **{f"angle1({i})": 90.0 for i in range(1, 5)},
                       **{f"angle2({i})": 90.0 * (i - 1) for i in range(1, 5)}},
            "electrons": {"mixing_beta": 0.3, "conv_thr": 1e-11}},
    ),
    # if_pos, as fixed
    "graphene-bilayer-d2-relax.in": dict(
        lattice="hexagonal", a=4.6511 * B, c=4.6511 * 4.3 * B,
        atoms=[("C", (0, 0, 0.32)), ("C", (0.333333333333, 0.666666666667, 0.32)),
               ("C", (0, 0, 0.68)), ("C", (0.666666666667, 0.333333333333, 0.68))],
        pseudos={"C": "C.pbe-hgh.UPF"}, masses={"C": 12.011},
        ecut=40 * RY_TO_EV, kmesh=(12, 12, 1),
        smearing=0.02 * RY_TO_EV, smearing_type="mv", fixed={i: "xy" for i in range(4)},
        qe={"control": {"calculation": "relax", "tprnfor": True,
                        "forc_conv_thr": 1e-5, "etot_conv_thr": 1e-8},
            "system": {"vdw_corr": "grimme-d2"},
            "electrons": {"conv_thr": 1e-10, "mixing_beta": 0.5}, "ions": {}},
    ),
    # OCCUPATIONS, from_input, one row per spin channel
    "o-paw-spin.in": dict(
        lattice="fcc", a=25.0 * B, atoms=[("O", (0, 0, 0))], coordinates="angstrom",
        pseudos={"O": "O.pz-kjpaw.UPF"}, masses={"O": 1.0}, ecut=30 * RY_TO_EV,
        kmesh="gamma", spin="collinear", nbands=6,
        band_occupations=[[1, 1, 1, 1, 0, 0],
                          [1, 0.3333333333, 0.3333333333, 0.3333333333, 0, 0]],
        qe={"electrons": {"conv_thr": 1e-10}},
    ),
}


#: Where the file's ``celldm(3)`` is not exactly ``c/a`` of the two converted
#: lengths, one ulp of the ratio reaches ``at`` (3.6e-15 bohr for graphene).
#: Every other case is equal to the last bit.
ATOL = {"graphene-bilayer-d2-relax.in": 1e-13}


@pytest.mark.parametrize("name", sorted(CARD_CASES))
def test_every_card_builds_the_committed_input(name):
    import equinox as eqx

    ours = _system(**CARD_CASES[name])
    ref = build_system(read_pw_input(QE / name))
    assert eqx.tree_equal(ours, ref, atol=ATOL.get(name, 0.0)), name


def test_local_fields_and_tensor_moments_write_their_cards():
    """No committed input has these two; the card is the hand-written one."""
    import equinox as eqx

    common = dict(lattice="sc", a=3.0, atoms=[("Fe", (0, 0, 0), (0, 0, 2.0))],
                  pseudos={"Fe": "Fe.UPF"}, masses={"Fe": 55.845}, ecut=300.0,
                  kmesh=(2, 2, 2), smearing=0.1, spin="noncollinear")
    ours = _system(**common, fields=[(0.0, 0.0, 0.02 * RY_TO_EV)])
    ref = build_system(parse_pw_input(
        pw_input_text(**common) + "LOCAL_MAGNETIC_FIELDS\n  0.0 0.0 0.02\n"))
    assert eqx.tree_equal(ours, ref)
    assert ours.atomic_b_field == ((0.0, 0.0, 0.02),)

    text = pw_input_text(**common, tensor_moments=[("Fe-3d", 1, 1, 0, 0, -2.5)])
    assert "TENSOR_MOMENTS\n  Fe-3d  1 1 0 0  -2.5\n" in text


def test_a_card_passes_through_and_a_keyword_card_is_refused():
    text = pw_input_text(**DIAMOND_KEYWORDS,
                         cards={"ATOMIC_FORCES": ("", ["C 0 0 0", "C 0 0 0"])})
    assert "ATOMIC_FORCES\n  C 0 0 0\n  C 0 0 0" in text
    with pytest.raises(ValueError, match="written by the keyword hubbard"):
        pw_input_text(**DIAMOND_KEYWORDS, cards={"HUBBARD": "U C-2p 1.0"})


# --- named k-paths --------------------------------------------------------------

def _cell(lattice, **lengths):
    return _system(lattice=lattice, atoms=[("C", (0, 0, 0))], pseudos={"C": "C.UPF"},
                   ecut=300.0, kmesh=(2, 2, 2), **lengths).cell


@pytest.mark.parametrize("lattice, lengths, point, modulus", [
    ("fcc", {}, "X", 1.0), ("fcc", {}, "L", np.sqrt(3) / 2),
    ("fcc", {}, "K", 0.75 * np.sqrt(2)), ("fcc", {}, "W", np.sqrt(5) / 2),
    ("bcc", {}, "H", 1.0), ("bcc", {}, "N", np.sqrt(2) / 2),
    ("bcc", {}, "P", np.sqrt(3) / 2), ("sc", {}, "R", np.sqrt(3) / 2),
    ("hexagonal", {"c": 6.0}, "M", 1 / np.sqrt(3)),
    ("hexagonal", {"c": 6.0}, "K", 2 / 3), ("hexagonal", {"c": 6.0}, "A", 3.0 / 6.0 / 2),
    ("tetragonal", {"c": 6.0}, "A", np.sqrt(0.5 + (3.0 / 6.0) ** 2 / 4)),
    ("orthorhombic", {"b": 4.0, "c": 6.0}, "R",
     np.sqrt(0.25 + (3.0 / 4.0) ** 2 / 4 + (3.0 / 6.0) ** 2 / 4)),
])
def test_named_points_sit_where_they_should(lattice, lengths, point, modulus):
    """|k| in 2 pi / a: a mislabelled point or a wrong frame cannot pass."""
    from defuqumat.system.kpoints import labelled_path

    path = labelled_path(f"G-{point}", _cell(lattice, a=3.0, **lengths), npoints=4)
    k = np.asarray(path.coords)[-1]
    assert np.linalg.norm(k) == pytest.approx(modulus, abs=1e-12)


def test_a_named_path_has_the_corners_of_the_hand_built_one():
    """The silicon path of the bands page, L-G-X, typed by hand there."""
    from defuqumat.system.kpoints import labelled_path

    path = labelled_path("L-G-X", _cell("fcc", a=5.43), npoints=41)
    points = np.asarray(path.coords)
    corners = [0, int(np.argmin(np.linalg.norm(points, axis=1))), len(points) - 1]
    np.testing.assert_allclose(points[corners], [[0.5, 0.5, 0.5], [0, 0, 0], [1, 0, 0]],
                               atol=1e-14)
    assert path.nk == 41


def test_a_jump_adds_no_length_and_a_bad_name_is_refused():
    from defuqumat.system.kpoints import labelled_path

    cell = _cell("fcc", a=5.43)
    joined = labelled_path("X-U", cell, npoints=10)
    jumped = labelled_path("G-X|U-G", cell, npoints=20)
    assert float(jumped.path_length[-1]) == pytest.approx(
        1.0 + np.linalg.norm([1, 0.25, 0.25]), abs=1e-12)
    assert float(joined.path_length[-1]) == pytest.approx(np.sqrt(2) / 4, abs=1e-12)
    with pytest.raises(ValueError, match="no point 'Q'"):
        labelled_path("G-Q", cell)
    vectors = _system(lattice=[[3, 0, 0], [0, 3, 0], [0, 0, 3]], atoms=[("C", (0, 0, 0))],
                      pseudos={"C": "C.UPF"}, ecut=300.0, kmesh=(2, 2, 2)).cell
    with pytest.raises(ValueError, match="ibrav = 0"):
        labelled_path("G-X", vectors)
