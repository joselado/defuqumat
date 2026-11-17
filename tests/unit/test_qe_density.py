"""Continuing a converged ``pw.x`` run from its save directory.

The fixtures under ``tests/data/qe/save/`` are the ``charge-density.dat``,
``paw.txt`` and ``data-file-schema.xml`` of three ``pw.x`` 7.5 SCFs at
``conv_thr = 1e-10``, with the ``pw.in`` that made them: Si2 norm-conserving
under ``K_POINTS gamma`` (the half-sphere file), Si2 PAW on a 2x2x2 mesh, and
an O2 PAW triplet at ``nspin = 2`` (the ``(n, m_z)`` layout and the
``(up, down)`` ``becsum``).

What makes the check discriminating is that both codes hold the same pseudo
valence density: a wrong grid map, spin split, ``becsum`` packing or
projector order would leave the seeded run iterating, where the right one
converges in one or two iterations at the ``pw.x`` energy.

P180 added six more ``pw.x`` 7.5 save directories at ``conv_thr = 1e-10``:
Si2 in a triclinic cell (``ggen``'s G order off a cubic lattice), Pt2 PAW with
``lspinorb`` and no magnetization (a four-component ``paw.txt`` with three zero
blocks), an O2 PAW moment in the plane (the noncollinear ``becsum``), Ni with a
U at ``nspin = 2`` (a real ``occup.txt``) and as a spinor tilted off ``z``
(a complex ``ns_nc`` with spin-flip blocks), and Si2 under TPSS at
``nspin = 2`` (the ``(up, down)`` ``ekin-density``). The writer is checked
against ``pw.x``'s own files: the same records byte for byte, the same G order,
and the same numbers to the agreement of the two converged states.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from functools import lru_cache
import shutil
import xml.etree.ElementTree as ET

from defuqumat import Calculator
from defuqumat.io.qe_density import QESave, read_rhog, write_qe_save, write_rhog
from defuqumat.io.qe_gvectors import qe_gvector_order

SAVE = Path(__file__).resolve().parents[1] / "data" / "qe" / "save"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: ``!    total energy`` of each ``pw.x`` run, Ry.
QE_ENERGY = {
    "si2-nc-gamma": -14.56060838,
    "si2-paw": -89.09136627,
    "o2-paw-lsda": -80.72096086,
    "si2-tri": -15.48900572,
    "pt2-paw-soc": -1506.6572253,
    "o2-paw-noncol": -80.7209707,
    "ni-ldau-lsda": -85.64074932,
    "ni-ldau-tilt": -85.64074932,
}

#: Every committed save directory, for the checks that run no SCF.
ALL_SAVES = sorted(path.name for path in SAVE.iterdir() if path.is_dir())


def _calculator(name: str, text: str | None = None) -> Calculator:
    text = (SAVE / name / "pw.in").read_text() if text is None else text
    return Calculator.from_text(text, pseudo_dir=PSEUDO)


@pytest.mark.parametrize("name, counts", [
    ("si2-nc-gamma", [8.0]),
    ("si2-paw", [8.0]),
    ("o2-paw-lsda", [12.0, 2.0]),
])
def test_g0_is_the_electron_count_and_the_moment(name, counts):
    """``rho(G = 0) omega`` is N, and for LSDA the second component is M, not N_down."""
    save = QESave.read(SAVE / name)
    np.testing.assert_allclose(save.rho_g[:, 0].real * save.omega, counts, atol=1e-8)
    assert save.gamma_only == (name == "si2-nc-gamma")


def test_lsda_density_is_split_into_up_and_down():
    calc = _calculator("o2-paw-lsda")
    seed = calc.get_qe_seed(SAVE / "o2-paw-lsda")
    rho = np.asarray(seed["starting_density"])
    omega = QESave.read(SAVE / "o2-paw-lsda").omega
    channels = rho.reshape(2, -1).sum(axis=1) * omega / rho[0].size
    np.testing.assert_allclose(channels, [7.0, 5.0], atol=1e-8)
    becsum = [np.asarray(b) for b in seed["starting_becsum"] if b is not None]
    assert becsum[0].shape == (2, 2, 8, 8)
    np.testing.assert_array_equal(becsum[0], np.swapaxes(becsum[0], -1, -2))


@pytest.mark.parametrize("name", ["si2-nc-gamma", "si2-paw"])
def test_seeded_run_converges_at_once_to_the_pwx_energy(name):
    calc = _calculator(name)
    result = calc.get_scf(**calc.get_qe_seed(SAVE / name))
    assert result.converged
    # The first iteration lands at dr2 = 2.5e-11 (NC) and 4.0e-11 (PAW) against
    # conv_thr = 1e-10, a factor of 3 under it, so a second is allowed; a wrong
    # map took 9 against 7 from scratch.
    assert result.iterations <= 2
    assert abs(result.total_energy - QE_ENERGY[name]) < 1e-7


def test_paw_becsum_is_the_converged_one():
    """``paw.txt``'s packing (upper triangle, doubled off-diagonals) against our own."""
    calc = _calculator("si2-paw")
    seed = calc.get_qe_seed(SAVE / "si2-paw")
    own = calc.get_scf()
    for loaded, converged in zip(seed["starting_becsum"], own.becsum):
        if loaded is None:
            continue
        loaded, converged = np.asarray(loaded), np.asarray(converged)
        off = ~np.eye(loaded.shape[-1], dtype=bool)
        assert np.abs(converged[..., off]).max() > 1e-2  # the off-diagonals are tested
        np.testing.assert_allclose(loaded, converged, atol=2e-5)


@pytest.mark.parametrize("edit, error, words", [
    (("ecutrho=160.0", "ecutrho=120.0"), NotImplementedError, "ecutrho"),
    (("celldm(1)=10.20", "celldm(1)=10.30"), ValueError, "same cell"),
    (("ecutrho=160.0",
      "ecutrho=160.0, nspin=2, tot_magnetization=0, starting_magnetization(1)=0.2"),
     ValueError, "components"),
])
def test_a_save_directory_of_another_calculation_is_refused(edit, error, words):
    text = (SAVE / "si2-paw" / "pw.in").read_text()
    assert edit[0] in text
    calc = _calculator("si2-paw", text.replace(*edit))
    with pytest.raises(error, match=words):
        calc.get_qe_seed(SAVE / "si2-paw")


@pytest.mark.slow
def test_lsda_paw_seeded_run_converges_at_once():
    calc = _calculator("o2-paw-lsda")
    result = calc.get_scf(**calc.get_qe_seed(SAVE / "o2-paw-lsda"))
    assert result.converged
    assert result.iterations <= 2
    assert abs(result.total_energy - QE_ENERGY["o2-paw-lsda"]) < 1e-7


# ---------------------------------------------------------------- P180

def _alat_and_cell(name):
    out = ET.parse(SAVE / name / "data-file-schema.xml").getroot().find("output")
    structure = out.find("atomic_structure")
    at = np.stack([np.array(structure.find(f"cell/a{i}").text.split(), float)
                   for i in (1, 2, 3)])
    return float(structure.get("alat")), at, float(out.find("basis_set/ecutrho").text) * 2


@pytest.mark.parametrize("name", ALL_SAVES)
def test_the_g_order_is_pwx_own(name):
    """``ggen`` + ``hpsort_eps`` give the Miller record of ``pw.x``'s file exactly.

    ``read_rhog`` scatters a file's coefficients in their stored order, so a
    file written in any other order is read scrambled without a message. The
    set covers a cubic cell, a molecule in a box, the ``K_POINTS gamma`` half
    sphere and a triclinic cell.
    """
    gamma_only, _, miller, _ = read_rhog(SAVE / name / "charge-density.dat")
    alat, at, ecutrho = _alat_and_cell(name)
    np.testing.assert_array_equal(qe_gvector_order(at, alat, ecutrho, gamma_only), miller)


@pytest.mark.parametrize("name", ALL_SAVES)
def test_write_rhog_reproduces_the_pwx_file_byte_for_byte(name, tmp_path):
    source = SAVE / name / "charge-density.dat"
    gamma_only, b, miller, rho_g = read_rhog(source)
    write_rhog(tmp_path / "charge-density.dat", b, miller, rho_g, gamma_only)
    assert (tmp_path / "charge-density.dat").read_bytes() == source.read_bytes()


def test_a_spinor_paw_txt_is_read_for_every_atom():
    """``paw.txt`` of a nonmagnetic spin-orbit run has four components.

    ``scf_mod.f90`` allocates ``bec`` with ``lsda_mod``'s ``nspin = 4``, so the
    file holds the charge component and three zero blocks. Reshaped with the
    run's ``nspin_mag = 1``, the second atom's rows were the first zero block:
    loaded as zeros where defuqumat's own converged matrix peaks at 1.37. The
    two Pt atoms are a lattice translation apart, so their matrices are equal
    up to convergence: 8.9e-7 apart in ``pw.x``'s file, which treats the cell
    as a supercell and so never maps one atom onto the other.
    """
    save = QESave.read(SAVE / "pt2-paw-soc")
    calc = _calculator("pt2-paw-soc")
    assert calc.calculation.nspin_mag == 1
    nh = calc.calculation.pseudos[0].nh
    assert save.becsum.size == 4 * 2 * nh * (nh + 1) // 2
    becsum = np.asarray(next(b for b in calc.get_qe_seed(SAVE / "pt2-paw-soc")["starting_becsum"]
                             if b is not None))
    assert becsum.shape == (1, 2, nh, nh)
    assert np.abs(becsum[0, 1]).max() > 1.0
    np.testing.assert_allclose(becsum[0, 0], becsum[0, 1], atol=5e-6)


@lru_cache(maxsize=2)
def _seeded(name):
    calc = _calculator(name)
    seed = calc.get_qe_seed(SAVE / name)
    return calc, seed, calc.get_scf(**seed)


@pytest.mark.parametrize("name", [
    "si2-tri",
    "ni-ldau-tilt",
    pytest.param("o2-paw-noncol", marks=pytest.mark.slow),
    pytest.param("ni-ldau-lsda", marks=pytest.mark.slow),
    pytest.param("pt2-paw-soc", marks=pytest.mark.slow),
])
def test_p180_seeded_run_converges_at_once_to_the_pwx_energy(name):
    """Measured: 3, 1, 1, 1 and 3 iterations, at 1.3e-10 to 2e-11 Ry first."""
    calc, seed, result = _seeded(name)
    assert result.converged
    assert result.iterations <= 3
    assert abs(result.total_energy - QE_ENERGY[name]) < 1e-7


def test_the_spinor_occupation_matrix_is_read_in_pw_x_convention():
    """``ns_nc`` with spin-flip blocks: the layout, not its conjugate.

    The Ni moment is tilted (``angle1 = 60``, ``angle2 = 30``), so the
    up-down blocks are complex and 0.045 large. One iteration from the seed
    leaves ``ns`` where it was to 4e-6; read conjugated (equivalently, with the
    two spin-flip blocks swapped) it is 4.5e-2 off.
    """
    calc, seed, result = _seeded("ni-ldau-tilt")
    loaded, converged = np.asarray(seed["starting_ns"]), np.asarray(result.ns)
    assert loaded.shape == (4, 1, 5, 5) and np.iscomplexobj(loaded)
    assert np.abs(converged[1]).max() > 0.03
    assert np.abs(loaded - converged).max() < 2e-5
    assert np.abs(loaded.conj() - converged).max() > 1e-2
    assert np.abs(loaded[[0, 2, 1, 3]] - converged).max() > 1e-2


def test_the_written_occupations_are_pwx_own(tmp_path):
    """``occup.txt`` written from defuqumat's state against ``pw.x``'s file."""
    calc, seed, result = _seeded("ni-ldau-tilt")
    write_qe_save(tmp_path, result, calc.calculation)
    ours = QESave.read(tmp_path).occupations
    theirs = QESave.read(SAVE / "ni-ldau-tilt").occupations
    assert ours.shape == theirs.shape and np.iscomplexobj(ours)
    np.testing.assert_allclose(ours, theirs, atol=2e-5)
    # Read back, it is the state it was written from, to the last bit.
    np.testing.assert_array_equal(np.asarray(calc.get_qe_seed(tmp_path)["starting_ns"]),
                                  np.asarray(result.ns))


@pytest.fixture(scope="module")
def si2_paw_written(tmp_path_factory):
    calc = _calculator("si2-paw")
    result = calc.get_scf()
    directory = tmp_path_factory.mktemp("si2-paw") / "pwscf.save"
    calc.write_qe_save(directory, result)
    return calc, result, directory


def test_the_written_density_is_pwx_own(si2_paw_written):
    """Same records, same G order, the coefficients to 6e-9 on a peak of 0.03."""
    calc, result, directory = si2_paw_written
    ours = read_rhog(directory / "charge-density.dat")
    theirs = read_rhog(SAVE / "si2-paw" / "charge-density.dat")
    assert ours[0] == theirs[0]
    np.testing.assert_allclose(ours[1], theirs[1], rtol=1e-14)
    np.testing.assert_array_equal(ours[2], theirs[2])
    np.testing.assert_allclose(ours[3], theirs[3], atol=1e-7)
    np.testing.assert_allclose(QESave.read(directory).becsum,
                               QESave.read(SAVE / "si2-paw").becsum, atol=2e-5)


def test_the_written_state_is_read_back_as_itself(si2_paw_written):
    calc, result, directory = si2_paw_written
    seed = calc.get_qe_seed(directory)
    rho = np.asarray(seed["starting_density"])
    # Only what lies outside the G sphere is lost on the way.
    assert np.abs(rho - np.asarray(result.density)).max() < 1e-6 * np.abs(rho).max() + 1e-12
    for loaded, own in zip(seed["starting_becsum"], result.becsum):
        if loaded is not None:
            np.testing.assert_allclose(np.asarray(loaded), np.asarray(own), atol=1e-14)


def test_the_kinetic_energy_density_is_up_and_down():
    """``ekin-density`` at ``nspin = 2`` is ``(up, down)``, not ``(n, m)``.

    The TPSS run is nonmagnetic at ``nspin = 2``, so an ``(up, down)`` file has
    two equal components and an ``(n, m)`` file a zero second one. Each
    integrates to 3.3797 Ry, the kinetic energy per spin; defuqumat's TB09 state
    of the same cell has 3.4381, a different functional, where a factor of two
    in the units would read 1.69 or 6.76.
    """
    save = QESave.read(SAVE / "si2-tpss-lsda")
    np.testing.assert_allclose(save.rho_g[:, 0].real * save.omega, [8.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(save.ekin_g[:, 0].real * save.omega, [3.379683, 3.379676],
                               atol=1e-5)
    text = (SAVE / "si2-tpss-lsda" / "pw.in").read_text().replace("TPSS", "tb09")
    tau = np.asarray(_calculator("si2-tpss-lsda", text).get_qe_seed(
        SAVE / "si2-tpss-lsda")["starting_tau"])
    assert tau.shape[0] == 2
    np.testing.assert_allclose(tau[0], tau[1], atol=1e-6)


@pytest.mark.parametrize("edit, error, words", [
    (("<U_projection_type>atomic", "<U_projection_type>ortho-atomic"), ValueError,
     "projects on"),
    (("<lda_plus_u_kind>0", "<lda_plus_u_kind>1"), ValueError, "lda_plus_u_kind"),
    (("<U_projection_type>", "<Hubbard_back>x</Hubbard_back><U_projection_type>"),
     NotImplementedError, "background"),
    (("<U_projection_type>", "<Hubbard_V>x</Hubbard_V><U_projection_type>"),
     NotImplementedError, "Hubbard_V"),
])
def test_an_occupation_matrix_of_another_functional_is_refused(edit, error, words, tmp_path):
    shutil.copytree(SAVE / "ni-ldau-lsda", tmp_path / "save")
    xml = tmp_path / "save" / "data-file-schema.xml"
    head, output = xml.read_text().split("<output>")
    # The <input> section repeats <dftU>; the reader reads <output>'s.
    assert edit[0] in output
    xml.write_text(head + "<output>" + output.replace(*edit, 1))
    with pytest.raises(error, match=words):
        _calculator("ni-ldau-lsda").get_qe_seed(tmp_path / "save")


def test_a_hubbard_state_without_a_hubbard_term_is_refused():
    text = (SAVE / "ni-ldau-lsda" / "pw.in").read_text()
    text = text[: text.index("HUBBARD")]
    with pytest.raises(ValueError, match="no Hubbard term"):
        _calculator("ni-ldau-lsda", text).get_qe_seed(SAVE / "ni-ldau-lsda")


def test_a_meta_gga_without_its_kinetic_energy_density_is_refused():
    text = (SAVE / "si2-nc-gamma" / "pw.in").read_text().replace(
        "ecutwfc=16.0", "ecutwfc=16.0, input_dft='tb09'")
    assert "tb09" in text
    with pytest.raises(ValueError, match="ekin-density"):
        _calculator("si2-nc-gamma", text).get_qe_seed(SAVE / "si2-nc-gamma")


def test_the_hdf5_form_is_not_written(tmp_path):
    with pytest.raises(NotImplementedError, match="HDF5"):
        write_rhog(tmp_path / "charge-density.hdf5", np.eye(3), np.zeros((1, 3)),
                   np.zeros((1, 1)), False)
