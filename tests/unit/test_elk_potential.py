"""Elk's Kohn-Sham potential from its density: ``potks`` against Elk's own file.

Each committed Elk run directory holds the density and the potentials Elk
built from it. ``vclmt``/``vclir`` and ``vxcmt``/``vxcir`` (and ``bxcmt``/
``bxcir``) are functions of the density in the same file, so the transcription
must give them back to round-off; ``vsmt`` and ``bsmt`` are written after
Elk's potential mixing and agree only to the run's convergence. The fixtures
cover one atom unpolarized (``h_sc``, PW92), spin-polarized (``h_sc_spin``,
PW92 LSDA), and two species with different radial meshes (``sic_zb``, Elk's own PBE).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from defuqumat.io.elk import ElkState
from defuqumat.io.elk_potential import (coulomb_potential, crystal_symmetrizer,
                                        exchange_correlation_potential,
                                        kohn_sham_potential, radial_weights,
                                        read_elk_parameters)

ELK = Path(__file__).resolve().parents[1] / "data" / "elk"


def _read(name):
    return ElkState.read(ELK / name), read_elk_parameters(ELK / name)


def test_info_out_is_read():
    state, params = _read("sic_zb")
    assert params.spzn == (-6.0, -14.0)
    assert params.nrmti == (201, 273)
    assert (params.lmaxi, params.lmaxo, params.lradstp) == (1, 6, 4)
    assert params.xctype == (20, 0, 0) and params.point_nucleus


def test_the_radial_weights_integrate_a_cubic_exactly():
    r = np.geomspace(1e-5, 2.0, 300)
    wp = radial_weights(r)
    from defuqumat.io.elk_potential import _cumulative
    g = _cumulative(wp, (1.0 + r - r ** 2 + 0.5 * r ** 3).astype(complex))
    exact = r + r ** 2 / 2 - r ** 3 / 3 + r ** 4 / 8
    np.testing.assert_allclose(g.real, exact - exact[0], rtol=1e-11, atol=1e-14)


@pytest.mark.parametrize("name", ["h_sc", "h_sc_spin", "sic_zb"])
def test_the_coulomb_potential_is_elk_own(name):
    """Weinert's method, including the nucleus: 1e-12 relative on a 1e7 scale."""
    state, params = _read(name)
    vclmt, vclir = coulomb_potential(state, params)
    for ias in range(state.rhomt.shape[2]):
        s = int(np.repeat(np.arange(len(state.nrmt)), state.geometry.natoms)[ias])
        nr = state.nrmt[s]
        ref = state.vclmt[:, :nr, ias] / 2.0
        assert np.max(np.abs(vclmt[0, :nr, ias] - ref[0]) / np.abs(ref[0])) < 1e-11
        assert np.abs(vclmt[1:, :nr, ias] - ref[1:]).max() < 1e-13
    np.testing.assert_allclose(vclir, state.vclir / 2.0, atol=1e-12)


@pytest.mark.parametrize("name", ["h_sc", "h_sc_spin"])
def test_the_exchange_correlation_potential_is_elk_own(name):
    """``potxc`` with ``symrfmt``, which is not the identity on a symmetric density.

    Without the group average the muffin-tin potential carries 1e-7 (``h_sc_spin``)
    to 4e-4 (``h_sc``) in the channels the site symmetry forbids, which Elk
    holds at exactly zero; with it, round-off.
    """
    state, params = _read(name)
    moments = None
    if state.magmt is not None:
        r = state.rsp[0][:state.nrmt[0]]
        moments = [float(np.sqrt(4 * np.pi) * np.trapezoid(
            state.magmt[0, :state.nrmt[0], 0, 0] * r ** 2, r))]
    symmetry, nsym = crystal_symmetrizer(state, params.lmaxo, moments=moments)
    assert nsym == 48
    vxcmt, vxcir, bxcmt, bxcir = exchange_correlation_potential(state, params, symmetry)
    nr = state.nrmt[0]
    np.testing.assert_allclose(vxcmt[:, :nr], state.vxcmt[:, :nr] / 2.0, atol=1e-12)
    np.testing.assert_allclose(vxcir, state.vxcir / 2.0, atol=1e-14)
    raw, _, _, _ = exchange_correlation_potential(state, params, None)
    assert np.abs(raw[1:, :nr] - state.vxcmt[1:, :nr] / 2.0).max() > 5e-8
    if state.magmt is not None:
        np.testing.assert_allclose(bxcmt[:, :nr], state.bxcmt[:, :nr] / 2.0, atol=1e-13)
        np.testing.assert_allclose(bxcir.reshape(state.bxcir.shape), state.bxcir / 2.0,
                                   atol=1e-14)


def test_b_s_is_on_the_coarse_mesh_and_the_cover_points():
    """``bsmt`` agrees to the convergence of the run, 4e-8 on a scale of 0.07."""
    state, params = _read("h_sc_spin")
    r = state.rsp[0][:state.nrmt[0]]
    moments = [float(np.sqrt(4 * np.pi) * np.trapezoid(state.magmt[0, :state.nrmt[0], 0, 0] * r ** 2, r))]
    symmetry, _ = crystal_symmetrizer(state, params.lmaxo, moments=moments)
    potential = kohn_sham_potential(state, params, symmetry)
    nrc = state.nrcmt[0]
    np.testing.assert_allclose(potential["bsmt"][:, :nrc], state.bsmt[:, :nrc] / 2.0, atol=1e-6)
    assert np.abs(state.bsmt[:, :nrc]).max() / 2.0 > 0.05


def test_the_pbe_potential_is_elk_own():
    """Elk's own PBE (``xcgrad = 1``), two species, through ``ggamt_1``/``ggair_1``.

    The interstitial to round-off; the muffin tins to 1e-11 in every ``l > 0``
    channel and 2.6e-10 relative in ``l = 0``, the largest at the nuclear
    radius, where the derivative of ``|grad rho|`` meets the cusp.
    """
    state, params = _read("sic_zb")
    symmetry, nsym = crystal_symmetrizer(state, params.lmaxo)
    assert nsym == 24
    vxcmt, vxcir, _, _ = exchange_correlation_potential(state, params, symmetry)
    np.testing.assert_allclose(vxcir, state.vxcir / 2.0, atol=1e-14)
    for ias in range(2):
        nr = state.nrmt[ias]
        ref = state.vxcmt[:, :nr, ias] / 2.0
        assert np.max(np.abs(vxcmt[0, :nr, ias] - ref[0]) / np.abs(ref[0])) < 1e-9
        assert np.abs(vxcmt[1:, :nr, ias] - ref[1:]).max() < 1e-10


def test_a_libxc_functional_is_refused():
    import dataclasses

    state, params = _read("sic_zb")
    with pytest.raises(NotImplementedError, match="libxc"):
        exchange_correlation_potential(state, dataclasses.replace(params, xctype=(100, 101, 130)))


@pytest.mark.parametrize("name, nsym", [("h_sc", 48), ("sic_zb", 24), ("si2_template", 48)])
def test_the_space_group_is_elk_own(name, nsym):
    """``findsymcrys``: lattice symmetries with any fractional translation.

    The count is the one Elk prints in each run's ``INFO.OUT``; QE's search
    keeps a translation only when it is commensurate with the FFT grid, and
    for O2 off the origin in a box gives 8 where Elk has 16.
    """
    import re

    from defuqumat.io.elk_potential import elk_space_group

    state, _ = _read(name)
    rotations, maps = elk_space_group(state.geometry)
    info = (ELK / name / "INFO.OUT").read_text()
    assert len(rotations) == int(re.search(r"Number of crystal symmetries\s*:\s*(\d+)", info).group(1)) == nsym
    for rot in rotations:
        np.testing.assert_allclose(rot @ rot.T, np.eye(3), atol=1e-10)


def test_a_finite_nucleus_and_a_noncollinear_gga_are_refused():
    import dataclasses

    state, params = _read("h_sc_spin")
    with pytest.raises(NotImplementedError, match="finite nucleus"):
        coulomb_potential(state, dataclasses.replace(params, point_nucleus=False))
    vector = dataclasses.replace(state, ndmag=3,
                                 magmt=np.repeat(state.magmt, 3, axis=-1),
                                 magir=np.repeat(state.magir, 3, axis=-1))
    with pytest.raises(NotImplementedError, match="dncgga"):
        exchange_correlation_potential(vector, dataclasses.replace(params, xctype=(20, 0, 0)))


# ---------------------------------------------------------------- AUDIT-2026-11-20
# The counts below are the "Number of crystal symmetries" that Elk 11.0.2 (the
# elkpy build on the workstation) printed for each cell on 2026-11-20, from
# one-loop runs of the inputs written out in each test.

def _geometry(avec, positions, natoms):
    from types import SimpleNamespace

    geometry = SimpleNamespace(avec=np.asarray(avec, dtype=float),
                               positions=np.asarray(positions, dtype=float),
                               natoms=tuple(natoms))
    return SimpleNamespace(geometry=geometry)


def test_a_cell_that_is_not_primitive_keeps_every_translation():
    """Si in the 8-atom cubic cell: 48 rotations x 4 pure translations = 192 in Elk.

    ``findsymcrys`` calls ``findsym`` once per candidate translation, so a
    rotation is in the group once for each translation that works. Keeping one
    per rotation gave 48, a set of coset representatives and not a group.
    """
    from defuqumat.io.elk_potential import elk_space_group

    fcc = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
    positions = fcc + [[x + .25 for x in p] for p in fcc]
    state = _geometry(10.2 * np.eye(3), positions, (8,))
    rotations, maps = elk_space_group(state.geometry)
    assert len(rotations) == 192
    pure = [m for r, m in zip(rotations, maps) if np.allclose(r, np.eye(3))]
    assert len(pure) == 4 and len({tuple(m) for m in pure}) == 4


MONOCLINIC = [[6.0, 0.0, 0.0], [1.3, 6.7, 0.0], [0.0, 0.0, 7.3]]
PAIR = [[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]


@pytest.mark.parametrize("moments, nsym", [([0.1, -0.1], 1), ([0.1, 0.1], 2)])
def test_the_spin_rotation_is_a_proper_lattice_rotation(moments, nsym):
    """Two H related by inversion on a monoclinic lattice, ``bfcmt`` along the unique axis.

    The lattice symmetries are E, I, C2z and the mirror; their proper parts
    ``det(S) S`` are E and C2z, and neither reverses z. So Elk drops the
    inversion of the antiferromagnet (1 operation) and keeps it for the
    ferromagnet (2). A reversal -I among the candidates kept it for both.
    """
    _, n = crystal_symmetrizer(_geometry(MONOCLINIC, PAIR, (2,)), 2, moments=moments)
    assert n == nsym


def test_two_equal_moments_rounded_apart_still_match():
    """O2: 0.1235 +- 1e-8 on the two atoms round to 0.124 and 0.123.

    Elk keeps 16 operations for the ferromagnet. With a tolerance of 1e-3 of
    the largest moment, 1.24e-4, the operations that swap the atoms were lost.
    """
    state, params = _read("o2_template_collinear")
    _, n = crystal_symmetrizer(state, params.lmaxo, moments=[0.124, 0.123])
    assert n == 16


def test_a_small_antiparallel_pair_keeps_its_staggered_field():
    """O2 with moments +-0.01: the swap must carry a spin flip, not the identity."""
    state, params = _read("o2_template_collinear")
    symmetry, n = crystal_symmetrizer(state, params.lmaxo, moments=[0.01, -0.01])
    assert n == 16
    field = np.zeros(state.bxcmt.shape)
    field[0, :, 0, 0], field[0, :, 1, 0] = 1.0, -1.0
    np.testing.assert_allclose(symmetry(field, magnetic=True)[0, :, :, 0],
                               field[0, :, :, 0], atol=1e-12)


@pytest.mark.parametrize("product", [2.5, 2.5 - 1e-12])
def test_the_pseudocharge_constant_rounds_half_away_from_zero(product):
    """init0's ``nint(0.25 gmaxvr rmt)`` is 3 at 2.5, where ``np.rint`` gives 2.

    ``rsp(nrmt)``, the radius this module has, can sit an ulp from Elk's ``rmt``
    on either side (1.6750000499999973 against 1.67500005 in the O2 template),
    so a product Elk rounds to 3 can also arrive here just below 2.5.
    """
    import dataclasses

    state, params = _read("o2_template_collinear")
    rmt = state.rsp[0][state.nrmt[0] - 1]
    params = dataclasses.replace(params, gmaxvr=4.0 * product / rmt, npsd=3)
    coulomb_potential(state, params)


def test_a_pseudocharge_constant_that_is_not_init0_is_refused():
    import dataclasses

    state, params = _read("sic_zb")
    with pytest.raises(ValueError, match="pseudocharge"):
        coulomb_potential(state, dataclasses.replace(params, npsd=params.npsd + 1))


def _template_copy(tmp_path, name, elk_in="", info=None):
    import shutil

    directory = tmp_path / name
    shutil.copytree(ELK / name, directory)
    with open(directory / "elk.in", "a") as handle:
        handle.write(elk_in)
    if info is not None:
        text = (directory / "INFO.OUT").read_text()
        (directory / "INFO.OUT").write_text(info(text))
    return directory


def test_gradient_smoothing_is_refused_under_a_gga(tmp_path):
    """``stable`` sets ``msmgmt = 1``, and ``rfmtsm`` then smooths every PBE gradient."""
    directory = _template_copy(tmp_path, "sic_zb", "\nstable\n  .true.\n")
    state, params = ElkState.read(directory), read_elk_parameters(directory)
    assert params.msmgmt == 1
    with pytest.raises(NotImplementedError, match="msmgmt"):
        exchange_correlation_potential(state, params)


def test_gradient_smoothing_runs_under_the_lda(tmp_path):
    directory = _template_copy(tmp_path, "h_sc", "\nmsmgmt\n  2\n")
    state, params = ElkState.read(directory), read_elk_parameters(directory)
    assert params.msmgmt == 2
    exchange_correlation_potential(state, params)


def _with_scaled_xc(text):
    marker = "Number of Bravais lattice symmetries"
    return text.replace(marker, " scaled spin exchange-correlation enabled\n"
                                "  scaling factor :    0.8000000000\n\n" + marker, 1)


def test_a_scaled_spin_exchange_correlation_is_refused(tmp_path):
    directory = _template_copy(tmp_path, "h_sc_spin", info=_with_scaled_xc)
    state, params = ElkState.read(directory), read_elk_parameters(directory)
    assert params.tssxc
    with pytest.raises(NotImplementedError, match="sxcscf"):
        exchange_correlation_potential(state, params)


def test_a_scaled_spin_exchange_correlation_runs_unpolarized(tmp_path):
    directory = _template_copy(tmp_path, "h_sc", info=_with_scaled_xc)
    state, params = ElkState.read(directory), read_elk_parameters(directory)
    assert params.tssxc
    exchange_correlation_potential(state, params)


def test_nosource_is_refused_on_a_spin_polarized_state(tmp_path):
    directory = _template_copy(tmp_path, "h_sc_spin", "\nnosource\n  .true.\n")
    state, params = ElkState.read(directory), read_elk_parameters(directory)
    with pytest.raises(NotImplementedError, match="nosource"):
        exchange_correlation_potential(state, params)
    unpolarized = _template_copy(tmp_path, "h_sc", "\nnosource\n  .true.\n")
    exchange_correlation_potential(ElkState.read(unpolarized), read_elk_parameters(unpolarized))


def test_a_constant_electric_field_is_refused(tmp_path):
    def field(text):
        marker = "Number of Bravais lattice symmetries"
        return text.replace(marker, "Constant electric field applied across unit cell\n"
                                    " field strength :    0.0  0.0  0.01\n\n" + marker, 1)

    directory = _template_copy(tmp_path, "h_sc", info=field)
    state, params = ElkState.read(directory), read_elk_parameters(directory)
    assert params.tefield
    with pytest.raises(NotImplementedError, match="tefield"):
        coulomb_potential(state, params)


@pytest.mark.parametrize("text, switches", [
    ("stable\n  .true.\n\nmsmgmt\n  0\n", dict(symtype=1, msmgmt=0, nosource=False)),
    ("msmgmt\n  0\n\nstable\n  .true.\n", dict(symtype=1, msmgmt=1, nosource=False)),
    ("metagga\n  T\n", dict(symtype=1, msmgmt=4, nosource=False)),
    ("nosym\n  .false.\n\nnosource\n  .true.\n", dict(symtype=1, msmgmt=0, nosource=True)),
    ("!nosym\n  .true.\n\nsymtype\n  2\n", dict(symtype=2, msmgmt=0, nosource=False)),
    ("symtype\n  2\n\nnosym\n  .true.\n", dict(symtype=0, msmgmt=0, nosource=False)),
])
def test_the_elk_in_switches_are_read_in_file_order(text, switches, tmp_path):
    """``readinput`` reads the blocks in order: ``stable`` raises ``msmgmt``, a later one sets it."""
    from defuqumat.io.elk_potential import _elk_input_switches

    (tmp_path / "elk.in").write_text(text)
    assert _elk_input_switches(tmp_path / "elk.in") == switches


def test_a_template_without_elk_in_is_warned_about_where_it_matters(tmp_path):
    import warnings

    gga = _template_copy(tmp_path, "sic_zb")
    (gga / "elk.in").unlink()
    with pytest.warns(UserWarning, match="no elk.in"):
        exchange_correlation_potential(ElkState.read(gga), read_elk_parameters(gga))
    lda = _template_copy(tmp_path, "h_sc")
    (lda / "elk.in").unlink()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exchange_correlation_potential(ElkState.read(lda), read_elk_parameters(lda))
