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
