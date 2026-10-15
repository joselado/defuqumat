"""The sums over states of a collinear ``nspin = 2`` run: two channels that add.

The optical conductivity, the second-harmonic tensor and the shift current of
a spin-polarized insulator are each channel's band structure's own, summed; the
identity that checks the channel loop is an unpolarized crystal run both ways,
zincblende AlAs with ``tot_magnetization = 0``, whose two channels are each half
of the ``nspin = 1`` answer. ``PLAN.md`` P139 has the measurements.

That identity cannot see a channel mispairing, since the two channels are the
same arrays: channel 0's velocity contracted with channel 1's energies, or
channel 0 counted twice, gives the ``nspin = 1`` answer exactly. The shift
current is also checked on a magnet, AlAs with ``tot_magnetization = 2``,
against the same states written as spinors ``(u, 0)`` and ``(0, u)``, which is
P140's construction: the spinor route has one Hamiltonian on ``2 npwx`` and no
channel axis at all, so it shares the assembly and nothing of the fold.
"""

import re
from pathlib import Path

import jax
import numpy as np
import pytest

from defuqumat import Calculator

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"


@pytest.fixture(autouse=True)
def _bounded_compilation():
    yield
    jax.clear_caches()


def _alas(tmp_path, pseudo_dir, extra=""):
    text = (CASES / "alas-shg.in").read_text()
    text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", "ecutwfc = 10.0", text)
    text = re.sub(r"K_POINTS.*\n\s*[0-9 ]+\n", "K_POINTS (automatic)\n 2 2 2 0 0 0\n", text)
    if extra:
        text = re.sub(r"&system", "&system\n    " + extra, text, count=1)
    path = tmp_path / f"alas{len(extra)}.in"
    path.write_text(text)
    calculator = Calculator.from_file(path, pseudo_dir=pseudo_dir, announce=False)
    calculator.get_scf(conv_thr=1e-12)
    return calculator


def test_two_unpolarized_channels_are_the_scalar_run(pseudo_dir, tmp_path):
    """``get_shg`` and ``get_optical_conductivity`` at ``nspin = 2``, no moment, against ``nspin = 1``.

    At nine bands, a clean cut (the gap to the tenth is 4.9e-3 Ry at every
    k-point; a cut inside a multiplet makes either sum depend on the rotation
    the eigensolver returned, which is a statement about the cut and not about
    the channels). Measured: chi^(2) 1.4e-9 of its largest component apart.
    """
    one = _alas(tmp_path, pseudo_dir)
    two = _alas(tmp_path, pseudo_dir, "nspin = 2, tot_magnetization = 0,")
    options = dict(nbnd=9, nw=4, window=0.2, broadening=0.01)
    a, b = np.asarray(one.get_shg(**options).chi), np.asarray(two.get_shg(**options).chi)
    assert np.abs(a - b).max() < 1e-7 * np.abs(a).max()
    options = dict(nbnd=9, nw=4, window=0.4, broadening=0.01)
    a = np.asarray(one.get_optical_conductivity(**options).sigma)
    b = np.asarray(two.get_optical_conductivity(**options).sigma)
    assert np.abs(a - b).max() < 1e-7 * np.abs(a).max()


def test_the_shift_current_of_two_unpolarized_channels_is_the_scalar_run(pseudo_dir, tmp_path):
    """``get_shift_current`` at ``nspin = 2``, no moment, against ``nspin = 1``.

    The same cell, cut and band count as the test above (nine bands, a clean
    cut); the intermediate sum of the generalised derivative runs over one
    channel's nine bands, as it runs over the nine of the scalar run.
    """
    one = _alas(tmp_path, pseudo_dir)
    two = _alas(tmp_path, pseudo_dir, "nspin = 2, tot_magnetization = 0,")
    options = dict(nbnd=9, nw=4, window=0.6, broadening=0.01)
    a = np.asarray(one.get_shift_current(**options).sigma)
    b = np.asarray(two.get_shift_current(**options).sigma)
    print("unpolarized shift current: max|sigma|", np.abs(a).max(),
          "relative difference", np.abs(a - b).max() / np.abs(a).max())
    assert np.abs(a).max() > 0.0
    assert np.abs(a - b).max() < 1e-7 * np.abs(a).max()


def test_the_shift_current_of_a_collinear_magnet_written_as_spinors_is_itself(
        pseudo_dir, tmp_path):
    """The channel sum on AlAs at ``tot_magnetization = 2`` against its spinor form.

    Five electrons up and three down at 6 Ry on the whole 2x2x2 mesh (P140's
    cell). Each channel's twelve fixed-density states become ``(u, 0)`` and
    ``(0, u)``, ordered occupied first, so the spinor run's fixed filling of
    eight bands by index is the channels' five and three; its potential is
    built from ``(n, 0, 0, m)``, whose ``v0 +- B_z`` are the channels' own.
    The spinor Hamiltonian is block diagonal in spin, so every cross-channel
    velocity and second velocity is zero and the two tensors are one sum.
    """
    import jax.numpy as jnp

    from defuqumat.response.photocurrent import shift_current
    from defuqumat.scf.driver import Calculation
    from defuqumat.workflows.nscf import fixed_density_states

    def _input(extra):
        text = (CASES / "alas-shg.in").read_text()
        text = re.sub(r"ecutwfc\s*=\s*[0-9.dD+-]+", "ecutwfc = 6.0", text)
        text = re.sub(r"K_POINTS.*\n\s*[0-9 ]+\n", "K_POINTS (automatic)\n 2 2 2 0 0 0\n",
                      text)
        text = re.sub(r"&system", "&system\n    " + extra, text, count=1)
        path = tmp_path / f"alas-mag{len(extra)}.in"
        path.write_text(text)
        return path

    collinear = Calculator.from_file(
        _input("nspin = 2, tot_magnetization = 2, starting_magnetization(1) = 0.3,"),
        pseudo_dir=pseudo_dir, announce=False)
    rho = np.asarray(collinear.get_scf(conv_thr=1e-12).density)
    computed = 12
    calc_c, _, energies, bands = fixed_density_states(
        collinear.system, collinear.pseudos, jnp.asarray(rho), nbnd=computed + 1,
        conv_thr=1e-12, k_batch=None)
    energies, bands = np.asarray(energies)[..., :computed], np.asarray(bands)[..., :computed, :]
    v_c = np.asarray(calc_c.potential(jnp.asarray(rho)).v_scf)

    spinor = Calculator.from_file(
        _input("noncolin = .true., starting_magnetization(1) = 0.3, "
               "angle1(1) = 0.0, angle2(1) = 0.0,"),
        pseudo_dir=pseudo_dir, announce=False)
    calc_n = Calculation(spinor.system, spinor.pseudos)
    assert np.allclose(np.asarray(calc_n.system.kpoints.coords),
                       np.asarray(calc_c.system.kpoints.coords))
    rho_n = np.zeros((4,) + rho.shape[1:])
    rho_n[0], rho_n[3] = rho[0] + rho[1], rho[0] - rho[1]
    v_n = np.asarray(calc_n.potential(jnp.asarray(rho_n)).v_scf)
    assert np.abs(v_n[0] + v_n[3] - v_c[0]).max() < 1e-12
    assert np.abs(v_n[0] - v_n[3] - v_c[1]).max() < 1e-12

    def spinors(up, down):
        return np.concatenate([np.concatenate([up, np.zeros_like(up)], axis=-1),
                               np.concatenate([np.zeros_like(down), down], axis=-1)], axis=1)

    up, down = 5, 3
    basis_n = np.concatenate([spinors(bands[0][:, :up], bands[1][:, :down]),
                              spinors(bands[0][:, up:], bands[1][:, down:])], axis=1)
    energies_n = np.concatenate([energies[0][:, :up], energies[1][:, :down],
                                 energies[0][:, up:], energies[1][:, down:]], axis=1)
    options = dict(nw=6, window=1.0, broadening=0.02, degeneracy_tol=0.02, k_batch=None)
    a = shift_current(calc_c, jnp.asarray(bands), jnp.asarray(energies), jnp.asarray(v_c),
                      **options)
    b = shift_current(calc_n, jnp.asarray(basis_n)[None], jnp.asarray(energies_n)[None],
                      jnp.asarray(v_n), **options)
    x, y = np.asarray(a.sigma), np.asarray(b.sigma)
    print("magnet shift current: max|sigma|", np.abs(x).max(),
          "relative difference", np.abs(x - y).max() / np.abs(x).max())
    assert np.abs(x).max() > 0.0
    assert np.abs(x - y).max() < 1e-9 * np.abs(x).max()
