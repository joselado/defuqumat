"""P182: the local mBJ, ``c`` as a field (Rauch, Marques and Botti, arXiv:1911.00368).

Three statements, each checked by something that shares no machinery with the
route it checks:

* **The Gaussian average is the paper's Eq. (5).** A periodic sum of Gaussians
  of width ``s`` averaged under one of width ``sigma`` is a sum of Gaussians of
  width ``sqrt(s^2 + sigma^2)`` with the weight ``(s^2/(s^2 + sigma^2))^(3/2)``;
  evaluated here by summing the images in real space, against the G-space
  product of :func:`defuqumat.basis.gradients.gaussian_average`, on a
  non-orthogonal cell.
* **In a bulk crystal the local form is TB09.** With ``sigma`` and the
  threshold radius at 1000 bohr the Gaussian removes every ``G != 0`` and the
  blend never acts, so ``c(r)`` is uniform and equal to
  :func:`defuqumat.xc.mgga.tb09_coefficient`'s plain mean, a route with no
  convolution in it. The SCF lands on TB09's eigenvalues.
* **``c -> 1`` in vacuum**, for both parameter sets, where the density is zero
  or the slightly negative values of a truncated Fourier series.

And every regime lifted for TB09 with one ``c`` refuses the local form by name.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import jax.numpy as jnp

from defuqumat import Calculator
from defuqumat.xc.mgga import (
    LMBJ_PARAMETERS,
    lmbj_coefficient_from_average,
    lmbj_integrand,
    lmbj_vacuum_g,
)

pytestmark = pytest.mark.unit

PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"

#: Silicon at a small cutoff and mesh: the identity holds at any of them.
_SI = """&control
 calculation='scf'
/
&system
 ibrav=2, celldm(1)=10.20, nat=2, ntyp=1, ecutwfc=15.0, nbnd=8,
 input_dft='{dft}'{extra}
/
&electrons
 conv_thr=1.0d-10
/
ATOMIC_SPECIES
 Si 28.086 Si.pz-vbc.UPF
ATOMIC_POSITIONS alat
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""


def _calculator(dft="lmbj", extra="", **options):
    return Calculator.from_text(_SI.format(dft=dft, extra=extra), pseudo_dir=PSEUDO,
                                announce=False, **options)


@lru_cache(maxsize=2)
def _converged(dft, extra=""):
    calc = _calculator(dft, extra)
    return calc, calc.get_scf()


def test_the_gaussian_average_of_a_gaussian_is_the_wider_gaussian():
    from defuqumat.basis.gradients import gaussian_average

    calc = _calculator()
    calculation = calc.calculation
    cell, dense = calculation.system.cell, calculation.basis.dense
    at = np.asarray(cell.at)  # rows, bohr
    n1, n2, n3 = (int(n) for n in dense.grid)
    frac = np.stack(np.meshgrid(np.arange(n1) / n1, np.arange(n2) / n2,
                                np.arange(n3) / n3, indexing="ij"), axis=-1)
    points = frac @ at  # (n1, n2, n3, 3)
    centre = np.array([0.3, -0.7, 1.1])
    s, sigma = 1.2, 3.78
    wide = s * s + sigma * sigma

    def image_sum(width2, weight, reach):
        span = int(np.ceil(reach / np.min(np.linalg.norm(at, axis=1)))) + 1
        total = np.zeros(points.shape[:3])
        for i in range(-span, span + 1):
            for j in range(-span, span + 1):
                for k in range(-span, span + 1):
                    shift = centre + i * at[0] + j * at[1] + k * at[2]
                    d2 = np.sum((points - shift) ** 2, axis=-1)
                    total += weight * np.exp(-0.5 * d2 / width2)
        return total

    field = image_sum(s * s, 1.0, 10.0 * s)
    expected = image_sum(wide, (s * s / wide) ** 1.5, 10.0 * np.sqrt(wide))
    averaged = np.asarray(gaussian_average(jnp.asarray(field), dense, cell, sigma))
    assert np.abs(averaged - expected).max() < 1e-9 * np.abs(expected).max()
    # the mean is the G = 0 coefficient, kept exactly
    assert abs(averaged.mean() - field.mean()) < 1e-12 * field.mean()


@pytest.mark.parametrize("parameters", sorted(LMBJ_PARAMETERS))
def test_c_is_one_in_vacuum_and_the_plain_ratio_in_matter(parameters):
    alpha, beta, exponent = LMBJ_PARAMETERS[parameters]
    g_vac = lmbj_vacuum_g(alpha, beta, exponent)
    rho_th = 3.0 / (4.0 * np.pi * 5.0 ** 3)
    rho = jnp.asarray([-1e-5, 0.0, 1e-12, 1.0])
    grad = jnp.asarray([[3e-6, 0.0, 1e-13, 0.4], [0.0] * 4, [0.0] * 4])
    g = np.asarray(lmbj_integrand(rho, grad, rho_th, g_vac))
    assert np.allclose(g[:3], g_vac, rtol=0, atol=1e-8)
    assert g[3] == pytest.approx(0.4, abs=1e-12)  # erf(1/rho_th) = 1
    c = np.asarray(lmbj_coefficient_from_average(jnp.asarray(g_vac), alpha, beta, exponent))
    assert c == pytest.approx(1.0, abs=1e-14)


def test_in_the_bulk_limit_the_local_form_is_tb09():
    _, tb09 = _converged("tb09")
    _, local = _converged("lmbj", ", lmbj_sigma = 1000.0, lmbj_rs = 1000.0")
    assert local.converged and tb09.converged
    assert np.ptp(local.meta_c_field) < 1e-12
    assert local.meta_c == pytest.approx(tb09.meta_c, abs=1e-10)
    assert np.abs(np.asarray(local.eigenvalues) - np.asarray(tb09.eigenvalues)).max() < 1e-9


def test_the_settings_are_refused_where_they_do_not_apply():
    with pytest.raises(ValueError, match="local mBJ's settings"):
        _calculator("tb09", ", lmbj_sigma = 3.0").calculation
    with pytest.raises(ValueError, match="mbj_c imposes one"):
        _calculator("lmbj", ", mbj_c = 1.1").calculation
    with pytest.raises(ValueError, match="lmbj_parameters"):
        _calculator("lmbj", ", lmbj_parameters = 'nonsense'").calculation
    with pytest.raises(ValueError, match="positive"):
        _calculator("lmbj", ", lmbj_sigma = -1.0").calculation


def test_a_paw_dataset_is_refused():
    text = _SI.format(dft="lmbj", extra="").replace("Si.pz-vbc.UPF", "Si.pz-n-kjpaw_psl.0.1.UPF")
    text = text.replace("ecutwfc=15.0", "ecutwfc=15.0, ecutrho=120.0")
    with pytest.raises(NotImplementedError, match="local mBJ.*PAW"):
        Calculator.from_text(text, pseudo_dir=PSEUDO, announce=False).calculation


def test_single_precision_is_refused():
    with pytest.raises(NotImplementedError, match="band_precision.*local mBJ"):
        _calculator(band_precision="single").calculation


def test_the_regimes_lifted_for_tb09_with_one_c_refuse_the_local_form():
    from defuqumat.forces.energy import reject_potential_only
    from defuqumat.realtime.selfconsistent import require_a_potential_mode
    from defuqumat.scf.residual import make_residual
    from defuqumat.scf.spin_torque import torque_of_result
    from defuqumat.ultracell.potential import require_an_ultracell_functional

    calc = _calculator()
    calculation = calc.calculation
    words = "local mBJ"
    with pytest.raises(NotImplementedError, match=f"residual SCF solver.*{words}"):
        make_residual(calculation, 8, 1e-8)
    with pytest.raises(NotImplementedError, match=f"Sternheimer Jacobian.*{words}"):
        reject_potential_only(calculation, potential_only_ok=True)
    with pytest.raises(NotImplementedError, match=f"ultracell.*{words}"):
        require_an_ultracell_functional(calculation.functional)
    with pytest.raises(NotImplementedError, match=f"spin torque.*{words}"):
        torque_of_result(calculation, None)
    with pytest.raises(NotImplementedError, match=f"potential updated.*{words}"):
        require_a_potential_mode(calculation, "hxc", None, None)
    # and none of them fires for TB09, so the guard is the local form's
    reject_potential_only(_calculator("tb09").calculation, potential_only_ok=True)


def test_a_topological_invariant_is_refused():
    from defuqumat.workflows.topology import _source

    calc, result = _converged("lmbj")
    source = _source(calc.system, calc.pseudos, result.density, 4, 8, 1e-8, None,
                     calculation=calc.calculation, tau=result.tau)
    with pytest.raises(NotImplementedError, match=f"topological invariant.*local mBJ"):
        source.states(np.zeros((1, 3)))


def test_the_local_form_cuts_its_potential_at_1e9_and_tb09_at_1e6():
    """The gate is the local mBJ's own (the user's choice of 2026-11-18).

    Between 1e-9 and 1e-6 the local form's potential is evaluated, so the
    vacuum carries no step at 1e-6 (at the old gate graphane's vacuum state at
    ``Gamma`` moved 1.53 eV with the vacuum's size); in a cell the hold below
    1e-5 then replaces it there. TB09 and BJ06 keep 1e-6, where with one cell-averaged ``c`` the lower
    gate made the same sheet run away.
    """
    from defuqumat.xc.functional import resolve_functional

    rho = jnp.asarray([1e-8, 1e-10])
    sigma, lap, tau = 4.0 * rho ** 2, 4.0 * rho, 0.5 * rho
    local = resolve_functional(["PZ"], "lmbj")
    tb09 = resolve_functional(["PZ"], "tb09")
    v_local = np.asarray(local.meta_exchange_potential(rho, sigma, lap, tau, 1.0))
    v_tb09 = np.asarray(tb09.meta_exchange_potential(rho, sigma, lap, tau, 1.0))
    assert v_local[0] != 0.0 and v_local[1] == 0.0
    assert np.all(v_tb09 == 0.0)


def test_the_band_structure_route_rebuilds_c_of_r_and_lands_on_the_scf():
    """Bands at the SCF's own k-points are the SCF's eigenvalues.

    The fixed-density route rebuilds the potential from the density and
    ``tau`` (``workflows/nscf.py``), so under the local form it rebuilds
    ``c(r)`` too; a route that fell back to one ``c``, or to none, would read
    off this by the shift between the two. Six of the eight bands: the SCF does
    not converge its two highest ones tightly, and they differ by 0.022 Ry
    under TB09 exactly as under the local form, while the six below agree to
    1.6e-6 Ry in both (measured 2026-11-18).
    """
    calc, result = _converged("lmbj")
    bands = calc.get_bands(kpoints=calc.system.kpoints)
    shift = np.abs(np.asarray(bands.eigenvalues) - np.asarray(result.eigenvalues))[:, :6].max()
    assert shift < 5e-6


def test_an_unpolarized_two_channel_run_is_the_one_channel_run():
    """``nspin = 2`` with no moment is ``nspin = 1``: ``c(r)`` is the total's.

    ``test_mbj.py``'s identity for TB09, under the local form: the Gaussian
    average is of the total density's ratio, so the two regimes must build the
    same field and the same eigenvalues.
    """
    smearing = ", occupations='smearing', smearing='gaussian', degauss=0.02"
    _, one = _converged("lmbj", smearing)
    _, two = _converged("lmbj", smearing + ", nspin=2, starting_magnetization(1)=0.0")
    assert np.abs(np.asarray(two.meta_c_field) - np.asarray(one.meta_c_field)).max() < 1e-10
    for channel in np.asarray(two.eigenvalues):
        assert np.abs(channel - np.asarray(one.eigenvalues)).max() < 1e-9


@pytest.mark.parametrize("extra, words", [
    (", noncolin=.true.", "spinor"),
    (", nosource=.true., noncolin=.true.", "spinor"),
])
def test_the_unmeasured_magnetic_regimes_are_refused(extra, words):
    with pytest.raises(NotImplementedError, match=f"local mBJ.*{words}"):
        _calculator("lmbj", extra).calculation


#: A laterally uniform sheet in a 40 bohr cell. Only the dense grid and the
#: cell are used: the density, ``tau`` and the potential are laid on them by
#: hand, so the hold is checked on a tail whose two faces are chosen.
_SHEET = """&control
 calculation='scf'
/
&system
 ibrav=6, celldm(1)=4.0, celldm(3)=10.0, nat=1, ntyp=1, ecutwfc=20.0
/
&electrons
/
ATOMIC_SPECIES
 H 1.008 H.pbe-hgh.UPF
ATOMIC_POSITIONS bohr
 H 0.0 0.0 20.0
K_POINTS automatic
 1 1 1 0 0 0
"""
_SHEET_LENGTH = 40.0


@lru_cache(maxsize=1)
def _sheet_grid():
    calculation = Calculator.from_text(_SHEET, pseudo_dir=PSEUDO, announce=False).calculation
    cell, dense = calculation.system.cell, calculation.basis.dense
    n3 = int(dense.grid[2])
    return cell, dense, np.arange(n3) * _SHEET_LENGTH / n3


def _sheet_profile(z, kappa_above, kappa_below, peak=0.1, centre=20.0, a=3.0):
    """``n(z)`` and ``dn/dz``: ``exp(-2 kappa_above u)`` above, ``exp(-2 kappa_below |u|)`` below.

    ``n0 exp(-(ka + kb) sqrt(u^2 + a^2) - (ka - kb) u)``, summed over the
    neighbouring images, with ``n0`` putting about ``peak`` at the centre. It is analytic in a strip of half-width ``a``, so its
    Fourier series is converged on the dense grid to about 1e-13, and the meta
    potential evaluated on it is a clean tail down to the gate.
    """
    n0 = peak * np.exp((kappa_above + kappa_below) * a)
    n = np.zeros_like(z)
    dn = np.zeros_like(z)
    for image in (-1, 0, 1):
        u = z - centre - image * _SHEET_LENGTH
        root = np.sqrt(u * u + a * a)
        f = n0 * np.exp(-(kappa_above + kappa_below) * root - (kappa_above - kappa_below) * u)
        n += f
        dn += f * (-(kappa_above + kappa_below) * u / root - (kappa_above - kappa_below))
    return n, dn


def _sheet_tau(n, dn):
    """``tau`` in Ry: von Weizsaecker plus a third of Thomas-Fermi, any positive ``D``."""
    return 2.0 * (dn * dn / (8.0 * n) + 0.3 * 0.6 * (3.0 * np.pi ** 2) ** (2.0 / 3.0) * n ** (5.0 / 3.0))


def _sheet_potential(channels, monkeypatch=None, functional=None):
    """``meta_exchange`` on laterally uniform channels; with ``monkeypatch``, the hold off.

    Returns the potential along ``z``, ``(nspin, n3)``.
    """
    import jax
    import defuqumat.xc.mgga as mgga
    from defuqumat.basis.fft import r_to_g
    from defuqumat.scf.potential import meta_exchange
    from defuqumat.xc.functional import resolve_functional

    cell, dense, _ = _sheet_grid()
    grid = tuple(int(n) for n in dense.grid)
    rho = jnp.asarray(np.stack([np.broadcast_to(n, grid) for n, _ in channels]))
    tau = jnp.asarray(np.stack([np.broadcast_to(t, grid) for _, t in channels]))
    rho_g = jax.vmap(r_to_g, in_axes=(0, None))(rho.astype(jnp.result_type(rho, 1j)),
                                                dense.fft_index)
    functional = functional or resolve_functional(["PZ"], "lmbj")
    if monkeypatch is not None:
        monkeypatch.setattr(mgga, "hold_vacuum_potential", lambda v, *args, **kwargs: v)
    try:
        v, _ = meta_exchange(rho, rho_g, tau, dense, cell, functional)
    finally:
        if monkeypatch is not None:
            monkeypatch.undo()
    return np.asarray(v)[:, 0, 0, :]


def _held_reference(v_raw, rho):
    """The hold written out on one channel along ``z``: the shell and ``w`` on ``rho``."""
    from scipy.special import erf

    from defuqumat.xc.mgga import LMBJ_HOLD_WIDTH, LMBJ_RHO_HOLD

    positive = rho > 0.0
    weight = np.where(positive, 0.5 * (1.0 + erf(
        np.log(np.where(positive, rho, LMBJ_RHO_HOLD) / LMBJ_RHO_HOLD) / LMBJ_HOLD_WIDTH)), 0.0)
    shell = (rho > LMBJ_RHO_HOLD) & (rho < 10.0 * LMBJ_RHO_HOLD)
    tail = v_raw[shell].mean()
    return weight * v_raw + (1.0 - weight) * tail


def test_a_polarized_channel_is_held_on_its_own_density(monkeypatch):
    """Each channel's shell and weight are on ``2 n_sigma`` (audit 2026-11-20 #47).

    The channel potential is evaluated on the channel density, so where the
    minority channel is on its own tail the total can still be well inside
    the sheet. Here the minority carries 1e-4 of the majority: where the total
    is between 1e-3 and 1e-2 the minority's ``2 n_sigma`` is below 2e-6, its
    weight below 0.03, and its potential has to be held there. On the total's
    shell it was left as evaluated.
    """
    _, _, z = _sheet_grid()
    n, dn = _sheet_profile(z, 0.6, 1.0, peak=0.3)
    ratio = 1e-4
    channels = [(f * n, _sheet_tau(f * n, f * dn)) for f in (1.0 / (1.0 + ratio), ratio / (1.0 + ratio))]
    held = _sheet_potential(channels)
    raw = _sheet_potential(channels, monkeypatch)
    for s, (rho, _) in enumerate(channels):
        assert ((2.0 * rho > 1e-5) & (2.0 * rho < 1e-4)).sum() >= 3
        expected = _held_reference(raw[s], 2.0 * rho)
        assert np.abs(held[s] - expected).max() < 1e-12
    inside = (n > 1e-3) & (2.0 * channels[1][0] < 2e-6)
    assert inside.sum() >= 4
    assert np.abs(held[1][inside] - raw[1][inside]).min() > 1e-3


def test_an_unpolarized_pair_of_channels_is_held_as_one_channel():
    """``(n/2, n/2)`` is ``n`` at ``nspin = 1``, on a sheet whose vacuum the hold reaches.

    The SCF identity above runs on bulk silicon, where no point is in the
    shell and the hold never acts, so it cannot see the hold.
    """
    _, _, z = _sheet_grid()
    n, dn = _sheet_profile(z, 0.6, 1.0)
    tau = _sheet_tau(n, dn)
    assert ((n > 1e-5) & (n < 1e-4)).sum() >= 8
    one = _sheet_potential([(n, tau)])
    two = _sheet_potential([(0.5 * n, 0.5 * tau), (0.5 * n, 0.5 * tau)])
    assert np.abs(two - one[0][None]).max() < 1e-12


def test_the_vacuum_potential_is_held_at_the_clean_tails_value():
    """Below ``n = 1e-5`` the potential becomes its mean over ``1e-5 < n < 1e-4``.

    The vacuum check (P182): evaluated on the plane-wave density's noise floor
    the vacuum potential sat 3.3 Ry above PBE's and noisy, where the
    Becke-Johnson asymptote is 0.33 Ry; held, it is flat at 0.28. Matter is
    untouched, and a cell with no point in the shell keeps its potential.
    """
    from defuqumat.xc.mgga import LMBJ_RHO_HOLD, hold_vacuum_potential

    rho = jnp.asarray([1e-1, 3e-5, 5e-5, 1e-8, -1e-9])
    v = jnp.asarray([[0.7, 0.30, 0.34, 3.3, 9.0]])
    held = np.asarray(hold_vacuum_potential(v, rho))[0]
    tail = 0.5 * (0.30 + 0.34)
    assert held[0] == pytest.approx(0.7, abs=1e-12)           # matter
    assert held[3] == pytest.approx(tail, abs=1e-6)          # noise floor
    assert held[4] == pytest.approx(tail, abs=1e-12)         # negative noise
    assert 0.30 < held[1] < 0.34                             # the switch
    bulk = jnp.asarray([1e-1, 2e-2, 6e-4])
    assert np.array_equal(np.asarray(hold_vacuum_potential(v[:, :3], bulk)), np.asarray(v[:, :3]))
    assert LMBJ_RHO_HOLD == 1e-5
