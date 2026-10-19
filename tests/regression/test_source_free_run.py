"""``nosource`` on a whole calculation, and the torque it makes measurable.

:mod:`tests.unit.test_source_free_field` checks the projection on fields built
by hand. This file is the run: what the flag does to a converged magnet, what it
does to the exchange-correlation torque, and what it refuses.

**The finding this file exists to pin is a null.** In the local spin-density
approximation ``B_xc`` is parallel to ``m`` at every point of the grid, so the
torque ``int m x B_xc`` is zero at *every* density, converged or not, and a test
asserting that it vanishes at self-consistency would pass on a broken code. The
tilt table below is the guard firing: one site's moment is turned away from
where it sat, and the plain functional still reports zero at every angle where
the source-free one grows from 1.6e-4 to 6.3e-4 Ry.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat import Calculator
from defuqumat.basis.gradients import divergence
from defuqumat.scf.spin_torque import torque_of_result

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


@lru_cache(maxsize=2)
def _converged(stem):
    calc = Calculator.from_file(CASES / stem, pseudo_dir=PSEUDO)
    return calc, calc.get_scf()


def _field_divergence(calc, scf):
    """``int |div B_xc|^2``, which is what the projection is named after."""
    calculation = calc.calculation
    from defuqumat.scf.potential import v_of_rho

    density = jnp.asarray(scf.density)
    potential = v_of_rho(
        density, calculation.basis.dense, calc.system.cell,
        calculation.rho_core, calculation.functional, calculation.rho_core_g,
        calculation.quantization_axis, None, calculation.source_free,
    )
    div = divergence(jnp.real(potential.v_scf[1:]), calculation.basis.dense,
                     calc.system.cell)
    scale = calc.system.cell.volume / density[0].size
    return float(scale * jnp.sum(div ** 2))


def test_the_projection_removes_the_divergence_of_a_real_run():
    plain, scf_plain = _converged("ni-noncol-111.in")
    free, scf_free = _converged("ni-noncol-111-nosource.in")
    assert scf_plain.converged and scf_free.converged

    # The guard has to fire: a run without the flag must have a divergence
    # worth removing, or the one with it proves nothing.
    assert _field_divergence(plain, scf_plain) > 1.0e-3
    assert _field_divergence(free, scf_free) < 1.0e-25


def test_a_local_functional_can_exert_no_torque_at_all():
    """Not "vanishes at convergence" -- vanishes always, which is the claim."""
    plain, scf = _converged("ni-noncol-111.in")
    torque = torque_of_result(plain.calculation, scf)
    assert float(torque.parallel_fraction) == pytest.approx(1.0, abs=1.0e-12)
    assert np.linalg.norm(np.asarray(torque.total)) < 1.0e-15


def test_the_source_free_field_is_not_parallel_to_the_magnetization():
    free, scf = _converged("ni-noncol-111-nosource.in")
    torque = torque_of_result(free.calculation, scf)
    assert float(torque.parallel_fraction) < 0.99
    # The *total* is still zero, and by a symmetry rather than by convergence:
    # without spin-orbit coupling the energy does not change under a global
    # rotation of every spin, so the net torque on the cell cannot. Asserted so
    # that a later change which breaks it is seen.
    assert np.linalg.norm(np.asarray(torque.total)) < 1.0e-15


def test_the_moment_survives_and_moves():
    """A robust magnet keeps its moment under the projection, and changes it.

    Measured: 0.6576 to 0.7045 mu_B. The hydrogen helix at the same flag goes
    from 0.4675 to 0.0003, which is a Stoner-marginal cell crossing its own
    criterion rather than anything about the projection, and is why the cell
    here is nickel.
    """
    plain, scf_plain = _converged("ni-noncol-111.in")
    free, scf_free = _converged("ni-noncol-111-nosource.in")
    before = np.linalg.norm(np.asarray(scf_plain.site_moments)[0])
    after = np.linalg.norm(np.asarray(scf_free.site_moments)[0])
    assert before == pytest.approx(0.6576, abs=5.0e-3)
    assert after == pytest.approx(0.7045, abs=5.0e-3)


TILTS = (15.0, 45.0, 90.0)


def test_the_torque_fires_when_a_moment_is_turned():
    """The check that the diagnostic can return a surprise.

    One site's magnetization is turned about the moment's own axis inside that
    site's sphere alone, which is a configuration neither functional converged
    to. The source-free field pushes back; the plain one cannot, at any angle.
    """
    plain, scf = _converged("ni-noncol-111.in")
    free, _ = _converged("ni-noncol-111-nosource.in")
    regions = free.calculation.local_regions()
    density = jnp.asarray(scf.density)
    owner = np.asarray(regions.owner).reshape(density.shape[1:])
    mask = (owner == 0).astype(float)
    m = np.asarray(jnp.real(density[1:]))

    for degrees in TILTS:
        c, s = np.cos(np.deg2rad(degrees)), np.sin(np.deg2rad(degrees))
        turned = np.stack([c * m[0] - s * m[1], s * m[0] + c * m[1], m[2]])
        mixed = jnp.asarray(np.concatenate(
            [np.asarray(jnp.real(density[:1])), m + mask[None] * (turned - m)]
        ))
        bare = torque_of_result(plain.calculation, scf, density=mixed)
        projected = torque_of_result(free.calculation, scf, density=mixed)
        assert np.linalg.norm(np.asarray(bare.sites)[0]) < 1.0e-15
        assert np.linalg.norm(np.asarray(projected.sites)[0]) > 1.0e-6


def _with_nosource(stem, anchor="noncolin = .true."):
    import tempfile
    text = (CASES / stem).read_text()
    assert anchor in text, stem
    path = Path(tempfile.mkdtemp()) / stem
    path.write_text(text.replace(anchor, anchor + "\n    nosource = .true."))
    return path


def test_a_collinear_run_is_refused():
    """The projection is transverse, so a field along z gains x and y
    components a collinear density has nowhere to put. Elk refuses it too
    (``init0.f90`` requires ``ncmag``)."""
    with pytest.raises(NotImplementedError, match="noncollinear"):
        Calculator.from_file(_with_nosource("h2-mirror-afm.in", "nspin = 2"),
                             pseudo_dir=PSEUDO).calculation


def test_paw_is_refused():
    """The one-centre field on the spheres is a second copy this projection
    does not reach, so half the potential would belong to another functional."""
    with pytest.raises(NotImplementedError, match="PAW"):
        Calculator.from_file(_with_nosource("o2-paw-texture.in"),
                             pseudo_dir=PSEUDO).calculation


def test_a_derivative_of_the_energy_is_refused():
    """The potential moved and the energy did not, so the converged state is
    stationary for a functional nothing writes down."""
    free, _ = _converged("ni-noncol-111-nosource.in")
    with pytest.raises(NotImplementedError, match="nosource"):
        free.get_forces()


# --- I108 and I109: 'grad_m' and a potential-only meta-GGA under nosource ----
#
# The four-site hydrogen chain of ``h4-noncolin-force.in`` (starting moments 90 degrees
# apart), with the functional changed and ``conv_thr`` loosened to 1e-8, which
# is ``notes/incompat/sourcefree-harness/run_sourcefree.py``'s computation (D22,
# 2026-10-19, one sample each; ``notes/incompat/I108.md`` and ``I109.md``).


def _chain(functional, choice):
    text = (CASES / "h4-noncolin-force.in").read_text()
    text = text.replace("    tprnfor = .true.\n", "").replace("    tstress = .true.\n", "")
    text = text.replace("conv_thr = 1.0d-11", "conv_thr = 1.0d-8\n    electron_maxstep = 300")
    text = text.replace(
        "noncolin = .true.",
        f"noncolin = .true.\n    input_dft = '{functional}'\n    nosource = .true.\n"
        f"    noncollinear_gga = '{choice}'")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        calc = Calculator.from_text(text, PSEUDO, announce=False)
        return calc, calc.get_scf()


def _sphere_divergences(calc, scf):
    from defuqumat.scf.potential import v_of_rho

    c = calc.calculation
    tau = None if scf.tau is None else jnp.asarray(scf.tau)
    out = []
    for projected in (True, False):
        p = v_of_rho(jnp.asarray(scf.density), c.basis.dense, calc.system.cell,
                     c.rho_core, c.functional, c.rho_core_g, c.quantization_axis,
                     tau, projected, c.noncollinear_gga)
        div = divergence(jnp.real(p.v_scf[1:]), c.basis.dense, calc.system.cell)
        out.append(float(calc.system.cell.volume / div.size * jnp.sum(div ** 2)))
    return out


def test_grad_m_under_nosource_converges_source_free_on_a_texture():
    """I108. Measured: 62 iterations, ``int |div B|^2`` 6.97e-31 projected
    against 3.456 unprojected, site 0 moment 0.2432, parallel fraction 0.98865.
    The same chain without nosource does not converge under either PBE branch
    in 300 iterations (``pw.x`` limit-cycles on it too). Under ``'kubler'``
    with nosource it is at 9.8e-8 after 300 and converges at 371, to a nearly
    nonmagnetic state."""
    calc, scf = _chain("PBE", "grad_m")
    assert scf.converged and calc.calculation.noncollinear_gga == "grad_m"
    on, off = _sphere_divergences(calc, scf)
    assert off > 1.0 and on < 1.0e-25 * off
    assert np.linalg.norm(np.asarray(scf.site_moments)[0]) == pytest.approx(0.2432, abs=5e-3)
    torque = torque_of_result(calc.calculation, scf)
    assert float(torque.parallel_fraction) == pytest.approx(0.98865, abs=2e-3)
    with pytest.raises(NotImplementedError, match="nosource"):
        calc.get_forces()


def test_a_potential_only_meta_gga_under_nosource_converges_source_free():
    """I109. BJ06 (fixed ``c``; TB09's cell average does not converge on a
    chain in vacuum with or without nosource) under ``'grad_m'``. Measured: 117
    iterations, ``int |div B|^2`` 2.33e-29 projected against 83.9
    unprojected, site 0 moment 0.7362."""
    calc, scf = _chain("BJ06", "grad_m")
    assert scf.converged and calc.calculation.source_free
    on, off = _sphere_divergences(calc, scf)
    assert off > 10.0 and on < 1.0e-25 * off
    assert np.linalg.norm(np.asarray(scf.site_moments)[0]) == pytest.approx(0.7362, abs=5e-3)
    with pytest.raises(NotImplementedError):
        calc.get_forces()
    with pytest.raises(NotImplementedError, match="meta-GGA"):
        torque_of_result(calc.calculation, scf)
