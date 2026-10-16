"""The sum-over-states ``chi_0`` of a nonmagnetic spinor, with and without spin-orbit.

A spinor's pair density is the spin trace ``sum_s conj(u_i,s) u_j,s``, the
charge-charge block of ``chi_0``, and when the ground state carries no moment
(``nspin_mag = 1``) that block is the whole of what the Coulomb kernel and the
unpolarized exchange-correlation kernels see. Two checks, sharing nothing but
the code under test:

1. **The scalar identity.** Silicon without spin-orbit coupling run as a
   spinor is the scalar run on a doubled space: every scalar band becomes a
   Kramers pair holding one electron each, so a spinor cut at ``2 n`` bands
   is the scalar cut at ``n``, and ``chi_0(G, G', omega)`` must be the same
   matrix at every frequency. A factor of two in the spin sum, in the weights
   or in the occupied count is 100 per cent here.
2. **Spin-orbit coupling, against the Sternheimer solve.** Trigonal selenium
   on SG15's fully relativistic norm-conserving dataset, the one committed
   norm-conserving spin-orbit insulator on a whole unshifted grid: ``eps_M(0)``
   in RPA from the Dyson equation against the spinor Sternheimer
   ``epsilon_infinity`` with Hartree screening, through
   :func:`~defuqumat.workflows.tddft.run_absorption`'s own
   ``static_residual``. No external code has this number (QE's TDDFPT is out
   of scope, Elk's ``tddftlr`` has no spin-orbit reference), so the reference
   is the band-complete route in this code, and the residual is the sum over
   states' band truncation once the response cutoff is converged, which must
   shrink as ``nbnd`` grows and match the scalar cell's own residual.

``jax.clear_caches()`` between tests and ``lru_cache(maxsize=2)``, for
``CLAUDE.md``'s reason: two cells, each of which compiles the response stack.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import read_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf import Calculation, run_scf
from defuqumat.system import build_system
from defuqumat.tddft import independent_response, solve_dyson
from defuqumat.workflows.nscf import fixed_density_states

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _drop_compiled_code():
    yield
    jax.clear_caches()


#: The scalar cut. Silicon on the unshifted 4x4x4 grid has a gap above band 20
#: at every k-point (5.6e-2 Ry, asserted below), so neither cut splits a
#: multiplet and the identity is not spoiled by the eigensolver's rotation
#: inside one (rule D4). Every cut from 9 to 19 except 14 and 18 closes to
#: 1e-15 Ry somewhere on this grid (measured, D22), 16 included.
SCALAR_CUT = 20
#: Two more bands than the cut are computed, so that the top of the cut is
#: converged as well as the bottom and the gap above it can be read.
EXTRA = 2
ECUT_RESPONSE = 4.0
#: The frequencies the identity is asserted at: the static point with no
#: broadening and one inside the absorption, Ry, broadened.
FREQUENCIES = np.array([0.0, 0.3 + 0.01j])
#: Relative to the largest entry of the matrix, over every frequency.
#: Measured 5.8e-11 (3.2e-11 at the static point, 5.8e-11 at 0.3 + 0.01i Ry),
#: and 5.0e-10 in an ``eps_M(0)`` of 22.98, one run on D22. A factor of two
#: anywhere in the spin sum is 1 in these units.
IDENTITY = 1.0e-9

#: Selenium's response cutoff, Ry. **At 4 the local-field effect is a third
#: missing** and the residual is +2.7 per cent on both the spinor and the
#: scalar cell (+0.443 and +0.448), growing with ``nbnd`` rather than shrinking;
#: at 8 it is -0.20 and -0.19 per cent and at 12 -0.265 and -0.257 (D22). 12 is
#: where what is left is the band truncation, which is what is asserted.
SELENIUM_ECUT = 12.0
#: The spinor's band count, and the scalar companion's (half, the same states).
SELENIUM_NBND = 144
#: ``|residual| / eps_M`` on the spinor at 144 bands: measured 2.65e-3.
SELENIUM_TRUNCATION = 4.0e-3
#: The spinor residual against the scalar one, relative: measured 8.1e-5
#: (-2.650e-3 against -2.569e-3). Two different datasets (SG15's fully and
#: scalar relativistic selenium), so the dielectric constants differ by 0.042
#: and are not compared; what each sits from its own band-complete solve is.
SELENIUM_SPINOR_AGAINST_SCALAR = 5.0e-4


@lru_cache(maxsize=2)
def _silicon(noncolin: bool):
    """``si-epsilon-unshifted-nosym``, scalar or spinor, with empty states."""
    parsed = read_pw_input(CASES / "si-epsilon-unshifted-nosym.in")
    if noncolin:
        parsed.namelists["system"]["noncolin"] = True
    system = build_system(parsed)
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    assert calculation.noncolin == noncolin
    scf = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                  max_iterations=80)
    assert scf.converged
    degeneracy = 2 if noncolin else 1
    nbnd = degeneracy * (SCALAR_CUT + EXTRA)
    calculation, system, eigenvalues, wavefunctions = fixed_density_states(
        system, pseudos, scf.density, nbnd=nbnd, conv_thr=1e-12,
        calculation=calculation,
    )
    cut = degeneracy * SCALAR_CUT
    eigenvalues = np.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]
    gap = float((eigenvalues[..., cut] - eigenvalues[..., cut - 1]).min())
    potential = calculation.potential(jnp.asarray(scf.density))
    chi = independent_response(
        calculation, jnp.asarray(wavefunctions)[:, :, :cut],
        eigenvalues[:, :, :cut], potential.v_scf, FREQUENCIES,
        ecut_response=ECUT_RESPONSE, broadening=0.0,
    )
    return scf, chi, gap


def test_a_spinor_without_spin_orbit_gives_the_scalar_chi_0():
    """The whole matrix, at the static point and inside the absorption.

    Measured on D22: the matrices 5.8e-11 apart relative to their largest
    entry, ``eps_M(0)`` 22.977824642 against 22.977824642 (5.0e-10), the total
    energies 1.8e-15 Ry. 64 pairs a k-point against 256.
    """
    scalar_scf, scalar, scalar_gap = _silicon(False)
    spinor_scf, spinor, spinor_gap = _silicon(True)
    # The cut has to be through a gap on both sides, or the comparison reads
    # the eigensolver's choice inside a multiplet (rule D4).
    assert scalar_gap > 1e-3 and spinor_gap > 1e-3
    assert float(spinor_scf.total_energy) == pytest.approx(
        float(scalar_scf.total_energy), abs=1e-10)
    assert spinor.nocc == 2 * scalar.nocc and spinor.npairs == 4 * scalar.npairs
    a, b = np.asarray(scalar.x), np.asarray(spinor.x)
    assert np.abs(b - a).max() / np.abs(a).max() < IDENTITY

    rpa = [float(np.real(np.diag(np.asarray(
        solve_dyson(chi, "rpa", {}, static_index=0).epsilon)[0])).mean())
        for chi in (scalar, spinor)]
    assert abs(rpa[1] - rpa[0]) < IDENTITY * rpa[0]


@lru_cache(maxsize=2)
def _selenium_ground_state(case: str):
    system = build_system(read_pw_input(CASES / f"{case}.in"))
    pseudos = tuple(
        read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species
    )
    calculation = Calculation(system, pseudos)
    scf = run_scf(system, pseudos, calculation=calculation, conv_thr=1e-12,
                  max_iterations=80)
    assert scf.converged
    return system, pseudos, calculation, scf


def selenium_static(case: str, nbnd: int, ecut_response: float = SELENIUM_ECUT):
    """``(eps_M(0) in RPA, minus the Sternheimer solve's)`` at ``nbnd`` bands."""
    from defuqumat.workflows.tddft import run_absorption

    system, pseudos, calculation, scf = _selenium_ground_state(case)
    spectrum = run_absorption(
        system, pseudos, scf.density, frequencies=np.array([0.1]),
        kernel="rpa", nbnd=nbnd, broadening=0.01, ecut_response=ecut_response,
        conv_thr=1e-10, calculation=calculation, static_residual=True,
    )
    # 18 electrons: one per spinor band, two per scalar band.
    assert spectrum.nocc == (18 if calculation.noncolin else 9)
    return spectrum.static_rpa, spectrum.static_residual


def test_spin_orbit_selenium_sits_on_the_sternheimer_solve_to_its_truncation():
    """``eps_M(0)`` from the spinor sum over states against the spinor solve.

    Both sides are RPA and unshifted, so the residual is the sum over states'
    band truncation and the response cutoff, and at ``SELENIUM_ECUT`` the
    cutoff is converged: it must shrink as ``nbnd`` grows, and it must be what
    the scalar cell's own residual is at the same states, since spin-orbit
    coupling changes the states and ``dH/dk`` but not how the pairs are summed.

    Measured on D22 at 144 bands: 16.51019387 against the Sternheimer
    16.55406495, a residual of -4.39e-2 (-2.65e-3 relative), and the scalar
    cell at 72 bands -2.57e-3 (16.55194112 against 16.59458061).
    """
    soc = "se-trigonal-soc"
    calculation = _selenium_ground_state(soc)[2]
    assert calculation.noncolin and calculation.nspin_mag == 1
    assert calculation.system.lspinorb

    rpa, residual = selenium_static(soc, SELENIUM_NBND)
    _, fewer = selenium_static(soc, SELENIUM_NBND // 2)
    relative = residual / (rpa - residual)
    assert abs(relative) < SELENIUM_TRUNCATION
    assert abs(residual) < abs(fewer)

    scalar_rpa, scalar_residual = selenium_static("se-trigonal", SELENIUM_NBND // 2)
    scalar_relative = scalar_residual / (scalar_rpa - scalar_residual)
    assert abs(relative - scalar_relative) < SELENIUM_SPINOR_AGAINST_SCALAR
