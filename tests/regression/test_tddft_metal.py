"""The sum-over-states ``chi_0`` of a smeared run: a metal, and a gapped cell under a smearing.

``notes/incompat/I95.md`` and ``I96.md``. Three checks, each of which would
read differently with the term it is about switched off:

* **the Drude weight is the optical conductivity's**, on aluminium: the
  head-only ``eps = 1 - X_head`` against ``1 + 4 pi i sigma/w`` with the
  intraband term at ``gamma = eta``, which is the same expression rearranged,
  interband and intraband. Without the Drude block the two differ by the
  whole ``wp^2/(w + i eta)^2``.
* **the f-sum rule converges onto the diamagnetic weight with the k-mesh**,
  on aluminium: ``-Y^2 X_head(iY)`` at large ``Y`` is the spectral weight in
  closed form, and it tends to ``(8 pi/Omega) sum W_n <n|d2H/dk^2|n>`` as the
  mesh grows, because the Drude weight ``sum w delta v^2`` is the integral of
  ``sum f d2e/dk^2`` by parts only in the continuum. Without the Drude block
  aluminium, a nearly free-electron metal, keeps a small fraction of it.
* **silicon under a smearing falls onto its fixed-occupation spectrum as
  ``degauss`` falls**, on the same density, so that only the sum moves.

Slow: the aluminium meshes up to 8x8x8 and four silicon spectra.
"""

from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.response.conductivity import optical_conductivity
from defuqumat.response.velocity import VelocityOperator
from defuqumat.scf import run_scf
from defuqumat.system import build_system
from defuqumat.tddft.chi0 import independent_response
from defuqumat.units import E2, FPI
from defuqumat.workflows.nscf import fixed_density_states
from defuqumat.workflows.tddft import run_absorption

pytestmark = [pytest.mark.regression, pytest.mark.slow]

CASES = Path(__file__).resolve().parents[1] / "data" / "qe"
PSEUDO = Path(__file__).resolve().parents[1] / "data" / "pseudo"


@pytest.fixture(autouse=True)
def _bounded_compilation():
    """Three aluminium meshes and two silicon cells share no shape."""
    yield
    jax.clear_caches()


def _system(text):
    system = build_system(parse_pw_input(text))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    return system, pseudos


@lru_cache(maxsize=2)
def _aluminium(nk: int):
    """``al-conductivity`` on the whole unshifted ``nk^3`` grid, converged at 1e-10."""
    text = (CASES / "al-conductivity.in").read_text().replace(
        " 8 8 8 0 0 0", f" {nk} {nk} {nk} 0 0 0")
    system, pseudos = _system(text)
    result = run_scf(system, pseudos, conv_thr=1e-10, max_iterations=200)
    return system, pseudos, result


def drude_identity(nk: int = 4, nbnd: int = 12, eta: float = 0.01,
                   drude: bool = True):
    """``(eps from chi_0, eps from sigma, wp^2 in Ry^2)`` on aluminium, diagonals ``(nw, 3)``."""
    import defuqumat.tddft.chi0 as chi0_module

    system, pseudos, result = _aluminium(nk)
    calculation, _, eigenvalues, psi = fixed_density_states(
        system, pseudos, result.density, nbnd=nbnd, conv_thr=1e-10)
    v_scf = calculation.potential(result.density).v_scf
    w = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
    original = chi0_module._drude_block
    if not drude:
        chi0_module._drude_block = lambda weight, eig, element, zomega, volume, tol: (
            jnp.zeros((zomega.shape[0], 3, 3), dtype=zomega.dtype))
    try:
        chi = independent_response(calculation, psi, eigenvalues, v_scf, w,
                                   ecut_response=0.0, broadening=eta)
    finally:
        chi0_module._drude_block = original
    here = 1.0 - np.einsum("wii->wi", np.asarray(chi.head))

    energies = jnp.asarray(eigenvalues)
    energies = energies[None] if energies.ndim == 2 else energies
    fermi = calculation.occupations(energies)[1]["fermi_energy"]
    sigma = optical_conductivity(calculation, psi, eigenvalues, v_scf,
                                 fermi_energy=fermi, frequencies=w,
                                 broadening=eta, relaxation=eta, intraband=True)
    there = np.einsum("wii->wi", np.asarray(sigma.dielectric))
    plasma2 = float(np.mean(np.diag(np.asarray(sigma.plasma)) ** 2))  # Ry^2
    return here, there, plasma2, w + 1j * eta


def fsum(nk: int, nbnd: int = 16, height: float = 1.0e4, smearing: str | None = None,
         degauss: float | None = None):
    """``(spectral weight with the Drude block, without it, diamagnetic weight)``, Ry^2.

    On the 8x8x8 density of ``al-conductivity``, the states solved on the
    whole unshifted ``nk^3`` grid; ``smearing`` and ``degauss`` replace the
    input's for the fixed-density run, since the identity is one about the
    k-sum of ``d(f de/dk)/dk`` for any smooth ``f`` and not about the
    self-consistent density.

    The spectral weight is ``-Y^2 X_head(iY)`` at ``Y = height``, the
    asymptote ``X_head -> C/z^2``, read off one frequency rather than an
    integral; its correction is ``(e_max/Y)^2``, below 1e-6 here. The
    diamagnetic weight is ``(8 pi/Omega) sum_k sum_n wg_n <n|d2H/dk_a^2|n>``
    averaged over ``a``, the operator's second derivative at frozen states.
    """
    import equinox as eqx

    import defuqumat.tddft.chi0 as chi0_module
    from defuqumat.system.kpoints import KPoints

    _, pseudos, result = _aluminium(8)
    text = (CASES / "al-conductivity.in").read_text()
    if smearing is not None:
        text = text.replace("smearing = 'marzari-vanderbilt', degauss = 0.05",
                            f"smearing = '{smearing}', degauss = {degauss}")
    system, _ = _system(text)
    points = KPoints.automatic((nk, nk, nk), (0, 0, 0), system.cell,
                               precision=system.cell.precision)
    system = eqx.tree_at(lambda s: s.kpoints, system, points)
    calculation, _, eigenvalues, psi = fixed_density_states(
        system, pseudos, result.density, nbnd=nbnd, conv_thr=1e-10)
    v_scf = calculation.potential(result.density).v_scf
    z = np.array([1j * height])

    def weight():
        chi = independent_response(calculation, psi, eigenvalues, v_scf, z,
                                   ecut_response=0.0, broadening=0.0)
        head = np.asarray(chi.head)[0]
        # ``X_head -> C/z^2`` and ``z^2 = -Y^2``, so ``C = -Y^2 X_head``.
        return -float(np.real(np.trace(head)) / 3.0 * height ** 2)

    total = weight()
    original = chi0_module._drude_block
    chi0_module._drude_block = lambda weight_, eig, element, zomega, volume, tol: (
        jnp.zeros((zomega.shape[0], 3, 3), dtype=zomega.dtype))
    try:
        interband = weight()
    finally:
        chi0_module._drude_block = original

    energies = jnp.asarray(eigenvalues)
    energies = energies[None] if energies.ndim == 2 else energies
    wg = np.asarray(calculation.occupations(energies)[0])  # (nspin, nk, nbnd)
    operator = VelocityOperator(calculation, v_scf)
    second = np.asarray(operator.second_matrix_elements(jnp.asarray(psi), sequential=True))
    diagonal = np.real(np.einsum("aasknn->skn", second)) / 3.0
    volume = float(calculation.system.cell.volume)
    diamagnetic = E2 * FPI / volume * float(np.sum(wg * diagonal))
    return total, interband, diamagnetic


@lru_cache(maxsize=1)
def _silicon_density():
    text = (CASES / "si-epsilon-unshifted-nosym.in").read_text()
    system, pseudos = _system(text)
    result = run_scf(system, pseudos, conv_thr=1e-12, max_iterations=200)
    return text, pseudos, result.density


def smeared_silicon(degauss: float | None, drude: bool = True, nbnd: int = 16,
                    eta: float = 0.01, ecut_response: float = 2.0):
    """RPA ``eps_M`` (trace/3, ``(nw,)``) of silicon on the fixed run's density.

    ``degauss = None`` is the fixed-occupation run; otherwise a Gaussian
    smearing of that width, on the same density, so that the sum over states
    is the only thing that moves.
    """
    import defuqumat.tddft.chi0 as chi0_module

    text, pseudos, density = _silicon_density()
    if degauss is not None:
        text = text.replace(
            "ecutwfc = 18.0",
            "ecutwfc = 18.0, occupations = 'smearing', smearing = 'gaussian', "
            f"degauss = {degauss}")
    system, _ = _system(text)
    w = np.array([0.05, 0.1, 0.2, 0.3, 0.4])
    original = chi0_module._drude_block
    if not drude:
        chi0_module._drude_block = lambda weight, eig, element, zomega, volume, tol: (
            jnp.zeros((zomega.shape[0], 3, 3), dtype=zomega.dtype))
    try:
        spectrum = run_absorption(system, pseudos, density, w, kernel="rpa",
                                  nbnd=nbnd, broadening=eta,
                                  ecut_response=ecut_response,
                                  static_residual=False, conv_thr=1e-11)
    finally:
        chi0_module._drude_block = original
    return np.trace(np.asarray(spectrum.epsilon), axis1=1, axis2=2) / 3.0


# -- the tests ------------------------------------------------------------------


def test_the_drude_weight_of_aluminium_is_the_conductivitys():
    """The head-only ``eps`` is ``1 + 4 pi i sigma/w``, interband and Drude together."""
    here, there, plasma2, z = drude_identity()
    assert np.max(np.abs(here - there)) < 1e-8
    # The control: the Drude block is what carries the agreement.
    off, _, _, _ = drude_identity(drude=False)
    assert np.allclose(off - here, plasma2 / z[:, None] ** 2, rtol=1e-6)


def test_the_f_sum_rule_of_aluminium_converges_onto_the_diamagnetic_weight():
    """``C_inter + C_Drude -> (8 pi/Omega) sum W <d2H/dk^2>`` with the k-mesh."""
    ratios, interband = {}, {}
    for nk in (4, 8):
        total, inter, diamagnetic = fsum(nk)
        ratios[nk] = total / diamagnetic
        interband[nk] = inter / diamagnetic
    assert abs(ratios[8] - 1.0) < abs(ratios[4] - 1.0)
    assert ratios[8] == pytest.approx(1.0, abs=0.02)
    # Without the Drude block a nearly free-electron metal keeps a fraction.
    assert interband[8] < 0.5


def test_smeared_silicon_falls_onto_its_fixed_spectrum_as_degauss_falls():
    """The discrepancy is the smearing's tail across the gap, and it falls with it."""
    fixed = smeared_silicon(None)
    gaps = {d: np.max(np.abs(smeared_silicon(d) - fixed)) for d in (0.02, 0.01, 0.005)}
    assert gaps[0.005] < gaps[0.01] < gaps[0.02]
    assert gaps[0.005] < 1e-6
