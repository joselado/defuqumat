"""The conventions of the uniform orbital field: units, gauge, sign, translations, the hook.

Each check is an identity that shares no machinery with the code it tests: the flux
quantum from the SI constants, the curl and the divergence of ``a`` from its Jacobian,
the commutator of two magnetic translations, ``grad f = a_Lan - a_sym`` for the GIPAW
phase, the free Landau spacing, the Dirac ladder at the review's numbers, the regimes
refused by name, and the Davidson precondition hook (the default is the old rule bit for
bit, and a different preconditioner changes the step count, so the hook is live).
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from defuqumat.config import DOUBLE
from defuqumat.magnetic.conventions import MagneticCell, refuse_regime, signed_area
from defuqumat.magnetic.landau import dirac_levels, free_landau_levels
from defuqumat.units import (
    ANGSTROM_TO_BOHR, FLUX_QUANTUM_SI, TPI, flux_per_cell, rational_flux, tesla_from_flux,
)

pytestmark = pytest.mark.unit

A_GRAPHENE = 2.46
HEX = np.array([[1.0, 0.0, 0.0], [-0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 15.0 / A_GRAPHENE]]) \
    * A_GRAPHENE * ANGSTROM_TO_BOHR


def _area(lattice):
    return abs(np.linalg.det(lattice)) / np.linalg.norm(lattice[2])


def test_flux_quantum_and_the_check_values():
    """h/e to 1e-8 relative; graphene 78912 T and the 3x1x1 diamond cell 10877 T at one quantum.

    Both computed here from a = 2.46 A and a = 3.56 A exactly; the design's 78926 and
    10883 T are 1.8e-4 and 5e-4 above them (recorded). Lu et al.'s 10870.4 T for the
    diamond cell is at their own rounded a, and agrees to 0.1 per cent.
    """
    assert FLUX_QUANTUM_SI == pytest.approx(4.135667696e-15, rel=1e-8)
    area_g = _area(HEX)
    phi0_g = tesla_from_flux(1, 1, area_g)
    expected_g = 4.135667696e-15 / (A_GRAPHENE ** 2 * np.sqrt(3) / 2 * 1e-20)
    assert phi0_g == pytest.approx(expected_g, abs=1.0)
    assert phi0_g == pytest.approx(78912.0, abs=1.0)
    a_d = 3.56
    diamond = np.diag([3 * a_d, a_d, a_d]) * ANGSTROM_TO_BOHR
    phi0_d = tesla_from_flux(1, 1, _area(diamond))
    assert phi0_d == pytest.approx(4.135667696e-15 / (3 * a_d * a_d * 1e-20), abs=1.0)
    assert phi0_d == pytest.approx(10870.4, rel=2e-3)
    assert flux_per_cell(phi0_g, area_g) == pytest.approx(1.0, rel=1e-12)


def test_rational_flux_rounds_to_one_over_q_and_reports_the_field():
    area = _area(HEX)
    p, q, b = rational_flux(10.0, area)
    assert (p, q) == (1, round(tesla_from_flux(1, 1, area) / 10.0))
    assert b == pytest.approx(tesla_from_flux(1, q, area))
    assert abs(b - 10.0) / 10.0 < 1e-3
    assert tesla_from_flux(1, 7900, area) == pytest.approx(9.989, abs=2e-3)
    # a field above one quantum per cell: the nearest p/q under qmax
    p, q, b = rational_flux(2.5 * tesla_from_flux(1, 1, area), area, qmax=10)
    assert (p, q) == (5, 2)
    with pytest.raises(ValueError):
        rational_flux(1e-3, area, qmax=100)


def test_gauge_curl_divergence_and_the_field_axis():
    """curl a = 2 pi n_phi a3 / Omega, div a = 2 pi n_phi (b1 . b2), from the Jacobian of a."""
    cell = MagneticCell.build(HEX, 3, 10)
    jac = np.asarray(jax.jacfwd(cell.a)(jnp.array([0.3, -1.2, 0.7])))   # d a_i / d r_j
    curl = np.array([jac[2, 1] - jac[1, 2], jac[0, 2] - jac[2, 0], jac[1, 0] - jac[0, 1]])
    omega = abs(np.linalg.det(HEX))
    np.testing.assert_allclose(curl, TPI * 0.3 * HEX[2] / omega, atol=1e-15)
    b = np.asarray(cell.b_chem)
    np.testing.assert_allclose(b @ HEX.T, np.eye(3), atol=1e-14)
    assert np.trace(jac) == pytest.approx(TPI * 0.3 * (b[0] @ b[1]), abs=1e-15)
    assert abs(np.trace(jac)) > 1e-3          # not zero on a hexagonal cell
    assert float(cell.b_ph) == pytest.approx(np.linalg.norm(curl), rel=1e-13)
    # the magnetic cell and its reciprocal vectors
    np.testing.assert_allclose(np.asarray(cell.magnetic_reciprocal) @ np.asarray(cell.magnetic_lattice).T,
                               np.eye(3), atol=1e-14)


def test_magnetic_translations_commute_up_to_the_flux():
    """T(a1) T(a2) = exp(-i 2 pi n_phi) T(a2) T(a1), and a(a1) . a1 = 0 (T(a1)^X = T(X a1))."""
    cell = MagneticCell.build(HEX, 1, 7)
    a1, a2 = HEX[0], HEX[1]
    commutator = float(cell.translation_phase(a1, a2) - cell.translation_phase(a2, a1))
    assert commutator == pytest.approx(-TPI / 7, abs=1e-14)
    assert float(cell.translation_phase(a1, a1)) == pytest.approx(0.0, abs=1e-14)
    # the Bloch-sum factor vanishes on an a2 wrap and is 2 pi p m1 (b2 . c) on an a1 one
    c = np.array([1.1, 2.3, 0.4])
    assert float(cell.bloch_sum_phase(a2, c)) == pytest.approx(0.0, abs=1e-14)
    b2 = np.asarray(cell.b_chem)[1]
    assert float(cell.bloch_sum_phase(7 * a1, c)) == pytest.approx(-TPI * (b2 @ c), abs=1e-13)


def test_peierls_phase_is_the_line_integral_and_the_loop_sign_is_the_electron():
    """The straight-line phase is -int a.dl (Simpson on a linear a is exact), and a
    counterclockwise cell gives exp(-i 2 pi n_phi)."""
    cell = MagneticCell.build(HEX, 2, 9)
    r0, r1 = np.array([0.4, -0.3, 0.1]), np.array([3.0, 5.5, -0.2])
    s = np.linspace(0.0, 1.0, 3)
    path = r0 + s[:, None] * (r1 - r0)
    integrand = np.asarray(cell.a(path)) @ (r1 - r0)
    simpson = (integrand[0] + 4 * integrand[1] + integrand[2]) / 6
    assert float(cell.peierls_phase(r0, r1)) == pytest.approx(-simpson, abs=1e-13)
    corners = [np.zeros(3), HEX[0], HEX[0] + HEX[1], HEX[1]]
    assert signed_area(corners, HEX) == pytest.approx(1.0, abs=1e-14)
    assert signed_area(corners[::-1], HEX) == pytest.approx(-1.0, abs=1e-14)
    total = sum(float(cell.peierls_phase(corners[i], corners[(i + 1) % 4])) for i in range(4))
    assert np.exp(1j * total) == pytest.approx(np.exp(-1j * TPI * 2 / 9), abs=1e-13)


def test_gipaw_phase_gradient_is_landau_minus_symmetric():
    """grad_d f = a_Lan(d) - a_sym(d), a_sym = B x d / 2, with f(d) = a(d) . d / 2."""
    cell = MagneticCell.build(HEX, 1, 5)
    tau = jnp.array([1.0, 2.0, 0.0])
    d = jnp.array([0.7, -0.4, 0.3])
    f = lambda x: -cell.gipaw_phase(x, tau) - jnp.dot(cell.a(tau), x)   # = a(x) . x / 2
    grad = np.asarray(jax.grad(f)(d))
    bvec = TPI * 0.2 * np.cross(*np.asarray(cell.b_chem)[:2])
    a_sym = 0.5 * np.cross(bvec, np.asarray(d))
    np.testing.assert_allclose(grad, np.asarray(cell.a(d)) - a_sym, atol=1e-15)


def test_a_derivative_in_the_field_survives():
    """The traced scale on a: d B_ph / d scale = B_ph, and a grad through the phase."""
    def bph(scale):
        return MagneticCell.build(HEX, 1, 10, scale=scale).b_ph

    assert float(jax.grad(bph)(1.0)) == pytest.approx(float(bph(1.0)), rel=1e-14)

    def phase(scale):
        cell = MagneticCell.build(HEX, 1, 10, scale=scale)
        return cell.peierls_phase(jnp.array([0.0, 0.0, 0.0]), jnp.array([1.0, 3.0, 0.0]))

    assert float(jax.grad(phase)(1.0)) == pytest.approx(float(phase(1.0)), rel=1e-14)


def test_build_refuses_a_flux_that_is_not_reduced():
    with pytest.raises(ValueError, match="not reduced"):
        MagneticCell.build(HEX, 2, 400)
    with pytest.raises(NotImplementedError):
        MagneticCell.build(HEX, 1, 4, axis=1)


def test_free_landau_spacing_on_the_graphene_cell():
    """q = 10: B_ph = 2 pi / (10 x 18.7154 bohr^2) = 0.033573 Ry, the G3 = 1 branch at +0.049 Ry.

    The design's 18.73 bohr^2 and 0.03355 Ry are the same numbers rounded (5.2409 A^2
    at a = 2.46 A is 18.7154 bohr^2)."""
    cell = MagneticCell.build(HEX, 1, 10)
    area = A_GRAPHENE ** 2 * np.sqrt(3) / 2 * ANGSTROM_TO_BOHR ** 2
    assert float(cell.area) == pytest.approx(area, rel=1e-13)
    assert float(cell.area) == pytest.approx(18.73, abs=0.02)
    assert float(cell.b_ph) == pytest.approx(TPI / (10 * area), rel=1e-13)
    assert float(cell.b_ph) == pytest.approx(0.03355, abs=5e-5)
    levels, mult, labels = free_landau_levels(cell, nmax=3, g3max=1)
    assert levels[0] == pytest.approx(float(cell.b_ph), rel=1e-14)
    assert tuple(labels[0]) == (0, 0)
    g3 = (TPI / (15.0 * ANGSTROM_TO_BOHR)) ** 2
    assert g3 == pytest.approx(0.049, abs=1e-3)
    assert np.all(mult == 1)
    np.testing.assert_allclose(sorted(levels[labels[:, 1] == 0]),
                               [(2 * n + 1) * float(cell.b_ph) for n in range(4)], rtol=1e-14)


def test_dirac_ladder_at_the_review_numbers():
    """v_F = 3 |t| a_cc / 2 hbar = 8.74e5 m/s; E_1 = 0.1002 eV at q = 7900 on graphene."""
    hbar = 6.62607015e-34 / TPI
    v_f = 3 * 2.7 * 1.602176634e-19 * (A_GRAPHENE * 1e-10 / np.sqrt(3)) / (2 * hbar)
    assert v_f == pytest.approx(8.74e5, rel=1e-3)
    b = tesla_from_flux(1, 7900, _area(HEX))
    ladder = dirac_levels(b, v_f, 3)
    assert ladder[3] == 0.0
    assert ladder[4] == pytest.approx(0.10021, abs=1e-4)
    np.testing.assert_allclose(ladder[5:], ladder[4] * np.sqrt([2, 3]), rtol=1e-14)
    np.testing.assert_allclose(ladder[:3], -ladder[6:3:-1], rtol=1e-14)


def test_regimes_are_refused_by_name():
    plain = SimpleNamespace(npol=1, nspin=1, spiral_q=None, hubbard=None, input_dft=None,
                            kpoints=SimpleNamespace(gamma_only=False))
    nc = SimpleNamespace(is_paw=False, is_ultrasoft=False, element="C", functional="SLA-PW-PBX-PBC")
    refuse_regime(plain, (nc,), "a test")
    for change, word in ((dict(npol=2), "spinor"), (dict(nspin=2), "nspin = 2"),
                         (dict(spiral_q=(0, 0, 0.1)), "spiral"), (dict(hubbard=object()), "DFT\\+U"),
                         (dict(kpoints=SimpleNamespace(gamma_only=True)), "gamma-only")):
        with pytest.raises(NotImplementedError, match=word):
            refuse_regime(SimpleNamespace(**{**vars(plain), **change}), (nc,), "a test")
    us = SimpleNamespace(is_paw=False, is_ultrasoft=True, element="C", functional="SLA-PW-PBX-PBC")
    paw = SimpleNamespace(is_paw=True, is_ultrasoft=True, element="O", functional="SLA-PW-PBX-PBC")
    with pytest.raises(NotImplementedError, match="ultrasoft dataset C"):
        refuse_regime(plain, (us,), "a test")
    with pytest.raises(NotImplementedError, match="PAW dataset O"):
        refuse_regime(plain, (paw,), "a test")
    with pytest.raises(NotImplementedError, match="meta-GGA"):
        refuse_regime(SimpleNamespace(**{**vars(plain), "input_dft": "tb09"}), (nc,), "a test")


# --- the Davidson precondition hook -----------------------------------------------------

NDIM, NBND = 40, 4


class _Diagonal:
    """A dense diagonally dominant ``H`` with no overlap, as the solver duck-types it."""

    gamma_only = False
    has_overlap = False

    def __init__(self):
        rng = np.random.default_rng(3)
        coupling = rng.normal(size=(NDIM, NDIM)) + 1j * rng.normal(size=(NDIM, NDIM))
        kinetic = 2.0 * np.arange(NDIM, dtype=DOUBLE.real)
        self.h = (np.diag(kinetic) + 0.02 * (coupling + coupling.conj().T)).astype(DOUBLE.complex)
        self.ndim = self.space = NDIM
        self.dtype = DOUBLE.complex
        self.state_mask = jnp.ones((1, NDIM), dtype=bool)
        self.state_kinetic = jnp.asarray(kinetic[None, :])
        self._h = jnp.asarray(self.h)
        self._diagonal = jnp.asarray(np.diag(self.h).real)

    def diagonal(self, ik):
        return self._diagonal

    def overlap_diagonal(self, ik):
        return jnp.ones(NDIM)

    def apply_projected(self, psi, ik):
        empty = jnp.zeros((psi.shape[0], 0), dtype=psi.dtype)
        return psi @ self._h.T, empty, empty

    def s_projections(self, vectors, ik):
        empty = jnp.zeros((vectors.shape[0], 0), dtype=vectors.dtype)
        return empty, empty

    def s_correction(self, becq, ik):
        return jnp.zeros((becq.shape[0], NDIM), dtype=self.dtype)


class _Explicit(_Diagonal):
    """The default rule, called through the hook."""

    def precondition(self, residual, energies, ik, diagonal=None, overlap_diagonal=None):
        from defuqumat.solvers.davidson import _precondition

        return _precondition(residual, diagonal, overlap_diagonal, energies)


class _Identity(_Diagonal):
    """No preconditioner at all: the hook is live if the step count moves."""

    def precondition(self, residual, energies, ik, diagonal=None, overlap_diagonal=None):
        return residual


def _solve(operator):
    from defuqumat.solvers.davidson import davidson_eigensolver

    values, _, steps, unsettled = davidson_eigensolver(operator, 0, NBND, return_steps=True,
                                                       ethr=1e-10, max_iterations=200)
    return np.asarray(values), int(np.asarray(steps)), int(np.asarray(unsettled))


def test_the_precondition_hook_is_the_old_rule_and_is_live():
    exact = np.linalg.eigvalsh(_Diagonal().h)[:NBND]
    base, steps_base, unsettled = _solve(_Diagonal())
    explicit, steps_explicit, _ = _solve(_Explicit())
    identity, steps_identity, unsettled_identity = _solve(_Identity())
    assert unsettled == 0 and unsettled_identity == 0
    np.testing.assert_array_equal(explicit, base)
    assert steps_explicit == steps_base
    np.testing.assert_allclose(base, exact, atol=1e-8)
    np.testing.assert_allclose(identity, exact, atol=1e-8)
    assert steps_identity != steps_base, (steps_identity, steps_base)
