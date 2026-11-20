"""Potential-only meta-GGA exchange: Becke-Roussel, Becke-Johnson, Tran-Blaha.

This module is the one place in the package where the convention of `PLAN.md`
D1 -- *write down the energy, get the potential from ``jax.grad``* -- runs
backwards. The Tran-Blaha modified Becke-Johnson potential (TB09) is a
**potential**, not the derivative of anything: there is no ``E_x[rho]`` whose
functional derivative it is, and Tran and Blaha say so in the paper that
introduces it. What that costs is set out in :mod:`defuqumat.xc.functional` and
enforced there -- no variational total energy, and therefore no forces, no
stress, no relaxation and no response -- and what it buys is a band gap.

The chain is three functionals deep and each one is a correction to the one
before:

* **Becke-Roussel 89** (`J. Chem. Phys. 88, 2547 (1988)`; the potential is Eq.
  (25) of `Phys. Rev. A 39, 3761 (1989)`) models the exchange hole by a
  displaced exponential and fits its three parameters to ``rho``, ``grad rho``,
  ``lap rho`` and ``tau`` at each point. Fitting them means solving one
  nonlinear equation per grid point -- :func:`becke_roussel_x`, and the whole of
  the numerical difficulty here. The model is *exact* for a one-orbital density,
  which is the tolerance-free check this module is pinned by: on hydrogen's 1s
  it reproduces the exact Slater potential to 6e-13.
* **Becke-Johnson 06** (`J. Chem. Phys. 124, 221101 (2006)`) adds a term
  proportional to ``sqrt(2 tau / rho)`` to the Becke-Roussel potential, chosen
  so that the sum reproduces the exchange potential of the uniform electron
  gas. The *construction* is exact -- the added term evaluates there to
  ``+(1/2)(6 rho_s/pi)^(1/3)`` and the Slater potential to
  ``-(3/2)(6 rho_s/pi)^(1/3)``, which sum to ``v_x^LDA`` -- so the identity
  holds exactly to the extent that Becke-Roussel *is* the Slater potential
  there, which is to 6e-4 (see :data:`BR89_GAMMA`). It is this module's second
  test; the first is the hydrogen atom, where the model is exact.
* **Tran-Blaha 09** (`Phys. Rev. Lett. 102, 226401 (2009)`) puts a coefficient
  ``c`` in front of the Becke-Roussel term and ``(3c - 2)`` in front of the
  other, and fixes ``c`` from a *cell average* of ``|grad rho| / rho``. ``c = 1``
  gives Becke-Johnson back. This is the only ingredient here that is not a
  pointwise function of the density: the potential at one point depends on the
  density everywhere, which is why libxc declines to compute it and says so in
  the parameter's own description ("This parameter involves an average over the
  unit cell and must be calculated by the calling program").

**There is no Fortran to transcribe.** QE reaches TB09 only through libxc
(``XClib/dft_setting_routines.f90`` maps ``imeta = 3`` to libxc's 208), so the
reference followed here is libxc's own definition -- ``maple/mgga_vxc/
mgga_x_tb09.mpl`` and ``maple/mgga_exc/mgga_x_br89.mpl``, plus the bracketing
in ``src/mgga_x_br89.c``. Two things about QE's route are worth knowing, because
they mean a ``pw.x`` TB09 number is not the functional this module implements
and cannot be used as a reference for it:

* **QE passes a zero Laplacian.** ``XClib/xc_wrapper_mgga.f90`` declares
  ``lapl_rho`` "not used in QE" and sets it to zero before every libxc call.
  TB09 is flagged ``XC_FLAGS_NEEDS_LAPLACIAN`` and the Laplacian enters through
  ``Q``, which is where the whole Becke-Roussel fit comes from. In a plane-wave
  code the Laplacian is free -- it is ``-G^2 rho(G)`` -- so it is computed here.
* **QE never sets ``c``.** ``set_ext_params`` is called with libxc's *default*
  parameter list, and the default is ``c = 1``. So ``input_dft = 'tb09'`` in
  ``pw.x`` is Becke-Johnson, not Tran-Blaha. Both are available here, as
  ``BJ06`` and ``TB09``, precisely so the difference can be measured.

**Units.** Everything internal to this module is Hartree, because that is what
libxc's expressions are in and mixing the two is how a factor of two gets lost;
the ``E2`` that converts to the Rydberg the rest of the package uses is applied
once, at the boundary, in :func:`tb09_potential`. The kinetic energy density
``tau`` this module takes is the **Hartree** one,

    tau_sigma(r) = (1/2) sum_i f_i |grad psi_i(r)|^2,

which is QE's ``rho%kin_r / e2`` (``PW/src/v_of_rho.f90``, ``v_xc_meta``) and
libxc's convention both. :func:`defuqumat.scf.density.kinetic_energy_density`
accumulates the Rydberg one, as ``sum_band`` does, and the division happens here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.scipy.special

from defuqumat.units import E2

__all__ = [
    "becke_roussel_x",
    "becke_roussel_potential_hartree",
    "tb09_potential",
    "tb09_coefficient",
    "lmbj_integrand",
    "lmbj_coefficient_from_average",
    "lmbj_vacuum_g",
    "hold_vacuum_potential",
    "LMBJ_RHO_HOLD",
    "LMBJ_PARAMETERS",
    "LMBJ_SIGMA",
    "LMBJ_RS_THRESHOLD",
    "thomas_fermi_tau",
    "BR89_GAMMA",
    "TB09_ALPHA",
    "TB09_BETA",
    "BR89_MIN_Q",
    "META_RHO_THRESHOLD",
    "LMBJ_RHO_THRESHOLD",
    "TAU_FLOOR",
]

#: ``params_a_gamma`` as ``mgga_x_tb09.mpl`` sets it. Becke and Roussel's own
#: value is 1; 0.8 is what libxc registers as ``XC_MGGA_X_BR89`` and what the
#: TB09 definition inherits, so it is the one used here (``BR89_1``, with
#: gamma = 1, is the other registered variant and is not offered).
#:
#: **What 0.8 is, measured rather than looked up.** The ratio of ``v_x^BR`` to
#: the uniform gas's exact Slater potential is 1.02810 at gamma = 0.6, **0.99960
#: at 0.8** and 0.97446 at 1.0, so 0.8 is the uniform-gas fit to four digits and
#: the crossing is at 0.7997. That matters here beyond provenance: Becke-Johnson
#: adds a term chosen to make ``v_x^Slater + (...)`` equal ``v_x^LDA`` exactly in
#: that limit, so how well the identity holds *is* how well this gamma
#: reproduces the Slater potential -- 6.0e-4, and no member of the family does
#: better than about 4e-4. ``tests/unit/test_mgga.py`` pins both numbers.
BR89_GAMMA = 0.8

#: ``alpha`` (dimensionless) and ``beta`` (bohr^(1/2)) of the Tran-Blaha
#: coefficient, from the 2009 Letter and confirmed against the local-mBJ paper
#: `arXiv:1911.00368` which quotes them as "the original MBJ potential
#: (alpha = -0.012, beta = 1.023 bohr^(1/2), e = 0.5)".
TB09_ALPHA = -0.012
TB09_BETA = 1.023

#: ``br89_min_Q``: the reduced ``Q`` is pushed off zero by this much, keeping
#: its sign. ``Q = 0`` is ``x = 2``, where the nonlinear equation's right-hand
#: side is infinite and the implicit derivative is 0/0.
BR89_MIN_Q = 5.0e-13

#: Below this density the meta-GGA exchange potential is set to zero rather
#: than evaluated, and the Becke-Roussel solve floors the density at it. It is
#: the GGA gate of :mod:`defuqumat.xc.functional` (``rho_threshold_gga``), not
#: the LDA one: every ingredient here is a *ratio* to a power of the density --
#: ``x``, ``u`` and ``t`` all divide by ``rho^(4/3)`` or ``rho^(5/3)`` -- so the
#: low-density region is where this functional is most sensitive to the Fourier
#: noise of a truncated density, and it is exactly where a GGA already declines
#: to look. The local mBJ takes :data:`LMBJ_RHO_THRESHOLD` instead.
META_RHO_THRESHOLD = 1.0e-6

#: The local mBJ's gate on the potential, 1e-9 where TB09 and BJ06 keep 1e-6
#: (P182, the user's choice of 2026-11-18). It sits beneath
#: :func:`hold_vacuum_potential`, which blends the potential into the clean
#: tail's value below about 1e-4, centred on 1e-5, so in a cell with vacuum it
#: no longer decides anything.
#: The history, for whoever is tempted to move it: cut at 1e-6, the vacuum had a
#: well (the Becke-Johnson term tends to a positive constant in a tail, and the
#: cut dropped it to zero), and graphane's vacuum state at ``Gamma`` moved 1.53 eV
#: between 15 and 25 bohr of vacuum; evaluated down to 1e-9, the potential sat on
#: the density's noise floor, a 3.3 Ry barrier that only looked stable. **Not for
#: TB09**: with one ``c`` the lower gate lets more of the tail into the cell
#: average, and on the same sheet TB09's ``c`` went from 1.456 to 1.555 at 20 bohr
#: and ran away at 30 (2.39, no convergence in 100 iterations).
LMBJ_RHO_THRESHOLD = 1.0e-9

#: ``xc_reduced_floor`` in libxc's ``tb09_f``: what ``tau - alpha tau_W`` is
#: floored at before its square root is taken. ``alpha = 0`` for TB09 and BJ06,
#: so the quantity floored is ``tau`` itself, which a truncated plane-wave
#: expansion can make very slightly negative.
TAU_FLOOR = 1.0e-30

#: ``sqrt(5/12)/pi``: the coefficient of the Becke-Johnson term, which is fixed
#: by requiring that ``v_x^BJ`` reduce to ``v_x^LDA`` for a uniform gas.
_BJ_HEG = jnp.sqrt(5.0 / 12.0) / jnp.pi

#: How many bisection steps :func:`becke_roussel_x` takes. The bracket is at
#: most ``1/y`` wide (see the function), so 80 halvings put the root below
#: double precision for any ``y`` a physical density produces, and bisection is
#: chosen over Brent -- which is what libxc uses -- because it is branch-free
#: and fixed-length, which ``lax.fori_loop`` needs and Brent's bookkeeping is
#: not. The derivative does not come from these steps in any case; it is the
#: implicit one, attached as a ``custom_jvp``.
_BISECTION_STEPS = 80


def _br89_residual(x, y):
    """``x exp(-2x/3) - y (x - 2)``, the root of which is Becke-Roussel's ``x``.

    ``br89_x_Q`` of ``src/mgga_x_br89.c``, written as a polynomial in ``x``
    rather than as the ratio ``x exp(-2x/3)/(x - 2) = y`` it comes from, so that
    ``x = 2`` is an ordinary point rather than a pole. For large ``x`` the
    exponential underflows to zero and the residual becomes ``-y(x - 2)``, whose
    sign is still the right one -- so no guard is needed.
    """
    return x * jnp.exp(-2.0 * x / 3.0) - y * (x - 2.0)


@jax.custom_jvp
def becke_roussel_x(q_reduced: jnp.ndarray) -> jnp.ndarray:
    """The Becke-Roussel ``x``: the root of ``x e^(-2x/3)/(x - 2) = y``.

    Args:
        q_reduced: ``Q / rho_sigma^(5/3)``, already pushed off zero.

    with ``y = (2/3) pi^(2/3) / Q``. Which root is wanted is decided by the
    sign, and libxc's brackets are used unchanged (``mgga_x_br89.c``):

    * ``Q > 0`` -- so ``y > 0`` -- puts it in ``(2, 2 + 1/y)``. The residual is
      ``2 e^(-4/3) > 0`` at the left end whatever ``y`` is, and negative at the
      right end for every positive ``y``, so the bracket is always valid.
    * ``Q < 0`` puts it in ``(0, 2)``, where the residual is ``2y < 0`` at the
      left end and again ``2 e^(-4/3) > 0`` at the right.

    The two brackets have *opposite* orientation -- the residual falls through
    zero in one and rises through it in the other -- so the bisection tracks the
    sign at the left end rather than assuming one.

    **The derivative is not the bisection's.** A bisection is piecewise constant
    in its argument and its tangent is zero; the ``custom_jvp`` below carries the
    implicit derivative of the defining equation instead, which is what libxc's
    ``diff/br89_x`` does in Maple.
    """
    q_reduced = jnp.asarray(q_reduced)
    y = (2.0 / 3.0) * jnp.pi ** (2.0 / 3.0) / q_reduced

    positive = y > 0.0
    low = jnp.where(positive, 2.0, 0.0)
    # ``1/y`` and not ``|1/y|``: the ``Q < 0`` branch does not use it.
    high = jnp.where(positive, 2.0 + jnp.where(positive, 1.0 / y, 1.0), 2.0)
    sign_low = jnp.sign(_br89_residual(low, y))

    def step(_, bracket):
        low, high = bracket
        middle = 0.5 * (low + high)
        keep_left = jnp.sign(_br89_residual(middle, y)) == sign_low
        return (jnp.where(keep_left, middle, low), jnp.where(keep_left, high, middle))

    low, high = jax.lax.fori_loop(0, _BISECTION_STEPS, step, (low, high))
    return 0.5 * (low + high)


@becke_roussel_x.defjvp
def _becke_roussel_x_jvp(primals, tangents):
    """``dx/dQ = -(2/3) pi^(2/3) / (Q^2 f'(x))``, libxc's ``diff/br89_x``.

    ``f`` here is the *ratio* form ``x e^(-2x/3)/(x - 2)``, whose derivative
    ``br89_aux_dfdx`` is written down in ``mgga_x_br89.mpl``. Both factors blow
    up as ``x -> 2`` and the blow-ups cancel; what keeps the quotient finite in
    floating point is that the caller has already clamped ``|Q|`` to
    :data:`BR89_MIN_Q`, which holds ``x - 2`` at about 4e-13 and ``f'`` at about
    -1e25 -- large, but nowhere near overflow.
    """
    (q_reduced,), (dq,) = primals, tangents
    x = becke_roussel_x(q_reduced)
    dfdx = (
        -2.0 / 3.0
        * jnp.exp(-2.0 * x / 3.0)
        * (x * x - 2.0 * x + 3.0)
        / (x - 2.0) ** 2
    )
    dx_dq = -2.0 / 3.0 * jnp.pi ** (2.0 / 3.0) / (q_reduced**2 * dfdx)
    return x, dx_dq * dq


def _clamp_q(q_reduced):
    """``br89_cQ``: push ``Q`` off zero without changing its sign."""
    return jnp.where(
        jnp.abs(q_reduced) < BR89_MIN_Q,
        jnp.where(q_reduced > 0.0, BR89_MIN_Q, -BR89_MIN_Q),
        q_reduced,
    )


def becke_roussel_potential_hartree(rho, sigma, laplacian, tau, gamma=BR89_GAMMA,
                                    threshold=META_RHO_THRESHOLD):
    """``v_x^BR`` for one spin channel, in **Hartree**.

    Args:
        rho: ``rho_sigma``, the channel density -- half the total when
            unpolarized.
        sigma: ``|grad rho_sigma|^2``.
        laplacian: ``lap rho_sigma``.
        tau: ``tau_sigma = (1/2) sum_i f_i |grad psi_i|^2``, Hartree.
        gamma: Becke and Roussel's ``gamma``.

    The expression is libxc's ``br89_v``, which is Becke and Roussel's

        v_x^BR = -(1/b) [1 - e^(-x) - (x/2) e^(-x)],
        b = [x^3 e^(-x) / (8 pi rho)]^(1/3),

    with ``1/b`` folded in as ``(8 pi rho)^(1/3) e^(x/3) / x``. The bracket is
    evaluated as ``(1 + x/2)(1 - e^(-x))/x - 1/2`` rather than literally: the
    literal form is ``1`` minus something that tends to ``1`` as ``x -> 0`` and
    loses every significant digit there, while the rewritten one is a sum of two
    O(1) pieces and tends to ``1/2`` cleanly. That is libxc's own rewriting and
    the comment in ``mgga_x_br89.mpl`` explains it in the same terms.

    ``Q`` is built in reduced variables, exactly as ``br89_Q`` does:

        x_red = |grad rho| / rho^(4/3),  u_red = lap rho / rho^(5/3),
        t_red = tau / rho^(5/3),
        Q_red = (u_red - 4 gamma t_red + gamma x_red^2 / 2) / 6,

    which is ``Q = (lap rho - 2 gamma D)/6`` with ``D = 2 tau - |grad rho|^2/(4
    rho)`` divided through by ``rho^(5/3)``. Doing it in the reduced variables
    is not cosmetic: ``x`` is a function of ``Q_red`` alone, so the nonlinear
    solve never sees the density's dynamic range.
    """
    rho = jnp.asarray(rho)
    safe_rho = jnp.maximum(rho, threshold)
    rho_43 = safe_rho ** (4.0 / 3.0)
    rho_53 = safe_rho ** (5.0 / 3.0)

    x_red = jnp.sqrt(jnp.maximum(sigma, 0.0)) / rho_43
    u_red = laplacian / rho_53
    t_red = tau / rho_53
    q_red = _clamp_q((u_red - 4.0 * gamma * t_red + gamma * x_red**2 / 2.0) / 6.0)

    x = becke_roussel_x(q_red)
    # ``-expm1(-x) = 1 - e^(-x)``, and ``x`` is bounded away from zero by the
    # clamp on Q, so the division is safe.
    bracket = (1.0 + 0.5 * x) * (-jnp.expm1(-x)) / x - 0.5
    return -2.0 * (jnp.pi * safe_rho) ** (1.0 / 3.0) * jnp.exp(x / 3.0) * bracket


def tb09_potential(rho, sigma, laplacian, tau, c, gamma=BR89_GAMMA,
                   threshold=META_RHO_THRESHOLD):
    """``v_x^mBJ`` for one spin channel, in **Rydberg**.

        v_x^mBJ = c v_x^BR + (3c - 2)/pi sqrt(5/12) sqrt(2 tau_sigma/rho_sigma)

    Args:
        rho: ``rho_sigma``.
        sigma: ``|grad rho_sigma|^2``.
        laplacian: ``lap rho_sigma``.
        tau: ``tau_sigma``, **Hartree** -- see the module docstring.
        c: the Tran-Blaha coefficient, ``1`` for Becke-Johnson.

    ``c = 1`` collapses the prefactors to ``1`` and ``1``, which is Becke-Johnson
    06; ``c = 2/3`` would switch the second term off entirely and leave a scaled
    Becke-Roussel. ``c`` is one number for TB09 and BJ06, the cell average of
    :func:`tb09_coefficient`, and a field on the same grid as ``rho`` under the
    local mBJ (``input_dft = 'lmbj'``, :func:`lmbj_coefficient_from_average`);
    the expression is the same pointwise in both.

    **Zero below the density threshold**, and the points below it are evaluated
    on substitute values first, so that ``grad`` sees no singular arithmetic
    through the ``where``. This is the two-sided sanitisation the GGA slots use
    (:func:`defuqumat.xc.functional._sanitise`) and it matters more here: the
    linear-response Jacobian of an SCF under this functional differentiates the
    potential, and a NaN at one vacuum point poisons the whole solve.
    """
    rho = jnp.asarray(rho)
    active = rho > threshold
    safe_rho = jnp.where(active, rho, 1.0)
    safe_sigma = jnp.where(active, sigma, 0.0)
    safe_laplacian = jnp.where(active, laplacian, 0.0)
    safe_tau = jnp.where(active, tau, 1.0)

    v_br = becke_roussel_potential_hartree(
        safe_rho, safe_sigma, safe_laplacian, safe_tau, gamma, threshold
    )
    kinetic_term = jnp.sqrt(2.0 * jnp.maximum(safe_tau, TAU_FLOOR) / safe_rho)
    v_hartree = c * v_br + (3.0 * c - 2.0) * _BJ_HEG * kinetic_term
    return jnp.where(active, E2 * v_hartree, 0.0)


#: ``|grad rho|^2`` below which :func:`tb09_coefficient` treats a point as the
#: apex of the cone ``|grad rho|``. Round-off on a symmetry-forced zero is
#: about 1e-17 in ``|grad rho|`` on silicon, so 1e-24 in the square; 1e-20
#: keeps a margin above that and drops ``|grad rho| < 1e-10``, a contribution
#: to the cell average below any number the run reports.
CONE_SIGMA = 1.0e-20


def tb09_coefficient(rho, grad_rho, threshold=META_RHO_THRESHOLD):
    """``c = alpha + beta sqrt(<|grad rho| / rho>)``, the cell average.

    Args:
        rho: the **total** density on the real-space grid, any shape.
        grad_rho: its gradient, ``(3,) + rho.shape``.

    Tran and Blaha's Eq. (3): the average is over the unit cell,

        c = alpha + beta [ (1/V) int_cell |grad rho(r')| / rho(r') d3r' ]^(1/2),

    and since the grid is uniform the volume element cancels between the
    integral and the ``1/V``, leaving a plain mean over grid points. ``alpha``
    is dimensionless and ``beta`` carries bohr^(1/2), so ``c`` is dimensionless
    and is *not* invariant under a change of cell size at fixed structure -- it
    is a property of the material, which is the whole idea.

    **The integrand is gated, and it has to be.** ``|grad rho|/rho`` is finite in
    a real exponential tail, where it tends to twice the decay constant, but a
    plane-wave density is a truncated Fourier series that oscillates about zero
    in vacuum, and there the ratio is unbounded and signless. Points below
    ``threshold`` are dropped from the average -- from the numerator *and* from
    the count, so what is returned is the mean over the region the functional
    actually acts on. For a bulk crystal, which is what TB09 is for, the dropped
    fraction is a few parts in a thousand and the value of ``c`` moves in the
    fourth decimal; for an isolated atom in a large box it is most of the cell
    and the number would otherwise be meaningless.
    """
    rho = jnp.asarray(rho)
    # ``|grad rho|`` is a cone at ``grad rho = 0``, and a crystal's group puts
    # that zero on grid points (the atom sites of diamond silicon). The tangent
    # there is taken as zero, which is what a central difference reads at the
    # apex, through a two-sided ``where`` so ``sqrt``'s infinite slope is never
    # evaluated (I120: with the raw ``sqrt`` a direction off the crystal's
    # symmetry read ``dc`` 0.0091 against the difference's 0.0035, and ``dV``
    # 8.3e-4 off).
    sigma = jnp.sum(jnp.asarray(grad_rho) ** 2, axis=0)
    finite = sigma > CONE_SIGMA
    magnitude = jnp.where(finite, jnp.sqrt(jnp.where(finite, sigma, 1.0)), 0.0)
    active = rho > threshold
    ratio = jnp.where(active, magnitude / jnp.where(active, rho, 1.0), 0.0)
    count = jnp.maximum(jnp.sum(active), 1)
    average = jnp.sum(ratio) / count
    return TB09_ALPHA + TB09_BETA * jnp.sqrt(jnp.maximum(average, 0.0))


#: The local mBJ of Rauch, Marques and Botti (J. Chem. Theory Comput. 16, 2654
#: (2020), arXiv:1911.00368): ``c`` becomes a field, ``c(r) = alpha + beta
#: gbar(r)^e`` with ``gbar`` a Gaussian average of ``|grad rho|/rho`` (their
#: Eqs. 5 to 7). Two parameter sets, by the user's choice of 2026-11-18:
#:
#: * ``'tb09'`` (the default): Tran and Blaha's own ``alpha``, ``beta`` and
#:   ``e = 1/2``. In a bulk crystal ``gbar`` is uniform to four digits at the
#:   paper's ``sigma`` (the Gaussian kills every ``G != 0`` of a cell of a few
#:   bohr), so the local form *is* this code's TB09 there, and only the vacuum
#:   and the interfaces see the difference.
#: * ``'koller'``: the paper's own choice, ``alpha = 0.488``, ``beta = 0.5``
#:   bohr and ``e = 1`` (Koller, Tran and Blaha, PRB 85, 155109 (2012)), for
#:   comparing with published local-mBJ numbers. Those were fitted on
#:   all-electron densities, whose ``gbar`` the core makes larger: on this
#:   code's norm-conserving silicon it gives ``c`` about 1.01 where VASP's PAW
#:   run gives 1.11 to 1.13, so bulk gaps come out lower than the paper's.
LMBJ_PARAMETERS: dict[str, tuple[float, float, float]] = {
    "tb09": (TB09_ALPHA, TB09_BETA, 0.5),
    "koller": (0.488, 0.5, 1.0),
}

#: The paper's smearing length, 3.78 bohr = 2 angstrom, "large enough to keep
#: the properties of the original MBJ potential in locally homogeneous regions".
LMBJ_SIGMA = 3.78

#: The paper's threshold Wigner-Seitz radius, ``rho_th = 3 / (4 pi rs^3)``.
#: 5 bohr is ``rho_th = 1.91e-3`` bohr^-3; on norm-conserving silicon 6e-5 of
#: the grid lies below it and on diamond (HGH) 1.4e-4, so it acts in the vacuum
#: and at a surface, not inside these crystals.
LMBJ_RS_THRESHOLD = 5.0


#: Where the local mBJ's potential stops being evaluated and is held at the
#: vacuum tail's value (P182, the user's choice of 2026-11-18, after the vacuum
#: check). A plane-wave density is an exponential tail only down to a noise
#: floor (1.6e-8 to 6.6e-8 in graphane's vacuum), and below it ``tau/n`` and
#: ``|grad n|/n`` are noise: evaluated there the vacuum potential sat 3.3 Ry
#: above PBE's, noisy, where the Becke-Johnson asymptote is
#: ``(1/pi) sqrt(5/12) kappa`` = 0.33 Ry. Held at its mean over the shell
#: ``LMBJ_RHO_HOLD < n < 10 LMBJ_RHO_HOLD``, where the tail is clean, the vacuum
#: is flat; the switch is an erf in ``ln n`` about a decade wide.
LMBJ_RHO_HOLD = 1.0e-5
LMBJ_HOLD_WIDTH = 0.5 * 2.302585092994046  # ln(10)/2: the switch spans about a decade


def hold_vacuum_potential(v, rho, rho_hold=LMBJ_RHO_HOLD, width=LMBJ_HOLD_WIDTH):
    """The meta potential blended into its clean-tail value where the density is low.

    Args:
        v: ``(nspin, ...)``, the channel potentials.
        rho: ``(nspin, ...)``, each channel's density **doubled**, ``2 n_sigma``,
            the density each channel's exchange is evaluated at; a field of
            ``v.shape[1:]`` is taken as the same for every channel.

        w(n) = [1 + erf(ln(n / rho_hold) / width)] / 2,   v_out = w v + (1 - w) v_tail,

    ``v_tail`` per channel the mean of ``v`` over ``rho_hold < n < 10 rho_hold``,
    with ``n`` that channel's own ``2 n_sigma`` in the shell and in the weight
    (audit 2026-11-20 #47): the channel potential is evaluated on the channel
    density, so its clean tail is where that density is clean. At ``nspin = 1``
    and on an unpolarized ``nspin = 2`` run ``2 n_sigma`` is the total, and the
    two regimes hold the same potential. ``w = 0`` where the density is not
    positive. ``w`` is 0.5 at ``rho_hold``, 0.73 at ``2 rho_hold`` and 0.998
    at ``10 rho_hold``, so the blend acts below about ``10 rho_hold``, centred
    on ``rho_hold``, not only below it. A channel with no point in its shell (a
    bulk crystal whose lowest density is above ``10 rho_hold``) keeps ``v`` as
    it is.
    """
    v = jnp.asarray(v)
    rho = jnp.broadcast_to(jnp.asarray(rho), v.shape)
    positive = rho > 0.0
    log_ratio = jnp.log(jnp.where(positive, rho, rho_hold) / rho_hold)
    weight = jnp.where(positive, 0.5 * (1.0 + jax.scipy.special.erf(log_ratio / width)), 0.0)
    shell = (rho > rho_hold) & (rho < 10.0 * rho_hold)
    axes = tuple(range(1, v.ndim))
    count = jnp.sum(shell, axis=axes, keepdims=True)
    v_tail = jnp.sum(jnp.where(shell, v, 0.0), axis=axes, keepdims=True) / jnp.maximum(count, 1)
    held = weight * v + (1.0 - weight) * v_tail
    return jnp.where(count > 0, held, v)


def lmbj_vacuum_g(alpha: float, beta: float, exponent: float) -> float:
    """The ``g`` at which ``c = 1``: ``((1 - alpha) / beta)^(1/e)``.

    The paper's ``(1 - alpha)/beta`` is this for ``e = 1``. ``c -> 1`` in
    vacuum is the condition, since it restores the asymptotic ``-1/r`` of
    Becke-Roussel and switches off the ``(3c - 2)`` term's excess.
    """
    return ((1.0 - alpha) / beta) ** (1.0 / exponent)


def lmbj_integrand(rho, grad_rho, rho_threshold, g_vacuum):
    """The paper's Eq. (7): ``g`` blended into ``g_vacuum`` below ``rho_th``.

        g(r) = g_vac [1 - erf(rho/rho_th)] + (|grad rho|/rho) erf(rho/rho_th)

    Args:
        rho: the total density on the grid.
        grad_rho: its gradient, ``(3,) + rho.shape``.

    The second term is written ``|grad rho| erf(x)/x / rho_th`` with ``x =
    rho/rho_th``, which is finite as ``rho -> 0`` (``erf(x)/x -> 2/sqrt(pi)``),
    so the ratio is never formed where it is not defined. A plane-wave density
    goes slightly negative in vacuum; there ``g = g_vac`` exactly, the paper's
    limit, through two-sided ``where``s that keep the tangent finite. ``|grad rho|`` has the same cone guard as
    :func:`tb09_coefficient`.
    """
    rho = jnp.asarray(rho)
    sigma = jnp.sum(jnp.asarray(grad_rho) ** 2, axis=0)
    finite = sigma > CONE_SIGMA
    magnitude = jnp.where(finite, jnp.sqrt(jnp.where(finite, sigma, 1.0)), 0.0)
    positive = rho > 0.0
    x = jnp.where(positive, rho, 0.0) / rho_threshold
    small = x < 1.0e-6
    safe_x = jnp.where(small, 1.0, x)
    erf_over_x = jnp.where(small, 2.0 / jnp.sqrt(jnp.pi) * (1.0 - x * x / 3.0),
                           jax.scipy.special.erf(safe_x) / safe_x)
    erf_x = x * erf_over_x
    # Where the density is not positive the ratio has no meaning (it is the
    # Fourier noise of the vacuum), so only the vacuum value is kept there.
    return g_vacuum * (1.0 - erf_x) + jnp.where(
        positive, magnitude * erf_over_x / rho_threshold, 0.0)


def lmbj_coefficient_from_average(gbar, alpha, beta, exponent):
    """``c(r) = alpha + beta gbar(r)^e``, with ``gbar`` already averaged.

    ``gbar`` is a Gaussian average of a non-negative field and is positive, but
    a truncated Fourier series can dip below zero by round-off where the field
    is flat; it is floored at zero before the power, as :func:`tb09_coefficient`
    floors its mean.
    """
    gbar = jnp.maximum(jnp.asarray(gbar), 0.0)
    if exponent == 1.0:
        return alpha + beta * gbar
    if exponent == 0.5:
        positive = gbar > 0.0
        return alpha + beta * jnp.where(positive, jnp.sqrt(jnp.where(positive, gbar, 1.0)), 0.0)
    return alpha + beta * gbar ** exponent


def thomas_fermi_tau(rho, nspin: int = 1):
    """The Thomas-Fermi guess for ``tau``, in **Rydberg**, per channel.

    ``potinit.f90``'s starting ``rho%kin_r``: ``(3/5)(3 pi^2)^(2/3) rho^(5/3)``
    for one channel, and for two the same expression applied to ``nspin rho_s``
    and divided by ``nspin``, which is the spin-scaling of the kinetic energy
    density (Eq. 2.9 of `Phys. Rev. A 20, 397`).

    An SCF under a meta-GGA needs a ``tau`` before it has any wavefunctions to
    build one from, and this is the guess QE uses for that first iteration. It
    is not used again: from the second iteration on, ``tau`` comes from the
    states, and it is *not* mixed -- ``mix_rho.f90`` does not touch ``kin_r``.
    """
    factor = (3.0 / 5.0) * (3.0 * jnp.pi**2) ** (2.0 / 3.0)
    rho = jnp.asarray(rho)
    if nspin == 1:
        return factor * jnp.abs(rho) ** (5.0 / 3.0)
    return factor * jnp.abs(rho * nspin) ** (5.0 / 3.0) / nspin
