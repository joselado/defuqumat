"""The exchange-correlation kernels of the Dyson equation, behind a name.

``PLAN.md`` P37. Every kernel here returns the same object -- the **symmetrised**
kernel

    F = v^{-1/2} f_xc v^{-1/2},

dimensionless, ``(nw, nm, nm)``, on :class:`~defuqumat.tddft.chi0.ChiZero`'s
index set (three head directions then the body). Symmetrised because
:mod:`defuqumat.tddft.chi0` stores ``v^{1/2} chi_0 v^{1/2}`` and the two have to
meet: ``genvfxc.f90`` builds exactly this array and calls it
``v^-1/2 f_xc v^-1/2``.

``rpa``
    ``f_xc = 0``. Not a kernel so much as the absence of one, and the reference
    every other entry here is read against -- a spectrum with an excitonic peak
    means nothing until the same code with this kernel is shown not to have one.
``alda``
    the adiabatic local density approximation: ``f_xc(r, r') = dv_xc/drho
    delta(r - r')``, which in reciprocal space is a matrix depending on
    ``G - G'`` alone. **Its head and wings are zero** -- ``f_xc`` is finite at
    ``q = 0`` where ``v`` diverges, so ``F = f_xc / v`` vanishes there, which is
    Elk's ``vfxc(1:3,...) = 0`` and is the reason ALDA cannot bind an exciton
    however large it is. It is one ``jvp`` of the exchange-correlation potential
    this code already writes down (rule D1), not a transcription of
    ``setup_dmuxc``.
``lrc``
    the long-range correction of Reining, Olevano, Rubio and Onida
    (PRL **88**, 066404 (2002)) and Botti *et al.* (PRB **69**, 155112 (2004)):
    ``f_xc = -alpha / q^2``, so ``F = -alpha / 4 pi`` on the diagonal and
    nothing else. Two lines, one empirical parameter, and the kernel the
    bootstrap exists to produce without one.
``bootstrap``
    Sharma, Dewhurst, Sanna and Gross (PRL **107**, 186401 (2011)); Elk's
    ``fxctype = 210``:

        f_xc^BS = - eps^-1(q, 0) v(q) / (eps_0(q, 0) - 1),

    which symmetrised is ``F = eps^-1(omega = 0) / X_00(omega = 0)``, with
    ``eps^-1`` taken from the Dyson equation this kernel is then fed back into.
    It is **self-consistent** and **static**: one matrix, reused at every
    frequency.
``bootstrap-1``
    Elk's ``fxctype = 211``: the same expression evaluated once, from the RPA
    ``eps^-1``, without iterating. Byun and Ullrich call this family's members
    the "0-bootstrap" and "RPA-bootstrap" kernels and find they differ from the
    self-consistent one by about 10% in ``alpha``.

**The head convention.** In the optical limit ``eps_0``'s head is a 3x3 block,
and the scalar the bootstrap divides by is its **trace over three** --
``genvfxc.f90``'s ``z1 = (eps0(1,1) + eps0(2,2) + eps0(3,3))/3``. On a cubic
crystal that is the same as any diagonal entry; on a uniaxial one it is not, and
it is what makes the kernel a scalar rather than a tensor.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from defuqumat.basis.fft import g_to_r, r_to_g
from defuqumat.units import FPI

__all__ = [
    "get_kernel",
    "kernel_names",
    "register_kernel",
    "XCKernel",
    "DEFAULT_KERNEL",
]

DEFAULT_KERNEL = "bootstrap"

#: Below this absolute magnetization (electrons a cell) an ``nspin = 2``
#: density counts as unpolarized for the ``alda`` kernel. A run started at zero
#: moment keeps its two channels equal to round-off, and a polarized one
#: carries a moment of order one, so the value sits between the two rather
#: than at either.
ALDA_UNPOLARIZED_MOMENT = 1.0e-6

_KERNELS: dict = {}


class XCKernel:
    """A kernel of the Dyson equation: how to build ``F``, and whether to iterate.

    ``build(chi, epsi, context)`` returns ``(nw, nm, nm)``. ``epsi`` is the
    inverse dielectric matrix from the previous pass, or ``None`` on the first
    one, and only a self-consistent kernel reads it.
    """

    def __init__(self, name, build, *, self_consistent=False, iterations=None,
                 static=False, description=""):
        self.name = name
        self.build = build
        #: Whether the kernel has to be rebuilt from its own answer until it
        #: stops moving. ``genvfxc``'s 210 against 211.
        self.self_consistent = self_consistent
        #: A fixed number of passes for a kernel that is neither -- 211 is one.
        self.iterations = iterations
        #: Whether ``F`` is the same matrix at every frequency and is built
        #: from the ``static_index`` slice alone -- true of every kernel
        #: registered here. :func:`~defuqumat.tddft.dyson.solve_dyson` then
        #: iterates on that one frequency and screens the rest once, a chunk at
        #: a time. ``False`` is the safe default for a new kernel: it is built
        #: over the whole axis, as before.
        self.static = static
        self.description = description

    def __repr__(self) -> str:  # pragma: no cover -- diagnostics
        return f"XCKernel({self.name!r}, self_consistent={self.self_consistent})"


def register_kernel(kernel: XCKernel) -> None:
    _KERNELS[kernel.name.lower()] = kernel


def get_kernel(name: str | None) -> XCKernel:
    key = (name or DEFAULT_KERNEL).lower()
    if key not in _KERNELS:
        raise ValueError(
            f"unknown exchange-correlation kernel {name!r}; available: "
            f"{sorted(_KERNELS)}"
        )
    return _KERNELS[key]


def kernel_names() -> tuple[str, ...]:
    return tuple(sorted(_KERNELS))


# --- rpa ---------------------------------------------------------------------

def _rpa(chi, epsi, context):
    return jnp.zeros_like(chi.x)


register_kernel(XCKernel(
    "rpa", _rpa, static=True,
    description="f_xc = 0: the random phase approximation",
))


# --- lrc ---------------------------------------------------------------------

def _lrc(chi, epsi, context):
    """``F = -alpha / 4 pi`` on the diagonal, head included.

    ``genvfxc.f90``'s ``fxctype = 200`` without its ``beta omega^2`` term, which
    is a dynamic extension nothing here asks for. ``alpha`` is dimensionless in
    the sense that matters: ``F`` is ``f_xc / v`` and so carries no units, which
    is why the literature's values may be used as they stand whether the code
    writes ``v = 4 pi / q^2`` or ``8 pi / q^2``.
    """
    alpha = context.get("alpha")
    if alpha is None:
        raise ValueError(
            "the 'lrc' kernel needs its parameter: pass alpha = ... . It is "
            "material-dependent and empirical, which is the whole reason the "
            "bootstrap kernel exists"
        )
    nw, nm, _ = chi.x.shape
    diagonal = jnp.full((nm,), -float(alpha) / FPI, dtype=chi.x.dtype)
    return jnp.broadcast_to(jnp.diag(diagonal), (nw, nm, nm))


register_kernel(XCKernel(
    "lrc", _lrc, static=True,
    description="f_xc = -alpha/q^2, the empirical long-range correction",
))


# --- alda --------------------------------------------------------------------

def _alda(chi, epsi, context):
    """``F`` for the adiabatic LDA: a body matrix in ``G - G'``, no head.

    The kernel itself comes from one ``jvp`` of
    :func:`~defuqumat.scf.potential.exchange_correlation` with a tangent of ones.
    That is exact where ``v_xc`` is a pointwise function of the density -- which
    is what "adiabatic **local** density approximation" means. A
    gradient-corrected functional is not truncated to its local part: its
    adiabatic kernel is built a column at a time with the gradient terms in
    (:func:`_adiabatic_gga_matrix`, I97), under the same name, since it is the
    same approximation (adiabatic, ground-state functional) one rung up.
    """
    matrix = context["alda_matrix"]  # (nbody, nbody), built once
    nw, nm, _ = chi.x.shape
    full = jnp.zeros((nm, nm), dtype=chi.x.dtype).at[3:, 3:].set(matrix)
    return jnp.broadcast_to(full, (nw, nm, nm))


register_kernel(XCKernel(
    "alda", _alda, static=True,
    description="the adiabatic local density approximation",
))


def alda_matrix(calculation, density, sphere) -> jnp.ndarray:
    """``f_xc(G, G') / sqrt(v_G v_G')`` over the response sphere's body.

    Built once per run and handed to :func:`_alda` through the context, because
    it depends on the ground state and not on the frequency or on the iteration.
    """
    from defuqumat.scf.potential import exchange_correlation

    functional = calculation.functional
    density = jnp.asarray(density)
    nspin = int(density.shape[0])
    if functional.is_meta:
        raise NotImplementedError(
            f"the 'alda' kernel with the {functional.name} functional is not "
            "implemented: a meta-GGA's potential depends on the kinetic energy "
            "density, which comes from the states and not from the density, so "
            "dv_xc/drho at fixed tau is not its kernel. The tau response is the "
            "missing term"
        )
    # A belt for a direct call: a nonmagnetic spinor reaches here with a
    # one-component density, and a textured one is refused upstream by
    # ``require_a_sum_over_states_regime``.
    if functional.is_gradient and nspin not in (1, 2):
        raise NotImplementedError(
            f"the 'alda' kernel with the {functional.name} functional on a "
            f"{nspin}-component density is not implemented: the noncollinear "
            "gradient correction rotates onto |m|, and with a moment the kernel "
            "is a 4x4 tensor in (n, m) that a charge-only chi_0 cannot use. The "
            "adiabatic GGA kernel is built for a one-component density and an "
            "unpolarized nspin = 2 one only"
        )

    cell = calculation.system.cell
    refuse_a_polarized_alda(density, cell)
    if functional.is_gradient:
        return _adiabatic_gga_matrix(calculation, density, sphere)

    def potential(rho):
        return exchange_correlation(rho, cell, calculation.rho_core, functional)[0]

    # A pointwise kernel's ``jvp`` along a tangent of ones *is* the kernel:
    # ``dv_xc/drho`` at every grid point, in one pass rather than one per point.
    # At ``nspin = 2`` the channels are ``(up, down)`` and a charge
    # perturbation of an unpolarized density goes half into each, so the
    # tangent is ``1/2`` per channel and ``kernel_r[0]`` is
    # ``(f_upup + f_updn)/2 = dv_up/dn`` at fixed ``m``, the ``nspin = 1``
    # kernel exactly. A tangent of ones there would double it.
    tangent = jnp.full_like(density, 1.0 / nspin)
    _, kernel_r = jax.jvp(potential, (density,), (tangent,))

    # **On the dense box, not the smooth one.** The density and the potential
    # live on the dense grid, so that is the box this field is sampled on, while
    # the response sphere was *selected* on the smooth one -- which is why it
    # carries signed Miller indices rather than a flat index into one particular
    # box. This is :func:`~defuqumat.basis.fft.r_to_g` without its gather,
    # because the index below is a difference ``G - G'`` and reaches beyond
    # either sphere.
    grid = np.asarray(calculation.basis.dense.grid)
    kernel_g = jnp.fft.fftn(kernel_r[0]) / float(np.prod(grid))

    # ``f_xc(G, G') = ktilde(G - G')``: the Miller difference, folded into the
    # box the way every other G index here is.
    body = np.asarray(sphere.miller)
    index = (body[:, None, :] - body[None, :, :]) % grid
    matrix = kernel_g[index[..., 0], index[..., 1], index[..., 2]]

    coulomb = sphere.sqrt_coulomb
    return matrix / (coulomb[:, None] * coulomb[None, :])


def _xc_potential(rho, gvectors, cell, functional, rho_core, rho_core_g):
    """``v_xc(rho)``, the whole exchange-correlation potential and nothing else.

    :func:`~defuqumat.scf.potential.v_of_rho` with its Hartree term taken off
    again, so that the gradient correction is the one the SCF and the
    Sternheimer stack use (``gradcorr`` with the core charge on both grids),
    not a second copy of it. Hartree is linear in the density, so subtracting
    it changes the tangent by exactly ``v_H[drho]`` and nothing else.
    """
    from defuqumat.scf.potential import hartree, total_charge, v_of_rho

    total = v_of_rho(rho, gvectors, cell, rho_core, functional, rho_core_g).v_scf
    rho_g = jax.vmap(r_to_g, in_axes=(0, None))(rho, gvectors.fft_index)
    v_g, _ = hartree(total_charge(rho_g), gvectors, cell)
    v_h = jnp.real(g_to_r(v_g, gvectors.fft_index, gvectors.grid))
    return total - v_h[None]


def _xc_potential_of(calculation):
    """:func:`_xc_potential` with a calculation's grid, functional and core charge."""
    def potential(rho):
        return _xc_potential(rho, calculation.basis.dense, calculation.system.cell,
                             calculation.functional, calculation.rho_core,
                             calculation.rho_core_g)

    return potential


def _kernel_columns(potential, density, miller, grid):
    """``f_xc(G, G')`` for every ``G'`` of ``miller``, one ``jvp`` per column.

    The adiabatic kernel of a functional that is not pointwise is not a
    function of ``G - G'``, so it cannot come from one transform of one field
    as ALDA's does. What is still true for any functional is that the kernel is
    the linear map ``drho -> dv_xc``, so its column ``G'`` is that map applied
    to ``e^{i G'.r}``,

        f_xc(G, G') = (1/N) sum_r e^{-i G.r} [dv_xc/drho](e^{i G'.r})(r),

    which is the definition ALDA's matrix satisfies too: for a pointwise
    ``f(r)`` the bracket is ``f(r) e^{i G'.r}`` and the sum is ALDA's
    ``ktilde(G - G')`` (``test_the_column_route_is_alda_for_a_local_functional``
    holds the two to round-off). ``v_xc`` is real in a real density, so the
    complex tangent is its cosine and sine taken as two real ``jvp``.

    The tangent is ``1/nspin`` in each channel, as in :func:`alda_matrix`: a
    charge perturbation of an unpolarized density goes half into each, and the
    column read off channel 0 is ``dv_up/dn`` at fixed ``m``.
    """
    nspin = density.shape[0]
    shape = tuple(int(n) for n in grid)
    axes = [jnp.arange(n, dtype=density.dtype) / n for n in shape]
    frac = jnp.stack(jnp.meshgrid(*axes, indexing="ij"))  # (3, n1, n2, n3)
    body = jnp.asarray(miller)
    index = jnp.mod(body, jnp.asarray(shape))

    def column(m):
        phase = 2.0 * jnp.pi * jnp.tensordot(m.astype(density.dtype), frac, axes=1)
        tangents = jnp.stack([jnp.cos(phase), jnp.sin(phase)]) / nspin
        tangents = jnp.broadcast_to(tangents[:, None], (2,) + density.shape)

        def one(t):
            return jax.jvp(potential, (density,), (t,))[1][0]

        dv_cos, dv_sin = jax.vmap(one)(tangents)
        response = jax.lax.complex(dv_cos, dv_sin)
        transform = jnp.fft.fftn(response) / float(np.prod(shape))
        return transform[index[:, 0], index[:, 1], index[:, 2]]

    # ``(nbody, nbody)`` with the column index leading; transposed below.
    return jax.lax.map(column, body).T


@jax.jit
def _gga_columns(density, miller, gvectors, cell, functional, rho_core, rho_core_g):
    """:func:`_kernel_columns` compiled once per shape, every array an argument.

    The density, the core charge and the sphere are arguments rather than
    constants of a closure, so a second density (or a second sphere of the same
    size) reuses the executable instead of compiling one with the first folded
    in. The grid is ``gvectors``'s static field, the functional's every field is
    static, so both are part of the signature.
    """
    def potential(rho):
        return _xc_potential(rho, gvectors, cell, functional, rho_core, rho_core_g)

    return _kernel_columns(potential, density, miller, gvectors.grid)


def _adiabatic_gga_matrix(calculation, density, sphere) -> jnp.ndarray:
    """The adiabatic kernel of a gradient-corrected functional, symmetrised.

    ``f_xc(G, G') / sqrt(v_G v_G')`` over the response sphere's body, with the
    gradient terms in: ``v_xc = v1 - div(v2 grad rho)`` differentiated as it
    stands, which is the same ``jvp`` of the same ``v_of_rho`` that the
    Sternheimer screening (``dv_of_drho``, :func:`~defuqumat.response.efield.
    _screening_kernel`) takes, so its static limit is the ``epsilon_infinity``
    of that solve once ``chi_0`` is converged in bands and in the response
    cutoff (I97). The head and wings are still zero in the optical limit:
    ``f_xc(q, q + G')`` stays finite as ``q -> 0`` while ``v^{-1/2}`` vanishes.

    Cost: ``2 nbody`` ``jvp`` of the potential on the dense grid, against one
    for ALDA, done one column at a time (``lax.map``), so the working set is two
    dense-grid tangents and their responses beside the ``(nbody, nbody)`` result.
    """
    matrix = _gga_columns(
        density, jnp.asarray(np.asarray(sphere.miller)), calculation.basis.dense,
        calculation.system.cell, calculation.functional, calculation.rho_core,
        calculation.rho_core_g,
    )
    coulomb = sphere.sqrt_coulomb
    return matrix / (coulomb[:, None] * coulomb[None, :])


def refuse_a_polarized_alda(density, cell) -> None:
    """Refuse the ``alda`` kernel on an ``nspin = 2`` density that carries a moment.

    Host-side, on the converged density, so that :func:`~defuqumat.workflows.
    tddft.run_absorption` can ask before it pays for the empty states. The
    density is ``(up, down)``; the moment is the absolute magnetization
    ``int |n_up - n_dn|``, as ``pw.x`` prints it.
    """
    density = np.asarray(density)
    if density.shape[0] != 2:
        return
    moment = float(np.abs(density[0] - density[1]).mean() * cell.volume)
    if moment > ALDA_UNPOLARIZED_MOMENT:
        raise NotImplementedError(
            f"the 'alda' kernel on a magnetized ground state (absolute "
            f"magnetization {moment:.3e}) is not implemented: with a "
            "moment the LSDA kernel is a 2x2 tensor in (n, m) and the "
            "charge couples to the magnetization through f_nm, so the "
            "Dyson equation is a 2x2 block one (Elk's tddftsplr.f90). "
            "chi_0 here is the charge-charge block alone. Use 'rpa', "
            "'lrc' or 'bootstrap', whose kernels act on the charge only"
        )


# --- bootstrap ---------------------------------------------------------------

def _bootstrap(chi, epsi, context):
    """``F = eps^-1(omega = 0) / X_00(omega = 0)``, the same matrix at every omega.

    ``genvfxc.f90``'s 210/211 branch. Written there as
    ``z1 = -1/(eps_0(1,1,1) - 1)`` with ``eps_0 = 1 - X``, so ``z1 = 1/X_00``;
    and the head, in the optical limit, is the trace over the three directions.

    On the first pass ``epsi`` is the initialisation ``tddftlr.f90`` uses --
    ``1 + X`` -- rather than the converged inverse of anything. That is the
    first-order expansion of ``eps^-1``, and it is where the paper's algorithm
    starts ("start by setting ``f_xc = 0``").
    """
    static = context["static_index"]
    x = chi.x[static]
    if epsi is None:
        epsi = jnp.eye(x.shape[-1], dtype=x.dtype) + x
    else:
        epsi = epsi[static]
    head = jnp.trace(x[:3, :3]) / 3.0
    kernel = epsi / head
    return jnp.broadcast_to(kernel, chi.x.shape)


register_kernel(XCKernel(
    "bootstrap", _bootstrap, self_consistent=True, static=True,
    description="the bootstrap kernel, iterated to self-consistency (Elk 210)",
))
#: **Two passes, not one.** ``tddftlr.f90``'s 211 branch increments its counter
#: *after* the first Dyson solve and rounds the loop again while ``it <= 1``, so
#: the kernel it ends on was built from the first pass's ``eps^-1`` rather than
#: from the initialisation. "Single iteration" counts the update, not the pass.
#: Byun and Ullrich's "0-bootstrap" is a third thing again -- one pass from the
#: *true* RPA ``eps^-1`` -- and is not this.
register_kernel(XCKernel(
    "bootstrap-1", _bootstrap, iterations=2, static=True,
    description="the bootstrap kernel, a single update (Elk 211)",
))
