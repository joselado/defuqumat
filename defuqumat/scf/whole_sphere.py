"""Gamma-only states put back on the whole sphere, for the consumers after the SCF.

``K_POINTS gamma`` stores one plane wave of each ``(G, -G)`` pair, because at
``k = 0`` a state can be chosen real and then ``c(-G) = conj(c(G))``
(:func:`~defuqumat.basis.gvectors._half_sphere`). The SCF consumes that storage
through three helpers that carry the rule ``2 Re(sum) - (the G = 0 term)``
(``g_to_r_gamma``, ``gamma_inner``/``calbec_gamma``, ``force_real_g0``). The
consumers that run *after* the SCF -- the projected density of states, the site
angular momenta, the sum-over-states ``chi_0`` and its transverse twin, the
tunnelling spectrum and the two tunnelling transmissions -- each contract the
states in a form of their own (an ``einsum`` against atomic projectors, a pair
density through ``g_to_r``, ``psi(r)`` at an arbitrary point, an exit-plane Gram
matrix), and each used to refuse the half sphere
(:func:`~defuqumat.basis.gvectors.refuse_gamma_storage`).

**The rule is carried here once, as data, rather than six times as
arithmetic.** The half sphere holds the whole sphere exactly, so the conjugate
half is written out -- ``c(G)`` where ``G`` is stored, ``conj(c(-G))`` where it
is not, ``G = 0`` once -- and the consumer's own full-sphere contraction is then
the ``2 Re(sum) - G0`` form by construction. Each of the six forms derived again
would be one more place for the ``G = 0`` term to be dropped, which is the
defect ``gamma_inner``'s docstring says happens in one place out of ten.

**What it costs is the factor of two the trick saves, for the consumer only.**
The SCF keeps the half sphere; the states handed to the consumer are
``(nspin, nk, nbnd, npwx_full)``, ``npwx_full = 2 npwx - 1``, which is the
working set the same consumer has on an explicit ``K_POINTS automatic 1 1 1 0 0
0`` run. The precedent is :meth:`~defuqumat.scf.driver.Calculation.at_kpoints`,
which gives the memory back for a sub-run on another k-set in the same way
(``_adopt_rebuilt_sphere``).

The map is aligned by **Miller index**, never by position in the two lists
(``CLAUDE.md``'s trap: neighbouring sets do not share a G ordering).
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np

from defuqumat.scf.streaming import is_host_store

__all__ = ["whole_sphere_calculation", "unfold_gamma_states"]


def whole_sphere_calculation(calculation):
    """The same calculation at an explicit ``k = 0`` on the whole sphere.

    Returns ``calculation`` itself when it does not store a half sphere.
    ``dataclasses.replace`` rather than a rebuilt ``KPoints``, for the reason
    :func:`~defuqumat.scf.driver._without_gamma_storage` gives: the weights of an
    ``nspin = 2`` set are already divided by ``degspin`` and ``spin_normalized``
    must survive. :meth:`~defuqumat.scf.driver.Calculation.at_kpoints` builds
    the plane waves without ``gamma_only`` and re-derives the three sphere
    fields, so what comes back is the whole sphere, sharing every
    k-independent array with ``calculation``.
    """
    if not getattr(calculation, "gamma_only", False):
        return calculation
    kpoints = dataclasses.replace(calculation.system.kpoints, gamma_only=False)
    moved = calculation.at_kpoints(kpoints)
    assert not moved.gamma_only and moved.fft_index_minus is None
    return moved


def _unfolding_map(half, full):
    """``(source, conjugate, mask)``, each ``(nk, npwx_full)``, host arrays.

    ``full[k, j] = c[k, source]`` or its conjugate where ``conjugate`` is set,
    and zero on the padding (``mask`` false).
    """
    miller_half = np.asarray(half.basis.planewaves.miller(half.basis.smooth))
    mask_half = np.asarray(half.basis.planewaves.mask)
    miller_full = np.asarray(full.basis.planewaves.miller(full.basis.smooth))
    mask_full = np.asarray(full.basis.planewaves.mask)
    nk, width = mask_full.shape
    source = np.zeros((nk, width), dtype=np.int64)
    conjugate = np.zeros((nk, width), dtype=bool)
    for k in range(nk):
        stored = {tuple(m): i for i, m in enumerate(miller_half[k])
                  if mask_half[k, i]}
        if tuple(miller_half[k, 0]) != (0, 0, 0):
            # ``gamma_inner`` and ``force_real_g0`` both read G = 0 at index 0;
            # a half sphere that does not start there is not the one the SCF
            # wrote.
            raise ValueError("the stored half sphere does not start at G = 0")
        found = 0
        for j in np.flatnonzero(mask_full[k]):
            m = tuple(int(v) for v in miller_full[k, j])
            if m in stored:
                source[k, j] = stored[m]
            else:
                source[k, j] = stored[tuple(-v for v in m)]
                conjugate[k, j] = True
            found += 1
        if found != 2 * len(stored) - 1:
            raise ValueError(
                f"the whole sphere has {found} plane waves at k-point {k} and the "
                f"half {len(stored)}, where the pairing needs 2 n - 1"
            )
    return source, conjugate, mask_full


def unfold_gamma_states(calculation, wavefunctions):
    """``(calculation, wavefunctions)`` on the whole sphere.

    ``wavefunctions`` is ``(..., nk, nbnd, npwx)`` on ``calculation``'s half
    sphere, a device array or a streamed host store; the states come back in
    the same kind of container, ``(..., nk, nbnd, npwx_full)``. Both are
    returned untouched when ``calculation`` does not store a half sphere, so a
    consumer calls this unconditionally.

    The gather is differentiable in the coefficients. ``Im c(0)`` is not
    imposed here: it is zero by ``force_real_g0`` on every state the SCF or a
    fixed-density run on this calculation wrote, and zeroing it here would hide
    a state that did not come from one.
    """
    if not getattr(calculation, "gamma_only", False):
        return calculation, wavefunctions
    full = whole_sphere_calculation(calculation)
    source, conjugate, mask = _unfolding_map(calculation, full)
    if is_host_store(wavefunctions):
        states = np.asarray(wavefunctions)
        k = np.arange(source.shape[0])[:, None]
        gathered = states[..., k, :, source].copy()          # (nk, npwx_full, ..., nbnd)
        gathered = np.moveaxis(gathered, (0, 1), (-3, -1))     # (..., nk, nbnd, npwx_full)
        gathered = np.where(conjugate[:, None, :], gathered.conj(), gathered)
        gathered = np.where(mask[:, None, :], gathered, 0)
        return full, np.ascontiguousarray(gathered.astype(states.dtype))
    states = jnp.asarray(wavefunctions)
    index = jnp.asarray(source)[:, None, :]                    # (nk, 1, npwx_full)
    gathered = jnp.take_along_axis(
        states, jnp.broadcast_to(index, states.shape[:-1] + (source.shape[1],)),
        axis=-1,
    )
    gathered = jnp.where(jnp.asarray(conjugate)[:, None, :], gathered.conj(), gathered)
    gathered = jnp.where(jnp.asarray(mask)[:, None, :], gathered,
                         jnp.zeros((), dtype=gathered.dtype))
    return full, gathered
