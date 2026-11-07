"""The optical conductivity by a two-index Chebyshev expansion, with no bands computed.

The Kubo-Greenwood formula for the absorptive part of the conductivity,

    sigma'_ab(w) = (pi / (Omega w)) sum_k w_k int dE [f(E) - f(E + w)]
                   Tr[v_b delta(E - H) v_a delta(E + w - H)],

has no band index in it: the transition from ``E`` to ``E + w`` is weighted by
the velocity matrix elements between whatever states sit there. **The order of
``a`` and ``b`` is the band sum's**: the trace is ``sum <c|v_b|o><o|v_a|c>`` over
occupied ``o`` at ``E`` and empty ``c`` at ``E + w``, which is ``z_ab = <o|v_a|c>
<c|v_b|o>`` of :mod:`defuqumat.response.conductivity`, so the Hermitian tensor
here is the absorptive part of that one and an absorptive Hall part has its
sign. With ``a`` and ``b`` the other way round the diagonal is the same and the
antisymmetric part changes sign. Expanding both
deltas in Chebyshev polynomials of ``H' = (H - centre) / half``,

    delta(E - H) = sum_n g_n (2 - delta_n0) T_n(x) T_n(H') / (pi half sqrt(1 - x^2)),

turns the trace into the two-index moments

    mu^ab_nm = Tr[v_a T_n(H') v_b T_m(H')] = <T_n(H') v_a r | v_b T_m(H') r>,

averaged over random vectors ``r`` (or summed over the unit vectors of the
basis for the exact trace). For one ``r`` they are the matrix of inner products
between two families of vectors: the Chebyshev vectors started from ``v_a r``,
one recursion per direction, and the velocity applied to the Chebyshev vectors
of ``r`` itself. The cost per random vector and k-point is
``(1 + ndir) N`` applications of ``H``, ``ndir N`` of the velocity and a Gram
product of ``ndir^2 N^2 ndim`` flops, not ``N^2`` applications. Weisse et al.,
Rev. Mod. Phys. 78, 275 (2006), arXiv:cond-mat/0504627, section IV.A, is the
method; pyqula's ``kpmtk/kpm2d.py`` is where the two-index form was taken from.

**The velocity is the derivative of the Hamiltonian, not ``p``.** ``v_a = dH/dk_a``
at a frozen sphere, which carries ``[r, V_NL]`` and the Hubbard term, as in
:mod:`defuqumat.response.velocity`. Only three parts of ``H(k)`` move with ``k``:
``|k+G|^2``, the projectors and the Hubbard orbitals. Their tangents along each
direction are built once on the host by one ``jvp`` of
:meth:`~defuqumat.scf.driver.Calculation.at_kcart`
(:func:`velocity_tangents`), and inside the recursion ``v_a psi`` is the ``jvp``
of ``H psi`` in those parts alone (:func:`apply_velocity`). The local potential
is not an argument of that ``jvp``, so its FFTs have a zero tangent and the
primal ``H psi`` the ``jvp`` also returns is never used, which XLA removes. One
direction's tangent is one more projector table, ``nk npwx nkb`` complex
numbers, and the three are held at once.

**Memory.** Per k-point in flight, ``(ndir + 2) N ndim`` complex numbers: the
``ndir`` left families, the right family and its image under one velocity; the
random vectors are taken one at a time by a ``lax.scan``. The moments are summed
over k inside the walk, so the result is ``ndir^2 N^2`` complex numbers however
many k-points there are. At ``N = 1000`` and ``ndim = 3000`` in double
precision that is 240 MB per k-point in flight with three directions (and a
transient copy of one family when it is assembled), and 144 MB of moments.

**The broadening of a transition is that of two deltas.** With the Lorentz
kernel each delta is close to a Lorentzian of half width
``1.19 lambda half sqrt(1 - x^2) / N`` (:data:`~defuqumat.kpm.kernels.LORENTZ_HWHM`),
and two Lorentzians convolve to one of the summed widths. The Jackson kernel
gives a Gaussian per delta, and two give a Gaussian ``sqrt(2)`` times as wide.
:attr:`KPMConductivity.transition_broadening` is that, at the Fermi level.

**It is not the band sum at that broadening, and the difference is physics.**
The Kubo-Greenwood integral occupies by energy, ``f(E) - f(E + w)``, after the
deltas are broadened, while the band sum
:func:`~defuqumat.response.conductivity.optical_conductivity` occupies by band
and then broadens the transition. So the tails of a broadened state that cross
the Fermi level are cut here and kept there, and the factor is ``1/w`` here and
``1/e_mn`` there. The two agree as the broadening goes to zero (``PLAN.md``
P169 measures by how much at finite broadening); what is checked to round-off
is the pair sum over the eigenpairs in this form.

**The memory statement assumes stored projectors.** On a calculation that
rebuilds its projectors per k-point (the lazy set of ``memory`` mode), the
primal of the moving parts is taken from ``at_kcart``, which stores them, so the
velocity holds one ``nk npwx nkb`` table besides its tangents.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from defuqumat.batching import sum_k
from defuqumat.kpm.control import FILTER_DEGREE
from defuqumat.kpm.kernels import get_kernel
from defuqumat.kpm.moments import live_mask, random_phases, unit_vectors

#: Cartesian axes by name.
AXES = {"x": 0, "y": 1, "z": 2}


# -- the velocity --------------------------------------------------------------

def _moving(hamiltonian):
    """The parts of ``H(k)`` that carry ``k``: ``(kinetic, projectors, hubbard)``."""
    return hamiltonian.kinetic, hamiltonian.projectors, hamiltonian.hubbard


def _inexact(tree):
    return eqx.partition(tree, eqx.is_inexact_array)


def require_a_velocity(calculation, hamiltonians) -> None:
    """Refuse the regimes whose velocity the recursion does not carry."""
    if any(getattr(h, "berry", None) is not None for h in hamiltonians):
        raise NotImplementedError(
            "the KPM conductivity does not carry lelfield's H_E term: it couples "
            "neighbouring k-points and has no velocity at one k")
    if calculation.is_hubbard:
        # The tangent of the Hubbard orbitals is built (``hubbard_terms`` at the
        # moved k), but no test has compared it with the band sum's.
        raise NotImplementedError(
            "the KPM conductivity with a Hubbard U is not checked: the Hubbard orbitals "
            "live at k+G and carry a velocity of their own, and nothing has compared "
            "that term with get_optical_conductivity's. Use get_optical_conductivity")


def velocity_tangents(calculation, hamiltonians, directions, ns=None) -> tuple:
    """``(moving, tangents)``: the moving parts of each channel's ``H(k)`` and their tangents.

    ``directions`` is a sequence of cartesian 3-vectors (1/bohr). ``moving`` is a
    tuple over channels of ``(kinetic, projectors, hubbard)`` as
    :meth:`~defuqumat.scf.driver.Calculation.at_kcart` builds them, with the
    projector table stored even when the calculation keeps a lazy set (whose
    pytree a tangent of the stored one does not fit); ``tangents`` is a tuple
    over directions of tuples over channels of the inexact leaves of their
    tangent. Both go to :func:`apply_velocity`. ``ns`` is the converged Hubbard
    occupation matrix, whose orbitals move with ``k`` like the projectors do.
    """
    require_a_velocity(calculation, hamiltonians)
    # the k the arrays are at, as the velocity operator reads it (a strained
    # calculation keeps its own)
    kcart = getattr(calculation, "_kcart", None)
    if kcart is None:
        kcart = calculation.system.kpoints.cartesian(calculation.system.cell)
    kcart = jnp.asarray(kcart)
    real = hamiltonians[0].kinetic.dtype
    ns = None if ns is None else jnp.asarray(ns)

    def parts(kc):
        moved = calculation.at_kcart(kc)
        hubbard = None if ns is None else moved.hubbard_terms(ns)[2]
        return tuple((moved.kinetic.astype(real), moved.projectors,
                      None if hubbard is None else hubbard[channel])
                     for channel in range(len(hamiltonians)))

    moving = parts(kcart)

    def inexact_parts(kc):
        return tuple(_inexact(m)[0] for m in parts(kc))

    tangents = []
    for direction in directions:
        step = jnp.broadcast_to(jnp.asarray(direction, dtype=kcart.dtype), kcart.shape)
        _, tangent = jax.jvp(inexact_parts, (kcart,), (step,))
        tangents.append(tangent)
    return moving, tuple(tangents)


def apply_velocity(hamiltonian, moving, tangent, psi, ik):
    """``v psi = dH/dk psi`` at k-point ``ik``, for ``psi`` of shape ``(..., ndim)``.

    The ``jvp`` of ``H psi`` in the moving parts alone, with ``moving`` and
    ``tangent`` one channel's entries of :func:`velocity_tangents`.
    """
    primal, static = _inexact(moving)
    psi = jnp.where(live_mask(hamiltonian)[ik], psi, 0.0)

    def applied(leaves):
        kinetic, projectors, hubbard = eqx.combine(leaves, static)
        moved = eqx.tree_at(lambda h: (h.kinetic, h.projectors), hamiltonian,
                            (kinetic, projectors))
        if hubbard is not None:
            moved = eqx.tree_at(lambda h: h.hubbard, moved, hubbard)
        return moved.apply(psi, ik)

    return jax.jvp(applied, (primal,), (tangent,))[1]


# -- the moments ---------------------------------------------------------------

def _family(apply, start, n_moments: int):
    """``(N, ...)``: the Chebyshev vectors ``T_n(H') start`` for ``n < N``."""
    first = apply(start)

    def step(carry, _):
        previous, current = carry
        following = 2.0 * apply(current) - previous
        return (current, following), following

    _, rest = lax.scan(step, (start, first), None, length=n_moments - 2)
    return jnp.concatenate([start[None], first[None], rest], axis=0)


@functools.partial(jax.jit, static_argnames=("n_moments", "n_random", "kind", "k_batch"))
def conductivity_moments(hamiltonian, moving, tangents, centre, half, key, weights,
                         n_moments: int, n_random: int, kind: str = "stochastic",
                         k_batch=None):
    """The k-weighted two-index moments of one channel, and their guard.

    ``moving`` and ``tangents`` are this channel's entries of
    :func:`velocity_tangents`. Returns ``(mu, norms, dos)``: ``mu`` is ``(ndir, ndir, N, N)`` complex,
    ``sum_k weights_k Tr[v_a T_n(H') v_b T_m(H')]`` estimated by the start
    vectors; ``norms`` is ``(N,)``, ``sum_k weights_k sum_r <T_n r|T_n r>``
    over ``sum_k weights_k sum_r <r|r>``, which a spectrum inside the interval
    keeps at or below one; ``dos`` is ``(N,)``, the trace moments
    ``sum_k weights_k Tr T_n(H')`` of the same start vectors.

    ``kind`` is ``"stochastic"`` (``n_random`` random phases per k-point,
    each sample scaled by the number of live components so that it estimates
    the trace) or ``"exact"`` (every unit vector of the live sphere). A
    module-level ``jit`` closing over nothing, so a second call on another
    potential compiles nothing; the key is folded with the k index.
    """
    mask = live_mask(hamiltonian)
    dtype = hamiltonian.dtype
    ndir = len(tangents)

    def one_k(args):
        ik, weight = args
        live = mask[ik]
        count = jnp.sum(live).astype(weight.dtype)

        def apply(v):
            out = (hamiltonian.apply(v, ik) - centre * v) / half
            return jnp.where(live, out, 0.0)

        def velocity(a, v):
            return apply_velocity(hamiltonian, moving, tangents[a], v, ik)

        if kind == "stochastic":
            starts = random_phases(jax.random.fold_in(key, ik), live, n_random, dtype)
            scale = weight * count
        else:
            starts = unit_vectors(live, dtype)
            scale = weight

        def one_vector(carry, r):
            mu, norms, dos = carry
            right = _family(apply, r, n_moments)  # (N, ndim): T_m r
            lefts = [_family(apply, velocity(a, r), n_moments) for a in range(ndir)]
            rows = []
            for b in range(ndir):
                image = velocity(b, right)  # v_b T_m r
                rows.append(jnp.stack([jnp.conj(left) @ image.T for left in lefts]))
            block = jnp.stack(rows, axis=1)  # (a, b, n, m)
            norms = norms + jnp.real(jnp.sum(jnp.conj(right) * right, axis=-1))
            dos = dos + jnp.real(right @ jnp.conj(r))
            return (mu + block, norms, dos), None

        real = jnp.zeros((), dtype).real.dtype
        zero = (jnp.zeros((ndir, ndir, n_moments, n_moments), dtype),
                jnp.zeros((n_moments,), real), jnp.zeros((n_moments,), real))
        (mu, norms, dos), _ = lax.scan(one_vector, zero, starts)
        if kind == "stochastic":
            mu, norms, dos = mu / n_random, norms / n_random, dos / n_random
        norm0 = jnp.real(jnp.sum(jnp.conj(starts) * starts)) / (
            n_random if kind == "stochastic" else 1)
        return scale * mu, weight * norms, scale * dos, weight * norm0

    indices = jnp.arange(mask.shape[0])
    mu, norms, dos, norm0 = sum_k(one_k, (indices, weights), batch=k_batch)
    return mu, norms / norm0, dos


# -- the control variate -------------------------------------------------------

#: Degree of the low-pass filter for the conductivity's variate: the DOS's, 16
#: (:data:`~defuqumat.kpm.control.FILTER_DEGREE`). The conduction states of the
#: window need Ritz values well inside a delta width, which degree 8 misses by
#: 0.0035 Ry (median) on si16; the error of ``sigma_xx`` falls from 0.037 to 0.012
#: at R = 1 for 1.6 times the time.
FILTER_DEGREE_SIGMA = FILTER_DEGREE

@functools.partial(jax.jit, static_argnames=("n_moments", "n_random"))
def variate_block(hamiltonian, moving, tangents, ik, key, theta, vectors, centre, half,
                  n_moments: int, n_random: int):
    """``((ndir, ndir, N, N), (N,))``: the control variate's corrections at one k-point.

    The first is the two-index moments'; the second is P170's correction of the
    trace moments ``<r|T_n r>`` of the same random vectors, kept on the result
    as :attr:`KPMConductivity.dos_moments`.

    With the Ritz pairs ``(theta_j, y_j)`` of :func:`~defuqumat.kpm.control.ritz_pairs`,
    ``P = sum_j |y_j><y_j|`` and ``V^a_jl = <y_j|v_a|y_l>``, the model sample of a
    random vector ``r`` is ``n_live <r|v_a P T_n(H_P) P v_b P T_m(H_P) P|r>``,
    ``n_live sum_jl <r|v_a|y_j> T_n(theta_j) V^b_jl T_m(theta_l) c_l`` with
    ``c = <y|r>``, and its expectation is ``sum_jl V^a_lj T_n(theta_j) V^b_jl
    T_m(theta_l)`` exactly, since ``E[n_live <r|A|y_j> <y_l|r>] = <y_l|A|y_j>`` for
    random phases on the live sphere. The correction is the expectation minus
    the mean of the model samples over the same random vectors
    :func:`conductivity_moments` draws at ``ik``, so it is unbiased whatever the
    pairs are. Not weighted by ``w_k``.

    **The left leg is ``<r|v_a|y_j>``, not ``<r|P v_a|y_j>``.** The true sample is
    ``sum <r|v_a|o><o|v_b|c><c|r>`` over the states at ``E`` and ``E + w``; when
    ``P`` holds them all, the model with ``v_a r`` itself equals it and the
    difference has no noise left, while the model with ``P v_a P`` misses the
    part of ``r`` outside ``P`` that ``v_a`` carries onto the valence states, and
    that part is most of the noise: on si16 that form gained 1.3x at every filter
    degree, with the window's Ritz values exact to 1e-8 Ry at degree 24
    (``PLAN.md`` P169). It costs one velocity application per vector and
    direction.
    """
    live = live_mask(hamiltonian)[ik]
    dtype = hamiltonian.dtype
    ndir = len(tangents)
    vectors = jnp.where(live, vectors, 0.0)
    elements = [jnp.conj(vectors) @ apply_velocity(hamiltonian, moving, tangents[a],
                                                    vectors, ik).T for a in range(ndir)]
    x = jnp.clip((theta - centre) / half, -1.0, 1.0)
    n = jnp.arange(n_moments, dtype=x.dtype)
    cheb = jnp.cos(n[:, None] * jnp.arccos(x)[None, :]).astype(dtype)  # (N, M)
    starts = random_phases(jax.random.fold_in(key, ik), live, n_random, dtype)
    c = jnp.conj(vectors) @ starts.T  # (M, R): <y_j|r>
    # <r|v_a|y_j> = conj(<y_j|v_a r>), with v_a r itself rather than P v_a P r
    legs = [jnp.conj(jnp.conj(vectors) @ apply_velocity(hamiltonian, moving, tangents[a],
                                                        starts, ik).T) for a in range(ndir)]
    count = jnp.sum(live).astype(x.dtype)
    rows = []
    for a in range(ndir):
        row = []
        for b in range(ndir):
            exact = cheb @ (elements[a].T * elements[b]) @ cheb.T
            left = cheb[:, :, None] * legs[a][None]  # (N, M, R): T_n(theta_j) <r|v_a|y_j>
            right = jnp.einsum("mp,pr,lp->mlr", cheb, c, elements[b])
            model = count * jnp.einsum("nlr,mlr->nm", left, right) / n_random
            row.append(exact - model)
        rows.append(jnp.stack(row))
    # P170's variate of the trace moments <r|T_n r> of the same vectors, for the level
    norms = jnp.sum(jnp.abs(vectors) ** 2, axis=1).astype(cheb.dtype)
    weight = jnp.mean(jnp.abs(c) ** 2, axis=1).astype(cheb.dtype)  # (M,)
    trace = jnp.real(cheb @ norms - count * (cheb @ weight))
    return jnp.stack(rows), trace


def variate_corrections(calculation, hamiltonians, moving, tangents, weights, centre, half,
                        n_moments, n_random, seed, cut, degree):
    """``((nspin, ndir, ndir, N, N), (nspin, N), ritz)``: the k-weighted corrections of every channel.

    The second is the correction of the trace moments, scaled as
    :func:`conductivity_moments` scales them (``w_k n_live``); ``ritz`` is the
    sorted Ritz values of every k-point and channel, for :func:`ritz_gap_level`.
    """
    from defuqumat.kpm.control import ritz_pairs
    from defuqumat.projwfc.projections import atomic_orbitals

    orbitals = jnp.swapaxes(atomic_orbitals(calculation), 1, 2)  # (nk, natw, ndim)
    top = centre + half
    cut_centre, cut_half = 0.5 * (top + cut), 0.5 * (top - cut)
    key = jax.random.PRNGKey(seed)
    weights = np.asarray(weights)
    out, traces, values = [], [], []
    for channel, hamiltonian in enumerate(hamiltonians):
        tangent = tuple(t[channel] for t in tangents)
        channel_key = jax.random.fold_in(key, channel)
        total, trace = 0.0, 0.0
        for ik in range(orbitals.shape[0]):
            if weights[ik] == 0.0:
                continue
            theta, vectors = ritz_pairs(hamiltonian, ik, orbitals[ik], cut_centre, cut_half,
                                        degree)
            values.append(np.sort(np.asarray(theta)))
            block, moments = variate_block(
                hamiltonian, moving[channel], tangent, ik, channel_key, theta, vectors,
                centre, half, n_moments, n_random)
            total = total + weights[ik] * np.asarray(block)
            trace = trace + weights[ik] * np.asarray(moments)
        out.append(total)
        traces.append(trace)
    return np.stack(out), np.stack(traces), values


# -- from the moments to a spectrum -------------------------------------------

def delta_table(factors, centre: float, half: float, energies):
    """``(N, nE)``: ``g_n (2 - delta_n0) T_n(x) / (pi half sqrt(1 - x^2))`` on ``energies``.

    Contracted with the moments of an operator ``A`` over ``n`` it gives
    ``Tr[A delta(E - H)]`` per Ry, and zero outside the interval.
    """
    factors = jnp.asarray(factors)
    x = (jnp.asarray(energies) - centre) / half
    inside = jnp.abs(x) < 1.0
    xc = jnp.where(inside, x, 0.0)
    theta = jnp.arccos(xc)
    n = jnp.arange(factors.shape[0], dtype=theta.dtype)
    weights = jnp.where(n > 0, 2.0, 1.0) * factors
    table = weights[:, None] * jnp.cos(n[:, None] * theta[None, :])
    table = table / (math.pi * half * jnp.sqrt(1.0 - xc**2))[None, :]
    return jnp.where(inside[None, :], table, 0.0)


def fermi_function(energies, fermi_energy: float, temperature: float):
    """Occupation at ``energies``: a step (one half at the level) at zero temperature."""
    energies = jnp.asarray(energies)
    if temperature <= 0.0:
        return jnp.where(energies < fermi_energy, 1.0,
                         jnp.where(energies > fermi_energy, 0.0, 0.5))
    return 0.5 * (1.0 - jnp.tanh(0.5 * (energies - fermi_energy) / temperature))


def kubo_greenwood(moments, factors, centre: float, half: float, volume: float,
                   fermi_energy: float, temperature: float, emin: float, delta_e: float,
                   n_frequencies: int):
    """``(nw, ndir, ndir)`` complex: ``sigma'_ab(w_j)`` at ``w_j = j delta_e``.

    The energy grid is uniform from ``emin`` in steps of ``delta_e``, so
    ``E_i + w_j`` is on it. ``Gamma_ab(E, E') = Tr[v_a delta(E - H) v_b
    delta(E' - H)]`` is ``A^T mu A`` with ``A`` the :func:`delta_table`, and
    ``sigma'_ab`` is built from ``Gamma_ba`` (the band sum's order, see the module
    docstring); the
    ``w = 0`` entry is the limit ``(pi / Omega) int dE (-f'(E)) Gamma(E, E)``,
    the Kubo-Greenwood DC conductivity, which is zero for an insulator. In
    Hartree atomic units when the energies are in Ry and the velocity in
    Ry bohr, the same homogeneity as the band sum
    (:mod:`defuqumat.response.conductivity`).
    """
    top = fermi_energy + 12.0 * max(temperature, 0.0)
    n_low = int(math.floor((top - emin) / delta_e)) + 2
    energies = emin + delta_e * jnp.arange(n_low + n_frequencies)
    table = delta_table(factors, centre, half, energies)
    gamma = jnp.einsum("ni,abnm,mj->abij", table[:, :n_low], jnp.asarray(moments), table)
    occupation = fermi_function(energies, fermi_energy, temperature)
    rows = jnp.arange(n_low)
    shifts = jnp.arange(1, n_frequencies)
    columns = rows[:, None] + shifts[None, :]  # (nlow, nw - 1): E_i + w_j
    weight = occupation[rows][:, None] - occupation[columns]
    sigma = delta_e * jnp.einsum("ij,abij->jba", weight, gamma[:, :, rows[:, None], columns])
    w = delta_e * np.arange(1, n_frequencies)
    sigma = math.pi / volume * sigma / jnp.asarray(w)[:, None, None]
    # w -> 0: [f(E) - f(E + w)] / w -> -f'(E)
    if temperature > 0.0:
        slope = occupation * (1.0 - occupation) / temperature
        dc = delta_e * jnp.einsum("i,abi->ba", slope[:n_low],
                                  gamma[:, :, rows, rows])
    else:
        x = (fermi_energy - centre) / half
        if abs(x) >= 1.0:
            dc = jnp.zeros(gamma.shape[:2], gamma.dtype)
        else:
            at_level = delta_table(factors, centre, half, jnp.asarray([fermi_energy]))[:, 0]
            dc = jnp.einsum("n,abnm,m->ba", at_level, jnp.asarray(moments), at_level)
    dc = math.pi / volume * dc
    return jnp.concatenate([dc[None], sigma], axis=0)


def channel_levels(fermi_energy, nspin: int) -> np.ndarray:
    """``(nspin,)``: one Fermi level per channel from a number or a pair."""
    levels = np.asarray(fermi_energy, dtype=float).reshape(-1)
    if levels.size not in (1, nspin):
        raise ValueError(f"fermi_energy must be one level or one per spin channel ({nspin}), "
                         f"got {levels.size}")
    return np.broadcast_to(levels, (nspin,)).copy()


def channel_sum(moments, factors, centre, half, volume, levels, temperature, emin, delta_e,
                n_frequencies):
    """:func:`kubo_greenwood` of ``(nspin, ...)`` moments, each channel at its own level."""
    if np.all(levels == levels[0]):
        return np.asarray(kubo_greenwood(np.asarray(moments).sum(axis=0), factors, centre, half,
                                         volume, float(levels[0]), temperature, emin, delta_e,
                                         n_frequencies))
    return sum(np.asarray(kubo_greenwood(np.asarray(m), factors, centre, half, volume,
                                         float(level), temperature, emin, delta_e,
                                         n_frequencies))
               for m, level in zip(moments, levels))


def ritz_gap_level(ritz, nelec: float, nspin: int, noncolin: bool):
    """Mid-gap between the highest occupied and the lowest empty Ritz value, or ``None``.

    The Ritz values of the filtered orbitals are the band energies to about 1e-6 Ry
    in the valence band and 1e-4 Ry just above it (filter degree 16), so when the
    band holding the last electron stays below the next one at every k-point the
    gap is read off them. ``ritz`` is :func:`variate_corrections`' list, channel
    first and then k-point. At ``nspin = 2`` with one level for both channels the
    two channels' values at each k-point are one ladder of singly occupied states;
    a constrained moment, with a level per channel, does not come here. ``None``
    for an odd filling, too few pairs, or no gap.
    """
    if not ritz or nspin not in (1, 2):
        return None
    if nspin == 1:
        per_state = 1.0 if noncolin else 2.0
        ladders = [np.sort(np.asarray(v, dtype=float)) for v in ritz]
    else:
        if len(ritz) % 2:
            return None
        nk = len(ritz) // 2
        per_state = 1.0
        ladders = [np.sort(np.concatenate([np.asarray(ritz[ik], dtype=float),
                                           np.asarray(ritz[nk + ik], dtype=float)]))
                   for ik in range(nk)]
    occupied = nelec / per_state
    if abs(occupied - round(occupied)) > 1e-8:
        return None
    occupied = int(round(occupied))
    if occupied < 1 or any(len(v) <= occupied for v in ladders):
        return None
    homo = max(float(v[occupied - 1]) for v in ladders)
    lumo = min(float(v[occupied]) for v in ladders)
    if lumo <= homo:
        return None
    return 0.5 * (homo + lumo)


# -- the result and the entry point -------------------------------------------

@dataclass(frozen=True)
class KPMConductivity:
    """The absorptive optical conductivity by the kernel polynomial method.

    Attributes:
        frequencies: ``(nw,)`` photon energies in Ry, ``j delta_e`` from zero.
        sigma: ``(nw, ndir, ndir)`` complex, Hartree atomic units: the
            absorptive (Hermitian) part of ``sigma_ab(w)``. Its diagonal is
            ``Re sigma_aa``; the antisymmetric imaginary part of an off-diagonal
            entry is the absorptive Hall part, with the sign of
            :func:`~defuqumat.response.conductivity.optical_conductivity`'s.
            Entry ``0`` is the DC conductivity
            ``(pi / Omega) int dE (-f'(E)) Tr[v_b delta v_a delta]``.
        directions: the cartesian unit vectors of the rows, ``(ndir, 3)``.
        moments: ``(nspin, ndir, ndir, N, N)`` complex, summed over k with the
            k weights, so another kernel, Fermi level or temperature is
            :meth:`reconstructed` with no recursion.
        dos_moments: ``(nspin, N)``, ``sum_k w_k Tr T_n(H')`` from the same start
            vectors, with P170's correction when the variate is on.
        centre, half_width: the interval, Ry.
        kernel, n_moments: the kernel and the moment count.
        broadening: half width at half maximum of one delta at the Fermi level
            (the mean of a pair), Ry, with the ``N`` the run used; it is the
            ``broadening`` asked for where that level is the reference energy
            ``N`` was chosen at.
        transition_broadening: of a transition, two deltas convolved: ``2``
            times the above for the Lorentz kernel, ``sqrt(2)`` times for a
            Gaussian (Jackson), at the Fermi level, so it is
            :meth:`broadening_for` of the run's kernel.
        fermi_energy, temperature: of the occupations, Ry; a pair, one per
            channel, for a run with a constrained moment.
        volume: bohr^3.
        trace, n_random, seed: how the trace was taken.
    """

    frequencies: np.ndarray
    sigma: np.ndarray
    directions: np.ndarray
    moments: np.ndarray
    dos_moments: np.ndarray
    centre: float
    half_width: float
    kernel: str
    n_moments: int
    broadening: float
    transition_broadening: float
    fermi_energy: float | tuple
    temperature: float
    volume: float
    delta_e: float
    emin: float
    trace: str
    n_random: int
    seed: int

    @property
    def frequencies_ev(self) -> np.ndarray:
        from defuqumat.units import RY_TO_EV

        return np.asarray(self.frequencies) * RY_TO_EV

    @property
    def sigma_s_per_cm(self) -> np.ndarray:
        """``sigma`` in S/cm."""
        from defuqumat.units import AU_TO_S_PER_CM

        return np.asarray(self.sigma) * AU_TO_S_PER_CM

    @property
    def epsilon_imaginary(self) -> np.ndarray:
        """``Im eps_ab(w) = 4 pi sigma'_ab / w`` (Hartree units), ``nan`` at ``w = 0``."""
        w = 0.5 * np.asarray(self.frequencies)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = 4.0 * math.pi * np.asarray(self.sigma) / w[:, None, None]
        out[0] = np.nan
        return out

    def broadening_for(self, kernel: str) -> float:
        """The half width of a transition (Ry) with ``kernel`` at this ``N``, at the Fermi level.

        What :meth:`reconstructed` with that kernel gives, so a band sum can be
        handed the same width.
        """
        chosen = get_kernel(kernel)
        if chosen.width_at is None:
            return math.nan
        x = (float(np.mean(self.fermi_energy)) - self.centre) / self.half_width
        one = float(self.half_width * chosen.width_at(self.n_moments, x))
        return (2.0 if chosen.name == "lorentz" else math.sqrt(2.0)) * one

    def reconstructed(self, *, kernel: str | None = None, n_moments: int | None = None,
                      fermi_energy: float | None = None, temperature: float | None = None,
                      n_frequencies: int | None = None) -> np.ndarray:
        """``sigma`` again from the stored moments, with another kernel, ``N``, level or temperature."""
        kernel = self.kernel if kernel is None else kernel
        n = self.n_moments if n_moments is None else int(n_moments)
        factors = get_kernel(kernel).factors(n)
        moments = np.asarray(self.moments)[..., :n, :n]
        levels = channel_levels(self.fermi_energy if fermi_energy is None else fermi_energy,
                                moments.shape[0])
        temp = self.temperature if temperature is None else float(temperature)
        nw = len(self.frequencies) if n_frequencies is None else int(n_frequencies)
        return channel_sum(moments, factors, self.centre, self.half_width, self.volume, levels,
                           temp, self.emin, self.delta_e, nw)


def _directions(directions):
    out = []
    for d in directions:
        if isinstance(d, str):
            if d not in AXES:
                raise ValueError(f"a direction is 'x', 'y', 'z' or a 3-vector, not {d!r}")
            vector = np.zeros(3)
            vector[AXES[d]] = 1.0
        else:
            vector = np.asarray(d, dtype=float)
            if vector.shape != (3,) or not np.linalg.norm(vector) > 0:
                raise ValueError(f"a direction must be a non-zero 3-vector, got {d!r}")
            vector = vector / np.linalg.norm(vector)
        out.append(vector)
    return np.stack(out)


def run_kpm_conductivity(system, pseudos, density, *, fermi_energy: float | None = None,
                         directions=("x", "y", "z"), window: float = 1.5,
                         broadening: float = 0.02, n_moments: int | None = None,
                         kernel: str = "jackson", trace: str = "stochastic",
                         n_random: int = 4, seed: int = 0, temperature: float = 0.0,
                         delta_e: float | None = None, control_variate: bool | str = "auto",
                         filter_degree: int | None = None, filter_cut: float | None = None,
                         lanczos_steps: int | None = None,
                         edge_margin: float | None = None, k_batch="default", ns=None,
                         tau=None, becsum: tuple = (), field=None, field_scale=None,
                         calculation=None,
                         reference_energy: float | None = None) -> KPMConductivity:
    """``sigma'_ab(w)`` of a converged density by two-index Chebyshev moments.

    Args:
        system, pseudos, density: the converged run.
        fermi_energy: Ry, where the occupation steps (smeared by
            ``temperature``): the SCF's level for a metal, any energy in the
            gap for an insulator. A pair, one level per spin channel, for a
            run with a constrained moment, where a single number raises. If
            not given, ``reference_energy`` must be, and the level is put
            mid-gap from the variate's Ritz values (:func:`ritz_gap_level`);
            that raises with the variate off or with no gap in the Ritz
            values, since the count of a few random vectors does not place a
            level. A level at the top of the valence band would cut the
            broadened top states in half.
        reference_energy: Ry, where ``broadening`` holds and where the filter
            cut is measured from when ``fermi_energy`` is not given, as for
            :func:`~defuqumat.kpm.dos.run_kpm_dos`; an energy near the gap,
            such as the highest occupied level, which
            :meth:`~defuqumat.calculator.Calculator.get_kpm_conductivity`
            passes for fixed occupations. Ignored when ``fermi_energy`` is
            given, whose mean is then the reference.
        directions: the cartesian axes of the tensor, names or 3-vectors; each
            costs one more recursion and multiplies the Gram product.
        window: the largest photon energy, Ry.
        broadening: half width at half maximum of one delta at the reference
            energy, Ry; it sets the number of moments, and the result reports
            the width at the level it found with that number. A transition is
            broader (:attr:`KPMConductivity.transition_broadening`).
        n_moments: overrides ``broadening``.
        kernel: ``"jackson"`` (default, Gaussian deltas) or ``"lorentz"``
            (close to Lorentzian ones, the line shape of the band sum).
        trace: ``"stochastic"`` or ``"exact"`` (one recursion per plane wave,
            for small cells).
        n_random, seed: the random vectors per k-point.
        temperature: of the Fermi function, Ry; zero is a step.
        delta_e: the energy and frequency step, Ry; a fifth of the broadening
            at the Fermi level if not given.
        control_variate: subtract from each random sample the same trace in the
            span of the low-pass-filtered atomic orbitals and add back its exact
            value (:func:`variate_block`): unbiased, with the noise of the
            transitions inside that span removed. ``"auto"`` (default) uses it for
            random vectors when the datasets carry orbitals
            (:func:`~defuqumat.kpm.control.resolve_variate`); ``True`` raises with
            the exact trace.
        filter_degree, filter_cut: the low-pass filter, degree 16
            (:data:`FILTER_DEGREE_SIGMA`) and a cut at ``E_F + max(1, window + 0.4)``
            Ry when not given (the higher level of a pair, or
            ``reference_energy`` when no level is given).
        lanczos_steps, edge_margin: the interval (:func:`~defuqumat.kpm.bounds.spectral_interval`).
        ns, tau, becsum, field, field_scale: the rest of the converged state.
        calculation: the SCF's own.
    """
    from defuqumat.kpm.bounds import EDGE_MARGIN, LANCZOS_STEPS
    from defuqumat.kpm.bounds import spectral_interval
    from defuqumat.kpm.dos import kpm_setup
    from defuqumat.kpm.kernels import moment_count
    from defuqumat.kpm.guard import SpectralIntervalError
    from defuqumat.kpm.traces import whole_grid_calculation
    from defuqumat.workflows.nscf import threaded_calculation

    if trace not in ("stochastic", "exact"):
        raise ValueError(f"trace must be 'stochastic' or 'exact', got {trace!r}")
    vectors = _directions(directions)
    calculation, system, _, k_batch = threaded_calculation(calculation, system, None, k_batch)
    if calculation is None:
        from defuqumat.scf.driver import Calculation

        calculation = Calculation(system, pseudos, k_batch=k_batch)
    # a velocity tensor on a wedge is not completed by its weights alone
    calculation = whole_grid_calculation(calculation)
    calculation, hamiltonians, batch = kpm_setup(
        system, pseudos, density, k_batch=k_batch, calculation=calculation, ns=ns, tau=tau,
        becsum=becsum, field=field, field_scale=field_scale)
    if getattr(calculation, "gamma_only", False):
        raise NotImplementedError("the KPM conductivity does not run on gamma_only storage")
    require_a_velocity(calculation, hamiltonians)
    from defuqumat.kpm.control import resolve_variate

    # The level, or the energy it will be found near, is decided before the
    # recursion: N is chosen there and the filter cut is measured from it. A
    # level read off the count of the random vectors afterwards is what this
    # replaces (PLAN.md P172): it put Al 0.06 Ry above its SCF level and an
    # insulator without the variate inside its valence band for half the seeds.
    nspin = len(hamiltonians)
    use_variate = resolve_variate(control_variate, trace, calculation)
    if use_variate and trace != "stochastic":
        raise ValueError("control_variate reduces the noise of random vectors; the exact "
                         "trace has none")
    if fermi_energy is not None:
        if calculation.two_fermi_energies and np.size(fermi_energy) != 2:
            raise ValueError(
                "a run with a constrained moment has one level per spin channel, pass the pair "
                "(scf_result.fermi_energy_up, scf_result.fermi_energy_down)")
        channel_levels(fermi_energy, nspin)
        reference = float(np.mean(fermi_energy))
    elif calculation.two_fermi_energies:
        raise ValueError(
            "a run with a constrained moment has one Fermi level per spin channel: pass "
            "fermi_energy as the pair (scf_result.fermi_energy_up, scf_result.fermi_energy_down) "
            "for a smeared run, or one energy in the gap of each channel for fixed occupations")
    elif reference_energy is None:
        raise ValueError(
            "the conductivity needs its Fermi level before the recursion: pass fermi_energy "
            "(scf_result.fermi_energy for a smeared run, (homo + lumo) / 2 or any energy in "
            "the gap for an insulator), or reference_energy (an energy near the top of the "
            "valence band, such as scf_result.homo) to have the level put mid-gap from the "
            "Ritz values of the atomic orbitals")
    elif not use_variate:
        raise ValueError(
            "the exact trace has no Ritz values to put the level in the gap: pass fermi_energy"
            if trace == "exact" else
            "with no control variate the count of random vectors puts the level inside the "
            "valence band for about half the seeds (measured on Si2: 6 of 12 seeds at R = 2); "
            "pass fermi_energy")
    else:
        reference = float(reference_energy)
    centre, half, bottom = spectral_interval(
        hamiltonians, seed=seed, steps=LANCZOS_STEPS if lanczos_steps is None else lanczos_steps,
        margin=EDGE_MARGIN if edge_margin is None else edge_margin, k_batch=batch)
    emin = bottom - 10.0 * broadening
    chosen = get_kernel(kernel)
    if n_moments is None:
        n_moments = moment_count(chosen, broadening, half, (reference - centre) / half)
    moving, tangents = velocity_tangents(calculation, hamiltonians, vectors, ns=ns)
    weights = jnp.asarray(calculation.system.kpoints.weights)
    key = jax.random.PRNGKey(seed)
    moments, dos = [], []
    for channel, hamiltonian in enumerate(hamiltonians):
        tangent = tuple(t[channel] for t in tangents)
        mu, norms, trace_moments = conductivity_moments(
            hamiltonian, moving[channel], tangent, centre, half,
            jax.random.fold_in(key, channel), weights,
            n_moments, n_random, kind=trace, k_batch=batch)
        norms = np.asarray(norms)
        if not np.all(norms <= 1.0 + 1e-6):
            raise SpectralIntervalError(
                f"the Chebyshev vectors grew (largest norm^2 {norms.max():.3g} of 1): the "
                f"spectrum leaves the interval [{centre - half:.4f}, {centre + half:.4f}] "
                "Ry. Pass a larger edge_margin")
        moments.append(np.asarray(mu))
        dos.append(np.asarray(trace_moments))
    moments = np.stack(moments)
    factors = chosen.factors(n_moments)
    dos = np.stack(dos)
    if use_variate:
        base = (float(np.max(fermi_energy)) if fermi_energy is not None else reference)
        cut = base + (max(1.0, window + 0.4) if filter_cut is None else float(filter_cut))
        corrections, dos_corrections, ritz = variate_corrections(
            calculation, hamiltonians, moving, tangents, np.asarray(weights), centre, half,
            n_moments, n_random, seed, cut,
            FILTER_DEGREE_SIGMA if filter_degree is None else int(filter_degree))
        moments = moments + corrections
        dos = dos + dos_corrections
    if fermi_energy is None:
        # mid-gap from the Ritz values, deterministic and 1e-4 Ry accurate at the gap
        fermi_energy = ritz_gap_level(ritz, float(calculation.nelec), nspin,
                                      calculation.noncolin)
        if fermi_energy is None:
            raise ValueError(
                "the Ritz values show no gap (or the filling is not a whole number of "
                "bands): a metal needs the Fermi level of its SCF, pass fermi_energy")
    levels = channel_levels(fermi_energy, nspin)
    # the width is reported at the level the occupations step at, with the N used,
    # which is what broadening_for gives on the result
    x_level = (float(np.mean(levels)) - centre) / half
    width = (float(half * chosen.width_at(n_moments, x_level))
             if chosen.width_at is not None else math.nan)
    if delta_e is None:
        delta_e = width / 5.0 if math.isfinite(width) else 0.002
    n_frequencies = int(round(window / delta_e)) + 1
    sigma = channel_sum(moments, factors, centre, half, float(calculation.system.cell.volume),
                        levels, temperature, emin, delta_e, n_frequencies)
    factor = 2.0 if chosen.name == "lorentz" else math.sqrt(2.0)
    return KPMConductivity(
        frequencies=delta_e * np.arange(n_frequencies), sigma=sigma, directions=vectors,
        moments=moments, dos_moments=dos, centre=centre, half_width=half,
        kernel=kernel, n_moments=n_moments, broadening=width,
        transition_broadening=factor * width,
        fermi_energy=float(levels[0]) if levels.size == 1 or np.all(levels == levels[0])
        else tuple(float(level) for level in levels),
        temperature=float(temperature), volume=float(calculation.system.cell.volume),
        delta_e=float(delta_e), emin=float(emin), trace=trace,
        n_random=n_random if trace == "stochastic" else 0, seed=seed)
