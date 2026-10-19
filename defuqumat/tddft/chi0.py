"""The independent-particle response ``chi_0(G, G', omega)`` at ``q -> 0``.

``PLAN.md`` P37. This is the one object in the TDDFT phase that costs anything,
and it is the one thing the rest of :mod:`defuqumat.tddft` cannot get from code
that already exists: everywhere else in ``defuqumat/response/`` the
susceptibility appears as an *operator*, ``drho = chi_0 dV`` from a Sternheimer
solve, and a Dyson equation needs it as a **matrix** over reciprocal lattice
vectors, at every frequency.

**Sum over states, and the tempting shortcut is a trap.** The kernel of P37 is
static, so ``chi_0`` is needed at ``omega = 0`` for it and at every ``omega``
only for the spectrum -- which suggests taking the kernel from the Sternheimer
stack, where ``omega = 0`` already lives, and the spectrum from a sum over
states. ``tddftlr.f90`` says why not: it hands ``genvfxc`` the ``omega = 0``
*slice of the same array* it then inverts at every frequency, so the kernel and
the spectrum carry one band truncation, one broadening and one response cutoff.
Two builders would put a band-complete, unbroadened kernel inside a truncated,
broadened spectrum, and the bootstrap's self-consistency would then couple them.
Nothing in the answer shows that. So there is one builder, it is Adler-Wiser
(``genvchi0.f90``), and the Sternheimer stack is the independent referee
instead (``tests/regression/test_tddft.py``).

**What is stored is the symmetrised ``v^1/2 chi_0 v^1/2``**, not ``chi_0``.
That is what keeps the optical limit finite: ``chi_0`` itself vanishes as
``q^2`` and ``v`` diverges as ``1/q^2``, and only the product has a limit. In
that limit the ``G = 0`` row and column become **direction-dependent**, so they
are a 3x3 block rather than one number and the matrix has

    nm = 3 + (ngrf - 1)

rows: three head directions, then every ``G != 0`` inside the response cutoff.
This is Elk's ``t3hw`` layout (``nm = ngrf + 2``) with its index 1 doing double
duty removed.

**The whole matrix is a sum of rank-one terms, which is what makes it cheap.**
For one pair of states,

    r_a    = -sqrt(8 pi) <u_i| dH/dk_a |u_j> / (eps_i - eps_j)      (head)
    r_G    =  sqrt(8 pi / |G|^2) <u_i| e^{-iG.r} |u_j>              (body)

and the pair contributes ``cw(omega) r r^dagger`` with the scalar

    cw(omega) = (wg_i - wg_j) / Omega / (eps_i - eps_j + omega + i eta).

So the pair vectors are built once, per k-point, and the frequency axis costs
one matrix product each -- ``(pairs, nm)`` scaled by a frequency-dependent
weight and contracted against itself.

**The head is the one line of Elk that must not be transcribed.**
``genvchi0.f90`` reads momentum matrix elements off a file, which is exactly
right in an all-electron code and silently wrong in a pseudopotential one:
``[H, r] != p`` when the pseudopotential is nonlocal, and the difference is not
small. What is wanted is ``dH/dk``, and that is
:class:`~defuqumat.response.velocity.VelocityOperator` -- one ``jvp`` of ``H(k)``
at a frozen sphere, rule D2. This module is otherwise a transcription; this is
the place where the autodiff the project is built on is load-bearing rather
than decorative.

**Both orderings of every pair are summed**, as ``genvchi0`` does -- the
resonant term and the antiresonant one. Keeping only ``v -> c`` and doubling
gets the real part roughly right and the imaginary part wrong. The reversed
pair needs no second transform, because

    <u_j| e^{-iG.r} |u_i> = conj(<u_i| e^{+iG.r} |u_j>),

so its vector is the first one reflected through the origin and conjugated. The
head reflects with a **sign**, since ``rho_0`` is linear in ``q`` and the
reflection is ``q -> -q``.

**Scissors is part of the method**, not a convenience: PRL 107, 186401's Eq. (3)
replaces ``chi_0`` by a gap-corrected model response, and every published
bootstrap spectrum is computed that way. The shift is applied to the empty
states' eigenvalues, and the velocity matrix elements are renormalised by
``e_ij / (e_ij -+ Delta)`` with it -- Elk's ``getpmat.f90``, which is the
Del Sole-Girlanda factor and is what keeps ``<u|r|u'>`` right when the energies
it was divided by have moved.

**What it costs, and where.** Two working sets, and they scale differently:

* the **stored matrix**, ``nw nm^2`` complex, which is what the caller keeps.
  With ``nm = 115`` (silicon at ``ecut_response = 8``) and 150 frequencies that
  is 25 MB, and it is quadratic in the response cutoff.
* the **assembly**, the ``einsum`` below, which is the larger of the two. Over
  the whole frequency axis XLA contracts it through whichever intermediate is
  smaller -- ``(nw, 2 npairs, nm)`` when ``nw < nm``, ``(2 npairs, nm, nm)``
  otherwise -- and at 60 bands, ``nm = 115`` and 200 frequencies that is a
  90 MB block of a **131 MB** temporary per k-point (compiled, CPU). It is a
  stated trade rather than an accident -- the alternative is one matrix product
  per frequency, which halves the flop rate on CPU -- and the frequency axis is
  the dial that bounds it: ``w_batch`` frequencies at a time hold ``w_batch``
  blocks of ``(2 npairs, nm)``, **32.4 MB at 32 and 8.7 MB at 8** on the same
  shapes (``notes/GPU-MEMORY-NEXT.md`` item 12). The whole axis is the CPU default;
  an accelerator's is budgeted (:func:`~defuqumat.batching.resolve_w_batch`).
  ``PERFORMANCE.md`` carries the time.

The transforms are ``nk npairs`` of them, one per occupied-empty pair, and the
**pair densities are a working set of their own**: one is a whole complex field
on the smooth grid, so forming every pair's at once is ``npairs`` times the FFT
box -- 26 GB on a cell with 50 occupied and 150 empty bands and a 60^3 grid,
where the wavefunctions it is built from are a few hundred megabytes. The pair
axis is therefore chunked (``pair_batch``, defaulting to the **band** dial,
because one pair density in flight is exactly what one band in flight is), and
what stays live whatever the chunk is ``fields`` -- the ``nbnd`` states in real
space, which is the natural working set and is ``nbnd/npairs`` of the other.

This bounds the grid-sized allocation and **not** the assembly above: the
``(nw, 2 npairs, nm)`` einsum is still linear in ``npairs``, and it is the
stated trade the previous paragraph records rather than an oversight. It is
also the smaller of the two by the ratio of ``nm`` to the FFT box, which is
where the two hundred separating 100 MB from 26 GB comes from.

**The k axis is walked a chunk at a time** (:mod:`defuqumat.response.walk`): a
chunk's states cross to the device, its velocity matrix elements and pair
vectors are built on its own rows and contracted into the matrix, and the next
chunk follows, so the states are never on the device for the whole k axis. The
chunk is the calculation's ``k_batch`` and moves the matrix at round-off, through
the order of the k sum.

**Collinear spin is a sum over the two channels** (``nspin = 2``). The pair
``(i, j)`` of the Adler-Wiser sum is a pair of states of one spin, since
``e^{-iG.r}`` and ``dH/dk`` are both diagonal in spin, so the charge response
is ``chi_0 = chi_0^up + chi_0^dn``, each channel with its own occupied count
(:func:`~defuqumat.scf.occupations.occupied_counts`, QE's ``nbnd_occ``, which
is what makes the two depths of a magnetic insulator different) and its own
``wg``, which already carries one electron a band rather than two. The pair
lists differ in length between the channels, so each channel is its own
static shape and the k walk contracts both. What the matrix does *not* carry
is the magnetization's response: it is the charge-charge block of the 2x2
``(n, m)`` response, and a kernel that couples ``n`` to ``m`` -- ALDA on a
polarized density -- is refused where it is built
(:func:`~defuqumat.tddft.kernels.alda_matrix`), not here.

**A spinor is a doubled space, not a new formula.** With ``noncolin`` the
states are ``(up, down)`` on one sphere, both components are transformed with
the same ``fft_index``, and the pair density is the spin trace
``sum_s conj(u_i,s) u_j,s``. That is the charge-charge block of ``chi_0``, and
when the ground state carries no moment (``nspin_mag = 1``) it is all the
Coulomb and the unpolarized kernels see, with or without spin-orbit coupling,
whose whole effect is in the states and in ``dH/dk``. Each band then holds one
electron, so ``nocc`` is ``nelec`` and ``npairs`` is four times the scalar
run's on the same cell; ``fields`` is ``nbnd npol`` boxes. Checked in
``tests/regression/test_spinor_chi0.py``: silicon as a spinor against the scalar
run, and spin-orbit selenium against the spinor Sternheimer solve.

**A smeared run carries its intraband term**, the ``q -> 0`` limit of the
``i = j`` pairs that ``f_i - f_j`` removes: ``wp_ab^2/(omega + i eta)^2`` on
the head and nothing on the wings or the body (:func:`_drude_block`). The
pair list then runs over every band that is partly filled anywhere, ``(low,
high)`` per channel (:func:`_band_windows`), so a metal has
``low (nbnd - low) + (high - low)(2 nbnd - high - low - 1)/2`` pairs where
the rectangle has ``nocc (nbnd - nocc)``, and the assembly grows with it.

An **ultrasoft** pair density carries ``Q_ij(G)`` and its head the generalised
velocity (:func:`require_a_sum_over_states_regime`, I94).

Refused by name: finite ``q``, PAW, an ultrasoft run that is smeared,
polarized or a spinor, tetrahedron occupations, a smeared run that is
polarized, a spinor or on a reduced k-set (no number yet), a
spinor carrying a moment (``nspin_mag = 4``), a spin spiral, a scissors shift
on a smeared run, and a **reduced k-set** at ``nspin = 2`` -- see
:func:`require_a_sum_over_states_regime`.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from defuqumat.basis.fft import g_to_r, r_to_g
from defuqumat.batching import (
    map_axis, map_windows, resolve_pair_batch, resolve_w_batch, sum_k,
)
from defuqumat.response.velocity import VelocityOperator
from defuqumat.response.velocity import with_connections
from defuqumat.scf.occupations import occupied_counts
from defuqumat.scf.whole_sphere import unfold_gamma_states
from defuqumat.response.walk import (
    chunk_size, padded, row_leaves, store_rows, walk, with_rows,
)
from defuqumat.tddft.spinchi0 import augmentation_factors, state_projections
from defuqumat.system.kpoints import is_reduced
from defuqumat.units import E2, FPI
from defuqumat.system.wedge import wedge_group

__all__ = [
    "ResponseSphere",
    "ChiZero",
    "independent_response",
    "require_a_sum_over_states_regime",
    "response_sphere",
]

#: Below this the occupation difference of a pair is treated as zero and the
#: pair is dropped. ``genvchi0.f90``'s own ``1.d-8``.
_OCCUPATION_TOLERANCE = 1.0e-8


class ResponseSphere(eqx.Module):
    """The reciprocal lattice vectors ``chi_0`` is built on -- Elk's ``ngrf``.

    A cutoff of its own, and a small one: the body of ``chi_0`` is what carries
    the local-field effects, so it cannot be dropped, but it converges long
    before ``ecutrho`` and inheriting that cutoff makes ``nm`` -- and the cost,
    which is quadratic in it -- explode for nothing.

    ``G = 0`` is **not** in the body. In the optical limit it is the head, and
    the head is a 3x3 block of directions rather than a single entry.

    **An empty body is allowed and is a named approximation**, not a degenerate
    case: ``ecut = 0`` leaves the 3x3 head alone, which is the **head-only**
    kernel of the long-range-correction literature and is what Elk's own
    ``LiF-bootstrap`` example asks for with ``gmaxrf = 0.0``. Byun and Ullrich
    use it throughout and report that it differs little from the full matrix for
    the bootstrap family. It removes the local-field effect entirely, so
    ``eps_M`` then equals ``1 - X_head``; with the ``alda`` kernel, whose head
    and wings are identically zero, it is therefore *exactly* RPA.
    """

    #: ``(nbody,)`` flat indices into the smooth FFT box, for the gather that
    #: reads pair densities out of a transform.
    fft_index: jnp.ndarray
    #: ``(nbody,)`` ``sqrt(v(G)) = sqrt(8 pi / |G|^2)`` in Rydberg units, the
    #: factor that symmetrises ``chi_0``.
    sqrt_coulomb: jnp.ndarray
    #: ``(nbody,)`` the position of ``-G`` in this same list. A G-sphere is
    #: closed under inversion whatever the crystal's symmetry, so this always
    #: exists; it is what gives the antiresonant term for free.
    reflection: jnp.ndarray
    #: ``(nbody, 3)`` the signed Miller indices, carried rather than recovered
    #: from :attr:`fft_index`. A flat index is a statement about *one* FFT box,
    #: and the exchange-correlation kernel is built on the **dense** one where
    #: these vectors were selected on the smooth one -- so anything that has to
    #: index a second box needs the integers, not the wrapping.
    miller: jnp.ndarray
    ecut: float = eqx.field(static=True)

    @property
    def nbody(self) -> int:
        return self.fft_index.shape[0]

    @property
    def nm(self) -> int:
        """The full matrix dimension: three head directions plus the body."""
        return 3 + self.nbody


def response_sphere(calculation, ecut: float) -> ResponseSphere:
    """The G-vectors inside ``ecut`` (Ry), taken from the smooth set.

    The **smooth** set and not the dense one, because a pair density is formed
    on the smooth grid -- that is where the wavefunctions live -- and a
    component the smooth box cannot hold is an aliased one. The smooth cutoff is
    ``4 ecutwfc``, which is exactly the largest ``|G|`` a product of two
    wavefunctions has, so nothing is lost by the restriction.
    """
    gvectors = calculation.basis.smooth
    cell = calculation.system.cell
    g2 = np.asarray(gvectors.kinetic(cell))  # |G|^2 in 1/bohr^2 == Ry
    if ecut < 0.0:
        raise ValueError(
            f"the response cutoff cannot be negative, got {ecut}: it selects "
            "the G-vectors chi_0 is a matrix over (Elk's gmaxrf), and zero "
            "means the head alone"
        )
    inside = np.flatnonzero(g2 <= ecut)
    body = inside[inside != 0]  # G = 0 is the head, not a body entry

    miller = np.asarray(gvectors.miller)
    where = {tuple(m): n for n, m in enumerate(miller[body])}
    try:
        reflection = np.array([where[tuple(-m)] for m in miller[body]], dtype=np.int32)
    except KeyError as exc:  # pragma: no cover -- a G-sphere is always closed
        raise RuntimeError(
            "the response G-set is not closed under G -> -G, which a sphere "
            f"sorted by |G|^2 always is; missing {exc}"
        ) from exc

    precision = cell.precision
    return ResponseSphere(
        fft_index=jnp.asarray(np.asarray(gvectors.fft_index)[body]),
        sqrt_coulomb=precision.as_real(np.sqrt(E2 * FPI / g2[body])),
        reflection=jnp.asarray(reflection),
        miller=jnp.asarray(miller[body]),
        ecut=float(ecut),
    )


class ChiZero(eqx.Module):
    """``v^1/2 chi_0 v^1/2`` at every requested frequency, and what built it.

    ``x`` is ``(nw, nm, nm)``. The leading 3x3 block of each frequency's matrix
    is the head -- the optical limit, resolved by direction -- and the rest is
    the body over ``sphere``'s G-vectors.
    """

    x: jnp.ndarray
    frequencies: jnp.ndarray  # (nw,) complex: omega + i eta
    sphere: ResponseSphere
    #: How many valence-conduction pairs went into it, per k-point, summed over
    #: the spin channels. Reported because band truncation is this phase's one
    #: unrefusable error.
    npairs: int = eqx.field(static=True)
    #: The occupied count, the deeper channel's at ``nspin = 2``
    #: (:attr:`occupied` has both).
    nocc: int = eqx.field(static=True)
    nbnd: int = eqx.field(static=True)
    #: One occupied count per spin channel, ``(nocc,)`` at ``nspin = 1``.
    occupied: tuple = eqx.field(static=True, default=())
    #: Whether the occupations were smeared, so that the matrix carries the
    #: partly filled pairs and the intraband (Drude) head block, and has no
    #: finite value at ``omega + i eta = 0``. For a smeared run
    #: :attr:`occupied` is each channel's last band carrying weight, plus one.
    smeared: bool = eqx.field(static=True, default=False)

    @property
    def nm(self) -> int:
        return self.x.shape[-1]

    @property
    def head(self) -> jnp.ndarray:
        """``(nw, 3, 3)``: the optical block, which is what a spectrum reads."""
        return self.x[:, :3, :3]


def require_a_sum_over_states_regime(calculation) -> None:
    """What P37's ``chi_0`` refuses, checked on the calculation alone.

    Each of these is a missing term rather than a missing convenience, and each
    would otherwise produce a plausible spectrum:

    ``q != 0``
        not an argument here at all yet. The perturbed states would live at
        ``k + q``, which is the two-sphere machinery P19 built for spin spirals.
    ultrasoft and PAW
        ``<u_i| e^{-iG.r} |u_j>`` is not the plane-wave overlap when the charge
        is not all in ``|psi|^2``: the augmentation charge ``Q_ij(q+G)`` enters
        every matrix element. **It has been tried and it does not close**
        (``PLAN.md`` P40): with ``Q_ij(G)`` in the body -- gathered from the
        dense table by Miller index, and multiplied by the cell volume, which
        is a factor of 265 and was the first thing wrong -- and the head's
        ``q``-linear ``dpqq`` term beside it, ultrasoft silicon's ``eps_M(0)``
        is still **2.1%** from the Sternheimer solve where a norm-conserving
        control on the same machinery is at **0.06%**. So a third term is
        missing and the refusal stands until it is found; what is *excluded* is
        worth as much as what is not.

        **A second attempt reached ``Q_ij(G)`` by another route and disagrees
        with the first** (``PLAN.md`` P94): through
        ``topology/augmentation.py``'s ``augmentation_at_q`` rather than a
        gather from the dense table, the same cell at the same ``nbnd = 60``
        gives 57.200 with a residual of **+0.540** against P40's 55.5 and
        -1.20. And the residual is positive at every band count and does not
        tend to zero where the norm-conserving control does, so it is neither
        a truncation nor a missing normalisation. Only the *body* was corrected
        there; the head still carried no ``-e_n dS/dk`` and no ``dpqq``.

        **The third term was the head, and an ultrasoft insulator is no
        longer refused** (I94). With the body's ``Q_ij(G)`` from
        ``augmentation_at_q`` (P94's route) and the head built from the
        generalised velocity ``<n|dH_a - e_m dS_a|m> + (e_m - e_n) K^a_nm``
        (P99, which also holds the augmentation dipole ``dpqq``), the same
        cell at ``ecut_response = 8`` gives residuals of **-0.3369 and
        -0.0522** at 30 and 60 bands, negative and falling by 6.5, against the
        norm-conserving control's -0.2885 and -0.0601 on the same cell; the
        plain head reproduces P94's +0.2498 and +0.5405. Still refused: PAW
        (its one-centre Hartree and exchange-correlation terms are a
        different term), and an ultrasoft run that is smeared, polarized or a
        spinor, for want of a number.
    tetrahedron occupations
        the ``f_i - f_j`` weight kills the ``i = j`` term, so the intraband
        (Drude) response has to be added by hand (Elk's ``tddftlr`` does not
        add it at all), and its weight is a Fermi-surface delta function the
        tetrahedron method does not supply. A **smeared** run, metal or gapped
        cell, is not refused: its pair list runs over the partly filled
        window (:func:`_band_windows`), its degenerate pairs and its ``i = j``
        terms are the Drude head block :func:`_drude_block`, built from the
        smearing's own delta, and its weights are the smeared ``wg``. Checked
        in ``tests/regression/test_tddft_metal.py``: the Drude weight is the
        optical conductivity's plasma frequency, the f-sum rule converges onto
        the diamagnetic weight with the k-mesh on aluminium, and silicon
        smeared falls onto its fixed-occupation spectrum as ``degauss`` falls.
    a spinor with a moment (``nspin_mag = 4``)
        the kernel acquires spin components (Elk's ``nscfxc``, and a separate
        routine ``tddftsplr.f90``): the charge couples to the magnetization
        response through ``f_xc``. Collinear ``nspin = 2`` is *not* refused:
        the matrix is the sum of the two channels' (see the module docstring),
        and what has spin components there is the kernel, which
        ``alda_matrix`` refuses on a polarized density. A **nonmagnetic** spinor, with or without
        spin-orbit coupling, is not refused: its charge-only pair density is
        the spin trace, and that is the whole response. Until 2026-10-16 every
        spinor was refused, by the collinear branch, since a noncollinear
        :class:`~defuqumat.scf.driver.Calculation` reads ``nspin = 4``.
    a spin spiral
        the two components live on spheres at ``k +- q/2``, so a pair density
        is not one gather. Checked first, since a spiral carries a moment too.
    gamma-only storage
        **not refused**: the pair densities go through the full-sphere
        ``g_to_r`` and ``dH/dk`` leaves the half sphere, so
        :func:`independent_response` puts the states on the whole sphere first
        (:func:`~defuqumat.scf.whole_sphere.unfold_gamma_states`), and the
        workflow moves the calculation there before its fixed-density solve.
    a reduced k-set
        **not refused** when the group it was cut with tiles it: the wedge's
        matrix is averaged over that group with both G labels rotated at once
        (:func:`_completed`). A magnetic group, a reduced set on a ``nosym`` run
        and a set the group does not tile (a shifted grid) are refused by
        :func:`~defuqumat.system.wedge.wedge_group`.
    """
    system = calculation.system
    if calculation.is_paw:
        raise NotImplementedError(
            "a sum-over-states chi_0 with a PAW dataset is not implemented: "
            "beside the augmentation charge Q_ij(G) of every matrix element, "
            "which the ultrasoft branch carries, PAW's Hartree and "
            "exchange-correlation energies have one-centre parts on the "
            "spheres whose response the plane-wave matrix does not hold. Use a "
            "norm-conserving or an ultrasoft dataset"
        )
    if calculation.is_ultrasoft and (system.noncolin or _is_smeared(system)
                                     or int(system.nspin) != 1):
        raise NotImplementedError(
            "a sum-over-states chi_0 with an ultrasoft dataset is measured for "
            "an unpolarized insulator only (silicon against the Sternheimer "
            "solve): a spinor's augmentation pairs the projections of the two "
            "components through qq_so, and a smeared or spin-polarized "
            "ultrasoft run reaches the same matrix with no number behind it"
        )
    if calculation.spiral:
        # Checked before the spin regime, which a spiral also fails (it always
        # carries a moment, so ``nspin_mag = 4``): the spiral is the one regime
        # whose missing term is a different one, and it should say so.
        raise NotImplementedError(
            "chi_0 on a spin spiral is not implemented: the two spinor "
            "components live on spheres centred at k +- q/2, so a plane-wave "
            "matrix element is not a single gather"
        )
    if system.noncolin and calculation.nspin_mag != 1:
        raise NotImplementedError(
            "a sum-over-states chi_0 of a textured spinor (nspin_mag = 4, a "
            "run carrying a magnetization) is not implemented: the "
            "charge-charge chi_0 is only the first of four spin-trace blocks, "
            "and once the ground state carries a moment the exchange-"
            "correlation kernel couples the charge to the magnetization "
            "response, which nothing here builds. The Sternheimer referee "
            "refuses the same regime (P83). A nonmagnetic spinor, with or "
            "without spin-orbit coupling, is supported"
        )
    if getattr(calculation, "is_hubbard", False):
        raise NotImplementedError(
            "chi_0 with a Hubbard U is not implemented: the Hubbard term "
            "responds to the density too (QE's adddvhubscf), and leaving it "
            "out of the screening is a silent approximation rather than a "
            "missing feature"
        )
    if _is_smeared(system) and (system.noncolin or is_reduced(system.kpoints)):
        raise NotImplementedError(
            "a smeared chi_0 on a spinor or on a symmetry-reduced k-set is not "
            "implemented: the pair window and the Drude block are written for "
            "both, but the numbers behind the smeared path are for a scalar run "
            "on the whole grid (aluminium and silicon), and these reach the "
            "same matrix with none. Pass the whole unshifted grid with nosym, "
            "or use fixed occupations"
        )
    if _is_smeared(system) and int(system.nspin) == 2:
        raise NotImplementedError(
            "a smeared chi_0 at nspin = 2 is not implemented: the pair window, "
            "the Drude block and the per-channel Fermi level are written for "
            "it, but the number behind the smeared path is unpolarized "
            "(aluminium and silicon), and a polarized metal reaches the same "
            "matrix with none. Use nspin = 1, or fixed occupations"
        )
    if not _is_an_insulator(system) and not _is_smeared(system):
        # A smeared run carries its intraband term (:func:`_drude_block`); a
        # tetrahedron run has no smeared delta to build it from.
        raise NotImplementedError(
            "a sum-over-states chi_0 with tetrahedron occupations is not "
            "implemented: the occupation difference f_i - f_j vanishes for "
            "i = j, so the intraband (Drude) term has to be added by hand, and "
            "its weight is a Fermi-surface delta function that the tetrahedron "
            "method does not supply (tetra.f90 integrates the step function "
            "itself). Run with occupations = 'smearing', where the Drude term "
            "is built from the smearing's own delta"
        )
    # A symmetry-reduced k-set is completed by the group it was cut with, both
    # G labels rotated at once (:func:`_completed`); a set that group does not
    # tile and a magnetic group are still refused there.
    if is_reduced(system.kpoints) and (int(system.nspin) != 1 or system.noncolin):
        raise NotImplementedError(
            "chi_0 on a symmetry-reduced k-set is measured for an nspin = 1 "
            "run only (silicon and AlAs against their whole grids): a "
            "collinear nspin = 2 or a spinor wedge reaches the same completion "
            "with no number behind it. Pass the whole unshifted grid as kpoints="
        )
    wedge_group(calculation.system, quantity="chi_0")


def _is_an_insulator(system) -> bool:
    occupations = getattr(system, "occupations", "fixed")
    return str(occupations).lower() in ("fixed", "from_input")


def _is_smeared(system) -> bool:
    """Whether the occupations are a smooth function of energy, with a delta to match.

    The same test :func:`~defuqumat.response.conductivity._fermi_surface_delta`
    makes: a smearing with a positive ``degauss``. That is what makes a
    degenerate pair's two orderings cancel onto the Drude form, and what
    supplies the Drude term's delta function.
    """
    scheme = str(getattr(system, "occupations", "fixed")).lower()
    degauss = float(getattr(system, "degauss", 0.0) or 0.0)
    return "smearing" in scheme and degauss > 0.0


def _kpoints_are_reduced(calculation) -> bool:
    """Whether the k-set is a wedge rather than the whole grid.

    Read off the weights: an unreduced grid gives every k-point the same weight,
    and a reduction is exactly what makes them differ. That is a weaker test
    than comparing against the grid the input asked for, and it is the one that
    survives an explicit ``K_POINTS`` list -- which is how the closed-grid cases
    in ``tests/data/qe`` are written.
    """
    return is_reduced(calculation.system.kpoints)


def independent_response(
    calculation,
    wavefunctions,
    eigenvalues,
    v_scf,
    frequencies,
    *,
    ecut_response: float,
    broadening: float,
    scissor: float = 0.0,
    k_batch: int | None | str = "default",
    pair_batch: int | None | str = "default",
    w_batch: int | None | str = "default",
) -> ChiZero:
    """``v^1/2 chi_0 v^1/2`` over a response sphere and a frequency grid.

    Args:
        calculation: the :class:`~defuqumat.scf.driver.Calculation` the states
            belong to, on the whole k-grid or on a wedge, which is completed by
            its group (:func:`_completed`).
        wavefunctions: ``(nspin, nk, nbnd, npwx)``, from a fixed-density run
            with empty states -- :func:`~defuqumat.workflows.nscf.
            fixed_density_states`. ``nbnd`` is the convergence parameter of this
            whole phase and there is no refusal that can catch it being too
            small, which is why :attr:`ChiZero.nbnd` is carried out.
        eigenvalues: ``(nspin, nk, nbnd)`` or the squeezed ``(nk, nbnd)``, Ry.
        v_scf: the converged potential, which ``dH/dk`` is built at.
        frequencies: ``(nw,)`` Ry. Real is the usual case and ``broadening``
            supplies the imaginary part; a **complex** array is accepted too and
            has ``broadening`` added to it, which is how a caller asks for one
            point at ``eta = 0`` beside a broadened grid. Elk's ``wrf`` is
            complex for the same reason.
        ecut_response: the cutoff (Ry) selecting the G-vectors of the matrix.
        broadening: ``eta``, Ry -- Elk's ``swidth``. Every frequency carries it,
            **including omega = 0**: ``init3.f90`` sets the bootstrap's static
            point to ``0 + i swidth`` rather than to zero, and the kernel is
            built there.
        scissor: a rigid shift (Ry) of the empty states, PRL 107, 186401's
            Eq. (3). The velocity matrix elements are renormalised with it.
        k_batch: how many k-points each chunk of the walk over k holds;
            ``"default"`` is the calculation's own ``k_batch`` and ``None`` the
            whole axis in one chunk (:func:`~defuqumat.response.walk.chunk_size`).
        pair_batch: how many occupied-empty **pair densities** are in flight
            inside one k-point. One of them is a whole complex field on the
            smooth grid, so this is the dial that decides whether the phase's
            largest array is ``npairs`` FFT boxes or a handful; the default is
            the band dial (``DEFUQUMAT_BAND_BATCH``, one on a CPU), since a pair
            density and a band in real space are the same object -- except on
            an accelerator, where the band dial's "all" would be every pair,
            and the default is a chunk budgeted to about 256 MB of boxes
            (:func:`~defuqumat.batching.resolve_pair_batch`). Every chunk goes
            through the same transform, so the answer does not depend on it.
        w_batch: how many **frequencies** each k-point's assembly forms at
            once. The whole axis (the CPU default) is one contraction whose
            intermediate is ``2 npairs nm min(nw, nm)`` complex; a chunk of
            ``n`` forms ``n`` pair-weighted blocks, ``2 npairs nm`` each, and
            contracts them -- one matrix product per frequency, as the module
            docstring's trade describes. On an accelerator the default is
            budgeted (:func:`~defuqumat.batching.resolve_w_batch`). The
            frequencies are independent, so it moves nothing beyond round-off.

    Returns:
        A :class:`ChiZero`. On the whole grid nothing is symmetrised, since
        there is nothing to put back; on a wedge the sum is completed by the
        group it was cut with (:func:`_completed`).
    """
    require_a_sum_over_states_regime(calculation)
    # A half sphere's pair densities through the full-sphere ``g_to_r`` lose
    # the conjugate half, and ``dH/dk`` is not a real operator; both are the
    # full-sphere ones on the unfolded states (``scf/whole_sphere.py``).
    calculation, wavefunctions = unfold_gamma_states(calculation, wavefunctions)

    # The states stay where they are, a streamed store in host memory or a
    # device array, and cross to the device a k-chunk at a time below.
    if wavefunctions.ndim == 3:
        wavefunctions = wavefunctions[None]
    eigenvalues = jnp.asarray(eigenvalues)
    if eigenvalues.ndim == 2:
        eigenvalues = eigenvalues[None]

    nbnd = int(eigenvalues.shape[-1])
    smeared = _is_smeared(calculation.system)
    precision = calculation.system.cell.precision
    zomega = jnp.asarray(frequencies) + 1j * precision.as_real(broadening)
    zomega = zomega.astype(precision.complex)
    if smeared:
        if scissor:
            raise NotImplementedError(
                "a scissors shift on a smeared chi_0 is not implemented: the "
                "shift moves the empty states, and with fractional occupations "
                "there is no empty set to move. Use occupations = 'fixed' for "
                "a scissor-corrected spectrum"
            )
        if np.any(np.abs(np.asarray(zomega)) == 0.0):
            raise ValueError(
                "a smeared chi_0 carries the intraband (Drude) term "
                "wp^2/(omega + i eta)^2, which is infinite at omega + i eta = "
                "0: give every frequency a broadening"
            )
        windows = _band_windows(calculation, eigenvalues, nbnd)
        counts = tuple(high for _, high in windows)
    else:
        counts = _occupied_bands(calculation, eigenvalues, nbnd)
        windows = tuple((nocc, nocc) for nocc in counts)
    sphere = response_sphere(calculation, ecut_response)
    # An ultrasoft dataset's augmentation charge on the response sphere, at
    # ``q = 0`` (the body's ``G != 0`` only); PAW is refused above.
    augmented = getattr(calculation, "augmentation", None) is not None
    factors = (augmentation_factors(calculation, np.zeros(3), sphere)
               if augmented and sphere.nbody else None)

    # The scissors shift moves the empty states and nothing else. It is applied
    # here rather than by the caller so that the matrix elements below can be
    # renormalised consistently -- the two are one approximation, not two. Each
    # channel's empty states start at its own count.
    shifted = eigenvalues
    if not smeared:
        for spin, nocc in enumerate(counts):
            shifted = shifted.at[spin, :, nocc:].add(precision.as_real(scissor))

    # Built on the whole set for its refusals and for the point it
    # differentiates around; each chunk's operator is handed its rows of
    # ``kcart`` (:mod:`defuqumat.response.walk`).
    kcart = np.asarray(VelocityOperator(calculation, v_scf).kcart)

    weights, levels = calculation.occupations(eigenvalues)
    # (nspin, nk, nbnd), summing to nelec over both channels: one electron a
    # band at nspin = 2, two at nspin = 1.
    weights = np.asarray(weights)

    # One pair list per channel, ``(i, j)`` with ``i`` occupied in *that*
    # channel. A channel with nothing occupied (a hydrogen atom's minority)
    # has no pair and is skipped rather than given an empty static shape. A
    # smeared channel's list runs over its partly filled window as well
    # (:func:`_band_windows`), and its pairs closer than ``DEGENERACY_TOL``
    # go to the Drude block instead of the interband sum (:func:`_drude_block`).
    channels = [(spin,) + _pairs(low, high, nbnd)
                for spin, (low, high) in enumerate(windows) if high > 0]
    tol = _degeneracy_tol() if smeared else None
    # ``w_k delta(E_F - e)`` per state, ``(nspin, nk, nbnd)`` in 1/Ry, the
    # Drude term's weight; ``None`` for an insulator, which has none.
    drude = _drude_weights(calculation, eigenvalues, levels) if smeared else None

    grid = calculation.basis.smooth.grid
    volume = calculation.system.cell.volume
    batch = chunk_size(calculation, k_batch)
    itemsize = np.dtype(precision.complex).itemsize

    def channel_term(rows, columns):
        """One spin channel's ``one_k``, with its own pair list and dials."""
        pairs = resolve_pair_batch(
            pair_batch, npairs=int(rows.size),
            box_bytes=int(np.prod(grid)) * itemsize)
        # One frequency's pair-weighted block, ``(2 npairs, nm)`` complex.
        chunk = resolve_w_batch(
            w_batch, nw=int(zomega.shape[0]),
            block_bytes=2 * int(rows.size) * sphere.nm * itemsize)

        def one_k(arrays):
            psi, fft_index, band_mask, eig, occupation, element, *rest = arrays
            weight = rest.pop(0) if drude is not None else None
            augmentation = (factors, rest.pop(0)) if factors is not None else None
            vectors, scalars = _pair_terms(
                psi, fft_index, band_mask, eig, occupation, element,
                rows, columns, sphere, grid, volume, zomega, scissor, precision,
                pairs, tol, augmentation,
            )
            # ``(nw, nm, nm)``: one matrix product per frequency, the pair
            # axis contracted away. This is the whole frequency cost.
            x = _assemble(scalars, vectors, chunk)
            if weight is not None:
                x = x.at[:, :3, :3].add(
                    _drude_block(weight, eig, element, zomega, volume, tol))
            return x

        return one_k

    terms_by_spin = [(spin, channel_term(rows, columns))
                     for spin, rows, columns in channels]

    # The potential on the smooth grid, k-independent, built once here rather
    # than in every chunk and direction.
    terms = calculation.local_terms(v_scf)

    def elements_of(rowset, psi, kcart):
        """One chunk's ``<m|dH/dk_a|n>``, ``(nspin, chunk, 3, nbnd, nbnd)``."""
        # ``dH/dk``, the head's ingredient and the reason this is not a pure
        # transcription. One jvp per cartesian direction over the chunk's rows;
        # each channel's states see that channel's Hamiltonian.
        velocity = VelocityOperator(with_rows(calculation, rowset), v_scf, kcart=kcart,
                                    local_terms=terms)
        elements = velocity.matrix_elements(psi, sequential=True)  # (3, nspin, chunk, nb, nb)
        return jnp.moveaxis(elements, 0, 2)

    # An ultrasoft dataset's head is the **generalised** velocity,
    # ``<n|dH_a - e_m dS_a|m> + (e_m - e_n) K^a_nm`` (P99,
    # :meth:`~defuqumat.response.velocity.VelocityOperator.
    # generalised_matrix_elements`), whose ratio to ``e_n - e_m`` is the dipole
    # of the true states ``T|psi>``, built in the conductivity's two stages.
    # ``dpqq``, k-independent, is built here once.
    dipole = None
    if augmented:
        from defuqumat.response.efield import _augmentation_dipole

        dipole = _augmentation_dipole(calculation)

    def velocity_on(rowset, kcart):
        return VelocityOperator(with_rows(calculation, rowset), v_scf, kcart=kcart,
                                dipole=dipole, local_terms=terms)

    def tangents_of(rowset, psi, energies, kcart):
        """``<n|dH_a - e_m dS_a|m>`` on one chunk, ``(3, nspin, chunk, nb, nb)``."""
        return velocity_on(rowset, kcart).tangent_elements(psi, energies, sequential=True)

    def connections_of(tangents, rowset, psi, kcart):
        """The augmentation dipole's connection, and ``<beta|psi>`` per channel."""
        connections = velocity_on(rowset, kcart).connections(psi, sequential=True)
        chunk = with_rows(calculation, rowset)
        projections = jnp.stack([state_projections(chunk, psi[spin])
                                 for spin in range(psi.shape[0])])
        return connections, projections

    def contract(elements, *rest):
        """One chunk's share of ``(nw, nm, nm)``, both spin channels summed."""
        projections = None
        if augmented:
            connections, projections = rest[0]
            psi, fft_index, mask, energies, occupation, bare, *extra = rest[1:]
            elements = jnp.moveaxis(
                with_connections(elements, bare, connections), 0, 2)
        else:
            psi, fft_index, mask, energies, occupation, *extra = rest
        total = 0.0
        for spin, one_k in terms_by_spin:
            more = tuple(x[spin] for x in extra)
            if projections is not None and factors is not None:
                more = more + (projections[spin],)
            total = total + sum_k(
                one_k, (psi[spin], fft_index, mask, energies[spin],
                        occupation[spin], elements[spin]) + more, batch=batch)
        return total

    shifted = np.asarray(shifted)
    fft_index = np.asarray(calculation.fft_index)
    mask = np.asarray(calculation.basis.planewaves.mask)

    bare = np.asarray(eigenvalues)

    def arguments(chunk_rows, live):
        # The padded rows are a repeat of a real one and carry zero ``wg``,
        # which every pair's weight ``(wg_i - wg_j)/Omega`` is linear in, and
        # zero Drude weight, which the Drude block is linear in.
        psi = store_rows(wavefunctions, chunk_rows)
        rowset, kc = row_leaves(calculation, chunk_rows), jnp.asarray(kcart[chunk_rows])
        last = (psi, jnp.asarray(fft_index[chunk_rows]), jnp.asarray(mask[chunk_rows]),
                jnp.asarray(shifted[:, chunk_rows]),
                jnp.asarray(padded(weights, chunk_rows, live, axis=1)))
        if augmented:
            # The *unshifted* eigenvalues for the generalised velocity's
            # ``e_m dS``, the ones the eigenproblem was solved with.
            last = last + (jnp.asarray(bare[:, chunk_rows]),)
        if drude is not None:
            last = last + (jnp.asarray(padded(drude, chunk_rows, live, axis=1)),)
        if augmented:
            energies = jnp.asarray(bare[:, chunk_rows])
            return (rowset, psi, energies, kc), (rowset, psi, kc), last
        return (rowset, psi, kc), last

    stages = ((tangents_of, connections_of, contract) if augmented
              else (elements_of, contract))
    x = walk(int(weights.shape[1]), batch, arguments, *stages)
    x = _completed(x, sphere, wedge_group(calculation.system, quantity="chi_0"))
    return ChiZero(
        x=x,
        frequencies=zomega,
        sphere=sphere,
        npairs=int(sum(rows.size for _, rows, _ in channels)),
        nocc=max(counts),
        nbnd=nbnd,
        occupied=tuple(counts),
        smeared=smeared,
    )


def _completed(x, sphere, group):
    """A wedge's ``(nw, nm, nm)`` averaged over the group it was cut with.

    **Both labels at once**, which is what the refusal this replaces named. For
    an operation ``c -> c M + t`` (crystal rows, :class:`~defuqumat.system.
    symmetry.Symmetries`'s convention) the pair density at the image k-point is
    ``rho^{gk}(m) = exp(-2 pi i m.t) rho^k(M m)``, the Coulomb factor being a
    function of ``|G|`` alone, and the head, the ``q -> 0`` limit of the same
    vector, rotates as the velocity does, ``r(gk) = R r(k)``. So each operation
    acts on the matrix as ``U X U^dagger`` with ``U`` the rotation on the head
    and a phased permutation on the body.

    **Time reversal is a transpose with a parity**: ``r_G(-k) = conj(r_{-G}(k))``
    and ``r_a(-k) = -conj(r_a(k))``, so the time-reversed image of a k-point's
    term is ``P X^T P`` with ``P`` the reflection ``G -> -G`` on the body and
    ``-1`` on the head -- the reflection :attr:`ResponseSphere.reflection`
    already carries for the antiresonant term.

    Every term is ``jnp`` and the maps are static, so a derivative through the
    matrix passes through the average.
    """
    if group is None:
        return x
    miller = np.asarray(sphere.miller, dtype=int).reshape(-1, 3)
    nbody = miller.shape[0]
    where = {tuple(m): n for n, m in enumerate(miller)}
    head = np.arange(3)
    reflect = np.concatenate([head, 3 + np.asarray(sphere.reflection, dtype=int)])
    sign = np.concatenate([-np.ones(3), np.ones(nbody)])
    transposed = jnp.swapaxes(x, -1, -2)[:, reflect[:, None], reflect[None, :]]
    reversed_ = transposed * jnp.asarray(sign[:, None] * sign[None, :], dtype=x.dtype)

    total = jnp.zeros_like(x)
    for rotation, crystal, shift, anti in zip(group.rotations, group.crystal,
                                              group.translations, group.antiunitary):
        images = np.rint(miller @ np.asarray(crystal).T).astype(int)
        try:
            index = np.array([where[tuple(m)] for m in images], dtype=int)
        except KeyError:
            raise NotImplementedError(
                "chi_0 on a symmetry-reduced k-set needs a response sphere the "
                "point group maps onto itself, and this one is cut through a "
                "shell of equal |G| by round-off at ecut_response: move the "
                "cutoff off the shell, or pass the whole grid"
            ) from None
        full = np.concatenate([head, 3 + index])
        phase = np.concatenate([
            np.ones(3, dtype=complex),
            np.exp(-2j * np.pi * (miller @ np.asarray(shift, dtype=float))),
        ])
        source = reversed_ if anti else x
        # The precision of ``x`` and not a literal one: the phases and the
        # rotation are host tables, cast to the matrix they act on.
        term = source[:, full[:, None], full[None, :]] * jnp.asarray(
            phase[:, None] * np.conj(phase)[None, :], dtype=x.dtype)
        rot = jnp.asarray(np.asarray(rotation, dtype=float), dtype=x.dtype)
        term = term.at[:, :3, :].set(jnp.einsum("ab,wbn->wan", rot, term[:, :3, :]))
        term = term.at[:, :, :3].set(jnp.einsum("wnb,ab->wna", term[:, :, :3], rot))
        total = total + term
    return total / group.order


def _assemble(scalars, vectors, batch):
    """``(nw, nm, nm)``: ``sum_p s_wp v_pa conj(v_pb)``, ``batch`` frequencies at a time.

    ``batch = None`` is one ``einsum`` over the whole axis, and XLA contracts
    it through ``v_pa conj(v_pb)`` first -- a ``(2 npairs, nm, nm)`` block,
    measured at 90 MB of a 131 MB temporary on silicon at ``ecut_response =
    8`` with 60 bands and 200 frequencies -- whenever ``nw`` exceeds ``nm``,
    and through ``s_wp v_pa``, ``(nw, 2 npairs, nm)``, otherwise. A chunk forms
    ``s_wp v_pa`` for its own frequencies only and contracts that with
    ``conj(v)``, so what is in flight is ``batch`` blocks of ``(2 npairs,
    nm)`` whatever ``nw`` and ``nm`` are (``notes/GPU-MEMORY-NEXT.md`` item 12). The
    frequencies are independent, so the chunk moves nothing beyond the order
    of a sum.
    """
    conjugate = jnp.conj(vectors)
    if batch is None or batch >= scalars.shape[0]:
        return jnp.einsum("wp,pa,pb->wab", scalars, vectors, conjugate)

    def one_frequency(weights):
        return (weights[:, None] * vectors).T @ conjugate

    return map_windows(one_frequency, scalars, batch=batch)


def _pairs(low: int, high: int, nbnd: int):
    """The ordered ``(i, j)``, ``i < j``, whose occupations can differ.

    ``i`` runs below ``high``, the first band that is empty everywhere, and
    ``j`` from ``low``, the first band that is not full everywhere, so a pair
    of two full bands or of two empty ones, whose weight ``f_i - f_j`` is zero,
    is never listed. An insulator has ``low = high = nocc`` and this is the
    valence-conduction rectangle, in the same order.

    Only that half is enumerated. The reversed pairs -- which
    ``genvchi0.f90``'s loop runs explicitly, and which are the antiresonant
    term -- come from reflecting these, in :func:`_pair_terms`, without a
    second transform.
    """
    rows, columns = [], []
    for i in range(high):
        start = max(low, i + 1)
        rows.extend([i] * (nbnd - start))
        columns.extend(range(start, nbnd))
    return np.asarray(rows, dtype=int), np.asarray(columns, dtype=int)


def _pair_terms(psi, fft_index, band_mask, eig, occupation, element,
                rows, columns, sphere, grid, volume, zomega, scissor,
                precision, pair_batch=None, tol=None, augmentation=None):
    """One k-point's pair vectors ``r`` and their frequency weights.

    Returns ``(vectors, scalars)`` of shapes ``(2 npairs, nm)`` and
    ``(nw, 2 npairs)``: the resonant pairs followed by the antiresonant ones,
    which are the same states read the other way round.

    ``tol`` is a smeared run's degeneracy tolerance: a pair closer than it
    carries no weight here, because its two orderings together are the Drude
    form and are counted in :func:`_drude_block`. ``None`` (an insulator)
    masks nothing.

    ``augmentation`` is an ultrasoft dataset's ``(factors, projections)``:
    ``q_ij(G)`` on the response sphere, ``(nbody, nkb, nkb)``
    (:func:`~defuqumat.tddft.spinchi0.augmentation_factors` at ``q = 0``), and
    this k-point's ``<beta_i|psi_n>``, ``(nbnd, npol, nkb)``. The pair density
    then gains ``sum_ij <psi_i|beta_a> q_ab(G) <beta_b|psi_j>``, the charge
    that is not in ``|psi|^2``.
    """
    if sphere.nbody:
        # The states in real space, once, one field per spinor component:
        # ``(nbnd, npol, n1, n2, n3)``. A spinor is ``(up, down)`` on the one
        # sphere at ``k``, each ``npwx`` long, so both components go through
        # the same ``fft_index`` -- which is exactly what a spiral, whose two
        # components sit on spheres at ``k +- q/2``, does not allow. Padding
        # must be zeroed before the scatter, since padding entries share the
        # index of G = 0.
        npwx = band_mask.shape[-1]
        npol = psi.shape[-1] // npwx
        components = psi.reshape(psi.shape[:-1] + (npol, npwx))
        fields = g_to_r(components * band_mask, fft_index, grid)

        def one_pair(index):
            # ``<u_i| e^{-iG.r} |u_j>``, from one transform. The product is
            # formed on the smooth grid, which holds every G a product of two
            # wavefunctions has, and it is **the** allocation of this phase:
            # one whole complex box per pair, against ``nm`` numbers kept.
            # Forming every pair's at once is what ``pair_batch`` bounds.
            # For a spinor the pair density is the **spin trace**
            # ``sum_s conj(u_i,s) u_j,s``, the charge-charge block, which is
            # the whole of what the Coulomb and the unpolarized kernels see
            # when the ground state carries no moment (``nspin_mag = 1``).
            i, j = index
            product = jnp.sum(jnp.conj(fields[i]) * fields[j], axis=0)
            density = r_to_g(product, sphere.fft_index)
            if augmentation is not None:
                # Contracted as ``spinchi0``'s matrix element and the Berry
                # phase's augmented overlap are: the bra's projection
                # conjugated, the ket's not, the block between them.
                factors, projections = augmentation
                density = density + jnp.einsum(
                    "pa,gab,pb->g", jnp.conj(projections[i]),
                    factors.astype(density.dtype), projections[j])
            return density * sphere.sqrt_coulomb

        body = map_axis(one_pair, (jnp.asarray(rows), jnp.asarray(columns)),
                        batch=pair_batch)
    else:
        # **The head-only kernel needs no transform at all**, and the saving is
        # the whole of the phase's cost rather than a corner of it: with an
        # empty body there is no plane-wave matrix element to form, so the
        # states never go to real space and ``chi_0`` is built from the velocity
        # matrix elements alone. Writing the gather over an empty index would
        # still transform every pair and then keep none of it.
        body = jnp.zeros(rows.shape + (0,), dtype=psi.dtype)

    # The head. ``<u_i|du_j/dk_a> = <u_i|dH/dk_a|u_j> / (eps_j - eps_i)``, and
    # the Coulomb factor sqrt(8 pi / q^2) cancels the q the numerator is linear
    # in, which is what leaves a finite, direction-resolved 3x3 block.
    difference = eig[rows] - eig[columns]  # eps_i - eps_j, negative
    if tol is not None:
        # A smeared run's degenerate pairs, whose ``1/(eps_i - eps_j)`` is a
        # 0/0 that the two orderings cancel analytically. They are moved to the
        # Drude block whole, so here the division is made safe and the weight
        # zeroed, ``jnp.where`` on both so that no tangent is ``0 * inf``.
        finite = jnp.abs(difference) > tol
        difference = jnp.where(finite, difference, 1.0)
    matrix = element[:, rows, columns]  # (3, npairs)
    if scissor:
        # ``getpmat.f90``: the eigenvalues have already moved, so the matrix
        # element has to move with them or <u|r|u'> silently changes. Here
        # ``i`` is occupied and ``j`` empty, which is Elk's second branch.
        matrix = matrix * (difference / (difference + scissor))[None, :]
    head = -jnp.sqrt(precision.as_real(E2 * FPI)) * matrix / difference[None, :]
    vectors = jnp.concatenate([head.T, body], axis=-1)  # (npairs, nm)

    # The reversed pair. ``<u_j|e^{-iG.r}|u_i> = conj(<u_i|e^{+iG.r}|u_j>)``,
    # so the body is this body reflected through the origin and conjugated;
    # the head reflects with a sign, because ``rho_0`` is linear in ``q``.
    reflected = jnp.concatenate(
        [-jnp.conj(vectors[:, :3]),
         jnp.conj(vectors[:, 3:][:, sphere.reflection])],
        axis=-1,
    )

    weight = (occupation[rows] - occupation[columns]) / volume
    if tol is not None:
        weight = jnp.where(finite, weight, 0.0)
    resonant = weight[None, :] / (difference[None, :] + zomega[:, None])
    antiresonant = -weight[None, :] / (-difference[None, :] + zomega[:, None])
    return (
        jnp.concatenate([vectors, reflected], axis=0),
        jnp.concatenate([resonant, antiresonant], axis=-1),
    )


def _degeneracy_tol() -> float:
    """Below this gap (Ry) a smeared pair is intraband: the conductivity's own.

    :data:`~defuqumat.response.conductivity.DEGENERACY_TOL`, so that the two
    modules partition the pairs the same way and the Drude weight of one is
    the Drude weight of the other on the same states.
    """
    from defuqumat.response.conductivity import DEGENERACY_TOL

    return float(DEGENERACY_TOL)


def _drude_weights(calculation, eigenvalues, levels):
    """``w_k delta(E_F - e_nk)`` per state, ``(nspin, nk, nbnd)`` in 1/Ry.

    The smearing's own delta, ``w0gauss`` at the argument
    :func:`~defuqumat.response.conductivity._fermi_surface_delta` documents,
    which is minus the derivative of the occupations the SCF converged with.
    One Fermi level, or one per channel when ``tot_magnetization`` fixes the
    moment. ``w_k`` is the k-point weight with the spin degeneracy in it, the
    one ``wg`` is a fraction of.
    """
    from defuqumat.response.conductivity import _fermi_surface_delta

    energies = np.asarray(eigenvalues)
    nspin = energies.shape[0]
    if "fermi_energy_up" in levels:
        fermi = (levels["fermi_energy_up"], levels["fermi_energy_down"])
    else:
        fermi = (levels["fermi_energy"],) * nspin
    wk = np.asarray(calculation.system.kpoints.weights)
    return np.stack([
        wk[:, None] * _fermi_surface_delta(calculation, float(fermi[s]),
                                           energies[s], enabled=True)
        for s in range(nspin)
    ])


def _drude_block(weight, eig, element, zomega, volume, tol):
    """One k-point's intraband head, ``(nw, 3, 3)``: ``wp_ab^2(k) / (omega + i eta)^2``.

    **What ``f_i - f_j`` leaves out at ``q = 0``, as the limit of the same
    expression.** The ``i = j`` pair at ``q -> 0`` has the pair density
    ``<u_i,k|u_i,k+q> -> 1`` and the weight ``(f(e_k) - f(e_k+q)) / (e_k -
    e_k+q + z)``, with ``e_k+q - e_k = q.v``; times ``v(q) = 8 pi/q^2`` and
    expanded in ``q``, the first order is odd in ``v`` and cancels between
    ``k`` and ``-k``, and the second is

        X_ab^intra = (8 pi / Omega) sum_k w_k sum_n delta(E_F - e_n) v_a v_b / z^2,

    which is ``wp_ab^2 / z^2`` with ``wp^2`` in Ry^2 (the free-electron gas
    gives ``16 pi n``, ``hbar wp = sqrt(4 pi n)`` Hartree). Its wings are
    first order in ``v`` and cancel the same way, since ``rho_ii(-G) =
    conj(rho_ii(G))`` for any state, and its body vanishes as ``q``; so it is
    the head alone. ``eps_M = 1 - X_head`` then carries ``-wp^2/(omega + i
    eta)^2``, which is :class:`~defuqumat.response.conductivity.
    OpticalConductivity`'s ``-wp^2/(omega (omega + i gamma))`` at ``gamma =
    eta`` up to terms of order ``eta/omega``, with the same weight.

    **The multiplet block, not the diagonal** (rule D4): a pair degenerate
    within ``tol``, the pairs :func:`_pair_terms` excluded, contributes its
    two orderings' limit, ``-2 f' Re(v_a^ij v_b^ji) / z^2``, which is this
    same expression with ``v_a^nn v_b^nn`` widened to ``sum_m v_a^nm
    v_b^mn`` over the multiplet -- invariant under the rotation a degenerate
    eigensolver is free in, and :func:`~defuqumat.response.conductivity.
    _plasma_sum`'s partition exactly.
    """
    gap = eig[:, None] - eig[None, :]
    multiplet = (jnp.abs(gap) <= tol).astype(weight.dtype)
    block = jnp.real(jnp.einsum("n,nm,anm,bmn->ab", weight, multiplet,
                                element, element))
    return (E2 * FPI / volume) * block[None] / (zomega ** 2)[:, None, None]


def _band_windows(calculation, eigenvalues, nbnd: int) -> tuple:
    """Each channel's ``(low, high)``: the first band not full everywhere, the first empty everywhere.

    A smeared run's version of :func:`_occupied_bands`. "Full" and "empty"
    are within :data:`_OCCUPATION_TOLERANCE` of the fractional filling, which
    is ``genvchi0.f90``'s own threshold for dropping a pair, so a pair of two
    bands both outside the window is one that sum would drop. A gapped cell
    under a small smearing has ``low = high = nocc`` and is the insulator's
    list; a metal, or a gap the smearing's tail crosses, widens it.

    **The window must close below ``nbnd``**: a band set whose top band still
    carries weight has pairs above it that are missing, and nothing could say
    how much they carry.
    """
    weights, _ = calculation.occupations(jnp.asarray(eigenvalues))
    weights = np.asarray(weights)
    wk = np.asarray(calculation.system.kpoints.weights)
    filling = weights / np.where(wk > 0.0, wk, 1.0)[None, :, None]
    windows = []
    for spin in range(filling.shape[0]):
        partial = np.abs(filling[spin] - 1.0).max(axis=0) > _OCCUPATION_TOLERANCE
        carrying = np.abs(filling[spin]).max(axis=0) > _OCCUPATION_TOLERANCE
        low = int(np.argmax(partial)) if partial.any() else nbnd
        high = int(np.flatnonzero(carrying)[-1]) + 1 if carrying.any() else 0
        if high >= nbnd:
            raise ValueError(
                f"a smeared chi_0 needs bands above every occupied one: band "
                f"{nbnd - 1} of spin channel {spin} still carries filling "
                f"{np.abs(filling[spin][:, -1]).max():.3e} at some k-point. "
                "Run the fixed-density calculation with more bands (nbnd)"
            )
        windows.append((min(low, high), high))
    return tuple(windows)


def _occupied_bands(calculation, eigenvalues, nbnd: int) -> tuple:
    """How many bands each channel fills, and a check that the question is well posed.

    An insulator has the same number at every k-point, which is what makes the
    valence-conduction split a *static* shape and the pair list a compile-time
    object. A run where it is not is a metal, and metals are refused above --
    but the refusal reads the input's ``occupations``, and a fixed-occupation
    input can still be handed a state that crosses. This catches that.

    One count per spin channel (:func:`~defuqumat.scf.occupations.
    occupied_counts`), since a magnetic insulator fills its two channels to
    different depths; ``nelec / 2`` for both would put a majority state among
    the empty ones. A channel may be empty (``0``), but not every one.
    """
    counts = tuple(int(n) for n in occupied_counts(calculation))
    if max(counts) < 1 or max(counts) >= nbnd:
        raise ValueError(
            f"a sum-over-states chi_0 needs empty states: this run has "
            f"{max(counts)} occupied bands out of {nbnd}. Run a fixed-density "
            "calculation with more bands (nbnd), which is the convergence "
            "parameter of the whole spectrum"
        )
    weights, _ = calculation.occupations(jnp.asarray(eigenvalues))
    weights = np.asarray(weights)
    for spin, nocc in enumerate(counts):
        empty = np.abs(weights[spin][:, nocc:]).max()
        if empty > _OCCUPATION_TOLERANCE:
            raise NotImplementedError(
                f"band {nocc} of spin channel {spin} carries occupation "
                f"{empty:.3e} at some k-point, so this state has no clean "
                "valence-conduction split -- it is a metal or a zero-gap "
                "system. The sum-over-states chi_0 here is the interband "
                "expression and has no intraband term"
            )
    return counts
