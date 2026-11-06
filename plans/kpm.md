# The kernel polynomial method in defuqumat: a plan (2026-11-06)

This plan says which quantities a Chebyshev expansion of the Kohn-Sham Hamiltonian
should give in this code, in which order, and how each one is checked. The inspiration
is pyqula's `kpmtk/`. The constraint is that a plane-wave basis is not a tight-binding
basis.

## Where KPM can win in a plane-wave basis

Let us first ask what the expansion costs here. The polynomial is in
`H' = (H - c)/s`, with `[c - s, c + s]` containing the whole spectrum of `H(k)` in the
basis. In plane waves the top of that spectrum is set by the basis and not by the
physics: it is roughly `ecutwfc + max(v_loc) + 1.1 max(nonlocal)`, which is
`spectral_bounds` in `realtime/propagate.py`. So `s` is roughly `ecutwfc / 2`.
With the Jackson kernel the delta function has a half width at half maximum of
`delta = sqrt(2 ln 2) pi s / N`, so the number of moments is

    N = sqrt(2 ln 2) pi s / delta.

At `ecutwfc = 40` Ry and `delta = 0.01` Ry (0.14 eV), `s` is about 22 Ry and `N` is
about 8000, meaning 4000 applications of `H` per start vector and per k-point with the
doubling trick. A whole SCF of `si-1k.in` applies `H` to a few hundred band vectors
in total, while a KPM DOS with `R = 10` random vectors applies it about 40000 times, so on
a two-atom cell KPM loses by roughly two orders of magnitude, and a timing there would
only say that.

What makes KPM worth having is two things. First, its cost does not depend on `nbnd`.
Davidson goes as `nbnd^2 npw` in the orthogonalization and the subspace, and `nbnd`
grows with the number of atoms, so there is a crossover cell above which a DOS by KPM is
cheaper than Davidson plus a DOS. We will measure where it is on `si16`, `si64` and
`si128` (`benchmarks/`). Second, KPM gives the whole spectrum up to the cutoff at no
extra cost, which Davidson cannot give at any reasonable `nbnd`: the empty-state PDOS,
the unfolded spectral function at high energy, and the optical conductivity up to
large frequencies. Also, a trace in a large cell self-averages, so the number of random
vectors per atom needed for a given noise falls as the cell grows.

## What we will compute, in order

### P167: the engine, and the total DOS

The engine is the recursion `V_{n+1} = 2 H' V_n - V_{n-1}` on a block of `R` start
vectors at one k-point, `(R, npwx)` or `(R, 2 npwx)` for a spinor, applied with the
existing block `ham.apply(psi, ik)`. The recursion is a `lax.scan` over `n`, the k-points
go through `map_k`/`sum_k` of `batching.py`, and the start vectors are batched in chunks
so the memory is `3 R ndim` per k-point in flight. The doubling trick gives `2n` moments
from `n` steps: `mu_{2n} = 2 <V_n|V_n> - mu_0`, `mu_{2n+1} = 2 <V_{n+1}|V_n> - mu_1`.

The pieces, each its own file under `defuqumat/kpm/`:

- `bounds.py`: `c` and `s` per k-set, self-contained (the real-time `spectral_bounds` is
  welded to its chunk and its `kappa_max` widening, and is not refactored). The default is
  a short Lanczos run, 40 steps from one random vector per k-point, for both edges, with a
  margin of 2 per cent of the width, as pyqula's BSE resolvent does it; Weyl's inequality
  (kinetic maximum, local potential maximum, nonlocal top) is computed beside it as a sanity
  bound that the Lanczos top must not exceed. The local potential minimum (about -14 Ry on
  silicon) is not used as a lower edge, since it doubles `s`. The guard below is what makes
  an estimated edge safe.
- `moments.py`: the recursion with doubling, for three kinds of start vectors: random
  (a stochastic trace), given (a rank-`r` trace, deterministic), and all basis vectors
  of the sphere (the exact trace, for tests on small cells).
- `kernels.py`: Jackson (default), Lorentz (lambda = 3 by default, the one that gives a
  Lorentzian, used for optics) and Fejer, behind a name registry, with the conversion
  from a broadening to `N` and back. The Jackson width in the rescaled energy `x` is
  `pi sqrt(1 - x^2) / N` to leading order, so it narrows towards the edges; the bands sit
  at `x` of about -0.85, where the width is half the one at the centre. The conversion
  takes the energy window and asks for the width at its centre, which roughly halves `N`
  against the formula at `x = 0`.
- `profile.py`: moments to a function of energy, evaluated directly on the grid by the
  `T_n(x)` recursion (at `N nE` of about `10^7` there is no need for a DCT), plus the
  closed-form integrated DOS.
- `guard.py`: the a-posteriori check of pyqula's `scaleguard.py`. For a Hermitian `H'`
  with its spectrum in `[-1, 1]` and normalized start vectors, `|mu_n| <= 1`; a NaN or a
  moment above `1 + tol` (1e-6 in double, 1e-3 in single) raises a named error that says
  the scale was too small.

The random vectors are random phases on the sphere, `exp(2 pi i u)` on the `npw_k` live
plane waves and zero on the padding, from `jax.random` with the seed recorded on the
result. The trace is normalized by `npw_k` (or `2 npw_k` for a spinor), not by `npwx`.
The result object `KPMDensityOfStates` carries the moments themselves, per spin channel,
as well as `energies`, `dos`, `integrated` and the spread over the random vectors as a
noise estimate; with the moments stored, a different broadening or kernel is a host-side
operation of milliseconds and no new recursion. This is the one thing a user gets from KPM
that a sum over states does not give.

Regimes: norm-conserving, `nspin = 1` and `2` (per channel, as `Calculation.hamiltonian`
returns them), and the spinor with `npol = 2`. Ultrasoft and PAW are refused by name: the
polynomial must be in `S^-1 H`, with moments in the `S` inner product, and we measure what
ignoring `S` does on ultrasoft silicon (the guard firing or not, and the DOS error) so
the refusal is a number. `gamma_only` is refused by name too, because the half sphere's
inner product is `2 Re - G=0`, and a run is unfolded with `whole_sphere` first if needed.

The checks, none of which shares machinery with the recursion:

- On `pw_scf/scf.in` silicon, the dense `ham.matrix(ik)` fixture, sliced to the `npw_k`
  live plane waves (the padding rows are zero and would add `npwx - npw_k` spurious
  eigenvalues at `x = -c/s`), gives every eigenvalue `e_i` of `H(k)` in the basis, so the
  exact moments are `(1/npw_k) sum_i T_n(e_i')`. The full-sphere trace must match them to
  round-off at a small `N` (40 is enough, since an exact moment does not need a fine
  broadening), and the seeded stochastic trace must match them with an error that falls as
  `1/sqrt(R)` between `R = 4` and `R = 64`.
- `mu_1 = Tr H' / npw_k` from the diagonal (kinetic, `V(G=0)`, the nonlocal diagonal), which
  is computable on any cell, si128 included, and `mu_2` from the Frobenius norm of the dense
  matrix where it exists. `mu_0` is not a check, because it is 1 for a normalized start
  vector whatever the recursion does.
- The occupied DOS against Davidson: the SCF eigenvalues put through the same Jackson
  kernel in the rescaled variable `x` (not a fixed Gaussian in `E`, since the Jackson width
  depends on `x`), below the top of the computed bands, on `si8`. The difference is the
  stochastic noise only, and it is compared with the spread the result reports.
- The guard fires on a scale 10 per cent too small (a test that it fires, not only that a
  good run passes).
- The spinor's DOS equals twice the scalar one on a run without spin-orbit and without a
  magnetization.

Timing: there is no KPM in `pw.x` and none in Elk, so the reference is this code's own
Davidson DOS on the same cell and the same broadening (an NSCF with enough bands plus
`run_dos`), with the time of `pw.x` for the same NSCF beside it. The cells in `benchmarks/`
are at `ecutwfc = 12` Ry, where `s` is about 7 to 8 Ry and `N` at `delta = 0.01` Ry is
about 1500 at `x = -0.85`. The stochastic error at an energy goes as
`1/sqrt(R x states inside the broadening)`, so at a fixed noise `R` falls as one over the
number of atoms, and only then is KPM linear in the cell. With the `PERFORMANCE.md` numbers
for si64 (128 bands), a Davidson NSCF is about `2 nat x 25` band applications and KPM about
`R N / 2`, and at 10 per cent noise the two meet around 100 to 200 silicon atoms. So si16
and si64 are expected to lose and si128 is near the line. We measure si16 and si64 at a
stated noise target, and si128 if its Davidson reference (256 bands and more, on one CPU
slot) fits; the number to give is the crossover, measured or extrapolated, and said which.

### P168: deterministic traces, the PDOS, the LDOS and unfolding

These are rank-`r` traces with one recursion per start vector and no randomness, and they
are the strongest physics case for KPM in this code.

- The PDOS: the start vectors are the Lowdin-orthogonalized atomic orbitals of
  `projwfc/`, so the KPM PDOS converges to exactly `run_pdos`'s projections, with the
  empty states up to the cutoff included. Checked against `run_pdos` with the Jackson
  kernel applied to its sum over states, in the window of the computed bands. A per-orbital
  trace is not invariant on a reduced k-set, so it runs on the whole grid or goes through
  `build_projection_symmetry`.
- The spin-resolved DOS of a spinor, `Tr[sigma_i delta(E - H)]`, the operator-weighted trace
  of pyqula's `random_trace_A`, for one extra application per step: what spin-polarized STM
  measures. And the Fermi level from the integrated DOS by bisection, with no bands, which
  P169 needs on a supercell.
- The LDOS at a point: the start vector is `delta(r - r0)` in plane waves,
  `exp(-i (k+G) r0) / sqrt(Omega)` on the sphere, so `LDOS(r0, E) = <r0|delta(E - H)|r0>`.
  A map at a fixed set of energies is cheaper with random vectors on the real-space grid,
  accumulating `sum_n c_n(E) T_n(H) r` for each energy and reading the diagonal
  `conj(r(x)) (A r)(x)`, so the memory is one field per energy and not one per moment. This
  is what feeds `stm/`. The delta start vector has norm squared `npw_k / Omega`, so the
  guard bound is that norm, and a point does not self-average, so the grid map needs `R`
  in the hundreds. The tip-inside-a-sphere refusal of `stm/` is inherited. Checked against
  `get_stm` at the same energy and broadening.
- Unfolding: in a supercell, the projector onto the plane waves `k_prim + G_prim` of a
  primitive k-point is diagonal in the plane-wave basis, so the spectral function
  `A(k_prim, E) = sum over G_prim of <k_prim + G_prim|delta(E - H)|k_prim + G_prim>` is a
  rank-`npw_prim` trace, done exactly or with random vectors supported on that subset.
  A defect or a disordered supercell then gives an unfolded band structure with its
  broadening, with no supercell bands computed. The subset is built by a Miller-index test
  (neighbouring k-points do not share a G-sphere), and each `k_prim` folded to its `K` is a
  caller-built k-set, so a `for_spin` boundary. Checked on a pristine `si8` supercell
  against the primitive cell's bands (the weight must sit on them), and exactly by the sum
  over the `k_prim` folding to one `K` of `A(k_prim, E)`, which is the total DOS at `K`.

### P169: the optical conductivity by a two-index expansion

The Kubo-Greenwood conductivity is

    sigma_ab(w) = (pi / (Omega w)) int dE [f(E) - f(E + w)] Tr[v_a delta(E - H) v_b delta(E + w - H)],

meaning that the transition from `E` to `E + w` is weighted by the velocity matrix
elements, with no band index anywhere. With `delta(E - H) = sum_n g_n c_n(E) T_n(H')`,
the trace is a sum over the two-index moments
`mu_nm = <r|v_a T_n(H') v_b T_m(H')|r>`, which for one random vector are the matrix of
inner products between `T_n(H') v_a^dagger r` and `v_b T_m(H') r`, built in blocks of a few
hundred `n` as a matrix product, so the memory is two blocks of vectors and not `N` of each.
The cost is `2N` applications per random vector and direction plus a Gram product of
`N^2 npw`, not `N^2` applications.
The velocity is `VelocityOperator.apply`, whose `jvp` of `H(k)` already contains
`[r, V_NL]` and the Hubbard term. The Lorentz kernel gives each delta a Lorentzian of
half width `lambda s / N`, and the two deltas convolve to a Lorentzian of twice that, so
the comparison with the existing band sum `optical_conductivity` is at `eta = 2 lambda s / N`,
at low `w`, on silicon. On silicon the band sum from a cheap NSCF already covers this, so
there P169 tests the engine; where it pays is a disordered supercell, and with the
Kubo-Bastin `sigma_xy`. The bare
f-sum rule is not used as a check, because the nonlocal pseudopotential changes it. On a
symmetry-reduced k-set the tensor is a rank-2 object per k-point, and the wedge sum is
symmetrized as the existing conductivity does it (the trap "a wedge sum completes only for
a quantity linear in a covariant per-k object"). The Kubo-Bastin form for the transverse
part, `sigma_xy`, uses the same moments with a Green's function kernel and is a named
follow-up checked against the existing anomalous Hall conductivity of iron.

### What we leave out, and why

- The density by a Chebyshev expansion of the Fermi function, which is an SCF with no
  `nbnd` (stochastic DFT): at room temperature `N` is of order `s / kT`, about `10^4` to
  `10^5` here, and with stochastic noise in the density it only wins past hundreds of
  atoms. It is the natural successor of P167, not part of it.
- Chebyshev-filtered subspace iteration, which is an eigensolver; it is discussed in
  `notes/DAVIDSON-LITERATURE.md` and belongs there.
- A Chebyshev step for `exp(-i H dt)` registered through `register_propagator`: cheap
  once P167 exists, and measured against `taylor4` on the same run. A side item.

## Rules that bind all three phases

- Differentiable and GPU-ready: `lax.scan` over `n`, `jnp.where` for the sphere mask, no
  `np.asarray` on the path, no dtype literals (dtypes from the Hamiltonian and
  `config.py`, the random phases included), k as the leading axis through `map_k`/`sum_k`.
  The moments are differentiable; the moments-to-DOS step is in `jnp` too, so the DOS is.
- The moments kernel is a module-level `jit` that takes the Hamiltonian, the k indices and
  the start block as arguments and never closes over them, so a second call compiles
  nothing (the recompiling-closure trap), which is a test.
- Ultrasoft and PAW, when they come: `S^-1 H` is diagonalizable with real eigenvalues, so
  `Tr T_n(S^-1 H')` with plain random vectors is `sum_i T_n(e_i')`; the guard bound
  `|mu_n| <= 1` holds only in the S-form `<r|S T_n(S^-1 H')|r>` with `<r|S|r> = 1`, which is
  the form to use. The inverse is `_Chunk.inverse_overlap`'s Woodbury form, and the top edge
  is the generalized one from Lanczos on `S^-1 H`, not `H`'s own.
- `Calculator.get_kpm_dos`, `get_kpm_pdos`, `get_kpm_ldos`, `get_kpm_unfolded`,
  `get_kpm_conductivity` are one-line delegations to `run_kpm_*` entry points written in the
  same pass.
- Every phase leaves the five deliverables: a number in `PLAN.md` §3, a row in the README
  table (the QE and Elk ticks checked by the `reference-source` agent), a section of a
  guide page with its Limits note, a notebook, and the timing in `PERFORMANCE.md`.
- Every calculation runs on D22 through `review/guide/slot.sh`, on its CPU slots, with the
  code moved by a git bundle of branch `kpm`.

## Added after P168 (2026-11-06): the control variate, measured

The user's low-energy subspace (atomic orbitals and their leak into the interstitial) was measured by
a subagent on D22; the numbers are in `PLAN.md` at the end of P168. The form worth implementing is a
control variate in moment space built on the Ritz pairs of the atomic orbitals after a Chebyshev
low-pass of degree 8: unbiased, 50 to 100x more accurate than plain random vectors at equal cost at
R = 1, and it keeps the result object as it is. It is the candidate next to P169. Chebyshev-filtered
subspace iteration as an NSCF solver is a separate item for the solvers.
