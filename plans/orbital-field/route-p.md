# Route P, step 1: `pi^2 + V_loc` on the magnetic box, what was built and what it measured

Branch `mag-p1`, from the foundation's tip ef527648. Every number below was measured on D22
(`ladovj1@D22-0161`) in the worktree `/l/ladovj1/w-mag-p1`, through
`/l/ladovj1/review/guide/slot.sh` (one core, 6 GB), and the logs are in
`/l/ladovj1/review/mag/p1/`, written `L/<log>`. Each log starts with its `# slot N cpu M`
line; the load average was 2 to 5 during the runs, and the route U implementer ran beside them
under `big.lock`, so a time here is a range of about 20 per cent, not a pair.

The paper is Lu et al., PRL 135, 196401 (2025), arXiv:2505.06585; the design is section 3 of
`DESIGN.md`, step 1 only. Three things in the operator differ from the design as written. Each
was measured against the design's form, and the design's form stays as an option, so a reader
can see the number the change rests on.

## What is here

- `defuqumat/magnetic/blochgrid.py`: `MagneticGridHamiltonian` (frozen `equinox.Module`:
  `potential_box`, `potential_fine`, `kpoints`, `metric`; static `p`, `q`, `grid`, `ecutwfc`,
  `local`, `preconditioner`, `window`, `band_batch`, `precision`) with the duck-typed members
  Davidson reads (`nk`, `npwx = ndim`, `npol = 1`, `space`, `dtype`, `gamma_only = False`,
  `has_overlap = False`, `kinetic`, `state_kinetic`, `state_mask`, `diagonal`,
  `overlap_diagonal`, `apply`, `apply_projected`, `s_projections`, `s_correction`, and the hook
  `precondition`); `momentum_multipliers` (the three `pi_a` as row-by-row Fourier multipliers),
  `local_term` (the two forms of `V u`), `wavefunction_box`, `filter_to_box`, `tile_potential`,
  `start_block`, `estimate_bytes`, `refuse_memory`, `landau_problem`, `dense_matrix`,
  `reference_matrix` (the same operator from explicit Kronecker products, for the dense check),
  `grid_positions`, `chemical_copies`, `sphere_local_levels` (the a = 0 reference: the sphere
  Davidson with `D_ij` zeroed), `magnetic_cell` (the conventions' cell, `p = 0` for zero
  field). `kinetic`, `state_kinetic` and `state_mask` are `KRows`, rows computed for the `ik`
  asked, so nothing of size `nk ndim` lives on the operator.
- `defuqumat/workflows/magnetic_bands.py`: `run_magnetic_bands(system, pseudos, scf, *, flux,
  kmesh, nbnd, local_only=True, free=False, chern=False, david=4, ...)` and `MagneticBands`
  (eigenvalues `(nk, nbnd)` Ry, kpoints, `steps`, `unsettled`, `solver` per k, `memory_estimate`,
  `chern` as `(first, last, C, max plaquette flux)` per isolated group of levels, `sorted_levels`,
  timings). One k-point at a time through `davidson_eigensolver_all(..., indices=[ik])`, one
  executable for every k.
- `tests/unit/test_magnetic_bloch.py`: 17 tests. At the final code, with the foundation's two
  files of the same package beside it: 37 passed in 38.6 s, the slowest test 4.0 s
  (`L/t5.log`, `--durations=6`); the file alone 17 passed in 13.3 s (`L/t4.log`).
- `tests/regression/test_magnetic_bloch_slow.py` (slow, not run here): item 1 at `q = 10` and
  `2/11`, items 4 and 5 at `q = 3`.
- Probes: `plans/orbital-field/mag_route_p.py` (`landau`, `herm`, `sphere`, `strong`, `local`),
  `plans/orbital-field/mag_route_p_precondition.py` (the preconditioner variants).
- `plans/orbital-field/route-p-q{1,2,3,4,6}-local.npz`: the level sets of item 5 with the default
  operator (two-grid local term, centred window); `route-p-q{1,2,3,4}-local-filtered.npz` the same
  with the design's filtered local term, for the record.

## The acceptance numbers

**1. The Landau problem**, graphene geometry (`a = 2.46` A, `c = 15` A, box 12 x 12 x 64 per
cell), `p/q = 1/10` (7891.2 T), kmesh (4, 4), `nbnd = 12`, all potentials off, at the final code
(`L/landau-10-1-final.log`, d8e9fcad and later: centred window, two-gauge preconditioner, four
buffer bands): `B_ph = 0.03357231492638168` Ry; every level against `(2n+1) B_ph + (2 pi m/c)^2`
within **2.9e-12 Ry**, flat over the 16 magnetic k within **2.7e-12 Ry**, one state per level per
k per `G3` branch (the `m = +-1` pairs degenerate to 1e-13). FHS Chern numbers of each isolated
group: **1** for each of the three lowest `m = 0` levels (bands 0, 3, 6, `C = 1.0` to the printed
twelve digits), 2 for each `m = +-1` pair, the largest plaquette flux 0.39 (pi/8 for the 16
plaquettes of the 4 x 4 mesh) for a single level. The sign is +1 with the repository's FHS
convention and no sign was adjusted. Davidson, `david = 4`, `ethr = 1e-12`: **119 to 154 steps**
per k; 630 s for the 16 k-points; memory estimate 0.597 GB, peak RSS 1.72 GB (the 16 x 12 states
kept for the Chern number are 0.28 GB of it). The same run before the buffer bands
(`L/landau-10-1c.log`) gave 6.7e-12, 5.8e-12, the same Chern numbers and 146 to 179 steps, and
at the fixed window (`L/landau-10-1b.log`, commit 247038aa) 5.1e-12, 4.0e-12, 146 to 169 steps.

The preconditioner, one k-point, the same problem, 200 steps allowed (`L/landau-10-1-identity.log`,
`L/landau-10-1-kinetic.log`, and the variant probe `L/pre-10-d2.log`, `L/pre-10-d4.log`,
`L/pre-10-d.log`, `L/pre-10-e10.log`, `L/pre-10-e10-d2.log`, `L/dbg2.log`):

| preconditioner | `david` | `ethr` | steps | bands unsettled | worst level error |
|---|---|---|---|---|---|
| identity | 4 | 1e-12 | 200 | 12 of 12 | 0.196 Ry |
| kinetic (the design's: `g_psi` on the periodic FFT of `u`) | 2 | 1e-12 | 200 | 12 of 12 | |
| kinetic | 4 | 1e-12 | 200 | 9 of 12 at kappa 0, 5 at (0.13, 0.27) | 1.1e-3 Ry |
| kinetic | 4 | 1e-10 | 200 | 2 of 12 | |
| `h` gauge only | 4 | 1e-12 | 200 | 12 of 12 | |
| two-gauge (the mean of the two) | 2 | 1e-12 | 200 | 12 of 12 | |
| **two-gauge** | **4** | 1e-12 | **160** | 0 | 1e-13 (first six levels) |
| two-gauge with the plane waves' own diagonal `<m|pi^2|m>` | 4 | 1e-12 | 150 | 0 | |

The rows from the variant probe are at `kappa = (0.13, 0.27)`, the identity and kinetic rows of the
workflow at `kappa = 0`; all of them ran before the centred window and the buffer bands existed
(commits 66ae6545 to 56a6cdff), which the free problem barely feels (the two full item-1 runs
above without the buffer, fixed and centred, took 146 to 169 and 146 to 179 steps).

At zero field the same box and the kinetic rule settle local-only graphene in 22 steps
(`L/pre-g1-p0.log`; identity 200 steps, 10 unsettled), and the same local-only graphene at
`p/q = 1/1` in 164 (two-gauge 93, `L/pre-g1-d4.log`; both at the fixed window and the filtered
local term). So it is the field, not the box, that slows the solve.

**2. Hermiticity and the dense check** (`L/herm2.log`, 5f414272, the final operator). Random `v`, `w`, `kappa =
(0.37, 0.29)` (both wraps active): `|<v|Hw> - <Hv|w>|` relative to `|<v|Hw>|` is **5.0e-15**
(two-grid) and 9.2e-15 (filtered) at `q = 10` on the full box (ndim 92160), 6.0e-16 at `q = 2` on
4 x 4 x 8 per cell; relative to `|v||Hw|` 1.3e-17 and 2.8e-17. The operator built by application
on unit vectors against the independent Kronecker-product matrix (`reference_matrix`, `q = 2`,
4 x 4 x 8, graphene potential, filtered local term): **3.3e-16 and 3.7e-16** of the largest entry
at the two kappa, `|H x - R x| / |R x|` 3.6e-16 and 3.9e-16, the reference Hermitian to 2.5e-16.
The two-grid local term at zero field against the plane-wave Galerkin matrix `V(m' - m)` read off
the dense potential: within 1e-12 of the largest entry (unit test). The three momenta against the
analytic derivatives of a smooth function obeying the twist, `p = 1, 2`: within 1e-10 (unit test).

**3. a = 0, q = 1, graphene local-only, the lowest 8 levels at k = 0, box against the sphere
Davidson of the same Hamiltonian with `D_ij` zeroed**, the two-grid rows at 5f414272, the filtered
ones at bb2182c9 (`L/sphere-50-tg.log`, `L/sphere-70-tg.log`,
`L/sphere-50.log`, `L/sphere-70.log`). The expected `< 5e-3` Ry was **not reached** by either form
of the local term; the difference falls with the cutoff, as the design asks:

| ecut | box per cell | local term | box - sphere, deep pair | box - sphere, the next six | max |
|---|---|---|---|---|---|
| 50 Ry | 12 x 12 x 64 | filtered (the design's) | -0.269, -0.262 | +0.0035 to -0.060 | 0.269 Ry |
| 70 Ry | 15 x 15 x 75 | filtered | -0.067, -0.076 | +0.0015 to -0.016 | 0.076 Ry |
| 50 Ry | 12 x 12 x 64 | **two-grid** | -0.141, -0.123 | -0.0007 to -0.029 | 0.141 Ry |
| 70 Ry | 15 x 15 x 75 | **two-grid** | -0.035, -0.041 | -0.0002 to -0.0078 | 0.041 Ry |

What the numbers say. The two-grid box is below the sphere at every level and both cutoffs, which
is what a larger basis with exact matrix elements gives: the box holds about 2.9 times the sphere's
plane waves, and local-only HGH carbon is not converged at 50 Ry in the sphere either (its deep
level moves from -6.820 to -6.951 Ry between 50 and 70 Ry, and the 50 Ry two-grid box, -6.961, is
already below the 70 Ry sphere). The gap is then the sphere's own incompleteness on a hard local
potential, not an error of the box. The filtered form over-binds through aliasing (its box is 0.27
Ry below the sphere where the two-grid one is 0.14), which is why it is not the default. The Gamma
doublet the sphere holds to 1e-10 Ry at 50 Ry (-0.3446922105) and 1.4e-9 at 70 is split on the box: by **2.1e-3 Ry**
(two-grid) and 4.9e-4 Ry (filtered) at 50 Ry, 1.2e-4 Ry (two-grid) at 70 Ry. The box is a
parallelepiped in crystal coordinates with an asymmetric Nyquist slot, so it breaks the hexagonal
group at its corner plane waves; that is a limit of the route, under "What is left".

**4. The strong translational symmetry**, graphene local-only, `p/q = 1/3`, kmesh (2, 6), `nbnd =
20`, the final operator (5f414272: two-grid, centred; before the buffer bands, which change the
solver and not the operator), `residual_threshold = 1e-9` (`L/strong-tg.log`): spectra at `kappa2` and
`kappa2 - 1/3` equal within **5.1e-12 Ry** over the mesh; the density of the 12 occupied bands
(a fixed `mu = -0.487` Ry in the gap of at least 0.083 Ry, weights by an erfc of width 0.01 Ry so
that no multiplet is cut) folded onto the chemical cell, spread over the three copies **4.0e-12**
relative; the same at the two representatives alone (an open mesh) **1.6e-2**. 469 s.
Without the residual bound the eigenvalue test alone leaves the states converged to about
`sqrt(ethr)`: 6.1e-8 for the density spread (`L/strong-c.log`, filtered local term), 7.4e-12 with
the bound (`L/strong-c-r.log`). With the fixed window (Lu's FFT as written, `L/strong-tg-fixed.log`)
the spectra at the two kappa differ by **6.5e-3 Ry** and the density spread is 2.5e-3: the
alternative reads differently, so the centred window is what makes the theorem exact on the grid.

**5. Graphene local-only at `q = 1, 2, 3, 4, 6`**, kmesh (2, 2q), `nbnd = 4q + 8`, final operator
with the four buffer bands (`L/local-f-q{1,2,3,4,6}.log`, npz in `L/f/`, copied to
`plans/orbital-field/`): the sorted set over the full mesh against `q` copies of the set at the
representatives `kappa2 in {0, 1/(2q)}`, all within the design's 1e-9 Ry:

| q | ndim | nk | full mesh against q copies | Davidson steps per k | memory estimate | peak RSS | solve |
|---|---|---|---|---|---|---|---|
| 1 | 9216 | 4 | 0.0 | 60 to 69 | 0.035 GB | 1.05 GB | 18 s |
| 2 | 18432 | 8 | 9.0e-13 Ry | 64 to 70 | 0.085 GB | 1.13 GB | 81 s |
| 3 | 27648 | 12 | 6.7e-13 Ry | 69 to 81 | 0.150 GB | 1.25 GB | 227 s |
| 4 | 36864 | 16 | 1.1e-12 Ry | 55 to 64 | 0.231 GB | 1.35 GB | 455 s |
| 6 | 55296 | 24 | 3.5e-12 Ry | 72 to 107 | 0.438 GB | 1.63 GB | 2149 s (89 s a k-point) |

Before the buffer bands (`L/local-tg-q{1,2,3,4}.log`) the same runs read 0.0, 6.2e-12, 1.0e-12 and
**8.1e-10** Ry with 80 to 187 steps, and `q = 6` was refused (1 of 32 bands unsettled after 200
steps at k-point 0, `L/local-tg-q6.log`); the 8.1e-10 sat in the topmost band of the block (band
23) at `kappa1 = 0` while every other band agreed within 3.4e-12. The buffer takes that band off
the returned set and halves the steps. The level sets of `q = 1` and `q = 4` are the same to the
printed digits with and without it. Load average 7 to 14 from other users during the `q = 4` and
`q = 6` runs.
With the filtered local term (`L/local-q{1,2,3,4}.log`, the `-filtered` npz files): 0.0, 9.7e-12,
8.7e-12, 6.9e-11 Ry, solves of 9, 47, 147 and 279 s. The two-grid term costs 1.40 times the
filtered one at `q = 3` without the buffer (206 s against 147 s, with 120 to 145 steps against
125 to 157).

The estimate counts Davidson's blocks at `nbnd + 4` bands, six box buffers and three dense-grid
buffers per band in flight (the two-grid term: `3 q 24 x 24 x 128 x 16` B, 21 MB at `q = 6`,
106 MB at `q = 30`, small beside the blocks); the peak RSS is 1.0 to 1.2 GB above it at every `q`,
which is the process and its compiled programs, not the solve.

**6. q = 10, 20, 30**: not run. The `q = 6` run (36 minutes on one core at a load of 9 to 14) is
the largest that fitted this step's time; the time per k-point at `q = 1, 2, 3, 4, 6` is 4.6, 10,
19, 28 and 89 s, and what it is at `q = 30` is not measured here. The free problem at `q = 10` is
the largest box solved (ndim 92160, 39 s a k-point, item 1).

## Decisions taken here, and why

No fable subagent could be started from this step (the Agent tool is not available to it), so each
choice is the conservative one that meets the design's own acceptance, with the design's form kept
as an option and its number beside it.

- **The preconditioner is `two-gauge` and `david = 4`.** The design's rule (`kinetic`) at the
  design's `david = 2` left 12 of 12 bands unsettled after 200 steps on the free problem at `q =
  10`, which is the design's refuse_if; at `david = 4` 9 of 12. The reason is topological: the
  twisted bundle has Chern number `p`, so no gauge makes a state periodic along both `xi1` and
  `xi2`, and a periodic transform of `u` sees a jump at `xi1 = 0`, one of `h` a jump at `xi2 = 0`.
  The mean of the rule in the two gauges settles in 160 steps (`david = 4`, the QE default). The
  plane waves' own diagonal (`<m|pi^2|m>`, with the sawtooth's mean and mean square) took 150 and
  was not adopted, being a second change for 6 per cent.
- **The frequency window of each row is centred on zero mechanical momentum** (`window =
  'centred'`). With Lu's FFT as written the grid operator is neither covariant under `T(a1)` nor
  periodic in kappa (a twisted row multiplies `u` by `exp(i 2 pi p xi2)`, which moves its
  frequencies past the box's Nyquist frequency): 7.5 Ry apart at the top of a 4 x 4 x 4 box (the
  first unit-test run), 6.5e-3 Ry for item 4 at the full box. Centred, both hold to round-off on
  the coarse box (unit test) and item 4 reads 5.1e-12.
- **The local term is `two-grid`** (QE's `vloc_psi` order on the potential's own dense grid), not
  the design's filtered potential on the box, for the aliasing of item 3. It is the design's own
  condition ("the box-against-sphere difference is measured at a = 0 rather than assumed").
- `p = 2, q = 10` is not a reduced flux and `MagneticCell.build` refuses it, as in the
  foundation; two states per level was checked at `2/5` and `2/11` instead (below).
- **Four buffer bands** above `nbnd`, at QE's empty-band threshold 1e-5 Ry, are solved and
  dropped. Without them the top band of the block is the solve's straggler: the `q = 6` run of
  item 5 was refused after 200 steps with 1 of 32 bands unsettled at k-point 0
  (`L/local-tg-q6.log`), and at `q = 4` the top band read 8.1e-10 Ry off its translate where the
  others read 3.4e-12 (`L/local-tg-q4.log`). Item 5's table is the run with the buffer
  (`L/f/`, `L/local-f-q*.log`); items 1 to 4 were measured before it, without it.
- A `residual_threshold` on the states for item 4, since the eigenvalue test does not converge
  them (6.1e-8 against 7.4e-12, above).
- The conventions' traced `scale` is not used: the field enters through the integer `p` of the
  twist, so `B` is quantised by the box; `kappa` is traced.

## Two states per level

`2/10` is refused by the conventions (not reduced). At `2/5` (ndim 46080, `L/landau-5-2c.log`,
without the buffer bands) the solve was refused by the design's budget: 1 of 12 bands unsettled
at k-point 1 after 200 steps. At `2/11` (14347.7 T, ndim 101376) with 400 steps allowed
(`L/landau-11-2c-400.log`, without the buffer): every level within **1.7e-11 Ry** of the free
formula, flat within 1.3e-11 Ry over the 4 x 4 mesh, **two states per level** (the members of the
lowest four levels equal within 1e-13 Ry, of the 4-fold `n = 1, m = +-1` level within 1.3e-11),
265 to 361 steps per k, 1041 s. With the four buffer bands (the final code) `2/5` settles within the
design's budget (`L/landau-5-2-buffer.log`, ndim 46080, 31564.9 T): **101 to 118 steps** per k, every
level within **3.2e-12 Ry**, flat within 2.0e-12 Ry, two states per level (the lowest pair equal to
1e-12), 242 s for the 16 k. So `p = 2` holds the exact degeneracy, and what slowed it was the block's
top band, which the buffer takes off the returned set. `2/11` was not rerun with the buffer.

## Refusals, each tested to fire (unit tests)

`local_only=False` (the GIPAW-dressed nonlocal term is step 2), a field not along `a3`, a memory
estimate above the limit (by number, in GB), a flux that is not reduced, a Chern number on a k-list
that is not the closed plane mesh, a spinor (`npol = 2`) and an ultrasoft dataset through
`refuse_regime` (which also names `nspin = 2`, PAW, a spiral, DFT+U, meta-GGA, gamma-only storage),
`p = 0` with `q != 1`. An unconverged Davidson solve falls back to the dense matrix (tested at
`max_iterations = 1`) and is refused by name above `ndim = 20000` (tested by lowering the limit);
`unconverged = 'keep'` keeps the Davidson values for measuring a preconditioner and marks them.

## What is left

- The nonlocal term with the GIPAW-dressed projectors (step 2), the fold and the SCF.
- The broken hexagonal group of the box (the 2.1e-3 Ry Gamma split at 50 Ry): an odd box removes
  the Nyquist asymmetry; a spherical mask at zero field would remove the corners, and with the
  field it needs a definition in the mixed representation, which the review dropped.
- A better preconditioner for `q >= 10`: the free Landau operator in the representation where
  `pi_2` is diagonal (the `xi1` segments chained by the twist into one line, a 1D oscillator) has
  an exact dense inverse of size `q n1 n2` per k, at a cost of `(q n1 n2)^2 n3` per band; not built.
- Item 6 (q = 10, 20, 30), and the memory of the two-grid buffers measured rather than estimated.
- The crosscheck stage reads `route-p-q*-local.npz` (two-grid, centred); route U's step 1 builds
  its operator on the sphere, so the comparison of item 1 of the crosscheck carries item 3's
  box-against-sphere difference (0.14 Ry on the deep pair at 50 Ry) and should be read with it.

## For `docs/d22-notes.local.md` (the Record stage)

- Several slot jobs in sequence are a shell script of `slot.sh` calls started with `nohup bash
  script.sh > out 2>&1 < /dev/null & disown`; the scripts of this step are
  `review/mag/p1/chain_{a,b,c,d,e}.sh`. Killing the script (by its PID from `ps`) stops the chain
  after the running job.
- `slot.sh` passes the caller's environment to the job (`MAG_MAXIT=400 bash slot.sh ...` reached
  the probe).
- The worktree `/l/ladovj1/w-mag-p1` is removable once `mag-p1` is merged.

## Slow files (listed, not run)

`tests/regression/test_magnetic_bloch_slow.py`: the free Landau problem at `1/10` with the Chern
numbers (about 10 minutes) and at `2/11`, items 4 and 5 at `q = 3` (about 4 minutes).
