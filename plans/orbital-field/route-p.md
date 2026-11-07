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

# Route P, step 2: the GIPAW-dressed nonlocal term, the SCF by the fold, the magnetizability

Branch `mag-p2`, from step 1's tip 84253a62. Every number below was measured on D22 in the
worktree `/l/ladovj1/w-mag-p2`; the logs are in `/l/ladovj1/review/mag/p2/`, written `M/<log>`.
A log starting with `# slot N cpu M` ran in `slot.sh` (one core, 6 GB); one starting with `# big
job on cpus 0-3` ran under `big.lock` (four cores, 14 GB, `M/big.sh`). The load average was 4 to
16 from other users, so a time here is a range, not a pair. The paper is Lu et al., PRL 135,
196401 (2025), arXiv:2505.06585 (eq. 6 for the dressed projectors, eq. 7 and Table 2 for the
magnetizability, Fig. 3 for the band shift, the Chern numbers of graphene at `n_Phi = 1`).

## What is here (step 2)

- `defuqumat/magnetic/blochgrid.py`: `GridProjectors` (frozen `equinox.Module`: sphere indices
  `(natom, npts)` into the flat magnetic fine grid, the actual displacements `(natom, npts, 3)`,
  the k-independent bra values `(natom, nproj, npts)`, `D` per atom, the magnetic reciprocal
  vectors; `bra(kappa)`, `project`, `apply`) and `grid_projectors` (host NumPy, `# fold back into
  magnetic/gipaw.py`); `MagneticGridHamiltonian` gains `projectors`, `interpolate` and `restrict`
  (the two halves of the two-grid local term, now shared with the nonlocal term and the density),
  `grid_ratio`, `with_potential` (a new SCF potential without a recompile); `magnetic_cell` takes
  `p < 0`; `sphere_levels` (the sphere Davidson of the full H, with its states) and
  `sphere_state_on_box`.
- `defuqumat/magnetic/fold.py`: `representatives`, `fold_to_chemical`, `band_density` (module
  `jit`), `magnetic_occupations`, `run_magnetic_scf` and `MagneticSCF`, `fold_check`,
  `molar_magnetizability`, `run_magnetizability` and `Magnetizability`.
- `defuqumat/workflows/magnetic_bands.py`: `local_only=False` is the full Hamiltonian;
  `magnetic_mesh` for a 2D or 3D closed mesh.
- `tests/unit/test_magnetic_bloch.py`: 7 new tests and 2 more refusals in `test_the_refusals` (below); `tests/regression/test_magnetic_bloch_slow.py`:
  3 new slow tests (listed, not run).
- Probe `plans/orbital-field/mag_route_p2.py` (`item1`, `si`, `chern`, `shift`, `sets`, `fold`,
  `diamond0`, `diamond`, `chi`); `plans/orbital-field/route-p-q{1,...,6}-full.npz`, the level sets
  of item 6.

How the nonlocal term is put on the box. Each projector is a real-space function, `beta_lm(d) =
(r beta_l)(|d|)/|d| Y_lm(d)` with the same real harmonics as `vkb`, times the GIPAW phase `exp(i
phi(d))`, `phi = -a(tau).d - a(d).d/2`, on the points of the **potential's (dense) grid** within its
radius of every atom of the magnetic cell, each image enumerated with its own `d` and the magnetic
Bloch factor `exp(-i a(P).r)` of the lattice vector `P` that brings it into the cell. It rides on
step 1's two-grid interpolation `x = I u`: `y = V x + c W^dagger D W x`, then `I^dagger y`, so the
whole operator is one Galerkin matrix and stays Hermitian, and the quadrature of `beta x` is the
dense grid's, exact up to the projector's tail beyond twice that grid's Nyquist frequency. The radius
is where `|r beta|` falls below 1e-6 of its maximum (1.726 bohr for C HGH, 2997 points per atom at
50 Ry; 4.100 bohr for Si vbc, 4417 points), and the dropped norm is 3e-14 and 2e-14 of `int (r
beta)^2`. The scatter is `.at[].add` (the CPU; `segment_sum` on a card is not built).

## The acceptance numbers (step 2)

**1. a = 0, q = 1 graphene full H, k = 0, the lowest 8 levels, box against the sphere Davidson of
the same Hamiltonian** (`M/item1-50.log`, `M/item1-70.log`; the shared potential).

| ecut | box - sphere, deep pair | the Gamma doublet | the next four | max | projector part alone |
|---|---|---|---|---|---|
| 50 Ry | -1.13e-2, -7.41e-3 | -2.00e-2, -1.78e-2 | -1.8e-4 to -8.4e-4 | **2.00e-2 Ry** | 5.1e-7 Ry |
| 70 Ry | -2.90e-3, -2.34e-3 | -7.20e-3, -7.08e-3 | -4.5e-5 to -2.6e-4 | **7.2e-3 Ry** | 5.3e-7 Ry |

The expected `< 5e-3` Ry is **not reached** at 50 Ry, and 2.00e-2 sits just under the refuse line of
2e-2. What the numbers say: the box is below the sphere at every level, and the difference falls by
2.8 from 50 to 70 Ry, as a larger basis does; the projector part alone is 5e-7 Ry. The projector
part is measured on the sphere's own eigenstates put on the box: there the kinetic term and the
two-grid local term are exact Galerkin matrix elements, so the box operator's Rayleigh quotient minus
the sphere eigenvalue is the grid projectors' error alone, 5.1e-7 Ry at most over the eight levels,
and `becp` against `vkb`'s is 1.1e-7 of its size 0.25. So the projector normalisation and the
`(-i)^l` phase are right, and the 2e-2 is step 1's box-against-sphere basis difference with the
nonlocal term in (step 1 local-only: 0.141 Ry on the deep pair at 50 Ry; the nonlocal s term pushes
the deep states out of the region where the sphere is least converged). The Gamma doublet is split
on the box by 2.1e-3 Ry at 50 Ry and 1.2e-4 at 70 Ry, the broken hexagonal group of step 1.

The projector part on silicon (`M/si.log`; QE's `pw_scf/scf.in` cell, Si.pz-vbc, 12 Ry, our
own SCF at `conv_thr = 1e-10`, the four lowest bands at `k = (0,0,0)`, `(1/4,1/4,1/4)`, `(1/2,0,1/4)`
crystal): `<psi|V_NL|psi>` from the grid projectors against `vkb` within **6.6e-6 Ry**, `becp` within
2.7e-6 of its size 0.42 (s and p projectors, so the `(-i)^l` of `vkb` is exercised). The foundation
had no grid projectors, so this is the design's silicon number, measured here. Its 16^3 dense grid is
coarser for these projectors than graphene's, and that is the larger error.

**2. Chern numbers of graphene at `n_Phi = 1`** (`M/chern12.log`; `p/q = 1/1`, 78912 T, kmesh (12,
12), nbnd 8, the shared potential, 368 s): FHS on each band and group,

| bands | C | largest plaquette flux |
|---|---|---|
| 0 | **+1** | 0.061 |
| 1 | **0** | 0.511 |
| 2 | -3 | 1.091 |
| 3 | +6 | 1.329 |
| 2 and 3 | **+3** | 1.338 |
| 0 to 3 | +4 | 1.320 |

Lu's pattern (1, 0, and 3 for the third and fourth together) holds as exact integers on our 12 x 12
mesh, without forcing. Bands 2 and 3 do not touch on this mesh (smallest gap 2.8e-2 Ry), so each has
its own number, -3 and +6; Lu give only the pair. On a 24 x 24 mesh (`M/chern24.log`, big job, 1287
s, 52 to 101 Davidson steps) every number is the same, **+1, 0, -3, +6, and +3 for the pair**, with
the largest plaquette flux halved (0.016, 0.235, 0.582, 0.880, 0.882), so the integers are converged
in the mesh; the smallest gap between bands 2 and 3 over the finer mesh is 7.6e-3 Ry, so they stay
apart and the split -3/+6 is a statement about this density and cutoff, where Lu's 40 x 40 at 30 Ha
on their own SCF density report the pair. The sign convention is the repository's FHS one, as in
step 1.

**3. The fold** (`M/fold3b.log`; graphene `p/q = 1/3`, kmesh (2, 6), nbnd 20, the 4 representatives
`kappa2 in {0, 1/6}`, from the shared zero-field density): the SCF converges to `conv_thr = 1e-8` Ry in
**9 iterations** (dr2 2.2e-3, 6.0e-4, 2.1e-5, 1.8e-5, 5.0e-7, 4.6e-7, 9.8e-8, 1.0e-8, 5.1e-9), 121 s,
memory estimate 0.193 GB, peak RSS 1.27 GB. At the potential of its last diagonalisation, the folded
density of the 4 representatives against that of all 12 points of the closed mesh, each solved
explicitly to `ethr = 1e-12` with a residual bound of 1e-9 on the returned bands: max relative
difference **1.35e-12**; the sorted levels, three copies of the representatives' against the full
mesh's, 3.7e-13 Ry (710 s). The density change from the zero-field one (Lu Fig. 3c: small): max
`|drho|` 3.3e-2 e/bohr^3 against a maximum density of 0.333, `int |drho|` 0.147 electron of 8.000.
The total energy per chemical cell is -22.7079645172 Ry (the zero-field sphere SCF: -22.6648208203;
the difference is the box's larger basis and the field together, not separated here).
The first attempt (`M/fold3.log`) converged the same SCF and then refused in the check: the residual
bound was applied to the buffer bands too, and one of 24 stayed unsettled after 400 steps; the bound
is now on the returned bands only.

**4. Diamond, Lu's cell** (3x1x1 conventional, `a = 3.56` A, 24 atoms, C.pbe-hgh, 60 Ry, kmesh
(2, 6, 6), box **50 x 18 x 18** (ndim 16200; the design's 54 x 18 x 18 was a guess, this is
`wavefunction_box` at 60 Ry), nbnd 52 with 48 occupied (fixed occupations, the gap checked), memory
estimate **1.24 GB**, `conv_thr = 1e-8` Ry; the start is the zero-field sphere SCF, `M/diamond0.log`,
-272.4330337195 Ry in 8 iterations, 74 s). One flux quantum through the 3x1x1 cell is **10877.4 T**
(Lu 10870.4 T at their rounded `a`).

| run | log | iterations | dr2 | E (Ry, the 24-atom cell) | time | peak RSS |
|---|---|---|---|---|---|---|
| `p = 0` | `M/diamond-p0.log` (big job) | 4 | 9.3e-9 | -273.105259083159 | 947 s | 3.60 GB |
| `p = +1` | `M/diamond-p1.log` (slot) | 7 | 3.1e-9 | -273.043606009502 | 3417 s | 3.52 GB |
| `p = -1` | `M/diamond-m1.log` (slot) | 7 | 3.3e-9 | -273.043606172621 | 3599 s | 3.66 GB |

`E(+dB) - E(-dB)` = **1.63e-7 Ry** (`M/chi.log`): the design's 1e-7 is **not reached**; the refuse
line of 1e-6 (a sign error) is not crossed. A sign error would make the two fields different
operators at first order in `B`, against `E(+) - E(0) = 6.17e-2` Ry here; what is left is the grid.
`H(-B, -k) = conj H(B, k)` holds on the grid only where every row's frequency window is
mirror-symmetric, which needs an odd magnetic box and no window centre at a tie (measured on
small boxes, `M/tr2.log`, and in the unit test: exact to 1e-12 on 5 x 5 x 9 at `q = 1`; 0.09 of the
largest entry on 10 x 5 x 9 at `q = 2`, whose `xi1` axis is even, and 0.34 on 4 x 4 x 8 at `q = 2`). Diamond's
box is 50 x 18 x 18, even on every axis, so each row's half-open Nyquist slot breaks time reversal
by the states' weight there; the two SCFs stopped at dr2 3.1e-9 and 3.3e-9 as well. Which of the two
makes the 1.6e-7 is not separated here (an odd box, 75 x 25 x 25, at three times the size, would).

`chi = -(E+ - 2 E0 + E-)/dB^2` (`M/chi.log`): **-11.40 x 1e-6 cm^3/mol per C2** (the 24-atom cell
is 12 primitive cells), **-5.70 per C atom**. Lu's `chi_val` (LDA, their dataset unnamed) is -11.16 in
their Table S3 and -11.2 in Table 2: the difference is **0.24**, inside the design's 0.6 (5 per cent)
for the functional, PBE here. `E(+) - E(0)` = 6.1653e-2 Ry, so the asymmetry above is 2.6e-6 of the
second difference. The `p = 0` energy is the box's own, 0.67 Ry below the sphere SCF of the same cell
(the box's larger basis); it cancels in the second difference since the three runs share the box. The
three SCFs took 4, 7 and 7 iterations from the zero-field density, the first iteration (random start
blocks, 25 to 60 Davidson steps per k) 680 s on four cores, later ones 67 to 250 s.

**5. The band shift** (`M/shift.log`; graphene at `n_Phi = 1`, 78912 T, kmesh (6, 6), nbnd 12,
non-SCF on the shared potential, the atoms moved by `a1/2` through `Calculator.with_positions`, to
crystal (0.5, 0, 0) and (0.833333, 0.666667, 0), and the potential rolled by half the dense grid
along `a1`, which is the shared density translated with them): `eps(k)` at the same named `k` differ
by up to **0.291 Ry**; the sorted level set over the mesh agrees to **9.7e-13 Ry** and the band
energy (mv smearing 0.01 Ry, 8 electrons) to **2.0e-14 Ry**. The same run at `p = -1` gives the
time-reversal partner: the sorted set at `+B` against `-B` within 6.3e-11 Ry and the band energy
within 1.6e-13 Ry (on the grid time reversal holds only up to the rows whose window centre is a tie,
below). 98 to 105 s a run.

**6. Graphene full H at `p/q = 1/q`, `q = 1, ..., 6`**, kmesh (2, 2q), nbnd `4q + 8`, the shared
potential (`M/sets-q{1,...,6}.log`, big jobs; npz in `M/sets/`, copied to
`plans/orbital-field/route-p-q{q}-full.npz`): the full mesh against `q` copies of the
representatives `kappa2 in {0, 1/(2q)}`:

| q | ndim | nk | full mesh against q copies | Davidson steps | memory estimate | peak RSS | solve | per k |
|---|---|---|---|---|---|---|---|---|
| 1 | 9216 | 4 | 0.0 | 69 to 74 | 0.035 GB | 1.01 GB | 19 s | 4.8 s |
| 2 | 18432 | 8 | 8.4e-13 Ry | 79 to 100 | 0.085 GB | 1.10 GB | 81 s | 10.1 s |
| 3 | 27648 | 12 | 1.0e-12 Ry | 92 to 119 | 0.150 GB | 1.23 GB | 257 s | 21.4 s |
| 4 | 36864 | 16 | 2.9e-12 Ry | 90 to 119 | 0.231 GB | 1.42 GB | 586 s | 36.6 s |
| 5 | 46080 | 20 | 1.4e-12 Ry | 94 to 114 | 0.327 GB | 1.55 GB | 1042 s | 52.1 s |
| 6 | 55296 | 24 | 1.8e-12 Ry | 91 to 148 | 0.438 GB | 1.78 GB | 2227 s | 92.8 s |

All within the design's 1e-9 Ry. Against step 1's local-only runs at the same sizes (60 to 107 steps,
4.6 to 89 s a k-point) the nonlocal term costs 1.0 to 1.3 times the time a k-point (1.04, 1.01, 1.13, 1.31, 1.04 at `q = 1, 2, 3, 4, 6`); the step counts
rose by 7 to 40 per cent (the preconditioner does not see the nonlocal diagonal). These ran as big
jobs (four cores) at a load of 8 to 16, so a time is a range; `q = 5` (`M/sets-q5.log`) waited 12
minutes for the lock.

The SCF at `q = 6` by the fold (`M/fold6.log`, slot 0, CPU 4; graphene `p/q = 1/6`, 13152 T, kmesh
(2, 12), the 4 representatives, nbnd 32, ndim 55296, from the shared zero-field density): converged to
`conv_thr = 1e-8` Ry in **7 iterations** (dr2 8.8e-4, 1.2e-4, 1.1e-4, 8.3e-7, 1.5e-6, 4.2e-7, 1.9e-9),
**644 s** (448 s of it the first iteration from random start blocks, 83 to 108 Davidson steps; later
iterations 9 to 87 s), memory estimate 0.565 GB, **peak RSS 2.19 GB**. E = -22.7426968657 Ry per
chemical cell (at `1/3`: -22.7079645172). The density change from the zero-field one: max `|drho|`
3.1e-2 e/bohr^3 (the maximum density is 0.333), `int |drho|` 0.119 electron of 8.000.

## Decisions taken here (step 2), and why

No fable subagent could be started from this step (the Agent tool is not available to it), so each
open choice is the conservative one, recorded with the number it rests on.

- **The projectors live on the potential's grid, through the two-grid interpolation, not on the
  wavefunction box.** At 50 Ry the box spacing is 0.39 bohr and the HGH carbon `s` projector has a
  width of about 0.3 bohr, so a quadrature on the box would alias at the per-cent level; on the dense
  grid the projector part is 5e-7 Ry (item 1). It costs one sphere gather and scatter beside an
  interpolation the local term already does.
- **The density is `|u|^2` on the dense grid from the same interpolation** (QE's `sum_band` order),
  not the design's `|u|^2` on the box Fourier-padded: the product of two box-limited functions is not
  box-limited, and squaring on the box aliases it. The probe measures the padded form against the one
  used on the same states (`padded` in `mag_route_p2.py`, printed in `M/fold6.log`): on the converged `1/6` states the two differ by up to **2.7e-2 e/bohr^3**
  (8 per cent of the largest density), which as a residual is dr2 = **5.4e-5 Ry**, 5000 times
  `conv_thr`, with the same 8.000000 electrons. So the padded form would converge to another density.
- **The field along `-a3` is the conventions' cell at `|p|` with its traced `scale` at -1**
  (`magnetic_cell`), since `MagneticCell.build` refuses `p < 1` and `conventions.py` is the
  foundation's file. Every phase of the conventions changes sign with it, and the box operator's twist
  is the integer `p < 0`. Tested: the spectrum at `(-1, -kappa)` equals the one at `(+1, kappa)` to
  1e-10 relative (a k with no ambiguous window centre), and differs at `(-1, kappa)`.
- **Occupations**: smearing by the system's own `degauss` and kind, refused when the top band holds
  more than 1e-8 of a state; fixed only across a gap, checked over the whole mesh (diamond's).
  Tetrahedra refused by name.
- **The energy** is QE's sum per chemical cell with `eband` over `q`, `deband` paired as the
  zero-field loop pairs it (output density, input potential), the Hartree and exchange-correlation
  energies at the output density on convergence, the smearing term over `q`, Ewald unchanged.
- **The SCF's Davidson threshold** starts at 1e-8 (the start is a converged zero-field density) and
  follows `next_ethr` from the second iteration; a solve that does not settle in 200 steps is refused,
  not used (none did).
- **Time reversal on the grid is exact only on an all-odd magnetic box with no tie row**, and it is
  not changed. An even axis has a half-open window (the Nyquist slot belongs to one side), and the
  centred window rounds its centre with a small offset (`_TIE_OFFSET`, step 1), so a row whose centre
  is within the offset of a half-integer gets a window shifted by one frequency under `B -> -B, k ->
  -k`. Both are what keep the magnetic translation and the period in `kappa` exact, which the fold
  rests on, where no rule can keep all three at an exact tie. Measured: `H(-B,-k)` against `conj
  H(B,k)` exact to 1e-12 on an odd 5 x 5 x 9 box at a tie-free `kappa` (unit test), 0.09 of the
  largest entry on 10 x 5 x 9 and 0.34 on 4 x 4 x 8 (`M/tr2.log`; the first two forms of the unit test,
  at `q = 2`, failed for this reason, `M/t1.log`, `M/t2.log`); on the converged graphene box at `n_Phi =
  1` the sorted 6 x 6 set at `+B` and `-B` agree to 6.3e-11 Ry (item 5); on diamond it is inside the
  1.6e-7 Ry of `E(+) - E(-)` (item 4).

## Refusals (step 2), each tested to fire

Ultrasoft and PAW in `grid_projectors` (by name, with `C.pz-rrkjus.UPF`), on top of
`refuse_regime` (spinors, `nspin = 2`, spirals, DFT+U, meta-GGA, gamma-only storage); the nonlocal
term without a potential or with the filtered local term; `run_magnetic_scf` without a starting
density; a mesh whose `N2` is not a multiple of `q`; fixed occupations across crossing levels;
occupations other than smearing and fixed (tetrahedra by name); a band set whose top band holds charge
("raise nbnd"); a Davidson solve unsettled after its budget with the nonlocal term in. The guards
that are checks rather than refusals also fire in a test: the projectors without the GIPAW phase
break the magnetic translation (spectra at `kappa2` and `kappa2 - p/q` 1e-6 or more apart), a bra at
the wrong k is far from `vkb`, and the nonlocal term moves the local-only spectrum by more than 1e-2 Ry.

## Unit tests (step 2)

`tests/unit/test_magnetic_bloch.py` (seven new, and the ultrasoft and no-density refusals added to `test_the_refusals`): the grid projectors against `vkb` on silicon (s and
p, random states); Hermiticity, the magnetic translation with and without the GIPAW phase; time
reversal at `p = -1`; the rigid shift of the crystal at `n_Phi = 1` on a 2 x 2 mesh (eps(k) moves,
the set does not, kappa2 -> kappa2 - 1/2 pairs them); the fold's weights, the fold and the charge
of `band_density`; the occupations and the molar conversion; item 1 at 50 Ry (projector part under
2e-6 Ry, the box below the sphere and within 2.5e-2). With the foundation's two files of the package:
44 passed in 54.1 s, the slowest test 9.2 s (`M/t3.log`, at 960dbe3f).

## What is left (step 2)

- Item 1's `< 5e-3` Ry at 50 Ry: the box-against-sphere basis difference, not the projectors; it is
  2e-2 at 50 Ry and 7.2e-3 at 70 Ry, and closing it is the box's corners and Nyquist slot (step 1's
  "What is left").
- The orbital magnetization identity of section 12 item 6 on a Chern insulator cell: left to a
  later phase, as the design says.
- `q = 10` graphene SCF with the fold is measured (`M/fold10.log`, slot 1, CPU 6: 7891 T, kmesh
  (2, 20), 4 representatives, nbnd 48, ndim 92160): converged in **6 iterations** (dr2 8.3e-4 to
  2.6e-9), **1631 s** (1027 s the first iteration, 74 to 85 Davidson steps), memory estimate 1.34 GB,
  **peak RSS 2.87 GB**; E = -22.7542103850 Ry per chemical cell; the padded density 2.7e-2 e/bohr^3
  off again (dr2 5.4e-5 Ry). `q = 20`: FOLD20. `q = 30` not run (estimate below).
- `segment_sum` for the projector scatter on a card; the projectors' memory is small
  (`natom nproj npts` complex, 12 x 1 x 2997 at `q = 6`, 0.6 MB) and is not in `estimate_bytes`.
- The preconditioner does not include the nonlocal term's diagonal; the Davidson steps with it in
  are in the tables (91 to 148 at `q = 6` with the buffer bands, against 72 to 107 local-only).

## Slow files (step 2, listed, not run)

`tests/regression/test_magnetic_bloch_slow.py`: `test_the_chern_numbers_at_one_flux_quantum`
(about 6 minutes), `test_the_band_shift_of_a_rigid_translation` (about 5 minutes),
`test_the_fold_at_one_third` (about 14 minutes, beyond `slot.sh`'s 10-minute per-test timeout).
