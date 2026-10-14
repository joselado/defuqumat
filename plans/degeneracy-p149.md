# Plan: P149, the degeneracy items left after P147 and P148

Follows `notes/DEGENERACY-NEXT.md` (the inventory, items 1 to 6, with the measurements of items 2, 4
and 6 in `PLAN.md` P147 and P148) and `OPEN.md` Part XXV item 1 and Part XXVI. Fable's order was
6, 4, 5, 3; 6 and 4 are P148 (`cf1ebe2`), so this phase is items 5 and 3, in that order. Every
number is taken on `D22-0161`, on a worktree at `cf1ebe2` or later (`w-p149`), never on a tree
without P148's averaged `becsum` and regularised kernel, since both arms of a baseline taken there
would re-measure what P148 fixed.

## What is open, and the one term each is missing

- **Item 5a**, `hubbard/occupations.py:593`: `starting_ns` writes a requested eigenvalue into
  `eigh(ns)`'s eigenvalue `m`, and inside a degenerate block of `ns` the index `m` is LAPACK's
  order. A starting guess, so one docstring sentence; nothing to measure.
- **Item 5b**, `OPEN.md` Part XXV item 1: `get_electron_phonon` at `threshold=None` (the scheduled
  CG threshold `min(0.1 sqrt(dr2), 1e-2)`) gives a mode pair at `q1` split by 0.015 cm^-1 where
  `ph.x`'s is 7.8e-5, and a random unitary inside the degenerate multiplets at `k` moves it. The
  tests pin `CG_THRESHOLD = 1e-12`; the default is not fixed. The missing term is a threshold rule,
  not a kernel: which CG threshold makes the dynamical matrix a function of the multiplet and not of
  its basis, under the schedule.
- **Item 3**, nine per-module degeneracy tolerances, a hard step each, five of them below the
  solver's promise on an empty band (`empty_band_threshold(ethr) = max(5 ethr, 1e-5)` Ry). The
  missing terms are two: one resolution rule for the tolerance, and the smooth kernel
  `Delta/(Delta^2 + eta^2)` in place of the step, site by site, behind a measurement against that
  site's own validation number.
- **Item 4's tail** (the divided difference of the occupation in the density assembly's tangent)
  and **Part XXVI** (the remaining amplifier in a Davidson step on a degenerate block; the exact
  route is P22c's implicit tangent) are **not** this phase. Their triggers are written in `PLAN.md`
  "What is outstanding" and in Part XXVI, and nothing here fires them.

## Item 5b: the measurement before the rule

The experiment is `OPEN.md` Part XXV's own, on `tests/regression/test_electron_phonon.py`'s metal at
`q1 = (0.25, 0, 0)`, with P148's tree:

1. The pair at the default schedule, and at CG thresholds 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12,
   each run twice: once with the converged states as the solver returns them and once with a random
   unitary applied inside every degenerate multiplet at `k` below the cut (the rotation goes into
   the converged eigenvectors handed to the phonon solve, never through the SCF start, which is a
   span and a null test). The reading per threshold is `|pair(rotated) - pair(unrotated)|` and the
   split of the pair against `ph.x`'s 7.8e-5 cm^-1.
2. One arm with the warm start between passes (`phononq.py:904` in Part XXV's words; the line is
   located first) switched off at the default schedule, under the same rotation, which is the A/B
   Part XXV named and did not run.

The rule is then whichever of these the numbers support, and fable picks: (a) the schedule's floor
`1e-2` lowered to the threshold at which the rotated and unrotated pairs agree to 7.8e-5; (b) the
threshold held at `CG_THRESHOLD` for the last pass only, once `dr2` has converged, so the warm
passes keep their speed; (c) the warm start between passes dropped, if arm 2 shows it carries the
basis. Acceptance is Part XXV's: the default reproduces the 1e-12 pair to 7.8e-5 under the
rotation. The cost is recorded as a call count against the current schedule on the same cell.

## Item 3: the rule, then the kernel

### The nine sites, re-read 2026-10-14

| site | what it guards | manifold | tolerance | the rule applies | the kernel applies |
|---|---|---|---|---|---|
| `response/conductivity.py:836` `DEGENERACY_TOL` | `1/e_mn`, `1/e_mn^2` of the interband conductivity | occupied/empty pairs of nscf states | `EMPTY_ETHR_FLOOR` since B1; `degeneracy_tol=None` resolves to it at `:593` | yes, already the constant; gains `ethr` | yes |
| `topology/kubo.py:105` `DEGENERACY_TOL` | `1/gap^2` of the Kubo Berry curvature | occupied/empty pairs of nscf states | 1e-6 | yes | yes |
| `transport/green.py:211` `DEGENERACY_TOL` | the multiplet blocks of the lead at `E` (`channel_basis`, `multiplets`) | bands at `E`, empty included | 1e-6 | yes | **no**: it decides a block, not a weight; a kernel has no meaning for a block membership |
| `response/photocurrent.py:180` `DEGENERACY_TOL` | `1/e_mn` of the shift current and SHG | occupied/empty pairs | 1e-8, raised to `max(broadening, 1e-8)` at the entry point (`:611`) | yes, below the entry point only (a before/after at the entry point is a null) | yes |
| `ultracell/driver.py:202` `DEGENERATE_CUT` | whether the multiplet ladder's cut lies inside a multiplet (`:1324`) | the band at the cut | 1e-8 | yes | no: a refusal's threshold |
| `topology/berry.py:323` `MODEL_DEGENERACY_TOL` | `1/gap^2` on a model | model, exact, no `ethr` | 1e-8 of the band width | **no**: no solver and no `ethr` | left alone |
| `tddft/spinchi0.py:654` `_DEGENERACY_TOLERANCE` | the `0/0` of the transverse `chi_0` (`:639`) | pairs near `E_F` | 1e-8, the limit `df/de` taken inside | the rule, for the switch | no: the divided difference is already the exact kernel there |
| `scf/tetrahedra.py:94` `_BAND_DEGENERATE` | the degenerate-weight average (`:648`) | bands carrying weight, at `ethr` | 1e-6 | yes, with `ethr` not `empty_ethr` | no: `opt_tetra_weights_only` shares weight by a step, as `pw.x` does |
| `topology/parity.py:73` `TOLERANCE` | the parity window and `|P^2 - 1|` | occupied, at `ethr` | 1e-6 | yes, with `ethr` | no: a sign |

So the rule touches seven sites and the kernel three (the conductivity, the Kubo curvature, the
photocurrent), and those three are the ones that build a `1/gap` or `1/gap^2` on occupied/empty
pairs. The ultracell cut and `green.py` are block decisions and get the rule only.

### What the rule is, stated so it promises no more than it does

`resolve_degeneracy_tol(asked, ethr, *, manifold)` in one module (`solvers/davidson.py` beside
`empty_band_threshold`, or a new `scf/degeneracy.py` that imports it; fable picks):
`max(asked, empty_band_threshold(ethr))` for a site reading the empty manifold and
`max(asked, ethr)` for one reading the occupied manifold only, with `asked` the site's own constant
and `ethr` the threshold the states handed in were converged at. **At the end of a converged SCF
`5 ethr` is about 5e-10, so for SCF states the rule collapses to the 1e-5 floor, and the change is
"1e-6 and 1e-8 become 1e-5" at the four empty-manifold sites that are below it.** The rule is a
function of `ethr` only for nscf or bands eigenvalues solved at a loose `conv_thr`
(`workflows/nscf.py:285`, `ethr = max(ETHR_MIN, 0.1 min(1e-2, conv_thr/nelec))`), which is the
case that matters: a `get_berry_curvature` on a `conv_thr = 1e-6` nscf run has empty bands
converged to 1e-5 and a guard at 1e-6 below it.

Where `ethr` comes from: `SCFResult.ethr` exists (`scf/driver.py:1824`); `NSCFResult` and
`BandStructure` do not carry it and gain a field (one line each, the value the solve was run at),
and the states objects the response sites consume (`topology/states.py`'s `PlaneWaveStates`, or
whatever `get_optical_conductivity`, `get_berry_curvature` and `get_shift_current` hand down) carry
it through. A site handed eigenvalues with no `ethr` (a caller-built array) resolves with
`ethr = None`, which gives `asked` alone and a one-line note in the docstring.

The user-facing name is one shared option, `degeneracy_tol`, default `None`, in `SHARED_OPTIONS`,
forwarded **by named parameter** (so it is added to every entry point's signature that the rule
touches, or the facade drops it silently, `calculator.py:123`), meaning "at least this, and never
below the solver's promise". The sites that take `tol=` today (`green.py`, `photocurrent.py`,
`kubo.py`'s `degeneracy_tol`) keep their keyword and resolve it through the rule.

### The kernel, behind the measurement

The step `where(|gap| > tol, 1/gap, 0)` becomes `gap/(gap^2 + eta^2)` and `1/gap^2` becomes
`gap^2/(gap^2 + eta^2)^2`, with `eta` the resolved tolerance, at the three pair sites. A kernel moves
a converged number by about `(eta/gap)^2` of each pair's contribution, so each site is read against
its validation number before and after, and where the move exceeds that number's tolerance the step
stays and only the rule lands. The numbers to locate in `PLAN.md` §3 (fable fills the table at the
review, since the phase headers do not name them by the words above):

| site | validation number, phase | tolerance there | after the kernel |
|---|---|---|---|
| `conductivity.py` | `OPEN.md` B1's table; the Kubo-against-real-time identity (P139, P141) | | |
| `kubo.py` | the Chern number of the model and the Berry curvature against `berry.py`'s Wilson loop (the topology phase) | | |
| `photocurrent.py` | silicon's `chi^(2)` against the hierarchy (P137) and the shift current's reference | | |

The baseline for each site is the same experiment, on the P148 tree: a cell with a symmetry-forced
degeneracy (silicon's threefold at `Gamma` and the pairs at `X` for the curvature and the
conductivity; the same for SHG), the converged eigenvectors handed to the site once as returned and
once with a random unitary applied inside each degenerate block, and the output read before and
after. A site is degeneracy-safe when that move is round-off. The acceptance for the kernel is the
same rotation at a pair the solver splits by its own 3e-7 (an nscf at `conv_thr = 1e-6`), where a
step at 1e-6 or 1e-8 lets the pair through into a `1/gap^2`: before, the output moves by the gauge;
after, by round-off, and the validated number by less than its tolerance.

### Order inside item 3

1. The rule and the two result fields, with a unit test per site that the guard fires at the
   resolved value and not at the old constant (the trap: a check whose null result cannot be told
   from a pass).
2. The rotation harness, one script, reused from `review/degeneracy/cut_jvp*.py`'s rotation.
3. The three kernel sites in the order conductivity (already at the constant, so the kernel alone is
   measured), Kubo, photocurrent; each a before/after against its validation number.
4. The texts: `features.tex`'s box for each `get_*` the option reaches, `scope.md`'s line, the
   docstrings, and the P149 record.

## Questions for fable

1. P148's header still reads IN PROGRESS although its record and gate are complete and `cf1ebe2`
   names items 6 and 4 only. Flip it to DONE in this phase's first commit, and make items 5 and 3
   P149? (yes/no)
2. Item 5b's rule, (a), (b) or (c), after the numbers.
3. The module the rule lives in, and whether `degeneracy_tol` joins `SHARED_OPTIONS` or stays a
   per-method keyword (it reaches `get_optical_conductivity`, `get_berry_curvature`, `get_chern`,
   `get_shift_current`, `get_shg`, `get_vertical_transport`, `get_momentum_transport`,
   `get_ultracell*`, `get_z2*`, the tetrahedron DOS).
4. For the occupied-manifold sites (`tetrahedra.py`, `parity.py`), `ethr` as proposed, or leave
   them at 1e-6 with a sentence (they are four orders above `ethr` and the B1 argument does not
   apply).
5. Whether the kernel lands at all three pair sites or only where the rotation test shows a move
   above round-off at the solver's own split.

## Fable's review and decisions (2026-10-14)

Read: this plan, `notes/DEGENERACY-NEXT.md` items 3, 4 and 5, `PLAN.md` P148 (lines 25718 to 25908)
and the "What is outstanding" entry at line 207, `OPEN.md` Part XXV item 1 (line 9117) and Part XXVI
(line 9133), the nine sites at their quoted lines with the function around each, the workflows that
feed them, and the facade. Nothing was run. One finding changes the shape of item 3 and is stated
first, since every row of the table and three of the five answers rest on it.

**The three pair sites do not read states converged to the empty-band floor.** `optical_conductivity`,
`shift_current` and `shg` take the four-tuple of `fixed_density_states`
(`workflows/conductivity.py:141`, `workflows/photocurrent.py:114`, `workflows/shg.py:115`), and the
Kubo curvature takes `DFTSource.states` (`workflows/topology.py:307-314`); both hand
`calculation.diagonalize` a **scalar** `ethr = max(ETHR_MIN, 0.1 min(1e-2, conv_thr/nelec))`
(`workflows/nscf.py:285`, `workflows/topology.py:307`), and a scalar holds every band to it, empty
ones included (`scf/driver.py:6291`'s docstring, and `calculator.py:201-205`: "a band-structure or
response run diagonalises every band at `ethr` regardless"). `empty_band_threshold(ethr)` is applied
only through `band_thresholds` inside the SCF loop, so the sentence "a `get_berry_curvature` on a
`conv_thr = 1e-6` nscf run has empty bands converged to 1e-5" is false: those bands are converged to
`ethr`, about 1e-8 on silicon, and what a symmetry-degenerate pair is left split by is B1's measured
shape, an arithmetic floor near 5e-12 and then **a few per cent of `ethr`** above it (`OPEN.md`
765-767: 5.5e-12 at `ethr` 5.6e-13, 5.5e-12 at 5.6e-11, 1.24e-10 at 5.6e-9, which is 0.022 `ethr`;
and `OPEN.md` 788's 3e-7 on an SCF's empty band against its 1e-5 floor is 0.03). At the three sites'
defaults (`conv_thr` 1e-10, 1e-10 and 1e-8) that split is on the floor, 5e-12 to 3e-10, and every one
of the existing constants already catches it, by two decades or more. So the headline "1e-6 and 1e-8
become 1e-5 at four sites" is withdrawn, and so is the plan's "five of them below the solver's
promise": at the defaults none is. What the rule delivers is a tolerance that is a function of the
states' provenance rather than a constant, so that a run at a loose `conv_thr` (the ultracell's
`states_conv_thr` at 1e-4 and beyond, row 5) is still guarded, and no default changes anywhere unless
step 2 measures a ratio that says otherwise. The floor 1e-5 belongs to a consumer of **SCF** states,
and there is exactly one of those in the table, `green.py`'s `resolve=False` branch
(`workflows/transport.py:245`, `result.eigenvalues_by_spin`), plus any caller-built array of unknown
provenance, which is B1's own argument for the constant and still holds.

### A. The nine sites, verified

| site | what it guards, as read | the states it reads, and the promise on them | the rule | the kernel |
|---|---|---|---|---|
| `response/conductivity.py:836` | `_pair_weights` (`:846`) builds `t = W_n (1 - f_m)/e_mn` behind a step shared by both routes; `_curvature_sum` (`:894`) divides by `e_mn` once more. The frequency route's `1/e_mn` cancels analytically for a smeared `f` (B1, `OPEN.md` 712-722), so the step is inert there; the curvature route keeps `1/g`. | nscf, scalar `ethr` from `conv_thr = 1e-10`; the split on B1's floor, 5e-12 | yes; `degeneracy_tol` already reaches the facade by name (`workflows/conductivity.py:56`) | **curvature route only** (see D: the frequency route's identity against P37 is held at 1e-10 and a kernel at `eta = 1e-5` moves a pair at a gap of 0.1 Ry by 1e-8) |
| `topology/kubo.py:105` | `kubo_from_matrices` (`:185-186`) weights **every** pair by `1/gap^2`; `total` keeps occupied/empty pairs only (`:190-192`), and `DFTSource._check_gap` has refused any occupied/empty gap below `gap_tol = 1e-4` before the sum is built (`workflows/topology.py:131`, `:337-357`), so **the step never fires on `total`**; it fires on `curvature_by_band` (`:194`), which sums each occupied band against every other and is gauge dependent inside a multiplet by rule D4, as its own docstring says (`:70-76`). | nscf, scalar `ethr` from `conv_thr = 1e-8` | yes, and the option has to be **named**: today it reaches the method through `**kwargs` (`workflows/topology.py:452`) and `_defaults_for` filters by named parameter (`calculator.py:2449-2470`), so `get_berry_curvature(degeneracy_tol=...)` is dropped silently | **no**: on `total` it is a null by construction, and on `by_band` it bounds a number D4 says is not defined |
| `transport/green.py:211` | `multiplets` (`:254-280`, host numpy) groups eigenvalues into blocks from each block's opening value and `channel_basis` diagonalises `S` inside each block: a block decision, as the plan says | SCF states on the `resolve=False` branch (`workflows/transport.py:245`), where an in-window empty band is at `empty_band_threshold(ethr)`, 1e-5, and a symmetry pair there is split by about 3e-7 (`OPEN.md` 788), under the 1e-6 block tolerance by a factor of three; nscf states on the other branch (`:250`) | yes, the one row that reads SCF states, with `SCFResult.ethr` in hand; the margin of three is the thinnest in the table and is what the rule's floor term is for | no, as the plan says |
| `response/photocurrent.py:180` | `tol` is used two ways: block membership in `band_velocity_difference` (`:325`, the D4 average) and a weight guard in `_safe_ratio` (`:252`) on `r_nm` and the three `1/w_nm` of `generalized_derivative` (`:331-396`), over **all** pairs including occupied/occupied, since the intermediate sum runs over everything. Resolved to `max(broadening, 1e-8)` at `:611` and `shg.py:656`; a third consumer the plan missed, `_warn_about_the_cut` (`:896`), compares `band_cut_gap` to the raw constant. | nscf, scalar `ethr` from `conv_thr = 1e-10` | yes, on the floor and on `:896`; inert at the entry point, as the plan says | **no**: with `eta = broadening` (1e-2 Ry) the dipole becomes a Lorentzian, different physics, and `(eta/gap)^2` is one to ten per cent where the Elk peak agrees to 0.8 per cent; with `eta = 1e-5` under a step at 1e-2 it is never reached, and inside a block it would re-admit `r_nm` at `O(split/eta^2)`, basis dependent, where dropping the block is the degenerate-perturbation answer |
| `ultracell/driver.py:202` | `multiplet_cut`'s gap across the `nbnd` cut against 1e-8, warned only when `nbnd <= 2 occupied` (`:1324`) | the frozen solve at `states_conv_thr`, default 1e-6 (`:932`, `:960-963`), scalar `ethr` about 1.25e-8 on silicon, so by B1's ratio a symmetry multiplet at the cut is split by about 3e-10 and the 1e-8 guard catches it; the guard is expected to miss only past `states_conv_thr` of roughly 1e-4 on silicon, **and that is a forecast, measured in step 4**; the docstring at `:195-201` says the threshold was never re-measured against this gap | yes, with the ultracell's own `ethr`; the one site whose `conv_thr` a user loosens on purpose for a large `nbnd` | no |
| `topology/berry.py:323` | `1/gap^2` on a model, 1e-8 of the band width (`:288`); `:360` is the same step | model, no solver | no, left alone | left alone |
| `tddft/spinchi0.py:654` | the switch between `(f_i - f_j)/(e_i - e_j)` and `df/de` at 1e-8 (`:639-645`) | SCF pairs near `E_F` | **none**, correcting the plan's "the rule, for the switch": the two branches agree at the switch to `O(gap)` (`f' + f'' gap/2` against `f'`), so the step is not a discontinuity and the solver's promise does not enter; one docstring sentence | no, the divided difference is the exact kernel |
| `scf/tetrahedra.py:94` | `_BAND_DEGENERATE` groups bands that share weight (`:648`), inside the SCF loop on every iteration's eigenvalues | SCF, at the **scheduled** `ethr`, which starts near 1e-2 | **no**, withdrawing the plan's "with `ethr`": a tracking rule would group distinct bands in the early iterations; the constant stays, as `pw.x`'s does | no |
| `topology/parity.py:73` | `parity_eigenvalues` checks `|P - P^H|` and `||P| - 1|` (`:99-125`): a closure check on the window, not a gap tolerance | occupied, SCF | **no**, correcting the plan: there is no gap here to resolve | no |

Three constants the sweep found that the table does not list, all already at the floor and to be
named in the rule's docstring as reached and unchanged: `response/sternheimer.py:262`
`DEGENERATE_CUT_RY = 1e-5` (the refusal of a cut multiplet, on an SCF result whose first empty band is
at the floor, so it is exactly the rule's value), `response/effmass.py:118` `DEFAULT_DEGENERACY_TOL =
1e-5` (the multiplet at the centre k), and `workflows/topology.py:131` `gap_tol = 1e-4` (the refusal
that makes the Kubo step inert). `response/spectra.py:108` `DEGENERACY_TOLERANCE = 1.0` is in cm^-1 on
phonon frequencies and is not a site. No caller puts the three assemblies under `jax.grad` (the
`jax.jvp` in `kubo.py` is the velocity input), so the kernel's tangent argument is forward looking and
the rotation test alone is the acceptance.

So the rule touches five sites (conductivity, Kubo, `green.py`, photocurrent with its three consumers,
the ultracell cut), the kernel one (the conductivity's curvature route), and four rows are left as they
are with a sentence each (`berry.py`, `spinchi0.py`, `tetrahedra.py`, `parity.py`).

### B. The validation numbers, located

| site | number, phase, `PLAN.md` lines | tolerance asserted | what the kernel may move |
|---|---|---|---|
| `conductivity.py`, frequency route | `eps = 1 + 4 pi i sigma/w` reproduces `run_absorption`'s `epsilon_no_local_fields` on silicon to **3.6e-14** in the diagonal (P51, 9492-9493) | `tests/regression/test_conductivity.py:135`, 1e-10 | nothing: the kernel does not land on this route |
| `conductivity.py`, curvature route | B1's nonmagnetic SOC nickel, 102 Kramers pairs at `E_F`: `max|sigma|` 4.669e-4 and `sigma_xy` 5.07e-5 S/cm at the 1e-5 guard against 1.2576 and 3.23e-2 at the `ethr`-derived one (`OPEN.md` 746-751); the Kubo sum against the real-time first order 2.4e-6 of the scale (P141, 25122-25131, a dense-model Kubo sum, to keep unchanged rather than a validation of this module), test bound 5e-5 | the nickel number is a residue with no reference tolerance; the test is `test_a_nonmagnetic_crystal_has_no_hall_conductivity` and the isotropy bounds at `test_conductivity.py:324-325` (2 per cent) | the synthetic sweep across `eta` is the measurement (E, step 5); the nickel residue must not grow, and the rotation is a control |
| `kubo.py` | `Omega_kubo = 0.964705` at `k = (0.1875, 0.3125, 0)`, `nbnd = 30`, against the FHS plaquette to **3.2e-3** relative, and the 24x24 mesh to 1.45e-4 (P47, 7858-7869) | `tests/regression/test_kubo_curvature.py:247-250`, 5e-2 and 1e-2 | nothing: no kernel lands; the rule at the default changes no number |
| `photocurrent.py` and `shg.py` | AlAs `sigma^xyz` **-1.78e-5 A/V^2 at 4 eV**, silicon zero (P53, 10237-10241); SHG against Elk: resonance 0.5 per cent, peak **0.8 per cent**, static 8 per cent, the three parts under 35 (P54, 10358-10373); the hierarchy against the propagation 1.6e-5 to 1e-4 at 800 steps (P137, 24836-24846, the real-time route, which does not pass through this constant) | `tests/regression/test_shg.py:141-142` (allowed spread 1e-6 relative, forbidden below 1e-3), `:176` (silicon below 1e-3 of AlAs), `:323` (the static sign) | nothing: no kernel lands |

### C. The five questions

1. **Yes.** P148's gate reads 3654 passed, its "What a user sees" is written and "What is
   outstanding" already attributes items 6 and 4 to it; flip the header to DONE in P149's first
   commit, and items 5 and 3 are P149.

2. **The electron-phonon rule, decided per outcome.** Let `T*` be the loosest CG threshold at which
   both `|pair(rotated) - pair(unrotated)| <= 7.8e-5` cm^-1 and the pair's split against `ph.x`'s is
   within 7.8e-5. If arm 2 (the warm start at `phononq.py:902-905` off, the schedule kept) meets the
   acceptance, pick **(c)**, unless its CG call count exceeds that of one tight final pass, in which
   case **(b)**. If arm 2 fails but one more `respond` at `T*` from the converged `dvscf` meets it,
   pick **(b)** with the last pass at the measured `T*`, not at an assumed 1e-12. If neither does,
   pick **(a)** with the floor lowered to `T*` at every pass, and if `T*` is 1e-12 the default becomes
   `CG_THRESHOLD` and the 1.4x is the recorded price. Two more readings decide between the mechanisms
   before the rule is written: the CG iteration count and the residual norm each pass actually reached
   (a linear fall-off of the split with the threshold says the residual carries it, a plateau and then
   a cliff says the warm start does), and the size argument, since `D(q)` is linear in `dpsi` and a
   residual of 1e-8 cannot by itself give 0.015 cm^-1 on 74, so if the split does not fall linearly
   the item is not a threshold rule at all and goes back to `OPEN.md` with the numbers.

3. **The rule lives in `solvers/davidson.py` beside `empty_band_threshold`**, since it reads the same
   promise, and no new module. Its form: `resolve_degeneracy_tol(asked, ethr=None, *,
   empty_floor=False)` returning `max(asked, EMPTY_ETHR_FLOOR)` for `ethr=None` (an array of unknown
   provenance, B1's argument), `max(asked, SPLIT_PER_ETHR * ethr)` for states every band of which was
   held to the scalar `ethr`, and the same with `EMPTY_ETHR_FLOOR` joined for an SCF result whose empty
   bands were at the floor; `SPLIT_PER_ETHR` is measured in step 2 below, the prior being 0.02 to
   0.03 from B1's two readings (0.022 on the nscf route, 0.03 on an SCF's empty band), and cegterg's
   5 being the promise on the eigenvalue change rather than on the split, so it is not copied. With
   that prior the `c ethr` term is below every site's `asked` at every realistic `conv_thr`, so the
   term is there for the loose runs and the record says so. `degeneracy_tol` joins `SHARED_OPTIONS` (it means the same thing everywhere, which is the
   criterion), is **not** added to `_TOLERANCE_OPTIONS` (there a smaller value wins; here a larger one
   is the safe one), and is named in each of the eight signatures the rule reaches: `run_conductivity` (already),
   `run_berry_curvature` (today `**kwargs`), `run_shift_current` and `run_shg` (already),
   `run_vertical_transport`, `run_momentum_transport` and `run_ultracell_transport` (today `tol`, kept
   as an alias for one release), and `run_ultracell` for the cut. It does not reach `get_z2*` or the
   tetrahedron DOS, whose constants are not gap tolerances (A). `fixed_density_states` returns a
   four-tuple to many callers, so the `ethr` formula at `nscf.py:285` and `topology.py:307` is factored
   into one helper the workflow calls again to resolve the tolerance, rather than changing that return;
   the `NSCFResult` and `BandStructure` fields are one line each and are a convenience for a
   caller-built path, not the rule's input.

4. **Leave both at 1e-6, for different reasons stated in each docstring**: the tetrahedron grouping
   runs at the scheduled `ethr`, which starts near 1e-2, so a tracking rule would merge distinct
   bands in the first iterations; the parity tolerance is a closure check and not a gap at all.

5. **The kernel lands at one site, the conductivity's curvature route**, since that is the only one
   with a reachable discriminating case (B1's nickel, a `1/g` left after the numerator cancellation).
   The Kubo `total` cannot be moved by a solver-split pair (`gap_tol`), the photocurrent's operative
   guard is the broadening and not the solver, so neither gets it; the plan's "three" becomes one,
   beside P148's subspace solve.

### D. What is wrong or missing in the plan

- **The route.** The premise that nscf states carry the empty-band floor is false (the opening of this
  section), and the split above the arithmetic floor is a few per cent of the threshold rather than a
  multiple of it, so "what the rule delivers" is: no change at any site's default, and a tolerance
  that is a function of the states' provenance rather than a constant, which guards a run at a loose
  `conv_thr` that a constant does not. The phase record says that, not "1e-6 and 1e-8 become 1e-5"
  and not "five below the solver's promise".
- **The plan's kernel acceptance transplants the SCF split onto the nscf route.** "A pair the solver
  splits by its own 3e-7 (an nscf at `conv_thr = 1e-6`)" is `OPEN.md` 788's number for an SCF's empty
  band at the 1e-5 floor; an nscf at 1e-6 leaves about 3e-10 on silicon, which every guard in the
  table catches. The pair the acceptance needs is synthetic (next bullet but one).
- **Where `ethr` has to be carried.** None of the three sites consumes `NSCFResult`, `BandStructure`
  or `PlaneWaveStates` for the guard; `build_plane_wave_states` (`topology/states.py:816`) carries no
  `ethr` and the Kubo sum takes `energies` arrays. The `ethr` is in hand where the states are solved
  and is passed down as a named argument; the result fields are optional.
- **Two experiments whose null cannot be told from a pass.** A rotation inside an occupied multiplet
  read on the Kubo `total` moves nothing by construction and proves nothing; the kubo entry of the
  harness reads `curvature_by_band`'s block trace instead, which is the invariant the D4 average
  defines, and `total` as a control. And a rotation applied to `SCFResult.wavefunctions` is a null at
  the three workflow sites, which re-solve their own states inside `run_*`; the harness must call the
  pure assemblies (`optical_conductivity(calculation, wavefunctions, eigenvalues, ...)`,
  `kubo_from_matrices`, `shift_current`) with the rotated states, as `test_electron_phonon.py`'s gauge
  test did for the phonon.
- **The guard-fires test.** On the nscf route the split is 5e-12 to 1e-9 and both the old and the new
  value catch it, so a unit test on real states passes with either; the test takes a synthetic pair
  placed between the old constant and the resolved value, as
  `tests/unit/test_conductivity_machinery.py:259` does at 3e-7, and asserts the count.
- **The kernel's site inside `conductivity.py`.** Applied in `_pair_weights` it enters both routes and
  moves the frequency route's diagonal by `(eta/gap)^2`, about 1e-8 at a gap of 0.1 Ry, which breaks
  the 3.6e-14 identity asserted at 1e-10 (`test_conductivity.py:135`). It lands in `_curvature_sum`
  only, as `W_n (1 - f_m) gap^2/(gap^2 + eta^2)^2`; `_pair_weights` keeps its step and its `dropped`
  count, and `OpticalConductivity.degeneracy_tol` and `degenerate_pairs` keep their meaning (the
  count becomes "pairs inside `eta`", said in the docstring and in `_report_dropped_pairs`).
- **The rotation arm at the conductivity is a null.** The step already sits at 1e-5, so at the
  solver's split (5e-12) and at a synthetic 3e-7 the step drops the pair before and the kernel
  suppresses it after; both arms read zero and a rotation moves nothing either way. What tells a
  kernel from a step is a synthetic pair swept across `eta` (0.3, 1, 3, 10 `eta`, the machinery test's
  form at `test_conductivity_machinery.py:187`), where the step jumps from 0 to the full `1/g` at
  `gap = eta` and the kernel traces the continuous curve `gap^2/(gap^2 + eta^2)^2`; that sweep is
  the discriminating arm of step 5, the nickel residue and the P37 identity are its bounds, and the
  rotation is a control.
- **Differentiability.** `gap/(gap^2 + eta^2)` and its square are rational and smooth at `gap = 0`
  (slope `1/eta^2`), so the kernel needs no `where` and no clamp; the existing double `where` in
  `_pair_weights` and `_safe_ratio` is already the correct differentiable form of a step, so nothing
  there is a clamp to fix. The `dropped` count is an integer and stays outside any tangent.
- **The multiplier.** `empty_band_threshold`'s 5 is not the split a symmetry pair is left with; B1
  read 0.022 `ethr` above the floor on one cell and 0.03 of the floor on an SCF's empty band.
  `SPLIT_PER_ETHR` is measured on the harness cell at three `conv_thr` (1e-4, 1e-6, 1e-8, since at
  1e-10 the split is on the arithmetic floor and reads nothing about `ethr`) before the rule is
  written, and a k-point `_say_what_did_not_converge` flagged is outside the promise, said once in
  the docstring.
- **Item 5b's size argument** (C.2): the plan reads the split against the threshold but not the
  residual actually reached, and without that reading a result at `T*` cannot be told from a
  coincidence of the CG path.
- **The ultracell row** is the one real default change and the plan's table has it as a refusal
  threshold only; its re-measurement (a cut inside a symmetry multiplet at `states_conv_thr = 1e-6`,
  the gap read, the warning's firing checked) is a step of its own below.

### E. The order of work

1. Flip P148's header to DONE; open the P149 record with the opening finding of this section.
   Acceptance: the record names the route and withdraws the "become 1e-5" claim.
2. The harness, one script on D22 reused from `review/degeneracy/`'s rotation: silicon's threefold
   at `Gamma` and the `X` pairs on the nscf route at `conv_thr` 1e-4, 1e-6, 1e-8, the split of each
   symmetry pair read against `ethr`, and the same on an SCF's empty bands against
   `empty_band_threshold(ethr)`. Acceptance: `SPLIT_PER_ETHR` as a number with its spread against
   the prior 0.02 to 0.03, and the arithmetic floor confirmed near B1's 5e-12.
3. The rule in `solvers/davidson.py`, the option in `SHARED_OPTIONS` and named at the seven
   signatures of C.3, the `ethr` helper factored out of `nscf.py:285` and `topology.py:307`, the two
   optional result fields, one synthetic-pair unit test per site (D). Acceptance: each test fails
   with the old constant and passes with the resolved value; `tests/unit/test_calculator.py`'s
   forwarding test sees `degeneracy_tol` arrive at every named entry point.
4. The ultracell cut re-measured: a cut inside a symmetry multiplet, the gap `multiplet_cut` returns
   at `states_conv_thr` 1e-6, 1e-5 and 1e-4, and the `states_conv_thr` at which the 1e-8 constant
   stops firing and the resolved tolerance still does. Acceptance: the three gaps recorded with the
   threshold they were solved at, the crossover `states_conv_thr` named, and
   `UltracellResult.multiplet_gap` unchanged at the default.
5. The conductivity kernel on the curvature route. The discriminating arm is the synthetic pair of
   `test_conductivity_machinery.py:187`'s form swept across `eta` (0.3, 1, 3, 10 `eta`): the step
   reads 0 then the full `1/g`, the kernel the continuous curve. The bounds are B1's nickel case
   (`ni-soc-nosym.in` with `starting_magnetization = 0`, the 102 Kramers pairs), `sigma_xy` and
   `max|sigma|` before and after within the residue of `OPEN.md` 751 (5.07e-5 and 4.669e-4 S/cm),
   the P37 identity still at 1e-10 and P141's 2.4e-6 unchanged; the rotation inside the pairs is a
   control, read at round-off before and after and stated as a control.
6. The rotation controls at the sites that get no kernel: the Kubo `by_band` block trace invariant
   to round-off under rotation with `total` as the control, the SHG unchanged under a rotation
   inside silicon's `Gamma` triplet. Acceptance: round-off, stated as controls rather than results.
7. Item 5b's sweep (the plan's two arms, plus the per-pass CG count and residual), the decision by
   the tree of C.2, and the default changed. Acceptance: the default reproduces the 1e-12 pair
   (73.995394, 73.995396) to 7.8e-5 under the rotation, the cost recorded as a call count against
   the current schedule.
8. Item 5a's docstring sentence at `hubbard/occupations.py:593`.
9. The texts: `features.tex`'s box for each `get_*` the option reaches, `scope.md`'s line, the
   docstrings of the four left-alone sites and the three reached-and-unchanged constants, the P149
   record with every number above, and `notes/DEGENERACY-NEXT.md` item 3's table corrected to A.

### Follow-up (2026-10-14, later)

1. **The step stays at the curvature route; the rule alone lands, and the arithmetic goes in the
   record.** The two orderings of a pair leave `w (f_n - f_m) Im z K(gap)`, and with
   `K = gap^2/(gap^2 + eta^2)^2` and `f_n - f_m` about `f' gap` that is `w f' Im z gap^3/eta^4`:
   at the synthetic 3e-7 with `eta = 1e-5` the factor is 2.7, times `f'` of order `1/degauss` (50 at
   0.02 Ry), so about 135 `w Im z`, against exactly zero from the step and against 100 `w Im z` from a
   legitimate pair at 0.1 Ry; at the solver's real split, 5e-12, the factor is 1e-14 and the kernel
   is harmless. So the kernel buys continuity in `eta` and not gauge safety below it, and the number
   it gives for a symmetry pair split by noise is a fiction of the same size as a real pair's, where
   the step's zero is the physical answer (the true gap is zero and the two members cancel). The
   continuity it would buy already exists in the module: a Lorentzian regularisation of the
   curvature route is the frequency route at `frequencies = [0]` with `broadening = eta`, and that
   call is what a user who wants a smooth `eta` dependence makes. So E step 5 is withdrawn, the
   sweep across `eta` is recorded as the reason, C.5's answer becomes "no site beyond P148's", and
   the phase's kernel content is the measured argument above rather than a change.

2. **Yes, a third term of the max, `ARITHMETIC_FLOOR`, seeded at 1e-10.** B1 read 5.5e-12 at
   every `ethr` below about 2.5e-10 on one cell, nickel with spin-orbit coupling at one cutoff, and
   the floor is the round-off of a Rayleigh quotient, so it scales with the Hamiltonian's norm (the
   cutoff) and with `npw`; a factor of twenty over one cell's reading is the margin for that, and the
   split harness (E.2) reads the floor on its own cells and raises the constant if any reads above
   1e-11. With every site's `asked` at 1e-8 or above the term is inert at the defaults and acts only
   on an explicit ask below 1e-8, which is item 3's case.

3. **(a): an explicit number is honoured as asked, and only `None` resolves through the rule**, with
   one addition, a warning (never a raise) when the explicit ask is below the resolved value, naming
   both, so that the knob says what it is doing. The reason is the project's own: B1's bimodality
   reading, 42 pairs at 5.56e-13 against 102 at 1e-10, is a measurement, and a knob that is silently
   raised cannot make it; the safety belongs in the default, where `None` now means "the solver's
   promise on these states". `tests/regression/test_conductivity.py:452-453` keeps its 42 and gains
   `pytest.warns` on the raise, which is also the guard-fires test for that site, and the option's
   docstring in `SHARED_OPTIONS` says "`None` resolves, a number is yours". The plan's phrase "at
   least this, and never below the solver's promise" is replaced by that sentence.

On item 5b's sweep as reported (`review/degeneracy/elph`): arm 2 at 5.6e-4 fails the acceptance, so
by C.2's tree the running one-tight-final-pass arm decides between (b) at `T* = 1e-8` and (a) at
`T* = 1e-8` with its 1.7x; and the fixed 1e-6 result, gauge-safe at 6e-5 but 1.4e-2 off the tight
pair, says the gauge split and the frequency accuracy are set by different thresholds, so `T*` is the
accuracy's (the 6e-3 bound against the 1e-12 pair) and the record says which of the two each
threshold meets.

### Follow-up 2 (2026-10-14, the cap against the periodicity test)

**(C), in the one form the table has already measured: the cap stays at 1e-8 by default, and `tr2`
defaults to 1e-16 whenever the cap is active**, the `None` arm keeping `tr2 = 1e-14` as `ph.x`'s
loop for like-for-like pass counts. Not (A), because moving a test to the setting that makes it
pass while the default leaves a less converged potential is the trap of a tolerance the default does
not deliver, and the periodicity residue at 1.027e-5 is a fact about the default and not about the
test. Not (B), because the dynamical matrix of a metal is one object and two entry points with two
defaults for it would leave `get_phonons_at_q` on aluminium with the 1.42e-2 split that
`get_electron_phonon` on the same cell no longer has. Not the "two consecutive passes below `tr2`"
variant, which is unmeasured, where the `tr2 = 1e-16` variant is rows four to six of the table on
silicon and the aluminium reading on the other cell: silicon capped at 1e-16 reads 8.22e-7 on
`D(q) - D(q + G)` (the Ewald floor, 8.25e-7 uncapped at 1e-14) and the TA pair split 4.3e-5 (1.0e-2
under the bare schedule at 1e-14, 2.6e-4 uncapped at 1e-16), in 11 passes, the same count as the
old default, at 16.8 against 12.0 CG steps a solve, 1.4x; aluminium capped at 1e-16 is 2.2e-7 under
the rotation and 9 passes at 301 steps, identical to capped at 1e-14, 1.7x the uncapped 178. So the
default pair is clean on both cells and costs 1.4x to 1.7x in CG steps, which is the price the tests
already paid at `CG_THRESHOLD`.

**What the stop-short is.** With the CG noise gone the change `|ddv_scf|^2` falls below 1e-14 two
passes earlier on silicon at a potential whose periodicity residue is twelve times the Ewald floor,
so at 1e-14 the stopping test reads the pass-to-pass change and not the distance to the fixed point,
and the noise was what kept the uncapped loop going to a better potential. That sentence goes in the
`tr2` docstring and in the P149 record with the table above; it does not go to `OPEN.md`, since the
default pair cures it and the `None` arm is documented as what it is.

**Test consequences.** `test_phonons_at_q.py::test_the_matrix_is_periodic_in_the_reciprocal_lattice`
keeps its 1e-5 bound and its Ewald-floor docstring and reads 8.22e-7 at the new default with nothing
retuned; every other phonon-at-`q` and electron-phonon number is re-read under the default pair
through `tools/run_regression.sh` on the two files, with any move recorded against its old value;
`test_electron_phonon.py` may drop its `CG_THRESHOLD` pin for the default and keep the 1e-12 pair
(73.995394, 73.995396) as the reference it is read against, since the capped default reproduces the
fixed 1e-8 arm at 3e-6 from it.

**The silicon TA pair's 1.0e-2 under the bare schedule at `tr2 = 1e-14` does not go to `OPEN.md`.**
It is the aluminium disease on an insulator and it is cured by the default; it goes in the P149
record as the second cell the cap was measured on, and the `threshold_cap = None` docstring says
that `ph.x`'s loop leaves a degenerate pair split by the CG path, 1.0e-2 cm^-1 on this cell and
1.42e-2 on aluminium, which is what the cap is for.
