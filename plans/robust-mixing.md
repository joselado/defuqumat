# Plan: a mixing mode that converges almost every cell (`mixing_mode = 'robust'`)

Repository: /u/40/ladovj1/data/Documents/programs/claude/defuqumat (package `defuqumat/`).
Written 2026-10-15 on master at `78fb499`, at the user's request: "a type of mixing in the SCF
that is intelligent, meaning that depending on how the SCF evolves it switches to one mode or
another with different parameters ... it does not have to be the most efficient one, but it
should be the most robust: if you choose this mode, it is guaranteed, or almost guaranteed, that
the SCF converges". Builds on `notes/MAGNETISM-NEXT.md`'s controller-hook item.

**Status: plan, reviewed by fable on 2026-10-15 (its decisions are section 7), no code.** The
review checked the plan against the code at `78fb499` and the paper's text, found the central
claim true in a form the draft did not have (the merit function is the loop's *converged*
energy branch, run every iteration, not its mid-run total), found that no change to
`AndersonMixer` is needed for backtracking, moved the guaranteed rung from density mixing to
Hamiltonian mixing, and cut R0 to ten cells.

## 0. The question, and the answer in one paragraph

What makes an SCF fail? Three different things, and a mixer that is robust has to deal with each
by name rather than by a smaller `beta`:

1. **Ill-conditioning**, the long-wavelength charge sloshing of a metal and the metal beside
   vacuum of a slab: the Jacobian of the SCF map has eigenvalues far from 1, and Anderson needs a
   deep history or diverges. The cure is a preconditioner, and `PLAN.md` P129 measured which one:
   `'ldos'` is flat in the vacuum where `TF` and `local-TF` are not.
2. **Nonlinearity far from the solution**: the first steps of an elongated rattled supercell, a
   magnet started from the wrong order. Anderson's secant model is wrong there, a full step
   overshoots and the history it then fits is garbage. The cure is a **line search**: take the
   step only when a merit function says it went downhill.
3. **Anderson's own stagnation**: the residual stops falling at 1e-4 to 1e-6 with no physical
   reason, on transition-metal cells above all. Herbst and Levitt found that a restart with a
   tight bound on the least-squares conditioning (1e2) unjams it where a restart alone does not.

The design is a **supervisor** around the mixers we already have. Anderson with a preconditioner
proposes a step; a **safeguard** accepts it only if the Kohn-Sham free energy of the output states
went down, or the residual did (Herbst and Levitt's adaptive damping), and otherwise shrinks it;
and a **stagnation monitor** walks a fixed, recorded ladder of remedies from the best iterate seen
so far: a history restart with tighter conditioning, a different preconditioner, a smaller
magnetization step, a shorter history, Newton-Krylov, and at the bottom damped **Hamiltonian**
mixing with an energy-only Armijo search, which is the one rung with a convergence theorem. What
"almost guaranteed" means is section 2; where it does not hold, the mode refuses or warns by name,
and a run that reaches the bottom rung and still cannot lower the energy stops with a status that
says so.

## 1. What exists, and what each piece contributes

- `scf/mixing.py`: `linear`, `anderson` (`condition_limit = 1e12` trimming, the Gram cache, the
  `rho_ddot` fit by default since P146), `kerker` = QE's `TF`, `local-TF`, `'ldos'` (Herbst and
  Levitt's LDOS preconditioner, P129), `adaptive` (Elk's `mixadapt`), `mixing_beta_mag`,
  `mixing_space = 'r' | 'g'`. `broyden` is an alias of Anderson.
- `scf/solvers.py`: `newton_krylov` with a finite-difference JVP, which already backtracks on the
  residual norm (four halvings) and takes `(x0, psi0)`; since P150 also the exact Sternheimer
  Jacobian, which lost on wall time in every P150 row and refuses DFT+U, meta-GGA, magnetic
  noncollinear runs and a streamed store (`scf/residual.py:300`).
- `scf/driver.py` `run_scf`: the loop is Python, the iteration body jitted. `_mix` packs density,
  `becsum`, `ns`, `tau` and returns all four mixed. `accuracy` is QE's `dr2`, split into charge
  and magnetic parts. **There is no `descf`** (the word is only in the comment at `:8405`): the
  mid-run `total` (`:8693-8727`) is `eband + deband + E_H[rho_in] + E_xc[rho_in] + ...`, a
  Harris-like value with no lower bound. What reacts to the run today: the `ethr` schedule, QE's
  first-iteration redo, `mixing_fixed_ns`, the field updates, the orientation stepper's trust
  radius, and Anderson's own conditioning trim. **Nothing detects a stall, changes `beta` during a
  run, clears the history on a blow-up or swaps the mixer.**
- `notes/MAGNETISM-NEXT.md` (the item "A per-iteration controller hook") already specifies the
  hook: a callback on the `history` entry that may change `beta`, clear the history or swap the
  mixer, every decision written to `history` and its state checkpointed, with Herbst and Levitt
  named as the principled policy for it. The supervisor is that hook with its first policy.

## 2. What "guaranteed" can mean here

The merit function is Herbst and Levitt's, written for this code:

    I(rho_in) = E_KS[P_out],   P_out = the states and occupations of the Hamiltonian built from rho_in

the Kohn-Sham free energy of an actual density matrix, so `I(rho_in) >= E_min` for every input
(the minimum over admissible `P`, occupations in `[0, 1]`, `Tr P = N`), with equality at a
self-consistent minimum. **It is the loop's converged energy branch run every iteration**
(`driver.py:8653-8672`): `calculation.potential(rho_out)` for Hartree and XC, `onecenter(becsum_out)`
for PAW, `hubbard_energy(ns_out)`, because the other terms are already at the output:
`deband = -int rho_out v_in` (`_iteration_scalars`, `:1510`), `_paw_deband(ddd_paw_in, becsum_out)`
(`:8445`), the Hubbard `ns_out v_ns_in` (`:8690`), and `-TS` from this iteration's eigenvalues
(`:8340`). So `eband + deband` is exactly `Tr[(T + V_ext) P_out]`. A fixed external field or a
penalty constraint is admissible only if `I` adds `potential.e_field` and `e_constraint` at
`rho_out` (reported today and never added to the total, `:8795-8796`); the saw-tooth term comes
with `potential(rho_out)`. **Cost:** one `v_of_rho` on the dense grid (genuinely extra, since the
next iteration builds it at the mixed density), one jitted `_paw_onecenter`, one `hubbard_energy`;
no diagonalization. The PAW term is the one to time.

**What Herbst and Levitt proved is narrower than what the mode does**, and the plan keeps the two
apart. Their Theorems 1 and 2 (arXiv:2109.14018, section II) are for **Hamiltonian mixing along
the plain SCF direction**, `H_{n+1} = H_n + alpha (H_KS(P(H_n)) - H_n)`, a fixed step below some
`alpha_0 <= 1` (Theorem 1) or Armijo backtracking from `alpha_max < 1` (Theorem 2). They state it
for Fermi-Dirac and say it "applies just as well to Gaussian smearing" and not to "non-monotonous
occupations such as Methfessel-Paxton". No Anderson and no residual clause: they contrast
energy-decreasing schemes with "residual-based schemes, which afford no such guarantee", and their
production algorithm is outside both theorems. So:

- **The guarantee belongs to the bottom rung only** (rung 6): damped Hamiltonian mixing, no
  Anderson, no preconditioner, energy-only Armijo acceptance. With Fermi-Dirac or Gaussian
  smearing, or a gap, it converges to a stationary point of the free energy. It is slow, on a slab
  above all since it cannot be preconditioned and keep its descent property, and it is there so
  that the ladder has a floor with a theorem under it.
- **Everything above it is "almost guaranteed" in a measured sense**: the fraction of R0's cells
  that converge, against the same fraction under the existing modes.

**Why the bottom rung mixes the Hamiltonian and not the density.** Along `rho_in + alpha r`, with
`r = rho_out - rho_in`, the slope of `I` at `alpha = 0` is `<chi0 R_V, K r>`, with
`R_V = V(rho_out) - V(rho_in)` and `K` the Hartree-XC kernel. Only for a linear `V_Hxc` is
`R_V = K r` and the slope `<K r, chi0 K r> <= 0`, meaning that the density direction is downhill.
The size of the exception: for LDA exchange, `v_x ~ rho^(1/3)`, the pointwise difference is
`K r (1 - r/(3 rho)) + O(r^3)`, so the XC part flips sign only where `|r| > 3 rho`, the Hartree
part is exact, and `chi0` weights the sum by the states at the Fermi level. In practice this does
not happen at an accepted iterate, which is why the upper rungs mix the density. Along
`H_n + alpha (H_KS - H_n)` the slope is exactly `<R_V, chi0 R_V> <= 0` with no caveat, and
`H_KS(P_out) = (v_scf[rho_out], ddd_paw[becsum_out], v_ns[ns_out])` is a by-product of computing
`I`. The sign of the density slope is one of the model's own coefficients (section 4), so it is
logged on every iteration of every cell rather than sampled.

**Not a ground state.** A magnet can land on a saddle or another minimum (the hydrogen ultracell
has three, the DFT+U nickel cell four or five, `PLAN.md` P113), so the mode reports the state it
reached, its energy and moments, and a Tier C "convergence" to a saddle is not counted as a
success.

**Refused, by name:** tetrahedra (no smooth `I`; `'ldos'` already refuses them, `:7811-7821`); a
potential-only meta-GGA (no energy functional, so no `I`); `lelfield` (a Berry-phase energy on the
states, `:8709`); the feedback fields (`FEEDBACK` in `scf/fields.py:106`, `reducebf`, the
orientation stepper), whose functional changes between iterations; the ultracell driver, the
response mixers and the spiral workflow, for the first version.
**Warned about:** Methfessel-Paxton and Marzari-Vanderbilt smearing (occupations leave `[0, 1]`,
`I` is not a lower bound and the bottom rung's theorem does not apply; the energy clause stays as
a heuristic, and the Co film, which uses Marzari-Vanderbilt, stays in the test bed); fixed
occupations (the degenerate cut is already named by `SCFResult.cut_gap`, P147).
**Delayed:** under DFT+U the safeguard starts after `mixing_fixed_ns` iterations, since `ns_adj`
and the frozen `ns` change the map before that (`:8531-8547`).

## 3. The design

### 3.1 Setup, read from the cell

Decided once before iteration 1 and recorded on the result:

- **Preconditioner.** `'ldos'` when the cell is smeared and the vacuum fraction (on the
  superposition of atomic densities: the fraction of the cell along a lattice vector where the
  planar average falls below a threshold) is above about 0.3; none otherwise. A monolayer misfire
  costs about one iteration (P129: NbSe2 10 against 9), so there is no film-versus-monolayer
  detector. Never `'ldos'` on fixed occupations.
- **Step.** The input's `mixing_beta` is the starting trial damping, as in QE; HL found the start
  matters little. `alpha_min = 0.5` on a run with `nspin > 1` and a starting moment, 0.2 otherwise
  (the split HL measured: 0.5 strictly better on every transition-metal cell, 0.2 needed on GaAs).
- **History.** `mixing_ndim = 8`, `condition_limit = 1e6` (HL's value, against 1e12 today).
- **Its own `max_iterations` default**, larger than `run_scf`'s 100, since rejected trials count.

### 3.2 Every iteration: the safeguard

1. One `mix()` per iteration, exactly as today. `mix()` appends the *current* pair `(rho_n, r_n)`
   (`mixing.py:417`), and the Anderson coefficients depend on the residuals only, so the history
   and the Gram cache are right whatever happens to the step. The supervisor takes the packed step
   `mixed - rho_n` (all four of density, `becsum`, `ns`, `tau`) and scales it by `alpha / beta`.
   **No change inside `mix()` or `step()`**, which is what keeps every existing mode
   bit-identical: `beta` is folded into each preconditioner at build (`kerker` at `:920`,
   `LDOSPreconditioner._beta`), so scaling outside is the only route that is not a last-bit change
   on every preconditioned mode. The high-frequency shell follows the same scale: linear in the
   `becsum` step on the augmentation branch (`rebuilt_shell`), `beta` times the residual otherwise
   (`shell_step`).
2. Diagonalize at the trial (the ordinary iteration), evaluate `I` and `accuracy`.
3. **Accept** if `I` fell by more than its noise, or `accuracy` fell. `accuracy` and not a
   preconditioned norm, since a preconditioner switch changes that norm discontinuously and would
   fake a decrease. The noise is the older iterate's `nelec * ethr_n`, which under QE's schedule is
   about `0.1 accuracy`, the scale of the expected decrease: the energy clause resolves about ten
   per cent after iteration 2 and nothing at iterations 1 and 2, where the residual clause carries
   the run. Never reject on the energy alone while any band is unconverged
   (`davidson_unconverged > 0`).
4. **Reject** otherwise: the model of section 4 gives the next `alpha` when it is good
   (`r_n < 0.1`, a minimum exists), halving otherwise, `|alpha|` strictly decreasing. The rejected
   pair does not enter the history (Algorithm 1's semantics; it costs nothing, since the current
   pair is already in). A rejected trial costs one diagonalization, is counted apart from
   `iteration`, and must not re-fire `next_ethr`'s `iteration == 2` reset (`:1671`) or QE's
   first-iteration redo (`:8269`, `:8573`).
5. Between iterations, the trial damping shrinks to the accepted `alpha` and grows to
   `max(alpha, 1.1 argmin phi)` after an immediate acceptance, never below `alpha_min`.
6. The **best iterate** is the lowest `accuracy` among accepted steps (`I` has no floor under
   Marzari-Vanderbilt), with its `I` recorded beside it, kept as density, `becsum`, `ns` and `tau`
   on the host. Its wavefunctions are not kept, since they would double the largest array; a
   restart diagonalizes warm from the current states.

### 3.3 Over a window: the stagnation monitor and its ladder

A stall: the minimum `accuracy` over the last `w = 8` accepted iterations has not fallen by a
factor of 2, or `alpha` has sat at `alpha_min` for `w` iterations, or `accuracy` has risen four
times in a row. The charge and magnetic halves of `dr2` say which part stalls. Each stall advances
one rung, a rung that works is kept, and every rung taken is written to the result:

1. Restart the Anderson history from the best iterate, `condition_limit` to 1e2 (HL's Fe2CrGa).
2. Charge stalls: switch the preconditioner (none to `'ldos'` on a smeared cell; `'ldos'` to
   Kerker if its inner CG does not converge). The loop already reads `wants_ldos` per iteration
   (`:8341-8376`), so the cost is compiling `density_and_ldos` and the CG once, and the new
   preconditioner is built at the current trial damping.
3. Magnetization stalls: `mixing_beta_mag = beta / 4`. Elk's `adaptive` and the flat-manifold
   projection of `notes/MAGNETISM-NEXT.md` F2 are not in the first version.
4. `mixing_ndim` to 4 and `alpha_min` to 0.05.
5. Newton-Krylov from the best iterate, `jvp='finite-difference'` only (it covers every regime and
   already backtracks), outside the mode's promise.
6. **The bottom rung**: damped Hamiltonian mixing of `(v_scf, ddd_paw, v_ns)` along
   `H_KS(P_out) - H_n`, no Anderson, no preconditioner, Armijo on `I` from `alpha_max < 1`, `ethr`
   held at `0.01 accuracy / nelec` so that the Armijo decrease clears the eigensolver's noise. The
   loop builds `potential` from `rho` (`:8215-8222`), so this needs a branch that takes the
   Hamiltonian as given, and it stops on `rho_ddot(rho_out(n) - rho_out(n-1))`, since there is no
   `rho_in` to compare with.
7. **Terminal status, not a remedy.** If the bottom rung cannot lower `I` at the eigensolver's
   floor, the run stops and says so: the residual is at the floor, the map is not single-valued
   there (the class of P147 and P148, a degenerate cut or a limit cycle such as `co-helix4-nosoc`),
   and no mixer will fix it.

Smearing continuation (converge at a larger `degauss`, then step it down) is not a rung, because
it changes which state the run reaches.

### 3.4 What crosses which boundary

- All of it is Python-side loop logic around the jitted body. Nothing changes inside a `jit`, so
  the derivatives of the converged state are untouched; a gradient through the path was never
  offered.
- **Checkpoint.** `_MIXER_DERIVED` (`checkpoint.py:577`) rebuilds `beta`, `history`,
  `condition_limit` and `beta_mag` from the input on resume, so the supervisor's record must
  reinstall them after `get_mixer` (`driver.py:7553`) or a resumed run silently returns to rung 0.
  `save_mixer` refuses anything but arrays, lists and scalars, so the supervisor's state and the
  best iterate go in a file of their own beside `save_optimizer`, policed by their own
  `unhandled_*` test.
- **Pools.** Every piece of `I` is already all-reduced, so the decision is taken on rank 0 and
  broadcast with `pools.broadcast_flag`, as `converged` is (`:8647`).
- `relax` and `vc-relax` start each ionic step at the rung and damping that worked on the last.

## 4. The model for the step, in density mixing

HL write the model for potential mixing, `V_{n+1} = V_n + alpha dV_n`:

    phi(alpha) = I(V_n) + alpha <R_n, chi0 dV_n> - alpha^2/2 <chi0 dV_n, (1 - K chi0) dV_n>

with `alpha chi0 dV_n ~ rho(V_n + alpha dV_n) - rho(V_n)`, so every coefficient comes from the
trial already computed. For density mixing the same model is taken along the curve
`V(rho_n + alpha d_n)`, with `dV = K d` to first order, so `chi0 dV ~ (rho_out(trial) - rho_out(n)) / alpha`,
and the quadratic coefficient needs one application of the Hartree-XC kernel to that change (a
Hartree division in G and one `jvp` of the XC potential, no diagonalization). The linear
coefficient `<R_V, rho_out(trial) - rho_out(n)>` is the logged slope sign of section 2. The check
is the model's own ratio `r_n`: if the density form is wrong, `r_n` is large and the code halves,
so a wrong model costs iterations and not correctness.

## 5. Phases

- **R0, the test bed, before any code.** One cell per class, one perturbation, under `anderson`
  at the input's `beta` plus the preconditioner that applies, run on `D22-0161`: the Al slab at
  32 bohr unpreconditioned (sloshing), the Co film c10 (film, Marzari-Vanderbilt),
  `fe-unstable` at 4x4x4 (magnetic state), `co-helix4-nosoc` (limit cycle), `ni-kind1-force`
  (DFT+U minima), a rattled elongated Al8 without Kerker (HL's nonlinearity; pseudopotentials
  committed), `si64-1k-ecut30` (a large cell, where the Gram conditioning reached 1.7e8, so
  `condition_limit = 1e6` fires from the start), and `si-1k`, `al2-metal` and graphene as
  controls. The output is the success matrix, the diagonalization counts, `pw.x`'s count where
  there is one, and the state each run reached.
- **R1, the merit function.** `I` assembled as in section 2 for `nspin = 1, 2, 4`, US, PAW, +U.
  Checks: equal to the converged `etot` at convergence; `I(rho_* + eps d) - I(rho_*)` quadratic
  in `eps` and positive along random `d` on a gapped cell; its cost per call against one
  iteration, the PAW one-centre term above all; the slope sign logged from iteration 1.
- **R2, the safeguard with halving only, and the bottom rung as the reference arm.** The
  acceptance test, backtracking by rescaling, the best iterate, the trial counter; the Hamiltonian
  mixing rung and its stop test. Run R0's matrix, both arms.
- **R3, the model and the adaptive trial damping** (section 4). Run R0's matrix; the controls
  stay within 1.2 times `anderson`'s count.
- **R4, the monitor and the ladder**, each rung tested by a cell that needs it, the terminal
  status by a limit-cycle cell, and a test that each guard fires on a cell made to stall.
- **R5, the setup heuristics and the front door**: `Calculator` forwards the mode, the result
  carries the record of rungs and dampings, `docs/features.tex`, the README row, a notebook (the
  success matrix as its one figure), `PERFORMANCE.md` pairs against `pw.x`.

## 6. The cells this code has found hard

Compiled 2026-10-15 from `PLAN.md`, `PERFORMANCE.md`, `OPEN.md`, `docs/features.tex` and
`notes/`. **Counts from three periods do not compare**: before P107 `becsum` was in the Anderson
fit; `22842dc` changed Davidson's collapse and moved several counts and the root the iron Newton
tests find; P146 made the `rho_ddot` fit the default (bulk about 14% slower, the Co film and the
Ni slab now converge under plain Anderson). The header of `benchmarks/fe-unstable.in` is stale
(it says Anderson reaches 3.41 mu_B; current code lands at `M = 0`, `PLAN.md` P129).

There is **no portable set of hard SCF cells**: `tools/vacuum_sweep/` is hard-wired to D22 paths,
P107's nine-cell and P146's twelve-row tables were run by hand. Phase R0 builds one.

The proposed test bed, in three tiers by what each tests:

**Tier A, the charge (sloshing, vacuum, nonlinearity).**
- `benchmarks/al-slab.in`, vacuum 16 to 64 bohr: Anderson 13/15/18/15 now, Kerker 14 to 35, `'ldos'`
  10 to 11; Anderson reached +105 Ry at iteration 2 in P22. `pw.x` 16 / 14 (TF).
- `tests/data/qe/co-slab-forcetheorem-sr.in` and `tools/vacuum_sweep/co/co-c{6,10,14}.in`: Co film,
  ultrasoft LSDA, **Marzari-Vanderbilt smearing**: flat-fit Anderson not converged at 150
  (oscillating by hundreds of Ry, +335 Ry in P58), `rho_ddot` 30 to 36, `'ldos'` 20 to 22. `pw.x` 24
  (local-TF). Note that the headline vacuum cell is under a smearing the theorem does not cover.
- `tests/data/qe/efield-ni-slab.in`: Ni slab with the dipole correction: flat fit at 0.3 diverges
  (643 at iteration 3), `rho_ddot` 27, at 0.7 diverges. `pw.x` 26.
- A Si `(8,1,1)` supercell under the applied cosine potential (P88): not converged in 100, linear
  at 0.5 diverges.
- New, from HL's Table I: a rattled elongated Al supercell without Kerker (buildable now), a
  rattled elongated GaAs supercell (needs a committed scalar-relativistic Ga dataset first:
  `tests/data/pseudo/` has Ga only fully relativistic).
- `benchmarks/si64-1k-ecut30.in`, a large cell: the Gram conditioning reached 1.7e8 there
  (`mixing.py:500-504`), so the new `condition_limit` fires from the first iterations.
- Control, must not get slower: `si-1k`, `al2-metal`, the three monolayers (NbSe2, graphene, hBN:
  Anderson 9/8/11, flat in the vacuum).

**Tier B, the magnetization (slow, stalling, wrong state).**
- `benchmarks/fe-unstable.in`: bcc Fe at a moment instability; lands on `M = 0`, 0.42 or 1.99 mu_B
  depending on the k-mesh and the mixer; `pw.x` 23.
- `benchmarks/fe-mag-1k.in`, `tests/data/qe/fe-noncolin-pbe-stress.in`: noncollinear Fe, 11 and 16
  now against `pw.x` 12 and 19; the counts moved with `mixing_ndim`, `beta_mag` and the fit.
- `tests/data/qe/ni-tetragonal-relaxed-mae-paw.in`: fully relativistic PAW Ni, stalls at every
  orientation (200 iterations, `dr2` 1e-6 to 1e-8); cured only by `degauss = 0.05` (78); `pw.x`
  diverges from iteration 1.
- `tests/data/qe/co-helix4-nosoc.in`: limit cycle near 1e-7 under every mixer tried.
- `tests/data/qe/h4-noncolin-force.in` under PBE: limit cycle at 5e-6.
- `tests/data/qe/fe2-canted-soc.in`: oscillates 1e-8 to 1e-7; `pw.x` the same at its cap.
- GaAs spinor ultrasoft at zero field (P57): not converged in 80, drifts to a spurious moment.
- `ni-soc.in` (`notes/AUDIT-2026-09-18.md`): diverges at 0.7.
- New, from HL's Table I: one Heusler (Fe2CrGa or Mn3Si from a ferromagnetic guess), which needs
  a committed Mn (and Ga) dataset first; the Cr supercell with a vacancy is buildable now.

**Tier C, multiple solutions and DFT+U (the state, not the count).**
- `tests/data/qe/ni-kind1-force.in`: DFT+U Ni, four or five minima within 7.5e-3 Ry, the state
  set by the path; `rho_ddot` 100 not converged against flat 79 at `conv_thr = 1e-12`.
- `benchmarks/ni-ldau-1k.in`: 82 to 92 against `pw.x` 98.
- `tests/regression/test_ten_site.py`'s ten-site Ni DFT+U supercell: not converged at 200.
- `benchmarks/ni-u-unstable.in`, the iron pair of `tests/regression/test_scf_solvers.py`: a saddle
  and a minimum, Anderson and Newton-Krylov finding different ones.
- `tests/data/qe/h-noncolin-ultracell.in` at a weak field: 263 at 0.7, not converged at 0.3,
  three solutions at `(4, 2, 2)`; lowering `beta` makes it worse.
- `tests/data/qe/o-atom-fixed-lsda.in` and the PAW atom: degenerate p shell, no fixed point below
  1e-9 under fixed occupations, which is section 2's "not smooth" case and should be refused or
  warned about rather than counted as a failure.

**Out of the mode's scope, listed so that the refusal is tested:** the feedback fields
(`fe2-canted-nosoc.in` with fsm, `h2-texture-120.in`, `noncolin-constrain_atomic.in`), the
orientation stepper (`co-tetragonal-relaxed-mae.in`), the spirals (`o-chain-spiral-paw.in`),
tetrahedra (`al-slab` with `tetrahedra_opt`, not converged in 200 under any mode), NiBr2 on Triton.

## 7. Decisions, from fable's review (2026-10-15)

Taken by fable under the user's delegation of such decisions (2026-10-10), one line of reason each.

1. **The merit function** is the converged energy branch run every iteration (section 2), not the
   mid-run total; the upper rungs mix the density, the bottom rung the Hamiltonian; the density
   slope's sign is logged from the model's coefficient on every iteration.
2. **A rejected pair does not enter the history**: Algorithm 1's semantics, and free, since the
   current pair is already in.
3. **`alpha_min` = 0.5 with `nspin > 1` and a starting moment, 0.2 otherwise**, 0.05 at rung 4:
   the split HL measured, and the regime is known at setup.
4. **The input's `mixing_beta` is the starting trial damping**: QE's meaning kept, and HL found
   the start matters little.
5. **No smearing continuation**: it changes which state the run reaches.
6. **Feedback fields refused; fixed fields and penalty constraints admitted** with their energies
   in `I`.
7. **The magnetic rung is `mixing_beta_mag = beta / 4` only**; Elk's `adaptive` and F2's
   projection wait.
8. **The name is `'robust'`.**
9. **The guarantee survives the `ethr` schedule** with section 3.2's provisions, and the bottom
   rung holds `ethr` at `0.01 accuracy / nelec`.
10. **Newton-Krylov stays as rung 5 with the finite-difference JVP only**, outside the promise:
    the Sternheimer JVP lost on wall time in every P150 row and refuses the magnets.
11. **A vacuum-fraction detector at setup** (above about 0.3 with a smearing gives `'ldos'`), no
    film-versus-monolayer distinction: a misfire on a monolayer costs one iteration.

## References

- M. F. Herbst and A. Levitt, "A robust and efficient line search for self-consistent field
  iterations", arXiv:2109.14018 (the safeguard, the model, the Table I cells).
- M. F. Herbst and A. Levitt, "Black-box inhomogeneous preconditioning for self-consistent field
  iterations in density functional theory", arXiv:2009.01665 (the `'ldos'` mode, P129).
- E. Cances, G. Kemlin and A. Levitt, "Convergence analysis of direct minimization and
  self-consistent iterations", arXiv:2004.09088 (local convergence of damped iterations).
- J. Zhang, B. O'Donoghue and S. Boyd, "Globally convergent type-I Anderson acceleration for
  non-smooth fixed-point iterations", arXiv:1808.03971 (safeguarding and restarts in general).
- "Preconditioning magnetic systems in Kohn-Sham density functional theory", arXiv:2606.26693
  (the Stoner term, P130, not merged).
