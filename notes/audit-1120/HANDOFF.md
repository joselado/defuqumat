# Audit fixes of 2026-11-20: the state at the end of 2026-11-21

**The fixes are finished. What is left is the user's.** This first section is the state now.
The sections below it are the checkpoint of the night before, kept as the history of how the
work was run; where they say "left" or "not run", this section wins.

## The state now (2026-11-21)

- Branch `audit-fixes-1120`, 147 fix commits to `277ce4d5`, then the record. Not merged into
  master and not pushed.
- Round three is done: group F (#13, #15, #14, #18) and group G (#24 landed; #23 held as a
  patch; #17 deferred, with its diff kept). `PLAN.md` P183 is the record, with the decisions
  taken without the user, the held patches, the open choices and the slow files.
- The proposed text is applied: dated corrections in the phases of `PLAN.md`, `PERFORMANCE.md`,
  `OPEN.md` Part XXXII (ten items), `notes/incompat/I28.md`, the `with_kpoints` docstring, and
  notebook 40. **The rule files are two commits of their own**, so that the user can drop
  them: `b7e645d3` (`.claude/rules/scope.md`, `reference-source-map.md`, the audit row of
  `CLAUDE.md`) and `2ef8103b` (`.claude/rules/calculator.md`).
- **The fast gate ran on Triton** (the workstation takes light runs only): job 21736920 at
  `6b7bb998`, 4938 passed, 65 skipped, 1940 deselected and the 5 failures of that venv (3
  watchdog without `psutil`, 2 of `test_check_guide.py` without `yaml`), 1 h 29 min 44 s, 5.50
  GB; and job 21739219 at `ceedcaf6`, after groups F and G and the record, 4969 passed and
  the same 5 failures, 1 h 20 min 19 s, 6.07 GB. The commits after `ceedcaf6` touch only
  notebook prose and the record. Both are in the gate line of P183. The four files that venv cannot
  run passed here at the head (`test_check_guide.py`, `test_memory_watchdog.py`,
  `test_result_plots.py`, `test_notebook_conventions.py`, 550 tests), and `mkdocs build
  --strict` builds.
- **No slow test was run.** The 99 files the changes reach are in
  `notes/audit-1120/slow-files.txt`, one per line. The array is prepared and not submitted
  (`tools/cluster/slow_files.sbatch`, `--array=0-98%12`, checked with `sbatch --test-only`);
  the command is in `docs/triton-notes.local.md`, "Added 2026-11-21".
- On Triton: the worktree `apps/defuqumat-audit1120`, the ref
  `refs/remotes/bundle/audit-1120` of `defuqumat-qtt` and the bundles `audit-1120.bundle` and
  `audit-1120b.bundle` are still there. Their removal waits for the user's word.
- `session/agent-reports.md` (untracked) has the three reports of 2026-11-21 at its end, and
  `session/scratch/r3f` and `r3g` the scripts of groups F and G.

Decided by the user on 2026-11-21, by pick list: the 99 slow files run on Triton (array job
21740810 at `ceedcaf6`, logs `calculations/audit-1120/slow-21740810_<task>.out`, task n is
line n+1 of `slow-files.txt`); the two rule commits stay; `notes/audit-1120/` stays.

**The slow array ran and the user stopped it** ("Kill the slow array", then "Cancel it too"
for the control). 83 of the 99 files passed; `tests/regression/test_realtime_regimes.py` did not
finish (cancelled after 1 h 21 min); 15 files have failures, 18 tests, listed with what each
read in P183, "The slow files were then run". A control at master `e22a1c5d` (job 21742448)
was cancelled after 1 min 28 s: only `test_elastic_one_trace.py` (3 tests) and
`test_elk_allelectron.py` (1 test) ended, and they fail at master too. **The other 14 tests,
in 13 files, are not assigned to the branch or to master, and nothing was read against the
code.** No job of this work is left in the queue.

What waits, in this order: the 14 slow failures (a control at master for the 13 files needs
the user's word, since the user stopped both arrays; reading the code is free); the open
choices of P183, one pick list at a time; the merge and the push; the removal of the
leftovers on Triton, which now include the worktree `apps/defuqumat-audit1120-base`.

# The checkpoint of the night of 2026-11-20 (history)

This note was for the session that continued the work. It says what was done on the branch,
what was left, how the fix agents were run, and what waited for the user.

## State

- Branch `audit-fixes-1120`. Not merged into master and not pushed. The user decides both.
- Three reading audits of 2026-11-20, each with a fix log at its end:
  - `notes/AUDIT-2026-11-20.md` (round one, 49 findings): all done.
  - `notes/AUDIT-2026-11-20b.md` (round two, 34 findings): all done.
  - `notes/AUDIT-2026-11-20c.md` (round three, 87 findings, the user chose to fix #1 to #29
    only): groups A to E done. Groups F and G are left.
- `notes/audit-1120/decisions.md` is the list of "Decisions taken without the user". The user
  said "Take the next decisions by yourself" on the afternoon of 2026-11-20. The rule since then
  is to pick the conservative option (a refusal or a deferral over a new form) and never to move
  a number `PLAN.md` records without a remeasured row. Every such decision goes in that list
  with its reason, and the list goes into the P183 record for the user to review.
- `notes/audit-1120/session/` is untracked on purpose, because it holds local paths. It has the
  three briefs (`brief.md`, `brief2.md`, `brief3.md`) and `agent-reports.md`, the verbatim final
  report of every fix agent and every audit workflow of the session. The proposed record text of
  round one lives only there (search it for "Proposed" and "scope.md"). Its `scratch/` folder
  is a copy of the session scratchpad's scripts and notes (2 MB, no arrays): the scripts behind
  the numbers measured outside a test, for example the hcp Al phonon (round two #12), the PAW
  wedge against its nosym twin (round three #2), the finite difference of #20 and the timing
  pairs of round two #8.
- **The gate has not been run on this branch at any point.** Every fix ran its own fast files
  one at a time. `tools/test-fast.sh` runs once at the end, before anything is merged.

## What is left

### Round three, group F and group G

Run them one at a time, with one Opus agent each (never Sonnet, pass `model: opus`), in the
background, in the main tree.

- Group F, magnetism, tests and docs: #13 (with_turning_moments refused with "needs nosym" at
  every turned orientation, although the guard offers it as the cure), #14 (the I119
  single-precision refusal bypassed by relax_orientation, frozen_expectation and
  spiral_spin_orbit_energy), #15 (the torque wedge refusal tested on its helper only), #18 (the
  spirals guide says a spiral PDOS works under nosym, but every spiral projection is refused).
- Group G, ultracell, sizing and checkpoint: #17 (every ultracell state box-transformed in the
  density, tau and LDOS passes, zero-weight states included; an optimization), #23 (k_batch='fit'
  leaves the device store out of the estimate), #24 (a constructor-level checkpoint_dir crosses
  into other runs).

The brief is `session/brief3.md`. Copy it to the new session's scratchpad and replace the
scratchpad path in it (the `old3` worktree) with the new one. The old code of each agent is the
branch head at its start. The prompts of groups A to E had the same shape: the brief path, the
head, the finding list with one line each, the context from earlier rounds on the same code, and
these rules: refusal by name over new physics; a fix that moves a recorded number becomes
`notes/audit-1120/round3-<N>-<short>.patch`; no edits to `PLAN.md`, `PERFORMANCE.md`,
`.claude/rules/`, `.claude/skills/` (proposed text instead); no slow tests, one capped pytest at
a time; copy the new test into old3 to see it fail there and print `defuqumat.__file__`;
trap-check before each commit; no merge, no push; simple English. After each agent, log its
choices in `decisions.md`.

### The end of the work

**Status of steps 1 and 2 (2026-11-21, the record agent, branch `audit-1120-record` from
`6b7bb998`; this branch is merged into `audit-fixes-1120` by the lead).** Each item below is
marked DONE, DONE WITH A CHANGE, AS A PATCH or LEFT.

- Step 1, P183: DRAFTED, status IN PROGRESS. Left in it: the paragraph "TO FILL: round three,
  groups F and G" (their numbers, refusals, slow files, gate time and decisions are in no list
  of P183 yet), the gate line, and the status. One more correction is carried beside the one
  below: `tests/regression/test_gauge_multiplets.py` has no mark at all, so it is in the gate
  too (round three #20 lists it as slow).
- Step 2, `PLAN.md`: DONE as dated notes in P4 item (3), the line on `expand_to_subgroup`,
  P77c, P148, P179, P181 and P182. DONE WITH A CHANGE: P122 (the proposed sentence was for
  the case that round two #3 lands; it is held, so the note says the history is dropped under
  the default fit); P179's V3 (0.0 stays, the note says the slow test was not run); the round
  three paragraphs (each is in P183, #8 also in P61, and one pointer each in P8, P16, P58, P65,
  P71 and P162; the paragraph of #6 is merged with its siblings, which landed). Added beyond
  this list: a dated note in P62e, whose numbers belong to the contraction before round one #8
  and are not remeasured.
- Step 2, `.claude/rules/scope.md`, `.claude/rules/reference-source-map.md` and `CLAUDE.md`:
  AS A PATCH, `notes/audit-1120/record-rules-and-claude-md.patch` (applies clean with
  `git apply`). An agent may not change these files on the word of another agent, so the user
  applies it. Two proposals are changed in it: the spiral DFT+U text (the "DFT+U" of that list
  means symmetry beside DFT+U, which is still refused, so a sentence is added and the word is
  kept), and the Elk refusals (`tefield` is refused in any template, not only a
  spin-polarized one; `sxcscf` is named with its flag `tssxc`; `msmgmt` under a GGA and the
  group order are added from the report of that fix).
- Step 2, `notes/incompat/I28.md`: DONE. `PERFORMANCE.md`: DONE, and DONE WITH A CHANGE for
  round two #8 ("card times" is "CPU times at the default chunk", since a card takes the whole
  box); the per-face hold pair, proposed in a report only, is added too. `OPEN.md`: DONE (Part
  XXXII, five entries, and Part XXI item 1). The `Calculator.with_kpoints` docstring: DONE.
- Step 2, `notebooks/40_stm_images`: LEFT, its export needs a run.
- Steps 3 to 6: LEFT.

1. Write the `PLAN.md` phase P183 from the three fix logs and `decisions.md`, with the slow files
   to run and these corrections: `tests/regression/test_input_sweep.py` has no slow mark and is
   in the gate (round two #7 and #24 call it slow).
2. Apply the proposed text, which the agents could not edit:
   - `PLAN.md`: P179 (the WKB wording, V3 reads about 1e-16 where 0.0 is recorded); P181 (the
     spin rotation rule, "External fields are not refused", the core join lines near 30910 to
     30934); P182 (the hold is one mean per face, the fcc Ar 3.6e-6 Ry of #46, polarized lmbj
     admitted); P77c (round two #32); the line near 2216 (expand_to_subgroup except 'bands');
     P122 (equivariance under rho_ddot); P148 ("Corrected 2026-11-20", round two #20, and the
     open question whether P148's autodiff numbers went through `ScfResidual.jvp`); P4 item (3)
     (QE's budget is maxter x 6 = 120 outside an SCF, round two #9); every round-three entry's
     own proposed paragraph.
   - `.claude/rules/scope.md`: write_* refuse under pools; the spiral DFT+U bullet (DFT+U
     refused only on an ultrasoft or PAW spiral); the P181 refusals (tefield, sxcscf, nosource
     on a spin-polarized run; a DFT+U template); the lmbj hold wording (one mean per face); the
     Gamma-only storage paragraph; the spiral refusal list gains the structure factors.
   - `notes/incompat/I28.md`: dE/dq, the spiral scan with gradients and relax_spiral_q carry the
     Hubbard energy.
   - `PERFORMANCE.md`: round two #8 (the balanced plane chunk, the pairs and the 13 inputs whose
     card timings are now a few per cent too slow); round two #9 near line 3386 (QE's Davidson
     budget as it is: 120 steps outside an SCF, and inside one pw.x moves on once five or fewer
     bands are unconverged); round three #16 (the overlap metric pair); the P181 pair is
     already there.
   - `.claude/rules/reference-source-map.md` line 32 (not for `calculation = 'bands'`).
   - `defuqumat/calculator.py`: one sentence in the `Calculator.with_kpoints` docstring (a
     tetrahedron run refuses the whole grid on a run with symmetry, as pw.x does).
   - `notebooks/40_stm_images`: the constant-current paragraph says `heights=(lo, hi)` is needed.
   - `OPEN.md`: round one #13 (the exact-trace wedge, about a day); the held patches below; the
     hcp Al wedge against nosym SCF energy, 6.8e-7 Ry apart on a 2x2x2 grid, not investigated;
     starting_ns_eigenvalue applied again by every fresh run_scf (pw.x applies it once), the
     final SCF of a vc-relax included; `run_ultracell_stm` passing `heights` in bohr to
     `_refuse_an_augmented_plane` as crystal coordinates; Part XXI item 1 gains tau (round three
     #25).
   - `CLAUDE.md`: the audit row points to the newest audits (2026-11-20, -20b, -20c).
   - P183: the P178 Se2 spin-DOS test cell is diamond (it has inversion), so it cannot see wedge
     weights; zincblende SeI would.
3. Run `tools/test-fast.sh` once (the gate). Slow tests only when the user asks.
4. Give the user the slow files to run. Each fix-log entry names its own; collect the union.
5. Hand the user the open choices below, one pick list at a time.
6. Decide whether `notes/audit-1120/` stays in the repository (its tracked part) or moves into
   the P183 record.

## Held patches (in this folder, not applied)

Checked at 9750e507 with `git apply --check`: every patch applies clean except the superseded
local-tail one, which needs `--3way`.

- `round2-3-rotate-fit-vectors.patch`: turn the rho_ddot fit vectors under rotate_moments. Right
  by construction, no saving shown, would move I78, I112, I113.
- `round2-9-scf-stragglers.patch`: pw.x's allowance of five unsettled bands inside an SCF.
- `round2-11-kubler-cut-channels.patch`: the kubler GGA on the cut channels of the PAW sphere.
  Reaches several slow PAW spinor files.
- `round2-21-symmetrize-ddd.patch`: QE's group average of ddd_paw (3.2e-7 of 0.22 on AlAs).
- `round3-2-paw-wedge-becsum.patch` and `round3-2-raman-wedge-experiment.py`: the value split of
  becsum for the PAW third derivatives on a wedge; the complete fix also needs a full-zone tangent.
- `round3-29-mixed-newton-refusal.patch`: refuse band_precision='mixed' with Newton-Krylov.
- `local-tail-9-10-on-900af157.patch`: superseded by the per-face hold the user picked; a record.

## Open choices for the user

- Round one: write the external field into Elk's B_s (#21, moves the P181 O2 start); write
  dftu=0 instead of refusing (#1); the Elk symmetrizer caches (#23, about 3 h); 'bands' with
  K_POINTS automatic reduces with the crystal group, pw.x with the lattice group; whether the
  28 s fast part of `tests/unit/test_spiral_hubbard_gradient.py` stays in the gate;
  get_stress(terms=True) in memory mode (needs a jacfwd over the chunked walk).
- Round two: the full external force (#5); the Thomas-Fermi tau seed at nspin_mag = 4 (#14,
  four SCFs on nickel decide it); #17 and #27; land #3, #9, #11 or #21 from the patches; #29
  (vkb at one k-point: store at nk = 1, store under a byte budget, or keep); #28 (the xy pass by
  rows, at most 1 to 2 per cent); #30 (the second becsum jvp, 2.2 per cent of a PAW Si phonon);
  #31 (the magnon kernel built once, 12 and 2.7 per cent at three q); #4 (the fixed GGA axis
  follows pw.x's compute_ux literally, so two moments of 0.01 count as parallel at any angle;
  keep, or a rule that does not depend on the lengths).
- Round three: the complete PAW wedge fix of #2; whether to keep #20 (PAW deformation
  potentials move by up to 4.59e-3 of 1.13, toward a finite difference; no recorded number; a
  revert of 48955549 would hold it); #28 (the Raman ort blocks on the host, 7.37 GB and 64 s on
  the P25 yardstick); #29 (leave, refuse, or Newton in double, which needs a GPU timing); a
  default for the constant-current STM range at the vacuum midpoint (#9, about 2 h); the spiral
  structure factors with the charge kept or the satellites H -/+ q (#8, about half a day); the
  Kramers-pair check of the Wannier centres in the Z2 (#4).

## Gate time added by the fixes

Round one: spiral fast tests 28 s, #37 14 s, lmbj 8 s and 5.5 s, #22 7.8 s, stress 2.5 s. Round
two E: about 24 s. Round two F: not given by its agent (the new files are
`tests/unit/test_one_shot_solves_report.py` and `tests/unit/test_plane_chunk.py`, plus one test
in `tests/unit/test_subspace_eigh_tangent.py`). Round three: A 33.4 s, B 9.2 s, C about 23 s, D about 21 s, E about 2 s. Each
was measured as its own process, so the gate grows by less.
