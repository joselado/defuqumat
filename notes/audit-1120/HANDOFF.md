# Audit fixes of 2026-11-20: where we stopped (checkpoint of 2026-11-21)

This note is for the session that continues the work. It says what is done on the branch,
what is left, how the fix agents were run, and what waits for the user.

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
