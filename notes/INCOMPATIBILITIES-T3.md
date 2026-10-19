# Tier T3 of the incompatibility work, integration one

This file is the record of how the units of `notes/INCOMPATIBILITIES.md` in tier T3 (I86 to
I122) are brought onto the branch `incompat-t3`, which starts from master at `63804fc5`
(`6c5b3a36`, the merge of T2, plus the two commits of the T3 workflow files). The
implement/verify workflow stopped early at a usage limit, so this integration covers what had
finished. The rules are T2's: merge in item order with `--no-ff` and a commit after each,
record files only from a partial, refuted or unverified unit, one fast gate on D22 of the head,
a merge that fails the gate reverted. Slow tests are not run (the user's rule of 2026-10-16);
they are listed for the user. The outcome of each unit is taken from the integration input as
given (`t3-integ-input.json`, whose verifier findings are cut at about 1500 characters each); the
full verifier reports are in the D22 log directories each one names, `/l/ladovj1/review/incompat/<ID>-ver*/`.

## The units, as reported, and what this integration does

| unit | branch | commit | reported | integration |
|---|---|---|---|---|
| I86 | incompat/I86 | 90d92f55 | done, confirmed | merge |
| I87 | incompat/I87 | 9d858042 | done, confirmed (two record sentences to fix) | merge |
| I88 | incompat/I88 | 6967d36b | done, refuted on scope | record only |
| I89 | incompat/I89 | 3ad688ca | done, confirmed | merge |
| I90 | incompat/I90 | 0082ea99 | done, refuted (wedge regime) | record only |
| I91 | incompat/I91 | 3a36ecfb | partial, guard holds | record only |
| I92 | incompat/I92 | 3aae8e8e | done, confirmed | merge |
| I93 | incompat/I93 | f8891a9d | done, refuted (crossing filter) | record only |
| chi0 (I94 to I96) | incompat/chi0 | c2a83c97 | done, verifier did not finish | record only |
| I97 | incompat/I97 | c189b039 | done, confirmed | merge |
| I98 | incompat/I98 | 549c2b3a | done, confirmed (one record sentence false) | merge |
| I99 | incompat/I99 | 48f06b81 | done, refuted (not reachable from `Calculator`) | record only |
| I100 | incompat/I100 | 59f27d1f | done, confirmed | merge |
| I101 | incompat/I101 | fff605f3 | done, refuted (T=800 breaks the tolerance) | record only |
| I102 | incompat/I102 | 39f91dbb | done, confirmed | merge |
| I103 | incompat/I103 | ee7325ae | done, confirmed | merge |
| I104 | incompat/I104 | 24003058 | partial, refuted on its precision claims | record only |
| I105 | incompat/I105 | d1acce97 | partial, guard holds | record only |
| I106 | incompat/I106 | 65f9f183 | partial, confirmed as partial | record only |
| sourcefree (I107 to I109) | incompat/sourcefree | c5401217 | I108, I109 done, I107 partial; verifier did not run | record only |
| I110 | incompat/I110 | 8efa0ec4 | done, confirmed | merge |
| I114 | incompat/I114 | e80bc66f | partial; verifier did not run | record only |
| I111, rotate (I112, I113), I115, ultraseed (I116, I117), I118 to I122 | none | none | not started (session limit) | nothing |

## Progress

- Step 0: branch made from master `63804fc5`, this file written. Every branch tip matched the
  reported sha, and every branch sits on `6c5b3a36`.
- Step 1: merged, with `--no-ff` and one commit each, I86, I87, I89, I92, I97, I98, I100, I102,
  I103 and I110. Two merges conflicted, both plainly, and both sides were kept. I100 against I98
  in `realtime/propagate.py` (two docstring lines) and `workflows/realtime.py` (the update dict
  takes I100's `ns` under I98's condition for the one-centre `D`); I102 against I98 in
  `realtime/hierarchy.py` (I102's order check and I98's `route="hierarchy"` on the same guard).
  After each overlap the fast files of both sides were run here: `test_overlap_metal_phonon.py`,
  `test_elastic_augmented.py` and `test_sibling_refusals.py` after I87 (49 passed),
  `test_realtime.py` and `test_realtime_hubbard.py` after I100 (23 passed, 11 deselected), and
  those two with `test_hierarchy_two_grids.py` after I102 (32 passed, 11 deselected). Record files
  only, each path under `notes/incompat/` taken by name from its branch, from I88, I90, I91, I93,
  chi0 (I94 to I96), I99, I101, I104, I105, I106, sourcefree (I107 to I109) and I114.
- Step 2: Limits notes on seven guide pages (`phonons.md` I86, I87; `stress.md` I87; `piezo.md`
  I87, I89; `nonlinear.md` I92, I98, I100, I102; `optics.md` I97; `electric-field.md` I103;
  `magnetism.md` I110, a sentence in the torque section and one in the Limits note), README's
  `lelfield` cell (I103) and phonon bullet (I86), and `.claude/rules/scope.md`'s real-time
  clause (I98, I100). Two record sentences were corrected on the way, from the verifiers: I87's
  chunked route walks only in memory mode (`memory_mode='memory'` with `k_batch < nk` on a CPU,
  any chunk on a GPU), not for any `k_batch` on a CPU; I97's 2e-3 agreement is stated for the
  silicon test it was measured on. `tools/check_guide.py` passes, and
  `tests/unit/test_facade_documentation.py` passes locally (6 passed). Ten ledger entries moved
  into "Closed since", the twelve unlanded ones (eighteen IDs) marked in place; `PLAN.md` P156,
  and its outstanding index for I86, I87, I98, I100 and I102.
- Step 3: the fast gate launched on D22 at `8819131e` (the code head; steps 2 and on touch only
  `docs/`, `README.md`, `notes/`, `PLAN.md` and `.claude/rules/`), worktree `/l/ladovj1/w-inc-t3`,
  logs in `/l/ladovj1/review/incompat/t3/`, `taskset -c 0-3,12-15`, 16 GB scope,
  `DEFUQUMAT_QE_ROOT` set.
