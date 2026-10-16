# The overnight incompatibility work of 2026-10-16, integration one

This file is the record of how the units of `notes/INCOMPATIBILITIES.md` that were worked on
overnight are brought onto the branch `incompat-night`, which starts from master at `862235d`.
Only I1 to I48 were attempted this night, and I49 to I141 are left for another session, so
they are not failures. Integration one covers the first 23 units in rank order, and on the
user's request to save tokens this session integrates only the first third of them, I1 to I8;
the other fifteen units are left for the next integration session, with their reported outcome
below so that session can start from this file.

## The units of integration one, as reported

| unit | branch | commit | reported | this session |
|---|---|---|---|---|
| I1 | incompat/I1 | 5b31cac | done, confirmed | to merge |
| I2 | incompat/I2 | b6c74ac | done, confirmed | to merge |
| I3 | incompat/I3 | 800c880 | partial | record only |
| I4 | incompat/I4 | 5f8408d | done, confirmed | to merge |
| I5 | incompat/I5 | db9848e | partial | record only |
| I6 | incompat/I6 | 67c051f | done, confirmed | to merge |
| I7 | incompat/I7 | c216761 | done, confirmed | to merge |
| I8 | incompat/I8 | 96f270f | done, confirmed | to merge |
| I9 | incompat/I9 | 30b2574 | done, confirmed | next session |
| strain (I10, I11) | incompat/strain | 853731e | partial | next session |
| I12 | incompat/I12 | 129e30b | done, confirmed | next session |
| I13 | incompat/I13 | 41e6e90 | done, confirmed | next session |
| finitefield (I14, I15, I56, I57) | incompat/finitefield | efa53c9 | I14, I15 confirmed; I56, I57 done, not verified | next session |
| I16 | incompat/I16 | 746853c | blocked, confirmed as blocked | next session |
| I17 | incompat/I17 | cbece28 | done, confirmed | next session |
| I18 | incompat/I18 | 92f079f | done, confirmed | next session |
| I19 | incompat/I19-v2 | 132267d | done, confirmed | next session |
| spiral (I20 to I30) | incompat/spiral-fix | 0b24bd2 | I20, I21, I24, I27, I28 confirmed; I22, I23, I25, I26, I29 partial; I30 blocked | next session |
| ultracell (I31 to I36) | incompat/ultracell-i32fix | cbbad0d | I32 confirmed after its fix; I31 partial; I33 to I36 not addressed | next session |
| I37 | incompat/I37 | 6fdd9a2 | done, confirmed | next session |
| I38 | incompat/I38 | e9a02fe | done, confirmed | next session |
| I39 | incompat/I39 | ef35116 | done, confirmed | next session |
| wedge (I40 to I48) | incompat/wedge-r2 | cf684fb | I40, I41 confirmed; I42, I46 done, not reverified; I47, I48 partial; I43 to I45 blocked | next session |

## Progress

- Step 0: branch made, this file written.
- Step 1: merged I1, I2, I4, I6, I7, I8 with `--no-ff`, one commit each; record files only from I3 and I5. Every branch tip matched the reported sha. I8 conflicted with I7 in `defuqumat/tddft/chi0.py`, `defuqumat/workflows/tddft.py` and `tests/unit/test_tddft_machinery.py`, neighbouring arms of one guard; resolved by keeping both lifts, dropping the sibling test of each branch that asserted the other's refusal, and taking I7's `max(occupied_counts(...))`. `tests/unit/test_tddft_machinery.py` passes locally after the merge (20 passed).
- Step 2: the guide Limits notes of the six (`bands.md`, `phonons.md` twice plus the linewidth paragraph, `nonlinear.md`, `optics.md` with I7 and I8 written as one sentence) and the README phonon bullet; the six entries moved into "Closed since" of `notes/INCOMPATIBILITIES.md` with their numbers, I3 and I5 marked partial in place; `PLAN.md` P154. Of the stale texts, none is touched by I1 to I8: `.claude/rules/scope.md`'s real-time bullet is I39's, `notes/ELK-FEATURES.md` §9 the ultracell's, `response/piezo.py:374` the strain family's, and `tddft/spinkernel.py:51-67` is the magnon kernel's, not the `chi_0` of I7 and I8, so all four are left for the session that lands those units.
