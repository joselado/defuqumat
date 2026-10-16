# The rest of tier T2 of the incompatibility work, integration one

This file is the record of how the units of `notes/INCOMPATIBILITIES.md` in the rest of tier T2
(I33 to I36 and I49 to I85, with I83 left out because its prerequisite I5 is partial) are brought
onto the branch `incompat-t2`, which starts from master at `126d6ec` (the merge of I1 to I48,
`notes/INCOMPATIBILITIES-NIGHT.md`). Integration one takes the first sixteen units in rank order;
I70 to I82, I84 and I85 were still running when it started and are left for integration two, so
they are not failures. The rules are the night's: merge in rank order with `--no-ff` and a commit
after each, record files only from a partial, blocked or refuted unit, a mixed unit merged only
when its unconfirmed items still refuse, one fast gate on D22 of the head, a merge that fails the
gate reverted. Slow tests are not run (the user's rule of 2026-10-16); they are listed for the user.

## The sixteen units, as reported, and what integration one did

| unit | branch | commit | reported | integration |
|---|---|---|---|---|
| ultracell2 (I33 to I36) | incompat/ultracell2-fix | da3c996 | I33, I34, I35 confirmed; I36 partial | pending |
| I49 | incompat/I49 | 91b1b9b | done, confirmed | pending |
| I50 | worktree-wf_93b36bbe-189-57 | 9ec1c09 | done, confirmed | pending |
| I51 | incompat/I51-merged | 704b157 | done, confirmed | pending |
| I52 | incompat/I52-r2 | 87f1691 | partial, nothing in the confirmed list | pending |
| I53 | incompat/I53 | bc0b81c | done, confirmed | pending |
| I54 | incompat/I54 | f1d8760 | done, confirmed | pending |
| I55 | incompat/I55 | 6e55d88 | done, confirmed | pending |
| magnon (I58 to I62) | incompat/magnon | 05303c1 | I58, I61 confirmed; I59, I60 partial; I62 blocked | pending |
| I63 | incompat/I63 | e401a6e | done, confirmed | pending |
| I64 | incompat/I64 | 8a6b6e9 | done, confirmed | pending |
| I65 | incompat/I65 | 29b3d82 | done, confirmed | pending |
| I66 | incompat/I66-r2 | 9d58999 | done, but refuted in the confirmed list | pending |
| I67 | incompat/I67-fix | 7e130a5 | done, confirmed | pending |
| I68 | incompat/I68 | 967c9c8 | done, confirmed | pending |
| I69 | incompat/I69 | 6bb262e | done, confirmed | pending |

## Progress

- Step 0: branch made from `126d6ec`, this file written. Every branch tip matched the reported sha.
