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
| ultracell2 (I33 to I36) | incompat/ultracell2-fix | da3c996 | I33, I34, I35 confirmed; I36 partial | merged (I36's refusals are fast tests that fire) |
| I49 | incompat/I49 | 91b1b9b | done, confirmed | merged |
| I50 | worktree-wf_93b36bbe-189-57 | 9ec1c09 | done, confirmed | merged |
| I51 | incompat/I51-merged | 704b157 | done, confirmed | merged |
| I52 | incompat/I52-r2 | 87f1691 | partial, nothing in the confirmed list | record only |
| I53 | incompat/I53 | bc0b81c | done, confirmed | merged |
| I54 | incompat/I54 | f1d8760 | done, confirmed | merged, record corrected, walked-route guard added |
| I55 | incompat/I55 | 6e55d88 | done, confirmed | merged |
| magnon (I58 to I62) | incompat/magnon | 05303c1 | I58, I61 confirmed; I59, I60 partial; I62 blocked | merged, with the texture guard on the bare `X_0` added |
| I63 | incompat/I63 | e401a6e | done, confirmed | merged |
| I64 | incompat/I64 | 8a6b6e9 | done, confirmed | merged |
| I65 | incompat/I65 | 29b3d82 | done, confirmed | merged |
| I66 | incompat/I66-r2 | 9d58999 | done, but refuted in the confirmed list | record only |
| I67 | incompat/I67-fix | 7e130a5 | done, confirmed | merged |
| I68 | incompat/I68 | 967c9c8 | done, confirmed | merged |
| I69 | incompat/I69 | 6bb262e | done, confirmed | merged |

## Progress

- Step 0: branch made from `126d6ec`, this file written. Every branch tip matched the reported sha.
- Step 1: merged, with `--no-ff` and one commit each, ultracell2, I49, I50, I51, I53, I54, I55, magnon, I63, I64, I65, I67, I68 and I69; record files only from I52 (partial, empty confirmed list) and I66 (listed refuted). No merge conflicted; the overlaps in `scf/driver.py`, `response/phonon.py`, `realtime/` and `tests/unit/test_parallel.py` were separate hunks. Two guards were added where a lift met something unmeasured. After I54: a walked response with a closed subset under symmetry is refused as the streamed route is (I51 lifted the closed subset on the whole-k route only, and I54's row map was measured on `nosym`), with a fast test; `test_partial_walked_phonons.py` and `test_partial_phonon_refusals.py` pass locally (13 passed). I54's record and its slow test's docstring are corrected (the old whole-k route raised `IndexError` on US AlAs for `atoms=(1,)`, it was not 5.5e-3 off; verifier's `/l/ladovj1/review/incompat/I54-ver/h-rev.log`). After magnon: the I59 verifier's gap (a spinor moment along x reached the bare `X_0` with no refusal, `review/incompat/magnon/ver2/texture.log`) is closed in `run_spin_susceptibility`, with a fast test for both values of `interacting` that fails with the check patched out; `test_magnon_regimes.py` and `test_magnon_machinery.py` pass locally (30 passed). The magnon merge also corrects the stale `tddft/spinkernel.py` text (it now names the U, the field and PAW, not ultrasoft).
- Step 2: Limits notes on eleven guide pages (`ultracell.md` I33 to I36, `nonlinear.md` I49, I55, I63, I64, `spirals.md` I50, `phonons.md` I51, I54, `magnons.md` I58, I59, I61, `dft-plus-u.md` I61, `performance.md` I53, I65, I68, `ground-state.md` I65, `magnetism.md` I67, `elk-seed.md` I69); README's pools paragraph (I53, I68) and spiral bullet (I50); `.claude/rules/scope.md`'s spiral-symmetry, real-time-field and Elk-spin clauses (I50, I64, I69); `PLAN.md` P72's two Elk-spin sentences (I69). `tools/check_guide.py` passes, and `tests/unit/test_facade_documentation.py` passes locally (6 passed). Seventeen ledger entries moved into "Closed since", I36, I52, I59, I60, I62 and I66 marked in place; `PLAN.md` P155.
- Step 3: the fast gate launched on D22 at the code head `f69cdfb` (worktree `/l/ladovj1/w-inc-t2`, 20:36, logs in `/l/ladovj1/review/incompat/t2/`), before step 2's commits, which touch only `docs/`, `README.md`, `notes/`, `PLAN.md` and `.claude/rules/` and were checked locally by the two documentation checks above. The worktree has no `quantum_espresso/` symlink, as at the night's gates, so the skip count compares with their 71.
- Step 3, result: the gate at `f69cdfb` read 4 failed, 3848 passed, 71 skipped, 1715 deselected in 1125.45 s (20:36 to 20:55). Two failures are `tests/unit/test_check_guide.py`'s missing `yaml`, the D22 venv's. The other two are `tests/unit/test_one_calculation_per_workflow.py::test_the_refusals_read_the_calculation_the_run_diagonalises_in[spin-susceptibility]` and `::test_the_calculator_hands_the_spectra_its_own_calculation[spin-susceptibility]`, `IndexError: tuple index out of range` at `np.shape(density)[0]`: my own texture check, called with `density=None` by tests that reach a refusal. No merge is at fault, so none is reverted; the check now needs an array (`1dd9b6d`), and the two files pass locally (33 passed). Re-gate on a fresh worktree `/l/ladovj1/w-inc-t2-c` at `1dd9b6d`, logs in `/l/ladovj1/review/incompat/t2c/`.
