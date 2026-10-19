# The rest of tier T2 of the incompatibility work, integrations one and two

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
- Step 3, result: the gate at `f69cdfb` read 4 failed, 3848 passed, 71 skipped, 1715 deselected in 1125.45 s (20:36 to 20:55). Two failures are `tests/unit/test_check_guide.py`'s missing `yaml`, the D22 venv's. The other two are `tests/unit/test_one_calculation_per_workflow.py::test_the_refusals_read_the_calculation_the_run_diagonalises_in[spin-susceptibility]` and `::test_the_calculator_hands_the_spectra_its_own_calculation[spin-susceptibility]`, `IndexError: tuple index out of range` at `np.shape(density)[0]`: my own texture check, called with `density=None` by tests that reach a refusal. No merge is at fault, so none is reverted; the check now needs an array (`1dd9b6d`), and the two files pass locally (33 passed). Re-gate on a fresh worktree `/l/ladovj1/w-inc-t2-c` at `1dd9b6d`, logs in `/l/ladovj1/review/incompat/t2c/`. It read 2 failed, 3850 passed, 71 skipped, 1715 deselected in 1009.43 s (20:57 to 21:14), both failures `test_check_guide.py`'s missing `yaml`. Nothing attributable to a merge, and no merge is reverted. The commits after `1dd9b6d` touch only `notes/` and `PLAN.md`.

## Outcome per ID

- I33 landed: the TB09 ultracell against the TB09 supercell at matched grids, 8.144e-3 at 48 bands and falling; `tau` frozen at the tiled one rises 1.246e-1 to 1.636e-1.
- I34 landed: `nosource` ultracell, charge 7.13e-5, magnetization 1.47e-4 to 1.74e-4 at 16 bands; without the projection at `Q != 0`, `m_x` 2.5e-1.
- I35 landed: saw-tooth ultracell with `dipfield`, field energy per cell 3.585e-6 Ry off the supercell at 48 bands, against 1.118e-4 for a frozen saw-tooth (the record's E_f assertion is what separates them).
- I36 partial (merged): a fixed `B_field` reference converges to the supercell under the same field, free energy +8.838e-7 Ry at 24 bands; feedback schemes, `reducebf < 1`, penalties and a spinor under a field refuse, and whether the ultracell should run a feedback scheme on the box is the user's design question.
- I49 landed: shifted-grid wedge against the whole shifted grid 1.398e-8 (frozen), 1.717e-8 (`hxc`); all eight operations 0.823.
- I50 landed: the spin-space-group wedge against `nosym`, force 5.6e-8, stress 2.1e-10, energies 7e-14 to 1.6e-12 Ry; `dE/dq` refused with symmetry after the verifier's fix round.
- I51 landed: closed subset against the whole symmetrised block 1.5e-11 to 3.7e-10 (nc, US, PAW, fluorite, `nspin = 2`), the inverse table 2.73e-1.
- I52 partial, record only: pooled bands, NSCF and DOS against 1 pool 1.3e-14 Ry and 1.9e-12 of the DOS peak, verified by its verifier, but the batch's confirmed list was empty, so not merged.
- I53 landed: Newton-Krylov under 2 and 3 pools against 1 pool, 1.8e-14 and 1.1e-14 Ry; tetrahedra refused on every rank before any collective.
- I54 landed: walked subset bit for bit against the same response's block; the record's 5.5e-3 for the old route on US AlAs corrected to `IndexError`; the walked route with symmetry refused at integration.
- I55 landed: two-grid hierarchy against the dense two-grid solve 1.9e-11 to 5.1e-11 (Hartree), 9.5e-12 to 2.5e-11 (`hxc`).
- I58 landed: Goldstone residual 0.00585 at the Larmor frequency `2B`, 0.171 at 0.
- I59 partial (merged): a collinear spinor along z matches `nspin = 2` to 1.1e-5 in `lambda_max`; the texture now refuses on the bare `X_0` too (added at integration, the verifier's gap).
- I60 partial: the PAW one-centre kernel sized, not written; refusal kept.
- I61 landed: Goldstone residual 0.00473 with the Hubbard channel, 0.197 without.
- I62 blocked: needs I59's texture response and a tensor kernel; its guard fires for its own reason now.
- I63 landed: shifted kick against the run started at the kick, 1.1e-12 to 2.2e-11 on US and PAW.
- I64 landed: 4.31e-10 electrons out of the occupied subspace under the converged field, 8.93e-3 under the input field.
- I65 landed: fsm resume, field -0.079999913 on both routes, 1.4e-17 apart; 26 against 26 iterations on `fe-fsm.in`.
- I66 refuted, record only: the batch's list has it refuted on the first fix's digest, which was blind to `k_batch`. The third verifier round at the fix commit `592240e` (`/l/ladovj1/review/incompat/I66-ver3/hA.log`, `hM.log`, `hOld.log`) reads a resume bit for bit, a file of another `k_batch` ignored (`resumed=0`, `max|dJ| = 0`) and the old digest crashing in the scan, which reads as a confirmation; the user decides whether that round counts.
- I67 landed: spiral to plain at `q = b3`, 1 iteration against 11, E 4.9e-10 Ry; spiral into `nspin = 2` refused, `tau` dropped.
- I68 landed: seed moved between pool layouts, 1.39e-13 Ry in 1 iteration against 8.
- I69 landed: `MAG3D.OUT` pointwise 1.5e-11; seeded against unseeded dE 4.3e-14 Ry.
- Counts: 17 landed (I33, I34, I35, I49, I50, I51, I53, I54, I55, I58, I61, I63, I64, I65, I67, I68, I69); 4 partial (I36 and I59 merged with their kept refusals, I52 record only, I60 record with the magnon merge); 1 blocked (I62); 1 refuted (I66); none skipped.

## Slow test files for the user to run

None of these was run here. The merges add or change:

- `tests/regression/test_ultracell_regimes.py`: `test_a_tb09_ultracell_converges_to_the_tb09_supercell`, `test_a_source_free_ultracell_converges_to_the_supercell`, `test_a_saw_tooth_ultracell_bounds_the_supercell_from_above` (both parameters), `test_a_field_converged_reference_converges_to_the_supercell_under_it`.
- `tests/regression/test_realtime.py` (I49's new test, both parameters, about 9 minutes; and the rest, since I63 refactored `_block_function` and I64 the workflow), `tests/regression/test_realtime_regimes.py` (I63), `tests/regression/test_realtime_hierarchy.py` (I55, and the tests whose `_dense_cell` gained `ecutrho`), the slow tests of `tests/unit/test_realtime.py` and `tests/unit/test_realtime_field.py` (I63, I64).
- `tests/regression/test_spiral_symmetry.py` (I50, new), `tests/regression/test_spin_spirals.py` (I50, and I67's new test), `tests/regression/test_spiral_soc.py`, `tests/regression/test_spiral_relaxation.py` (I50 touched the reader and driver for every spiral).
- `tests/regression/test_phonons.py` (I51's four new tests and the two subset regressions), `tests/regression/test_streamed_phonons.py` (I54), `tests/regression/test_streamed_third.py` (I54's walked Raman tests).
- `tests/unit/test_parallel.py`, its slow tests (I53's two, I68's new one and the four pooled continuation and checkpoint tests), and `tests/regression/test_scf_solvers.py` (I53 split `_solve_residual`).
- `tests/regression/test_magnon_regimes.py` (new: I58, I59, I61), `tests/regression/test_magnons.py`, `tests/regression/test_ultrasoft_magnon.py`.
- `tests/unit/test_scf_restart.py` and `tests/unit/test_checkpoint.py`, their slow tests (I65's new `test_an_interrupted_fixed_spin_moment_run_costs_the_same_iterations`).
- `tests/regression/test_elk_seed.py` (I69, new test and a changed message).

## Follow-up deliverables

Only the Limits notes, two README sentences, three `scope.md` clauses and P72's Elk-spin sentences are written. Every landed item still needs its guide section with an executed snippet where the regime is worth showing, a notebook cell, and a timing where there is reference code (I69 against Elk's own seed run; the others are identities with no reference code for the regime). `tests/unit/test_parallel.py::_start_pools`'s stdout-to-file fix (a 76 kB JSON blocking a rank on a full pipe) is in I52's branch, which is not merged, so the pooled process tests keep the old pipe. The I54 verifier noted that reverting `driver.py` alone leaves I68's gate test green, since it calls `Pools.redistribute` directly; only the slow test covers the call in `run_scf`. The `ultrasoft subset solve` 3.7e-8 against the whole-cell block, found by I54 and outside it, is not in `OPEN.md` yet, and nor is the fluorite bug the I51 verifier found next to I51 (its findings name it a separate pre-existing item).

## What the user decides

- I52: merge `incompat/I52-r2` (87f1691), the pooled `get_bands`, `get_nscf` and `get_dos`, verified on D22 by its verifier, or keep it for a further round.
- I66: count the third verifier round at `592240e` as the confirmation and merge `incompat/I66-r2` (9d58999), or send it to another round.
- I36: whether the ultracell should run a feedback scheme (`fsm`, `reducebf`) on the box itself.

## D22 notes from the reports, for `docs/d22-notes.local.md`

- The worktree names the recipe prescribes are often taken by an earlier round of the same unit (ultracell2, I49, I50, I51, I53, I54, I55, I63, I64, I65, I66, I67, I68, magnon all hit it). Run `ls -d /l/ladovj1/w-inc-<ID>*` and `git worktree list` first, take a fresh suffix, and do not chain `git worktree add` with `&&`, since a failed add stops the chain or, behind `| tail`, hides itself. Read an earlier round's logs in `review/incompat/<ID>*` before relaunching.
- `git checkout <rev> -- <files>` stages the old content, so restoring needs `git checkout HEAD -- .` (ultracell2, magnon, I51). Several verifier worktrees are left with uncommitted reverts: `w-inc-I64-ver-rvf`, `-rvu`, `-rvv`, `w-inc-I65-ver2-rev`, `-ver2-drv`, `w-inc-I69-ver2-rev`.
- The six-slot queue was full several times on 2026-10-16 between 18:50 and 20:05, with waits of 3 to 16 minutes; a log holding only `# waiting for a slot` is not stalled, and a script can be replaced while it waits. `pgrep -af slot.sh | wc -l` is the quick reading (29 waiting at 19:27).
- A harness that changes the ultracell modulation axis must patch `tests/regression/test_ultracell_regimes.py::_modulation` before building the supercell reference, or the supercell runs another potential and the ladder reads -5.6e-2 Ry gaps that look like a broken method (ultracell2, `v35d-y.log` against `v35d-y2.log`).
- A slow test body can be measured without `pytest -m slow` by importing the test module with `importlib` and calling the function in a harness through `slot.sh` (magnon, `review/incompat/magnon/slowbodies.py`); `from tests.unit.test_realtime import _silicon` works because `slot.sh` puts the worktree on `PYTHONPATH` (I66).
- A pooled failure reading `Shutdown barrier has failed ... 1/2` with exit -6 can be a rank blocked on a full stdout pipe (a 76 kB JSON, I52), or a Python exception raised inside a collective, whose traceback is only in another rank's stderr (I53); read every rank's stderr first.
- `KPoints.band_path` takes one count per vertex, including the last; a list one short raises "a band path needs one count per vertex", which reads like a pools failure (I50, I52).
- `tests/unit/test_scf_restart.py` ignores `DEFUQUMAT_QE_ROOT` and skips 6 tests unless `<worktree>/quantum_espresso/qe-7.5-ReleasePack/qe-7.5` is a symlink to `/l/ladovj1/qe-7.5/qe-7.5` (gitignored) (I67, I68). A run naming `tests/regression/test_elk_seed.py` under `-m 'not slow'` deselects the whole file (module-level slow mark), which reads like a pass (I69).
- `tail -5 a b` over ssh fails with "option used in invalid context"; use `tail -n 5` (I51, I63, I65, I69, magnon). The local worktree guard refuses any Bash text containing `git` in a compound command or a heredoc, `sed` on a shell variable, and an `scp` with a glob; write the script with the Write tool, scp it by name, run `ssh host 'bash script'` (I53, I69, magnon). A local run from an agent worktree needs the worktree first on the path, or the editable install imports the main checkout (I49).
- An ssh line that sleeps 120 s or more trips the 120 s tool timeout; keep remote sleeps near 100 s or raise the timeout (I67). One ssh call with an `until grep -q "# exit"` loop and a 600 s tool timeout waits for several slot jobs at once (I66); a harness running each phase with `subprocess.run([sys.executable, ...])` inside one slot inherits its taskset, scope and `PYTHONPATH` (I66).
- With `dt = 0.2`, `graphene-monolayer.in` at `ecutwfc = 25` is refused by the taylor4 stability guard ("use dt <= 0.1293"); a cell with vacuum needs `dt = 0.1` (I49). On `h-fcc-magnon.in` (degauss 0.01, 4x4x4) the moment is pinned at 34/64, so a small uniform field only shifts the bands rigidly and a field check there tests only the sign (magnon). `Calculator.get_dielectric_tensor` refuses a noncollinear spinor with `nspin_mag = 4` even when its magnetization has decayed (I55). JAX clamps an out-of-bounds static index instead of raising, so a channel-count mismatch can pass silently (I67).
- Costs on one slot: TB09, nosource and saw-tooth ultracell harnesses 15 to 120 s; a kept-response harness with `raman_tensors(keep_internals=True)` and six dynamical matrices fits 6 GB, 258 s on 64-k Si, 445 s on 8-k US AlAs; ultrasoft Ni magnon cell SCF 15 s, 20 s per `transverse_response`, 110 s with a U; `fe-fsm.in` to convergence 9 to 12 s per three SCFs; spinor Si2 real-time harness 223 s; a linear SiO2 molecule in a 9-bohr box does not converge in 100 iterations, CO2 with `C.pz-rrkjus` and `O.pz-rrkjus` does (I51); pooled spiral bands, NSCF and DOS 75 s at 2 pools on `h-chain-spiral.in` (I52). The fast gate itself ran in 1125 s at 20:36 beside a loaded slot queue.
- The D22 refs `incompat-<ID>` can lag the branch tip (I64 at e760cfe, magnon at 738b2438, I54 at fe14ec8); compare `git rev-parse` before trusting them.

## D22 worktrees that can be removed (none removed)

Of the merged units, once `incompat-t2` is accepted: `w-inc-ultracell2`, `-v2`, `-v4`, `-ver`, `-ver-rev`, `-ver2`, `-ver2-rev`; `w-inc-I49`, `-t2`, `-ver`, `-ver-rev`, `-ver2`, `-ver2-rev`, `-verify`, `-verify-rev`; `w-inc-I50`, `-t2`, `-t3`, `-v2`, `-ver`, `-ver-ax`, `-ver-nf`, `-ver-rev`, `-ver-rz`, `-ver3`, `-ver3-rev`; `w-inc-I51`, `-t2`, `-v2`, `-ver`, `-ver-rev`; `w-inc-I53`, `-t2`, `-t2-v2`, `-ver`, `-ver-rev`, `-ver3`; `w-inc-I54`, `-v2`, `-ver`, `-ver-rev`; `w-inc-I55`, `-ver`, `-ver-rev`; `w-inc-magnon`, `-v2` to `-v5`, `-ver`; `w-inc-I63`, `-base`, `-ver`, `-ver-rev`, `-ver2`; `w-inc-I64`, `-v2`, `-ver`, `-ver2`, `-ver-rvf`, `-ver-rvu`, `-ver-rvv`; `w-inc-I65`, `-v2` to `-v5`, `-v4-rev`, `-ver`, `-ver-base`, `-ver-drv`, `-ver-nofix`, `-ver-rev`, `-ver2`, `-ver2-drv`, `-ver2-rev`, `-ver3`, `-ver3-drv`, `-ver3-rev`; `w-inc-I67`, `-v2` to `-v4`, `-ver`, `-ver-rev`, `-ver3`, `-ver3-old`; `w-inc-I68`, `-base`, `-ver`, `-ver-rev`, `-ver2`; `w-inc-I69`, `-v2`, `-v3`, `-ver`, `-ver-rev`, `-ver-mut1` to `-ver-mut3`, `-ver2`, `-ver2-rev`. Those of I52 (`w-inc-I52`, `-t2`, `-v2`, `-ver`, `-ver-revfull`, `-ver-revnscf`) and I66 (`w-inc-I66`, `-rev`, `-v2`, `-ver`, `-ver3`, `-ver3-old`) stay until the user decides. `w-inc-t2` (the first gate, at `f69cdfb`) and `w-inc-t2-c` (the re-gate) are this integration's own.

## Integrator's changes that no second session has seen, and stale refs

Three code changes on this branch are mine and are in no `notes/incompat/<ID>.md`, so no verifier ran them: the walked-route refusal of a closed subset with symmetry in `response/phonon.py` (where I51 meets I54), the texture check in `workflows/magnons.py:run_spin_susceptibility` (I59's gap), and its `np.ndim` repair after the first gate. Each has a fast test that fails with the change patched out, and the re-gate covers them.

Three `incompat/*` refs lag what was merged, because each is checked out in another agent worktree that this one may not touch: `incompat/I50` at 3e584d6 (merged from `worktree-wf_93b36bbe-189-57` 9ec1c09), `incompat/I51` at 72a0d2e (merged from `incompat/I51-merged` 704b157), `incompat/I67` at 87ff5bb (merged from `incompat/I67-fix` 7e130a5). Once those worktrees are released, `git branch -f` brings each up, or the merge of `incompat-t2` makes them moot.

# Integration two

Integration two takes the units that were still running at integration one: I70 to I82, I84 and
I85. The rules are the same. It works on the branch in the agent worktree
`.claude/worktrees/wf_93b36bbe-189-84` (branch `worktree-wf_93b36bbe-189-84`, fast-forwarded to
`incompat-t2` at `1f8957e`), since `incompat-t2` itself is checked out in the worktree
`wf_93b36bbe-189-82`, which this agent may not touch; `incompat-t2` is brought up to this branch's
head once that worktree is released (the head is a descendant, so it is a fast-forward).

## The fifteen units, as reported, and what integration two does

| unit | branch | commit | reported | integration |
|---|---|---|---|---|
| I70 | incompat/I70 | 0ce3e5e | done, confirmed | to merge |
| I71 | incompat/I71 | d58e46d | done, confirmed | to merge |
| I72 | incompat/I72 | 50cb57a | done, confirmed (fix round: the fixed-point identity and the one-iteration discriminator) | to merge |
| I73 | incompat/I73 | 4fd7d6b | done, confirmed (fix round: the loop-iteration discriminator) | to merge |
| I74 | incompat/I74-fix | 693c186 | done, confirmed (fix round: the check at `conv_thr = 1e-13`) | to merge |
| I75 | incompat/I75 | 74b9dea | done, confirmed | to merge |
| I76 | incompat/I76 | e627366 | done, confirmed | to merge |
| I77 | incompat/I77 | 91e35a2 | partial, nothing in the confirmed list | record only |
| I78 | incompat/I78-fix | 084598e | done, confirmed (fix round) | to merge |
| I79 | incompat/I79 | b6ff6d0 | partial, nothing in the confirmed list | record only |
| I80 | incompat/I80 | 58fc138 | done, confirmed | to merge |
| I81 | incompat/I81 | 5db26ea | partial, nothing in the confirmed list | record only |
| I82 | incompat/I82-fix | 89e0787 | done, confirmed (fix round: the GGA refused) | to merge |
| I84 | none | none | skipped, prerequisite not confirmed | nothing to take |
| I85 | incompat/I85 | 106978a | done, confirmed | to merge |

Every branch tip matched the reported sha.

## Progress of integration two

- Step 0: this section written.
- Step 1, I70: merged (`03b383a`). `io/elk.py`, `calculator.py` and `tests/unit/test_elk_reader.py` conflicted with I69 (integration one). Both readers are kept, the magnetic records before the DFT+U record as `writestate.f90` writes them, and a spin-polarized DFT+U state, which neither unit measured, is refused by name (a new parameter of the refusal test). `tests/unit/test_elk_reader.py` passes locally (41 passed).
- Step 1, I71: merged (`9868cd5`). `scf/residual.py` conflicted with I53's refusal of tetrahedra and `from_input` inside `_weights` and `make_residual`. That refusal is now the pooled one alone (the driver's pool guard, on every rank), since one process takes both and the pooled pair is unmeasured; the message reads "has no pooled form" and I53's fast and slow pool tests match it.
- Step 1, I72: merged (`b59ff1b`). The docstring conflicted with I71's; I53's pool test asserted `make_residual`'s old spiral refusal as a sibling and now asserts the Hubbard one that stays.
- Step 1, I73: merged (`51db087`). `make_residual` conflicted with I72: I73's field branch and I72's spiral branches are both kept, I73's spiral refusal is dropped, and a spiral under a field or a constraint, which neither unit measured, is refused ahead of both (this keeps I72's `test_a_spiral_under_a_field_is_still_refused` firing). I53's pool test now reads the feedback refusal instead of the old blanket field refusal. The touched fast files pass locally (41 passed, `test_parallel.py`, `test_residual_spiral.py`, `test_residual_fields.py`, `test_scf_solvers.py`, `test_magnetic_constraints.py`).
- Step 1, I74: merged clean (`3a51eaa`); its two fast files pass locally (14 passed).
- Step 1, I75: merged clean (`8d9a321`), but its `test_newton_krylov_still_refuses_tetrahedra` asserted the residual's refusal of tetrahedra, which I71 lifted, so it would have run a Newton solve on aluminium. The three together (Newton, tetrahedra and the tetrahedron LDOS as the warm-up) were measured by neither unit and are refused by name before the warm-up (`4a8d2c0`, `scf/driver.py`); the test reads the new refusal, and `tests/unit/test_ldos_mixing.py` passes locally (23 passed).
- Step 1, I76: merged clean (`c0b2478`); `test_robust_mixing.py`, `test_parallel.py` and `test_ldos_mixing.py` pass locally (64 passed).
- Step 1, I77: record only (`notes/incompat/I77.md`).
- Step 1, I78: merged clean (`64397c8`); `test_rotating_moments.py` and `test_anisotropy.py` pass locally (26 passed). No test covers a pooled `rotate_moments` run on the turning group (the record's own note).
- Step 1, I79: record only (`notes/incompat/I79.md`).
- Step 1, I80: merged clean (`04ea04f`); `test_phonon_q_nlcc.py` passes locally (11 passed).
- Step 1, I81: record only (`notes/incompat/I81.md`).
- Step 1, I82: merged (`bb90e6b`). `response/phononq.py` conflicted with I80; both lifts are kept and their pair, a core charge at `nspin = 2`, is refused by name as measured by neither. The two stand-in files are retargeted (I82's core sibling is the `nspin = 2` case, with a passing test at one channel; I80's stand-in carries the flags I82's guard reads). `test_phonon_q_nlcc.py`, `test_two_sphere_refusals.py`, `test_metallic_q_refusals.py` and `test_sibling_refusals.py` pass locally (78 passed).
- Step 1, I85: merged clean (`4df74c1`); `tests/unit/test_berryfield.py` passes locally (25 passed).
- Step 1 in all: eleven merges (I70, I71, I72, I73, I74, I75, I76, I78, I80, I82, I85), record files from I77, I79 and I81, nothing from I84 (no branch). Four integration guards, each with a fast test that reads it: a spin-polarized Elk DFT+U state, a spiral under a field in `make_residual`, the tetrahedron LDOS warm-up of a Newton solve, and a core charge at `nspin = 2` at `q != 0`; and one refusal narrowed, I53's occupation refusal, now pooled only.
- Step 3 launched before step 2: the fast gate on D22 at the code head `d07df07` (worktree `/l/ladovj1/w-inc-t2b`, logs in `/l/ladovj1/review/incompat/t2b/`, 21:29), fetched there as `incompat-t2` from the bundle `126d6ec..worktree-wf_93b36bbe-189-84`. Step 2's commits touch only `docs/`, `README.md`, `notes/`, `PLAN.md` and `.claude/rules/`.
- Step 2: Limits notes on seven guide pages (`elk-seed.md` I70, `ground-state.md` I71, I73, I75, I76, `spirals.md` I72, I74, `anisotropy.md` I78, `phonons.md` I80, I82, `electric-field.md` I85), each naming the refusal the integration added where two lifts meet; README's `rotate_moments` bullet (I78); `.claude/rules/scope.md`'s Elk-seed clause (I70). `tools/check_guide.py` passes and `tests/unit/test_facade_documentation.py` passes locally (6 passed). Eleven ledger entries moved into "Closed since" with their numbers, I77, I79, I81 marked partial in place and I84 skipped; `PLAN.md` P155 extended with the integration-two table.

- Step 3, result: The fast gate at `d07df07` (worktree `/l/ladovj1/w-inc-t2b`, `taskset -c 0-3,12-15`, 16 GB scope, logs in `/l/ladovj1/review/incompat/t2b/`, 21:29 to 21:46) read 2 failed, 3961 passed, 71 skipped, 1750 deselected in 1007.48 s. Both failures are `tests/unit/test_check_guide.py`'s missing `yaml`, the D22 venv's, as at integration one's re-gate (2 failed, 3850 passed, 71 skipped, 1715 deselected); `tools/check_guide.py` itself passes locally on the final head. The 71 skips are the same set (the worktree has no `quantum_espresso/` symlink, so `test_scf_restart.py` skips six, `test_units.py` one, and the rest are QE inputs this code refuses), with none in the new Elk or Berry-field files. Nothing is attributable to a merge, and no merge is reverted. The commits after `d07df07` touch only `docs/`, `README.md`, `notes/`, `PLAN.md` and `.claude/rules/`.

## Outcome per ID, integration two

- I70 landed: seeded against unseeded DFT+U SCF of sc hydrogen at `U = 4` eV, dE -6.7e-16 Ry, `max|drho|` 7.0e-9, `max|dns|` 5.3e-9; `vmatmt` against `VMATMT.OUT` to every printed digit. A spin-polarized DFT+U state refuses (added at integration).
- I71 landed: Newton-Krylov with tetrahedra 2.6e-9 Ry from `pw.x` on `al-tetrahedra.in`, with `from_input` 8.8e-10 on `scf-occ.in`. The verifier puts the record's unexplained 1 to 7 per cent autodiff residue in the eigensolver, not in I71. Under pools the two occupations refuse (I53's refusal, narrowed).
- I72 landed: `F(x*) - x*` at the mixing loop's converged state 1.32e-10 (NC), 2.70e-10 (US), 1.35e-11 (PAW), one loop iteration after the hand-off. A spiral with a U or a meta-GGA refuses, and a spiral under a field refuses (added at integration). Newton on a spiral was measured on `nosym` cells only; I50's spin-space-group wedge (integration one) is in the same code path through `Calculation.density`, so `F` should still be the loop's map, but Newton on a symmetrised spiral has not been run.
- I73 landed: Newton under a penalty against `pw.x` 7.8e-8, 2.7e-9 and 1.3e-8 Ry, under a collinear uniform field against mixing 1.7e-8 Ry; the loop-count discriminator (1 against 11 to 41 with the field dropped from `F`) is the assertion. The last commit after the verifier, `4fd7d6b`, changes the wording of the autodiff guard only.
- I74 landed: on `o-chain-spiral-us.in` at `conv_thr = 1e-13`, E 1.1e-13 Ry from `'r'` and `|m|` 1.2e-7, where the cutting layout does not converge and stops 3.5e-9 Ry off.
- I75 landed: Bloechl and Kawamura tetrahedra under `'ldos'` to `pw.x` (2e-10 and 0 at the printed digits), 19 against 55 iterations on `al10-metal-tetra`; the LDOS sum equals the tetrahedron DOS to 1e-10. The tetrahedron LDOS as a Newton warm-up refuses (added at integration, where I71 meets I75).
- I76 landed: robust mode at 2 pools against 1 and resumed against uninterrupted, the same decisions and counts, dE at most 8.9e-15 Ry.
- I77 partial, record only: `'ldos'` in the ultracell against Kerker 1.19e-7 and 3.47e-7 in the density, 3.4e-13 and 2.1e-12 Ry, the box LDOS 27.528 against 27.542; `'robust'` not written. The new slow test has never run under pytest; its body, called as a function at 91e35a2, passed both cases (verifier).
- I78 landed: `{E, I}` against `nosym` 5.7e-14 Ry (10 against 18 points) and -2.8e-14 on the shifted grid. Both cells are null tests of the choice between `{E, I}` and the starting group (their minima lie in the starting group's plane), which the record says. No test covers a pooled run on the turning group.
- I79 partial, record only: the magnetic spinor phonon against `nspin = 2`, 8.3e-6 Ry/bohr^2 on O2 and 1.9e-6 on the chain, of which about 2e-6 belongs to the spinor response route and is unlocated. The axial displacement average and the `becsum` axial average at `nspin_mag = 4`, which the verifier confirmed, are on `incompat/I79` and not merged.
- I80 landed: `D(b1/2)` against the doubled cell at Gamma 2.79e-6 (ONCV) and 4.81e-6 (broad core) at a floor of 4.40e-6, against 1.26e-1 and 7.8e-2 with no core terms. A core charge at `nspin = 2` refuses (added at integration, where I80 meets I82).
- I81 partial, record only: the refusal stays, and the record shows the entry undercounts (P39's orthogonality set at `q`, and a frozen Hessian that is not atom-diagonal for an augmented dataset). The verifier notes the branch's new tests cannot tell the old refusal message from the new one.
- I82 landed: the H3 chain at `b3/2` under the LDA against the doubled cell, 1.522e-5 Ry/bohr^2 and 0.051 cm^-1; a GGA refuses at every `nspin` (it was silently wrong by 60 to 77 cm^-1).
- I84 skipped: its prerequisite I79 is partial; no branch.
- I85 landed: the spiral at `b3/4` against the 90-degree supercell, `E(E) - E(0)` 8.3e-12 Ry and `el_pol` 8.9e-8; the supercell against a new `pw.x` run 2.2e-9 and 1.0e-9 Ry. The magnetized-spinor half is a guard lift with no code change, supported by the `pw.x` reference.
- Counts, integration two: 11 landed (I70, I71, I72, I73, I74, I75, I76, I78, I80, I82, I85); 3 partial, record only (I77, I79, I81); none blocked; none refuted (I72, I73, I74, I78 and I82 were each refuted by a first verifier and confirmed after a fix round, and the confirmed commit is the one merged); 1 skipped (I84).

## Integrator's changes in integration two that no second session has seen

Four refusals where two lifts meet unmeasured, and one narrowing, each with a fast test that reads it:

- `io/elk.py`: a spin-polarized DFT+U Elk state (I69 with I70), `tests/unit/test_elk_reader.py::test_a_state_carrying_physics_this_cannot_transfer_is_refused` (its `kwargs2` case).
- `scf/residual.py:make_residual`: a spiral under a field or a constraint (I72 with I73), `tests/unit/test_residual_spiral.py::test_a_spiral_under_a_field_is_still_refused` (its match "magnetic field" reads the new message).
- `scf/driver.py:_solve_residual`: the tetrahedron LDOS as a Newton warm-up (I71 with I75), `tests/unit/test_ldos_mixing.py::test_newton_krylov_still_refuses_tetrahedra` (retargeted).
- `response/phononq.py:require_a_two_sphere_regime`: a core charge at `nspin = 2` at `q != 0` (I80 with I82), `tests/unit/test_two_sphere_refusals.py::test_each_sibling_refusal_still_fires[lsda-nlcc]` and `tests/unit/test_phonon_q_nlcc.py::test_the_sibling_refusals_still_fire_with_a_core_charge`.
- `scf/residual.py:refuse_residual_occupations`: I53's refusal of tetrahedra and `from_input` is now called only by the driver's pool guard, with the message "has no pooled form"; `tests/unit/test_parallel.py::test_a_pooled_residual_refuses_tetrahedra_before_any_collective`.

Four sibling-refusal assertions of earlier units were retargeted because a later unit lifted what they asserted: I53's `make_residual` spiral and field siblings in `test_parallel.py` (now the Hubbard and the feedback refusals), I80's stand-in in `test_phonon_q_nlcc.py` (now carrying the flags I82's guard reads), I82's core sibling in `test_two_sphere_refusals.py` (now at `nspin = 2`, with a passing test at one channel).

## Slow test files for the user to run, integration two

None of these was run here. The merges add or change:

- `tests/regression/test_elk_seed.py` (I70's new `test_a_dftu_seed_and_an_atomic_dftu_run_converge_to_the_same_state`, and the rest, since the reader changed; the module is marked slow).
- `tests/regression/test_scf_solvers.py` (I71's `test_newton_krylov_with_tetrahedra_is_pw_x_s_fixed_point`, `test_newton_krylov_with_from_input_is_pw_x_s_fixed_point`, `test_autodiff_jacobian_with_tetrahedra_matches_finite_differences`, `test_weights_agree_with_the_driver`; I73's `test_newton_krylov_finds_the_mixer_s_fixed_point_under_a_field`; and the rest, since `make_residual` changed), `tests/regression/test_tetrahedra.py` and `tests/regression/test_scf.py` (the mixing loop now goes through `Calculation.tetrahedron_table()`).
- `tests/unit/test_residual_spiral.py::test_newton_krylov_finds_the_mixer_s_fixed_point_on_a_spiral` (I72, three cases).
- `tests/regression/test_magnetic_constraints.py::test_newton_krylov_reproduces_pw_x_under_a_constrained_moment` (I73, three cases, about 70, 50 and 370 s on one D22 core).
- `tests/unit/test_spiral_mixing_space.py::test_a_spiral_mixed_in_g_reaches_the_real_space_fixed_point` (I74, both parameters).
- `tests/unit/test_ldos_mixing.py::test_optimised_tetrahedra_under_ldos_reach_pw_x_on_ten_sites` and `::test_the_ldos_warm_up_reaches_the_mixer_s_fixed_point` (I75).
- `tests/unit/test_parallel.py::test_two_pools_take_the_robust_modes_decisions_of_one` and `::test_a_robust_checkpoint_resumes_where_it_stopped` (I76, four cases each), and `::test_a_pooled_newton_run_refuses_tetrahedra_on_every_rank`, whose message the integration changed to "has no pooled form" and which has never run with it; `::test_pools_reproduce_one_when_the_moments_are_turned` (I78's code path).
- `tests/regression/test_anisotropy.py::test_turning_the_moments_on_the_turning_group_is_the_nosym_run` and `::test_turning_the_moments_inside_the_scf_converges_on_the_easy_axis` (I78).
- `tests/regression/test_phonons_at_q_nlcc.py` (I80, new, five tests), `tests/regression/test_phonons_at_q.py` (I82's `test_a_magnetic_insulator_at_half_b3_is_the_doubled_cell_at_gamma`, about 5 minutes, and the rest), `tests/regression/test_streamed_phonons.py`, `tests/regression/test_electron_phonon.py` (I80 and I82 changed the `q` route).
- `tests/regression/test_berryfield.py` (I85's `test_a_spin_spiral_is_its_commensurate_supercell` and `test_a_magnetized_spinor_is_pw_x`, about 30 s together, and the rest, since `_gathered` and `BerryFieldSetup.build` changed for every `lelfield` run, 816 s before).

## Follow-up deliverables, integration two

Only the Limits notes, one README sentence and one `scope.md` clause are written. Every landed item still needs its guide section with an executed snippet where the regime is worth showing, a notebook cell, and a timing where there is reference code (I71, I73 and I75 against `pw.x` on the same input; I70 against Elk's own seed run; I80 and I82 against `ph.x` at `q`, which has not been run for either regime; the rest are identities with no reference code for the regime). I82's record asks for two `OPEN.md` entries, not written here: the phonon at `q != 0` under a GGA was silently wrong before I82 (1.33e-2 Ry/bohr^2 and 77 cm^-1 on the H2 chain at `nspin = 1`), the missing term being the `iq` of `dgradcorr.f90`'s gradients; and the PBE fold at Gamma on the same H2 chain, `|D(0) - (S_aa + S_ab)| = 4.8e-4`, 125 times the LDA's 3.8e-6, with no GGA phonon at Gamma checked against `ph.x` anywhere (an open item nothing refuses). The I80 record notes that no `ph.x` number at `q != 0` with a core charge exists here.

## What the user decides, integration two

- I77: merge `incompat/I77` (91e35a2), whose `'ldos'` half is verified and only `'robust'` is unwritten (the same shape as I52), or keep it for a round that writes `'robust'`.
- I79: take the two confirmed sub-pieces of `incompat/I79` (b6ff6d0), the axial displacement average and the axial `becsum` average at `nspin_mag = 4`, with the phonon refusal kept, or leave the branch whole until the 2e-6 is located.
- I81: whether to pin the new refusal message by a test (the verifier's suggestion), since the branch's tests cannot tell it from the old one; the message change itself is not merged.
- `incompat-t2` itself: see the branch note below.

## D22 notes from the reports of integration two, for `docs/d22-notes.local.md`

- The verifier worktree names were taken by an earlier round again (I70, I72, I73, I74, I78, I82): check `ls -d /l/ladovj1/w-inc-<ID>*` and pick a fresh suffix before suggesting a `-ver` name; check `ls` in `review/incompat/<ID>/` before an `scp` into it, since a shared run directory lost I73's `run4.sh`.
- `git checkout <commit> -- <files>` stages the old content, so `git checkout -- .` leaves the revert in place (status shows `M ` in the index); restore with `git checkout HEAD -- .`, or `git reset -q HEAD -- <files>` then `git checkout -- <files>` (I70, I72, I73, I77, I78, I79, I81).
- The D22 refs `incompat-<ID>` lagged the branch tip again (I71 at d505c94, I75 at c4e0e2bf, I76 at 2076a99, I78 at 7d35376, I81 at 152615f1, I82 at 46a859d); compare `git rev-parse` with the local tip before testing.
- The worktree-isolated local agent sandbox refuses: an `scp` whose source is a glob (name each file), any compound command mixing `cd`, heredocs, `git bundle`, `scp` and `ssh`, a `sed`/`scp` with a shell variable or command output in the arguments, and any ssh line or heredoc containing `git`. Write the script with the Write tool, `scp` it by literal path, run `ssh host 'bash /path/script.sh'` (I71, I72, I73, I74, I77, I78, I81). QE test-suite names contain `git` (`benchmark.out.git.inp=...`), so reach them with a glob (I73). Foreground `sleep` is blocked locally; an `until ssh ... grep -q '^# exit' log; do sleep 30; done` loop works (I77, I82).
- `tail -5 a b` over ssh fails with "option used in invalid context"; write `tail -n 5` (I72, I80).
- The six-slot queue (`slot.sh`, CPUs 4, 6, 8, 10, 5, 7) was full from about 19:24 to 20:10 on 2026-10-16, with about 25 to 30 verifier jobs and waits of 3 to 18 minutes; a log holding only `# waiting for a slot` is queued, not hung, and a job that never got a slot can be killed by pid and relaunched. A verifier answering within one session should submit its harness first (I73, I74, I76, I77, I79, I82, I85). A glob such as `tail *.log` right after launching can find nothing and exit 1 before `slot.sh` writes the logs (I73). A pytest path that does not exist ends at once with `# exit 4 after 1 s` and "file or directory not found", which reads like a fast pass if only the exit line is checked (I82). `slot.sh` passes environment variables from the launching line (`CONV=1e-13 nohup bash slot.sh ...`) (I85).
- `tests/unit/test_scf_restart.py`'s `qe_silicon` fixture ignores `DEFUQUMAT_QE_ROOT` and skips six tests unless `<wt>/quantum_espresso/qe-7.5-ReleasePack/qe-7.5` links to `/l/ladovj1/qe-7.5/qe-7.5` (I76, as I67 and I68 found). D22's venv has no `yaml`, so `tests/unit/test_check_guide.py` fails twice in every gate there (I75).
- Two k-point pools inside one slot (`test_parallel._run_pools` with `DEFUQUMAT_THREADS=1`) take 25 to 50 s per 2-atom Si or Al SCF on a 2x2x2 grid against about 6 s for one pool. When one rank crashes, `_run_pools` prints only the tail of the first nonzero rank's stderr, the coordination-service "Shutdown barrier has failed ... 1/2" dump; the Python traceback is above the cut or in the other rank's stderr (`review/incompat/I76/debug.py` prints them all). When pools enter different collectives the run dies after about 100 s with "CoordinationServiceError ... check the task logs for an earlier error", not at the timeout (I76). In a patch script, the line `accuracy = pools.broadcast_scalar(accuracy)` at 12 spaces also matches a deeper one at `driver.py:8954`; match it with the comment above (I76).
- A collection `ImportError` in one test file (a revert under a test importing a new class) gives "Interrupted: 1 error during collection" and stops every file on the same pytest line (I74). A harness that imports another harness must set `sys.argv` first, since the imported script runs its main block (I79); a harness can import a test module's helpers by adding `<worktree>/tests/unit` to `sys.path` (I79).
- `SCFResult.iterations` after a Newton-Krylov solve counts the mixing loop after the hand-off: 1 when Newton found the loop's fixed point, more when it did not, since the loop silently re-converges; this is the cheap discriminator for a residual-solver test (I72, I73). A Newton run on a spiral with a soft mode can run 6 to 45 minutes before it raises, silent until both solvers finish; Newton on a spiral under `tb09` with `'kubler'` was still running at 42 CPU minutes where the plain cell takes 7 s (I72).
- Matching a pw input with `"spiral_q" not in text` fails on `h-chain-spiral.in`, whose header comment contains the word; match lines starting with the keyword (I72). `h3-chain-lsda-fixed.in` has "nspin = 2" in its header comment, so a substitution needs `"\n    nspin = 2\n"` (I82). `Calculation` has no `.kpoints`; read `system.kpoints.weights` (I82). In `SCFResult.berry_dipoles` the tuple is `(el_pol, ion_pol, acc)`; `[2]` is `acc`, 0.0, which reads as a perfect identity (I85).
- To get a norm-conserving spiral with a separate smooth grid from `h-chain-spiral.in`, `ecutwfc` 24 to 25 with `ecutrho` 6 or 8 times it gives a smooth box 40x40x15 that `displaced_gvectors` refuses ("the smooth FFT box (40, 40, 15) does not hold the sphere ... spans (39, 39, 16)") within 1 to 2 s, which looks like a bad input; `ecutwfc = 20, ecutrho = 160` or `22, 132` works, 2 to 15 s a run (I74).
- `Calculator.with_turning_moments()` seeded from a different k-set warns "the converged wavefunctions are not being carried over ... 8 k-points ... target has 10" and starts from atomic orbitals; expected (I78).
- Serial `pw.x` 7.5 (`/l/ladovj1/qe-7.5/bin/pw.x`) runs `lelfield` with `noncolin`, `starting_magnetization`, `angle1`, `angle2` and `occupations = 'fixed'` with no `tot_magnetization`, and restarts into `lelfield` from a plain run with `startingwfc = 'file'`, `startingpot = 'file'` on the same manual k-list. I85 ran its `pw.x` reference outside `slot.sh`, under `systemd-run --user --scope -p MemoryMax=4G` with `OMP_NUM_THREADS=1`, about a minute, which breaks the recipe's queue rule.
- The scratchpad of a workflow run is shared between sibling agents, so `scp ver/*.py` copied about 25 unrelated scripts into `/l/ladovj1/review/incompat/I80-ver/` (`consumers.py`, `_fbreak.py`, `harness_d22.py`, `harness.py`, `harness_ver.py`, `newton.py`, `patch.py`, `pdos.py`, `restore.py`, `round2.py`, `smear.py`, `v33.py`, `v34.py`, `v35a.py`, `v35b.py`, `v35d.py`, `ver_field.py`, `ver_hub.py`, `ver_ni0.py`, `ver_ni.py`, `ver_revert.py`, `ver_texture.py`, `vgrid.py`, `vharness.py`, `weights.py`); they do nothing there and may be deleted. Use a unit-named local directory and copy by name (I80).
- Costs on one slot: the I70 harness (eight pairs of hydrogen SCFs) about 20 s; Al 10 k-point harnesses 11 to 41 s (I71); one `o-chain-spiral-us` SCF 12 to 25 s, 50 s when it stalls to 100 iterations (I74); a cobalt `rotate_moments` arm (3x3x2, US, SOC, `conv_thr = 1e-12`) 20 to 45 s, 4x4x2 50 to 70 s (I78); the I80 `b1/2` identity about 100 s, `b1/3` 210 s, smeared silicon 170 s; the fixed-occupation H-chain `lelfield` pair 66 s (I85); the O2 ultrasoft spinor Gamma phonon about 2 minutes, the AFM H chain about 1 minute, while the 4-hydrogen textured metal (`h4-cycloid-90-nosym`, 24 bands, 4 k) runs past 41 minutes at 4.7 GB RSS, near the 6 GB cap, printing nothing (I79, pid 2408805 started 19:24, `review/incompat/I79/helix.log` if it finishes).

## D22 worktrees that can be removed after integration two (none removed)

Of the merged units, once `incompat-t2` is accepted: `w-inc-I70`, `-v2`, `-ver`, `-ver2`, `-ver2r`; `w-inc-I71`, `-v2`, `-ver`, `-ver-rev` (dirty on purpose); `w-inc-I72`, `-rev`, `-v2`, `-v3`, `-v4`, `-ver`, `-ver2`, `-ver2r`; `w-inc-I73`, `-v2` to `-v6`, `-ver`, `-ver-rev`, `-ver2`, `-ver2-rev` (dirty on purpose); `w-inc-I74`, `-rev`, `-v2`, `-v3`, `-ver`, `-ver2`; `w-inc-I75`, `-v2`, `-ver`, `-ver-rev`; `w-inc-I76`, `-v2`, `-v3`, `-ver`, `-ver-old`, `-ver-rev`; `w-inc-I78`, `-base`, `-v2`, `-v3`, `-ver`, `-ver-rb`, `-ver-rd`, `-ver2`, `-ver2-rev`; `w-inc-I80`, `-v2`, `-v3`, `-ver`, `-ver-rev`; `w-inc-I82`, `-v2`, `-v3`, `-v4`, `-ver`, `-ver-rev` (dirty), `-ver2`; `w-inc-I85`, `-v2`, `-ver`. Those of the partial units (`w-inc-I77`, `-rev`, `-v2`, `-ver`; `w-inc-I79`, `-v2`, `-ver`, `-ver-rev`; `w-inc-I81`, `-ver`) stay until the user decides. `w-inc-t2b` is this integration's own gate.

## The branch

`incompat-t2` is checked out in the agent worktree `.claude/worktrees/wf_93b36bbe-189-82`, which this agent may not touch, so the local ref `incompat-t2` still stands at `1f8957e` (integration one). Integration two is on `worktree-wf_93b36bbe-189-84`, a descendant of it, and D22's `incompat-t2` was force-fetched to this branch's code head `d07df07` for the gate. Once the other worktree is released, `git branch -f incompat-t2 worktree-wf_93b36bbe-189-84` (or, from inside the worktree that holds `incompat-t2`, `git merge --ff-only worktree-wf_93b36bbe-189-84`) brings it up as a fast-forward.
