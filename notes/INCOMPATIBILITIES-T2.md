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
