# The overnight incompatibility work of 2026-10-16, integrations one and two

This file is the record of how the units of `notes/INCOMPATIBILITIES.md` that were worked on
overnight are brought onto the branch `incompat-night`, which starts from master at `862235d`.
Only I1 to I48 were attempted this night, and I49 to I141 are left for another session, so
they are not failures. The night's 23 units, I1 to I48 in rank order, were brought in by two
sessions: integration one took I1 to I8, and integration two the other fifteen units, I9 to the
wedge family. Both sessions followed the same rules (merge in rank order with `--no-ff`, a
commit after each, record files only from a partial or blocked unit, a mixed unit merged only
when its unconfirmed items still refuse, one fast gate on D22 of the head, a merge that fails
the gate reverted).

## The 23 units, as reported, and what each integration did

| unit | branch | commit | reported | integration |
|---|---|---|---|---|
| I1 | incompat/I1 | 5b31cac | done, confirmed | one: merged |
| I2 | incompat/I2 | b6c74ac | done, confirmed | one: merged |
| I3 | incompat/I3 | 800c880 | partial | one: record only |
| I4 | incompat/I4 | 5f8408d | done, confirmed | one: merged |
| I5 | incompat/I5 | db9848e | partial | one: record only |
| I6 | incompat/I6 | 67c051f | done, confirmed | one: merged |
| I7 | incompat/I7 | c216761 | done, confirmed | one: merged |
| I8 | incompat/I8 | 96f270f | done, confirmed | one: merged |
| I9 | incompat/I9 | 30b2574 | done, confirmed | two: merged |
| strain (I10, I11) | incompat/strain | 853731e | partial | two: record only |
| I12 | incompat/I12 | 129e30b | done, confirmed | two: merged |
| I13 | incompat/I13 | 41e6e90 | done, confirmed | two: merged |
| finitefield (I14, I15, I56, I57) | incompat/finitefield | efa53c9 | I14, I15 confirmed; I56, I57 done, not verified | two: merged (I56, I57 covered by the verifier's file run) |
| I16 | incompat/I16 | 746853c | blocked, confirmed as blocked | two: record only |
| I17 | incompat/I17 | cbece28 | done, confirmed | two: merged |
| I18 | incompat/I18 | 92f079f | done, confirmed | two: merged |
| I19 | incompat/I19-v2 | 132267d | done, confirmed | two: merged |
| spiral (I20 to I30) | incompat/spiral-fix | 0b24bd2 | I20, I21, I24, I27, I28 confirmed; I22, I23, I25, I26, I29 partial; I30 blocked | two: merged |
| ultracell (I31 to I36) | incompat/ultracell-i32fix | cbbad0d | I32 confirmed after its fix; I31 partial; I33 to I36 not addressed | two: merged |
| I37 | incompat/I37 | 6fdd9a2 | done, confirmed | two: merged, reverted after the gate |
| I38 | incompat/I38 | e9a02fe | done, confirmed | two: merged |
| I39 | incompat/I39 | ef35116 | done, confirmed | two: merged |
| wedge (I40 to I48) | incompat/wedge-r2 | cf684fb | I40, I41 confirmed; I42, I46 done, not reverified; I47, I48 partial; I43 to I45 blocked | two: record only (I42, I46 unverified) |

## Progress

- Step 0: branch made, this file written.
- Step 1: merged I1, I2, I4, I6, I7, I8 with `--no-ff`, one commit each; record files only from I3 and I5. Every branch tip matched the reported sha. I8 conflicted with I7 in `defuqumat/tddft/chi0.py`, `defuqumat/workflows/tddft.py` and `tests/unit/test_tddft_machinery.py`, neighbouring arms of one guard; resolved by keeping both lifts, dropping the sibling test of each branch that asserted the other's refusal, and taking I7's `max(occupied_counts(...))`. `tests/unit/test_tddft_machinery.py` passes locally after the merge (20 passed).
- Step 2: the guide Limits notes of the six (`bands.md`, `phonons.md` twice plus the linewidth paragraph, `nonlinear.md`, `optics.md` with I7 and I8 written as one sentence) and the README phonon bullet; the six entries moved into "Closed since" of `notes/INCOMPATIBILITIES.md` with their numbers, I3 and I5 marked partial in place; `PLAN.md` P154. Of the stale texts, none is touched by I1 to I8: `.claude/rules/scope.md`'s real-time bullet is I39's, `notes/ELK-FEATURES.md` §9 the ultracell's, `response/piezo.py:374` the strain family's, and `tddft/spinkernel.py:51-67` is the magnon kernel's, not the `chi_0` of I7 and I8, so all four are left for the session that lands those units.
- Step 3: the fast gate on D22, worktree `/l/ladovj1/w-inc-night` at `136c768`, cores 0-3 and 12-15, `MemoryMax=16G`, followed by the two slow regression files of the I7 and I8 merge (`test_tddft_spin.py`, `test_spinor_chi0.py`); logs in `/l/ladovj1/review/incompat/night/`. Result: the fast gate read 2 failed, 3707 passed, 71 skipped, 1634 deselected in 1621.67 s (08:08 to 08:35), and both failures are `tests/unit/test_check_guide.py`'s, `ModuleNotFoundError: No module named 'yaml'`, the D22 venv's and not ours; the two slow files read 5 passed in 334.90 s. Nothing attributable to I1 to I8.

### Integration two, the fifteen units I9 to wedge

- Step 0: every branch tip matched the table's sha.
- Step 1, so far: merged I9, I12, I13, finitefield, I17, I18 and I19 (`incompat/I19-v2`) with `--no-ff`; record files only from strain and I16. I12 conflicted with I9 in `workflows/topology.py`, where both had added the same `DFTSource.spin` field: I12's field and checks kept, I9's refusal of a one-channel velocity operator kept beside I12's channel selection, and the channel message worded as `0 or 1 (up or down)` so both branches' tests match. I13 conflicted with both: `_source` takes `spin` and `tau`, and the polarization's kept meta-GGA refusal is applied to the first of I9's per-channel sources. `test_topology_spin_channels.py`, `test_polarization_machinery.py`, `test_sibling_refusals.py` (58 passed) and `test_berryfield.py` (22 passed) pass locally after these merges. Finitefield is merged although the table has I56 and I57 as not verified: the verifier's whole-file run of `tests/regression/test_berryfield.py` at `ffa9faa` (`review/incompat/finitefield-verify/vf-file.log`, 12 passed in 816 s) includes `test_a_streamed_store_is_the_in_memory_run` and `test_an_automatic_grid_is_pw_x_l3dstring_field`, and `ffa9faa` differs from the tip `efa53c9` only in `notes/incompat/I57.md`.
- Step 1, done: then merged spiral (`incompat/spiral-fix`), ultracell (`incompat/ultracell-i32fix`), I37, I38 and I39; record files only from wedge (`incompat/wedge-r2`). Spiral merged cleanly, and its partial and blocked items' refusal tests pass on the merge (`test_realtime.py`, `test_orbital_magnetization.py`, `test_polarization_machinery.py`, `test_stm_machinery.py`, `test_spinor_refusals.py`, `test_sibling_refusals.py`, `test_source_free_field.py`, `test_stress_machinery.py`, `test_topology_spin_channels.py` and `test_ultracell.py::test_every_refusal_fires` for I30, 157 passed). Ultracell merged cleanly, and I31's polarized-GGA refusal and the untouched refusals of I33 to I36 fire (`test_ultracell.py::test_the_refusals_name_themselves`, `::test_every_refusal_fires`, `test_ultracell_regimes.py::test_what_the_box_still_refuses`, 4 passed). I38 conflicted with I18 and spiral in `projwfc/angular_momentum.py` (the states unfolded onto the whole sphere before I18's spin-angle branch; the guard keeps spiral's augmented-spiral refusal and drops the gamma one), in `workflows/stm.py` (the gamma refusal goes, spiral's `charge_image` argument stays), in `tddft/chi0.py` (both imports) and in `tests/unit/test_sibling_refusals.py` (I38's version, with the angular-momentum sibling made the ultrasoft spiral, since a norm-conserving spiral is now projected); the six touched unit files pass (139 passed). I39 conflicted with spiral in `tests/unit/test_realtime.py`, two new tests at one place, both kept (21 passed). Wedge is not merged: I42 and I46 lift their guards with implementer runs only (`review/incompat/wedge/chi0-*.log`, `sts2.log`), no verifier ran them, so the rule takes only the records, and I40 and I41, confirmed, wait with them.
- Step 2: the guide Limits notes of the nineteen items then landed (eighteen after the I37 revert) and of the merged partial ones (I22, I25, I31), on fourteen pages, `tools/check_guide.py` passing; README: the `lelfield` row's "Run with" cell (I57), the Chern row's cell (I12) and the Tran-Blaha bullet (I13). I12's suggested sentence named the polarization among what still refuses `nspin = 2`, which I9 lands, so it is left out. Nineteen entries of `notes/INCOMPATIBILITIES.md` moved into "Closed since" with their numbers, the others of this integration marked in place; `PLAN.md` P154 extended. Of the four stale texts, two are fixed: `.claude/rules/scope.md`'s real-time bullet (I39) and `notes/ELK-FEATURES.md` §9 (the ultracell unit). `response/piezo.py:374` belongs to strain, which is records only, and `tddft/spinkernel.py:51-67` to the magnon family, outside I1 to I48, so both stay stale.
- Step 3: the fast gate on D22, new worktree `/l/ladovj1/w-inc-night2` at `30d5e80`, launched 09:20, followed by the fourteen slow regression files the merged records name, one 16 GB scope each; logs in `/l/ladovj1/review/incompat/night2/`. The fast gate read 3 failed, 3759 passed, 71 skipped, 1677 deselected in 865.35 s (09:20 to 09:35): the two `test_check_guide.py` failures of the missing `yaml`, and `tests/unit/test_host_store_uploads.py::test_the_band_velocities_take_a_host_store_through_device_put`, `DID NOT RAISE _Stop`, which is I37's: the test patches `VelocityOperator.both` to watch the host store reach the tangent, and `incompat/I37` makes `band_velocities` call `_tangent` directly (base and spiral still call `both`), so it fails on I37's own branch too. So the I37 merge `483ea65` is reverted (`git revert -m 1`), its Limits sentence, ledger entry and P154 row undone, its record kept. Its `scf/driver.py` hunk is a comment and a message, so the slow files of the other merges stand as run. The slow loop was stopped by the coordinator at `test_spiral_supercell.py` on the user's new rule (fast tests only); what ran and what did not is under "Follow-up deliverables". The re-gate is the fast gate alone, on a new worktree `w-inc-night3` at `903fb1f`: 2 failed, 3760 passed, 71 skipped, 1677 deselected in 797.20 s (10:01 to 10:15), both failures `test_check_guide.py`'s missing `yaml`, so nothing attributable to a merge remains. `test_host_store_uploads.py` passes again, and passes locally too (5 passed).

## Outcome per ID

- I1 landed: `o-atom-fixed-lsda` HOMO and LUMO -6.633898 eV against `pw.x` -6.6339; `atom-lsda` gap 6.164453 against 6.1680.
- I2 landed: triplet O2 transverse block 3.9e-5 Ry/bohr^2 against `ph.x`'s `.dyn`; PAW stretch 2374.6975 against 2374.699451 cm^-1.
- I3 partial: spin-orbit bismuthene B_1g 69.308519 against `ph.x` 66.727386 cm^-1 at `tr2 = 1e-14`, onsite block only, unlocated; refusal kept.
- I4 landed: q route against Gamma route 7.2e-7 Ry/bohr^2 on an off-centre metal with `def` 1.2e-2; `al2-metal` 2.9e-3 cm^-1 against `ph.x`.
- I5 partial: direct eps_xx 2.056656 against `ph.x` 7.5 2.055320696 (1.3e-3), a miss the BN cell has without a U too (2.1e-3); the U's own change agrees to 7.4e-4; refusal kept.
- I6 landed: unpolarized AlAs `nspin = 2` against `nspin = 1` 6.4e-10 relative; magnet against the spinor route 1.6e-16.
- I7 landed: unpolarized Si `chi_0` 3.0e-9 relative; magnetic H3 chain eps_M(0) 8.65363 against the Sternheimer 8.64013.
- I8 landed: Si spinor against scalar 5.8e-11; spin-orbit Se eps_M(0) -2.65e-3 from the spinor Sternheimer, the scalar control at -2.57e-3.
- I9 landed: triplet O2 channel phases 0.19500066 and 0.42500024 against `pw.x` `lberry`'s 0.19500 and 0.42500, a new `pw.x` 7.5 reference.
- I10, I11 partial (strain): records only, refusals kept.
- I12 landed: nonmagnetic AlAs per channel against `nspin = 1` 9.7e-13 on fluxes of 4.1e-2; O2 channel sources against their SCF eigenvalues to 1.5e-7 Ry.
- I13 landed: fixed-density source against the converged `tb09` eigenvalues 1.1e-7 Ry (NC), 8.9e-7 (spinor), 2.5e-6 (PAW).
- I14 landed: Si8 `lelfield` at `tot_magnetization = 0` against `nspin = 1`, `E(E) - E(0)` 1.1e-10 Ry.
- I15 landed: the same on a nonmagnetic spinor, 2.1e-14 Ry.
- I16 blocked: record only, refusal kept.
- I17 landed: Pt wedge against closed grid 3.7e-7 (US), 5.2e-9 (PAW); against `projwfc.x` 1.7e-6, 2.2e-6.
- I18 landed: Pt Loewdin charges to every digit `projwfc.x` prints; Co `|<L>|` 0.117097947 along x and y.
- I19 landed: Ni at `tot_magnetization = 2`, levels 20.704251 and 14.299028 eV against 20.7043 and 14.2991.
- I20 landed: NiI spiral force against the four-cell supercell 1.17e-8 Ry/bohr.
- I21 landed: NiI spiral stress 7.8e-10 Ry/bohr^3.
- I22 partial (merged): the velocity operator on a spiral, trace 5.5e-10; its consumers refuse.
- I23 blocked (merged): refusals kept and tested.
- I24 landed: site moments of the NiI spiral against the supercell 2.1e-10.
- I25 partial (merged): the charge STM image of a spiral 2.9e-9; the rest refuses.
- I26 blocked (merged): refusal kept and tested.
- I27 landed: H chain spiral under `tb09`, `kubler`, levels against the supercell 4.8e-14 Ry.
- I28 landed: NiI spiral with U, supercell `ns` against the turned spiral 1.2e-10.
- I29 blocked (merged): physics, the refusal stays.
- I30 blocked (merged): refusal and test unchanged.
- I31 partial (merged): a GGA on an unpolarized ultracell, +4.1e-7 Ry/cell above the PBE supercell at 48 bands; polarized refuses.
- I32 landed: ultracell with DFT+U, +4.9e-6 and +7.9e-7 Ry/cell above the DFT+U supercell at 12 and 24 bands.
- I33 to I36 not addressed (merged unchanged with the ultracell unit); their refusals fire.
- I37 unmerged: merged and reverted after the fast gate (`test_host_store_uploads.py`, see Progress); its number, 2.3e-13 on every block of `<m|dH/dk|n>`, stands in its record.
- I38 landed: the real-space consumers on gamma storage against `k = 0`, PDOS 3.9e-12 to STS 8.2e-8.
- I39 landed: real-time current from a gamma ground state against `k = 0`, 1.8e-13.
- I40, I41 unmerged (confirmed, held back with the wedge unit); I42, I46 unmerged (done, not reverified); I47, I48 partial; I43 to I45 blocked. Records only.
- I56 landed: `lelfield` on a streamed store, bit-identical total, `el_pol` 3.9e-14.
- I57 landed: `lelfield` on `K_POINTS automatic` against a new `pw.x` 7.5 run, total 1.2e-8 Ry.
- I49 to I141 were not attempted this night.

## Follow-up deliverables

Only the Limits notes and the README phonon bullet are written. Still to do for each landed item: a guide section with an executed snippet where the regime is worth showing, a notebook or a notebook cell, and a timing against `pw.x` or `ph.x` where there is one (I1, I2, I4; I6, I7, I8 have no reference code for the regime). I1's record also leaves a finding for `tests/regression/test_lsda.py`'s comment on `CASES`: the `atom-lsda` LUMO's 3.6 meV sits with `pw.x`'s `gamma_only` route, not with QE's interpolation table. I17's verifier and I12's note weak unit tests, for the session that lands them.

For integration two's eighteen landed items the same is left: only the Limits notes and three
README cells (the `lelfield` and Chern rows, the Tran-Blaha bullet) are written, and each still
needs its guide section with an executed snippet, a notebook cell and, where there is reference
code, a timing (I9, I14, I15, I57 against `pw.x`; I17, I18 against `projwfc.x`; I19 against
`pw.x`'s Fermi levels; the spiral, ultracell and gamma items have none). Two stale texts remain:
`response/piezo.py:374`, which is the strain family's and was not merged, and
`tddft/spinkernel.py:51-67`, which is the magnon family's, outside I1 to I48.

Seven slow regression files of the merged units were not run, on the user's standing rule of
2026-10-16 that sessions run fast tests only (`CLAUDE.md`, master `6a432d6`):
`tests/regression/test_spiral_supercell.py` (killed by the coordinator part way), `test_stress.py`,
`test_ultracell.py`, `test_ultracell_regimes.py`, `test_gamma_only.py`, `test_gamma_consumers.py`
and `test_realtime_gamma.py`. The seven that had finished before the rule all passed, on
`w-inc-night2` at `30d5e80` (`review/incompat/night2/slow-summary.txt`): `test_polarization.py`
11, `test_topology.py` 17, `test_mbj.py` 19, `test_berryfield.py` 12, `test_spinor_pdos.py` 17,
`test_spinor_projection_symmetry.py` 5, `test_nesting.py` 7. None of the seven touches I37.

## D22 notes from the reports, for `docs/d22-notes.local.md`

- `git checkout <commit> -- <paths>` stages the old content, so `git checkout -- .` does not undo it; the recipe's restore step has to be `git checkout HEAD -- .` (I1, I4, I19, spiral, I37 all hit it). After a partial revert of a directory, files that exist only on the branch stay in place (ultracell).
- The local worktree guard refuses one Bash call that mixes local writes, scp and a remote `git worktree add`, a git command inside a heredoc or chained ssh line, an scp with a glob or a path from a shell variable, a tar pipe, and grep or sed on a path containing `.git.` (the QE benchmark names). Write the script with the Write tool, scp it by explicit path, run `ssh host 'bash script'` (I1, I2, I5, I6, I13, I38).
- `git worktree add ... | tail -1` hides a failed add on an existing path, and the next command runs anyway; check `git worktree list` first and suffix worktree names with something unique (I4, I13, I16). A branch held by another local worktree is checked out with `git checkout --detach <branch>` (I2, I9).
- Before queueing for a unit, look in `/l/ladovj1/review/incompat/<ID>/` and `<ID>-verify/` and in `git worktree list`, since a sibling verifier may already have run the same arms (I7, I8, I2, strain, I12).
- The D22 ref `incompat-<ID>` can lag the local branch (I2, strain, I17, I18), and the implementer's worktree can sit at an older WIP commit (I3, I17, I37, strain with uncommitted changes); compare `git rev-parse` before trusting either.
- The six-slot queue was saturated around 07:15 to 07:36 on 2026-10-16, with waits of 7 to 20 min; a job's `# slot` header can carry an earlier time than its `# waiting` line (I3, I4, I6, I8, I13, finitefield).
- Costs on one slot core: `ph_insulator_paw_magn` about 70 s; bismuthene-epsilon-us-soc SCF 37 s and Gamma DFPT 354 s at `tr2 = 1e-14`, 695 s at 1e-16; spin-orbit AlAs DFPT 48 s; vendored 7.5 pw.x/ph.x on the BN U cells 2+7 s (US) and 3+18 s (PAW); triplet O2 pw.x lberry 3 s, AlAs 11 s; Si8 lelfield chain 70 to 90 s, spinor 266 s; Pt SOC 3x3x3 wedge pair 46 s (I2, I3, I5, I9, finitefield, I17).
- `Calculator.with_positions` fails when the displacement changes the number of k-points (`TypeError: add got incompatible shapes for broadcasting: (10, 1, 3), (4, 2688, 3)` in `planewaves.kinetic`), which reads like a basis bug; a finite difference that lowers the symmetry builds each cell from the input (I3).
- The 6 GB cap of slot.sh kills `test_topology.py::test_bismuthene_parity_invariant_is_well_formed_and_reproducible` after about a minute with only `# exit 137` in the log; deselect it in a whole-file run (I9). `pw.x lberry` on AlAs at `nspin = 2`, `tot_magnetization = 2` prints a string phase of exactly 0.00000 that looks converged and is not (I9).
- The QE 6.5 test-suite benchmarks for `ph_U_insulator` have drifted from 7.5 (zz 1.229085293 against 1.229073580); a fresh 7.5 run is cheaper than trusting them (I5).
- The D22 venv has no PyYAML, so `tests/unit/test_check_guide.py` fails twice in every gate run there with `ModuleNotFoundError: No module named 'yaml'` (spiral).
- A pytest-timeout traceback in a log is not a failure on its own; read the summary line (spiral). `tail -5 a b c` over ssh fails, use `tail -n 5` (wedge). A Python syntax error in a harness shows only as `# exit 1 after 0 s` (wedge). A Monitor or background `until ssh ... test -f` loop reported success before the marker existed; a foreground loop over `ssh -n ... ls` matching the name worked (I19). slot.sh passes extra arguments to the harness and runs it with cwd at the worktree (I37, I18, I39).
- The fast gate ran in 865 s on 2026-10-16 09:20 against 1622 s at 08:08 on the same cores and caps; so one gate time on D22 is not a timing (integration two).
- A foreground `for` loop of `ssh ... ls` and `sleep 60` was refused once by the local guard as "Git Destructive" right after a refused `git filter-branch`, and a `Monitor` over the same `ssh` worked; a background `until ssh ... | grep -qx NAME` loop also worked (integration two).
- A wedge or nosym pair on fcc needs an unshifted mesh; a shifted 2x2x2 differs by 1.7e-2 Ry (I17). A gamma against whole-sphere check needs the atom off the inversion centre, or the coefficients are real and a dropped conjugation is invisible (I38).

## D22 worktrees that can be removed (none removed)

Named in the reports as left in place: `w-inc-I1-ver`, `w-inc-I1-ver-rev`, `w-inc-I2-vf-tip`, `w-inc-I2-vf-rev`, `w-inc-I2-vf-nomask`, `w-inc-I2-v2`, `w-inc-I2-v3`, `w-inc-I3`, `w-inc-I4-ver`, `w-inc-I4-ver2`, `w-inc-I4-ver-rev`, `w-inc-I6-v3`, `w-inc-I7-verify`, `w-inc-I7-verify-rev`, `w-inc-I7-verify-mA`, `w-inc-I7-verify-mB`, `w-inc-I7-v2`, `w-inc-I8-ver-rev`, `w-inc-I8-ver-clean`, `w-inc-I8-ver-mut`, and `w-inc-I16-base`. Those of units 9 to 23 (for example `w-inc-I9-adv2-rev`, `w-inc-I12-vr2`, `w-inc-I13-v29`, `w-inc-finitefield-ver`, `w-ver-I17`, `w-inc-I18-vf`, `w-inc-I19-v2-ver34`, `w-inc-spiral-fix`, `w-inc-ultracell-ver48`, `w-ver-I38-revert`, `w-inc-I39-rev`, `w-inc-wedge-v10`) stay until their units are integrated. `w-inc-night` (integration one's gate, at `136c768`), `w-inc-night2` (integration two's first gate, at `30d5e80`) and `w-inc-night3` (its re-gate after the I37 revert) are the integrations' own. The worktrees of the units integrated in integration two can go once this branch is accepted; those of I37 and the wedge (`w-inc-wedge-v10` and the rest) stay until those units land.
