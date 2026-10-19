# Tier T3 of the incompatibility work, integrations one and two, and the re-landing of sourcefree

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
- Step 3, result: the gate at `8819131e` read 2 failed, 4061 passed, 71 skipped, 1815 deselected,
  34 warnings in 1101.37 s. Both failures are `tests/unit/test_check_guide.py`'s missing `yaml`, the
  D22 venv's; `tools/check_guide.py` itself passes locally on the final head. The 71 skips are the
  set T2 saw (the worktree has no `quantum_espresso/` symlink, so `test_scf_restart.py` skips six
  and `test_units.py` one, and the rest are QE inputs this code refuses), none in a file a T3 merge
  touched. Nothing is attributable to a merge, and no merge is reverted.

## Integrator's changes that no second session has seen

The two conflict resolutions are code no verifier read: in `workflows/realtime.py` the update dict
of `run_realtime_dielectric` carries I100's `ns` under I98's condition for the one-centre `D`
(`potential == "frozen" or strength is not None`), and in `realtime/hierarchy.py`
`hierarchy_self_consistent` runs I102's order check and then I98's `require_a_potential_mode(...,
route="hierarchy")`. The evidence is the fast files of both sides passing locally after each merge
and the gate; the real-time slow files above are what would test them further.

## Outcome per ID

| ID | outcome |
|---|---|
| I86 | landed: US/PAW al2 metal phonon against `ph.x` at most 0.0136 and 0.023 cm^-1 |
| I87 | landed: US/PAW Si2 `C_ijkl` against a five-point energy difference at the 1e-7 floor (22.1 per cent with the terms off) |
| I88 | refuted twice, not merged: the fix round's smearing cutoff (`small = 1e-12`) holds, but the top-band guard at 6.96e-5 admits Fermi-Dirac states with 2e-5 (C_1111 6 to 7 per cent off) and 1.7e-6 (4e-6) in the top band; needs a limit below 1e-6 or tied to 1e-12 (`I88/ver3/`) |
| I89 | landed: PAW AlAs `e_14` onto `pw.x`'s Berry value, -0.001 per cent at the 12 mesh |
| I90 | landed in integration two: polar norm-conserving AlAs, wedge against closed grid 1.1e-9 (1.1e-3 off), the verifier's 3m cell 4.2e-9 |
| I91 | partial: US/PAW sum-rule floor 4.1e-3, flat in the step (5.29e-2 without the new term), refusal kept |
| I92 | landed: Hubbard velocity against a difference of `H(k)` falling as h^2 to 4.20e-7; DFT+U shift current and SHG reachable |
| I93 | landed in integration two: Al `tetrahedra_lin` within 0.035 eV of the smeared limit from 40^3 to 72^3, 12.43 to 12.49 eV with the mesh |
| I94 | landed in integration two: US Si residual -0.3369 to -0.0091 from 30 to 76 bands; ALDA, wedge and scissor measured |
| I95 | landed in integration two: smeared Si onto fixed occupations, 2.2e-8 at degauss 0.005 Ry |
| I96 | landed in integration two: Al Drude head 3.8e-13; f-sum 0.99410 at 40 bands, 0.99638 at 48 |
| I97 | landed: adiabatic PBE Dyson against PBE Sternheimer, -0.0020 at 60 bands and 20 Ry |
| I98 | landed: US/PAW kick local-field shift equal to the nc control on three meshes (+1.79e-1 with the augmentation held) |
| I99 | fix round confirmed, not merged: the merge conflicts with I98 in `realtime/selfconsistent.py` (the density and potential signatures), not a keep-both resolution, so it was aborted |
| I100 | landed: real-time first order with U against Kubo plus curvature, 8.07e-7 |
| I101 | landed in integration two: H2 polarizability within about 0.1 per cent from 16 bohr up, null pulse 1.1e-12 |
| I102 | landed: hierarchy orders 2 and 3 against the propagation, 3.7e-6 to 7.6e-6 at 800 steps, dt^2 |
| I103 | landed: US/PAW `lelfield` along y against `pw.x`, `el_pol` 1.7e-6 and 6.6e-6 |
| I104 | partial, landed in integration two: the Gamma phonon under `tefield` (with or without `dipfield`) to about 2e-4 Ry/bohr^2; the rest refused |
| I105 | partial: textured spinor `eps_zz` against the finite field, unlocated offset -5e-4, refusal kept |
| I106 | partial: field force on US O2 measured, Fe stress floor 4e-4 to 6e-4 Ry; PAW and noncollinear admitted untested |
| I107 | partial, nothing lifted; guard and argument confirmed by the verifier |
| I108 | landed after integration two (branch `t3-sourcefree`, I110's guard test retargeted): projected `grad_m` divergence 6.97e-31 against 3.456 |
| I109 | landed after integration two with I108: TB09 sphere divergence 1e-31 projected |
| I110 | landed: meta-GGA spin torque against m x B_xc of its own field, 5.2e-18 Ry |
| I111 | not started (worktrees `w-inc-I111*` exist on D22, no result reported) |
| I112, I113 | not started (rotate; `w-inc-rotate*` exist, no result reported) |
| I114 | partial, landed in integration two: `soc_scale` in (0, 1) on ultrasoft, coefficient 0.45804 at 0.37 against 0.4585; fully relativistic PAW refused |
| I115 | not started (`w-inc-I115*` exist, no result reported) |
| I116, I117 | not started (ultraseed) |
| I118 to I122 | not started |

## Slow test files for the user to run

The merges touch or add these; none was run (the user's rule). The three real-time merges (I98,
I100, I102) were combined by hand in `realtime/propagate.py`, `realtime/hierarchy.py` and
`workflows/realtime.py`, so the real-time files are best run together.

- `tests/regression/test_phonons.py` (I86: the new augmented-metal pair and the existing US/PAW and al2/al4 metal tests; I87 touched `phonon.py` too)
- `tests/regression/test_nonlinear.py` (I86)
- `tests/regression/test_electrostriction.py`, `tests/regression/test_streamed_strain.py`, `tests/regression/test_streamed_third.py` (I87; node ids of `test_the_elastic_constants_match_a_second_difference_of_the_energy` changed)
- `tests/regression/test_piezoelectric_paw_reference.py`, `tests/regression/test_polarization.py::test_a_paw_crystal_matches_pw_x`, `tests/regression/test_piezoelectric_paw.py`, `tests/regression/test_streamed_piezo.py` (I89)
- `tests/unit/test_hubbard_velocity.py`, `tests/regression/test_photocurrent.py` (I92)
- `tests/regression/test_tddft.py` (I97)
- `tests/regression/test_realtime_augmented_update.py` (I98, about 25 min), the slow tests of `tests/unit/test_realtime.py` (I98)
- `tests/regression/test_realtime_regimes.py`, `tests/regression/test_realtime.py` (I100, I102)
- `tests/regression/test_realtime_hierarchy.py` (I102)
- `tests/regression/test_berryfield_augmented.py`, `tests/regression/test_berryfield.py` (I103)
- `tests/regression/test_spin_torque_meta_run.py`, `tests/regression/test_source_free_run.py` (I110)

## Follow-ups

- The guide sections with executed snippets, the notebooks and the timings for every landed item;
  only the Limits notes, two README cells and one `scope.md` clause are written.
- I87: the record and the slow test's docstring on the roughness of the energy reference (the
  verifier's xc step of 4.7e-6 Ry, cause open, `/l/ladovj1/review/incompat/I87-ver/terms-us.log`),
  and the refusal message of `require_a_measured_elastic_regime`, which should say
  `memory_mode='speed'` rather than `k_batch=None` (on a GPU in memory mode `k_batch=None` still walks).
- I98: the record and docstring sentence "falls with the k-mesh" (false at 0.2 eV across 2^3, 3^3,
  4^3), and a test of the pulse path (the verifier's identity, 1.75e-4 against frozen 1.10e-4,
  `/l/ladovj1/review/incompat/I98-ver/identities.py`).
- I103: the record corrections of its verifier. The forces are 8.7e-8 (US) and 1.05e-7 Ry/bohr
  (PAW) from `pw.x`, not 5.7e-8 and 8.9e-8, so the docstring "under 1e-7 on both datasets" is false
  for PAW; the energy gap does not fall as the square of the state error; the frozen-enthalpy
  position gradient is 3.2e-4 Ry/bohr along `gdir`, nearly uniform, so after the net-force removal
  at most 3.0e-7 survives and the regression force assertion (1e-6) cannot see the cross term. The
  test gap: a fast unit test of that gradient against a central difference.
- I97: the 1.5e-2 bound is the committed cell's; a core-charge cell reads -0.019 at 20 Ry. The
  record's "60-band floor -0.0073" is not a floor (-0.0088 at 28 Ry), and at 90 bands the residue
  is -0.0054 at 32 Ry, past the RPA value, so the record should quote that point.
- Smaller record corrections the verifiers list: I86, the PAW Fermi shift of `becsum` is now
  measured (the force difference sides with the fixed code, 5.4e-6 against 5.0e-5 with it off);
  I89, the 35 Ry `pw.x` run reads 0.693330 and the one-centre term is 5.43e-4 at 8 8 8, not
  5.3e-4 "at every mesh"; I100, the floor of 8.07e-7 is the curvature stencil of the reference
  (3.76e-8 at step 1e-4); I102, the spinor and two-grid identities extended to `order=3` are the
  cheap tests to add.
- I89: notebook 28's sentence on PAW (the record has the replacement), with its `.md` export.
- I92: whether the result depends on `k_batch` with `ns` was not measured.
- Record sentences elsewhere that the landed records name and that are now stale:
  `notes/AUGMENTATION-NEXT.md` §1h ("PAW open" for the piezoelectric tensor, I89) and §3b's "What
  is left" on the elastic constants (I87), `PERFORMANCE.md:4560` ("PAW is what stays refused", I89),
  `PLAN.md` P100's closing paragraph (I87) and P50 (I89), and README's footnote mark 19 on the
  real-time row, which the I100 record found undefined.
- Fix rounds for I88, I90, I93, I99, I101 and I104, from their verifier findings (full text in each
  record's D22 directory and in the workflow's results).
- chi0 (I94 to I96): the verifier's queued jobs land in `/l/ladovj1/review/incompat/chi0-ver/`
  unattended; read them and verify. The verifier also found the ALDA kernel on an ultrasoft run
  admitted and unmeasured. The I90 `-ver2` jobs land in `/l/ladovj1/review/incompat/I90-ver2/out/`.
- sourcefree (I107 to I109) and I114: verify.
- I111, rotate, I115, ultraseed and I118 to I122: not started; I111, rotate and I115 left worktrees
  on D22 with no reported result.

## What the user decides

- I91, ultrasoft and PAW shift current. The new augmentation term takes the complete-basis
  sum-rule miss from 5.29e-2 to a floor of 4.1e-3 that does not fall with the step, so a piece
  is still missing. Options: keep the refusal with the measured floor in its message (as now);
  or lift it and document a 4.1e-3 relative error in the sum rule.
- I105, textured spinor dielectric tensor. Options: keep the refusal (as now); or lift it for
  epsilon with `born_charges=False`, accepting the unlocated -5e-4 offset (0.16 per cent of
  eps - 1).
- I108 and I109, the default of `noncollinear_gga` under `nosource`: superseded by the entry in
  "What the user decides, after integration two" (the verifier found `'kubler'` converges at 371).
- I102, `get_nonlinear_spectrum(order >= 2, potential != 'frozen')`. Options: keep its warning about
  the mesh artefact (as now); drop it, to match the real-time route; or add the same warning to
  `get_harmonic_orders`.

## D22 worktrees that can be removed (none removed)

Once `incompat-t3` is merged, the worktrees of the landed units: `/l/ladovj1/w-inc-I86*`,
`w-inc-I87*`, `w-inc-I89*`, `w-inc-I92*`, `w-inc-I97*`, `w-inc-I98*`, `w-inc-I100*`,
`w-inc-I102*`, `w-inc-I103*`, `w-inc-I110*`, and the gate's `w-inc-t3`. The revert and mutation worktrees of the other
units (`*-rev`, `*-verrev`, `*-ver2r`, `*-ver2-rev`, `w-inc-I93-ver-all`, `w-inc-I93-ver-nod4`,
`w-inc-sourcefree-ver-mg`, `-ver-mm`) are removable now. Keep the others until their fix round,
and keep `w-inc-chi0-ver` and `w-inc-I90-ver2` while their queued jobs run.

## D22 notes from the reports, for `docs/d22-notes.local.md`

Each unit's notes as reported (implementer, then verifier after the bar), 2026-10-19, for the user to append; nothing here was edited into that file.

- **I86**: Serial ph.x is only in /l/ladovj1/review/timing/ph.x. /l/ladovj1/qe-7.5/bin/ has pw.x alone, and calling ph.x there from a harness fails at once with FileNotFoundError, exit 1. Running pw.x and ph.x through subprocess inside a slot.sh harness works and keeps the queue rule. Costs on one core: 2 s pw.x and 4 s ph.x for the ultrasoft al2 cell, 3 s and 45 s for PAW. Our ultrasoft al2 metal phonon takes 40 to 50 s, PAW 145 s. A 6x6 finite-difference matrix (12 SCFs) takes 30 to 40 s. The not-slow gate took 1213 s on one slot (3 failed, 4017 passed). On 2026-10-19 between 10:36 and 11:20 the slot queue held 18 to 29 waiters, and waits were up to 25 minutes. | 2026-10-19: the shared CPU queue (review/guide/slot.sh, 6 slots) held about 45 to 58 waiting jobs from parallel verifier sessions, and my first job waited 32 minutes for a slot. The flock polling is not first in, first out, so jobs launched together finish in a random order; launch the discriminating job alone and first if its answer is needed early. /l/ladovj1/w-inc-I86-ver, -ver-rev and -ver-tog already existed (a57c6e90, from an earlier verifier round, left with uncommitted edits to phonon.py), so I used w-inc-I86-ver2, -ver2-off and -ver2-tog and restored the last two with `git checkout HEAD -- .` afterwards. `git checkout <commit> -- file` stages the change, so a plain `git checkout -- .` does not undo it. Serial pw.x plus ph.x on the 2-atom al2 metal cell (nosym, 32 k, tr2_ph=1e-16) takes 10 s for the ultrasoft dataset and 63 s for PAW through slot.sh, launched by a Python wrapper with subprocess and the binaries from /l/ladovj1/review/timing/. ph.x runs on a non-equilibrium geometry without complaint. Calling dynamical_matrix with wavefunctions=None on a norm-conserving metal fails with "ValueError None is not a valid value for jnp.array" only after it passes every guard, so it works as a cheap probe that a refusal fires before any state is read.
- **I87**: Costs on one slot. The I87 harness (three Si2 cells, three forms of the elastic constants, 24 re-converged SCFs per cell) takes 218 s. The public-path harness (electrostriction with elastic=True plus Calculator.get_elastic_constants on three cells) takes 405 s. The whole-suite fast gate (-m 'not slow', no path) took 1564 s on one slot beside a harness, with 4018 passed, 73 skipped and only the 2 yaml failures. A harness that prints json.dumps of a report holding JAX scalars fails at the very end with 'Object of type ArrayImpl is not JSON serializable' (exit 1 after all the numbers are printed), so it needs default=float. | /l/ladovj1/review/incompat/I87-ver/ and the worktrees /l/ladovj1/w-inc-I87-ver (9d858042) and /l/ladovj1/w-inc-I87-ver-rev (9d858042 with the four modules checked out from 6c5b3a36, staged) are this verification's own; an earlier pass of the same verifier made them and this pass reused them, so a restarted verifier should look there before making new ones. ver2.py there is a reusable five-point plus forms harness (case path, tangent 00/01/gen, steps, forms full/ort/none/rev), stressfd.py the stress-finite-difference cross-check, which stays usable where the energy reference of an augmented cell is rough.
- **I88**: The D22 login shell rejects `tail -5` with "option used in invalid context", so `tail -n 5` is needed (this hit `tail` over several files). At about 10:27 on 2026-10-19 all six slot.sh slots were busy, and a job waited about 3 minutes before its `# slot` line appeared, so the first poll after a launch may show only the waiting line. slot.sh passes trailing arguments through to the harness script, so one harness can take a selector argument (`harness.py al`). The serial /l/ladovj1/qe-7.5/bin/pw.x runs a strained al2-metal cell in about 1 s, so a pw.x sign check of a curvature costs under 20 s through a slot. The full fast gate (`-m pytest -m "not slow" tests`) took 1565 s on one slot core. | - The six slot.sh slots were held by other verifiers (I89, I93, I98, I102) for 10 to 20 minutes at a time, so a batch of six or more jobs runs one after another. To free a slot, cancel your own waiting slot.sh with pgrep -f on a pattern holding a bracket class (for example "ver2/stress-[av]"), so the pgrep does not match the ssh command line, then kill by pid.
    - /l/ladovj1/w-inc-I88-ver and w-inc-I88-ver-rev already existed at 6967d36b, presumably from an earlier verifier, so run git worktree list before choosing a name. I used w-inc-I88-ver2 and w-inc-I88-ver2-rev, and the -rev one is restored to HEAD.
    - The implementer's w-inc-I88-v2 is at 07141133, not the 6967d36b the task names.
    - An environment variable placed before `nohup bash slot.sh ...` reaches the python through the systemd --user --scope. That is how SMALL=1e-12 got into the run, and the log shows "patched _SMALL".
    - `git checkout <rev> -- files` stages the files, so `git checkout -- .` does not undo it; use `git checkout HEAD -- files`.
    - The worktree-isolation guard on this workstation refuses compound commands that hold brace expansions or a variable in an option position (scp {a,b}, sed $W). Use plain separate commands, and put long launches in a script file.
- **I89**: - With DEFUQUMAT_MEMORY_MODE=memory on the CPU (k_batch=1), the taped piezoelectric route walks the k axis. On PAW AlAs at 25 to 35 Ry it takes 2.5 to 3.6 GB and 2 to 5 min per wedge rung (4 to 12). zstar_eu takes 4 to 18 min per rung. Both fit the 6 GB slot.sh cap.
    - Serial pw.x lberry on 2-atom PAW AlAs at 25 to 30 Ry takes 12 to 35 s per nscf with 15 strings over 8x8. A whole finite difference (2 scf, 6 nscf) takes 90 to 210 s, and 576 s at 21 strings over 12x12.
    - pw.x reports 34 strings, not 64, for an 8x8 transverse mesh even with nosym and noinv (time reversal pairs them). A harness that counts strings sees 34.
    - The P line in pw.x's polarization block, '(e/Omega).bohr', gives the phase to 7 decimals as P/mod*MOD_TOT. The TOTAL PHASE line gives only 5.
    - The slot queue was full from about 10:45 to past 12:05 on 2026-10-19. review/incompat/I89/pwx35.log was still waiting from 11:05 when I left. Nothing depends on it, and it writes only to review/incompat/I89/pwx35. | 2026-10-19. A pytest run that names a file failing at import (here an ImportError for a missing constant on the reverted worktree) stops the whole run with exit 2 after 3 s and reports none of the other files. On a revert, the file that imports the new symbol has to go in a run of its own. The slot queue was saturated by other agents (load about 8.5 from 15:08 to 16:05). Waiting jobs got a slot in random order rather than in the order they were queued (flock -n on each slot), so a job queued first could wait almost an hour while later ones ran. The pkill trap also holds for pgrep plus kill. `kill $(pgrep -f '<pattern>')` sent over ssh matches the remote shell whose command line contains the same pattern, so it kills that shell (ssh exit 255), and it killed the queued waiter as well. Match on something that appears only in the target's own command line, or read the pid first in a separate ssh call. Memory mode keeps the PAW AlAs 25/200 Ry piezo rung flat at about 2.8 to 3.0 GB RSS up to 16 16 16 (145 k-points, 334 s). The whole-k route (memory_mode 'speed', the CPU default) peaks at 5.1 GB at 4 4 4, close to the 6 GB slot cap, so a denser whole-k rung would be killed.
- **I90**: A harness that spawns child processes (multiprocessing 'spawn', one geometry per child) through slot.sh runs the children inside the same 6 GB systemd scope and the same taskset CPU as the parent, and this worked: 14 Berry geometries plus three response meshes up to 512 k on norm-conserving AlAs at 10 Ry fit in one slot, in 1484 s. The full fast gate (-m "not slow" over tests/) took 1199 s in one slot. The queue wait was about 8 minutes at 10:25 with I87 and I89 jobs in the pool. | - **The queue was full and stayed full.** On 2026-10-19 at about 15:08 all six slot.sh slots were busy with verifier jobs for I88, I89, I93, I99 and chi0, plus a long tail of waiters. My five jobs in /l/ladovj1/review/incompat/I90-ver2/ (wedge, conv, cell27, unit, unit-rev) waited for a slot for the whole session.
    - They will still run unattended. Their results land in /l/ladovj1/review/incompat/I90-ver2/out/result-{wedge,conv,cell27}.json and the matching .log files.
    - The worktrees are /l/ladovj1/w-inc-I90-ver2 (incompat-I90 at 90ee62d7) and /l/ladovj1/w-inc-I90-ver2-rev (piezo.py from 6c5b3a36 plus a shim).
    - **/l/ladovj1/w-inc-I90-ver already existed**, from an earlier verifier pass. So did w-inc-I90-ver-rev. I left both alone and used the -ver2 names instead.
    - **A k-set built by hand is not a wedge.** `with_kpoints(KPoints.automatic(grid, shift, cell))` without `rotations=` returns the full grid even when symmetry is on. That is why the prior verifier's wedge.log ran 64 points.
    - `Calculator.from_file` on a card with nosym removed gives the real reduced set: 18, 46 and 95 points at 4^3, 6^3 and 8^3 for the mm2 AlAs cell.
    - **The worktree-isolation hook refuses an ssh command whose text contains 'git' inside a heredoc or a long chain.** The fix is to write the setup script locally, scp it over, then run `ssh ... bash script.sh` as a plain command.
- **I91**: - The six-slot queue held three jobs for 26 minutes (10:33 to 10:59 on 2026-10-19), with about 20 jobs waiting.
    - Costs on one slot: a complete-basis dense harness on alas-piezo-tiny (US or PAW, 159 plane waves, SCF included) takes about 40 s. A step sweep of three steps also takes about 40 s. The six photocurrent-related unit files take 46 s. The fast gate took 1249 s.
    - `Calculation.at_kcart` wants a (1, 3) array. A bare (3,) array fails deep inside `basis/planewaves.py:_kinetic` with "IndexError: Too many indices: array is 1-dimensional, but 2 were indexed", which reads like a broken basis.
    - A harness that builds a `VelocityOperator(calc, ..., kcart=k')` at a k' different from the calculation's own k gets a wrong augmentation connection, because `augmentation_connection` reads the calculation's own projectors. Build the operator on `calc.at_kcart(k')` instead. This one is a code fact, recorded in notes/incompat/I91.md. | /l/ladovj1/w-inc-I91-ver already existed (clean, fb9f4087) when this verifier started. An earlier verifier session at 12:11 had made it and left logs in /l/ladovj1/review/incompat/I91-ver, including harness.py with GK/AA/CC env overrides for the k-point and the pair. So I made /l/ladovj1/w-inc-I91-ver2, plus /l/ladovj1/w-inc-I91-ver2r for the revert check, which is now restored clean. My logs are in /l/ladovj1/review/incompat/I91-ver2. A workflow re-run should check whether the "-ver" worktree name is already taken. The harness takes about 70 s per ultrasoft run (three steps, SKIP_L=1) on one slot, and the unit file about 10 s.
- **I92**: On 2026-10-19 around 11:20 the shared CPU queue held about 20 slot.sh jobs on 6 slots (I86, I87-ver, I89, I91, I93-ver, I97, I99, plus my own). My four jobs waited between 16 and 44 minutes for a slot, then ran in 4 to 200 s. A harness file can be replaced on D22 with scp while its slot.sh line still says '# waiting for a slot', because python only starts once a slot is taken. That is how I fixed an attribute error before it ran. A pytest run of tests/unit/test_hubbard_velocity.py, tests/regression/test_photocurrent.py and tests/regression/test_shg.py with -m 'not slow' runs nothing from those files, since every test in them is slow, so it reads '17 passed' only because of the machinery file beside them. The G-vector object of a Calculation's basis is basis.smooth (or basis.dense); there is no basis.gvectors. I created the worktree /l/ladovj1/w-inc-I92-ver and left it in place, and no job is running from it.
- **I93**: (1) The tetrahedra_opt weights with 9 bands on a 48^3 fcc grid go over the 6 GB slot.sh scope cap. The optimised scatter holds (ntetra, nbnd, 4, 20) reals, 3.8 GB before the jvp. On 64^3 the job was killed even with the Drude delta computed one band at a time. I did not measure which step holds the memory; the candidate is calculation.occupations building the interband weights of all bands. The log just ends with "# exit 137". (2) A harness that survives every row can still be killed at its last call (h5 died after all eight 64^3 rows even with jax.clear_caches() after each row), so print every row as soon as it is done. (3) Around 10:40 to 11:00 on 2026-10-19 about 25 jobs were waiting for the six slots, and my jobs waited up to 22 minutes before starting. (4) kill <pid> on a slot.sh that is still waiting is clean, since no python has started yet. Find the pid with pgrep -af, never pkill -f. | 2026-10-19: an earlier verifier session had already made /l/ladovj1/w-inc-I93-ver plus -ver-all and -ver-nod4 (mutant worktrees), with logs in /l/ladovj1/review/incompat/I93-ver/ (finished about 12:18). I did not touch them. I made /l/ladovj1/w-inc-I93-ver2 (at incompat-I93 fc29dc71) with logs in /l/ladovj1/review/incompat/I93-ver2/ (a72.log, b80.log, c72.log; harness ver.py copied from I93-ver). So a verifier should check whether the -ver worktree name is already taken and pick the next suffix. In the afternoon all six slot.sh slots were busy (load average about 8, 29 users), and a job waited 18 minutes for a slot. Al on 80^3 (11921 wedge k-points) takes about 175 to 300 s per smeared conductivity row on one core and fits in 6 GB. tetrahedra_lin on 80^3 does not fit (killed, exit 137, before its first row), while 72^3 (8797 k) runs in about 195 s per tetrahedron row. On this workstation the worktree-isolation guard refuses a Bash command whose sed path is held in a shell variable, and it refuses compound heredoc-plus-run commands, so scripts go in with Write and run as one plain command.
- **chi0**: A slot.sh launch whose log stays at "# waiting for a slot since ..." can be dead, with no process waiting behind it: u4.log from the earlier session is one. A waiting log and a running one look the same until a "# slot N" header appears, so check for that header (or pgrep) before assuming a job is queued. The full fast gate (pytest -m 'not slow') through one slot on D22 took 1513 s (25 min) at 9db69bb4: 4012 passed, 73 skipped, 2 failed (test_check_guide, no yaml). Six test_scf_restart skips read "QE reference tree not present at quantum_espresso/..." even with DEFUQUMAT_QE_ROOT set; those tests use the relative path, not the variable. | - On 2026-10-19 around 15:40, the six-slot CPU pool (review/guide/slot.sh) held about 25 queued jobs from several incompat verifiers (I86, I88, I89, I90, I100, I101). A new job waited more than 15 minutes for a slot even though each of mine needed only about 40 to 60 s. A verifier with a time budget should launch its single most discriminating harness first, not a batch.
    - An ssh line of the form 'kill <pid>; ... nohup bash slot.sh ... & disown; sleep 2; head log' hung until the 120 s tool timeout, even with stdin and stdout redirected. A launch belongs in a script file run over ssh, as the recipe says, including a relaunch.
    - tests/regression/test_tddft_metal.py:smeared_silicon reads chi0_module._drude_block unconditionally. A harness that imports it on a tree reverted to 6c5b3a36 dies with AttributeError before computing anything, unless the attribute is stubbed first with getattr(c, '_drude_block', None).
- **I97**: All six slots were busy when I launched at 16:11 (load 11). Jobs queued for 1 to 3 minutes and then ran: about 1 min for each 3x3x3 cell and 172 s for the 4x4x4 one with 90 bands at 32 Ry (nbody 868, well inside 6 GB). Harness inputs went to /tmp/i97ver-*.in on D22. I created the worktrees /l/ladovj1/w-inc-I97-ver and /l/ladovj1/w-inc-I97-ver-rev, both at 2a8ff6bc; the -rev one had its kernels.py restored with git checkout HEAD afterwards, so it is clean. Nothing new about running on D22 beyond the recipe.
- **I98**: The slot queue showed about 55 waiting slot.sh processes for six slots on 2026-10-19 around 15:10, but my first job started within three minutes, so a crowd of waiters does not mean a long wait (many of them are probably stale or other agents' short jobs). In an ssh command line, 'tail -3 a b c' fails with 'tail: option used in invalid context -- 3'; use 'tail -n 3'. The slot.sh pool accepts 'python -c ...' (it passes its arguments to python -u), which gives an import check of a slow test file without running pytest on it. | - The D22 ref incompat-I98-ver is at 549c2b3a (the implementer's incompat-I98 there is still 48a0f16b). I made two worktrees, /l/ladovj1/w-inc-I98-ver and /l/ladovj1/w-inc-I98-ver-rev; the second is restored to HEAD. Scripts and logs are in /l/ladovj1/review/incompat/I98-ver/.
    - Restoring a file reverted with git checkout <commit> -- <files>: that command stages the change, so git checkout -- . afterwards leaves the index alone and status still reads M. The restore that works is git checkout HEAD -- <path>.
    - The six-slot pool was shared with the I102-ver and I104 verifiers. Only one slot was free at launch, and 21 jobs took about 50 minutes.
    - On an unshifted grid the x-field little group takes 3x3x3 to 4 points and 2x2x2 to 3, so 3x3x3 costs about what 2x2x2 does (1079 s against 842 s for ultrasoft frozen + hxc + held).
    - The KSET identities at 8 Ry take 25 to 90 s per arm.
- **I99**: The local scratchpad directory is shared with other concurrent workflow agents: my harness.py there was overwritten by the I100 agent mid-run. Keep per-item files in a subdirectory named after the item (I used scratchpad/I99x/). Passing an empty string argument through slot.sh to a harness works when quoted in the run script (""). | The slot pool is shared and handed out by random polling, not first come first served: one of my jobs queued at 15:08 got a slot only after jobs queued at 15:53 had started, about an hour later, with about 60 slot.sh waiters on the machine. Plan for an hour's latency when the pool is busy. A git worktree add of the same commit as a second detached worktree, followed by a sed edit, is a cheap way to run a mutant of a module through slot.sh; git checkout <ref> -- . and then git worktree remove clean it up. Harness gotcha: realtime.pulse.Adiabatic carries its own polarization (default x), and dense_first_order's current is projected on its direction argument. A harness that changes the direction must set Adiabatic(polarization=...) and project the propagated current on the same unit vector, or it reads a spurious discrepancy (11 per cent here) that does not fall with dt.
- **I100**: I made two D22 worktrees: /l/ladovj1/w-inc-I100-ver, at incompat-I100-ver = 59f27d1f fetched from my bundle, and /l/ladovj1/w-inc-I100-rev, the revert arm, restored to clean afterwards with git checkout HEAD -- defuqumat. Both can be removed. The review directory is /l/ladovj1/review/incompat/I100-ver/.
  
  Queue: the six-slot pool had one free slot at 15:10 on 2026-10-19 with load 7 to 8 from other agents. My eleven to fourteen queued jobs started between 15:10 and 16:16, so allow about an hour of queueing for a dozen 100 to 400 s jobs.
  
  Small traps:
    - `tail -3 a.log b.log` errors over ssh ('option used in invalid context'); use `tail -n 3`.
    - A JSON argument holding a whole pw.x input passes through slot.sh unharmed as one quoted argument, but the quotes inside the namelist (calculation = 'scf') are eaten by the outer single quotes. It is harmless for the reader; write the input to a file instead if that matters.
    - The workstation's worktree guard refuses compound shell commands (heredoc plus && chains). Write a small generator script with the Write tool and run it as a plain command.
- **I101**: 2026-10-19: the six-slot CPU queue (slot.sh) sat 36 minutes (15:13 to 15:49), with every slot held by other T3 units. `fuser /l/ladovj1/review/guide/locks/slot<n>` gives the PID that holds each slot, and `ps -o etime,args -p <pid>` then names the job and its age. A smoke run of the harness (a few steps, about 20 s) caught two crashes before a full check run was spent. | - run2.sh in /l/ladovj1/review/incompat/I101-ver was started twice. The second start was at 17:16:17 and did not come from my command. When a second slot.sh is pointed at the same log path, it truncates the log and writes its "waiting" line, while the first job keeps appending through its open file descriptor. The log then looks like one job with its middle missing. Give every launch its own log name, and run pgrep -af <harness> before trusting a result.
    - At load average 7.4 the six-slot pool took about 20 minutes to free a slot, so eight queued jobs effectively ran one after another.
    - I created two worktrees: /l/ladovj1/w-inc-I101-ver (at fff605f3) and /l/ladovj1/w-inc-I101-verrev (fff605f3 with the three modules checked out from 6c5b3a36, uncommitted). Both can be removed.
- **I102**: - The slot pool in /l/ladovj1/review/guide/slot.sh is shared with other verifiers' jobs (I101, I106 ran alongside), so a queued propagation can wait 30 minutes or more for a slot. A scratchpad script can also be overwritten by another agent's workflow, so keep launch scripts on D22 under your own review directory.
    - Timings on one D22 core, propagate_orders with order=3 and potential='hxc':
    - AlAs 4 Ry, 2 k-points: 290 s at 1600 steps a period, 590 s at 3200; hartree 460 s at 1600, 550 s at 3200.
    - Two-grid cell (ecutrho=32): 360 s at 800 steps.
    - get_harmonic_orders on the nspin=2 AlAs magnet (6 Ry, 2x2x2 nosym, eta_t=20): 1710 s at 400 steps and 3380 s at 800.
    - tail -5 is rejected on D22 ("option used in invalid context"); use tail -n 5.
    - The nspin=2 AlAs magnet with symmetry on (nosym and noinv dropped) is refused by _solved_states: fixed occupations cut a degenerate multiplet at Gamma (gap 1.4e-15 Ry between bands 5 and 6 of the up channel). That is why the regimes test runs it with nosym, and it means no wedge check of nspin=2 is possible on that cell.
    - A harness that puts os.getcwd() first on sys.path and is launched through slot.sh from a sibling worktree imports that worktree's package; printing defuqumat.__file__ in each log confirmed it.
    - The verifier's worktree /l/ladovj1/w-inc-I102-ver at 39f91dbb is left in place. The ref incompat-I102 in /l/ladovj1/defumat was moved from 406afca3 to 39f91dbb by a forced fetch; the implementer's worktrees are detached and were not touched. The temporary mutation and revert worktrees (-ver-rev, -ver-m1, -ver-m2) were restored and removed.
- **I103**: On D22, 'tail -5 file1 file2' fails with 'tail: option used in invalid context -- 5'. Use 'tail -n 5'. The full fast gate run as 'pytest -m "not slow" tests/' in one slot took 27 min (1649 s): 4012 passed, 73 skipped, 2 failed, both test_check_guide (no yaml). A test_units skip says 'QE reference tree not present at /l/ladovj1/qe-7.5/qe-7.5/Modules/constants.f90' even with DEFUQUMAT_QE_ROOT set, so that tree has no Modules/ sources. The augmented Si8 lelfield SCF chain (10 k-points, ecutrho 160) takes 10 to 20 s per SCF in one slot, and the pw.x chain about 20 s per dataset. | Run directory: /l/ladovj1/review/incompat/I103-ver.
    - Worktree w-inc-I103-ver is at incompat-I103 (035e63be) and was left in place.
    - Worktree w-inc-I103-ver-rev was used for the revert. I restored it with `git checkout incompat-I103 -- defuqumat` and left it in place.
  
  Queue: at load 8 to 9, a job in the six-slot slot.sh pool waited 8 to 22 minutes for a free slot. The right shape is to launch every job at once and wait with one background until-loop on the "# exit" lines. Chained foreground sleeps are blocked by the harness.
  
  Not about D22, but it costs the next agent a round trip: the local scratchpad is shared by the parallel agents of one session. A `scp <scratchpad>/ver/*.py` from a shared subdirectory shipped about 60 files written by other agents (I deleted them on D22 from my own fresh directory). Two of my Write calls also overwrote existing ver_rev.py and run.sh files in that shared subdirectory. Each unit should use a scratchpad subdirectory named after itself (I used ver-I103 afterwards).
  
  The worktree-isolation guard on this workstation refuses shell commands that contain a git call it cannot verify or a sed fed a variable. Use plain absolute paths, or detach the worktree at the branch, instead of `git show <ref>:<path>` inside loops.
- **I104**: The slot queue is not FIFO. slot.sh polls six locks at random every 20 s, so a job queued behind about a dozen I102 jobs waited 19 minutes, and jobs queued later sometimes started first. The CO inputs tests/data/qe/efield-co*.in give both masses as 0.0. pw.x replaces a mass <= 0 with its table value (PW/src/input.f90:1915-1918), but defuqumat does not, so dynamical_matrix dies in numpy eigh with "Eigenvalues did not converge". That message reads like a broken solve, not a missing mass. Separately, not a D22 fact: the session scratchpad path was shared with another agent, which overwrote my scratchpad/harness.py. Use a unique subdirectory. | I made two new worktrees on D22, /l/ladovj1/w-inc-I104-ver (detached at 1bf5e801) and /l/ladovj1/w-inc-I104-ver-rev (1bf5e801 with phonon.py, sternheimer.py and sawtooth.py checked out from 6c5b3a36). Logs are in /l/ladovj1/review/incompat/I104-ver/. On 2026-10-19 around 17:40 the load average was 9.7 and slot.sh waited about 9 to 12 minutes for a free slot. A physics fact that is useful beyond this unit: for CO in the 10 bohr box at ecutrho=300, a central difference of the forces is limited by a deterministic roughness of about 3.5e-7 Ry/bohr. This roughness does not change between conv_thr=1e-11 and 1e-13, so steps below about 0.0035 bohr measure the roughness rather than the force constants. One other point on the worktree-isolated workstation side: a compound local command (a variable used inside sed, or heredoc file writing chained with scp and ssh) is refused by the isolation check. Use the Write tool for files and separate plain scp and ssh calls.
- **I105**: A Sternheimer epsilon of i-atom-soc on one string of k-points costs this much on one slot: 374 s at 3 k, 577 s at 6, 1582 s at 9, and 1859 to 2204 s at 12 (the moment-along-x state is slower). A plain SCF there takes 35 to 90 s. lelfield on the fully relativistic iodine, with the spin-orbit refusal bypassed, converges in 5 to 7 iterations from the plain state at nberrycyc = 3, about 20 to 30 s a run. The slot queue held 42 to 44 waiting slot.sh processes from 16:31 to 16:45 on 2026-10-19, and the first job waited about 20 minutes for a slot. A unit whose branch has no commits yet cannot ship a bundle (the range 6c5b3a36..branch is empty). For a standalone harness, a worktree detached at incompat-base3 does the job, and the branch goes to a -v2 worktree later. In the guard harness the berry-field refusal must be patched as defuqumat.scf.driver.require_a_berry_field_regime, because driver imports the name directly, while the dielectric one is patched on defuqumat.response.efield. | I used no D22 time: the item is partial, so I checked the guard and the record locally with fast tests only, and created no D22 worktree.
- **I106**: The workflow scratchpad (/tmp/claude-3291340/.../ff012999-.../scratchpad) is shared by the workflow agents running at the same time. A file with a generic name (run1.sh, run3.sh) can be overwritten by another agent between your write and your scp. I ran a run3.sh that had become the I101-ver unit's launcher, so I queued e0scan and mesh16 a second time in /l/ladovj1/review/incompat/I101-ver: their logs now show one run each, headed 17:19:56 and 17:24:17, and nothing was killed. Rule: every file an item puts in the scratchpad carries the item ID and lives in its own subdirectory, and you read a launcher back on D22 (head) before you run it. Also: 'tail -3 *.log' fails over ssh on D22 ('option used in invalid context'); use 'tail -n 3'. | - `git checkout <sha> -- <files>` in a D22 worktree stages the old files into the index, so the recipe's `git checkout -- .` does not undo it. `git checkout HEAD -- .` does. Measured in /l/ladovj1/w-inc-I106-ver-rev: 7 modified files remained after `git checkout -- .`, 0 after `git checkout HEAD -- .`.
    - /l/ladovj1/review/incompat/I106/run3.sh launches the I101-ver jobs (e0scan.log and mesh16.log in /l/ladovj1/review/incompat/I101-ver), not I106 jobs. It is misfiled under I106.
    - With the kernel cache warm, each I106 harness arm (O2 at 9 bohr, or Fe on 1 k-point, about 5 SCFs at conv_thr=1e-12) ran in 8 to 50 s through slot.sh. The pool has 6 slots, so 8 launches queue 2.
    - My worktrees /l/ladovj1/w-inc-I106-ver and /l/ladovj1/w-inc-I106-ver-rev (now clean at 22593e2e) and the directory /l/ladovj1/review/incompat/I106-ver are left in place.
- **sourcefree**: 2026-10-19: The fast gate run as one pytest process through slot.sh (one core, one thread, 6 GB cap) took 24:25 at ed915e24: 4016 passed, 73 skipped, and the 2 known test_check_guide.py yaml failures. It fits in the 6 GB scope. The slot queue was full (all six slots busy) for about 3 minutes at 17:27 and for longer later, so expect to wait. Separately, not D22-specific: the session scratchpad directory is shared with a parallel incompatibility agent (I111). My run1.sh, run2.sh and run3.sh there were overwritten by its files of the same names after mine had been copied to D22. Give scratchpad launch scripts names that include the unit.
- **I110**: The fast unit file tests/unit/test_spin_torque_meta.py takes 15 s on a D22 slot (one pinned core, one thread) and 4.8 s on this workstation, about 3x slower. A "five seconds" judgement for whether a test belongs in the gate should therefore be timed on the workstation, not read off a D22 log. | I made two worktrees, /l/ladovj1/w-inc-I110-ver at incompat-I110 (8efa0ec4) and /l/ladovj1/w-inc-I110-ver-rev, where spin_torque.py and calculator.py were checked out from 6c5b3a36 and then restored to the branch. The logs are in /l/ladovj1/review/incompat/I110-ver/. Four jobs ran in parallel through slot.sh without trouble. The harness, four SCFs on h-chain-90deg under TB09 in the 5 bohr box, took 80 s on one core, and the ecutwfc=35 run took 28 s. Nothing I hit is missing from the D22 notes.
- **I114**: With the shared /l/ladovj1/jaxcache warm, one noncollinear ultrasoft SCF of the one-atom tetragonal cobalt cell (18 k-points, nosym, ecutwfc 20) takes about 8 to 14 s on one slot core, compile included, so a sweep of 19 such SCFs as separate slot.sh jobs finishes in about 2 minutes with six slots.

## Integration two (2026-10-20)

The fix rounds of the overnight log (`night-log.md` in the session scratchpad), each verified again
in a separate session, merged with `--no-ff` in item order, one commit each. The harness would not
let this session work in the locked worktree `agent-a155b466dfdcaf047` (it refuses any git command
aimed outside the session's own worktree), so the merges were made on the session's own branch
`worktree-agent-a67b42707eb2a1190`, reset first to `incompat-t3`'s head `82a0ff05`. **The ref
`incompat-t3` in the workstation repository is still at `82a0ff05`**; the integrated head is a
fast-forward of it (`git merge --ff-only <head>` inside `agent-a155b466dfdcaf047`). On D22 the ref
`incompat-t3` was fetched to `3d715272`, the head the gate ran on (later commits are notes only).

| unit | tip | outcome |
|---|---|---|
| I90 | 101f5665 | merged; record file add/add conflict, the branch's taken |
| I93 | 06d1aadb | merged; record file conflict, the branch's taken |
| chi0 (I94 to I96) | c2a83c97 | merged clean |
| I99 | 624cf2ea | **not merged**: besides the expected `ns=None, tau=None` lines of `workflows/realtime.py`, `realtime/selfconsistent.py` conflicts with I98 in five hunks: I98 made the density callable return `(rho, becsum)` and the potential take `(rho, becsum)`, I99 made them carry a `(rho, tau)` state, and the block scan, the energy function and the start of `propagate_self_consistent` differ in the same way. That is a combination by hand, not keep-both, so the merge was aborted |
| I101 | cd14815d | merged; `run_nonlinear_spectrum` keeps I102's warning on orders 2 and 3 with the potential updated and adds I101's refusal of `tefield` (I101's side still carried the pre-I102 raise, which was dropped); `hierarchy.py` merged clean |
| I104 | 5f6b310b | merged (partial); `dynamical_matrix` keeps `sawtooth_ok=True` and `_require_a_measured_sawtooth_regime(calculation)` and takes `_require_a_measured_overlap_metal(calculation, atoms)` in place of `_require_a_moving_overlap_regime` (still used by `nonlinear.py`); the walked and streamed blocks auto-merged with both guards |
| sourcefree (I107 to I109) | 22cb6267 | merged clean (record files from the branch; `22cb6267` descends from the verified `c5401217` and changes only records, a harness line and test comments and docstrings), then **reverted** after the first gate (below); the fix-round records of I108 and I109 kept; re-landed on `t3-sourcefree` (below) |
| I114 | e80bc66f | merged clean (partial) |
| I88 | 325df4ea | not merged (refuted twice) |

After the conflicted merges the touched fast files were run on this workstation (they need no
SCF): `tests/unit/test_realtime.py` 15 passed, 11 deselected (after I101);
`tests/unit/test_sawtooth_response.py` with `tests/unit/test_overlap_metal_phonon.py` 24 passed
(after I104).

**Record corrections applied** (from the verifiers' reports, recovered from their session
transcripts): I90 (the `zstar_eu` column is a gap between methods, wedge against its own closed grid
1.3e-9 to 6.7e-9; the defect argued from its signature, since the split falls with the mesh on the
3m cell; the proposed Limits sentence reworded; the tolerance point was already resolved at the tip,
which tightened the test to the record's 1e-7, 1e-5, 1e-7 after the verifier read `< 2e-5` at
110efefb, unrun); I93 (36^3 and 44^3 rows, the linear kind within 0.035 eV of the fit from 40^3 to
72^3 and between 12.43 and 12.49 eV with the mesh, in the record, the guide text and the slow test's
docstring, which also now says 12.39 to 12.47); I94 (50 and 76 band rows, the `K`-off control, the
ALDA, wedge and scissor numbers); I96 (the band fall steepens, not `nbnd^-1.4`); I101 (the gauge gap
crosses zero between 16 and 20 bohr, -0.08 to -0.10 per cent at 24 bohr; the guide says within about
0.1 per cent from 16 bohr up); I104 (the DFT+U guard is a second, unreachable one behind
`sternheimer.py:1653`; the O z offset varies with the step pair and is largest at `eamp = 0`; NLCC
under `dipfield` and norm-conserving named as admitted unmeasured); I114 (measured against admitted
on the variational argument, in the record and the Limits note; fast count 53; the Pt ultracell at
`Q_d = 0` in place of "no ultracell has run below 1"). Code: `tests/regression/test_tddft_metal.py`'s
Drude identity relative, `< 1e-11 * max|there|` (was absolute 1e-10); the `_drude_block` docstring.
**I99's guide follow-up is open**: `docs/guide/nonlinear.md` (the meta-GGA sentence near line 543 on
the I99 branch) describes a meta-GGA refusal, but `get_nonlinear_spectrum(potential='hxc')` under
tb09 refuses first with "first order only"; it goes with I99's merge.

Guide and README: `piezo.md` (I90), `optics.md` (I93 to I96, with the optional I94 sentence on ALDA,
scissors and the wedge), `nonlinear.md` (I101), `electric-field.md` (I101, I104), `magnetism.md` and
`spirals.md` (I108, I109, then restored by the revert), `anisotropy.md` (I114); README's piezoelectric bullet, the optics
paragraph (a metal with smearing) and the slab-field bullet and table row (phonons at Gamma).
`tools/check_guide.py` passes and `tests/unit/test_facade_documentation.py` 6 passed locally. The
ledger moved I90, I93 to I96 and I101 into "Closed since", marked I104 and I114 partial in place
and updated I88, I99, I107, I108 and I109; `PLAN.md` P156 has the integration-two table, and its
outstanding index is struck for the ultrasoft `chi_0` and the Gamma phonon under `tefield`.

**Fast gate on D22** (worktree `/l/ladovj1/w-inc-t3b`, `tools/test-fast.sh -rs -q` under
`taskset -c 0-3,12-15` in a 16 GB `systemd-run` scope, `DEFUQUMAT_QE_ROOT` set, logs in
`/l/ladovj1/review/incompat/t3b/`). First run at `25aadef9` (`gate1.log`): 3 failed, 4103 passed, 71
skipped, 1833 deselected in 1080.58 s. Two failures are the known `tests/unit/test_check_guide.py`
pair (no `yaml` in the D22 venv); the third is
`tests/unit/test_spin_torque_meta.py::test_the_source_free_field_still_refuses_a_meta_gga`, round
one's I110 test, which asserts that `refuse_source_free` refuses a potential-only meta-GGA, the
refusal I109 lifts by design. The file is I110's, but the merge that broke it is
`incompat/sourcefree`, so that merge was reverted (`git revert -m 1`, commit `9b9d4bcc`), with the
fix-round records of I108 and I109 kept and their guide, ledger and P156 rows undone (`3d715272`).
Re-gate at `3d715272` (`gate.log`): **2 failed (the `yaml` pair), 4096 passed, 71 skipped, 1831
deselected, in 1039.59 s**. The 71 skips are the same set as round one's (compared by name). Landing I108 and I109 needed only
that one test changed to the lifted behaviour, which the next section does.

### The re-landing of sourcefree (I108, I109; I107 stays partial)

On branch `t3-sourcefree` from `incompat-t3` at `008403c7`: `git revert 9b9d4bcc` restores the merge,
and the sourcefree files then equal `22cb6267`'s tree exactly (`git diff` empty for `potential.py`,
`sourcefree.py`, `I108.md`, `I109.md`, the harness and the three test files). The rule that a sibling
test asserting the other's refusal is retargeted was applied to two tests, not one:

- `tests/unit/test_spin_torque_meta.py::test_the_source_free_field_still_refuses_a_meta_gga` is now
  `::test_the_source_free_field_admits_a_meta_gga_and_keeps_its_other_refusals`. It asserts that
  `refuse_source_free` admits a potential-only meta-GGA on a noncollinear cell, and that with a
  meta-GGA it still refuses a PAW dataset and a spin spiral, each by its own message (the two refusals
  I109 did not lift; the collinear refusal is `test_source_free_field.py`'s).
- `tests/regression/test_source_free_run.py::test_a_potential_only_meta_gga_under_nosource_converges_source_free`
  (I109's, slow) asserted `torque_of_result` refuses a meta-GGA, which I110 lifted in integration one.
  It now asserts I110's identity under the lift: the site torques of `torque_of_result` (which rebuilds
  the field with `calculation.source_free` and the run's own `tau`) against `exchange_torque` of the
  SCF's own projected potential, to 1e-12. **That number is unmeasured on this chain**; I110 measured
  5e-18 Ry on its spiral, and the rebuild is from the last iteration's own density and `tau`, so it
  should hold to rounding whatever `conv_thr` is. The forces refusal is kept.

Beside them: the `sourcefree.py` comment no longer lists the torque among the consumers that refuse
(so that file is not byte-equal to `22cb6267` any more), and `I109.md` has a paragraph saying that its
"no torque number exists" sentence is out of date. No grep hit for "potential-only meta-GGA" in a test
asserts the refusal elsewhere. Restored from `3d715272`'s undo: the Limits sentences in
`docs/guide/magnetism.md` and `docs/guide/spirals.md`, the "Closed since" bullets of I108 and I109 in
`notes/INCOMPATIBILITIES.md` (and their T3 entries removed), the P156 table rows and the struck
outstanding index line in `PLAN.md`, plus one P156 sentence on the re-landing. README needs no change
(both records). `tools/check_guide.py` passes; the two fast unit files: 20 passed locally.

**Fast gate on D22** at `9dcdb7e0` (worktree `/l/ladovj1/w-inc-t3c`, `tools/test-fast.sh -q -rs
--tb=line` under `taskset -c 0-3,12-15` in a 16 GB `systemd-run` scope, `DEFUQUMAT_QE_ROOT` set, log
`/l/ladovj1/review/incompat/t3c/gate.log`): **2 failed (the `yaml` pair), 4104 passed, 71 skipped, 1833
deselected, in 1057.68 s**. That is integration two's first gate with the I110 failure gone: the
sourcefree merge adds 8 fast tests and 2 slow ones, and the retarget renames a test rather than adding
one.

**Slow files for the user** (none run, the user's rule):
`tests/regression/test_source_free_run.py` (I108's and I109's new tests, the second with the torque
identity above as its first measurement) and `tests/regression/test_spin_torque_meta_run.py` (I110,
unchanged, the meta torque on the path the lift now reaches).

### Slow files the merges of integration two touch, for the user

None was run (the user's rule).

- `tests/regression/test_piezoelectric.py`, `tests/regression/test_piezoelectric_wedge.py`, `tests/regression/test_streamed_piezo.py` (I90; the wedge test's tolerances tightened at the tip, never run), and as unchanged-branch guards `test_piezoelectric_augmented.py`, `test_piezoelectric_paw.py`
- `tests/regression/test_conductivity.py` (I93, all slow; and the plasma frequency `chi_0` is checked against, I96)
- `tests/regression/test_tddft_ultrasoft.py`, `tests/regression/test_tddft_metal.py` (I94 to I96; the Drude identity's tolerance changed at integration), `tests/regression/test_tddft.py`, `tests/regression/test_spinor_chi0.py`
- `tests/regression/test_realtime_sawtooth.py`, `tests/regression/test_realtime_regimes.py`, `tests/regression/test_realtime.py`, `tests/regression/test_realtime_hierarchy.py` (I101; `run_nonlinear_spectrum` was combined by hand with I102)
- `tests/regression/test_sawtooth_phonons.py`, `tests/regression/test_phonons.py`, `tests/regression/test_streamed_phonons.py`, `tests/regression/test_sawtooth.py` (I104; `dynamical_matrix` was combined by hand with I103 and I86's overlap-metal guard)
- `tests/regression/test_relaxed_anisotropy.py`, `tests/regression/test_anisotropy.py`, and the slow tests of `tests/unit/test_soc_scale.py` (I114)

### D22 notes from the second verifiers, for `docs/d22-notes.local.md`

Not edited into that file.

- **I90**: a harness that uses `multiprocessing` with spawn re-imports its own script in every child, so it must not be overwritten while it runs, even after the parent started (use a new file). At 22:13 on 2026-10-19 the slot pool was empty; four jobs took 38 s (fast tests), 69, 71 and 354 s. Worktrees `/l/ladovj1/w-inc-I90-ver3` (110efefb) and `-ver3-base` (6c5b3a36), logs in `/l/ladovj1/review/incompat/I90-ver3/`.
- **I93**: at 22:30 on 2026-10-19 D22 was idle and eight jobs got slots within 20 s. Al 44^3 on the wedge (2168 k) takes 30 to 53 s per conductivity row, 36^3 18 to 49 s, and `tetrahedra_opt` on 44^3 fits in 6 GB. `-ver3-mf` and `-ver3-md` carry uncommitted mutants and must not be reused as clean trees.
- **chi0**: on al-conductivity at `ecutwfc = 15` the Davidson room check refuses 56 bands (2 nbnd must fit in 97 plane waves), so 48 is the largest; at 48 bands on 8x8x8 the solve falls back from Cholesky to canonical orthogonalisation on 80 of 512 k-points. Worktree `/l/ladovj1/w-inc-chi0-ver2`.
- **I99**: AlAs tb09 at 10 Ry, 27 k on the whole mesh, `'hxc'`: 64, 155, 180 s at dt 0.2, 0.1, 0.05; the 6-operation wedge 23, 42, 73 s. `/l/ladovj1/w-inc-I99-fix-mut` is clean (its mutant is not recoverable); `w-inc-I99-ver2-mut` carries an uncommitted mutant. The local sandbox blocks `sleep N; ssh ...` while `ssh host 'sleep N; ...'` works.
- **I101**: the ref `incompat-I101` on D22 was at fff605f3; the fix commits lived under `incompat-I101-fix*`, and cd14815d was fetched as `incompat-I101-ver2`. At 24 bohr 1x1x2 (29174 plane waves) a T=400 pair takes about 6 min and T=800 about 11; at 16 bohr 2.5 and 3.5 min.
- **I104**: at 00:06 on 2026-10-20 the queue was empty; a CO SCF at `ecutrho = 300`, `conv_thr = 1e-13` takes about 10 s; a `Calculation` with a HUBBARD card on CO about 20 s to build; `slot.sh` passes extra arguments through, so one harness can take a case name.
- **I114**: `tests/data/qe/pt-ultracell-soc.in` is used by no test; its SCF (nbnd 20, `conv_thr = 1e-11`) plus a two-cell spinor ultracell (nbnd 36) converges in about 37 s on one slot with the jaxcache warm. A relaxed-direction SCF of tetragonal Co takes 7 to 12 s. `w-inc-I114-ver` and `-ver-rev` predate the second verifier.
- **I88**: `/l/ladovj1/w-inc-I88-ver2` already existed, so the second verifier used `-ver3` at f458168f; run `git worktree list` before picking a suffix. `CONV=` and `FULL=` exported before `nohup bash slot.sh` reach the python. With 2e-5 in the top band the SCF free energy moves 3.9e-6 Ry between default and full diagonalisation accuracy, since the empty-band threshold `max(5 ethr, 1e-5)` applies to bands below 0.01 occupation.
- **sourcefree**: `incompat-sourcefree` on D22 was two commits behind c5401217, but the object was present, so `git worktree add --detach <path> c5401217` worked without a re-ship. The sandbox refuses an `scp` whose path is a shell variable in a compound command.
- **This integration**: the worktree-isolation guard of an agent session refuses `git -C <other worktree>` and `cd <other worktree> && git ...`, so an integrator cannot work in a worktree it was not started in; and it refuses a `python3 $VAR/script.py` line and inline Python containing `| I88 |`-style table text, so such edits go in a script file run by literal path.

### D22 worktrees that can be removed after integration two (none removed)

Once `incompat-t3` is merged: `/l/ladovj1/w-inc-t3b` (this gate), `/l/ladovj1/w-inc-t3c` (the re-landing's gate), and for the landed units
`/l/ladovj1/w-inc-I90*`, `w-inc-I93*`, `w-inc-chi0*`, `w-inc-I101*`, `w-inc-I104*`,
`w-inc-I114*` and `w-inc-sourcefree*` (I108 and I109 have landed). The mutant trees are removable now: `w-inc-I93-ver3-mf`, `w-inc-I93-ver3-md`,
`w-inc-I99-ver2-mut`. Keep `w-inc-I99*` until I99 is combined with I98, and `w-inc-I88*` until its
next fix round.

### What the user decides, after integration two

- I108 and I109 (landed): the default of `noncollinear_gga` under `nosource`, left at `'kubler'`. Options: keep `'kubler'`, which preserves every `nosource` GGA run made before I108; or switch to `'grad_m'`. The verifier's facts on the PBE h4 chain: `'kubler'` takes 371 steps to a nearly nonmagnetic state, `'grad_m'` 62 to up-up-down-down. For a meta-GGA there is no earlier run to preserve; BJ06 on the chains is 160 against 46 steps, on Ni TB09 22 each.
- I99: combine its `(rho, tau)` state with I98's `(rho, becsum)` in `realtime/selfconsistent.py` by hand (a new session), or leave it recorded.
- I88: a third fix round (an occupation limit below 1e-6 or tied to `small = 1e-12`), or leave it refused.
