# Fixable incompatibilities, ranked by how easy the result is to check

This file is the list of every pair in which a feature runs in one regime and refuses another,
where the missing piece could be written; each entry is one feature times one refused regime (or
one missing term shared by several features), ranked by how easy it would be to *check* the
result once written, not by how much work it is. It was merged on 2026-10-15 from five read-only
surveys of the refusals at commit `0feb4ee` (the reconciled queue of the `notes/` files, and
sweeps of the core, response, workflow and TDDFT paths); where two surveys disagreed on a fact
(the tier, whether a test exists, whether QE refuses), the code or the reference file was opened
and the entry says which way it went. Guard line numbers are those of `0feb4ee` and go stale, so
grep the function name. A test is a `pytest.raises` that fires the guard, named by file, line and
function. Out of scope per `CLAUDE.md` (EXX, Liouville-Lanczos TDDFPT, `CPV/`, `EPW/`, `HP/`,
`GWW/`): nothing in the five surveys touched them. Tiers, top first: **T1** a committed reference
(`tests/data/qe/*`, `test-suite/*/benchmark.out.*`) exercises exactly the pair; **T2** an exact
internal identity; **T3** an approximate identity or one needing a convergence; **T4** a new
`pw.x`, `ph.x` or Elk run; **T5** QE or Elk refuses the pair (the code has the feature and refuses the regime), so no reference
can exist; whatever identity remains is named in the entry, and a refused pair goes to T5 even when
that identity is exact. Otherwise each entry is ranked by the best check it has; the axis "execution mode" marks a layout, storage or solver
choice rather than a physics regime. The QE and Elk facts attributed to "queue.md's lookup"
come from a `reference-source` lookup made for the queue; no Elk tree was read for this file.

## T1: a committed reference exercises exactly the pair

Energies and eigenvalues first, then derivatives.

### I3. Gamma phonon x nonmagnetic spinor (`noncolin`, no moment, with or without `lspinorb`)

- Axis: spinor.
- Guard: `defuqumat/response/sternheimer.py:require_a_sternheimer_regime:1594` (`phonon.py:388` does not pass `noncollinear=True`).
- Test: `tests/unit/test_spinor_refusals.py:104` `test_the_sternheimer_refusal_still_stands`; `tests/regression/test_spinor_response.py:627` `test_what_a_spinor_response_still_refuses`.
- Missing: the caller's assembly on a two-component state; the solve (P81) and the augmented spinor (P98) need no new term (MAGNETISM-NEXT A, HOLES #10).
- Check: derivative: Gamma frequencies in `tests/data/qe/reference.out.ph-alas-epsilon-us-soc` (8 `freq` lines, 376.714126 cm^-1 triplet) and `reference.out.ph-bismuthene-epsilon-us-soc`; beside it T2, silicon as `noncolin` without SOC equals `nspin = 1`.
- QE/Elk: QE y (`dfpt_kernels.f90:212`, `nsolv = 1` without `domag`).
- Partial, 2026-10-16 (P154): written and measured, refusal kept; the discrepancy is in `notes/incompat/I3.md`.

### I5. Linear response with DFT+U at Gamma (dynamical matrix, dielectric tensor, Born charges)

- Axis: DFT+U.
- Guard: `defuqumat/response/sternheimer.py:require_a_sternheimer_regime:1643`; internal duplicate `scf/driver.py:restricted_hamiltonians:4345` (row-subset Hamiltonian, no caller reaches it with a U).
- Test: none fires `sternheimer.py:1643` (`tests/regression/test_scf_solvers.py:778` `test_the_sternheimer_jacobian_refuses_a_hubbard_u_and_a_meta_gga` is the residual solver's own Hubbard refusal, entry on `jvp='sternheimer'`).
- Missing: `adddvhubscf`: the self-consistent `dns` and its `dV`, `dnsbare` (HOLES #16).
- Check: derivative: `test-suite/ph_U_insulator_us/benchmark.out.git.inp=BN.phG.in.args=2` and `ph_U_insulator_paw/...BN.phG.in.args=2` (nspin = 1, fixed, `epsil`); datasets `B/N.pbe-n-rrkjus_psl.1.0.0.UPF` and `B/N.pbe-n-kjpaw_psl.1.0.0.UPF` are in neither `tests/data/pseudo/` nor `$QE/pseudo/`, so a fetch comes first.
- QE/Elk: QE y (`ph.x` refuses U only for non-atomic projectors, `lda_plus_u_kind /= 0`, background, `elph`, `lraman`).
- Partial, 2026-10-16 (P154): written and measured, refusal kept; the discrepancy is in `notes/incompat/I5.md`.

## T2: an exact internal identity, no new reference

Physics regimes first (an `nspin`, spinor, supercell or `nosym` identity), then execution modes with a bit-level identity, then Goldstone, Larmor and the time-shift checks, then the fixed-point and resumed-run identities, and last the identities that need another entry here to land first.

### I10. Strain response, elastic constants, electrostriction, piezoelectric tensor x `nspin = 2`

- Axis: spin.
- Guard: `defuqumat/response/strain.py:_require_one_spin_channel:818`; `response/sternheimer.py:1679` for the unflagged callers (`electrostriction.py:969`, `piezo.py:667`, `workflows/piezo_ladder.py:210`).
- Test: `tests/regression/test_lsda_response.py:640` `test_an_unflagged_quantity_still_refuses_nspin_two`; `tests/regression/test_lsda_response.py:675` `test_the_dynamical_matrix_and_the_strain_response_still_refuse_nspin_two`; `tests/unit/test_piezo_machinery.py:169` `test_the_regimes_this_was_never_run_in_are_refused`.
- Missing: the spin axis on the strain coordinate's `dpsi + ort` block and multipliers; for third derivatives the third derivative of `E_xc^LSDA` in two channels (HOLES #19).
- Check: nonmagnetic `nspin = 2` equals `nspin = 1`; on a magnet T3, a finite difference of the `nspin = 2` stress (pw.x-validated to 2.7e-7); no norm-conserving LSDA insulator is committed.
- QE/Elk: QE n (no strain perturbation); Elk `piezoelt.f90` (task 380) has no spin check per queue.md's lookup.
- Partial, 2026-10-16: not merged (record only, `incompat/strain`); what was measured and what is missing is in `notes/incompat/I10.md`.

### I11. Strain family (strain response, elastic constants, electrostriction, piezo) x nonmagnetic spinor

- Axis: spinor.
- Guard: `defuqumat/response/sternheimer.py:require_a_sternheimer_regime:1594` via `strain.py:289`, `electrostriction.py:969`, `piezo.py:667`; `forces/energy.py:reject_spinors:432` (the elastic constants reach the functional with `spinors=False`).
- Test: `tests/regression/test_spinor_dielectric.py:494` `test_the_opt_in_is_what_lifts_the_refusal`; `tests/unit/test_piezo_machinery.py:169` `test_the_regimes_this_was_never_run_in_are_refused`; `tests/unit/test_spinor_refusals.py:92` `test_energy_at_refuses_a_spinor_unless_asked`.
- Missing: each caller's assembly on a two-component state, and the elastic constants' first-order spinor wavefunctions (MAGNETISM-NEXT A, HOLES #20, #21).
- Check: `noncolin` without SOC equals `nspin = 1`; with SOC, T3, a five-point difference of the spinor energy.
- QE/Elk: neither.
- Partial, 2026-10-16: not merged (record only, `incompat/strain`); see `notes/incompat/I11.md`.

### I16. Collinear DFT+U x a group with time-reversed operations (`t_rev`)

- Axis: DFT+U.
- Guard: `defuqumat/scf/driver.py:Calculation._setup_hubbard:3183`.
- Test: none.
- Missing: the spin-index flip of `new_ns`'s `colin_mag == 2` branch (`hubbard/occupations.py:build_ns_symmetry`); the spinor branch has it (NONCOLLINEAR item 13).
- Check: the same cell with `nosym = .true.`, exactly.
- QE/Elk: QE y (`PW/src/new_ns.f90:168`, `(colin_mag==2) .AND. (t_rev(isym) == 1)`, also `new_nsb.f90:192`, `new_nsg.f90:306`).
- Note: queue.md says QE has no collinear `t_rev`; sweep-core says it does. Read: QE 7.5 has the branch, so a `pw.x` reference (T4) is possible beside the identity.
- Blocked, 2026-10-16: refusal kept, not merged (record only, `incompat/I16`); see `notes/incompat/I16.md`.

### I22. Velocity operator and what is built on it (optical conductivity, effective mass, shift current, SHG) x spin spiral

- Axis: spiral.
- Guard: `defuqumat/response/velocity.py:159`; `response/conductivity.py:require_a_conductivity_regime:412`; `response/effmass.py:330`; `response/photocurrent.py:require_a_velocity_sum_regime:742`.
- Test: `tests/unit/test_conductivity_machinery.py:168` `test_a_spin_spiral_is_refused_for_its_two_spheres`; `tests/unit/test_photocurrent_machinery.py:230` `test_each_refusal_names_its_own_missing_term`; none for velocity and effective mass.
- Missing: `dH/dk` is a pair across the spheres at `k +- q/2` (guard messages).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE n; Elk unread.
- Partial, 2026-10-16 (P154): the operator on a spiral lands, the trace against the supercell 5.5e-10 relative; the conductivity, effective mass, shift current and SHG keep their refusal (`notes/incompat/I22.md`).

### I23. Berry-phase polarization and orbital magnetization x spin spiral

- Axis: spiral.
- Guard: `defuqumat/workflows/polarization.py:run_polarization:112`; `workflows/orbital_magnetization.py:run_orbital_magnetization:116`.
- Test: `tests/unit/test_polarization_machinery.py:217` `test_a_spin_spiral_is_refused_by_name`; none for the orbital magnetization.
- Missing: the overlap between neighbouring k on two spheres is not one gather (guard messages).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE n; Elk unread.
- Blocked, 2026-10-16 (P154): refusals kept and sharpened, each with a test that fires it (`notes/incompat/I23.md`).

### I25. STM image, STS and both tunnelling maps x spin spiral

- Axis: spiral.
- Guard: `defuqumat/workflows/stm.py:_refuse_what_has_no_fermi_level:853` (callers `stm.py:160`, `:347`, `transport.py:196`, `:438`).
- Test: `tests/unit/test_stm_machinery.py:397` `test_a_spin_spiral_is_refused`.
- Missing: `|psi(r)|^2` of a two-sphere state is not the lattice-periodic object the sum builds (guard message).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE n; Elk STM on a spiral unread.
- Partial, 2026-10-16 (P154): the charge image of a spiral lands, 2.9e-9 relative against the supercell; the spin-resolved image, STS and both maps keep their refusal (`notes/incompat/I25.md`).

### I26. Real-time propagation x spin spiral

- Axis: spiral.
- Guard: `defuqumat/realtime/propagate.py:require_a_realtime_regime:190`.
- Test: none.
- Missing: the two spheres at `k +- q/2` under `k + kappa` (guard message).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE out of scope; Elk unread.
- Blocked, 2026-10-16 (P154): refusal kept, now tested (`tests/unit/test_realtime.py::test_a_spin_spiral_is_refused_by_name`; `notes/incompat/I26.md`).

### I29. Source-free exchange-correlation field (`nosource`) x spin spiral

- Axis: spiral.
- Guard: `defuqumat/scf/sourcefree.py:refuse_source_free:175`.
- Test: none.
- Missing: `div B_xc` in the laboratory frame against a rotating-frame magnetization (sweep-core).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE n; Elk unread.
- Blocked, 2026-10-16 (P154): physics, not a missing term; the projection does not commute with the spiral's turn, and the message says so (`notes/incompat/I29.md`). Belongs under "Physically undefined".

### I30. Ultracell x spin spiral

- Axis: spiral.
- Guard: `defuqumat/ultracell/driver.py:require_an_ultracell_regime:545`.
- Test: `tests/regression/test_ultracell.py:1082` `test_every_refusal_fires`.
- Missing: a second modulation of the same states (guard message).
- Check: the explicit supercell SCF of the same spiral and modulation (nested basis).
- QE/Elk: QE n; Elk accepts it untested, per the guard message.
- Blocked, 2026-10-16 (P154): refusal and test unchanged (`notes/incompat/I30.md`).

### I31. Ultracell x gradient-corrected functional

- Axis: functional.
- Guard: `defuqumat/ultracell/potential.py:require_an_ultracell_functional:91`.
- Test: `tests/regression/test_ultracell.py:1109` `test_the_refusals_name_themselves`.
- Missing: the gradient of the ultracell density including the envelope's own gradient, `O(Q)` (guard docstring, PLAN.md P88).
- Check: the explicit supercell SCF of the same modulation, exact by the nested basis (P88).
- QE/Elk: QE n; Elk y (`gndstulr.f90:159` calls `potksu` with GGA, per queue.md's lookup).
- Partial, 2026-10-16 (P154): a GGA on an unpolarized ultracell lands, above the PBE supercell by +4.1e-7 Ry per cell at 48 bands and falling; `nspin = 2` and `4` refuse by name (`notes/incompat/I31.md`).

### I36. Ultracell x a reference converged under a magnetic field or constrained moment

- Axis: magnetic field.
- Guard: `defuqumat/ultracell/driver.py:run_ultracell:1193`.
- Test: `tests/regression/test_ultracell.py:1217` `test_a_field_converged_ground_state_is_refused`.
- Missing: the final field (after `reducebf` or fixed spin moment) carried into `dV` (guard message).
- Check: the explicit supercell SCF of the same modulation, exact by the nested basis (P88), under the same field.
- QE/Elk: QE n; Elk unread.
- Partial, merged 2026-10-16 (P155): a reference under a fixed `B_field` or `LOCAL_MAGNETIC_FIELDS` lands, converging to the supercell under the same field (charge 1.447e-4, moment 1.229e-4, free energy +8.838e-7 Ry above it at 24 bands); feedback schemes, `reducebf < 1`, penalty constraints and a spinor under a field refuse, a design question for the user (`notes/incompat/I36.md`).
- Feedback on the box, merged 2026-10-17 on the user's decision (`incompat/I36-feedback`): `'fsm'` against the 2x1x1 supercell under the same scheme, field +1.32e-5, -9.4e-7, -4.6e-6 Ry and `E - B(M - Mt)` +3.6e-6, +1.1e-6, +6.0e-7 Ry per cell at nbnd 8, 16, 24, where the frozen reference field misses the target by 3.8e-3 mu_B; a reference faded below 1.5e-4 Ry under `reducebf` continues its decay (misses 3.0e-6 to 4.9e-6 mu_B, frozen 5.2e-5). Per-atom schemes, `'secant'`, penalties, `'fsm'` with `reducebf` and a spinor refuse (`notes/incompat/I36-feedback.md`).

### I43. Transverse spin susceptibility (magnons) x symmetry-reduced k-set

- Axis: execution mode.
- Guard: `defuqumat/tddft/spinchi0.py:require_a_transverse_regime:308`.
- Test: none.
- Missing: `X_0(G, G')` symmetrised over the group of `q`, and closure under `+q` (sweep-tddft).
- Check: the whole unshifted grid with `nosym` and `noinv` (an unshifted grid is closed under the point group, P24); `ni-fcc-magnon.in`.
- QE/Elk: QE out of scope.
- Merged with the wedge unit, 2026-10-16 (`incompat/wedge-r2`): blocked, the refusal stands and `tests/unit/test_wedge_refusals.py` now fires it; record in `notes/incompat/I43.md`.

### I44. Clamped-ion elastic constants x symmetry-reduced k-set

- Axis: execution mode.
- Guard: `defuqumat/response/elastic.py:_require_a_closed_grid:412`; `response/electrostriction.py:_require_closed_for_elastic:1217`.
- Test: `tests/regression/test_electrostriction.py:572` `test_the_elastic_constants_are_still_refused_on_a_wedge`.
- Missing: the energy functional symmetrises its own density as a scalar inside the chain rule (guard message).
- Check: the whole unshifted grid with `nosym` and `noinv` (an unshifted grid is closed under the point group, P24).
- QE/Elk: neither.
- Merged with the wedge unit, 2026-10-16 (`incompat/wedge-r2`): blocked, the refusal stands; record in `notes/incompat/I44.md`.

### I45. Phonon at `q != 0` x symmetry-reduced k-set (small group of `q`)

- Axis: execution mode.
- Guard: `defuqumat/response/phononq.py:require_a_two_sphere_regime:1450`.
- Test: `tests/regression/test_phonons_at_q.py:397` `test_a_reduced_k_set_is_refused_by_name`.
- Missing: symmetrisation over the small group of `q` (guard message).
- Check: the whole unshifted grid with `nosym` and `noinv` (an unshifted grid is closed under the point group, P24).
- QE/Elk: QE y.
- Merged with the wedge unit, 2026-10-16 (`incompat/wedge-r2`): blocked, the refusal stands; record in `notes/incompat/I45.md`.

### I47. Vertical, momentum-resolved and ultracell transport x symmetry-reduced k-set

- Axis: execution mode.
- Guard: `defuqumat/workflows/transport.py:_refuse_a_k_set_this_cannot_sum:1413` (callers `transport.py:274`, `:518`, `ultracell.py:329`).
- Test: `tests/regression/test_transport.py:620` `test_a_symmetry_reduced_k_set_is_refused`; `tests/regression/test_ultracell_sts.py:543` `test_a_wedge_is_refused_and_the_whole_grid_is_not`.
- Missing: the subgroup that keeps the exit plane in place, not the whole point group (guard message).
- Check: the `nosym` run of the same cell.
- QE/Elk: QE n.
- Merged with the wedge unit, 2026-10-16 (`incompat/wedge-r2`): partial: the vertical transmission map on a wedge is lifted (implementer's 3.3e-9 on graphene, not reverified), the momentum-resolved map and the ultracell transport still refuse; record in `notes/incompat/I47.md`.

### I48. Force-theorem, torque and relaxed anisotropy off the system's axis x symmetry-reduced k-set

- Axis: execution mode.
- Guard: `defuqumat/workflows/anisotropy.py:_force_theorem:567`; `_require_nosym_to_turn:1027`; `_refuse_relaxed:2130`.
- Test: `tests/regression/test_anisotropy.py:325` `test_turning_the_axis_needs_nosym_at_every_entry_point`; `tests/regression/test_anisotropy.py:327` `test_turning_the_axis_needs_nosym_at_every_entry_point`; `tests/regression/test_anisotropy.py:617` `test_a_direction_other_than_the_system_s_own_needs_nosym`.
- Missing: the magnetic group of each direction; the two directions are sampled on different wedges (sweep-workflows).
- Check: the `nosym` run of the same cell.
- QE/Elk: QE n (no anisotropy workflow).
- Merged with the wedge unit, 2026-10-16 (`incompat/wedge-r2`): partial: the force theorem on a wedge is lifted (implementer's 5.4e-11 meV on tetragonal Co, not reverified), the relaxed route, the torque and the frozen expectation value still refuse; record in `notes/incompat/I48.md`.

### I52. Every `Calculator.get_*` beyond SCF, forces, stress and relaxation, and the response stack x k-point pools

- Axis: execution mode.
- Guard: `defuqumat/calculator.py:guarded:2691` (wrapper on 60 of 64 `get_*`, `POOL_AWARE` at `:2680`); `response/efield.py:574`.
- Test: `tests/unit/test_parallel.py:215` `test_the_facade_refuses_what_is_not_pool_aware`.
- Missing: a pool-aware reduction in each workflow (comment at `calculator.py:2675-2679`).
- Check: one pool against N on the same input; easiest `get_bands`, `get_dos`, `get_nscf`.
- QE/Elk: QE y (pools everywhere).
- Partial, record only, 2026-10-16 (P155): `get_bands`, `get_nscf` and `get_dos` under 2 and 3 pools against 1 pool, eigenvalues 1.3e-14 Ry, DOS 1.9e-12 of its peak, the misplaced-gather guard 0.99 Ry; not merged, since the batch's confirmed list was empty; the other 57 `get_*` refuse (`notes/incompat/I52.md`, branch `incompat/I52-r2` 87f1691).

- Merged 2026-10-16 onto `incompat-t2` on the user's decision; what is refused above still refuses.
### I59. Transverse spin susceptibility (magnons) x noncollinear ground state (the spiral branch is behind it)

- Axis: spinor.
- Guard: `defuqumat/tddft/spinchi0.py:require_a_transverse_regime:254`; the spiral branch `:268` is unreachable as written.
- Test: `tests/unit/test_magnon_machinery.py:199` `test_a_noncollinear_run_is_refused`.
- Missing: the 4x4 spin-density response, which no longer block-diagonalises; the cheaper half is a spiral at its own `q` in the rotating frame (MAGNETISM-NEXT D).
- Check: a collinear moment run as a spinor (no SOC) reproduces the `nspin = 2` magnon; Goldstone; T4 against Elk tasks 330/331.
- QE/Elk: Elk y (`tddftsplr.f90` `ncmag` branches, `genspchi0.f90:147`, per queue.md's lookup).
- Partial, merged 2026-10-16 (P155): a spinor run with its moment along z, no SOC, norm-conserving, against `nspin = 2`, `lambda_max` 1.1e-5, Goldstone 0.005745 against 0.005708; a texture refuses, on the bare `X_0` as well since the integration (`workflows/magnons.py`); spin-orbit coupling, an ultrasoft spinor, a U or a field on a spinor refuse (`notes/incompat/I59.md`).

### I60. Transverse spin susceptibility (magnons) x PAW

- Axis: dataset (PAW).
- Guard: `defuqumat/tddft/spinchi0.py:require_a_transverse_regime:244`.
- Test: `tests/regression/test_ultrasoft_magnon.py:161` `test_paw_is_still_refused`.
- Missing: the one-centre part of `B_xc` in the kernel `B_xc/m`; the matrix element is already augmented (MAGNETISM-NEXT H, D).
- Check: the Goldstone identity `X_0(q=0, w=0) B_xc = m` (`goldstone_residual`; ultrasoft nickel 0.071 to 0.0128 with cutoff, so read at a converged cutoff).
- QE/Elk: QE out of scope; Elk y (LAPW).
- Note: `tddft/spinkernel.py:51-67` still names ultrasoft as a reason for refusal, which is stale since P93.
- Partial, 2026-10-16 (P155): sized, the one-centre kernel (sphere integrals of `B^1/m^1`, all-electron minus pseudo) not written, refusal kept and naming the channel; the stale `tddft/spinkernel.py` text is corrected by the magnon merge (`notes/incompat/I60.md`).

### I62. Transverse spin susceptibility (magnons) x `nosource`

- Axis: spinor.
- Guard: `defuqumat/tddft/spinchi0.py:require_a_transverse_regime:300`.
- Test: none.
- Missing: a tensor kernel, since the projection makes `B_xc` not parallel to `m` (guard comment).
- Check: the Goldstone identity `X_0(q=0, w=0) B_xc = m` (`goldstone_residual`; ultrasoft nickel 0.071 to 0.0128 with cutoff, so read at a converged cutoff).
- QE/Elk: QE out of scope; Elk unread.
- Blocked, 2026-10-16 (P155): the guard now fires for its own reason; the projection gives a z-collinear spinor transverse components, so it needs I59's texture response and a tensor kernel (`notes/incompat/I62.md`).

### I66. Checkpoint x real-time run with the potential updated

- Axis: execution mode.
- Guard: `defuqumat/realtime/propagate.py:propagate:1157`.
- Test: `tests/unit/test_realtime.py:362` `test_what_the_updated_potential_refuses_is_refused_by_name`.
- Missing: saving every state and two potentials (guard message).
- Check: resumed against uninterrupted, bit for bit.
- QE/Elk: QE out of scope.
- Not merged, 2026-10-16 (P155): the batch lists it refuted (the first fix's digest was blind to `k_batch`); the third verifier round's logs at 592240e (`/l/ladovj1/review/incompat/I66-ver3/hA.log`) read the resume bit for bit and a file of another `k_batch` ignored, a decision for the user; record only (`notes/incompat/I66.md`, branch `incompat/I66-r2` 9d58999).

### I77. Ultracell x `mixing_mode = 'robust'` and `'ldos'`

- Axis: execution mode.
- Guard: `defuqumat/ultracell/driver.py:run_ultracell:1087`; `:1093`.
- Test: `tests/unit/test_ldos_mixing.py:352` `test_places_without_an_ldos_refuse_the_mode` (`ldos`); none for `robust`.
- Missing: the robust supervisor lives in `run_scf`'s loop; an LDOS handed to the mixer (guard messages).
- Check: the ultracell fixed point does not depend on the mixer.
- QE/Elk: QE n.
- Partial, record only 2026-10-16 (P155, integration two): `'ldos'` in the ultracell lands on the branch `incompat/I77` (91e35a2), the ultracell fixed point under `'ldos'` against `'anderson'` with Kerker 1.19e-7 (`nspin = 1`) and 3.47e-7 (`nspin = 2`) in the density, 3.4e-13 and 2.1e-12 Ry; the box LDOS 27.528 against the Gaussian sum 27.542. `'robust'` is not written and keeps the refusal. Not merged: the batch's confirmed list is empty for a partial unit (`notes/incompat/I77.md`).

- Merged 2026-10-16 onto `incompat-t2` on the user's decision; what is refused above still refuses.
### I79. Gamma phonon x magnetic spinor (`noncolin` with `domag`), with the axial symmetrisation of a displacement response

- Axis: spinor.
- Guard: `defuqumat/response/sternheimer.py:require_a_sternheimer_regime:1594`; `scf/driver.py:Calculation.symmetrize_atom_displacement:5328`.
- Test: as the nonmagnetic spinor phonon; none for `:5328` (the `axial vector` match in `test_conductivity_machinery.py:180` is the conductivity's).
- Missing: everything of the nonmagnetic spinor phonon plus the second linear system for the time-reversed partners (`nsolv = 2`, Urru and Dal Corso), and for ultrasoft `int3_nc_save`, `dbecsum_aux` (HOLES #17); the axial rotation with `det(R)` and the `t_rev` sign.
- Check: after the `nspin = 2` Gamma phonon: a collinear `nspin = 2` dynamical matrix against the same cell as a spinor with the moment along z; `test-suite/ph_Ni_nc_spinorbit_mag` is at `q != 0`, a metal, with an uncommitted dataset.
- QE/Elk: QE y for phonons (`dfpt_kernels.f90:212-213`); `phq_readin.f90:546-548` refuses `epsil`, `zue`, `lraman`, `elop` under `magnetic_sym`, which sweep-response reads as never firing because `magnetic_sym` is assigned later (`phq_setup.f90:211`, see the textured-spinor dielectric entry); PAW refused at `:807`.
- Partial, record only 2026-10-16 (P155, integration two): the collinear limit, a spinor along z without SOC against `nspin = 2`, reads 8.3e-6 Ry/bohr^2 on triplet O2 (force constants of 1.68) and 1.9e-6 on `h-chain-afm`; on a shared ground state 2.3e-6 and 1.4e-6, so about 1e-6 to 2e-6 belongs to the spinor response route and is unlocated (the nonmagnetic spinor identity of I3 reaches 2.7e-10). The phonon refusal stays. The axial displacement average at `nspin_mag = 4` (atom sum against `symmetrize_directional` 7.1e-15) and the axial `becsum` average are on the branch `incompat/I79` (b6ff6d0), not merged (`notes/incompat/I79.md`).

- Merged 2026-10-16 onto `incompat-t2` on the user's decision; what is refused above still refuses.
### I81. Phonon at `q != 0` x ultrasoft or PAW (and the electron-phonon coupling on it)

- Axis: dataset (US/PAW).
- Guard: `defuqumat/response/phononq.py:require_a_two_sphere_regime:1413`.
- Test: none.
- Missing: four terms: `addusddens`'s `Q_ij(q+G)` in the response density, `adddvscf`'s `int3` with `e^{iqr}`, `dvanqq`'s `int1`/`int2`, the multipliers at `q != 0`; `q_ij(q+G)` exists in `tddft.spinchi0.augmentation_factors` (PLAN.md index P97, AUGMENTATION-NEXT §1c).
- Check: a commensurate `q` equals the Gamma phonon of the supercell (ultrasoft and PAW insulators run at Gamma); T1 only stacked with `nspin = 2` and a metal (`test-suite/ph_base/ni.phX.in`, `nipaw.phX.in`).
- QE/Elk: QE y, including `elphon.f90:79-82`.
- Partial, record only 2026-10-16 (P155, integration two): the refusal stays, and the entry undercounts. Beyond the four terms, the `q` route lacks P39's orthogonality set at `q` (`dpsi^ort`, `drhous`, `becsumort`, `dLambda`, `drhodvus`, the `deff`/`int1`/`int2` half of `drhodvnl`), cross-sphere objects (`compute_drhous.f90:99-114`), and the frozen Hessian is not atom-diagonal for an augmented dataset (`addusdynmat.f90:141-195` through `dvanqq.f90:106-191`), so `frozen_force_constants`' claim that it is the same at every `q` holds for norm-conserving datasets only (`notes/incompat/I81.md`).

- Merged 2026-10-16 onto `incompat-t2` on the user's decision; what is refused above still refuses.
### I83. Phonon at `q != 0` x DFT+U

- Axis: DFT+U.
- Guard: `defuqumat/response/phononq.py:require_a_two_sphere_regime:1477`.
- Test: none.
- Missing: `dnsq_bare`, `dnsq_scf` (HOLES #16).
- Check: after the DFT+U Gamma response: supercell identity; T1 stacked with ultrasoft, `ph_U_insulator_us/BN.phq.in`, `ph_U_insulator_paw/BN.phq.in` (datasets need a fetch).
- QE/Elk: QE y.

### I84. Phonon at `q != 0` and any Sternheimer response x spin spiral

- Axis: spiral.
- Guard: `defuqumat/response/phononq.py:require_a_two_sphere_regime:1459`; `response/sternheimer.py:require_a_sternheimer_regime:1674`.
- Test: none.
- Missing: four shifted spheres (sweep-response).
- Check: after the magnetic spinor Gamma phonon: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE n.
- Skipped 2026-10-16 (P155, integration two): its prerequisite, the magnetic spinor Gamma phonon (I79), is partial.

## T3: an approximate identity, or one that needs a convergence

### I91. Shift current x ultrasoft or PAW

- Axis: dataset (US/PAW).
- Guard: `defuqumat/response/photocurrent.py:require_a_shift_current_regime:712`.
- Test: `tests/unit/test_photocurrent_machinery.py:230` `test_each_refusal_names_its_own_missing_term`.
- Missing: the k-derivative of the augmentation dipole, hence `d^2(beta)/dk_a dk_b` of a bare projector (PLAN.md P47, P94, P99, AUGMENTATION-NEXT §2).
- Check: a central difference in k of `augmentation_connection`; the norm-conserving limit; SHG on the same states for the dipole.
- QE/Elk: QE n; Elk unread.
- Partial 2026-10-19 (P156), record in `notes/incompat/I91.md`: the new augmentation term takes the complete-basis sum-rule miss from 5.29e-2 to 4.1e-3, flat in the step, against a norm-conserving control falling as the step squared, so a piece is still missing; the refusal stays and names the numbers. The verifier found some numbers hold only at the one k-point and pair used.

### I104. Any Sternheimer response x saw-tooth or Berry-phase field (`tefield`, `lelfield`)

- Axis: applied electric field.
- Guard: `defuqumat/response/sternheimer.py:require_a_sternheimer_regime:1565` via `scf/sawtooth.py:refuse_sawtooth`.
- Test: `tests/unit/test_sawtooth.py:245` `test_gate_is_refused`.
- Missing: unchecked against any reference (sweep-response).
- Check: a finite difference of the forces under the field.
- QE/Elk: unread.
- Partial 2026-10-20 (P156, integration two), merged for the Gamma phonon, record in `notes/incompat/I104.md`: the dynamical matrix at Gamma of a fixed-occupation `nspin = 1` insulator under `tefield`, with or without `dipfield`, against a central difference of the forces to about 2e-4 Ry/bohr^2 (the force difference's roughness); 0.12 with the ion-dipole term off. Every other Sternheimer response keeps the saw-tooth refusal, and `lelfield` stays refused everywhere. PAW is refused by name; the unshifted k-mesh, a norm-conserving dataset and a core charge under `dipfield` are admitted unmeasured.

### I105. Dielectric tensor and Born charges x textured spinor (`nspin_mag = 4` with a magnetization)

- Axis: spinor (magnetic).
- Guard: `defuqumat/response/efield.py:require_a_measured_spinor_response:1306`.
- Test: `tests/regression/test_spinor_dielectric.py:332` `test_a_textured_spinor_is_refused_by_name`.
- Missing: unlocated: on `i-atom-soc.in` the component along the moment is 1.574482 against `ph.x`'s 1.494594 (5.3 per cent), across 4.6e-4; P104 cleared the magnetic kernel (0.238 per cent), leaving the field source term and `ph.x` itself (MAGNETISM-NEXT A2).
- Check: a finite-field difference of the dipole on the same atom, independent of both codes (whether `tefield` runs on a spinor is not checked; if not, T4).
- QE/Elk: QE ran it, but `phq_readin.f90:546` checks `magnetic_sym` before `phq_setup.f90:211` assigns it, so its own refusal never fires and its number is a suspect.
- Note: queue.md calls it T1; there is no `reference.out.ph-i-atom-soc*` in `tests/data/qe/`, the `ph.x` numbers are only in `notes/MAGNETISM-NEXT.md:180`. sweep-response is right.
- Partial 2026-10-19 (P156), record in `notes/incompat/I105.md`: along the moment the Berry-phase finite field minus the Sternheimer `eps_zz` falls to an unlocated offset of about -5e-4 (0.16 per cent of eps - 1), the Born charges are unchecked, and the refusal stays. Lifting it for epsilon with `born_charges=False` is the user's decision.

### I106. Energy derivatives under a field or constrained moment: forces, stress, `dE/dq`, force-theorem and relaxed anisotropy

- Axis: magnetic field.
- Guard: `defuqumat/forces/energy.py:reject_magnetic_field:545`; `stress/energy.py:require_a_differentiable_cell:122`; `forces/spiral.py:_require_a_differentiable_spiral:600`; `workflows/anisotropy.py:_refuse_system:445`, `_refuse_relaxed:2097`.
- Test: `tests/regression/test_spiral_relaxation.py:270` `test_a_gradient_is_refused_where_it_would_be_silently_wrong`; `tests/unit/test_sibling_refusals.py:254` `test_every_force_method_passes_the_same_guards` reads the source.
- Missing: the field or penalty energy inside the differentiated functional; it is deliberately outside the reported total (`scf/fields.py`) (guard messages).
- Check: a finite difference of the total including the penalty.
- QE/Elk: QE not a reference: `add_bfield.f90` prints `etcon` and leaves it out of `etot` (`electrons.f90:1133-1163`), `forces.f90` has no constraint term.
- Partial 2026-10-19 (P156), record in `notes/incompat/I106.md`: the field term of the force on ultrasoft O2 is measured (falling as h^2 against 8.6e-4 off), the ultrasoft Fe stress sits at a floor of 4e-4 to 6e-4 Ry from the existing strain derivative; the lifted guard admits PAW and noncollinear regimes nothing tests yet.

### I107. Source-free exchange-correlation field x PAW

- Axis: dataset (PAW).
- Guard: `defuqumat/scf/sourcefree.py:refuse_source_free:147`.
- Test: `tests/regression/test_source_free_run.py:160` `test_paw_is_refused`.
- Missing: one Poisson equation for `div B` across the smooth and the one-centre parts (AUGMENTATION-NEXT §1f).
- Check: `div B_xc = 0` after projection on the combined field; the norm-conserving limit; T4 Elk `nosource`.
- QE/Elk: QE n; Elk y (LAPW).
- Partial 2026-10-19 (P156), record in `notes/incompat/I107.md`: sized, nothing lifted (the one-centre divergence, its multipoles and one Poisson solve are missing); the guard and the argument confirmed by the verifier 2026-10-20 (`/l/ladovj1/review/incompat/sourcefree-ver2/`).

### I112. `rotate_moments` x DFT+U

- Axis: DFT+U.
- Guard: `defuqumat/scf/driver.py:_refuse_rotating_moments:6945`.
- Test: none.
- Missing: turning `ns`'s spin structure with the step (sweep-core).
- Check: the minimum of the relaxed anisotropy, which accepts a U.
- QE/Elk: QE n.
- Partial 2026-10-20 (P156, integration three), merged, record in `notes/incompat/I112.md`: the lift is kept for a U on norm-conserving and ultrasoft datasets. The torque with a U against the finite-difference derivative of the frozen band free energy on tetragonal Co at `U = 1` eV, -1.50e-5 and -1.55e-5 relative at a 0.5 degree stencil, falling as h^2 (+0.52 with the Hubbard share left out, which carries 52 per cent of the torque); the end state is a fixed point of the plain loop and a minimum on its branch of `ns`. The entry's check, the minimum of the relaxed anisotropy, is not met: the cell has six self-consistent or descending states spanning 1.58e-3 Ry, so the run finds the easy orientation of the `ns` solution it lands on and not the lowest. The user's decision, taken by Fable while the user was away: keep the lift, warn at the end of a run when `|M|` moved by more than 1e-3 mu_B (`test_a_rotation_with_a_hubbard_u_warns_when_its_moment_moves`), and say in the guide to run from two or three orientations. PAW and a fixed tensor moment still refuse (`test_the_sibling_refusals_of_rotate_moments_still_fire_on_the_turning_group`).

### I114. `soc_scale` strictly between 0 and 1 x ultrasoft or PAW

- Axis: dataset (US/PAW).
- Guard: `defuqumat/scf/driver.py:Calculation.__init__:2670`; `pseudo/spinorbit.py:__init__:300`.
- Test: `tests/unit/test_spiral_soc.py:176` `test_an_intermediate_soc_scale_on_a_norm_conserving_dataset_is_the_blend`; `tests/regression/test_anisotropy.py:402` `test_an_intermediate_soc_scale_is_refused_on_an_augmented_dataset`.
- Missing: a blend of the spin-orbit overlap `qq_so`; only 0 and 1 are defined (guard message).
- Check: the two endpoints, and the shape in `soc_scale` against the norm-conserving blend.
- QE/Elk: QE n (no `soc_scale`).
- Note: Only in sweep-core.
- Partial 2026-10-20 (P156, integration two), merged for ultrasoft, record in `notes/incompat/I114.md`: `soc_scale` strictly between 0 and 1 on an ultrasoft dataset; tetragonal Co, the Richardson slope 0.010121 against Hellmann-Feynman 0.010135 meV, the verifier's `s = 0.37` coefficient 0.45804 against the law's 0.4585 (0.154 against 0.0634 meV with the term off), MAE(1) 0.447302 against `pw.x` 0.4474, a Pt ultracell at 0.5 to 2.3e-12 Ry at `Q_d = 0`. Forces, stress, phonons and the dielectric response at an intermediate scale rest on the variational argument. Fully relativistic PAW stays refused (its small component is scaled as a strength, P117).

### I116. Ultracell `seed_from_field` x collinear reference

- Axis: spin.
- Guard: `defuqumat/ultracell/driver.py:run_ultracell:1161`.
- Test: none.
- Missing: an amplitude or staggered seed from a scalar field, not measured (guard message).
- Check: the converged state does not depend on the seed.
- QE/Elk: QE n.

### I117. Ultracell noncollinear seed x compensated (antiferromagnetic) reference

- Axis: spinor.
- Guard: `defuqumat/ultracell/seed.py:reference_axis:226`.
- Test: `tests/unit/test_ultracell_seed.py:260` `test_a_compensated_reference_has_no_direction_to_turn_from`.
- Missing: a per-site or staggered reference axis; the net moment is round-off (guard message).
- Check: the converged state against the explicit supercell.
- QE/Elk: QE n.

### I118. STM image, STS and tunnelling maps x applied magnetic field

- Axis: magnetic field.
- Guard: `defuqumat/workflows/stm.py:_refuse_what_has_no_fermi_level:865`.
- Test: `tests/unit/test_stm_machinery.py:408` `test_an_applied_field_is_refused`.
- Missing: the tip energy: the field's energy is outside the reported total, so the Fermi level is not the field-free one (guard message).
- Check: the weak-field image tends to the field-free one.
- QE/Elk: QE n.

### I119. Single-precision bands x ultrasoft or PAW, noncollinear, spiral, DFT+U, meta-GGA

- Axis: execution mode.
- Guard: `defuqumat/scf/driver.py:_require_band_precision_supported:3065` (one raise, five reasons).
- Test: `tests/unit/test_band_precision.py:88` `test_what_it_has_not_been_checked_on_is_refused`.
- Missing: a single-precision check per regime (docstring `:3030-3036`).
- Check: the double-precision state, to `conv_thr_floor`.
- QE/Elk: QE n.

### I120. Sternheimer Jacobian (`jvp='sternheimer'`) x DFT+U, meta-GGA, magnetic noncollinear, streamed store

- Axis: execution mode.
- Guard: `defuqumat/scf/residual.py:ScfResidual._refuse_sternheimer:307`, :313, :319, :325.
- Test: `tests/regression/test_scf_solvers.py:778` `test_the_sternheimer_jacobian_refuses_a_hubbard_u_and_a_meta_gga`; `tests/regression/test_scf_solvers.py:781` `test_the_sternheimer_jacobian_refuses_a_hubbard_u_and_a_meta_gga`.
- Missing: `adddvhubscf` (shared with the DFT+U Gamma response); a kinetic-like `dH/dtau`; a measurement only for magnetic noncollinear; a derivative of `stream_densities` (sweep-core).
- Check: a central difference of `F`, and `jvp='finite'`.
- QE/Elk: QE n.

### I121. Spiral `dE/dq` x tabulated cross augmentation

- Axis: execution mode.
- Guard: `defuqumat/scf/driver.py:Calculation.at_spiral_q:4569`.
- Test: `tests/regression/test_spiral_relaxation.py:733` `test_a_tabulated_augmentation_table_is_refused`; `tests/regression/test_spin_spirals.py:310` `test_what_a_spiral_refuses`.
- Missing: the radial table's extent is set from `|q|` on the host; the exact path works, the table is the memory fallback (`DEFUQUMAT_AUG_MAX_BYTES`) (sweep-core).
- Check: the untabulated path, up to the table's interpolation error; a finite difference in `q`.
- QE/Elk: QE n.

### I122. Spin spiral x an FFT box that folds the displaced sphere

- Axis: execution mode.
- Guard: `defuqumat/basis/gvectors.py:_require_the_box_holds:309`.
- Test: `tests/unit/test_spiral_gvectors.py:98` `test_a_box_that_would_fold_two_members_together_is_refused`.
- Missing: a box one FFT size larger along the folded axis (sweep-core).
- Check: a cutoff nudged so the box rounds up (the grid changes, so not exact).
- QE/Elk: QE n.

## T4: needs a new `pw.x`, `ph.x` or Elk run

`pw.x` and `ph.x` are built on D22 (`/l/ladovj1/qe-7.5/bin/pw.x`, `/l/ladovj1/review/timing/`), Elk under `review/harmonics-elk/bin`.

### I123. Fully relativistic dataset x a run without spin-orbit coupling (`average_pp`)

- Axis: dataset (fully relativistic).
- Guard: `defuqumat/scf/driver.py:Calculation.__init__:2464`; downstream `projwfc/channels.py:213`, `pseudo/atomic.py:spinor_orbital_blocks:335`.
- Test: `tests/regression/test_spinorbit.py:334` `test_a_relativistic_dataset_without_lspinorb_is_refused`; `tests/unit/test_spinorbit_coefficients.py:382` `test_a_relativistic_dataset_without_spin_orbit_is_refused`; `tests/unit/test_spinor_projection.py:176` `test_the_label_path_refuses_a_relativistic_dataset_without_lspinorb`; `tests/unit/test_spinor_projection.py:196` `test_a_relativistic_dataset_without_lspinorb_is_refused`.
- Missing: `average_pp.f90`'s radial refit of the two `j` channels into one projector with `sqrt(D)` weights (HOLES #11); unaveraged it is 20 Ry off on rhombohedral BN; for the projection, QE's mixed dispatch (orbitals on `has_so`, labels on `lspinorb`).
- Check: energy and eigenvalues: `pw.x` on `I.rel-pbe-nc-dojo.UPF` or `Ni.rel-pbe-nc-dojo.UPF` (committed) at `nspin = 1`; `projwfc.x` for the channels; no test-suite input uses a `rel-` file without `lspinorb`.
- QE/Elk: QE y for norm-conserving without U (`setup.f90:319`); `average_pp.f90:34` refuses `tvanp` and `lda_plus_u`, so those halves are T5.

### I124. Meta-GGA x DFT+U

- Axis: functional.
- Guard: `defuqumat/scf/driver.py:Calculation._require_meta_supported:3129`.
- Test: none.
- Missing: none in the physics; the residual solver reads `ns` off the end of the packed state where `tau` now sits; validation (GAPS §3, sweep-core).
- Check: energy: `pw.x` with libxc `tb09` and a U; beside it, mixer and residual solver landing on one state.
- QE/Elk: QE: no `errore` for the pair in `setup.f90` or `input.f90`.

### I125. Berry-phase finite field (`lelfield`) x DFT+U

- Axis: DFT+U.
- Guard: `defuqumat/scf/berryfield.py:require_a_berry_field_regime:353`.
- Test: none.
- Missing: unread (reason "never run with the field here").
- Check: `pw.x` with `lelfield` and a U.
- QE/Elk: QE: not refused and not referenced in `c_phase_field.f90`, so unverified.

### I126. Berry-phase finite field (`lelfield`) x `tefield` at the same time

- Axis: applied electric field.
- Guard: `defuqumat/scf/berryfield.py:require_a_berry_field_regime:353`.
- Test: none.
- Missing: unread.
- Check: `pw.x` with both.
- QE/Elk: unread.

### I127. Berry-phase finite field (`lelfield`) with `a_gdir` off the Cartesian axis `gdir`

- Axis: applied electric field.
- Guard: `defuqumat/scf/berryfield.py:require_a_berry_field_regime:353`.
- Test: none.
- Missing: a consistent ionic and electronic dipole direction (sweep-core).
- Check: `pw.x`, with the caveat that `pw.x` mixes the two conventions and its number is the one in question.
- QE/Elk: QE y, convention in question.

### I128. LSDA screening kernel x GGA where a channel density empties

- Axis: functional.
- Guard: `defuqumat/response/efield.py:_require_a_finite_kernel:759` (raise near `:833`, fires on a NaN kernel).
- Test: none (fires on data).
- Missing: `dgcxc_spin`'s own `rho_threshold_gga` and `zeta` gates; the LDA half is closed (`_fully_polarized`, 5.1e-6 against `ph.x`); the PAW one-centre `dmxc` (`paw_onecenter.f90:1792`) has no finiteness check (guard docstring).
- Check: derivative: `ph.x` `epsil` on `o2-fixed-lsda.in` with a PBE oxygen dataset (`O.pbe-rrkjus.UPF`, `O.pbe-kjpaw.UPF` committed).
- QE/Elk: QE y (`dgcxc_spin` in `XClib`).
- Note: sweep-response calls it T1 on `ph_insulator_paw_magn` and `reference.out.ph-o2-fixed-lsda`; both are LDA (`O2.scf.2.in` uses `O.pz-kjpaw.UPF`), so they test only the closed half. queue.md is right.

### I129. Any Sternheimer response x tetrahedra

- Axis: occupations (tetrahedra).
- Guard: `defuqumat/response/sternheimer.py:require_a_sternheimer_regime:1654`.
- Test: `tests/regression/test_response.py:650` `test_the_regimes_without_a_response_here_are_refused_by_name`.
- Missing: `dfpt_tetra_beta`, a response weight per band pair (guard message).
- Check: derivative: `ph.x` on `tests/data/qe/al-tetrahedra.in`; no committed tetrahedron `ph.x` case.
- QE/Elk: QE y (`LR_Modules/dfpt_tetra_mod.f90`; only `tetra_type = 0` refused, `phq_readin.f90:968`).

### I130. Sum-over-states `chi_0` x DFT+U

- Axis: DFT+U.
- Guard: `defuqumat/tddft/chi0.py:require_a_sum_over_states_regime:358`.
- Test: none.
- Missing: the Hubbard response in the screening (`adddvhubscf`) (guard message).
- Check: the static limit against a new norm-conserving `ph.x` `epsil` run with a U; after the DFT+U Gamma response lands, against the code's own Sternheimer (T3).
- QE/Elk: QE out of scope (TDDFPT); `ph.x` gives the static limit.

## T5: QE or Elk refuses the pair

Only identities remain; the entry names the one available.

### I131. Force-theorem anisotropy x DFT+U

- Axis: DFT+U.
- Guard: `defuqumat/workflows/anisotropy.py:_refuse_system:429`.
- Test: none.
- Missing: `ns` is a property of the wavefunctions and is not in the handoff (sweep-workflows).
- Check: the relaxed anisotropy, which accepts a U, to first order in the coupling.
- QE/Elk: QE refuses: `average_pp.f90:34` refuses `lda_plus_u` on this path.

### I132. Any Sternheimer response x applied field or constrained moment

- Axis: magnetic field.
- Guard: `defuqumat/response/sternheimer.py:require_a_sternheimer_regime:1629`.
- Test: `tests/unit/test_sibling_refusals.py:380` `test_the_sternheimer_stack_refuses_a_magnetic_field`; `tests/unit/test_sibling_refusals.py:406` `test_the_field_refusal_fires_before_the_regime_flags`.
- Missing: for an applied field, plumbing: `SCFResult.magnetic_field` and `.field_scale` through `make_sternheimer` and the `.potential(` sites; for a fixed spin moment, `dB/drho` of a feedback update, which is not a derivative and has to be written (GAPS §3, OPEN.md A2).
- Check: a central difference of two converged densities under the field (P104's longitudinal susceptibility, 0.238 per cent).
- QE/Elk: QE refuses both: `input.f90:1620` sets `i_cons = 4` for `B_field` and `phq_readin.f90:850` refuses `i_cons /= 0`.
- Note: queue.md splits it (applied field T3, constrained T4) and lists it in its top ten; the `i_cons = 4` line puts both halves at T5.

### I133. Raman tensor x spinor, `nspin = 2`, DFT+U

- Axis: spinor, spin, DFT+U.
- Guard: `defuqumat/response/nonlinear.py:462` via `sternheimer.py:require_a_sternheimer_regime:1594`, `:1679`, `:1643`.
- Test: `tests/regression/test_lsda_response.py:640` `test_an_unflagged_quantity_still_refuses_nspin_two`; `tests/regression/test_spinor_dielectric.py:494` `test_the_opt_in_is_what_lifts_the_refusal`.
- Missing: the assembly with a spin axis or two components; for LSDA the third derivative of `E_xc^LSDA` (HOLES #19).
- Check: a finite difference of the dielectric tensor over displaced cells (validated for the nonmagnetic spinor and LSDA insulators).
- QE/Elk: QE refuses: `phq_readin.f90:817-818` (`lraman` with `noncolin`), `PHonon/PH/raman.f90:28` (spin-polarized), `phq_readin.f90:775` (U).

### I134. Any Sternheimer response x constrained `tot_magnetization` with a smearing

- Axis: spin.
- Guard: `defuqumat/response/sternheimer.py:require_a_sternheimer_regime:1694`.
- Test: `tests/regression/test_lsda_response.py:619` `test_two_fermi_levels_are_refused_by_name`.
- Missing: a spin axis on `Smearing.ef` (sweep-response).
- Check: the two levels made equal reproduce the shared-level response.
- QE/Elk: QE refuses: `phq_readin.f90:854-855` (two Fermi energies with a smearing).
- Note: queue.md excludes it (HOLES "Rejected"); sweep-response keeps it. Two Fermi levels with a smearing is a defined state, so it is kept, at T5.

### I135. Orbital magnetization x ultrasoft or PAW

- Axis: dataset (US/PAW).
- Guard: `defuqumat/workflows/orbital_magnetization.py:run_orbital_magnetization:123`.
- Test: none.
- Missing: `q_ij(b)` in the neighbour overlap and duals that are dual in the `S` metric (sweep-workflows).
- Check: the norm-conserving limit (no exact identity found).
- QE/Elk: QE refuses: `setup.f90:130`, "Orbital Magnetization not implemented with USPP/PAW".

### I136. Stress x saw-tooth or Berry-phase field

- Axis: applied electric field.
- Guard: `defuqumat/stress/energy.py:109` via `refuse_sawtooth`.
- Test: none.
- Missing: the field's stress terms (sweep-response).
- Check: a finite difference of the energy under the field over strained cells.
- QE/Elk: QE refuses: `input.f90:704` switches the stress off for `tefield` and `gate`, its field stress terms are commented out.

### I137. Field response (dielectric tensor, Born charges, Raman, strain, piezo) x gamma-only storage

- Axis: execution mode.
- Guard: `defuqumat/response/sternheimer.py:require_a_sternheimer_regime:1573` (the phonon passes `gamma_ok`).
- Test: `tests/regression/test_gamma_only.py:359` `test_a_field_response_is_still_refused_under_gamma`; `tests/unit/test_sibling_refusals.py:467` `test_two_more_consumers_refuse_gamma_storage`.
- Missing: the field source term's plane-wave sums (`dvpsi_e` and what is built on it) under the 2 Re rule; with the guard lifted Si gives 501.7/213.1/253.1 against 190.8 (PLAN.md).
- Check: the same cell with an explicit `K_POINTS automatic 1 1 1 0 0 0` on the whole sphere.
- QE/Elk: QE refuses in `ph.x` (`phq_readin.f90:757`, `gamma_only`; `phcg.x` is the gamma code), so the identity is the only check.

### I138. Berry-phase finite field (`lelfield`) x k-point pools

- Axis: execution mode.
- Guard: `defuqumat/scf/driver.py:run_scf:8294`; `forces/chunked.py:101`.
- Test: none.
- Missing: unread.
- Check: one pool against N.
- QE/Elk: QE refuses: `setup.f90:722-723` (`npool > 1` with `lelfield`); the pool identity is exact.

### I139. First-order spin-orbit energy of a spiral x ultrasoft or PAW

- Axis: dataset (US/PAW).
- Guard: `defuqumat/workflows/spiral_soc.py:_refuse:323`.
- Test: `tests/unit/test_spiral_soc.py:214` `test_an_augmented_dataset_is_refused_by_name`.
- Missing: which blocks of `newd_so`'s `F B F` keep a component on its own sphere; PAW adds the one-centre small component (PLAN.md P123, SPIRAL-SOC-NEXT).
- Check: the norm-conserving limit; the supercell with `lspinorb` at small coupling needs the fractional `soc_scale` above.
- QE/Elk: QE supercell only; Elk drops `spinorb` under `spinsprl` (`init0.f90:106-108`, per queue.md's lookup).

### I140. Ultracell x symmetry

- Axis: execution mode.
- Guard: `defuqumat/ultracell/driver.py:require_an_ultracell_regime:552`.
- Test: `tests/regression/test_ultracell.py:1082` `test_every_refusal_fires`; `tests/regression/test_ultracell.py:1239` `test_a_symmetric_run_is_refused`.
- Missing: the ultracell's own group, or re-expanding a reference converged with symmetry on the whole folded set (guard message).
- Check: the `nosym` run.
- QE/Elk: QE n (no ultracell); Elk refuses: it forces `reducek = 0`, per the guard message.

### I141. Magnons at a `q` that is not a difference of two k-points

- Axis: execution mode.
- Guard: `defuqumat/tddft/spinchi0.py:commensurate_shift:347`.
- Test: `tests/unit/test_magnon_machinery.py:175` `test_a_wavevector_off_the_grid_is_refused`.
- Missing: a second fixed-density diagonalisation at `k + q` (guard message).
- Check: a grid that contains that `q` (a different grid, so up to k convergence).
- QE/Elk: Elk refuses (`vecql` commensurability, per the test's docstring).

## Plain missing features, not pairs

Not ranked with the pairs. A committed reference is named where one exists.

- `tot_charge`, and `gate` behind it (`system/builder.py:_refuse_unimplemented_switches:1975`, table `_REFUSED_SWITCHES`; `_efield:2317`).
- `assume_isolated` (Makov-Payne, Martyna-Tuckerman, `2D`, ESM); `test-suite/pw_cluster/`, and it blocks `test-suite/ph_2d` as a reference for the `q != 0` core correction.
- `twochem`, `one_atom_occupations`, `qcutz`, hand-pinned `nr1` to `nr3s`, and `no_t_rev`, `force_symmorphic`, `use_all_frac`, `nosym_evc` (same table); `test-suite/pw_twochem/`.
- Wyckoff input: `space_group` (`system/builder.py:_build_cell:2060`) and `ATOMIC_POSITIONS crystal_sg` (`system/structure.py:from_card_units:137`).
- `K_POINTS tpiba_c` and `crystal_c` (`system/builder.py:_build_kpoints:2831`).
- `calculation = 'md'`, `'vc-md'` (`system/builder.py:_check_calculation:1874`); `test-suite/pw_md/`. `cell_dynamics = 'sd'`, `'damp-pr'`, `'damp-w'`, `'pr'`, `'w'` (`workflows/vc_relax.py:_check_cell_dynamics:516`); `test-suite/pw_vc-relax/vc-relax1.in`, `vc-relax2.in`. Other `ion_dynamics` schemes (`relax/registry.py:28`). `cell_dofree` shape, 2Dshape, volume, ibrav (`relax/cell.py:147`).
- `efield_cart` (three-direction `l3dstring`, `system/builder.py:_berry_field:2346`).
- UPF v1, `.van`, RRKJ3 (`pseudo/upf.py:278`, `:282`, `:299`).
- Hubbard intersite `V`, orbital-resolved lines, background channels and `hubbard_occ(...,2)` (`system/builder.py:_hubbard:2665` to `:2699`), with `pw_lda+U/lda+U+V*.in`, `lda+U_orbital_resolved.in`, `lda+U_background_*.in`; `Hubbard_projectors` `'wf'` and `'pseudo'` (`hubbard/manifold.py:485`); Yukawa Slater integrals from a non-PAW dataset (`hubbard/manifold.py:632`).
- `vdw_corr` D3, TS, MBD, XDM (`vdw/registry.py:120`).
- An analytic stress total (`stres_us`, `addusstress`; `stress/__init__.py:125`).
- The phonon dispersion (star of `q`, `q2r`, `matdyn` with the acoustic sum rule) and the electron-phonon coupling summed over the zone (`la2F`, `elphsum`, `lambda.x`).
- `chi^(2)` and the electro-optic tensor by the 2n+1 route (`dvpsi_e2`, `response/nonlinear.py:require_a_complete_third_derivative:358`); T1 already exists, `reference.out.ph-alas-raman` was run with `elop = .true.`.
- The relaxed-ion piezoelectric constant (internal strain `d^2E/du d eps`).
- The injection current and the Fermi-surface terms of a metal's shift current and SHG (`response/photocurrent.py:778`); sweep-response lists it as a pair, but it is a different assembly rather than a missing term of the insulator's.
- The frequency-domain hierarchy above third order (`realtime/hierarchy.py:_hierarchy_channel:512`); continuation across a change of cutoff (`notes/CONTINUATION-NEXT.md`).
- Subtracting Elk's frozen core from an all-electron `STATE.OUT` (`io/elk_density.py:density_on:650`); Elk's `q0cut` in the ultracell.
- Constant-current STS (`workflows/stm.py:_refuse_a_window_under_an_axis:660`), a magnetic plane tip in momentum-resolved transport (no `tip_spin`), a tilted exit plane (`transport/substrate.py:40`).

## Closed since the notes were written

Landed on 2026-10-16 in P154 (branch `incompat-night`), each with its record in `notes/incompat/<ID>.md`:

- I1, band gap at `nspin = 2`: `o-atom-fixed-lsda` HOMO and LUMO -6.633898 eV against `pw.x`'s -6.6339 (`reference.out.o-atom-fixed-lsda:261`); `atom-lsda` gap 6.164453 against 6.1680, the LUMO 2e-5 eV from `pw.x`'s full-sphere run (`incompat/I1`).
- I2, dynamical matrix at Gamma at `nspin = 2`: triplet O2 transverse block to 3.9e-5 Ry/bohr^2 against `reference.dyn.o2-fixed-lsda-phonon` (0.119 before), stretch 1664.369 against 1664.402 cm^-1; PAW `ph_insulator_paw_magn` stretch 2374.6975 against 2374.699451 cm^-1. The k-chunked route still refuses (`incompat/I2`).
- I4, a metal at `q = 0` through the `q != 0` route: the q route against the Gamma route on an off-centre `al2-metal` where `def` is 1.2e-2, 7.2e-7 Ry/bohr^2 (5.1e-2 without the shift); against `ph.x` on `al2-metal.in` 2.9e-3 cm^-1. A metal at `q = G != 0`, the k-chunked route at `q = 0` and the electron-phonon coupling at any `q = G` still refuse (`incompat/I4`).
- I6, shift current at `nspin = 2`: unpolarized AlAs `nspin = 2` against `nspin = 1` 6.4e-10 of max|sigma| = 5.66e-5 A/V^2; magnetic AlAs (`tot_magnetization = 2`) channel sum against the same states as spinors 1.6e-16 of 1.08e-5 A/V^2. A filling that cuts a degenerate shell is refused (`incompat/I6`).
- I7, sum-over-states `chi_0` at `nspin = 2`: unpolarized Si, the whole `chi_0` 3.0e-9 relative, eps_M(0) 1.6e-11 (RPA) and 1.5e-11 (ALDA); magnetic H3 chain eps_M(0) 8.65363 against the Sternheimer 8.64013, the residual from the response sphere. ALDA on a polarized density refuses; the `ph.x` O2 check is ultrasoft and stays out of reach (`incompat/I7`).
- I8, sum-over-states `chi_0` on a nonmagnetic spinor: Si as a spinor against the scalar run 5.8e-11 relative over the whole matrix, eps_M(0) 5.0e-10; spin-orbit Se eps_M(0) 16.51019387 against the spinor Sternheimer 16.55406495 (-2.65e-3, the scalar control -2.57e-3). A textured spinor and a spiral refuse by name (`incompat/I8`).

Landed on 2026-10-16 in P154, integration two (branch `incompat-night`), each with its record in `notes/incompat/<ID>.md`:

- I9, Berry-phase polarization at `nspin = 2`: triplet O2 channel phases 0.19500066 and 0.42500024 against `pw.x` `lberry`'s 0.19500 and 0.42500 (`reference.out.o2-lsda-berry`, made for it), each channel's seven lowest eigenvalues against `eigenvalues_by_spin` to 1e-6 Ry; AlAs converging to `m = 0` against `reference.out.alas-berry`. `from_input` refuses (`incompat/I9`).
- I12, Berry curvature and Chern number of one collinear channel: nonmagnetic AlAs per channel against `nspin = 1`, largest plaquette difference 9.7e-13 on fluxes of 4.1e-2; each channel's source against its own SCF eigenvalues 6.6e-8 and 1.5e-7 Ry on triplet O2, against a 0.131 Ry splitting. A channel's Chern number is zero by symmetry, so the check is the curvature; `kubo` and the orbital magnetization still refuse (`incompat/I12`).
- I13, Chern and Z2 invariants under a potential-only meta-GGA: the fixed-density source against the converged `tb09` eigenvalues, 1.1e-7 Ry (norm-conserving), 8.9e-7 (spinor), 2.5e-6 (PAW), the self-consistency floor of the band path. The polarization and the orbital magnetization keep the refusal (`incompat/I13`).
- I14, `lelfield` at `nspin = 2`: the Si8 chain at `tot_magnetization = 0` against `nspin = 1`, `E(E) - E(0)` 1.1e-10 Ry, `el_pol` 2.3e-7, forces against `pw.x` at 1e-6 Ry/bohr. A magnet refuses (`incompat/finitefield`).
- I15, `lelfield` on a nonmagnetic spinor without spin-orbit coupling: against `nspin = 1`, `E(E) - E(0)` 2.1e-14 Ry, `el_pol` 5.2e-11. Spin-orbit coupling and a magnetized spinor refuse (`incompat/finitefield`).
- I17, symmetrised projected DOS with spin-orbit coupling, nonmagnetic: fcc Pt wedge against the closed grid, column charges 3.7e-7 (ultrasoft) and 5.2e-9 (PAW), against `projwfc.x` 1.7e-6 and 2.2e-6. A magnetic spin-orbit run refuses (`incompat/I17`).
- I18, site angular momenta on a fully relativistic ultrasoft or PAW dataset: on the `|l j m_j>` set, the Loewdin charges of Pt to every digit `projwfc.x` prints (5e-5); on tetragonal Co `|<L>|` 0.117097947 along x and y, as the C4 requires. The two `j` shells' radial functions are treated as one, 1e-3 hbar of spurious `<L>` with the coupling off (`incompat/I18`).
- I19, nesting function at a constrained `tot_magnetization`: Ni at `tot_magnetization = 2`, the two levels 20.704251 and 14.299028 eV against `pw.x` 7.5's 20.7043 and 14.2991; `D(E_F)` against the wedge DOS 2.2e-16 relative. Tetrahedra with a constrained moment refuse (`incompat/I19-v2`).
- I20, force on a norm-conserving spin spiral: the NiI chain at `q = b3/4` against its four-cell supercell at one density, 1.17e-8 Ry/bohr on forces of 1.23e-2. Ultrasoft, PAW and DFT+U spirals refuse (`incompat/spiral-fix`).
- I21, stress on a norm-conserving spin spiral: the same pair, 7.8e-10 Ry/bohr^3 on a stress of 2.80e-2 (`incompat/spiral-fix`).
- I24, site angular momenta on a norm-conserving spin spiral: `<S>`, `<L>` and the Loewdin charge of each supercell atom against the spiral's, turned by `-q . R_j`, 2.1e-10; the opposite turn misses by 0.69 (`incompat/spiral-fix`).
- I27, meta-GGA on a spin spiral, norm-conserving, `noncollinear_gga = 'kubler'`: the hydrogen chain under `tb09` against its four-cell supercell at the same `(rho, tau)`, lowest twelve levels 4.8e-14 Ry; dropping the transverse `tau` moves them 0.54 Ry. `'grad_m'` and PAW refuse (`incompat/spiral-fix`).
- I28, DFT+U on a norm-conserving spin spiral: NiI with `U = 4 eV` against the four-cell supercell at the turned occupation matrix, the supercell's measured `ns` equals the turned spiral one to 1.2e-10. Ultrasoft and PAW spirals, and the forces, stress and first-order spin-orbit energy with U, refuse (`incompat/spiral-fix`).
- I32, ultracell with DFT+U, norm-conserving, unpolarized, `'atomic'` projectors: Si with `U Si-3p 2.0`, the tiled null at dV 6.7e-16, the 3x1x1 ultracell above the explicit DFT+U supercell by +4.9e-6, +7.9e-7 Ry per cell at 12 and 24 bands, falling monotonically. Every other Hubbard regime refuses (`incompat/ultracell-i32fix`).
- I37, velocity operator on gamma-only storage: displaced Si, half sphere unfolded against `k = 0` on the whole sphere, every block of `<m|dH/dk|n>` 2.3e-13 on norms of 1.1 Ry bohr, `Tr(v_x v_x)` 4.2e-13. The conductivity, the Kerr angle, SHG and the shift current still refuse (`incompat/I37`).
- I38, the real-space consumers on gamma-only storage, against `K_POINTS automatic 1 1 1 0 0 0`: projected DOS 3.9e-12, `eps_M` from `chi_0` 1.4e-9, STM 3.0e-8, STS 8.2e-8, vertical transmission 3.9e-8, momentum weight 1.2e-8, `X_0^{+-}` 8.0e-13 (`incompat/I38`).
- I39, real-time propagation from a `K_POINTS gamma` ground state: displaced Si under a one-cycle pulse against `k = 0` on the whole sphere, `max |J_gamma - J_k0| / max |J_k0|` 1.8e-13. A half-sphere `Calculation` handed to the driver directly still refuses (`incompat/I39`).
- I56, `lelfield` on a streamed wavefunction store: against the in-memory run, total 0.0 Ry (bit-identical), `el_pol` 3.9e-14, forces 4.6e-15 Ry/bohr. Pools refuse (`incompat/finitefield`).
- I57, `lelfield` on `K_POINTS automatic`: against a new `pw.x` 7.5 run (`reference.out.si8-lelfield-automatic`), total 1.2e-8 Ry, `el_pol` 5.8e-6. `efield_cart` off one axis refuses (`incompat/finitefield`).

Landed on 2026-10-16 in P154, the wedge unit (branch `incompat-night`, merge of `incompat/wedge-r2`), each with its record in `notes/incompat/<ID>.md`. A wedge is completed by the group it was cut with (`system/wedge.py:wedge_group`); a magnetic group, a reduced set on a `nosym` run, and a shifted grid, whose wedge the group unfolds onto its star closure rather than the grid (256 points from fcc `4 4 4 1 1 1`), still refuse:

- I40, optical conductivity on a wedge: AlAs `alas-raman-wedge.in`, 8 of 64 points, `Td` with time reversal, against the whole unshifted 4x4x4 grid, 2.8e-10 of the largest component (the uncompleted wedge 1.1 off) (`incompat/wedge-r2`).
- I41, shift current and SHG on a wedge: the same cell, 2.1e-9 and 1.4e-8 (uncompleted 0.96 and 2.5 off); with `noinv` (`Td` alone, 10 points) 2.8e-9 and 1.4e-8 (`incompat/wedge-r2`).
- I42, `chi_0(G, G')` on a wedge, both G labels and time reversal: Si `si-epsilon-unshifted.in` 8 of 64 points against the whole grid 2.1e-10 of `max|x|` (implementer, reproduced to every printed digit by the verifier), AlAs 6.3e-10; the verifier's Si at 3x3x3 (4 of 27 points), `ecut_response = 6` (61 G vectors), whole arm first, 3.9e-10; with the completion removed 0.73. `nspin = 2` and spinor wedges refuse, unmeasured (`incompat/wedge-r2`).
- I46, STS on a wedge, averaged over the tip's images: Si 4x4x4 symmetric, 4.5e-8 of the peak at `conv_thr = 1e-12` (implementer 4.2e-8 on a 12x12 plane); the verifier's 3x3x3, plane normal to x at 0.13, whole arm first, 4.0e-7; with the completion removed 0.99. `nspin = 2` and spinor wedges refuse, unmeasured (`incompat/wedge-r2`).

Landed on 2026-10-16 in P155, integration one of the rest of T2 (branch `incompat-t2`), each with its record in `notes/incompat/<ID>.md`:

- I33, ultracell under a potential-only meta-GGA: the TB09 3x1x1 ultracell at matched grids against the TB09 supercell, 8.144e-3 at 48 bands and falling with the band count; with `tau` frozen at the tiled unit-cell `tau` the ladder rises, 1.246e-1 to 1.636e-1. A spin-polarized cell and an augmented dataset refuse (`incompat/ultracell2-fix`).
- I34, ultracell under `nosource`: charge 7.13e-5 and magnetization 1.47e-4, 1.74e-4, 1.60e-4 against the supercell at 16 bands; dropping the projection at `Q != 0` moves `m_x` to 2.5e-1 (`incompat/ultracell2-fix`).
- I35, ultracell under a saw-tooth field, one cell along `edir`: with `dipfield` the field energy per cell against the supercell's 0.53840044 Ry, 6.514e-5, 1.224e-5, 3.585e-6 Ry at 12, 24, 48 bands (a frozen saw-tooth 1.118e-4 at 48); the energy above the supercell +4.789e-7 Ry at 48. More than one cell along `edir` and `lelfield` refuse (`incompat/ultracell2-fix`).
- I49, real-time propagation on a shifted grid: Si `4 4 4 1 1 1`, field along [100], the 2 kept operations and 40 points against the whole 64-point grid, 1.398e-8 of the current (frozen), 1.717e-8 (`hxc`); all eight operations 0.823 off (`incompat/I49`).
- I50, spin spiral with crystal symmetry, the spin space group that keeps `q` and the moments: SCF, forces, stress and DOS against `nosym`, zigzag force 5.6e-8, stress 2.1e-10, clean cells 7e-14 to 1.6e-12 Ry. Augmented datasets, DFT+U, `lspinorb`, explicit k-lists, site angular momenta, projections and `dE/dq` keep `nosym` (`incompat/I50`, merged from `worktree-wf_93b36bbe-189-57` 9ec1c09).
- I51, partial dynamical matrix of a closed subset with symmetry: the subset block against the whole symmetrised block, AlAs 1.7e-10 (nc), 3.7e-10 (US), 3.6e-11 (PAW), the fluorite pair 1.5e-11, the verifier's L1_2 1.6e-11 (the inverse atom table 2.73e-1); the wedge subset against `nosym` 6.8e-10 at a floor of 7.9e-10. A subset that is not closed, the streamed and the walked routes with symmetry, and the acoustic sum rule refuse (`incompat/I51-merged`).
- I53, Newton-Krylov SCF under pools: Si 8k at 2 and 3 pools against 1 pool, 1.8e-14 and 1.1e-14 Ry, 39 F evaluations on every rank. Tetrahedra and `from_input` refuse on every rank before any collective; `jvp='autodiff'` and `'sternheimer'` refuse (`incompat/I53`).
- I54, partial dynamical matrix from a walked response: the subset against the block of the same kept response's whole matrix, 0 bit for bit on nc Si and US AlAs, the two routes 1.2e-14 and 4.2e-14 apart; the old whole-k route was 2.57 off on Si and raised `IndexError` on US AlAs (corrected at integration) (`incompat/I54`).
- I55, self-consistent frequency-domain hierarchy on two FFT grids, norm-conserving: two-grid AlAs against the dense two-grid solve, 1.9e-11 to 5.1e-11 (Hartree) and 9.5e-12 to 2.5e-11 (`hxc`); Hartree two-grid against one-grid 1.8e-14 to 3.5e-14. Ultrasoft on two grids and a meta-GGA kernel refuse (`incompat/I55`).
- I58, magnons under a uniform field: H fcc at `B = 0.01` Ry, Goldstone residual 0.00585 at `omega = 2B`, against 0.171 at 0 and 0.286 at `-2B` (zero field 0.00571). A constrained moment and a per-atom field refuse (`incompat/magnon`).
- I61, magnons under a simplified U: H fcc with `U = 2` eV, Goldstone residual 0.00473 with the Hubbard channel against 0.197 without, `lambda_max(q=0)` 1.0143 against 0.818. `J` and `BETA` refuse (`incompat/magnon`).
- I63, real-time kick inside the run on ultrasoft and PAW: against the run started at the kick, 1.71e-11 (US, `propagate`), 1.08e-12 (US, orders), 2.21e-11 and 1.10e-12 (PAW). A kick off the time grid refuses (`incompat/I63`).
- I64, real-time propagation from a ground state under a field: Si2 under a sphere field with `reducebf = 0.8`, 4.31e-10 electrons out of the occupied subspace against 8.93e-3 with the input field; the kick energy drift 1.33e-7 Ha against 1.03e-6 with the old closure. A penalty constraint with the potential updated refuses (`incompat/I64`).
- I65, checkpoint of a fixed-spin-moment run: hydrogen `fsm` stopped and resumed, field -0.079999913 on both resume routes, 1.4e-17 apart (the old manual route -0.04); `fe-fsm.in` through `checkpoint_dir` 26 against 26 iterations, dE 1.2e-11 Ry (`incompat/I65`).
- I67, continuation across the spiral boundary at a reciprocal-lattice `q`: H chain spiral to plain at `q = b3`, 1 iteration against 11, E 4.9e-10 Ry; PAW O chain 3 against 13, `becsum` 1.2e-5 against 1.37 with the per-atom phase. A spiral into `nspin = 2`, a non-integer `q` and DFT+U refuse; `tau` is dropped across the frame (`incompat/I67-fix`).
- I68, seed under another pool layout: Si 8k, two pools, blocks to longest-first, against an unseeded run 1.39e-13 Ry in 1 iteration against 8. A span of another layout outside a pooled `run_scf` refuses (`incompat/I68`).
- I69, Elk seed from a collinear spin-polarized state: the magnetization against `MAG3D.OUT` worst 1.5e-11, the muffin-tin moment to every digit of `INFO.OUT`; seeded against atomic start dE 4.3e-14 Ry at `conv_thr = 1e-13`, moment -0.29092007 against -0.29091922; a flipped moment lands on the mirror state. Noncollinear, fixed-spin-moment and mismatched-channel seeds refuse (`incompat/I69`).

Landed on 2026-10-16 in P155, integration two of the rest of T2 (branch `incompat-t2`), each with its record in `notes/incompat/<ID>.md`:

- I70, Elk seed from a DFT+U state: seeded against unseeded DFT+U SCF of sc hydrogen at `U = 4` eV, dE -6.7e-16 Ry, `max|drho|` 7.0e-9, `max|dns|` 5.3e-9, five iterations each (U itself moves E by 3.756e-2 Ry); `vmatmt` against Elk's `VMATMT.OUT` to every printed digit. Only the density crosses, with a warning. A fixed tensor moment refuses, and a spin-polarized DFT+U state, where I69 meets I70, refuses unmeasured (`incompat/I70`).
- I71, Newton-Krylov with tetrahedra and `from_input`: `al-tetrahedra.in` -4.1978985974 Ry, 2.6e-9 from `pw.x`'s -4.19789860; `scf-occ.in` 8.8e-10 from `pw.x`; the autodiff Jacobian within 1 to 7 per cent of a central difference (the verifier puts the residue in the eigensolver). The Sternheimer Jacobian and pools (I53's refusal, now pooled only) refuse (`incompat/I71`).
- I72, Newton-Krylov on a spin spiral: `F(x*) - x*` at the mixing loop's converged state 1.32e-10 (NC), 2.70e-10 (US), 1.35e-11 (PAW), and the loop takes 1 iteration after the hand-off (10 under the verifier's map with `m_x`, `m_y` scaled by 0.97). A Hubbard U (Im `ns` dropped, 2.34e-4) and a meta-GGA refuse, and a spiral under a field, where I72 meets I73, refuses unmeasured (`incompat/I72`).
- I73, Newton-Krylov under a field or a penalty: against `pw.x` 7.8e-8 Ry (`atomic`), 2.7e-9 (`atomic direction`), 1.3e-8 (`total`); a collinear uniform field against mixing 1.7e-8 Ry, 1.7e-6 mu_B; the mixing loop after the hand-off 1 iteration, 11 to 41 when Newton's F drops the field. Feedback schemes, `reducebf < 1`, the two unmeasured penalties, a local-field card and `jvp='autodiff'` refuse (`incompat/I73`).
- I73, the feedback field in the solved state, merged 2026-10-17 on the user's decisions (`incompat/I73-feedback` 4450fe9; measure `|dm| (|B| + tau |dm|)`): raw total against the mixing loop at its fixed point, `conv_thr = 1e-11`: `fe-fsm.in` `'fsm'` 7.8e-13 Ry (39 steps), canted iron pair `'atomic fsm'` 7.0e-13 (63 of 80), `'atomic fsm direction'` -1.01e-9, the reference loop's own arccos offset (60 steps; energy at the target 2.8e-14); field rows replaced by the identity: 45 to 102 loop iterations after the hand-over against 1. Refused by name: `'secant'`, noncollinear `'fsm'`, pools, `reducebf < 1`, and under a feedback field a Hubbard U, a meta-GGA, a non-smearing occupation, `K_POINTS gamma`; a noncollinear Hubbard U in the residual for every run (`pack` kept only the real part of `ns`). A non-converged hand-over warns (`notes/incompat/I73.md`).
- I74, `mixing_space = 'g'` on a magnetic spiral: `o-chain-spiral-us.in` at `conv_thr = 1e-13`, `SpiralSphereLayout` 13 iterations, E 1.1e-13 Ry from `'r'`, charge and `|m|` 1.2e-7; the cutting layout does not converge in 40 and stops 3.5e-9 Ry off (`incompat/I74-fix`).
- I75, `mixing_mode = 'ldos'` with tetrahedra and as a Newton warm-up: Bloechl `al-tetrahedra` -4.1978985966 Ry against `pw.x` -4.19789860; Kawamura `al10-metal-tetra` 19 iterations against `pw.x`'s 55; the summed LDOS weights equal the tetrahedron DOS at `e_F` to 1e-10; the warm-up then Newton 3.7e-9 Ry from mixing (Gaussian). The tetrahedron LDOS as a Newton warm-up, where I71 meets I75, refuses unmeasured (`incompat/I75`).
- I76, `mixing_mode = 'robust'` under pools and across a checkpoint: 2 pools against 1, the same iteration counts, rungs and decisions, dE at most 8.9e-15 Ry; resumed from iteration 5 against uninterrupted, the same count, dE at most 7.1e-15 Ry; without the supervisor file iteration 6 is 2.3e-3 Ry off (`incompat/I76`).
- I78, `rotate_moments` with crystal symmetry, on the turning group `{E, I}`: tetragonal Co started 45 degrees off c, 10 points against `nosym`'s 18, dE 5.7e-14 Ry; the shifted grid 9 against 18, -2.8e-14 Ry. The starting orientation's group refuses, and PAW keeps `nosym`. No test covers a pooled run on the turning group (`incompat/I78-fix`).
- I80, phonon at `q != 0` with a nonlinear core correction: `D(b1/2)` against the doubled cell at Gamma, 2.79e-6 Ry/bohr^2 (ONCV Si, 0.74 e core), 4.81e-6 (broad Gaussian core), at a floor of 4.40e-6 without a core; with no core terms 1.26e-1 and 7.8e-2. A core at `nspin = 2`, where I80 meets I82, refuses unmeasured (`incompat/I80`).
- I82, phonon at `q != 0` for a collinear `nspin = 2` insulator under the LDA: the H3 chain at `b3/2` against the doubled cell at Gamma, 1.522e-5 Ry/bohr^2, worst partner 0.051 cm^-1. A GGA at any `nspin` refuses (7.97e-3 and 60.55 cm^-1 under PBE at `nspin = 2`, 1.33e-2 and 77 cm^-1 at `nspin = 1`, silently wrong before), as do a spin-polarized metal, the k-chunked route at `nspin = 2` and the coupling at `nspin = 2` (`incompat/I82-fix`).
- I85, `lelfield` on a spin spiral and on a magnetized spinor: the spiral at `q = b3/4` against the 90-degree supercell, `E(E) - E(0)` 8.3e-12 Ry, `el_pol` 8.9e-8 mod the quantum (one gather for both components 6.9e-8 Ry and 5.5e-4); the supercell against a new `pw.x` 7.5 run 2.2e-9 Ry (zero field), 1.0e-9 Ry and `el_pol` 2.0e-7 (field). A metal, an ultrasoft dataset and spin-orbit coupling refuse (`incompat/I85`).

Landed on 2026-10-19 in P156, integration one of tier T3 (branch `incompat-t3`), each with its record in `notes/incompat/<ID>.md`:

- I86, Gamma phonon of an ultrasoft or PAW metal: al2-metal against `ph.x` on the same input (`reference.out.ph-al2-metal-us` and `-paw`, made for it), at most 0.0136 cm^-1 (ultrasoft) and 0.023 cm^-1 (PAW); acoustic sum rule 6.7e-6 and 4.2e-6 Ry/bohr^2, 4.3e-2 with the term off; a force difference 6.6e-6 against 3.25e-2 unfixed. The verifier's distorted cell measures the Fermi-shift branch the committed cells leave at zero. Symmetry, `nspin = 2`, `atoms=` and the walked route refuse (`incompat/I86`).
- I87, Elastic constants of an ultrasoft or PAW crystal: Si2 against a five-point second difference of the energy, `C_1111` -1.6e-8 to -3.8e-8 (US) and -7.5e-8 to -8.9e-8 (PAW) at h = 8e-3 to 2e-3, at the norm-conserving floor of 1e-7; 22.1 per cent off with both terms off, 10.0 with `ort` alone. The walked route refuses. The verifier found the energy reference rough on other cells and tangents (an xc step of 4.7e-6 Ry, cause open), so the stress difference is the independent confirmation (`incompat/I87`).
- I89, Piezoelectric tensor of a PAW crystal: AlAs `e_14` against `pw.x`'s Berry-phase difference 0.693273 C/m^2 (references made for it), 15.22, 3.71, 0.86, 0.17, -0.001 per cent at meshes 4 to 12, as the ultrasoft and norm-conserving cells fall; the reference is good to 0.10 per cent. `zstar_eu` still refuses PAW (`incompat/I89`).
- I92, Shift current and SHG with DFT+U, norm-conserving and collinear: the Hubbard velocity against a difference of `H(k)` with `vhpsi`, errors falling as h^2 (6.72e-6 to 4.20e-7) on a term of 8.54e-2; the complete-basis sum rule 1.63e-3 against 4.37e-2 with the Hubbard `w` off. What was missing was `ns` on the entry points. Noncollinear DFT+U and augmented DFT+U SHG refuse (`incompat/I92`).
- I97, ALDA kernel with a GGA: the adiabatic PBE Dyson static limit against the PBE Sternheimer `eps_inf` 21.8318 on Si, +0.0312 to -0.0042 as `ecut_response` goes 8 to 28 Ry (90 bands), the local part alone -0.27. The 1.5e-2 bound is the committed cell's; the verifier's core-charge cell reads -0.019 at 20 Ry, its RPA truncation -0.024. Meta-GGA and a GGA spinor refuse (`incompat/I97`).
- I98, Real-time potential updated on ultrasoft or PAW, `nspin = 1`: the kick local-field shift against the Sternheimer shift, residual equal to the norm-conserving control on 2x2x2, 3x3x3 and 4x4x4 (the verifier's 3x3x3 shows it is not monotone in the mesh at 0.2 eV); with the augmentation held, +1.79e-1; fixed point 5.3e-12, energy after a kick 1.07e-10 Ha. The verifier's pulse identity, 1.75e-4 against the frozen 1.10e-4, has no test yet. Spinors, `nspin = 2`, PAW `hartree`, the orders refuse (`incompat/I98`).
- I100, Real-time propagation with DFT+U, frozen `ns`: the first order from a kick against the Kubo sum plus the band curvature, 8.07e-7 relative, falling onto that floor with dt; 0.904 with the projectors held at kappa = 0, 0.691 with the term dropped. `ortho-atomic`, spinors, US/PAW, the updated potential, `get_hhg` and the perturbative orders refuse (`incompat/I100`).
- I102, Frequency-domain hierarchy with the potential updated at orders 2 and 3: against the self-consistent propagation on AlAs, 3.7e-6 to 7.6e-6 (hxc) at 800 steps a period, falling as dt^2 to 2.1e-7 to 4.4e-7 at 3200 (the verifier's); frozen 3.5e-3 to 5.0e-2 away; wedge against whole mesh 8.8e-11 to 6.0e-9. Ultrasoft and meta-GGA at order 3 and order 4 refuse (`incompat/I102`).
- I103, `lelfield` on an ultrasoft or PAW dataset at `nspin = 1`: Si8 along y against `pw.x` 7.5 (references made for it), total 4.7e-9 Ry (US) and 7.5e-9 (PAW), `el_pol` 1.7e-6 and 6.6e-6, forces 8.7e-8 and 1.05e-7 Ry/bohr (the verifier's comparison; the record's 5.7e-8 and 8.9e-8 misread it); the cross term off 0.134 in `el_pol`. Along z `pw.x` 7.5 evaluates the cross term at `q = 0` (`qvan2.f90:237`), reproduced here with `Q_ij(0)`. `nspin = 2`, spinors, spirals, the streamed store and `method='analytic'` refuse (`incompat/I103`).
- I110, Spin torque under a potential-only meta-GGA: the h-chain under TB09, `torque_of_result` against m x B_xc of `SCFResult.potential` to 5.2e-18 Ry per site at every `conv_thr`, 3.7e-6 to 1.2e-4 with the vector part of `tau` dropped. The site torques there are a symmetry residue, so the nonzero `tau`-dependent torque is shown on made-up fields (`incompat/I110`).

Landed on 2026-10-20 in P156, integration two of tier T3 (branch `incompat-t3`), each with its record in `notes/incompat/<ID>.md` (I108 and I109 merged, reverted and re-landed after integration two with I110's guard test retargeted, `notes/INCOMPATIBILITIES-T3.md`):

- I90, Piezoelectric tensor of a polar norm-conserving crystal: the proper tensor on mm2 AlAs against a Berry-phase finite difference, 2.8e-3, 5.4e-4, 1.6e-3 and 1.17e-2 C/m^2 at 8^3 on `e31`, `e33`, `e15`, `e14`; after the fix round the autodiff wedge equals the closed grid to 1.1e-9 (1.1e-3 with the completion off), and the verifier's 3m cell to 4.2e-9 (6.16e-4 off). An ultrasoft or PAW polar crystal still refuses.
- I93, Drude term of the optical conductivity with tetrahedra: fcc Al, `tetrahedra_lin` within 0.035 eV of the `degauss^2` limit of the smeared plasma frequency on every grid from 40^3 to 72^3, between 12.43 and 12.49 eV with the mesh; `intraband = False` gives 0 exactly; the band-edge filter and the D4 multiplet average each with a mutant-tested unit test.
- I94, Sum-over-states `chi_0` on an ultrasoft insulator at `nspin = 1`: Si RPA `eps_M(0)` minus Sternheimer -0.3369 to -0.0091 from 30 to 76 bands, monotone; +0.2498 with P94's plain head and -2.4579 without `Q_ij(G)`; ALDA xc shift 4.8395 against 4.8676, wedge 2e-9, scissor pole shift 1.35e-5. PAW, smearing, `nspin = 2` and spinors on ultrasoft still refuse.
- I95, Sum-over-states `chi_0` with smearing on a gapped cell: Si onto the fixed-occupation spectrum 0.844, 0.3815, 0.0329, 8.95e-4, 2.2e-8 at `degauss` 0.02 to 0.005 Ry, the Gaussian tail; the fixed-occupation array bit-identical to the old code.
- I96, Sum-over-states `chi_0` of a metal: Al Drude head against the optical conductivity 3.8e-13 (872.1 with the block off); f-sum ratio 0.99410 at 40 bands and 0.99638 at 48 (0.37 without the block).
- I101, Real-time propagation under `tefield` at a frozen potential: stationarity 1.1e-12 against 9.9e-6 with the term out; H2 static polarizability within about 0.1 per cent from a 16-bohr box up (1.4 per cent at 12 bohr), the error growing as `E0^2 T^4`; the updated potential and the frequency-domain hierarchy still refuse a saw-tooth.
- I108, `noncollinear_gga = 'grad_m'` under `nosource`: the projected divergence 6.97e-31 against 3.456 unprojected on the h4 chain, the SCF's own field divergence-free to 1e-30; 62 iterations to an up-up-down-down state, where `'kubler'` takes 371 to a nearly nonmagnetic one.
- I109, `nosource` with a potential-only meta-GGA: TB09 sphere divergence 1e-31 projected; the box residue on converged meta-GGA runs is 0.18 to 0.64 and does not fall, a property of the array outside the sphere the run reads.
- I99, Real-time propagation with the potential updated (`hxc`) under a potential-only meta-GGA, `nspin = 1`: `tau(t)` rebuilt at `k + kappa(t) + G` beside `rho(t)`, against the dense self-consistent solve 3.76e-5 (Si2 tb09, 800 steps a period) and 1.52e-5 (AlAs), falling as dt^2; `tau` held at `tau0` 0.281 and 0.0217. Through `Calculator.get_realtime` the 8-operation wedge against the whole 4x4x4 mesh 1.10e-8, 9.78e-2 with `tau` not completed by the little group (verifier: AlAs 3.8e-9 against 3.4e-2). Landed after integration two, combined by hand with I98 (the propagated state is `(rho, becsum, tau)`), and reproduced on the combined head to every printed digit. `energy` is NaN; `nspin = 2`, spinors, PAW under `hxc` (a pair neither side covered) and the hierarchy's meta-GGA kernel refuse (`incompat/I99`).

Partial in the same integration, merged for the part measured: I104 (the Gamma phonon under `tefield`, to about 2e-4 Ry/bohr^2) and I114 (`soc_scale` between 0 and 1 on ultrasoft), each marked in place above.

Landed on 2026-10-20 in P156, integration three of tier T3 (branch `incompat-t3`), each with its record in `notes/incompat/<ID>.md` (the sections of I111 and I113 above were removed; integration three in `notes/INCOMPATIBILITIES-T3.md`):

- I111, Fixed-spin-moment feedback field (`'fsm'`, `fsm_update = 'elk'`) under `lspinorb`: bcc Fe (ultrasoft fully relativistic, shifted 2x2x2), target 1.0 mu_B at 30 degrees; the converged field held fixed from the converged state is self-consistent at its first iteration (`dr2` 1.07e-10) and ends 2.2e-10 Ry from the fsm total; the field projected on the target, dropping its transverse 1.55e-4 Ry, starts at 5.8e-7. The transverse part is 1.55e-4 Ry on that mesh and target and 3.7e-7 to 1.55e-4 over the three cells; 1.7e-7 with the coupling off. The guard's old test is now `test_the_fixed_spin_moment_field_runs_under_spin_orbit_coupling` (`tests/unit/test_magnetic_fields.py`). The secant update and the per-atom schemes still refuse; the Newton solver refuses `'fsm'` in any noncollinear run (I73) (`incompat/I111`).
- I113, `rotate_moments` under an applied `B_field` (`nosym`): tetragonal Co, the two-field fit of `K1 sin^2 + K2 sin^4 - B M sin(theta)` gives `K1 + K2 = 3.1032e-5` Ry against the relaxed `E(x) - E(z) = 3.1054e-5`, -0.074 per cent and flat in `torque_conv_thr`; the residue is not `K3` (a three-field fit gives -1.35e-8 Ry), cause open, candidate the 0.076 per cent `|M|` difference between the z and x runs. With the field's share left in the step the run stays on the easy axis. Constraints, `reducebf` and a field with symmetry still refuse (`incompat/rotate`).

Partial in the same integration, merged: I112 (`rotate_moments` with a Hubbard U on norm-conserving and ultrasoft datasets, with the `|M|` warning), marked in place above.

Landed on 2026-10-20 in P156, integration four of tier T3 (branch `incompat-t3`, the section of I115 above removed; integration four in `notes/INCOMPATIBILITIES-T3.md`):

- I115, Ultracell transport, `exit_region = 'volume'` x ultrasoft or PAW: the ultracell's `S` summed over the `N` copies of every atom (`ultracell/augmentation.py:ultracell_overlap`). Si sheet `(2, 1, 1)`, US and PAW, spinless, `nspin = 2` and scalar-relativistic spinor: Gram `C^dagger S C` the identity to 3e-15 to 1e-13, the volume transmission the STM image to 3e-14 to 1.2e-13; with `S` off the Gram diagonal is above 1 by 0.28 and the image misses by 2.8e-2. On `(3, 1, 1)` the Gram identity holds to 1e-14 and the image only to 1.3e-8 to 1.6e-8, the same on norm-conserving, so not `S`'s. Verifier's magnetized `nspin = 2` (US `(3, 1, 1)`, `magnetic_field` `0.05 cos(2 pi x / 3)`, splitting 4.2e-2 Ry): Gram 1.2e-14, 0.27 with `S` off. `lspinorb` with the volume exit on an augmented dataset stays refused, unmeasured (`incompat/I115`).

Landed on 2026-10-20 in P156, integration five of tier T3 (branch `t3-I88`, merge `19744ed6`, the section of I88 above removed; integration five in `notes/INCOMPATIBILITIES-T3.md`):

- I88, Elastic constants (strain response) x metal: the phonon's `ef_shift` on the states' half of the strain density and `ef_shift_wfc` on `dpsi`, the `wg`/`wk` split written as two copies of the band block (`response/elastic.py:_split_weights`), the strain route's smearing cutoff at `STRAIN_SMEARING_SMALL = 1e-12`, and a highest band holding at most 1e-12 (`_require_an_empty_top_band`). Smeared norm-conserving metal on a closed `nosym` grid, whole-k route: `al2-metal.in` `C_1111` against a five-point difference of `F` 1.6e-7 (M-V), 1.3e-6 (M-P 0.04), 2.2e-7 (Gaussian 0.03), 3.6e-8 (F-D 0.05 at `nbnd = 17`), each falling as `h^4`; with QE's 6.96e-5 cutoff -4.6e-4 (M-P), flat in `h`. Window, documented by the user's decision: a Gaussian top band 4.97 to 5.20 `degauss` above the Fermi level, or a Marzari-Vanderbilt one 4.47 to 5.20, is admitted and reads 2.7e-6 to 7.0e-6 off where `nbnd` splits a degenerate pair (`degauss` 0.17 to 0.19). The limit also refuses Gaussian and M-P 0.05, Gaussian 0.08 and M-V 0.08 at `nbnd = 7`, measured right. Refused: ultrasoft and PAW metals, tetrahedra, a reduced k-set, the walked route (`incompat/I88`, three fix rounds, `notes/incompat/I88.md`).

Earlier:

- LSDA kernel at `|zeta| >= 1` for the LDA (HOLES #1): `_fully_polarized`, 5.1e-6 against `ph.x` on O2; the GGA half is the T4 entry on the screening kernel.
- LSDA Born charges (HOLES #2): no `nspin` raise remains.
- PAW Born charges (HOLES #13): P39a.
- Kubo curvature, optical conductivity and SHG on ultrasoft or PAW (HOLES #6): P99.
- Ultrasoft magnons: P93.
- Ultrasoft and PAW spin spirals and their `dE/dq` (HOLES #14): P95, P96.
- Ultrasoft and PAW third derivatives in strain: P100.
- Ultrasoft piezoelectric tensor: lifted 2026-09-20.
- Noncollinear DFT+U with symmetry (half of HOLES #8): P82.
- Nonmagnetic spinor dielectric constant and Born charges, ultrasoft spinor and its multipliers: P83, P98.
- Projection at `nspin = 4` without SOC, and its symmetrised form: 2026-09-12.
- Site angular momenta on a reduced k-set: completed rather than refused (P82).
- `fixed` with `nspin = 2` and `tot_magnetization`: GAPS §3.
- Forces, stress and relaxation of a magnetic spinor: P46.
- Relaxed anisotropy with PAW and DFT+U: P87.
- Ultracell with ultrasoft or PAW: P88 stage 5; ultracell at `nspin = 2` and `4` runs (`ultracell/driver.py:487`; `notes/ELK-FEATURES.md` §9 corrected in P154).
- Real-time propagation at a frozen potential on ultrasoft, PAW, `nspin = 2` and spinors: P139 to P141 (`.claude/rules/scope.md`'s real-time bullet corrected in P154).
- `promote_ns` into `nspin = 4`: P79.
- DFT+U with topology: no guard left in `workflows/topology.py`.
- `Functional.require_spin` (`xc/functional.py:461`): unreachable, every registered correlation has a spin form.
- Gamma-only storage with a GGA (`basis/gradients.py:90`) or with ultrasoft, PAW, DFT+U, spinors (`pseudo/augmentation.py:1233`): not live; the dense set is never halved and the full sphere is substituted with a warning.
- `at_kpoints`, `at_kcart`, `at_rows`, `with_texture` on a spiral (`scf/driver.py:4001`, `:4390`, `:4289`, `:4109`): internal API; `run_bands` takes its own spiral route (`workflows/nscf.py:544`), and the consumers are the spiral entries above.
- The handoff checks in `workflows/nscf.py` and `workflows/topology.py` (`becsum`, `ns`, `tau`, the final field): the facade passes them (`tests/unit/test_calculator.py:325`).

## Physically undefined, or a refusal by definition

- Dielectric constant, Born charges, Raman, piezoelectric tensor and Berry-phase polarization of a metal (`phq_readin.f90:957`, `bp_c_phase`).
- `lelfield` with smearing or tetrahedra (`input.f90:1206`) and with gamma-only storage (`input.f90:1236`): a field on a metal, and no strings at one k-point.
- Orbital magnetization at `nspin = 2`, and anisotropy, orbital magnetization or `Z2` without spin-orbit coupling: zero by a global spin rotation.
- Meta-GGA on an ultrasoft dataset (`scf/driver.py:_require_meta_supported:3108`): no partial waves for a one-centre `tau`; QE refuses too (`setup.f90:124`).
- Every energy derivative of a potential-only meta-GGA or of `nosource`, and a potential-only meta-GGA anisotropy: no energy to differentiate.
- Magnons of a potential-only meta-GGA (`tddft/spinchi0.py:289`): the kernel's rotation argument needs an energy; sweep-tddft calls it close to intrinsic.
- Magnetoelectric tensor at `nspin_mag != 4` (`response/magnetoelectric.py:179`): a run with no magnetization density has nothing for the field to couple to.
- A spin spiral with the coupling switched on exactly: it breaks the generalized Bloch theorem.
- A filling that cuts a degenerate multiplet (`response/sternheimer.py:1453`); `from_input` occupations in a response (`:1662`).
- STM, STS and tunnelling maps, unit cell and ultracell, with a constrained `tot_magnetization` (`workflows/stm.py:859`, `workflows/ultracell.py:619`): two Fermi levels and a tip that sees one energy; sweep-workflows lists them as T3 pairs.
- A tip or lead plane inside an augmentation sphere; a spin-selective lead on an unpolarized run; a transverse tip on a collinear run.
- Ultracell with tetrahedra, `from_input` occupations or gamma-only storage (no `k0 + Q` is Gamma).
- LDOS mixing in a response loop (`response/mixing.py:140`): a response has no LDOS; queue.md listed it as an execution mode.
- The nesting function of a spiral (`response/nesting.py:184`).
- Second-method refusals whose default covers the regime: analytic force with spinors, DFT+U, a tabulated table, gamma-only or pools; analytic stress terms with spinors or gamma-only; `method='zstar_eu'` on PAW.
- A directional response with `nosym` on a shifted grid (`response/efield.py:1339`): an input-quality refusal, the fix is an unshifted grid.
- `rotate_moments` without the coupling, under a penalty constraint; a collinear FTM (which is FSM); `amf` with the simplified functional.
