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

### I1. Band gap (`BandStructure.gap`) x `nspin = 2`

- Axis: spin.
- Guard: `defuqumat/workflows/bands.py:BandStructure.gap:131`.
- Test: none.
- Missing: the per-channel filling, taken from `eigenvalues_by_spin` with each channel's occupation (guard message).
- Check: eigenvalue: `tests/data/qe/reference.out.pw_atom-atom-lsda:295` ("highest occupied, lowest unoccupied level", -6.6338 and -0.4658 eV) and `reference.out.o-atom-fixed-lsda:261`.
- QE/Elk: QE y (`PW/src/print_ks_energies.f90` prints the LSDA HOMO and LUMO over both channels).
- Note: Only in sweep-workflows (C3); queue.md missed it.

### I2. Dynamical matrix at Gamma x `nspin = 2`

- Axis: spin.
- Guard: `defuqumat/response/phonon.py:_require_one_spin_channel:1850` (and the gate's `spin_polarized` flag, `response/sternheimer.py:require_a_sternheimer_regime:1679`).
- Test: `tests/regression/test_phonons.py:655` `test_a_spin_polarized_calculation_is_refused`; `tests/regression/test_lsda_response.py:673` `test_the_dynamical_matrix_and_the_strain_response_still_refuse_nspin_two`.
- Missing: the spin axis of the second-derivative assembly above the solve (`phonon.py` docstring, HOLES #3). Measured with the guards bypassed on triplet O2: the stretch 1664.3992 against 1664.401578 cm^-1 (1.4e-6), both `D_zz` entries shifted by a common -0.0022, `D_xx` off by up to 0.12; nonmagnetic `nspin = 2` equals `nspin = 1` to 1.6e-14. Next step named: per-term decomposition against a finite difference of the LSDA forces (`o2-lsda-force.in`).
- Check: derivative: `tests/data/qe/reference.dyn.o2-fixed-lsda-phonon` element by element (not the acoustic sum, which `ph.x` itself violates by 0.0087), with `reference.out.ph-o2-fixed-lsda-phonon`; second reference `test-suite/ph_insulator_paw_magn/benchmark.out.git.inp=O2.phG.in.args=2` (PAW, `O.pz-kjpaw.UPF`, committed).
- QE/Elk: QE y (`ph.x`; no LSDA refusal in `phq_readin.f90`, `phq_setup.f90`).
- Note: queue.md quotes 1.4e-6 (frequency) and sweep-response 7.8e-6 (matrix element); both are in the docstring, no disagreement.

### I3. Gamma phonon x nonmagnetic spinor (`noncolin`, no moment, with or without `lspinorb`)

- Axis: spinor.
- Guard: `defuqumat/response/sternheimer.py:require_a_sternheimer_regime:1594` (`phonon.py:388` does not pass `noncollinear=True`).
- Test: `tests/unit/test_spinor_refusals.py:104` `test_the_sternheimer_refusal_still_stands`; `tests/regression/test_spinor_response.py:627` `test_what_a_spinor_response_still_refuses`.
- Missing: the caller's assembly on a two-component state; the solve (P81) and the augmented spinor (P98) need no new term (MAGNETISM-NEXT A, HOLES #10).
- Check: derivative: Gamma frequencies in `tests/data/qe/reference.out.ph-alas-epsilon-us-soc` (8 `freq` lines, 376.714126 cm^-1 triplet) and `reference.out.ph-bismuthene-epsilon-us-soc`; beside it T2, silicon as `noncolin` without SOC equals `nspin = 1`.
- QE/Elk: QE y (`dfpt_kernels.f90:212`, `nsolv = 1` without `domag`).

### I4. Phonon of a metal at `q = 0` (or `q = G`) through the `q != 0` route

- Axis: occupations (metal).
- Guard: `defuqumat/response/phononq.py:_require_a_metallic_q:1493`.
- Test: `tests/unit/test_metallic_q_refusals.py:26` `test_a_metal_at_a_reciprocal_lattice_vector_is_refused`; `tests/regression/test_electron_phonon.py:256` `test_a_metal_at_q_zero_is_refused_by_name`.
- Missing: `ef_shift` on this route (`get_phonons` has it) (guard message).
- Check: derivative: `tests/data/qe/reference.out.ph-al2-metal` (12 `freq` lines); T2 beside it, equality with the Gamma route on the same state.
- QE/Elk: QE y (`lmetq0`).

### I5. Linear response with DFT+U at Gamma (dynamical matrix, dielectric tensor, Born charges)

- Axis: DFT+U.
- Guard: `defuqumat/response/sternheimer.py:require_a_sternheimer_regime:1643`; internal duplicate `scf/driver.py:restricted_hamiltonians:4345` (row-subset Hamiltonian, no caller reaches it with a U).
- Test: none fires `sternheimer.py:1643` (`tests/regression/test_scf_solvers.py:778` `test_the_sternheimer_jacobian_refuses_a_hubbard_u_and_a_meta_gga` is the residual solver's own Hubbard refusal, entry on `jvp='sternheimer'`).
- Missing: `adddvhubscf`: the self-consistent `dns` and its `dV`, `dnsbare` (HOLES #16).
- Check: derivative: `test-suite/ph_U_insulator_us/benchmark.out.git.inp=BN.phG.in.args=2` and `ph_U_insulator_paw/...BN.phG.in.args=2` (nspin = 1, fixed, `epsil`); datasets `B/N.pbe-n-rrkjus_psl.1.0.0.UPF` and `B/N.pbe-n-kjpaw_psl.1.0.0.UPF` are in neither `tests/data/pseudo/` nor `$QE/pseudo/`, so a fetch comes first.
- QE/Elk: QE y (`ph.x` refuses U only for non-atomic projectors, `lda_plus_u_kind /= 0`, background, `elph`, `lraman`).

## T2: an exact internal identity, no new reference

Physics regimes first (an `nspin`, spinor, supercell or `nosym` identity), then execution modes with a bit-level identity, then Goldstone, Larmor and the time-shift checks, then the fixed-point and resumed-run identities, and last the identities that need another entry here to land first.

### I6. Shift current x `nspin = 2`

- Axis: spin.
- Guard: `defuqumat/response/photocurrent.py:shift_current:581`.
- Test: none.
- Missing: the loop over two channels whose currents add, which SHG already has (guard message).
- Check: a nonmagnetic cell at `nspin = 2` equals `nspin = 1`; on a magnet the per-channel sum against SHG's own channel sum.
- QE/Elk: QE n; Elk unread.

### I7. Sum-over-states `chi_0` x `nspin = 2`

- Axis: spin.
- Guard: `defuqumat/tddft/chi0.py:require_a_sum_over_states_regime:339`.
- Test: none.
- Missing: the spin-tensor kernel (guard message).
- Check: nonmagnetic `nspin = 2` equals `nspin = 1`; then T3, the static limit against `tests/data/qe/reference.out.ph-o2-fixed-lsda` ("Dielectric constant", line 170), converged in `nbnd`.
- QE/Elk: QE out of scope (TDDFPT); Elk `tddftsplr` (tasks 330/331) per queue.md's reference-source lookup.
- Note: sweep-tddft ranks it tier 1 through the static limit; that is a `ph.x` quantity reached by an `nbnd` convergence, so T2 on the exact identity.

### I8. Sum-over-states `chi_0` x spinor (noncollinear, spin-orbit; the spiral branch is behind it)

- Axis: spinor.
- Guard: `defuqumat/tddft/chi0.py:require_a_sum_over_states_regime:346`; the spiral branch `:352` is unreachable as written, `noncolin` fires first.
- Test: none (`tests/regression/test_spinor_dielectric.py:494` `test_the_opt_in_is_what_lifts_the_refusal` matches the same words but fires the Sternheimer gate).
- Missing: two-component pair densities with a spin trace (guard message).
- Check: a nonmagnetic spinor without SOC equals `nspin = 1`; with SOC, T3, the static limit against the code's own spinor Sternheimer `eps_inf`.
- QE/Elk: QE out of scope; Elk `tddftlr` has no spin-orbit reference per queue.md's lookup.

### I9. Berry-phase polarization x `nspin = 2`

- Axis: spin.
- Guard: `defuqumat/workflows/polarization.py:run_polarization:105`.
- Test: `tests/unit/test_polarization_machinery.py:212` `test_two_spin_channels_are_refused_by_name`.
- Missing: a channel axis on the string loop, two folded phases summed with the `nspin = 1` doubling applied once (HOLES #9).
- Check: `tests/data/qe/alas-berry.in` at `nspin = 2` converging to m = 0 returns `reference.out.alas-berry` (electronic 0.00876, total -0.24124); a magnet needs a `pw.x` `lberry` LSDA run (T4).
- QE/Elk: QE y (`bp_c_phase.f90:181`, `:345` `nspin_lsda`).

### I10. Strain response, elastic constants, electrostriction, piezoelectric tensor x `nspin = 2`

- Axis: spin.
- Guard: `defuqumat/response/strain.py:_require_one_spin_channel:818`; `response/sternheimer.py:1679` for the unflagged callers (`electrostriction.py:969`, `piezo.py:667`, `workflows/piezo_ladder.py:210`).
- Test: `tests/regression/test_lsda_response.py:640` `test_an_unflagged_quantity_still_refuses_nspin_two`; `tests/regression/test_lsda_response.py:675` `test_the_dynamical_matrix_and_the_strain_response_still_refuse_nspin_two`; `tests/unit/test_piezo_machinery.py:169` `test_the_regimes_this_was_never_run_in_are_refused`.
- Missing: the spin axis on the strain coordinate's `dpsi + ort` block and multipliers; for third derivatives the third derivative of `E_xc^LSDA` in two channels (HOLES #19).
- Check: nonmagnetic `nspin = 2` equals `nspin = 1`; on a magnet T3, a finite difference of the `nspin = 2` stress (pw.x-validated to 2.7e-7); no norm-conserving LSDA insulator is committed.
- QE/Elk: QE n (no strain perturbation); Elk `piezoelt.f90` (task 380) has no spin check per queue.md's lookup.

### I11. Strain family (strain response, elastic constants, electrostriction, piezo) x nonmagnetic spinor

- Axis: spinor.
- Guard: `defuqumat/response/sternheimer.py:require_a_sternheimer_regime:1594` via `strain.py:289`, `electrostriction.py:969`, `piezo.py:667`; `forces/energy.py:reject_spinors:432` (the elastic constants reach the functional with `spinors=False`).
- Test: `tests/regression/test_spinor_dielectric.py:494` `test_the_opt_in_is_what_lifts_the_refusal`; `tests/unit/test_piezo_machinery.py:169` `test_the_regimes_this_was_never_run_in_are_refused`; `tests/unit/test_spinor_refusals.py:92` `test_energy_at_refuses_a_spinor_unless_asked`.
- Missing: each caller's assembly on a two-component state, and the elastic constants' first-order spinor wavefunctions (MAGNETISM-NEXT A, HOLES #20, #21).
- Check: `noncolin` without SOC equals `nspin = 1`; with SOC, T3, a five-point difference of the spinor energy.
- QE/Elk: neither.

### I12. Topological invariants x collinear `nspin = 2`

- Axis: spin.
- Guard: `defuqumat/workflows/topology.py:DFTSource.__post_init__:142`.
- Test: none.
- Missing: a per-channel definition (two Hamiltonians give two sets; spin Chern numbers) (guard message).
- Check: nonmagnetic `nspin = 2` per channel equals `nspin = 1`, an integer.
- QE/Elk: neither.

### I13. Topological invariants x potential-only meta-GGA (`tb09`, `bj06`)

- Axis: functional.
- Guard: `defuqumat/workflows/topology.py:DFTSource.states:261`.
- Test: none.
- Missing: none named; validation only, the bands being unaffected (guard message).
- Check: the states the Wilson loop builds reproduce the SCF's own eigenvalues; the invariant is an integer.
- QE/Elk: neither.

### I14. Berry-phase finite field (`lelfield`) x `nspin = 2`

- Axis: spin.
- Guard: `defuqumat/scf/berryfield.py:require_a_berry_field_regime:353` (one raise, reasons joined).
- Test: none for this reason (`tests/unit/test_berryfield.py:147` `test_three_directions_are_refused`, `:153` are `efield_cart` and the automatic grid).
- Missing: one string set per channel (guard message).
- Check: nonmagnetic `nspin = 2` equals `nspin = 1` on `test-suite/pw_electric/electric-1.in`; a magnet needs `pw.x` (T4).
- QE/Elk: QE y (`setup.f90:559-575` builds `nx_el` for `nspin = 2`).

### I15. Berry-phase finite field (`lelfield`) x spinor

- Axis: spinor.
- Guard: `defuqumat/scf/berryfield.py:require_a_berry_field_regime:353`.
- Test: none.
- Missing: the spinor duals (guard message).
- Check: a nonmagnetic spinor without SOC equals `nspin = 1` on `pw_electric/electric-1.in`.
- QE/Elk: QE y (`c_phase_field.f90:173`).

### I16. Collinear DFT+U x a group with time-reversed operations (`t_rev`)

- Axis: DFT+U.
- Guard: `defuqumat/scf/driver.py:Calculation._setup_hubbard:3183`.
- Test: none.
- Missing: the spin-index flip of `new_ns`'s `colin_mag == 2` branch (`hubbard/occupations.py:build_ns_symmetry`); the spinor branch has it (NONCOLLINEAR item 13).
- Check: the same cell with `nosym = .true.`, exactly.
- QE/Elk: QE y (`PW/src/new_ns.f90:168`, `(colin_mag==2) .AND. (t_rev(isym) == 1)`, also `new_nsb.f90:192`, `new_nsg.f90:306`).
- Note: queue.md says QE has no collinear `t_rev`; sweep-core says it does. Read: QE 7.5 has the branch, so a `pw.x` reference (T4) is possible beside the identity.

### I17. Symmetrised projected DOS x spin-orbit coupling

- Axis: spinor.
- Guard: `defuqumat/projwfc/projections.py:atomic_projections:293`.
- Test: `tests/unit/test_projwfc.py:204` `test_a_symmetrised_spinor_projection_is_refused_by_name`; `tests/regression/test_spinor_projection_symmetry.py:162` `test_spin_orbit_is_still_refused`; `tests/regression/test_spinor_pdos.py:292` `test_a_symmetrised_spinor_projection_is_refused`.
- Missing: `D^j` of `d_matrix_so` for the `|j m_j>` columns (`sym_proj_so`) (GAPS §3, NONCOLLINEAR item 13).
- Check: wedge against the whole unshifted grid; the whole-grid side is T1 against `reference.projwfc.pt-soc`, `reference.projwfc.pt-soc-paw`.
- QE/Elk: QE y (`PP/src/projections_mod.f90` `sym_proj_so`, `projwfc.f90:1045`).

### I18. Site angular momenta x fully relativistic ultrasoft or PAW dataset

- Axis: dataset (US/PAW, fully relativistic).
- Guard: `defuqumat/projwfc/angular_momentum.py:_refuse_what_is_not_written:278`.
- Test: `tests/unit/test_angular_momenta.py:348` `test_a_relativistic_ultrasoft_dataset_is_refused`.
- Missing: the spinor projector set with `qq_so` (`_as_spinors` orthogonalised against `_spinor_overlap`); the naive route over-counts the basis by two (AUGMENTATION-NEXT §1i, PAW-MODES 6).
- Check: `<L>` quenched exactly at `soc_scale = 0`, and `|<L>|` invariant under a rotation of the moment; T1 for the metric only, Loewdin charges against `reference.projwfc.pt-soc-paw` on `pt-soc-paw-nosym.in`.
- QE/Elk: QE n (`projwfc.x` gives charges, not a per-site `<L>`); Elk `LSJ.OUT` is a muffin-tin decomposition.
- Note: sweep-response ranks it tier 2 against `projwfc.x`; `projwfc.x` has no `<L>`, so only the metric has a reference, as queue.md says.

### I19. Nesting function x constrained `tot_magnetization`

- Axis: spin.
- Guard: `defuqumat/response/nesting.py:require_a_fermi_surface:194`.
- Test: `tests/unit/test_nesting_machinery.py:297` `test_it_refuses_two_fermi_levels`.
- Missing: `N(q)` resolved by channel pair at two Fermi levels (guard message).
- Check: the two levels made equal reproduce the shared-level result (this does not reach two distinct levels).
- QE/Elk: neither.

### I20. Force x spin spiral

- Axis: spiral.
- Guard: `defuqumat/forces/energy.py:reject_spinor_spiral:455`.
- Test: `tests/unit/test_spinor_refusals.py:117` `test_a_spiral_force_is_refused_by_name`.
- Missing: the nonlocal term with the pair `vkb(k+q/2)`, `vkb(k-q/2)`, as `forces/spiral.py` has for `dE/dq` (guard message).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry); `pw.x` on that supercell is T4.
- QE/Elk: QE n (no spirals); Elk unread.

### I21. Stress x spin spiral

- Axis: spiral.
- Guard: `defuqumat/stress/energy.py:require_a_differentiable_cell:116`; `scf/driver.py:Calculation.at_strain:3803`.
- Test: `tests/unit/test_stress_machinery.py:261` `test_a_spiral_is_refused` (calls `at_strain` directly; `tests/regression/test_stress.py:461` `test_an_input_asking_for_an_impossible_stress_warns_rather_than_raising` warns rather than raises).
- Missing: `q` is in lattice coordinates and turns with the strain (guard message).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE n; Elk unread.

### I22. Velocity operator and what is built on it (optical conductivity, effective mass, shift current, SHG) x spin spiral

- Axis: spiral.
- Guard: `defuqumat/response/velocity.py:159`; `response/conductivity.py:require_a_conductivity_regime:412`; `response/effmass.py:330`; `response/photocurrent.py:require_a_velocity_sum_regime:742`.
- Test: `tests/unit/test_conductivity_machinery.py:168` `test_a_spin_spiral_is_refused_for_its_two_spheres`; `tests/unit/test_photocurrent_machinery.py:230` `test_each_refusal_names_its_own_missing_term`; none for velocity and effective mass.
- Missing: `dH/dk` is a pair across the spheres at `k +- q/2` (guard messages).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE n; Elk unread.

### I23. Berry-phase polarization and orbital magnetization x spin spiral

- Axis: spiral.
- Guard: `defuqumat/workflows/polarization.py:run_polarization:112`; `workflows/orbital_magnetization.py:run_orbital_magnetization:116`.
- Test: `tests/unit/test_polarization_machinery.py:217` `test_a_spin_spiral_is_refused_by_name`; none for the orbital magnetization.
- Missing: the overlap between neighbouring k on two spheres is not one gather (guard messages).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE n; Elk unread.

### I24. Site angular momenta x spin spiral

- Axis: spiral.
- Guard: `defuqumat/projwfc/angular_momentum.py:269`.
- Test: none.
- Missing: projections on both spheres (guard message).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE n; Elk unread.

### I25. STM image, STS and both tunnelling maps x spin spiral

- Axis: spiral.
- Guard: `defuqumat/workflows/stm.py:_refuse_what_has_no_fermi_level:853` (callers `stm.py:160`, `:347`, `transport.py:196`, `:438`).
- Test: `tests/unit/test_stm_machinery.py:397` `test_a_spin_spiral_is_refused`.
- Missing: `|psi(r)|^2` of a two-sphere state is not the lattice-periodic object the sum builds (guard message).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE n; Elk STM on a spiral unread.

### I26. Real-time propagation x spin spiral

- Axis: spiral.
- Guard: `defuqumat/realtime/propagate.py:require_a_realtime_regime:190`.
- Test: none.
- Missing: the two spheres at `k +- q/2` under `k + kappa` (guard message).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE out of scope; Elk unread.

### I27. Meta-GGA x spin spiral

- Axis: spiral.
- Guard: `defuqumat/scf/driver.py:Calculation._require_meta_supported:3117`; duplicates `scf/density.py:spinor_band_kinetic_density:286`, `scf/potential.py:1143`.
- Test: none.
- Missing: a lattice-periodic `tau` from two spinor components on two spheres (PLAN.md index).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE n; Elk unread.

### I28. DFT+U x spin spiral

- Axis: spiral.
- Guard: `defuqumat/scf/driver.py:Calculation._setup_hubbard:3156`; restated by `workflows/spiral_soc.py:_refuse:332`.
- Test: none.
- Missing: a Hubbard projector built on each sphere (guard message).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE n; Elk: `spinsprl` with `dftu` not refused, `gendmatmt.f90` has no spiral branch, per queue.md's lookup.

### I29. Source-free exchange-correlation field (`nosource`) x spin spiral

- Axis: spiral.
- Guard: `defuqumat/scf/sourcefree.py:refuse_source_free:175`.
- Test: none.
- Missing: `div B_xc` in the laboratory frame against a rotating-frame magnetization (sweep-core).
- Check: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE n; Elk unread.

### I30. Ultracell x spin spiral

- Axis: spiral.
- Guard: `defuqumat/ultracell/driver.py:require_an_ultracell_regime:545`.
- Test: `tests/regression/test_ultracell.py:1082` `test_every_refusal_fires`.
- Missing: a second modulation of the same states (guard message).
- Check: the explicit supercell SCF of the same spiral and modulation (nested basis).
- QE/Elk: QE n; Elk accepts it untested, per the guard message.

### I31. Ultracell x gradient-corrected functional

- Axis: functional.
- Guard: `defuqumat/ultracell/potential.py:require_an_ultracell_functional:91`.
- Test: `tests/regression/test_ultracell.py:1109` `test_the_refusals_name_themselves`.
- Missing: the gradient of the ultracell density including the envelope's own gradient, `O(Q)` (guard docstring, PLAN.md P88).
- Check: the explicit supercell SCF of the same modulation, exact by the nested basis (P88).
- QE/Elk: QE n; Elk y (`gndstulr.f90:159` calls `potksu` with GGA, per queue.md's lookup).

### I32. Ultracell x DFT+U

- Axis: DFT+U.
- Guard: `defuqumat/ultracell/driver.py:require_an_ultracell_regime:540`.
- Test: `tests/regression/test_ultracell.py:1097` `test_the_hubbard_refusal_fires`.
- Missing: an occupation matrix per atom copy (guard message).
- Check: the explicit supercell SCF of the same modulation, exact by the nested basis (P88).
- QE/Elk: QE n; Elk `gndstulr`/`potksu` have no +U term and no refusal, per queue.md's lookup.

### I33. Ultracell x potential-only meta-GGA

- Axis: functional.
- Guard: `defuqumat/ultracell/potential.py:require_an_ultracell_functional:77`.
- Test: `tests/regression/test_ultracell.py:1114` `test_the_refusals_name_themselves`.
- Missing: `tau` from the states rather than the carried density (guard message).
- Check: the explicit supercell SCF of the same modulation, exact by the nested basis (P88).
- QE/Elk: QE n; Elk unread.

### I34. Ultracell x `nosource`

- Axis: spinor.
- Guard: `defuqumat/ultracell/potential.py:require_an_ultracell_functional:84`.
- Test: none.
- Missing: the `1/|G|^2` projection couples all cells, so the potential is not per cell (guard message).
- Check: the explicit supercell SCF of the same modulation, exact by the nested basis (P88).
- QE/Elk: QE n; Elk unread.

### I35. Ultracell x saw-tooth field (`tefield`)

- Axis: applied electric field.
- Guard: `defuqumat/ultracell/driver.py:require_an_ultracell_regime:507` (`refuse_sawtooth`).
- Test: none.
- Missing: a saw-tooth term in the ultracell potential (guard message).
- Check: the explicit supercell SCF of the same modulation, exact by the nested basis (P88).
- QE/Elk: QE n (no ultracell); Elk unread.

### I36. Ultracell x a reference converged under a magnetic field or constrained moment

- Axis: magnetic field.
- Guard: `defuqumat/ultracell/driver.py:run_ultracell:1193`.
- Test: `tests/regression/test_ultracell.py:1217` `test_a_field_converged_ground_state_is_refused`.
- Missing: the final field (after `reducebf` or fixed spin moment) carried into `dV` (guard message).
- Check: the explicit supercell SCF of the same modulation, exact by the nested basis (P88), under the same field.
- QE/Elk: QE n; Elk unread.

### I37. Velocity operator `dH/dk` x gamma-only storage

- Axis: execution mode.
- Guard: `defuqumat/scf/driver.py:Calculation.at_kcart:4420`.
- Test: `tests/regression/test_gamma_only.py:595` `test_the_velocity_operator_refuses_gamma_storage`.
- Missing: `dH/dk` is odd in G, so the stored half needs its own rule; measured 0.863 against 1.8e-15 (sweep-core, queue B).
- Check: the same cell with an explicit `K_POINTS automatic 1 1 1 0 0 0` on the whole sphere.
- QE/Elk: QE n (no velocity operator in `pw.x`).

### I38. Real-space wavefunction consumers (projected DOS, `chi_0`, magnons, STM, STS, transport) x gamma-only storage

- Axis: execution mode.
- Guard: `defuqumat/basis/gvectors.py:refuse_gamma_storage:359`, called from `tddft/chi0.py:322`, `tddft/spinchi0.py:235`, `workflows/stm.py:170`, `:348`, `workflows/transport.py:206`, `:439` and the projections.
- Test: `tests/unit/test_gamma_basis.py:207` `test_the_post_scf_consumers_refuse_the_half_sphere_by_name` (fires it); `tests/regression/test_gamma_only.py:359` `test_a_field_response_is_still_refused_under_gamma`; `tests/unit/test_sibling_refusals.py:486` `test_two_more_consumers_refuse_gamma_storage` (magnons); `test_gamma_basis.py:213` only reads the source.
- Missing: `psi(r)` is a bare `g_to_r` over the stored half sphere; needs the 2 Re(sum) minus G = 0 form (guard messages).
- Check: the same cell with an explicit `K_POINTS automatic 1 1 1 0 0 0` on the whole sphere.
- QE/Elk: QE y for the projection (`projwfc.f90` has `gamma_only` branches); the others QE n.
- Note: gamma x GGA (`basis/gradients.py:_reject_gamma_only:90`) and gamma x ultrasoft (`pseudo/augmentation.py:build_augmentation:1233`) are not live refusals: the dense set is never halved, and the full sphere is substituted with a warning (`driver.py:gamma_storage_is_consumable`).

### I39. Real-time propagation x gamma-only storage

- Axis: execution mode.
- Guard: `defuqumat/realtime/propagate.py:require_a_realtime_regime:194` (restates `refuse_gamma_storage` by hand).
- Test: none.
- Missing: `H(k + kappa)` is complex, so the half sphere cannot be stored; the fix is storing the full sphere (guard message).
- Check: the same cell with an explicit `K_POINTS automatic 1 1 1 0 0 0` on the whole sphere.
- QE/Elk: QE out of scope.

### I40. Optical conductivity x symmetry-reduced k-set

- Axis: execution mode.
- Guard: `defuqumat/response/conductivity.py:require_a_conductivity_regime:419`.
- Test: `tests/unit/test_conductivity_machinery.py:180` `test_a_symmetry_reduced_k_set_is_refused_because_sigma_xy_is_axial`; `tests/unit/test_conductivity_machinery.py:375` `test_the_ordinary_shifted_grid_is_seen_as_the_wedge_it_is`.
- Missing: an axial-vector average of the antisymmetric part (guard message).
- Check: the whole unshifted grid with `nosym` and `noinv` (an unshifted grid is closed under the point group, P24).
- QE/Elk: QE n.

### I41. Shift current and SHG x symmetry-reduced k-set

- Axis: execution mode.
- Guard: `defuqumat/response/photocurrent.py:require_a_velocity_sum_regime:748`; re-raised by `response/shg.py:797`.
- Test: `tests/regression/test_photocurrent.py:363` `test_a_symmetry_reduced_wedge_is_refused_by_name`; `tests/regression/test_shg.py:448` `test_a_symmetry_reduced_wedge_is_refused_by_name`.
- Missing: a rank-3 point-group average; `symmetrize_cartesian_tensor` exists (guard message).
- Check: the whole unshifted grid with `nosym` and `noinv` (an unshifted grid is closed under the point group, P24).
- QE/Elk: QE n.

### I42. Sum-over-states `chi_0` x symmetry-reduced k-set

- Axis: execution mode.
- Guard: `defuqumat/tddft/chi0.py:require_a_sum_over_states_regime:372`.
- Test: `tests/unit/test_tddft_machinery.py:159` `test_a_reduced_k_set_is_refused_before_anything_is_computed`.
- Missing: symmetrising `chi_0(G, G')`, a rotation of both G indices at once (guard message).
- Check: the whole unshifted grid with `nosym` and `noinv` (an unshifted grid is closed under the point group, P24); the `si-epsilon-unshifted` and `-nosym` pair.
- QE/Elk: QE out of scope.

### I43. Transverse spin susceptibility (magnons) x symmetry-reduced k-set

- Axis: execution mode.
- Guard: `defuqumat/tddft/spinchi0.py:require_a_transverse_regime:308`.
- Test: none.
- Missing: `X_0(G, G')` symmetrised over the group of `q`, and closure under `+q` (sweep-tddft).
- Check: the whole unshifted grid with `nosym` and `noinv` (an unshifted grid is closed under the point group, P24); `ni-fcc-magnon.in`.
- QE/Elk: QE out of scope.

### I44. Clamped-ion elastic constants x symmetry-reduced k-set

- Axis: execution mode.
- Guard: `defuqumat/response/elastic.py:_require_a_closed_grid:412`; `response/electrostriction.py:_require_closed_for_elastic:1217`.
- Test: `tests/regression/test_electrostriction.py:572` `test_the_elastic_constants_are_still_refused_on_a_wedge`.
- Missing: the energy functional symmetrises its own density as a scalar inside the chain rule (guard message).
- Check: the whole unshifted grid with `nosym` and `noinv` (an unshifted grid is closed under the point group, P24).
- QE/Elk: neither.

### I45. Phonon at `q != 0` x symmetry-reduced k-set (small group of `q`)

- Axis: execution mode.
- Guard: `defuqumat/response/phononq.py:require_a_two_sphere_regime:1450`.
- Test: `tests/regression/test_phonons_at_q.py:397` `test_a_reduced_k_set_is_refused_by_name`.
- Missing: symmetrisation over the small group of `q` (guard message).
- Check: the whole unshifted grid with `nosym` and `noinv` (an unshifted grid is closed under the point group, P24).
- QE/Elk: QE y.

### I46. Tunnelling spectrum (STS) x symmetry-reduced k-set

- Axis: execution mode.
- Guard: `defuqumat/workflows/stm.py:_refuse_a_reduced_k_set:640`.
- Test: `tests/unit/test_stm_machinery.py:567` `test_a_wedge_is_refused_because_nothing_here_symmetrises_it`.
- Missing: nothing symmetrises a point-sampled sum of `|psi(r)|^2` (sweep-workflows).
- Check: the `grid=` whole-grid run of the same cell.
- QE/Elk: QE n.

### I47. Vertical, momentum-resolved and ultracell transport x symmetry-reduced k-set

- Axis: execution mode.
- Guard: `defuqumat/workflows/transport.py:_refuse_a_k_set_this_cannot_sum:1413` (callers `transport.py:274`, `:518`, `ultracell.py:329`).
- Test: `tests/regression/test_transport.py:620` `test_a_symmetry_reduced_k_set_is_refused`; `tests/regression/test_ultracell_sts.py:543` `test_a_wedge_is_refused_and_the_whole_grid_is_not`.
- Missing: the subgroup that keeps the exit plane in place, not the whole point group (guard message).
- Check: the `nosym` run of the same cell.
- QE/Elk: QE n.

### I48. Force-theorem, torque and relaxed anisotropy off the system's axis x symmetry-reduced k-set

- Axis: execution mode.
- Guard: `defuqumat/workflows/anisotropy.py:_force_theorem:567`; `_require_nosym_to_turn:1027`; `_refuse_relaxed:2130`.
- Test: `tests/regression/test_anisotropy.py:325` `test_turning_the_axis_needs_nosym_at_every_entry_point`; `tests/regression/test_anisotropy.py:327` `test_turning_the_axis_needs_nosym_at_every_entry_point`; `tests/regression/test_anisotropy.py:617` `test_a_direction_other_than_the_system_s_own_needs_nosym`.
- Missing: the magnetic group of each direction; the two directions are sampled on different wedges (sweep-workflows).
- Check: the `nosym` run of the same cell.
- QE/Elk: QE n (no anisotropy workflow).

### I49. Real-time propagation with the field's little group x shifted Monkhorst-Pack grid

- Axis: execution mode.
- Guard: `defuqumat/workflows/realtime.py:_kset:146`.
- Test: none.
- Missing: a shifted grid is not closed under the point group, so its wedge does not sum the current (P24's rule) (sweep-workflows).
- Check: `little_group=False`, the whole shifted grid, which runs.
- QE/Elk: QE out of scope.

### I50. Spin spiral x crystal symmetry (only `nosym` runs)

- Axis: execution mode.
- Guard: `defuqumat/scf/driver.py:Calculation.__init__:2891`; `system/builder.py:_spiral_q:2455`.
- Test: `tests/unit/test_geometry_invalidation.py:133` `test_a_spiral_built_without_the_reader_still_refuses_symmetry`; `tests/regression/test_spin_spirals.py:277` `test_what_a_spiral_refuses`.
- Missing: the spin-space group: operations with `S^T q = q` acting with their own spin rotation (sweep-core).
- Check: the `nosym` run.
- QE/Elk: QE n; Elk unread.

### I51. Partial dynamical matrix (`atoms=`) x symmetry

- Axis: execution mode.
- Guard: `defuqumat/response/phonon.py:_require_a_subsettable_regime:1742`.
- Test: `tests/regression/test_phonons.py:759` `test_a_subset_of_a_symmetrised_run_is_refused`.
- Missing: `symdvscf` acts on the `3 nat` stack as one object; a subset needs the stabiliser of the subset (guard message).
- Check: the subset block of the whole-cell matrix, or the subset with `nosym`.
- QE/Elk: QE n (no subset).
- Note: queue.md excludes it as not fixable; sweep-response lists it. The quantity is defined and only the speed-up is refused, so it is kept as an execution mode.

### I52. Every `Calculator.get_*` beyond SCF, forces, stress and relaxation, and the response stack x k-point pools

- Axis: execution mode.
- Guard: `defuqumat/calculator.py:guarded:2691` (wrapper on 60 of 64 `get_*`, `POOL_AWARE` at `:2680`); `response/efield.py:574`.
- Test: `tests/unit/test_parallel.py:215` `test_the_facade_refuses_what_is_not_pool_aware`.
- Missing: a pool-aware reduction in each workflow (comment at `calculator.py:2675-2679`).
- Check: one pool against N on the same input; easiest `get_bands`, `get_dos`, `get_nscf`.
- QE/Elk: QE y (pools everywhere).

### I53. Residual (Newton) SCF solver x k-point pools

- Axis: execution mode.
- Guard: `defuqumat/scf/driver.py:_refuse_under_pools:7101`.
- Test: `tests/unit/test_parallel.py:117` `test_refusals_name_what_pools_do_not_cover`.
- Missing: it differentiates the whole set (guard message).
- Check: one pool against N.
- QE/Elk: QE n (no residual solver).

### I54. Partial dynamical matrix x k-chunked response

- Axis: execution mode.
- Guard: `defuqumat/response/phonon.py:410`.
- Test: none.
- Missing: unread.
- Check: the whole-store run, bit for bit.
- QE/Elk: QE n.

### I55. Self-consistent frequency-domain hierarchy x two FFT grids (norm-conserving, `ecutrho > 4 ecutwfc`)

- Axis: execution mode.
- Guard: `defuqumat/realtime/hierarchy.py:hierarchy_linear_self_consistent:663`.
- Test: none.
- Missing: the dense-to-smooth interpolation of `dv` at every product (guard message).
- Check: a norm-conserving density is band-limited at `4 ecutwfc`, so the two-grid run equals the one-grid run to round-off.
- QE/Elk: QE out of scope.
- Note: `realtime/dense.py:dense_first_order:161` refuses the same and is a reference tool for the tests, not a feature; folded in here.

### I56. Berry-phase finite field (`lelfield`) x streamed wavefunction store

- Axis: execution mode.
- Guard: `defuqumat/scf/driver.py:run_scf:8294`; `forces/chunked.py:101`.
- Test: none.
- Missing: unread.
- Check: the in-memory run.
- QE/Elk: QE n (no streamed store).

### I57. Berry-phase finite field (`lelfield`) x `K_POINTS automatic`

- Axis: execution mode.
- Guard: `defuqumat/system/builder.py:_berry_field:2353`.
- Test: `tests/unit/test_berryfield.py:153` `test_an_automatic_grid_is_refused`.
- Missing: `kpoint_grid_efield` and the string set it builds (sweep-core).
- Check: the same strings given as an explicit list.
- QE/Elk: QE y.
- Note: queue.md files it under plain missing features with `efield_cart`; `efield_cart` (three directions, `l3dstring`) stays there.

### I58. Transverse spin susceptibility (magnons) x applied field or constrained moment

- Axis: magnetic field.
- Guard: `defuqumat/tddft/spinchi0.py:require_a_transverse_regime:281`.
- Test: none (`tests/unit/test_sibling_refusals.py:486` `test_two_more_consumers_refuse_gamma_storage` imports the guard; not shown to fire this branch).
- Missing: the Zeeman gap; the kernel `B_xc/m` rests on global spin-rotation invariance, which the field breaks (guard message).
- Check: Larmor's theorem: the `q = 0` mode at the Zeeman energy of the applied field.
- QE/Elk: QE out of scope; Elk unread.

### I59. Transverse spin susceptibility (magnons) x noncollinear ground state (the spiral branch is behind it)

- Axis: spinor.
- Guard: `defuqumat/tddft/spinchi0.py:require_a_transverse_regime:254`; the spiral branch `:268` is unreachable as written.
- Test: `tests/unit/test_magnon_machinery.py:199` `test_a_noncollinear_run_is_refused`.
- Missing: the 4x4 spin-density response, which no longer block-diagonalises; the cheaper half is a spiral at its own `q` in the rotating frame (MAGNETISM-NEXT D).
- Check: a collinear moment run as a spinor (no SOC) reproduces the `nspin = 2` magnon; Goldstone; T4 against Elk tasks 330/331.
- QE/Elk: Elk y (`tddftsplr.f90` `ncmag` branches, `genspchi0.f90:147`, per queue.md's lookup).

### I60. Transverse spin susceptibility (magnons) x PAW

- Axis: dataset (PAW).
- Guard: `defuqumat/tddft/spinchi0.py:require_a_transverse_regime:244`.
- Test: `tests/regression/test_ultrasoft_magnon.py:161` `test_paw_is_still_refused`.
- Missing: the one-centre part of `B_xc` in the kernel `B_xc/m`; the matrix element is already augmented (MAGNETISM-NEXT H, D).
- Check: the Goldstone identity `X_0(q=0, w=0) B_xc = m` (`goldstone_residual`; ultrasoft nickel 0.071 to 0.0128 with cutoff, so read at a converged cutoff).
- QE/Elk: QE out of scope; Elk y (LAPW).
- Note: `tddft/spinkernel.py:51-67` still names ultrasoft as a reason for refusal, which is stale since P93.

### I61. Transverse spin susceptibility (magnons) x DFT+U

- Axis: DFT+U.
- Guard: `defuqumat/tddft/spinchi0.py:require_a_transverse_regime:274`.
- Test: none.
- Missing: the Hubbard term's response to `m` in the enhancement; `spinkernel.py` says a nonlocal piece (guard message).
- Check: the Goldstone identity `X_0(q=0, w=0) B_xc = m` (`goldstone_residual`; ultrasoft nickel 0.071 to 0.0128 with cutoff, so read at a converged cutoff); it fails without the term, so the check is sharp.
- QE/Elk: QE out of scope; Elk unread.

### I62. Transverse spin susceptibility (magnons) x `nosource`

- Axis: spinor.
- Guard: `defuqumat/tddft/spinchi0.py:require_a_transverse_regime:300`.
- Test: none.
- Missing: a tensor kernel, since the projection makes `B_xc` not parallel to `m` (guard comment).
- Check: the Goldstone identity `X_0(q=0, w=0) B_xc = m` (`goldstone_residual`; ultrasoft nickel 0.071 to 0.0128 with cutoff, so read at a converged cutoff).
- QE/Elk: QE out of scope; Elk unread.

### I63. Real-time kick after `t0` x ultrasoft or PAW

- Axis: dataset (US/PAW).
- Guard: `defuqumat/realtime/propagate.py:require_a_smooth_field:1012`.
- Test: none.
- Missing: `_Chunk.jump` applied at a step inside the run as well as at the start (sweep-tddft).
- Check: a kick at `t1` gives the current of a run started at `t1`, shifted in time.
- QE/Elk: QE out of scope.

### I64. Real-time propagation x a ground state under a magnetic field or constrained moment

- Axis: magnetic field.
- Guard: `defuqumat/realtime/propagate.py:require_a_realtime_regime:206`.
- Test: none.
- Missing: the frozen potential is rebuilt from the input field, while `reducebf` and the fixed-spin-moment scheme change it (guard message).
- Check: a zero-amplitude pulse leaves the states stationary, which fails today exactly because of the missing term.
- QE/Elk: QE out of scope.

### I65. Checkpoint x fixed-spin-moment (feedback field) run

- Axis: execution mode.
- Guard: `defuqumat/scf/checkpoint.py:save_state:238`.
- Test: `tests/unit/test_checkpoint.py:253` `test_a_driven_field_is_refused_because_no_input_determines_it`.
- Missing: the controller's driven field in the file (sweep-core).
- Check: resumed against uninterrupted.
- QE/Elk: QE n.

### I66. Checkpoint x real-time run with the potential updated

- Axis: execution mode.
- Guard: `defuqumat/realtime/propagate.py:propagate:1157`.
- Test: `tests/unit/test_realtime.py:362` `test_what_the_updated_potential_refuses_is_refused_by_name`.
- Missing: saving every state and two potentials (guard message).
- Check: resumed against uninterrupted, bit for bit.
- QE/Elk: QE out of scope.

### I67. Continuation across the spiral boundary (spiral source and plain target, or the reverse)

- Axis: execution mode.
- Guard: `defuqumat/scf/continuation.py:_check_spiral:453`.
- Test: `tests/unit/test_continuation_machinery.py:801` `test_a_vector_magnetization_does_not_cross_into_a_spirals_frame`; `tests/unit/test_continuation_machinery.py:812` `test_a_spirals_magnetization_does_not_cross_out_of_its_frame_either`.
- Missing: the frame rotation of the transverse magnetization by `q.r` (sweep-core).
- Check: the continued run reaches the fixed point of a fresh one.
- QE/Elk: QE n.

### I68. Wavefunction seed held under a different pool layout

- Axis: execution mode.
- Guard: `defuqumat/scf/driver.py:starting_wavefunctions:6161`.
- Test: `tests/unit/test_parallel.py:555` `test_a_span_from_another_pool_layout_is_refused_by_name`.
- Missing: redistribution of rows between layouts (sweep-core).
- Check: the same converged state as an unseeded run.
- QE/Elk: QE n.

### I69. Elk `STATE.OUT` seed x spin-polarized state

- Axis: execution mode.
- Guard: `defuqumat/io/elk.py:read_elk_state:458`; `io/elk_density.py:density_on:576`.
- Test: `tests/unit/test_elk_reader.py:326` `test_a_state_carrying_physics_this_cannot_transfer_is_refused`; `tests/regression/test_elk_seed.py:240` `test_a_spin_polarized_run_cannot_take_an_unpolarized_seed`.
- Missing: a transfer rule for the magnetization (PLAN.md P72).
- Check: seeded and unseeded runs reach the same fixed point.
- QE/Elk: Elk writes it.

### I70. Elk `STATE.OUT` seed x DFT+U or a fixed tensor moment

- Axis: execution mode.
- Guard: `defuqumat/io/elk.py:read_elk_state:467`.
- Test: `tests/unit/test_elk_reader.py:326` `test_a_state_carrying_physics_this_cannot_transfer_is_refused` (parametrized on `dftu` and `ftmtype`).
- Missing: reading the trailing occupation and potential records (guard message).
- Check: seeded and unseeded runs reach the same fixed point.
- QE/Elk: Elk writes it.
- Note: sweep-core says no test; the parametrized test covers it, read.

### I71. Residual (Newton) solver x tetrahedra or `from_input` occupations

- Axis: execution mode.
- Guard: `defuqumat/scf/residual.py:_weights:511`.
- Test: `tests/regression/test_scf_solvers.py:335` `test_unsupported_occupations_are_refused_by_name`.
- Missing: differentiable tetrahedron weights (sweep-core).
- Check: the same fixed point as the mixing loop, to `conv_thr`.
- QE/Elk: QE n.

### I72. Residual (Newton) solver x spin spiral

- Axis: execution mode.
- Guard: `defuqumat/scf/residual.py:make_residual:551`.
- Test: none.
- Missing: unread.
- Check: the same fixed point as the mixing loop, to `conv_thr`.
- QE/Elk: QE n.

### I73. Residual (Newton) solver x applied or constraining magnetic field

- Axis: execution mode.
- Guard: `defuqumat/scf/residual.py:make_residual:544`.
- Test: none.
- Missing: a field that is a function of the state; the field secant sits outside the density (sweep-core, queue A13).
- Check: the same fixed point as the mixing loop, to `conv_thr`.
- QE/Elk: QE n.

### I74. `mixing_space = 'g'` x magnetic spiral

- Axis: execution mode.
- Guard: `defuqumat/scf/driver.py:run_scf:7880`.
- Test: `tests/unit/test_spiral_gvectors.py:109` `test_g_space_mixing_is_refused_on_a_magnetic_spiral`.
- Missing: the transverse channel on the sphere displaced by `q` (sweep-core).
- Check: the same fixed point as `mixing_space = 'r'`.
- QE/Elk: QE n.

### I75. `mixing_mode = 'ldos'` x tetrahedra, and as the warm-up of a Newton-Krylov solve

- Axis: execution mode.
- Guard: `defuqumat/scf/driver.py:run_scf:7975` (`ValueError`); `Calculation.ldos_weights:5443`; `_solve_residual:6734`.
- Test: `tests/unit/test_ldos_mixing.py:341` `test_tetrahedra_are_refused`; `tests/unit/test_ldos_mixing.py:355` `test_places_without_an_ldos_refuse_the_mode`.
- Missing: a smooth Fermi-level delta for tetrahedra; a Gaussian substitute is unmeasured (sweep-core).
- Check: the same fixed point as the mixing loop, to `conv_thr`.
- QE/Elk: QE n (`local-TF` is a different mixer).

### I76. `mixing_mode = 'robust'` x k-point pools, and x a checkpoint or `mixing_from`

- Axis: execution mode.
- Guard: `defuqumat/scf/driver.py:_refuse_robust:7158`.
- Test: none for these clauses (`tests/unit/test_robust_mixing.py:118` `test_tetrahedra_are_refused_by_name` fires the tetrahedron clause, which is a refusal by definition).
- Missing: the supervisor's state in the checkpoint; pool support (docstring at `:7140-7146`).
- Check: the same fixed point as the mixing loop, to `conv_thr`; resumed against uninterrupted.
- QE/Elk: QE n.

### I77. Ultracell x `mixing_mode = 'robust'` and `'ldos'`

- Axis: execution mode.
- Guard: `defuqumat/ultracell/driver.py:run_ultracell:1087`; `:1093`.
- Test: `tests/unit/test_ldos_mixing.py:352` `test_places_without_an_ldos_refuse_the_mode` (`ldos`); none for `robust`.
- Missing: the robust supervisor lives in `run_scf`'s loop; an LDOS handed to the mixer (guard messages).
- Check: the ultracell fixed point does not depend on the mixer.
- QE/Elk: QE n.

### I78. `rotate_moments` x crystal symmetry

- Axis: execution mode.
- Guard: `defuqumat/scf/driver.py:_refuse_rotating_moments:6957` (`ValueError`).
- Test: none.
- Missing: rebuilding the magnetic group and the k-set as the moments turn (sweep-core).
- Check: the `nosym` run.
- QE/Elk: QE n.

### I79. Gamma phonon x magnetic spinor (`noncolin` with `domag`), with the axial symmetrisation of a displacement response

- Axis: spinor.
- Guard: `defuqumat/response/sternheimer.py:require_a_sternheimer_regime:1594`; `scf/driver.py:Calculation.symmetrize_atom_displacement:5328`.
- Test: as the nonmagnetic spinor phonon; none for `:5328` (the `axial vector` match in `test_conductivity_machinery.py:180` is the conductivity's).
- Missing: everything of the nonmagnetic spinor phonon plus the second linear system for the time-reversed partners (`nsolv = 2`, Urru and Dal Corso), and for ultrasoft `int3_nc_save`, `dbecsum_aux` (HOLES #17); the axial rotation with `det(R)` and the `t_rev` sign.
- Check: after the `nspin = 2` Gamma phonon: a collinear `nspin = 2` dynamical matrix against the same cell as a spinor with the moment along z; `test-suite/ph_Ni_nc_spinorbit_mag` is at `q != 0`, a metal, with an uncommitted dataset.
- QE/Elk: QE y for phonons (`dfpt_kernels.f90:212-213`); `phq_readin.f90:546-548` refuses `epsil`, `zue`, `lraman`, `elop` under `magnetic_sym`, which sweep-response reads as never firing because `magnetic_sym` is assigned later (`phq_setup.f90:211`, see the textured-spinor dielectric entry); PAW refused at `:807`.

### I80. Phonon at `q != 0` x nonlinear core correction

- Axis: dataset (NLCC).
- Guard: `defuqumat/response/phononq.py:require_a_two_sphere_regime:1441`.
- Test: none.
- Missing: `dynmatcc` at `q` (`set_drhoc(xq)`, `dynmatcc.f90:105`), the one frozen term that depends on `q` (PLAN.md index).
- Check: a commensurate `q` equals the Gamma phonon of the supercell, whose route carries the core correction.
- QE/Elk: QE y (`dynmatcc.f90`).
- Note: queue.md ranks it T1 on `test-suite/ph_2d`; its `scf.in` sets `assume_isolated='2D'`, which is refused (plain missing feature), so that reference is not usable. sweep-response's identity is the check.

### I81. Phonon at `q != 0` x ultrasoft or PAW (and the electron-phonon coupling on it)

- Axis: dataset (US/PAW).
- Guard: `defuqumat/response/phononq.py:require_a_two_sphere_regime:1413`.
- Test: none.
- Missing: four terms: `addusddens`'s `Q_ij(q+G)` in the response density, `adddvscf`'s `int3` with `e^{iqr}`, `dvanqq`'s `int1`/`int2`, the multipliers at `q != 0`; `q_ij(q+G)` exists in `tddft.spinchi0.augmentation_factors` (PLAN.md index P97, AUGMENTATION-NEXT §1c).
- Check: a commensurate `q` equals the Gamma phonon of the supercell (ultrasoft and PAW insulators run at Gamma); T1 only stacked with `nspin = 2` and a metal (`test-suite/ph_base/ni.phX.in`, `nipaw.phX.in`).
- QE/Elk: QE y, including `elphon.f90:79-82`.

### I82. Phonon at `q != 0` x `nspin = 2` (and noncollinear)

- Axis: spin.
- Guard: `defuqumat/response/phononq.py:require_a_two_sphere_regime:1434`.
- Test: none.
- Missing: the `k+q` ground state, projector and response density per channel, plus the Gamma assembly's spin axis (guard message).
- Check: after the `nspin = 2` Gamma phonon: supercell identity; T1 once the ultrasoft `q` phonon also lands, `test-suite/ph_base/ni.phX.in` (`Ni.pbe-nd-rrkjus.UPF`, committed).
- QE/Elk: QE y.

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

### I85. Berry-phase finite field (`lelfield`) x spin spiral

- Axis: spiral.
- Guard: `defuqumat/scf/berryfield.py:require_a_berry_field_regime:353`.
- Test: none.
- Missing: unread (reason string "a spin spiral").
- Check: after `lelfield` on a spinor: a commensurate spiral equals the noncollinear supercell, exactly (P19, 1e-12 Ry).
- QE/Elk: QE n; Elk unread.

## T3: an approximate identity, or one that needs a convergence

### I86. Gamma phonon x ultrasoft or PAW metal

- Axis: dataset (US/PAW) and occupations (metal).
- Guard: `defuqumat/response/phonon.py:_require_a_moving_overlap_regime:1883`.
- Test: `tests/regression/test_phonons.py:630` `test_an_ultrasoft_metal_is_refused`.
- Missing: which weight (`wg` or `2 wk`) the three tangents `dpsi^ort`, `becsumort`, `dLambda` take in P28's split; QE's `addusdynmat`, `drhodvus` (HOLES #12, PAW-MODES 7).
- Check: a finite difference of the pw.x-validated ultrasoft forces on `tests/data/qe/al2-metal.in` with `Al.pbe-n-rrkjus_psl.1.0.0.UPF`; T4, one `pw.x` and `ph.x` pair.
- QE/Elk: QE y.

### I87. Elastic constants x ultrasoft or PAW

- Axis: dataset (US/PAW).
- Guard: `defuqumat/response/elastic.py:require_a_measured_elastic_regime:390` (also from `electrostriction.py:986`).
- Test: none fires it (`tests/unit/test_sibling_refusals.py:222` `test_the_elastic_constants_call_the_refusal_that_names_them`/`:244` reads the source).
- Missing: `StrainResponse.ort` (written, not wired) and a strain-coordinate `multiplier_response`; measured 22 per cent on `C_1111`, 2.9 per cent on `C_1212` (PLAN.md P100, AUGMENTATION-NEXT §3b).
- Check: a five-point difference of the energy, the norm-conserving control at 6.2e-08 on the same script.
- QE/Elk: neither.

### I88. Elastic constants (strain response) x metal

- Axis: occupations (metal).
- Guard: `defuqumat/response/strain.py:289` via `sternheimer.py:require_a_sternheimer_regime:1668` (`_NO_METAL_YET`).
- Test: `tests/unit/test_sibling_refusals.py:307` `test_the_strain_response_refuses_a_metal_for_the_right_reason`.
- Missing: the Fermi-level shift and the `wg`/`2 wk` split carried onto the strain coordinate (GAPS §3).
- Check: a central difference of the pw.x-validated metal stress over strained cells, and a five-point energy difference, on `tests/data/qe/al2-metal.in`.
- QE/Elk: neither.

### I89. Piezoelectric tensor x PAW

- Axis: dataset (PAW).
- Guard: `defuqumat/response/piezo.py:require_a_measured_dataset:516`.
- Test: `tests/unit/test_piezo_machinery.py:456` `test_the_transcribed_route_refuses_paw_and_no_longer_refuses_ultrasoft`; `tests/unit/test_piezo_machinery.py:581` `test_an_ultrasoft_dataset_is_warned_about_its_mesh_and_paw_refused`.
- Missing: none named; a measurement on a non-centrosymmetric PAW crystal (ultrasoft lifted at 0.86 per cent; wedge completion 1.58e-6) (AUGMENTATION-NEXT §1h).
- Check: a central difference of the Berry-phase polarization over strained `tests/data/qe/alas-piezo-tiny-paw.in` cells (no `reference.out` committed).
- QE/Elk: QE n (no tensor); Elk `piezoelt.f90` (LAPW).

### I90. Piezoelectric tensor x polar crystal

- Axis: cell (polar).
- Guard: `defuqumat/response/piezo.py:require_a_nonpolar_crystal:374`.
- Test: `tests/regression/test_piezoelectric.py:358` `test_a_polar_crystal_is_refused_by_name`.
- Missing: the improper-to-proper correction `delta_ki P_j - delta_ij P_k`; the message saying no polarization exists is stale (`run_polarization`, `Calculator.get_polarization` exist) (sweep-response).
- Check: a Berry-phase finite difference under strain; `P` itself is T1 against `reference.out.alas-berry`.
- QE/Elk: QE n; Elk `piezoelt.f90`.
- Note: queue.md files it under plain missing features; it is a pair, nonpolar runs.

### I91. Shift current x ultrasoft or PAW

- Axis: dataset (US/PAW).
- Guard: `defuqumat/response/photocurrent.py:require_a_shift_current_regime:712`.
- Test: `tests/unit/test_photocurrent_machinery.py:230` `test_each_refusal_names_its_own_missing_term`.
- Missing: the k-derivative of the augmentation dipole, hence `d^2(beta)/dk_a dk_b` of a bare projector (PLAN.md P47, P94, P99, AUGMENTATION-NEXT §2).
- Check: a central difference in k of `augmentation_connection`; the norm-conserving limit; SHG on the same states for the dipole.
- QE/Elk: QE n; Elk unread.

### I92. Shift current and SHG x DFT+U

- Axis: DFT+U.
- Guard: `defuqumat/response/photocurrent.py:require_a_velocity_sum_regime:767` (SHG re-raises at `response/shg.py:797`).
- Test: `tests/regression/test_photocurrent.py:391` `test_dft_plus_u_does_not_fall_through_to_a_ragged_error`; `tests/unit/test_photocurrent_machinery.py:230` `test_each_refusal_names_its_own_missing_term`.
- Missing: the Hubbard term's velocity and second velocity from `wfcU(k+G)` with the converged `ns` (guard message).
- Check: the velocity against a finite difference of `H(k)` including `vhpsi`.
- QE/Elk: neither.

### I93. Intraband (Drude) term of the optical conductivity x tetrahedra

- Axis: occupations (tetrahedra).
- Guard: `defuqumat/response/conductivity.py:_fermi_surface_delta:953`.
- Test: none.
- Missing: a Fermi-surface delta from the tetrahedron weights (guard message).
- Check: the `degauss -> 0` limit of the smeared plasma frequency on the same metal.
- QE/Elk: unread.

### I94. Sum-over-states `chi_0` x ultrasoft or PAW

- Axis: dataset (US/PAW).
- Guard: `defuqumat/tddft/chi0.py:require_a_sum_over_states_regime:331`.
- Test: `tests/unit/test_tddft_machinery.py:165` `test_an_ultrasoft_dataset_is_refused_by_name`.
- Missing: `Q_ij(G)` in every pair density did not close twice: P40 2.1 per cent, P94 57.200 with a residual +0.540 that does not fall with `nbnd`; the head lacks `-e_n dS/dk` and `dpqq` (PLAN.md P40, P94, AUGMENTATION-NEXT §1a).
- Check: RPA `eps_M(0)` against `dielectric_tensor(screening='hartree')` on `si-us-nosym.in` (norm-conserving control 0.06 per cent), converged in `nbnd`; that lands on `reference.out.ph-si-epsilon-us`, `-paw`.
- QE/Elk: QE out of scope; Elk LAPW only.
- Note: sweep-tddft ranks it tier 1 through the static limit; an `nbnd` convergence stands between, so T3.

### I95. Sum-over-states `chi_0` x smearing on a gapped cell

- Axis: occupations (smearing).
- Guard: `defuqumat/tddft/chi0.py:require_a_sum_over_states_regime:365` (`_is_an_insulator:382` reads only the `occupations` string).
- Test: none.
- Missing: the occupation test that exists at `_occupied_bands:687` in place of the string test (sweep-tddft).
- Check: the spectrum of the `fixed` run to within the smearing tail, on `si-epsilon-unshifted-nosym` with smearing.
- QE/Elk: QE out of scope.

### I96. Sum-over-states `chi_0` x metal

- Axis: occupations (metal).
- Guard: `defuqumat/tddft/chi0.py:require_a_sum_over_states_regime:365`; `_occupied_bands:687`.
- Test: none.
- Missing: the intraband (Drude) term, absent from `f_i - f_j` for `i = j` (guard message).
- Check: the Drude weight against the optical conductivity's intraband term; the f-sum rule.
- QE/Elk: QE out of scope; Elk `tddftlr` does not add it either, per the guard message.

### I97. ALDA kernel x gradient-corrected functional

- Axis: functional.
- Guard: `defuqumat/tddft/kernels.py:alda_matrix:200`.
- Test: none.
- Missing: an adiabatic GGA kernel with the gradient terms (the `dgcxc` route the Sternheimer stack already differentiates) (guard docstring).
- Check: the static limit against the code's own PBE Sternheimer `eps_inf`, converged in `nbnd`; T1 only stacked with ultrasoft (`reference.out.ph-alas-epsilon-us`, PBE).
- QE/Elk: QE out of scope.
- Note: Only in sweep-tddft. The meta-GGA half is refused by `ph.x` as a reference.

### I98. Real-time propagation with the potential updated x ultrasoft or PAW

- Axis: dataset (US/PAW).
- Guard: `defuqumat/realtime/selfconsistent.py:require_a_potential_mode:110`.
- Test: `tests/unit/test_realtime.py:340` `test_what_is_refused_is_refused_by_name`.
- Missing: the augmentation charge of `rho(t)` from `becsum` at `k + kappa(t)`, `newd`'s `D(t)`, PAW's one-centre `D(t)` (PLAN.md index).
- Check: the linear limit against the optical conductivity (runs on ultrasoft and PAW since P99), and `eps(w -> 0)` of the kick spectrum against the code's own Sternheimer `eps_inf` (`reference.out.ph-si-epsilon-us`, `-paw`), at finite propagation time.
- QE/Elk: QE out of scope; Elk task 460 (LAPW).
- Note: sweep-tddft ranks it tier 1 through the static limit; a finite propagation stands between, so T3.

### I99. Real-time propagation with the potential updated (`hxc`) x meta-GGA

- Axis: functional.
- Guard: `defuqumat/realtime/selfconsistent.py:require_a_potential_mode:117`; `realtime/hierarchy.py:659` is unreachable behind it.
- Test: none.
- Missing: `tau(t)` propagated; for `tb09`/`bj06` there is no conserved energy (guard message).
- Check: the linear limit against the optical conductivity under the same functional.
- QE/Elk: QE out of scope.

### I100. Real-time propagation x DFT+U

- Axis: DFT+U.
- Guard: `defuqumat/realtime/propagate.py:require_a_realtime_regime:200`.
- Test: none.
- Missing: the Hubbard projectors' row at `k + G` moving with `k + kappa` at a frozen `ns`; refused until its linear check is run (guard message).
- Check: the linear limit against the optical conductivity, which accepts a U.
- QE/Elk: QE out of scope; Elk task 460 includes DFT+U (`tddft.f90:66-70`, per queue.md's lookup).

### I101. Real-time propagation x static saw-tooth (`tefield`)

- Axis: applied electric field.
- Guard: `defuqumat/realtime/propagate.py:require_a_realtime_regime:184` (`refuse_sawtooth`).
- Test: none.
- Missing: the length gauge (saw-tooth) and the velocity gauge (laser) in one Hamiltonian, unmeasured (guard docstring).
- Check: a zero-amplitude pulse gives zero current on a saw-tooth ground state; the linear response equals the field-free run plus the static polarization.
- QE/Elk: QE out of scope.

### I102. Frequency-domain hierarchy with an induced potential x orders 2 and 3

- Axis: functional (potential updated).
- Guard: `defuqumat/realtime/hierarchy.py:_orders_at:304`; `workflows/realtime.py:run_nonlinear_spectrum:680`.
- Test: `tests/regression/test_realtime_hierarchy.py:244` `test_the_updated_potential_above_first_order_is_refused`.
- Missing: the induced potentials at `2w` and `3w`, each a fixed point per frequency, and the velocity gauge's second-order density artefact (7 per cent of rho on 2x2x2) (sweep-workflows, HARMONICS-NEXT).
- Check: the time-domain orders with the potential updated (`propagate_orders_self_consistent`, `get_harmonic_orders(potential='hxc')`).
- QE/Elk: QE out of scope.
- Note: queue.md files it under plain missing features; the frozen-potential orders run, so it is a pair.

### I103. Berry-phase finite field (`lelfield`) x ultrasoft or PAW

- Axis: dataset (US/PAW).
- Guard: `defuqumat/scf/berryfield.py:require_a_berry_field_regime:353`.
- Test: none.
- Missing: the cross augmentation `q_dk` with the structure phase in the string overlap (`h_epsi_her_set`), and `forces_us_efield` (guard message, PLAN.md index).
- Check: `dP/dE` against the ultrasoft dielectric constant (pw.x and ph.x validated); the zero-field limit against `run_polarization`; T4 `pw.x` (no ultrasoft `lelfield` reference committed).
- QE/Elk: QE y (`c_phase_field.f90` `okvan` branches).

### I104. Any Sternheimer response x saw-tooth or Berry-phase field (`tefield`, `lelfield`)

- Axis: applied electric field.
- Guard: `defuqumat/response/sternheimer.py:require_a_sternheimer_regime:1565` via `scf/sawtooth.py:refuse_sawtooth`.
- Test: `tests/unit/test_sawtooth.py:245` `test_gate_is_refused`.
- Missing: unchecked against any reference (sweep-response).
- Check: a finite difference of the forces under the field.
- QE/Elk: unread.

### I105. Dielectric tensor and Born charges x textured spinor (`nspin_mag = 4` with a magnetization)

- Axis: spinor (magnetic).
- Guard: `defuqumat/response/efield.py:require_a_measured_spinor_response:1306`.
- Test: `tests/regression/test_spinor_dielectric.py:332` `test_a_textured_spinor_is_refused_by_name`.
- Missing: unlocated: on `i-atom-soc.in` the component along the moment is 1.574482 against `ph.x`'s 1.494594 (5.3 per cent), across 4.6e-4; P104 cleared the magnetic kernel (0.238 per cent), leaving the field source term and `ph.x` itself (MAGNETISM-NEXT A2).
- Check: a finite-field difference of the dipole on the same atom, independent of both codes (whether `tefield` runs on a spinor is not checked; if not, T4).
- QE/Elk: QE ran it, but `phq_readin.f90:546` checks `magnetic_sym` before `phq_setup.f90:211` assigns it, so its own refusal never fires and its number is a suspect.
- Note: queue.md calls it T1; there is no `reference.out.ph-i-atom-soc*` in `tests/data/qe/`, the `ph.x` numbers are only in `notes/MAGNETISM-NEXT.md:180`. sweep-response is right.

### I106. Energy derivatives under a field or constrained moment: forces, stress, `dE/dq`, force-theorem and relaxed anisotropy

- Axis: magnetic field.
- Guard: `defuqumat/forces/energy.py:reject_magnetic_field:545`; `stress/energy.py:require_a_differentiable_cell:122`; `forces/spiral.py:_require_a_differentiable_spiral:600`; `workflows/anisotropy.py:_refuse_system:445`, `_refuse_relaxed:2097`.
- Test: `tests/regression/test_spiral_relaxation.py:270` `test_a_gradient_is_refused_where_it_would_be_silently_wrong`; `tests/unit/test_sibling_refusals.py:254` `test_every_force_method_passes_the_same_guards` reads the source.
- Missing: the field or penalty energy inside the differentiated functional; it is deliberately outside the reported total (`scf/fields.py`) (guard messages).
- Check: a finite difference of the total including the penalty.
- QE/Elk: QE not a reference: `add_bfield.f90` prints `etcon` and leaves it out of `etot` (`electrons.f90:1133-1163`), `forces.f90` has no constraint term.

### I107. Source-free exchange-correlation field x PAW

- Axis: dataset (PAW).
- Guard: `defuqumat/scf/sourcefree.py:refuse_source_free:147`.
- Test: `tests/regression/test_source_free_run.py:160` `test_paw_is_refused`.
- Missing: one Poisson equation for `div B` across the smooth and the one-centre parts (AUGMENTATION-NEXT §1f).
- Check: `div B_xc = 0` after projection on the combined field; the norm-conserving limit; T4 Elk `nosource`.
- QE/Elk: QE n; Elk y (LAPW).

### I108. `noncollinear_gga = 'grad_m'` x `nosource`

- Axis: functional.
- Guard: `defuqumat/scf/potential.py:refuse_noncollinear_gga:725`.
- Test: `tests/unit/test_noncollinear_gga_grad_m.py:185` `test_the_choice_is_refused_where_it_would_mix_two_functionals`.
- Missing: the projection checked for a field with a transverse part; only a field parallel to `m` was checked (PLAN.md index).
- Check: `div B_xc = 0` after projection; agreement with the `grad_m` field when `m` is collinear.
- QE/Elk: QE n; Elk unread.

### I109. Source-free exchange-correlation field x potential-only meta-GGA

- Axis: functional.
- Guard: `defuqumat/scf/sourcefree.py:refuse_source_free:182`.
- Test: none.
- Missing: unread (message: both are potentials with no energy functional behind them).
- Check: `div B_xc = 0` after projection.
- QE/Elk: QE n.

### I110. Exchange-correlation spin torque x potential-only meta-GGA

- Axis: functional.
- Guard: `defuqumat/scf/spin_torque.py:torque_of_result:182`.
- Test: none.
- Missing: `tau` from the states rather than the density (sweep-core).
- Check: the torque against `m x B_xc` evaluated on the converged fields.
- QE/Elk: QE n.

### I111. Fixed-spin-moment feedback constraint x `lspinorb`

- Axis: spinor.
- Guard: `defuqumat/system/builder.py:build_system:1263` (`ValueError`).
- Test: `tests/unit/test_magnetic_fields.py:693` `test_the_fixed_spin_moment_search_refuses_spin_orbit_coupling`.
- Missing: a secant with a full 3x3 `dm_a/dB_b` (sweep-core).
- Check: a penalty scheme on the same moment; T4 Elk `fsmtype` with spin-orbit (unread).
- QE/Elk: QE n.

### I112. `rotate_moments` x DFT+U

- Axis: DFT+U.
- Guard: `defuqumat/scf/driver.py:_refuse_rotating_moments:6945`.
- Test: none.
- Missing: turning `ns`'s spin structure with the step (sweep-core).
- Check: the minimum of the relaxed anisotropy, which accepts a U.
- QE/Elk: QE n.

### I113. `rotate_moments` x applied `B_field`

- Axis: magnetic field.
- Guard: `defuqumat/scf/driver.py:_refuse_rotating_moments:6951`.
- Test: none.
- Missing: the Zeeman torque in the step (for a penalty constraint the refusal is physical) (sweep-core).
- Check: the relaxed anisotropy energy plus the Zeeman term, minimised by hand.
- QE/Elk: QE n.

### I114. `soc_scale` strictly between 0 and 1 x ultrasoft or PAW

- Axis: dataset (US/PAW).
- Guard: `defuqumat/scf/driver.py:Calculation.__init__:2670`; `pseudo/spinorbit.py:__init__:300`.
- Test: `tests/unit/test_spiral_soc.py:176` `test_an_intermediate_soc_scale_on_a_norm_conserving_dataset_is_the_blend`; `tests/regression/test_anisotropy.py:402` `test_an_intermediate_soc_scale_is_refused_on_an_augmented_dataset`.
- Missing: a blend of the spin-orbit overlap `qq_so`; only 0 and 1 are defined (guard message).
- Check: the two endpoints, and the shape in `soc_scale` against the norm-conserving blend.
- QE/Elk: QE n (no `soc_scale`).
- Note: Only in sweep-core.

### I115. Ultracell transport, `exit_region = 'volume'` x ultrasoft or PAW

- Axis: dataset (US/PAW).
- Guard: `defuqumat/workflows/ultracell.py:_refuse_an_augmented_volume_diagnostic:548`.
- Test: `tests/regression/test_ultracell_augmented.py:1029` `test_the_augmented_exit_plane_transmission_is_the_unit_cell_s_tiled`.
- Missing: `S` in an ultracell state's plane-wave representation across N unit-cell spheres (sweep-workflows).
- Check: `exit_region = 'plane'` on the same dataset; the norm-conserving limit.
- QE/Elk: QE n.

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
- Ultracell with ultrasoft or PAW: P88 stage 5; ultracell at `nspin = 2` and `4` runs (`ultracell/driver.py:487`; `notes/ELK-FEATURES.md` §9 is stale).
- Real-time propagation at a frozen potential on ultrasoft, PAW, `nspin = 2` and spinors: P139 to P141 (`.claude/rules/scope.md`'s real-time bullet is stale).
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
