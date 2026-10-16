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
