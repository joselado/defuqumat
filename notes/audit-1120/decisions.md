# Decisions taken without the user (audit fixes of 2026-11-20)

The user said on 2026-11-20 "Take the next decisions by yourself". Each entry is the choice,
the reason, and what would change it.

- **#9/#10 (round one), graphane at 20 bohr moves by 5.8e-8 eV in the gap.** Accepted as
  round-off and committed (b59f3277). The code before the fix gives exactly that difference
  between the performance and the efficiency cores, and the code after lands on the other
  end point to 1.9e-12 eV; the two face means of graphane differ by 4e-13 Ry. P182 records
  the gap to 1 meV. The slow `tests/regression/test_lmbj_vacuum.py` is the check.
- **Round two #14 (Thomas-Fermi tau seed at nspin_mag = 4), deferred.** The fix is 15 lines but
  changes the seed of every magnetic spinor or spiral meta-GGA run, and PLAN.md records three
  (TB09 ni-soc-nosym, I27, I110). The deciding measurement is the nickel pair with the old and
  new seed, four SCFs on the same cores. Kept deferred rather than moving recorded numbers.
- **Round two #5, #6, #34: refusals rather than new physics.** ATOMIC_FORCES (nonzero),
  CONSTRAINTS (nconstr > 0, also in an SCF), a nonzero TOTAL_CHARGE card, trism, nextffield,
  gate (even alone) and a saw-tooth field in single precision are refused by name. The full
  external-force fix is sized in the fix log (extfor on System, -sum F.tau in the energy, a slow
  CO relax against relax-extfor's benchmark).
- **Round two #26: the CLI's conv_thr fallback stays 1e-8** (the old CLI default), not pw.x's 1e-6,
  so no CLI output moves; an unconverged SCF in `dos`/`pdos` writes nothing and exits 1.
- Note for P183: `tests/regression/test_input_sweep.py` has no slow mark and is in the gate; the
  fix-log entries of round two #7 and #24 call it slow, which is wrong.
- **Round two #3 (rotate_moments drops the Anderson history under rho_ddot): the fix is held.**
  The patch `notes/audit-1120/round2-3-rotate-fit-vectors.patch` turns the fit vectors and is
  right by construction (fit vectors match a fresh evaluation to 6e-16), but on the oblique
  cobalt it moves the iterations from 59 to 48 on one arithmetic order and from 53 to 60 on
  another, so no saving is shown, and it would move I78 (53 iterations), I113 (25.974 and
  55.712 degrees), I112's 38.557 degree end state and the guide's "53 iterations". The call
  site's comment now says what the code does (it claimed the opposite).
- **Round two #17 and #27 (fit vector on the half sphere; tau on the smooth sphere under 'g'):
  deferred.** #17 needs a trajectory check and moves P146's recorded 4.77 GB; #27 fails the
  round-off rule because the Thomas-Fermi start of tau is not on the smooth sphere (2.4e-4 of
  its norm off it).
- **New, seen by the mixing agent:** at nspin_mag = 4 with rotate_moments, tau's vector part is
  not turned (the Route C comment says tau "is a scalar"), and nothing refuses a meta-GGA with
  rotate_moments. Handed to the xc group of round two to confirm and refuse.
- **Round two #3 patch refreshed** to apply on f3b7ba41 (`git apply --check` clean); the only
  conflict was the call-site comment of 2f0a776e, resolved to the patch's own text.
- **Round two #R2-extra: rotate_moments refuses any meta-GGA** (f92c02e3), rather than turning
  tau's vector part, which would be new code in the Route C state.
- **Round two #4: the fixed GGA axis follows pw.x's compute_ux literally** (12aa7918), with its
  dependence on the moment lengths (|m|^2 > 1e-12 seeds, |ux x m|^2 < 1e-6 counts as parallel).
  Kept because it is the reference code's rule and none of the 86 committed noncollinear inputs
  changes axis. What would change it: a case where two small cants written once as a card and
  once as per-species moments take two GGA branches and the user wants one rule.
- **Round two #15 landed although the ultrasoft nickel magnon moves** by +2.8e-6 Ry at
  (0,1/4,1/4) and +3.4e-6 Ry at (0,1/2,1/2): P90 records only the Goldstone residual of that cell
  (0.071), which is unchanged to the bit, and no magnon energy.
- **Round two #11 (kubler sphere on cut channels) and #21 (ddd_paw group average): held as
  patches**, `round2-11-kubler-cut-channels.patch` and `round2-21-symmetrize-ddd.patch`. #11
  reaches several slow PAW spinor files that only the user may run; #21 has three options (leave
  it, QE's average in the SCF only, average the energy) and the size is 3.2e-7 of 0.22 on AlAs.
  Leave both until the user has the slow numbers.
- **Round two #30 and #31: deferred** with their sizes (2.2 per cent of a PAW Si phonon; 12 and
  2.7 per cent of a three-q dispersion of hydrogen and ultrasoft nickel).
- **Seen, not followed (for OPEN.md):** on a 2x2x2 grid of hcp Al the wedge and nosym SCF
  energies differ by 6.8e-7 Ry. Not investigated; it may be the conv_thr of that run.
- **Gate cost added by the xc group:** 14 s (#15), 7 s (#12), 3.2 s (#4). Kept in the gate.
- **Round two #8 (balanced plane chunk) kept** (69c099c9). The transforms, the density and whole
  SCFs are bit-identical on si8-1k, si8-1k-ecut30 and si16-1k, and the fix log has the timing
  pairs (H psi on si8-1k 6.98 to 5.72 ms, median of 21; an SCF iteration 57.7 to 52.9 ms, median
  of 7). The card timings of 13 benchmark inputs now overstate defuqumat's time by a few per
  cent; PERFORMANCE.md gets a dated paragraph saying so rather than 13 retimed rows. What would
  change it: a reading of "never move a recorded number" that includes timings that only improve.
- **Round two #9 (SCF straggler allowance, pw.x's five bands): held** as
  `round2-9-scf-stragglers.patch`. No run moved, but it lets unsettled bands into the density in
  exactly the stall regimes PERFORMANCE.md records, and it shows no timing gain.
- **Round two #28 (xy pass by rows) and #29 (store vkb when nk = 1): deferred.** #28 is at most
  1 to 2 per cent and a loss on si8-1k-ecut30. #29 saves 5 to 8 per cent of an iteration at one
  k-point but costs 290 to 386 MB on the large one-k cells, which is the choice the memory default
  was made for; options (store at nk = 1, store under a byte budget, keep and note) go to the user.
- **Round two #20, open question for P183:** d719e1aa moved the Newton autodiff's tangent width
  from 1e-11 to 1e-5 (every production ethr is a row). Whether P148's autodiff numbers went
  through `ScfResidual.jvp` is not checked; the P148 record gets a "Corrected 2026-11-20"
  paragraph and the question.
- **Round three #1: the tetrahedra refuse the whole grid on a run with symmetry** (1f587ffb), with
  tetra.f90's words, rather than folding it with the identity, because pw.x refuses it too. All
  217 committed automatic inputs pass. Proposed for the end: one sentence in the
  `Calculator.with_kpoints` docstring (set nosym for that comparison).
- **Round three #26: the projected force theorem refuses a wedge at every direction**, the own
  axis included, without the torque's identity-and-inversion exception (ba31b484), because an
  inversion can swap two magnetic atoms. Wide on purpose.
- **Round three #2 (PAW third derivatives on a wedge): refused**, not fixed (8b8f031b). The audit's
  fix (average the value of becsum) closes the epsilon identity but makes the Raman tensor worse
  (1.77e-5 to 9.50e-5 off the whole grid); the complete fix, with a full-zone tangent of becsum,
  reproduces the whole grid to 2.0e-13 on the Raman tensor but is about 100 lines in three files
  and its strain half is not measured. Held as `round3-2-paw-wedge-becsum.patch` with the
  experiment's driver. Every recorded PAW third-derivative number is on si-paw-nosym, so none moves.
- **Round three #20 (PAW deformation potentials without dD_paw): landed** (48955549), although PAW
  `deigenvalues` move by up to 4.59e-3 of 1.13, because PLAN.md records no PAW deformation
  potential (I checked: P151, P174's table and the guide are norm-conserving, si-gamma) and the
  new value meets a finite difference of re-converged cells to 6.4e-6 where the old one missed by
  2.5e-3. Its test is in the slow `test_electrostriction.py`.
- **Round three #28 (Raman ort blocks on the host): deferred**, sized at 7.37 GB and 64 s before the
  first chunk on the P25 yardstick, with no round-off check on a real walked run.
- **Round three #3 (starting_ns_eigenvalue after a Newton solve) and #21 (spinor DFT+U force or
  stress under TENSOR_MOMENTS): refused by name** (c7832e0f, bc035fd8). The missing penalty term of
  #21 is 4.9e-3 Ry/bohr on BN and does not sum to zero; writing it is new physics. Seen and left
  for OPEN.md: every fresh run_scf applies starting_ns_eigenvalue again (pw.x resets it after its
  one ns_adj call), including the final SCF of run_vc_relax.
- **Round three #25 (spiral transverse tau on the resident sphere): landed** (a58a9b02). It moves
  only the double-grid case (ecutrho above 4 ecutwfc), which no committed meta-GGA spiral input
  sets; I27 at dual 4 is unchanged.
- **Round three #29 ('mixed' with Newton-Krylov): left as it is (option a), deferred.** The run
  always ends in double, so the result is right; the cost is 152 against 35 evaluations of F on
  si16. The refusal patch `round3-29-mixed-newton-refusal.patch` and a 3-line "Newton in double"
  option are for the user; the second needs a GPU timing.
- **Round three #5: a c= projection site is now read in angstrom**, as Wannier90 reads it
  (c19130e1). This changes what an existing Python input means (it was bohr), which I accepted
  because a .win written by write_w90 meant another site to wannier90.x, no committed file uses
  c=, and the Wannier guide's Limits note says so. wannier90.x 3.1.0 -pp on the written .win now
  gives the site defuqumat uses.
- **Round three #4: the Z2 refusal is wide** (e47da50a): an antiferromagnetic Z2 (time reversal
  times a half translation) is refused too. Deferred: a check that the Wannier centres come in
  Kramers pairs at the two ends of the pump, whose threshold needs the slow bismuthene and
  germanene flows.
- **Round three #6, the siblings:** the Kubo quantum metric and the frozen real-time routes crash
  the same way on PAW TB09. The two-line fix (`round3-6-onecenter-siblings.patch`) mirrors the
  call nscf.py already makes and can move no number (the routes raised before), so I hand it to
  group E to land with a test that fails on the old code.
- **Round three #9: a constant-current STM scan needs `heights`** (060b13f3), rather than a new
  default. The reader's half period fails on the guide's graphite cell; only a scan to the vacuum
  midpoint was right (1.5000064 against 1.5). A default that ends at the middle of the vacuum gap
  (about two hours with tests) is the user's choice. One full period is refused.
- **Round three #8: a spin spiral's structure factors are refused as a whole** (e65c87a2), the
  charge columns included, which were right. Keeping the charge and withholding the spin columns,
  or reporting the satellites H -/+ q (about half a day), is the user's choice.
- **Seen outside the note, for OPEN.md:** `run_ultracell_stm` passes `heights` (bohr along the
  normal) to `_refuse_an_augmented_plane` as crystal coordinates (`sampled += heights` in
  `workflows/ultracell.py`), so on an augmented dataset it checks the wrong tip positions.
- **Round three #6, the siblings, landed** in group E as 1024b5c5 with four tests in
  `tests/unit/test_optics_paw_meta.py`; the patch file was removed.
