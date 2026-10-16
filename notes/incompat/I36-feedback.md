# I36-feedback. The ultracell runs the fixed-spin-moment feedback and `reducebf` on the box

Status: done for the cell-wide scheme `'fsm'` under Elk's update rule and for `reducebf < 1`,
collinear. The user decided (2026-10-16) that the ultracell should run a feedback scheme on the
box rather than refuse it, which lifts the part of `I36.md` that was still refused. Still refused,
each by name and each with a test: the per-atom schemes `'atomic fsm'` and `'atomic fsm
direction'`, the secant rule `fsm_update = 'secant'`, a penalty, and a spinor reference under any
field.

## What changed

`defuqumat/ultracell/driver.py:run_ultracell` carries the reference's field as loop state, the
pair `(field, field_scale)` starting at `SCFResult.magnetic_field` and `SCFResult.field_scale`,
and moves it between iterations exactly as `run_scf` moves its own: after the mix, the scale is
multiplied by `reducebf` and the field takes Elk's step `B <- B - tau (M - M_target)` on the
**output** density of the iteration, never the mixed one. The frozen eigenvalues carry the
reference's field, so only the difference `tile(v_field(B, s) - v_field(B_ref, s_ref))` enters
`dV`, and the box's `magnetic_potential` is refreshed with the current field every iteration so
that `deband` and `field_energy` pair with the field the matrix was built from. The box density is
normalised per unit cell, so `MagneticField.total_moment(box_density, unit_cell)` reads the moment
per cell and the target is the unit cell's `fixed_magnetization`, unchanged; the supercell under
the same scheme targets `N` times it. The loop is converged only when `dr2 < conv_thr` **and**
`field.satisfied(output, cell)`, the second condition `run_scf` tests. `UltracellResult` now
carries `magnetic_field` and `field_scale`, the box's own converged pair, and a `reducebf` run that
converges with the field still on is warned by `run_scf`'s own `_warn_if_the_field_did_not_fade`.

## The numbers

Both against the explicit 2x1x1 supercell SCF of the same modulation under the same constraint, the
nested-basis identity of P88: the partly polarized hydrogen lattice of `I36.md` (`H.pz-vbc.UPF`,
`nspin = 2`, Gaussian 0.02 Ry), a 2x1x1 ultracell on `(2, 2, 2)` with the `0.05 cos` scalar
modulation, the band ladder 8, 16, 24 of the I31, I32 and I36 records, `mixing_mode = 'adaptive',
kerker = False` on the box and adaptive on the supercell. Charge and moment are relative to their
largest Fourier components, as in `I36.md`. D22, single runs, logs under
`/l/ladovj1/review/incompat/I36-feedback/`.

### `'fsm'`, target 0.626 mu_B per cell, `lambda = 0.25`, controller tolerance 1e-6

`r3-check.log`, commit d8bf94b. The unit cell converges to `B = +1.477889e-02` Ry, `M = 0.625999`,
in 112 iterations. The tiled null: `dV` 2.2e-16, density 1.1e-8, total +1.1e-10 Ry, one
iteration, field unchanged. The supercell ends at `B = +1.069693e-02` Ry, so the modulation moves
the field by `B_sup - B_ref = -4.082e-3` Ry. Per cell, with the energy taken as `E - B (M -
M_target)` on both sides (which removes the moment's miss to first order):

| nbnd | charge | moment | B_box - B_sup | M_box | energy above the supercell | iterations |
|---|---|---|---|---|---|---|
| 8 | 5.194e-4 | 6.240e-4 | +1.32e-5 Ry | 0.625999 | +3.598e-6 Ry | 97 |
| 16 | 2.344e-4 | 2.437e-4 | -9.4e-7 Ry | 0.625999 | +1.143e-6 Ry | 97 |
| 24 | 1.737e-4 | 1.443e-4 | -4.6e-6 Ry | 0.625999 | +6.000e-7 Ry | 97 |

The box's field agrees with the supercell's to 1.4e-5 Ry or better at every rung, 300 times below
the 4.1e-3 Ry the modulation moved it by; the field difference is not monotonic past `nbnd = 16`,
which is the controller's own resolution, `1e-6 / chi = 6e-6` Ry. Charge, moment and energy fall
with `nbnd` and the energy is above the supercell's at every rung, the variational bound at matched
moment. **The alternative the lift replaces reads differently**: the reference's field held fixed
on the box at `nbnd = 24` (the I36 path, a `MagneticField` with no constraint and the same `B`)
ends at `M = 0.629786` per cell, 3.8e-3 mu_B off the target, against below 1e-6 for the driven box.

### `reducebf = 0.8` under `B_field(3) = 0.02`

`r1-check.log` (5749b21) and `r2-check.log` (a5a564e), identical to every printed digit. The unit cell
ends at `field_scale = 5.903e-3`, a residual 1.181e-4 Ry, in 24 iterations; the null is `dV`
2.2e-16, total -2.6e-10 Ry; the supercell ends at `field_scale = 1.547e-3`. The box continues the
decay from the reference's scale and ends at 2.077e-4 (16 iterations at every rung):

| nbnd | charge | moment | total above the supercell |
|---|---|---|---|
| 8 | 5.174e-4 | 6.147e-4 | +3.575e-6 Ry |
| 16 | 2.344e-4 | 2.393e-4 | +1.140e-6 Ry |
| 24 | 1.740e-4 | 1.396e-4 | +6.013e-7 Ry |

All three fall and the total is above the supercell's at every rung. **This arm discriminates
weakly**: at `reducebf = 0.8` both sides land on the modulated field-free state with residual
fields of 4e-6 Ry (box) and 3.1e-5 Ry (supercell), and the frozen alternative would have held
1.2e-4 Ry, whose effect on the state is second order in the field (`FADED_FIELD`'s table: 1e-4 Ry
is near 1e-10 Ry in the total). What it does establish is that continuing the decay on the box
converges to the supercell under the same `reducebf`; it is not a measurement of a state with a
finite residual field, and a run that stops with one is warned.

## Two things the measurement found about the scheme, not about the box

- **The default gain cannot hold this cell, and neither can a large one.** The moment is stiff
  self-consistently, `chi = 0.17` mu_B/Ry (0.62343 free, 0.62510 under `B = +0.01`), and
  `B = -0.01` flips it to the other branch, so targets 0.4 and 0.5 never converged in 300 to 400
  iterations at gains 0.0025 to 0.02 (`d1.log`, `d2.log`). The box's first output at fixed input
  responds with `chi0` near 7 mu_B/Ry, so `lambda = 2`, which converges the unit cell in 17, threw
  the box's moment between +1 and -1 for 40 iterations (`d5.log`); 0.25 converges on both
  (`d6.log`). Elk's rule is stable for `tau chi0 < 2` and converges at the rate `tau chi`, and on
  the box the first is set by the modulation's transient, which the unit cell never sees.
- **`FSM_TOLERANCE = 1e-3` mu_B leaves the field undetermined by `1e-3 / chi = 6e-3` Ry on this
  cell**, above the 4.1e-3 Ry the modulation moves it by: at `lambda = 0.1` the unit cell stopped
  at `M = 0.62500`, a miss of exactly the tolerance (`d6.log`). The check is therefore taken with
  the tolerance patched to 1e-6 on all three runs (`mock.patch.object` on the test), which is what
  defines the state being compared. A user running the box at the default tolerance gets the same
  indeterminacy the supercell has, no more; whether the tolerance should be an input is a separate
  question and is not changed here.

## Refusals that still fire, each with a test

- per-atom feedback (`'atomic fsm'`, `'atomic fsm direction'`) and `fsm_update = 'secant'`:
  `test_ultracell_regimes.py::test_a_per_atom_or_secant_feedback_reference_is_refused` (replaces
  `test_a_feedback_or_fading_field_reference_is_refused`, whose `'fsm'` and `reducebf` arms are now
  measured instead);
- penalty: `test_a_penalty_converged_reference_is_still_refused` and
  `test_ultracell.py::test_a_field_converged_ground_state_is_refused`;
- spinor reference under a field: `test_a_spinor_reference_under_a_field_is_refused`.

`r1-fast.log` (5749b21) and `r3-fast.log` (d8bf94b): `-m 'not slow'` on both ultracell files, 14
passed, 38 deselected, no skips.

## User guide

`docs/guide/ultracell.md`, the Limits note. The paragraph that starts "A collinear ground state
converged under a fixed magnetic field" becomes

> A collinear ground state converged under a fixed magnetic field (`B_field`
> or a `LOCAL_MAGNETIC_FIELDS` card) runs. The code keeps the same field on
> every copy of the unit cell, so the result is the supercell under the same
> field. A ground state converged with `constrained_magnetization = 'fsm'`
> also runs. Here the long cell keeps moving the field until the moment per
> cell is on the target again, which is what a supercell under the same scheme
> does, because the modulation changes the moment. This uses the update of Elk
> (`fsm_update = 'elk'`) with the `lambda` of the unit cell as the gain. Keep
> the gain small, because the first steps of the long cell react much more
> strongly than the unit cell did: on a hydrogen lattice `lambda = 2` did not
> converge in the long cell and `lambda = 0.25` did. A ground state converged
> with `reducebf` below 1 also runs, and the long cell keeps reducing the field
> from where the unit cell stopped. The penalty schemes, the per-atom schemes
> `'atomic fsm'` and `'atomic fsm direction'`, and `fsm_update = 'secant'` give
> an error. A fixed total moment with `tot_magnetization` still runs, with one
> Fermi level for each spin channel. A noncollinear ground state converged
> under a field also gives an error. For these cases, you converge the unit
> cell without the field and give it with `magnetic_field` in the long cell
> instead.

## README

No cell changes: the ultracell row does not list field regimes.

## Slow tests for the user

- `tests/regression/test_ultracell_regimes.py::test_a_feedback_reference_converges_to_the_supercell_under_the_same_scheme` (new; its body ran as `r3-check.log`, 85 s)
- `tests/regression/test_ultracell_regimes.py::test_a_fading_field_reference_converges_to_the_supercell_under_it` (new; its body ran as `r1-check.log` and `r2-check.log`, 35 s)
- `tests/regression/test_ultracell_regimes.py::test_a_field_converged_reference_converges_to_the_supercell_under_it` (unchanged path for a fixed field, but the loop around it changed)
