# The Record stage of the orbital-field work

Branch `mag-record`, from `mag-cross` 00ab8475 (which holds the foundation, both route U steps, both
route P steps and the crosscheck). Every number below was measured on D22 in the worktree
`/l/ladovj1/w-mag-record`; the logs are in `/l/ladovj1/review/mag/record/`, written `R/<log>`. No
fable subagent could be started from this stage (the Agent tool was not available to it), so every
open choice took the conservative option, and each is listed under "Decisions". The phases are
`PLAN.md` P175 (the foundation and Peierls), P176 (the magnetic ultracell, refused) and P177 (the
plane-wave magnetic Bloch route and the crosscheck), which hold the route records whole.

## The review's fixes

The review kept every branch: the foundation with no fix, route P after one fix, route U after two,
the crosscheck with three record corrections. There was no "drop" item, so nothing was left out.

- **Route U, fix 1** (`magnetic/ultracell_field.py`, `_ket_block`): the `-i div a` term is built with
  `jax.lax.complex` on the traced `c1 * b12` instead of the literal `1j`. `R/dtype.log`
  (`R/mag_dtype.py`): as the kernel is called (`b12` a Python float) the old line kept a complex64
  block complex64; with a float64 array `b12` it gave complex128; the new line gives complex64 both
  ways. New test `test_the_field_block_keeps_a_single_precision_dtype` (two forms); as called it would
  also pass on the old line, which is said so that its pass is not read as the evidence. The
  analytic-state test still reads 1e-12 (`R/t1.log`, `R/t3.log`).
- **Route U, fix 2**: `u2.md` ("The outcome") and `route-u.md` say that the default path raises on
  every acceptance case run, so what is merged is a measured negative result with its guards, plus the
  basis-independent pieces, and that the nine `route-u-q*` sets stay labelled unconverged.
- **Route P, the fix**: `test_the_lowest_landau_level_carries_one_chern_number` asserts `chern == +1`
  (it asserted `|C| = 1`). On its 4 x 4 x 8 box at flux 1/2 it reads +1 (`R/t1.log`, passed), so the
  FHS orientation needs no note. `test_magnetic_bloch.py` gained the `jax.clear_caches()` autouse
  fixture.
- **Route P, the record sentences**: `route-p.md` drops "sits just under the refuse line" (the line did
  not measure the projectors, whose part is 5.1e-7 Ry), names item 4's 1e-7 as the SCF floor
  (the trap of a tolerance its `conv_thr` does not deliver), and calls step 1's solver change a
  design change, measured, not a conservative choice.
- **Crosscheck, (a) (b) (c)**: `cross.md` leaves the attribution of the 0.2 eV open (the facts: 0.20
  to 0.22 eV, ratio to `E_1` 0.077 to 0.079 constant to 2 per cent while `B` falls by 1.67, so
  `sqrt(B)`, five alternatives excluded), says that "linear in B" was untestable at `q <= 10` by
  construction (`(spread/l_B)^2` about 1.5e-4 at 10 T and 0.1 at `q = 10`), and says that the two
  Wannier sets, refused under the strict 10 meV criterion, are within 11.4 meV (`p_z`) and 1.8 meV
  (8 functions) of the sphere inside `E_D +- 2.5` eV on the folded k.
- **The slow fold test is split** (`tests/regression/test_magnetic_bloch_slow.py`): the SCF at 1/3 is a
  module fixture with its own test, the check against the closed mesh another. The check alone took
  710 s (`P2/fold3b.log`), still above `slot.sh`'s 10-minute per-test timeout, so the file runs under
  `tools/run_regression.sh`.

## The facade

`Calculator.get_landau_levels`, `get_magnetic_bands`, `get_magnetizability`, delegations within the
20-line budget; `get_landau_levels` calls `conventions.refuse_regime` first. Tests:
`tests/unit/test_magnetic_facade.py` (four tests: the Peierls delegation and the tesla rounding, the
set with no edge, the regime refusals, the route P delegation with no SCF for the free problem, the
`nbnd` refusals).

**Two defects the facade's first runs found, both fixed.**

- `get_wannier` with explicit projections leaves `edge = None`, and `run_landau_levels` did
  `float(wannier.edge)`: `TypeError` in the first figure and notebook runs (`R/fig-graphene.log` first
  run, `R/nb.log` first run). `run_landau_levels` takes `fermi_energy`, refuses by name without an
  energy, and the facade passes the ground state's Fermi level only when the set has no edge.
- The orbit centre on a ribbon is `q (K2 - kappa2)` cells from the first cell, not `q (kappa2 - K2)`
  as the first docstring had it: `R/ribbon-time.log` (165 cells, 8 functions, 1/7900) and
  `R/snippets.log` against `R/snippets2.log` (330 cells, `p_z`, 1/7891: at `K2 + 165/q` no Landau
  level near `E_D`, at `K2 - 165/q` the torus's `n = +-1, +-2` to the printed digits).

## The numbers of this stage

- Graphene at 10 T through the facade (`R/snippets2.log`, slot 1, CPU 6): `p/q = 1/7891`, 10.000 T,
  dimension 15782, `v_F` 8.4207e5 m/s, `E_1` 96.48 meV against Dirac 96.62 (0.99860), `n = 10` 0.99314,
  `n = -1` 1.00136, `n = -10` 1.00213; zero level -66.7 ueV, two states, site traces (1, 1);
  Hermiticity 3.9e-16 Ry; plaquette `exp(-i 2 pi/7891)`. The 330-cell ribbon at `kappa2 = 0.31242`:
  -136.87, -96.75, -7.96, -0.07, 96.48, 136.33 meV.
- The free Landau problem at 1/10 through the facade, full box (ndim 92160), one k: levels over `B_ph`
  1, 2.46351843 (twice), 3, 4.46351843 (twice), the formula to the eight printed digits.
- Notebook 63 executed in **124 s** (`R/nb.log`, slot 1, CPU 6), exported (`R/nb-md.log`).
- The figures: `landau_graphene` (`R/fig-graphene.log`, 95 s), `landau_butterfly`
  (`R/fig-butterfly.log`, 64 s), drawn locally from the committed `data.npz`.
- **The magnetizability end to end** (`R/chi-facade.log`): the zero-field sphere SCF 74 s, then the three SCFs in one call, 4, 7 and 7 iterations; energies of the 24-atom cell `E(0)` = -273.10525908315896, `E(+)` = -273.0436060095016, `E(-)` = -273.0436061726208 Ry, the same as the three separate probe runs of item 4 (`P2/diamond-p0.log`, `diamond-p1.log`, `diamond-m1.log`) to every printed digit; `E(+) - E(-)` = 1.631e-07 Ry; **chi = -11.401 x 1e-6 cm^3/mol per C2, -5.700 per C atom** (Lu, LDA: -11.16); 5711 s in all, peak RSS 4.93 GB, the lock taken at 19:56:28 at a load of 3.6, which rose to 12 to 13 during the run. The worktree was at ac9eae0c when it started and was moved forward three times while it ran; the files changed after ac9eae0c (`workflows/landau.py`, `get_landau_levels` in `calculator.py`, tests, docs, probes) are not on the magnetizability path, and `calculator.py` was imported before the first move, so the run is the final code for that path. The guide's diamond block is that run: its code is the probe's calls, and its two output lines are the probe's values printed with the identical `.3e` and `.3f` formats (the probe's own lines carry more text after the same prefix).
- Timings (all in `PERFORMANCE.md`): route P at 1/6, second call **524.7 s** (first 564.5 s, Davidson
  steps equal), 7 iterations, peak 1.93 GB (`R/time-routep.log`, slot 0, CPU 4, load 3.6 rising to 12
  to 13); `pw.x` on the 6x1x1 supercell at zero field **82.9 s** with symmetry (11 k), **167.5 s**
  `nosym` (24 k), 10 iterations each (`R/pwx.log`, slot 1, CPU 6, load 6 to 13); the Peierls torus at
  1/7900, dimension 63200, second call **40.41 s**, peak 8.79 GB (`R/peierls-time.log`, big job on CPU
  0, load 1.5 to 3.6); the 165-cell ribbon **3.56 to 4.83 s** over eight calls (`R/ribbon-time.log`,
  load about 12), and not the torus to 4 meV at `5 l_B`; route U at `q = 100` not run (u2's cold
  numbers instead); the KPM against the shift-invert from `foundation/kpm.log` (52.3 s against 2.69 s).
- **Tests**: the six magnetic files and `test_calculator.py`, 123 passed and the one facade test since
  fixed (`R/t1.log`, 81.5 s); the facade, notebook conventions, facade documentation, ultracell and
  bloch files, **546 passed** in 54.6 s (`R/t3.log`); `mkdocs build --strict` and
  `tools/check_guide.py` pass here. **The fast gate** in one slot at 11bd76e5 (`R/gate.log`): `-m 'not slow'`, **3915 passed**, 85 skipped, 1731 deselected, and the two known failures of `tests/unit/test_check_guide.py` (D22's venv has no `yaml`), 1713.65 s (`# exit 1 after 1722 s`, slot 0, CPU 4, started 20:14:57, load 6 to 12); the review's gate at 00ab8475 read 3901 passed and the same two failures; the commits after 11bd76e5 change only `PLAN.md`, `PERFORMANCE.md`, `plans/`, `docs/guide/` and `notebooks/README.md`

## Decisions (all conservative, no fable could be started)

- `get_magnetizability` returns the `Magnetizability` object, not the design's float, since a float
  cannot carry the per-atom value and the asymmetry beside it.
- `get_magnetic_bands` and `get_magnetizability` require `nbnd` and refuse without it, rather than a
  default band count computed in the facade.
- Route U's timing at `q = 100` was not run (30 minutes and 6 GB under `big.lock` for a refused
  route); u2's measured `q = 10, 25, 50` costs stand in its place.
- Route P's timing pair takes the SCF's second call in one process and compares it with `pw.x` with
  and without symmetry, saying that the starts differ.
- `run_magnetizability` was run end to end once under `big.lock`, so that the guide's snippet is an
  executed one and the review's "has not run end to end" is closed.
- The pyqula ribbon pair of the plan is replaced by the torus against the ribbon of the same code.
- `graphene-scf.npz` stays in `plans/` (the review: "may want to" move it).

## What is left

`PLAN.md` P177, "What is outstanding (P175 to P177)", is the list. One small item of this stage: `HAS_A_FACADE_METHOD`
in `tests/unit/test_notebook_conventions.py` does not name the three new pairs; not added, so that no test
file changed after the gate. The slow files, listed and not run:
`tests/regression/test_magnetic_bloch_slow.py`, `tests/regression/test_magnetic_ultracell_slow.py`,
and for the hook `tests/regression/test_davidson_zero_root.py`, `test_gamma_only.py`,
`test_spinorbit.py`, `test_noncollinear_magnetism.py`.

## For `docs/d22-notes.local.md` (not edited here)

- Worktree `/l/ladovj1/w-mag-record` (detached at the bundled `mag-record`, `bundles/mag-record.bundle`),
  logs in `/l/ladovj1/review/mag/record/`; removable once the branch is merged, with `w-mag-foundation`,
  `w-mag-foundation-base`, `w-mag-u1`, `w-mag-u2`, `w-mag-p1`, `w-mag-p2`, `w-mag-cross`.
- A notebook run by `jupyter nbconvert --execute --inplace` in a worktree modifies a tracked file
  (the executed notebook) and leaves its untracked `.md` and `_files/`; the next `git checkout` of a
  bundle that adds them stops. `git checkout -- <notebook>` and removing the untracked exports first
  is the way; the same for a figure's `data.npz`.
- `big.lock` is held for the whole of a long big job (the diamond magnetizability, about an hour or
  more on four cores at a load of 12), so a second big job queues behind it; put the short ones
  first in the same script.
