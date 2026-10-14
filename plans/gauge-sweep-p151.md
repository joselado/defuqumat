# P151 plan: the gauge sweep over the front door (`notes/DEGENERACY-NEXT.md` item 7)

Asked by the user on 2026-10-15 ("do the first two points": the sweep, and `OPEN.md` Part XXVI
item 3, the latter already done and passing on D22: `fermi_level_shift` takes the charge with
`charge_channel`'s rule, unit test with a 0.3 e charge and 0.7 magnetization row at nspin_mag=4).

## What is measured

For each of three cells with real multiplets, and for each `Calculator.get_*` that the cell admits,
the observable is computed in four arms on one converged ground state:

- `plain-1`, `plain-2`: unrotated, twice (the determinism floor of the method, same cores);
- `rot-A`, `rot-B`: two independent seeds of the rotation below.

Per numeric leaf of the returned object (dataclass / dict / tuple / array, walked recursively):
`d_X = max|O_X - O_plain1| / max(max|O_plain1|, tiny)`. A leaf whose `d_A`, `d_B` are at the
`plain-2` floor is gauge invariant at that number; a leaf that moves by a similar size under both
seeds and not under `plain-2` is gauge dependent, and is then classified by hand: covariant by
construction (wavefunctions, a band-resolved matrix inside a multiplet) or a rule-D4 defect.

## The injection (the mechanism, which is what decides it)

Two points, both active in the rotated arms only:

1. **The converged states.** The SCF is run once, unrotated, in a parent `Calculator`. Each arm
   gets a fresh `Calculator` from the same text whose one-slot cache is filled by hand
   (`_scf`, `_scf_options`) with a copy of the `SCFResult` whose `wavefunctions` are rotated.
   A fresh calculator per arm, because `_strain_response` and `_ultracell` are caches too, and a
   response built in the plain arm and reused in a rotated one is a null test.
2. **Every eigensolve a consumer makes.** `defuqumat.scf.driver.get_eigensolver` is patched to
   return the registered solver wrapped so that its returned eigenvectors are rotated. Every
   `Calculation(...)` (nscf, shg, sfac, nesting, conductivity, magnons, vc_relax build their own)
   gets the wrapper at construction, and `streaming.py`, which calls `calculation.eigensolver`
   directly, goes through it too. A consumer that re-diagonalises (P149: `starting_wavefunctions`
   is a span) is then rotated after its own solve, which is the place the rotation means
   something.

The rotation, per channel and k: `U = expm(i (A o M))`, `A` a fixed-seed random Hermitian
`nbnd x nbnd`, `M_ij = |e_i - e_j| < tol` with `stop_gradient`, so `U` is unitary, block diagonal
on the multiplets, and a constant under differentiation. It is written in `jnp` so that it traces
inside a `jit` or a `jvp` (some consumers diagonalise under a transform). The diagonal of `A`
also puts a random phase on every singlet; **open question 1** is whether that is in the same arm
or a separate one.

`tol`: a fixed 1e-6 Ry, with the wrapper logging per solve the largest gap counted as degenerate and
the smallest one not, so a run whose spectrum has no clean separation at 1e-6 is visible as such
(mixing two bands 1e-6 apart is not a gauge transformation and would read as a defect of size
tol/gap).

**Positive control, per cell** (a null result must be told from a pass): the density of the first
member of each multiplet built from `SCFResult.wavefunctions` must move by O(1) under `rot-A`, and
the eigenvalues of every wrapped solve must not move at all.

## What the injection does not reach

Dense `eigh` calls outside Davidson: `topology/berry.py`, `topology/states.py`, `response/effmass.py`
(2), `response/phonon.py`, `response/phononq.py`, `transport/green.py`, `ultracell/driver.py`,
`ultracell/kramers.py` (2), `forces/torque.py`, `scf/orientation.py`, `realtime/dense.py` (2),
`realtime/propagate.py`, `hubbard/*`, `scf/continuation.py`. Some diagonalise a small matrix built
from the states (an overlap, a 2x2 spin matrix, a k-derivative block), which is a second gauge
choice of its own. **Open question 2**: reach them in this pass (a per-module `jnp` proxy whose
`linalg.eigh` rotates its output the same way), or record them as not reached.

## The cells

- `si-gamma`: `si2-lda-gap.in` at 2x2x2 with `Gamma`, `nbnd = 10` (threefold blocks at `Gamma`,
  pairs at `L`), NC, fixed.
- `alas-soc`: `alas-epsilon-us-soc.in` at 2x2x2, US, spin-orbit, nonmagnetic (Kramers pairs
  everywhere).
- `o-smear`: the smeared oxygen atom without `tot_magnetization` (the 2p triplet at E_F), nspin=2.

Every `get_*` is called on every cell with the cheapest arguments that exercise it (a k-path of a
few points, three frequencies, a small `q`, `supercell = (2,1,1)` and so on; the table is in the
harness). A refusal is recorded with its message. `get_relax`, `get_spiral_*`, `get_elk_seed`,
`get_piezoelectric_kmesh_ladder` run a new SCF or write files; **open question 3**: whether they are
in the sweep (their SCF goes through the wrapper, so their move is at `conv_thr` and the two seeds
measure that noise rather than a gauge defect).

## Where it lives, and what is committed

The harness is `tools/gauge_sweep.py` (committed: it is the instrument of a recorded number), run on
D22 with one capped process per cell. A gate test pins the mechanism on `si-gamma` only: the
positive control, plus two cheap observables that must be invariant (the DOS and the Berry-free
`get_band_velocities`, or whichever the sweep shows to be cheap and degenerate-sensitive).

**Open question 4**: a defect the sweep finds is fixed in this pass when the fix is a block average
in one assembly (rule D4), and recorded in `OPEN.md` with its size otherwise. Or record all, fix
none.

The outcome goes into `notes/DEGENERACY-NEXT.md` item 7, `PLAN.md` P151 and `OPEN.md` for anything
that moved.

## Fable's review (2026-10-15), and what was changed for it

The mechanism holds. Taken from the review into the harness before the night's run:

- The registry entries `solvers.EIGENSOLVERS["davidson"]` and `["david"]` are patched rather than
  `driver.get_eigensolver` (read at call time, covers a future importer).
- The tolerance is the code's own rule, not a fixed 1e-6: `resolve_degeneracy_tol(0, scf.ethr,
  empty_floor=True)` on the cached states (the empty bands are converged to the 1e-5 floor, and
  `si-gamma`'s top pair is split by 2.1e-6) and `resolve_degeneracy_tol(0, ethr)` per wrapped solve.
- The multiplets are labels `cumsum(diff(e) >= tol)`, so `U` is block diagonal by construction; the
  mask `|e_i - e_j| < tol` is not transitive.
- The eigenvalue half of the positive control was a tautology. It is replaced by a reach column per
  method (the wrapper's solve count, and the reads of the cached `wavefunctions`; zero on both is
  "not reached") and by a second O(1) mover out of a re-diagonalising consumer
  (`fixed_density_states` in a gauged calculator).
- On `o-smear` the floor is not `plain-2`: a band-weighted leaf moves by about split/degauss under a
  rotation of the triplet at e_F. The harness prints the bands near e_F so the floor can be stated.
- A fourth cell, `si-gamma-nosym`, since the consumers fable ranks first (SHG, shift current,
  optical conductivity, absorption, the phonon at q) refuse a wedge.

Decisions: OQ1 the singlet phases are in the same arm. OQ2 the dense `eigh` sites are recorded as
not reached. OQ3 the calls that run their own SCF are out. OQ4 the sweep and its record are one
commit; a single-assembly rule-D4 defect is then fixed in a second commit with a before/after pair.
Dropped for cost: the real-time and harmonic calls, Raman, the vibrational spectrum,
electrostriction, the magnetoelectric tensor, the ultracell, the relaxed family and the full
`get_phonons`.

## Fable's decisions on the fixes (2026-10-15, after the sweep)

The principle: a first-order diagonal in a degenerate level has a gauge-free answer, the
eigenvalues of the block (degenerate perturbation theory: the slopes of the split bands, the
deformation potentials); a second-order per-band tensor has none, which is why the effective
mass's NaN and the block eigenvalues are the same rule.

- **A. Velocities and the strain `deigenvalues`**: block eigenvalues, sorted within the block, one
  spectrum per Cartesian direction and per strain pair `ab`. The trace is kept, so
  `inverse_mass_sum` is unchanged; NaN would poison it along Delta (the doublet stays degenerate at
  `k +- h`), and the block average reads 0 at X where the slopes are `+-v`. `BandVelocities` gains a
  `degenerate` mask, `speeds` is NaN there, and the docs say that inside a multiplet a row is three
  spectra, not one band's vector. The kernel `VelocityOperator.band_velocities` keeps the raw
  diagonal; the grouping lands in the wrapper `band_velocities` and after both strain routes,
  through one shared label rule `cumsum(diff(e) > tol)` beside `cut_gaps` in `scf/occupations.py`,
  which `effmass._multiplets` is rewritten over. Tolerance: the solver's promise, never 1e-5
  (an interband `<m|dH/dk|n>` between resolved bands is O(1)): on the `kpoints` route
  `fixed_density_ethr(conv_thr, nelec)`; on SCF states `result.ethr` with the empty floor; for the
  occupied-only strain array `resolve_degeneracy_tol(0, result.ethr)` without it, resolved in
  `Calculator.get_strain_response`.
- **B.** `asymmetry_by_spin` and `truncation_by_spin` NaN on refused bands.
- **C.** Warn in `optical_conductivity` when `band_cut_gap` is below the tolerance, reusing
  `photocurrent._warn_about_the_cut` as `shg.py` does.
- **D.** `tests/unit/test_effective_mass.py:241-243, 262-263` change to finite off `refused`;
  `tests/regression/test_conductivity.py:249-251` will warn and is wrapped in `pytest.warns`;
  notebook 19's velocities along Gamma-L change (pairs become sorted `+-v`) and it is re-executed.
