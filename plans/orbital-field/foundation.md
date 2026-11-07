# The orbital-field foundation: what was built and what it measured

Branch `mag-foundation`, from 7d4851af. Every number below was measured on D22
(`ladovj1@D22-0161`), in the worktree `/l/ladovj1/w-mag-foundation`, through
`/l/ladovj1/review/guide/slot.sh` (one core, 6 GB) unless the line says otherwise. Logs are
in `/l/ladovj1/review/mag/foundation/`, written below as `L/<log>`. The load average was 7 to 15
from other users during the runs, so the times are a range, not a pair.

## What is here

- `defuqumat/magnetic/conventions.py`: `MagneticCell` (frozen `equinox.Module`: the chemical
  `Cell`, a traced `scale` on `a`, static `p`, `q`, `axis = 0`), with `b_chem`,
  `magnetic_lattice`, `magnetic_reciprocal`, `area`, `flux`, `b_ph`, `field_tesla`, `a(r)`,
  `kvector`, `peierls_phase`, `bloch_sum_phase`, `gipaw_phase`, `translation_phase`; and
  `minimum_image`, `signed_area`, `refuse_regime` (spinor, `nspin = 2`, spiral, DFT+U,
  gamma-only, ultrasoft, PAW, meta-GGA, each by name).
- `defuqumat/units.py`: `FLUX_QUANTUM_SI`, `flux_per_cell`, `tesla_from_flux`,
  `rational_flux` (returns `(p, q, B_used)`, so the field taken is always at hand).
- `defuqumat/magnetic/peierls.py`: `Bonds` (`from_blocks`, `zero_field`), `bonds_from_wannier`
  (each Wigner-Seitz image its own bond at `1/(ndegen count)`), `honeycomb_nn_bonds` (the
  review's model), `PeierlsTorus` (`matrix(kappa)` as CSR, `operator(kappa)` as a banded
  `PeierlsOperator` applied by `jnp.roll`, `ribbon(width)` from the same bonds),
  `levels_near`, `cluster_levels`, `block_weight`, `hermiticity`, `hermiticity_guard`,
  `plaquette_check`.
- `defuqumat/magnetic/landau.py`: `free_landau_levels`, `dirac_levels`, `dirac_point`,
  `dirac_velocity_from_bands`, `landau_degeneracy`, `linear_deviation`.
- `defuqumat/magnetic/gipaw.py`: the GIPAW phase and its factor on minimum-image
  displacements. The grid projectors (if-time item a) are not here.
- `defuqumat/workflows/landau.py`: `run_landau_levels` and `LandauLevels`.
- The precondition hook: `Hamiltonian.precondition` and `SpinorHamiltonian.precondition`
  (`g_psi` on the diagonals, which the solver passes in), and `davidson_eigensolver` calls
  `hamiltonian.precondition` when the operator has one and `_precondition` otherwise.
- `plans/orbital-field/graphene-scf.npz`: the shared zero-field state (density and potential
  `(1, 24, 24, 128)`, `fermi_energy = -0.11872348716859683` Ry, eigenvalues on the 19
  irreducible k, total energy `-22.664820820312105` Ry, lattice in bohr). It is the SCF of the
  design's keywords at the default `conv_thr = 1e-6` Ry (4 iterations, `L/graphene2.log`);
  the routes compare operators on this one potential, so its own convergence does not enter a
  crosscheck. Its integrated density is 8.000 electrons.
- `plans/orbital-field/graphene-wannier-auto.npz` and `graphene-wannier-pz.npz`: the two
  Wannier sets of item 6 (all arrays the Peierls model needs), for the crosscheck.
- `plans/orbital-field/nn-butterfly.png`: the Hofstadter butterfly of the nearest-neighbour
  model, `q <= 50`, all coprime `p`, 6 x 6 magnetic mesh, for the guide.
- Probes: `mag_graphene_scf.py`, `mag_si_energy.py`, `mag_nn.py`, `mag_wannier_landau.py`,
  `mag_kpm.py`, `mag_zero_modes.py`, all in `plans/orbital-field/`.
- Tests: `tests/unit/test_magnetic_conventions.py` (12 tests), `tests/unit/test_magnetic_peierls.py`
  (8 tests); 20 passed in 14.9 s at d79308f8 (`L/t-all2.log`).

## The acceptance numbers

**1. Hermiticity, flatness, plaquette** (`L/nn-1245.log`, NN model, `q = 400`, `p = 1`).
`max |H - H^dagger|` = 3.8e-15, 3.8e-15, 5.2e-15, 5.1e-15 eV at kappa (0,0), (0.37,0),
(0,0.29), (0.21,0.63). With the Bloch-sum factor dropped: 4.68 eV at (0.37, 0.29), and the guard
raised. Spread of the 22 levels nearest zero over the four kappa: 4.6e-15 eV. Hexagon
plaquette, counterclockwise: `|product - exp(-i 2 pi/400)|` = 0.0 (exact to the last bit); a
flipped sign raised. Accepted.

**2. Degeneracy** (`L/nn-1245.log`). `p = 2, q = 400` is not a reduced flux and
`MagneticCell.build` refuses it (the conventions ask for coprime `p, q`), so the check ran at
`2/401` and `2/399`: zero level 4-fold and the next five positive levels 4-fold at all four
kappa; at `3/400`, 6-fold. That is `2 p` (two valleys times `p`). Accepted.

**3. Dirac at 10, 20, 40 T** (`L/nn-3c.log`, the final code; the same numbers in `L/nn-3b.log`).
`E_1` = 0.100201 eV at `q = 7900` (9.989 T) against Dirac 0.100212 eV with `v_F` = 8.739e5 m/s
read from the zero-field bands. Fit `E_n/E_n^Dirac = 1 - c n`, `n = 1..10`, intercept fixed at one:
`c` = 1.146e-4 (10 T), 2.297e-4 (20 T), 4.605e-4 (40 T), against the review's 1.1e-4, 2.3e-4,
4.6e-4; electron and hole sides equal to 1e-15 (the model is chiral). Two zero modes per kappa,
`tr P_A` over the pair = 1.0000000000000 at all three fields. Shift-invert time at dim 15800:
1.40 s (1.54 to 1.60 s in the earlier runs), matrix build 0.43 to 0.50 s. Accepted.

**The trap this item found.** In the first run (`L/nn-3.log`) the trace over the zero-mode pair
read 1.0129, 1.0889 and 0.6455 at 10, 20 and 40 T, where it is one. `L/zero.log` shows why:
ARPACK's shift-invert returned the two exactly degenerate zero modes with `|V^+ V - 1|` from 0.12
to 0.79 at residuals of 1e-16 Ry, whatever the shift (1e-7, 1e-4, 3e-3 Ry). The vectors are
eigenvectors, but not an orthonormal basis of the pair, so a projector built from them is wrong.
`levels_near` now does a Rayleigh-Ritz step on the returned block (orthonormalised by its singular
vectors, a rank check that raises), and the unit test runs that path at `q = 1975`.

**4. Ribbon against torus** (`L/nn-1245.log`, `q = 400`, a 400-cell ribbon, kappa2 = 0.29).
The first twelve positive torus levels (six Landau levels, two valleys) against the nearest ribbon
level whose vector has under 1e-6 of its weight on the outer 20 cells at each end (189 such
levels): largest difference 1.8e-15 eV. Accepted.

**5. Hofstadter identities** (`L/nn-5.log`). `E -> -E`: 5.4e-14 eV at every kappa, `q <= 50`,
774 fluxes. `p/q -> (q - p)/q`: **not an identity at one kappa**, 1.40 eV apart at every mesh of
one point; over the magnetic mesh N x N, `q <= 12`: 0.63 eV (N = 2), 0.78 eV (N = 3), 1.5e-14 eV
(N = 6 and 12); at `q <= 50` over 6 x 6, 6.7e-14 eV. The reason: `(q - p)/q` is `-p/q` plus one
flux quantum per hexagon, and the gauge transformation that removes that quantum is not periodic
over the magnetic cell (on the honeycomb graph it carries a `pi n1 n2` term), so composed with
time reversal it moves kappa; the two spectra are equal as sets over a mesh closed under that
map, which a 6 x 6 mesh is and a 2 x 2 one is not. The design's "at four kappa" is replaced by
the 6 x 6 mesh, in the probe and in the unit test. Figure `plans/orbital-field/nn-butterfly.png`
(101 s for the whole sweep). Accepted with that change.

**6. P162's set on the shared graphene state: refused on the 10 meV criterion.** No fable
subagent could be launched from this step (the Agent tool is not available to it), so the
conservative reading was taken and is recorded here: the criterion "band distance < 10 meV within
E_F + 2 eV" counts every band below `E_F + 2` eV, as Vitale's `eta_2` does, and no set reached it.

| set | log | `eta_2` (band n against n) | max over bands below E_F + 2 eV | max within E_F +- 2 eV |
|---|---|---|---|---|
| `auto`, 8 functions, 100 steps (unconverged) | `L/graphene.log`, `L/graphene2.log` | 75.6 meV | 284 meV | (not printed) |
| `auto`, 8 functions, 1000 steps (converged) | `L/graphene-auto1000.log` | 116.8 meV | 800 meV | 3.86 meV |
| `C:pz`, 8 bands, frozen below E_F + 2 | `L/graphene2.log` | refused: more frozen states than `num_wann` | | |
| `C:pz`, 8 bands, frozen E_F +- 2 | `L/graphene-pz.log` | | 82 meV | 82 meV |
| `C:pz`, 16 bands, frozen E_F +- 2, 100 steps | `L/graphene-pz16.log` | | 622 meV | 22.3 meV |
| `C:pz`, 16 bands, frozen E_F +- 3, 1000 steps | `L/graphene-pz16b.log` | | 500 meV | 11.4 meV (within +- 3) |
| `C:pz`, 10 bands, frozen E_F +- 3, 1000 steps | `L/graphene-pz10b.log` | | 5713 meV | 4226 meV |

All comparisons are on a 6 x 6 mesh shifted by (1/24, 1/24) off the 12 x 12 Wannier mesh (on the
mesh itself the frozen states are exact by construction, 0.000 meV, which was checked too). Each
Wannier band is compared with the nearest DFT band at the same k, since the `p_z` bands are not
the lowest two. What fails is far from the Dirac point: the deep sigma bands of the 8-function set
(its `px`, `py` functions spread 3.8 to 10.4 A^2, and the site symmetry fell back: "the bands left
free at k-point 5 do not carry the irreps the trial orbitals need beside the frozen ones, rank 3
of 4") and the bottom of the pi band near Gamma for the `p_z` sets (with eight bands the outer
window stops below pi* at Gamma, which gave rank 1 of 2 and centres 0.3 to 0.7 bohr off the
plane; sixteen bands fixed the centres and the spreads, 0.95 A^2). Near `E_F` the 8-function set is
within 3.9 meV. **The Wannier problem is handed to the record**: what is needed is a set whose
deep bands are also within 10 meV, or a decision that the criterion is the window `E_F +- 2` eV,
which the 8-function set passes.

The Landau run on both sets was made anyway, to exercise the workflow at the real size, and is
recorded as a measurement, **not as an acceptance** (`L/wl-auto.log`, `L/wl-pz16b.log`; the big
job ran under `big.lock`, `MemoryMax=14G`, cores 0 to 3, `L/big-auto.sh`):

- 8-function set (converged), `q = 7900`, dim 63200: hopping cutoff 1e-6 Ry (10294 bonds per
  cell of 10306; zero-field band error 2.1e-5 eV within `E_F +- 2`); `E_D - E_F` = -11.8 meV,
  `v_F` = 8.397e5 m/s (holes 8.398e5); `E_1/E_1^Dirac` = 0.99865 (`E_1 - E_D` = 96.16 meV against
  96.29), `E_-1` 1.00069; fit `c` = 6.6e-4 electrons, -1.5e-4 holes (`E_n/E_n^Dirac` for `|n| <= 10`
  in the log: 0.9939 to 0.9986 above, 1.0007 to 1.0011 below); two zero modes with `tr P_A` =
  0.999993 over the pair; **valley splitting** 6 ueV at `n = -1` growing to 1.2 meV at `n = 10`
  (this set breaks the symmetry that makes the valleys degenerate); Hermiticity 5.6e-16 Ry,
  7.0e-16 relative; the plaquette of one cell (one function, hops along a1 and a2) is
  `exp(-i 2 pi/7900)` within the guard's 1e-10;
  build 20.4 s, shift-invert 22.0 s, 45 s in all.
- `p_z` set (16 bands), dim 15800: 722 bonds per cell, no cutoff needed; `v_F` = 8.421e5 m/s;
  `E_1/E_1^Dirac` = 0.99860, `E_-1` 1.00136; the electron-hole asymmetry grows to 0.9931 and
  1.0021 at `|n| = 10`; zero level at -67 ueV from `E_D`, two modes, `tr P_A` = 1.0000000;
  valleys degenerate to 6e-13 eV; Hermiticity 1.1e-15 relative; 5.0 s in all.
- The first version of the workflow grouped levels at the 1e-7 Ry degeneracy tolerance, which
  counted each split valley of the 8-function set as its own `n` (ratios 0.71 at every even `n`,
  `L/wl-auto-ungrouped.log`). The Dirac comparison now groups at 2 per cent of `E_1` and reports
  the splitting per level.

**7. The precondition hook.** The silicon probe `mag_si_energy.py` (QE's `pw_scf/scf.in` at
`conv_thr = 1e-10`, on the k-point path, at `K_POINTS gamma` with `nosym` on the half-sphere
branch, and with `noncolin` on the spinor operator) before the hook (worktree
`/l/ladovj1/w-mag-foundation-base` at 58536447) and after it (b3ff5f3b):

| run | slot line | k-points | gamma | noncolin |
|---|---|---|---|---|
| before, `L/si-base.log` | `# slot 1 cpu 6` | -15.794495570882638 | -14.518759817907355 | -15.794495570882642 |
| after, `L/si-after-b.log` | `# slot 1 cpu 6` | -15.794495570882638 | -14.518759817907355 | -15.794495570882642 |
| after, `L/si-after-a.log` | `# slot 0 cpu 4` | -15.794495570882638 | -14.518759817907355 | -15.794495570882642 |
| before, `L/si-before.log` (gamma without `nosym`) | `# slot 0 cpu 4` | -15.794495570882638 | (full sphere) | -15.794495570882642 |

The SHA-256 of the eigenvalue arrays is the same before and after on all three paths
(17462b9d099bdf53, b68b3cd37c1d5e79, 4aafb202279f27cf). Bit-identical. The unit test
`test_the_precondition_hook_is_the_old_rule_and_is_live` checks that the hook fires: an identity
preconditioner changes the step count and not the eigenvalues, and the explicit default gives
the same eigenvalues bit for bit.

Test files of the touched modules (`L/t-all.log`, at b3ff5f3b): `test_magnetic_conventions.py`,
`test_magnetic_peierls.py`, `test_davidson_collapse_projection.py`, `test_davidson_one_executable.py`,
`test_davidson_single_calbec.py`, `test_davidson_zero_root.py`, `test_gamma_davidson_real.py`,
`test_operator.py`, `test_units.py`, `test_hamiltonian_static_npw.py` and
`test_qeref.py::test_scf_silicon`: 50 passed, 3 skipped in 27.2 s. The three skips need QE's
source tree, which D22 does not have; with `scf.in`, its committed benchmark and `constants.f90`
copied to `L/qe-ref/` and `DEFUQUMAT_QE_ROOT` pointed there, `test_qeref.py::test_scf_silicon`,
`test_operator.py` and `test_units.py` gave 9 passed in 3.8 s (`L/t-qe.log`). The new files at the
tip d79308f8: 20 passed (`L/t-all2.log`).

**8. Units.** In `test_magnetic_conventions.py`: `FLUX_QUANTUM_SI` = 4.135667696e-15 Wb to 1e-8;
one flux quantum through the graphene cell (a = 2.46 A) is **78912 T**, and through the 3 x 1 x 1
conventional diamond cell (a = 3.56 A) **10877 T**, both computed from `a` and the SI constants
and asserted to 1 T. The design's 78926 T and 10883 T are 1.8e-4 and 5e-4 above them (a rounded
area, the review's 18.73 bohr^2 for 18.7154); the conservative choice was to test the arithmetic
at the stated `a`, keep Lu's 10870.4 T as a check at 0.2 per cent (it passes at 0.07), and record
it here. 10 T is `p/q = 1/7891` and `q = 7900` is 9.989 T (the design's 9.991 T used the rounded
area).

## The if-time items

- (a) Grid projectors with the GIPAW phase: **not delivered**. `gipaw.py` holds the phase only;
  each route's step 2 writes its own `_grid_projectors` as the design allows.
- (b) KPM on the Peierls operator (`L/kpm.log`): NN model, `q = 7900`, dim 15800, kappa (0.11,
  0.23), Jackson kernel with 1286 moments (sigma 20 meV at the centre), 100 random-phase vectors:
  the DOS within `E_D +- 0.3` eV against the 50 shift-invert levels broadened by the same 20 meV
  Gaussian, relative L2 difference 4.8 per cent, which is the stochastic trace's noise at 100
  vectors on two-state levels. 52 s for the moments against 2.7 s for the shift-invert.
  `PeierlsOperator.apply` agrees with the CSR matrix to 1e-12 (unit test, torus and ribbon).

## Decisions taken here, and why

No fable subagent could be started from this step, so every open choice took the conservative
option, as rule 9 asks.

- `rational_flux` returns `(p, q, B_used)` rather than `(p, q)`, so that the field actually taken
  travels with the flux.
- `p = 2, q = 400` replaced by `2/401`, `2/399` (and `3/400`): the conventions require a reduced
  flux, and `build` refuses `2/400` by name.
- The Hofstadter `(q - p)/q` identity is checked over a 6 x 6 mesh, measured as the smallest of
  1, 2, 3, 6 that closes.
- Item 6 is refused on the strict reading of the band criterion; its Landau numbers are kept as
  measurements.
- The units check values are the arithmetic at the stated lattice constants.
- `levels_near` does a Rayleigh-Ritz step on ARPACK's block (the zero-mode trap above).
- The Dirac comparison groups levels into Landau levels at 2 per cent of `E_1`.
- The Hermiticity guard compares against the largest hopping as well as the largest matrix entry:
  a small torus at a zone-boundary kappa had every entry cancel to 8e-16 eV and the guard raised
  on its own round-off (`L/nn-1245.log`, first item-5 attempt).
- The hook passes the solver's hoisted diagonals to `precondition` (doubled under half-sphere
  storage), so `g_psi` sees the same numbers in the same order and nothing is recomputed per step;
  an operator without the method takes the old call through `getattr`.
- The host phase path of `PeierlsTorus` is NumPy in complex128 (the reference path); the
  `PeierlsOperator` blocks take the policy's complex dtype. The blocks are built on the host, so
  a derivative in `B` through the operator is not there yet: the traced `scale` reaches every
  phase of `conventions.py` (tested with `jax.grad`), not the banded operator.

## What is left

- The Wannier set of item 6: a set within 10 meV on every band below `E_F + 2` eV, or the
  decision that the criterion is the window about `E_F`.
- Grid projectors with the GIPAW phase (if-time a), and its silicon acceptance.
- A traced build of the `PeierlsOperator` blocks, for `d/dB` through the operator.
- `Calculator.get_landau_levels` (the Record stage). **It must call
  `conventions.refuse_regime(system, pseudos, "Landau levels")`**: `run_landau_levels` sees only
  the Wannier set, which carries `spin` (refused by `bonds_from_wannier`) but neither the system
  nor the datasets, so ultrasoft, PAW, DFT+U, meta-GGA and gamma-only storage are not refused
  inside the workflow. A spinor set never reaches it: `run_wannier` refuses `npol = 2` and a
  spiral by name (`workflows/wannier.py:341`).
- For `docs/d22-notes.local.md` (the Record stage; this step does not edit `docs/`): D22 has no
  QE source tree, so `test_qeref.py::test_scf_silicon`, one test of `test_operator.py` and one
  of `test_units.py` skip there; `scf.in`, its committed benchmark and `constants.f90` copied to
  `/l/ladovj1/review/mag/foundation/qe-ref/` with `DEFUQUMAT_QE_ROOT` pointed there make them run
  (9 passed, `L/t-qe.log`). `L/big-auto.sh` is the runner under `big.lock` for the 63200-row
  solve (45 s, under 14 GB). Two slot jobs that write the same `npz` path at once race; the shared
  `graphene-scf.npz` was checked afterwards (8.000 electrons, the same total energy).
- D22 worktrees: `/l/ladovj1/w-mag-foundation` (detached at d79308f8; the later commits add only
  files under `plans/`) and `/l/ladovj1/w-mag-foundation-base` (58536447, the pre-hook baseline of
  item 7, removable).

## Slow files a reader may want to run (not run here)

No new slow test was added. The hook changes the call in the Davidson solver for every
operator, bit-identical on the three silicon paths above; the slow files that exercise the same
solver on other regimes are `tests/regression/test_davidson_zero_root.py`,
`tests/regression/test_gamma_only.py`, `tests/regression/test_spinorbit.py` and
`tests/regression/test_noncollinear_magnetism.py`.
