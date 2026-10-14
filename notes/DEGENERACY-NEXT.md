# What to do about degenerate eigenvectors, and where a smearing is the cure

**Status, 2026-10-14 evening.** Item 2 is done as `PLAN.md` P147 (the user's pick, fable's review
and decision, every number on D22); its measurements are in that item below and in P147. Items 3 to
5 are open and indexed in `PLAN.md` "What is outstanding". The paragraph that follows is the plan as
first written, on master at `3ab298b`, before anything was measured. The question was whether the trouble a degenerate
multiplet gives this code could be solved by some sort of smearing. **The short answer is that
"degenerate eigenvectors" covers three different troubles and a smearing cures exactly one of
them**: a filling that cuts a degenerate shell. The second, a hard tolerance deciding whether
two bands are one multiplet, takes a smearing in a looser sense, a smooth kernel in place of a
step, and that is a measured trade rather than a cure. The third, a derivative taken through an
eigendecomposition, no smearing can touch, and its fix is a divided difference, which the code
already has in one place. Every number below is a measurement still to take, on `D22-0161`
(`w-ddot` at `3ab298b`, `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`, `DEFUQUMAT_QE_ROOT`
exported), and the first step of every item is the baseline that shows the defect, since what
was reported is "sometimes" rather than one symptom.

## 1. Why a degeneracy is a problem at all, and what a smearing does to it

An eigensolver handed a degenerate multiplet returns some orthonormal basis of it, and which
one is not a property of the Hamiltonian: it is whatever rotation Davidson's last subspace
diagonalization happened to land on, and it changes with the starting block, with the thread
count and with the commit (`OPEN.md` Part on `get_electron_phonon` saw it move at `22842dc`,
a collapse rule that touched nothing else). Any quantity that is a function of the multiplet
as a whole is fine, because the projector onto it, `sum_n |n><n|`, is invariant. Any quantity
that reads one member, `<n|A|n>` band by band or the lowest `nocc` of them when `nocc` falls
inside the shell, is a function of that rotation and so is not a number. That is rule D4, and
the package meets it in most places with a block average (`.claude/skills/trap-check/traps.md`).

A smearing enters in one way. With a smooth `f(e)` the occupation of two bands with the same
energy is the same number, so the density `sum_n f_n |psi_n|^2` sees only the projector and the
rotation drops out; at `occupations = 'fixed'` the two get `1` and `0` and the density is
multivalued. The same is true one derivative up: the response of the density to a perturbation
carries `(f_n - f_m)/(e_n - e_m)`, which at a degeneracy is `-df/de`, finite and equal for
every member, where the fixed-occupation version is `1/0`. So a smearing cures the filling
cut (item 2) and dictates the shape of the derivative rule (item 4). It does nothing for a
tolerance that decides whether two bands are degenerate (item 3), where what helps is making
the decision continuous, and nothing for `jnp.linalg.eigh`'s own derivative rule.

## 2. The filling cuts a degenerate shell: the one a smearing cures

**What goes wrong.** `occupations = 'fixed'` (or a fixed `tot_magnetization`) with the last
occupied and the first empty band of a channel degenerate at some k-point. The oxygen atom at
`tot_magnetization = 2`, one electron in a threefold 2p shell (`tests/data/qe/o-atom-fixed-lsda.in`),
is the clean case, and a metal run with fixed occupations by mistake is the common one. The
density is then a function of the rotation Davidson returned and nothing downstream can repair
it: the Sternheimer `chi_0` on that atom converges to 5e-12 and disagrees with a central
difference of the density by 100 per cent (`response/sternheimer.py:233`), and the Newton-Krylov
solver was said to die at its first line search with a non-finite residual (`scf/solvers.py:178`;
**retracted 2026-10-14**: it converges under `jvp = 'autodiff'` and diverges, without a NaN, under
the finite-difference Jacobian, see item 4).

**What the code does today.** It knows, in five places, with four tolerances and three
outcomes, and never where the defect is made:

| site | tolerance (Ry) | outcome |
|---|---|---|
| `response/sternheimer.py:1473`, `DEGENERATE_CUT_RY` | 1e-5 | refuses |
| `workflows/realtime.py:255` | 1e-5, a literal | refuses |
| `ultracell/driver.py:1324`, `DEGENERATE_CUT` | 1e-8, only when `nbnd <= 2 nocc` | refuses |
| `response/photocurrent.py:896`, also from `shg.py:600` | 1e-8 | warns |
| `scf/occupations.py:41` `fixed_occupations` | none | nothing; `scf/driver.py:6300` prints `homo` and `lumo`, which may coincide |

An `nbnd` that truncates inside a multiplet is the same defect one layer up, the "occupied"
set being the computed one rather than the filled one; the ultracell row is that case and the
other response modules see it as `band_cut_gap`.

**The plan, as reviewed 2026-10-14 (fable).** One helper, the gap across a cut, taking
`(nspin, nk, nbnd)` eigenvalues always (callers hand it `eigenvalues_by_spin`) and a cut index
per channel, returning `min_k (e[cut] - e[cut - 1])` per channel and NaN where there is no band
to measure against (`cut <= 0` or `cut >= nbnd`), so that "not measured" cannot be read as
"gapped", the convention `photocurrent._warn_about_the_cut` already uses; the outcome and the
message stay with each caller. Both meanings share the arithmetic exactly: the filling callers
pass the occupied count per channel (`sternheimer.py:1473`, `realtime.py:255`, `run_scf`, and
`scf/solvers.py:178`, which today diagnoses the defect after the NaN and should test the gap
before the first line search), the truncation callers pass `nbnd` in every channel
(`ultracell/driver.py:1324`, whose `ultracell/hamiltonian.py:multiplet_cut` is this helper
already and is replaced by it, `photocurrent.py:871` and `shg.py:600`, and
`workflows/conductivity.py:151`, which hand-rolls `band_cut_gap` a third time). The occupied
count has three sources today (`occupied_counts` at `sternheimer.py:1440`, realtime's
`wg > 1e-10` scan, `fixed_occupations` itself); `occupied_counts` moves beside
`fixed_occupations` and every caller takes it from there, and noncollinear is `nspin = 1` with
count `nelec`, so the count comes from the one rule. The tolerance is the per-band threshold of
the band *above* the cut from the solve that produced the eigenvalues, passed in and not
recomputed: `band_thresholds` (`driver.py:332`) gives that band `empty_ethr` unless
`diago_full_acc`, so for an SCF it is `empty_band_threshold(SCFResult.ethr)` and for
`fixed_density_states` it is `nscf.py:285`'s own `ethr`, where the 1e-5 floor dominates at the
default `conv_thr = 1e-6`; a factor of one on that unit is what the measurements support (the
oxygen pair 1.4e-14 apart, B1's empty pair 3e-7), and a real gap under 0.14 meV at fixed
occupations is a metal anyway. The gap is a host-side diagnostic and never enters a
differentiated quantity, and the minimum over a wedge is the minimum over the zone since star
members share spectra, so nothing is symmetrised. One fact bounds the whole item:
`default_nbnd` at `occupations = 'fixed'` is exactly the occupied count, so a `pw.x`-style
input has no band above the cut and `run_scf` records NaN; the oxygen test fires only because
it sets `nbnd = 6`, and `docs/features.tex` says that `nbnd = nocc + 1` is what buys the check.
The existing `homo` and `lumo` cannot serve as the guard, being extremes across channels: a
magnetic insulator whose majority HOMO lies above its minority LUMO reads `homo > lumo` with
both channels gapped, so the message at `solvers.py:178`, "check homo against lumo", is wrong
in both directions. The ultracell constant moves from 1e-8 to the solver's unit only after its
silicon cases are re-measured across the cut (its docstring says its figures were taken under
the old definition); its `nbnd <= 2 nocc` gate stays.

**The decision (fable, 2026-10-14): `run_scf` records the per-channel gap on `SCFResult` and
warns once by name at the end, naming the smearing as the cure, and never refuses**; the
refusals stay in the consumers that are measured multivalued (Sternheimer's 100 per cent,
realtime). The warning fires with symmetry on as well, since the wavefunctions and everything
band-resolved stay gauge dependent, with the sentence saying what the symmetrised density is
not. The trade was stated from the reference source (read 2026-10-14): `pw.x` has no check at
all. `iweights_only` (`PW/src/iweights.f90`, about lines 95 to 125) gives
`wg = wk` to the lowest `NINT(nelec)/degspin` bands in index order and reads no eigenvalue,
`get_homo_lumo` (`print_ks_energies.f90:219`) reads only the occupations and returns
`ehomo == elumo` without a word, and `setup.f90:354` and `:362` check the electron count only
for a metal or an unpolarized spin run. So the reference runs such a cell and prints a density
that depends on nothing visible, the mixer here damps its way to something as well, and a
refusal in `run_scf` would be stricter than the reference; a warning by name with the two
equal levels would be the first time either code says so.

**The baseline, measured 2026-10-14 on D22** (`/l/ladovj1/review/degeneracy/`, `w-ddot` at
`3ab298b`, CPU; `cut_baseline.py` and `run_all.sh` there). The oxygen atom of
`o-atom-fixed-lsda.in` run in two arms, cores 0-3 with four threads against cores 4-9 with six,
so that the rotation Davidson returns differs; the same with `nosym = .true.`; and both again
with `occupations = 'smearing'`, `smearing = 'gaussian'`, `degauss = 0.01`, keeping
`tot_magnetization = 2`:

| run | iterations | E (Ry) | gap at the down cut (Ry) | arm A against arm B: `|dE|`, `max|drho|` |
|---|---|---|---|---|
| fixed, symmetry on | 9, 9 | -31.404683564908 | 4.9e-12 | 1.2e-13 Ry, 6.2e-12 |
| fixed, `nosym` | 58, 50 | -31.412688052606, -31.412687726220 | 2.3e-2 | **3.3e-7 Ry, 1.0e-3** (`max rho` 0.61) |
| smeared, symmetry on | 8, 8 | -31.412396680897 | 0.46 (the cut moves above the shell) | 6.4e-14 Ry, 1.5e-13 |
| smeared, `nosym` | 11, 11 | -31.412396680910 | 0.46 | 9.6e-14 Ry, 2.3e-12 |

Three things follow, and the first corrects the premise of this item. **With the point group
on, the SCF at a cut shell is gauge-independent**: the down channel fills one member of an
exactly degenerate 2p triplet (eigenvalues equal to 5e-12, `homo == lumo` to 5e-12) and the
two arms still agree to round-off, because the symmetrisation of the density over the cubic
group averages `|p_x|^2` into the spherical shell `(|p_x|^2 + |p_y|^2 + |p_z|^2)/3`, which is
the projector onto the multiplet and not a member of it. The multivaluedness lives in the
response and in the Newton-Krylov Jacobian, where a perturbation breaks the symmetry and the
symmetrisation no longer repairs it; `run_scf` on this atom is right as it stands, and a refusal
there would refuse a correct number. **Without symmetry the shell breaks instead of being cut**:
the occupied orbital drops 23 mRy below the other two, the total energy falls by 8.0 mRy to a
symmetry-broken state, the gap at the cut is a real 2.3e-2 Ry, and the run takes six times as
many iterations; the two arms land on different orientations of that orbital, 3.3e-7 Ry and
1.0e-3 of the density apart at `conv_thr = 1e-10`, which is the anisotropy of the cubic box
between two orientations and not a convergence residue. A gap check at convergence would read
2.3e-2 Ry here and stay silent, so the guard this item builds does not see the `nosym` case,
and the signature of that case is the lifted degeneracy and the iteration count rather than a
gap. **A smearing is a cure in both cases**: every pair of arms agrees to 1e-12, the shell stays
spherical even without symmetry (the triplet within 6e-6 Ry, occupations 0.3334, 0.3333, 0.3333)
and the run takes 8 to 11 iterations. What it costs is the `-TS` term: at `degauss = 0.01` the
smeared total energy sits 7.7 mRy below the fixed symmetric one on the same density and the
same eigenvalues (to 2e-6 Ry), which is `degauss sum_n w1gauss(0)` over three levels at the
Fermi level, 3 x 0.01 x 0.282 = 8.5 mRy to leading order; it is linear in `degauss` and a user
who wants the fixed energy takes a small width.

So the measurement reshapes the item: the SCF reports the gap and the two equal levels and
warns, naming what will go wrong downstream (a response, a Newton solve), and the refusals stay
in the consumers that are actually multivalued; and the test is that the guard fires on this
atom with symmetry on, is silent on the gapped fixed LSDA cell, and is silent (by design, and
said so) on the `nosym` run.

**The rotation arm (fable's change 6), measured the same day, and half of it is a null.** The
converged wavefunctions with the minority 2p block turned by a random 3x3 unitary, handed back as
`starting_wavefunctions` with the converged density and `becsum`, against the same continuation
unturned: with symmetry `|dE| = 3.6e-15` Ry and `max|drho| = 5.6e-12`, without symmetry
`3.6e-15` Ry and `5.8e-15`, both in one iteration. The `nosym` half is a null by construction:
`starting_wavefunctions` is a *span* for the first Rayleigh-Ritz, a rotation inside a block does
not change the span, and on the broken-shell state the block is gapped, so the Rayleigh-Ritz
returns the same eigenvectors whatever rotation went in. The symmetric half is evidence only if
the Rayleigh-Ritz actually returned a different rotation of the exactly degenerate triplet, which
`eigh` is free to do and not bound to; `cut_overlap.py` read the overlap matrix of the two
continuations' minority blocks, and it is diagonal to 0.8 per cent (`|<plain_i|turned_j>|` =
0.7868 on the diagonal, the plain inner product of S-normalised ultrasoft states, 0.006 off it),
so the rotation did **not** survive and the symmetric half is a null too. The two-arm baseline
above stands on its own for the `nosym` case, where the arms did differ (3.3e-7 Ry).

**The direct test of the claim, `cut_symdensity.py`, no re-diagonalisation anywhere.** The
converged states with the minority 2p block turned by a random unitary, and separately with only
the *occupied* member replaced by a rotated one, handed to `Calculation.density` (`sum_band`,
the augmentation charge, `sym_rho`) beside the unturned states, same weights:

| | `max|rho(turned) - rho(plain)|` | `max rho` |
|---|---|---|
| point group on (48 operations) | **9.1e-8** (both rotations) | 0.607 |
| `nosym` | **0.202** (both rotations) | 0.614 |

So the symmetrisation does what the group-theory argument says (the traceless part of one
member's density spans `E_g + T_2g`, both nontrivial, so the average over the 48 operations is
the third of the shell), and the method could have returned a surprise, since it did without the
group: 0.2 of a density whose maximum is 0.6, a third of an electron moved. The 9.1e-8 residue
with symmetry is not round-off; it is the eigenvectors' own convergence, the triplet closed only
to `ethr = 2e-12` in the eigenvalues and so to about 1e-6 in the vectors, and a rotation mixes
in that error. It says what the warning should say: the symmetrised density of a cut shell is
single-valued to the solver's accuracy on the states, not to machine precision, and the
self-consistent fixed point (the two arms, 6e-12) is tighter than one density built from one
set of states.

## 3. A hard step decides what is degenerate: a smearing in the looser sense

**What goes wrong.** The inventory found one tolerance per module, and the eigensolver's
promise the tolerance has to be read against depends on which manifold the guard looks at:
an occupied band is converged to `ethr` (about `conv_thr / nelec`, 1e-10-ish at the end of an
SCF) and an empty one only to `empty_band_threshold(ethr) = max(5 ethr, 1e-5)`:

| site | what it guards | manifold | tolerance |
|---|---|---|---|
| `response/conductivity.py:836` | `1/e_mn`, `1/e_mn^2` of the interband conductivity | occupied/empty pairs | `EMPTY_ETHR_FLOOR`, since `OPEN.md` B1 |
| `topology/kubo.py:105` | `1/gap^2` of the Kubo Berry curvature | occupied/empty pairs | 1e-6 |
| `transport/green.py:211` | the channel blocks of the lead at the energy `E` | bands at `E`, empty included | 1e-6 |
| `response/photocurrent.py:180` | `1/e_mn` of the shift current and SHG | occupied/empty pairs | 1e-8, raised to the broadening at the call |
| `ultracell/driver.py:202` | the multiplet ladder's cut | the band at the cut | 1e-8 |
| `topology/berry.py:323` | `1/gap^2` on a model | model, exact | 1e-8 of the band width |
| `tddft/spinchi0.py:654` | the `0/0` of the transverse `chi_0` | pairs near `E_F` | 1e-8, the limit `df/de` taken inside |
| `scf/tetrahedra.py:94` | the degenerate-weight average | bands carrying weight, at `ethr` | 1e-6 |
| `topology/parity.py:73` | the parity window | occupied, at `ethr` | 1e-6 |

The first five read a band the solver converged only to `empty_ethr`, and their guards sit
below it. `OPEN.md` B1 measured a symmetry-degenerate empty pair coming back split by 3e-7 Ry
at the default `ethr`; such a pair passes every 1e-6 and 1e-8 guard in those rows, lands in a
`1/gap^2` and reads as a large number whose sign is the rotation. B1 moved one module onto the
solver's floor and left the rest where they were. The last two guard the occupied manifold,
where 1e-6 is four orders above `ethr` and the argument does not apply; they stay in the table
because the unified rule of this item should give them `ethr` rather than `empty_ethr`. And every guard is a step: a pair at `0.99 tol` is in
and at `1.01 tol` is out, so a quantity built on it jumps when a lattice constant, a cutoff or
a commit moves the gap across the line, and under `jax.grad` the step's tangent is zero on one
side and the full `1/gap^2` on the other.

**What a smearing means here.** The step `where(|gap| > tol, 1/gap, 0)` is replaced by the
smooth kernel `gap/(gap^2 + eta^2)`, which is the real part of `1/(gap + i eta)`, and
`1/gap^2` by `gap^2/(gap^2 + eta^2)^2`, with `eta` the solver's floor. Inside a multiplet the
kernel is then `O(gap/eta^2)` rather than a dropped term, so the dependence on the returned
rotation falls off as `(gap/eta)^2` instead of flipping, and the quantity is a continuous and
differentiable function of everything. **Two things to write down before any of this lands.**
A smooth kernel moves a converged number, by about `(eta/gap_typical)^2` of each pair's
contribution, so each site is measured against its own QE or Elk validation tolerance in
`PLAN.md` §3 before the switch; where the move exceeds it the hard step stays and only the
tolerance is unified. And for a Kubo quantity the occupied-occupied degenerate pairs already
cancel through `f_n - f_m`, so the pairs this helps are occupied/empty ones at the Fermi level,
which is item 2 in another form, and an insulator's Berry curvature should not move at all.

**The plan.** One resolution rule for the tolerance, `max(asked, empty_band_threshold(ethr))`,
in one module, forwarded by name through `SHARED_OPTIONS` to every site in the table (the
`Calculator` rule, `.claude/rules/calculator.md`). Then the kernel, site by site, behind the
measurement.

**Measurements, on D22.** The baseline for each site is the same experiment: a cell with a
symmetry-forced degeneracy, silicon's `Gamma_25'` triplet and the `X` pairs for the Berry
curvature and the conductivity, whichever transport cell `tests/regression` already runs for
`green.py`, a random unitary applied
inside each degenerate pair of the converged eigenvectors, and the output read before and
after. A site is degeneracy-safe when that move is round-off, and the acceptance test for the
kernel is the same rotation at a pair split by the solver's own 3e-7. Each landed kernel is
also read against its reference number: `PLAN.md`'s silicon `chi^(2)`, the Kubo Chern number
of the model, the conductivity against `OPEN.md` B1's table.

## 4. A derivative taken through `eigh`: what a smearing cannot touch

**What goes wrong.** `jnp.linalg.eigh`'s JVP carries `1/(w_i - w_j)` over every pair, written
as `1/(1 + w_j - w_i) - 1` so that an exact degeneracy gives a dropped term rather than a NaN,
and a near one at 1e-16 gives a wrong number: 42 per cent of the largest element at `Gamma` on
the Hubbard overlap (`notes/AUDIT-2026-09-18.md`, P20). The package has one protected site,
`hubbard/projectors.py:62` `_inverse_sqrt`, whose `custom_jvp` is the divided difference
`-1/(s_i s_j (s_i + s_j))`, finite at a degeneracy because the denominator is a sum. The one
unprotected site is Davidson itself under the Newton-Krylov solver's `jvp`
(`scf/residual.py:31`, "Davidson is differentiated as implemented"), which differentiates the
subspace `eigh` in `solvers/subspace.py:47` at every step.

**What to expect, and what to measure rather than assume.** The dropped in-block term is the
rotation inside a degenerate block, and the density's tangent does not depend on it when the
block is wholly occupied or wholly empty, so in exact arithmetic the Jacobian action is right
on silicon at `Gamma` and wrong exactly when the block is cut, which is item 2 again. In
floating point there is a second effect to tell apart from the first: a pair the solver leaves
split by 1e-9 rather than exactly gives in-block terms of order 1e9 in the eigenvector
tangent, which cancel pairwise in the density's tangent and leave precision loss behind. So a
disagreement at 1e-7 on silicon reads as that cancellation and not as the defect of the cut
case, and the measurement records both the size of the disagreement and the smallest in-block
gap it came with. The correct rule, the one a
Sternheimer solve uses and `_smeared_projection` (`response/sternheimer.py:650`) already
writes, is the divided difference of the occupation, `(f_n - f_m)/(e_n - e_m)`, `-df/de` at a
degeneracy: this is where the smearing of item 1 reappears as the thing that makes the
derivative well defined.

**Measured so far (2026-10-14, with P147, `cut_newton.py`).** The whole Newton-Krylov solve on
the symmetric cut-shell atom, both Jacobians, eight outer iterations: `autodiff` converges,
1.3e-2 to 3.8e-7 in eight damped steps and the mixer finishes at 5.2e-12 on the mixer's own
energy, so the JVP through Davidson's `eigh` at an exactly degenerate occupied/empty pair is a
usable direction (not quadratic, which is the dropped in-block term showing); `finite-difference`
diverges to 7.1e33 at the first step, GMRES at its cap with 444 JVPs. **Which component diverges
is measured** (`cut_becsum.py`): under a rotation of the cut shell at the converged symmetric
state the symmetrised density moves by 9.1e-8 and the raw `becsum` the state vector packs moves
by 0.342 on a maximum of 1.111, so the difference quotient is `O(1)/eps` in its `becsum` rows
and a derivative in its density rows. Neither arm produces a NaN. **A finding of its own, item
6**: `becsum` is not symmetrised in this code. An ultrasoft dataset does not notice in the mixer,
since `becsum` reaches the next Hamiltonian only through the augmentation charge, which
`sym_rho` symmetrises with the rest; a PAW one-centre term reads `becsum` raw, `pw.x` runs
`PAW_symmetrize` on it, and whether this code does the equivalent is unchecked. So the hard-cut case is the opposite of the expectation above in one respect: the autodiff
tangent is the *better* of the two there, and what a finite difference measures on a cut shell is
not a derivative at all.

**Still to measure.** `residual.jvp` against `residual.jvp_finite_difference` for a random
direction where the finite difference is a derivative: silicon at `Gamma` (two threefold blocks,
wholly occupied) and the oxygen atom with a smearing (the block cut by `f` but smoothly).
Expected: agreement to `ethr` on both, with the precision-loss caveat above. If the smeared cut
disagrees, the fix is a `custom_jvp` on the density projector with the divided difference, the
Lowdin pattern, and the reverse-mode question stays where P22 left it.

## 5. Two small ones, named

- `hubbard/occupations.py:593`: `starting_ns` writes the requested eigenvalues into
  `eigh(ns)`'s eigenvalues by index `m`, and inside a degenerate block of `ns` the eigenvector
  order is the solver's. A starting guess rather than a result, so it costs iterations and
  cannot be wrong, and `ns_adj` in `pw.x` does the same; worth one line in the docstring.
- `OPEN.md`, "`get_electron_phonon` at its default threshold depends on the gauge of a
  degenerate multiplet": the mode pair at `q1` moves by 0.015 cm^-1 under a random unitary in
  the degenerate blocks at `k` under the scheduled CG threshold and by nothing at 1e-12. The
  tests pin 1e-12 and the default is still the schedule, so the item is open as written; it is
  item 3's experiment on a phonon, and the fix is a threshold rule rather than a kernel.

## What is not here

A spin spiral's degeneracies, the Kramers pairs of a spin-orbit run (`OPEN.md` Part I item 1,
closed at the solver's promise) and the tetrahedron average (B2, closed) are degeneracies the
code handles and this plan does not reopen. Nothing here touches `pw.x`'s own behaviour:
`cegterg.f90` returns the same arbitrary rotation and `iweights.f90` fills the lowest bands in
index order whatever their energies (read 2026-10-14, item 2), so every refusal in item 2 is
stricter than the reference, which is the trade to decide.
