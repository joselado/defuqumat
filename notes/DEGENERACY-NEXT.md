# What to do about degenerate eigenvectors, and where a smearing is the cure

**Status, 2026-10-14, late.** Item 2 is done as `PLAN.md` P147 (the user's pick, fable's review
and decision, every number on D22); its measurements are in that item below and in P147. Items 6
(the ultrasoft `becsum` averaged) and 4 (the subspace `eigh` tangent regularised) are P148, in the
order fable decided (6, 4, 5, 3). **Items 5 and 3 are P149** (`plans/degeneracy-p149.md` holds the
plan, fable's review, which corrected item 3's premise, and its follow-up): item 5a is a docstring
sentence, item 5b is closed by a cap on the scheduled CG threshold at `q` (not by a threshold rule
of the kind sketched in item 3, see "Outcome" under item 5), and item 3 landed as one rule and no
kernel (see "Outcome" under item 3). What stays open is item 4's tail and `OPEN.md` Part XXVI,
indexed in `PLAN.md` "What is outstanding". The paragraph that follows is the plan as
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

**Outcome (P149, 2026-10-14, D22).** The table above was re-read site by site at fable's review
(`plans/degeneracy-p149.md` A) and its premise was wrong in one respect that reshapes the item:
the pair sites (the conductivity, the Kubo curvature, the shift current and SHG) read
*fixed-density* states, which a scalar `ethr = max(ETHR_MIN, 0.1 min(1e-2, conv_thr/nelec))` holds
every band to, the empty ones included, so none of them reads a band at the 1e-5 floor and "five
of them below the solver's promise" is withdrawn. Measured on silicon's symmetry multiplets at
`Gamma`, `X` and `L` (`review/degeneracy/split/`): a fixed-density solve leaves a multiplet split by
0.00 to 0.23 `ethr` at `conv_thr` 1e-4 to 1e-6, with the one outlier the highest solved pair at
1.7 `ethr`, and by 1e-16 to 8e-13 Ry at `conv_thr` 1e-10 and 1e-12, the arithmetic floor (B1's
nickel read 5.5e-12). So the rule landed as `solvers/davidson.py:resolve_degeneracy_tol`,
`max(asked, 5 ethr, 1e-10)` for a fixed-density solve, `max(asked, 1e-5)` for states of unknown
provenance, the empty-band floor on top for an SCF's states, one `degeneracy_tol` in
`SHARED_OPTIONS` reaching eight entry points by name, `None` resolving and a number honoured as
asked with a warning below the promise (fable's three follow-up decisions). It touches five sites
(the conductivity, the Kubo per-band curvature, `green.py`'s block decision, the photocurrent's
guard and cut warning, the ultracell cut) and leaves four alone with a sentence each
(`berry.py`'s model, `spinchi0.py`'s switch, `tetrahedra.py`'s grouping at the scheduled `ethr`,
`parity.py`'s closure check). At every default on the fixed-density route it changes no number;
what it changes is the ultracell cut at a loose `states_conv_thr` (at 1e-4 the solver splits the
cut multiplet by 1.3e-7, which the 1e-8 constant read as a gap and the resolved 6.25e-6 does not)
and the transport channel basis on SCF states (the 1e-5 floor over the 1e-6 constant). **The
smooth kernel lands nowhere beyond P148's subspace solve**, on the arithmetic: on the curvature
route a pair at the synthetic 3e-7 with `eta = 1e-5` leaves `w f' Im z gap^3/eta^4`, about 135
`w Im z` at `degauss = 0.02`, against 100 from a legitimate 0.1 Ry pair and the step's exact zero,
which is the physical answer for a symmetry pair; at the real 5e-12 split the kernel is harmless and
buys nothing. The rotation controls at the sites without a kernel are round-off where rule D4 says
they must be: the Kubo manifold total and the per-band block trace move by 8.5e-11 on 3e-2 and 11
under a rotation inside silicon's multiplets while one band inside a multiplet moves by 7, and
AlAs's `chi^(2)` moves by 1e-10 on 1.1e3 under a rotation inside 222 blocks below the cut.

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

**Measured 2026-10-14 evening (D22, `cut_jvp.py`, `cut_jvp_steps.py`, `cut_jvp_mech.py`), and
the expectation above is falsified in both directions.** `residual.jvp` against
`residual.jvp_finite_difference` at `x` = the output density of the converged run (`|r(x)|` 7e-6 to
4e-5, so near the fixed point and not at it), a random unit direction `v`, `J = dF/dx - I`, so
`|Jv|` near 1 means `dF.v` is small:

| cell | degenerate block in the subspace | warm autodiff `|Jv|` (Davidson steps) | FD `|Jv|`, every start and `eps` | warm autodiff against warm FD |
|---|---|---|---|---|
| NC silicon, `nosym`, one generic k (`si-generic`) | none, smallest occupied gap 9.2e-2 Ry | 0.99971 (8) | 0.99971 to 0.99977 | **6.1e-4, 3.3e-5, 3.5e-7, 1.5e-8** at `eps` 1e-4 to 1e-7 `|x|` (the `eps^2` law); cold pair 1.5e-9 |
| NC silicon, 2x2x2 with `Gamma` (`si-gamma`) | threefold at `Gamma`, pairs at `L` | 1.039 (11/19/8); cold 9.58 (26/27/24); `ethr` 1e-13 85.5 (20/24/15) | 1.0000 to four figures at `eps` 1e-4 to 1e-7, warm, cold and tight | **28 per cent**, independent of `eps` |
| smeared symmetric oxygen (`o-smear`), triplet split 1.8e-14 | the 2p triplet | 1.0003 (8/9); cold 3.21 (29/27) | 1.0002 to 1.0004 | 0.5 to 0.9 per cent; on the `becsum` rows (norm 3e-2) 14 to 23 per cent |
| smeared `nosym` oxygen, triplet split 6e-6 | the split 2p triplet | 1.0016 | 1.0012 to 1.0016 | 1.5 to 2.5 per cent |

So on a cell without a degenerate block the autodiff action **is** the Jacobian, to the finite
difference's own accuracy, and `residual.py`'s "accurate to about `ethr`" holds there; with one it
is not, and the three autodiff arms contradict each other by factors of 9 and 30 while every FD arm
agrees with every other to four figures. The mechanism run (`cut_jvp_mech.py`: the eigensolver
alone, a smooth potential perturbation, the Davidson cap swept at a fixed warm start, the occupied
weight per `G` against its own central difference) puts it inside the step: on `si-gamma` the
tangent is 4.5e-2 off at caps 1 and 2, then **36x** at cap 4, 17x at 8, 23x at 16, 40x uncapped,
NaN at `ethr = 1e-13` and 895x from a cold start, while the FD at every cap sits at 8.4e-2 to
8.8e-2; on the smeared oxygen 0.9x to 1.1x at caps 1 to 4, 3.1x from cap 8, 5200x cold. The
culprit is the subspace `eigh`'s JVP at every Davidson step: `jax`'s rule is
`Fmat = 1/(eye + w_j - w_i) - eye` (`jax/_src/lax/linalg.py`, `_eigh_jvp_rule`, read 2026-10-14),
which is **zero on the diagonal and `1/(w_j - w_i)` unprotected off it** (the sentence above about
an exact degeneracy giving a dropped term is wrong: it gives `inf`, which is the NaN at `ethr`
1e-13 from the exactly degenerate parked rows), so Ritz values a degenerate block splits only at the
solver's noise give an in-block rotation tangent of `1/noise`; it is gauge and cancels in the
density after one or two steps (the 4.5e-2), and once it has passed through a normalised correction
vector and the next step's projections it no longer cancels. A degeneracy item after all, and the
1e-14 against 6e-6 control did not separate it because both cells carry a degenerate block.

**The decision (fable, 2026-10-14 evening): a small fix now, inside this plan, after item 6; nothing
goes to `OPEN.md` unless its acceptance fails.** A `custom_jvp` on the standard `eigh` inside
`solvers/subspace.py`'s subspace solve: same primal, the eigenvalue tangent unchanged, and the
kernel `F_ij = (w_j - w_i)/((w_j - w_i)^2 + eta^2)`, zero on the diagonal, linear in the tangent
so that transposition keeps reverse mode where it exists; bounded by `1/(2 eta)`, so no NaN on
exactly degenerate parked rows, and `1/Delta` to `(eta/Delta)^2` away from a degeneracy. **This is
item 3's kernel, landed at its first measured site.** The `eta` rule is
`empty_band_threshold(ethr) = max(5 ethr, 1e-5)` Ry with `ethr` the one the residual's Davidson is
called at, passed in and not recomputed, because a degenerate block's members are converged
separately only to the solver's promise on them and any `eta` below that unit is below the split
the solver itself puts into the block (B1's 3e-7). What it costs, stated: a pair of Ritz values
within about `10 eta` of each other (1e-4 Ry at the defaults) has its mixing tangent suppressed by
`(eta/Delta)^2`, one per cent there, which matters only where the two carry different occupations,
a near-cut or a smeared Fermi level, where the divided difference of the occupation in the density
assembly is the exact rule; that stays as this item's tail in "What is outstanding", motivated when
a smeared metal's Newton autodiff arm disagrees with FD by more than the `si-generic` figure.
Acceptance: `si-gamma` warm autodiff against FD within 10x of `si-generic`'s at each `eps` (3.5e-7
at 1e-6); the cap sweep flat, the occupied-weight tangent against its FD within 1e-2 at every cap
and no NaN at `ethr = 1e-13`; `si-generic` unchanged to 1e-8; the smeared oxygen's `becsum` rows
(symmetrised by item 6) down to the density rows' level; the cut-shell oxygen Newton autodiff arm
rerun and recorded, with the note that dropping the in-block mixing across a cut is a rigid-block
Jacobian and not a derivative, so the texts keep the smearing as the cure. The Newton default is
already `jvp = 'finite-difference'`, which after item 6 is a derivative on every symmetric gapped
cell; `jvp = 'autodiff'` stays an option whose docstring carries these numbers.

**The outcome (D22, `after4/`, 2026-10-14 night): the kernel is a measured partial fix and stays, and
the rest is `OPEN.md` Part XXVI.** The `inf` and the NaN are gone; `si-gamma` warm goes from 28 per cent
to 1.2e-2, cold from 9.6x to 16 per cent, the eigensolver's cap sweep from 36x to 1.9x at cap 4 and
from NaN to 300x at `ethr = 1e-13`; the smeared oxygen's sweep is flat at 0.2x where it was 0.9 to
3.1x; the Newton autodiff arm on the cut-shell atom converges in five undamped steps to 2.3e-12 where
it stalled at 3.8e-7 after eight damped ones; `si-generic` is unchanged to the last digit. The
acceptance (within 10x of `si-generic`, the sweep flat) is not met, and the width diagnostic says why
not: the residue does not depend on `eta` from 1e-5 to 1e-1, so another amplifier remains. Two frozen
variants locate it: `stop_gradient` on the correction vectors' norm changes nothing (bit-identical
sweep), while freezing the whole normalised expansion block flattens `si-gamma` (0.17 down to 3.1e-2
cold, the Jacobian action 1.6e-3) and costs `si-generic` its exactness (5.9e-3 from 3.5e-7), so the
amplifier is one member's correction vector on a degenerate block, which is not a smooth function of
the density when LAPACK's basis inside the block changes between steps, and the frozen-expansion
tangent is a candidate approximation and not a fix. The exact route is the implicit tangent of the
converged eigenproblem (P22c). The Newton default stays `finite-difference`, which with item 6 is a
derivative on the cut shell too (four undamped steps to 5.2e-12, 38 `F` evaluations against 26 JVPs).

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

**Outcome (P149, 2026-10-14, D22, `review/degeneracy/elph/`).** The first is one sentence in
`adjust_ns`'s docstring. The second was measured as a sweep of the CG threshold under a random
unitary inside every degenerate pair at `k` (49 pairs, 6 carrying weight) on `al-elph-nosym.in`
at `q1`, with the warm start on and off, and with one more tight pass of every mode at the
converged potential, warm or cold: the bare schedule moves the pair by 1.42e-2 cm^-1 (Part XXV's
number to the digit), every pass from zero by 5.6e-4, a fixed 1e-6 by 6e-5 but 1.4e-2 off the
tight pair, a fixed 1e-8 by 2.2e-7 at 1.7x the CG steps, 1e-10 and 1e-12 by 2.9e-6 and 6.7e-8 at
2.4x, a fixed 1e-2 or 1e-4 never converges the loop, and a tight finishing pass changes nothing
(the error is in `dV_scf`, which the self-consistency test cannot see). So the rule is a cap,
`dynamical_matrix_at_q(threshold_cap=1e-8)` by default (`SCHEDULE_CAP_AT_Q`), which reproduces the
fixed 1e-8 arm to the digit; the entry is closed in `OPEN.md` Part XXV with the table.

## 6. `becsum` was not symmetrised for an ultrasoft species

**Found with P147, half closed by reading, half measured (2026-10-14 evening).** The PAW half:
`pw.x` runs `PAW_symmetrize` on `rho%bec` and this code does the equivalent, `paw/symmetry.py`'s
`build_becsum_symmetry` (the D matrices derived from the project's own harmonics, the atom orbit
mapped, the `nspin_mag = 4` magnetization rotated as an axial vector), applied at the end of
`Calculation.becsum` and `finish_becsum`, the streamed path included; the sentence in P147 and in
`solvers.py` that a PAW one-centre term "would read `becsum` raw" was asserted and is wrong. The
ultrasoft half: the builder gave a species `None` unless `pseudo.is_paw` and the driver built the
object only `if self.paw is not None`, so an ultrasoft `becsum` was the raw sum over the k-wedge,
which `pw.x` also leaves raw on purpose (`sum_band.f90:259-267`: "only used to compute the
density, which is symmetrized later"; `addusforce.f90` and `addusstress.f90` read it raw and the
force and stress are symmetrised after). Every consumer here is linear in it and symmetrised
downstream too (`addusdens` then `sym_rho`; the forces through `symmetrize_vector`, the stress
through `symmetrize_matrix`; the Anderson fit excludes it, `mixing.py:371`; the shell is rebuilt by
the linear `sym_rho(addusdens(.))`), so the energy and the density cannot move. What moves is the
state: the Newton-Krylov vector, the checkpoint and the mixer's history carry `becsum` itself, and a
raw one-k sum follows the rotation the eigensolver returned inside a degenerate multiplet, which is
the whole of the finite-difference Jacobian's 7e33 in P147.

**The decision (fable, 2026-10-14): average every augmented species, unconditionally**, not only
under the Newton solver, since a gate would make the checkpoint's state and the mixer's history
depend on the solver choice and the response modules' `apply_strain`, `apply_atom_displacement`
and `apply_directional` need the object regardless. One condition in each of two places
(`paw/symmetry.py`, `is_ultrasoft` for `is_paw`; `scf/driver.py`, `is_ultrasoft and
use_symmetry`). The relabelling caveat in the builder (the inverse atom permutation, visible only
on a three-fold orbit) is covered by `tests/unit/test_symmetrisation_projector.py`, now
parametrised over the ultrasoft dataset too, with the identity added on that orbit.

**Measured before the change, D22, `us_becsum.py` on `w-ddot` at `904031c`** (the group average
built through a proxy and applied to the converged raw `becsum`; the identity read on it):

| cell | `nsym` | iterations, E (Ry) | `max|becsum|`, `|sym(b) - b|` | `|aug(sym b) - sym(aug(b))|` on `max|aug|` | `|aug(b) - sym(aug(b))|` |
|---|---|---|---|---|---|
| `si8-us-1k` | 24 | 9, -91.013925889499 | 1.552, **0.105** | 6.9e-17 on 4.26e-2 | 6.7e-3 |
| `si2-us-1k` | 48 | 8, -22.212973427529 | 1.351, 0.205 | 6.7e-17 on 3.87e-2 | 1.3e-2 |
| `si-epsilon-us` | 48 | 10, -22.766005381176 | 1.580, 0.092 | 5.6e-17 on 4.36e-2 | 2.5e-3 |
| oxygen atom (P147) | 48 | 9, -31.404683564908 | 1.111, 0.533 | 5.0e-16 on 2.63e-1 | 1.5e-1 |
| `fe-mag-1k` (`nosym`) | 16 found, none in force | 11, -55.574264635419 | 61.1, 24.9 | 0.405 on 1.04, **a null**: the run uses no symmetry, `symmetrize` is the identity and the average does not apply | 0 |

So the raw ultrasoft `becsum` is far from symmetric on every cell (7 to 48 per cent of its
maximum) and the identity holds to round-off wherever the symmetry is in force, which is what
makes the change exact for the energy; and under the shell rotation of P147 the averaged `becsum`
moves by 4.1e-7 (the states' own convergence) where the raw one moved by 0.342
(`cut_becsum_sym.py`). The cost of the average is one `(nsym, nh, nh)` contraction per species per
call; the `becsum` call read 1.7 to 3.4 ms on these cells before the change. **The after arm, the
continuation from a pre-change checkpoint, the noncollinear cell with symmetry in force
(`ni-noncol-111-nosource`, `nspin_mag = 4`) and the nonmagnetic spinor one (`alas-epsilon-us-soc`)
are in `PLAN.md` P148.**

## What is not here

A spin spiral's degeneracies, the Kramers pairs of a spin-orbit run (`OPEN.md` Part I item 1,
closed at the solver's promise) and the tetrahedron average (B2, closed) are degeneracies the
code handles and this plan does not reopen. Nothing here touches `pw.x`'s own behaviour:
`cegterg.f90` returns the same arbitrary rotation and `iweights.f90` fills the lowest bands in
index order whatever their energies (read 2026-10-14, item 2), so every refusal in item 2 is
stricter than the reference, which is the trade to decide.
