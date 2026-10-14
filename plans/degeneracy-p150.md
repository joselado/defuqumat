# P150: the plan as sent to fable, and fable's review

*The plan is the scratchpad file of 2026-10-14 as reviewed; the review follows it verbatim. What was done differently from the plan is in `PLAN.md` P150.*

## The plan: the implicit (Sternheimer) Jacobian of the SCF map

The repository at master `764bc3e`.

### What is open, and why one route closes both items

`PLAN.md` "What is outstanding", "Degenerate multiplets, beyond the filling cut", has two items:

1. `OPEN.md` Part XXVI: `ScfResidual.jvp` (`defuqumat/scf/residual.py`) differentiates Davidson as
   implemented. With a degenerate block in the subspace (si-gamma: NC silicon, 2x2x2 mesh with Gamma)
   it is 1.2e-2 off the central difference warm, 16 per cent cold, growing with the Davidson cap,
   after P148's regularised subspace `eigh` tangent. The amplifier is one member's correction vector
   on a degenerate block. The exact route named in the record is the implicit tangent of the converged
   eigenproblem (P22c).
2. The divided difference `(f_n - f_m)/(e_n - e_m)` in the density's tangent at a smeared Fermi level,
   which P148's regularised kernel suppresses within ~10 eta.

Claim: both are closed by **not differentiating the eigensolver at all**. At a point `x`, `F(x)` is
built from the eigenpairs of `H[x]` (Davidson at `ethr`). The derivative of the density assembly along
`v` is the static independent-particle response of those eigenpairs to `dH = dH/dx . v`, which is
exactly what `SternheimerSolver` (`defuqumat/response/sternheimer.py`) already computes for the
response stack: the insulator branch needs only `P_c`, invariant under any rotation inside the occupied
manifold (so a degenerate block that is wholly occupied is exact), and the metal branch
(`_smeared_projection`, de Gironcoli's `wwg`) already carries the divided difference with its
degenerate limit `-alpha_pv t w0gauss` (item 2). The Fermi level of `F` is re-bisected at every `x`, so
the tangent carries the `ef_shift` (`fermi_level_shift`, `fermi_level_shift_states`).

### The construction (new `jvp='sternheimer'` backend of `newton_krylov`)

`ScfResidual.sternheimer_jacobian(x, psi0, threshold=None) -> callable v -> (J v)` built once per outer
Newton iteration (`x` is fixed during GMRES, so the solver, Hamiltonians, eigenpairs, `ldos` are built
once and every matvec is one Sternheimer solve plus one linear assembly):

1. `rho, becsum_ = unpack(x)`; `potential = calc.potential(rho)`; `ddd_paw = calc.onecenter(becsum_)[1]`;
   `ham = calc.hamiltonian(potential.v_scf, ddd_paw)`; `eig, psi = calc.diagonalize(ham, nbnd, psi0,
   ethr)` -- the same eigenpairs `F(x)` is built from (warm start: psi0 is the converged psi at x, so
   this is a one-step Davidson).
2. `wg, info = calc.occupations(eig)`; `Smearing(ef=info['fermi_energy'], ...)` for a smeared run.
3. `solver = SternheimerSolver(calc, ham, psi, eig, wg, occupied_counts(calc), threshold,
   v_scf=potential.v_scf, becsum=becsum_, smearing=..., kpoint_weights=calc.system.kpoints.weights)`.
4. Per `v`: `drho_in, dbecsum_in = unpack(v)`; `dv = jvp(rho -> calc.potential(rho).v_scf)(drho_in)`
   (Hartree + XC kernel, compiled); `dddd_paw = paw_response(calc, dbecsum_in, becsum_)` (PAW only);
   `dpsi = solver.solve(solver.perturbation(dv, dddd_paw)).dpsi`.
5. Metal: `drho_raw = solver.response_density(dpsi)`; `shift = -int(charge_channel(drho_raw)) / dos_ef`
   (computed here with `calc.charge_channel`, not with `solver.fermi_level_shift`, which sums every
   channel axis and so would add the magnetization for `nspin_mag = 4`; that path is unreachable today
   because phonons refuse noncollinear, recorded rather than fixed); `dpsi =
   solver.fermi_level_shift_states(dpsi, shift)`.
6. The output tangent: `dpsi` padded to `nbnd` with zeros (insulator: `nocc < nbnd`), weights
   `W = solver.density_weights` padded with zeros; `(drho_out, dbecsum_out) = jvp over psi of
   (psi -> (calc.density(psi, W, calc.becsum(psi, W)), calc.becsum(psi, W)))` along `dpsi`. This is
   F's own assembly, so it carries the augmentation, the US/PAW becsum group average (P148) and
   `sym_rho`, all linear. With the metal's `wk` weights rather than `wg` because the occupation is
   already inside `dpsi` (the module's own rule).
7. `J v = flatten(drho_out, dbecsum_out) - v`.

Symmetry: `F` sums the wedge and applies `sym_rho`; for a non-symmetric `v` the derivative of *that map*
is `sym_rho` of the wedge sum of the responses, which is what step 6 does. (The trap "a response on a
reduced k-set is symmetrised as a polar vector" is about the physical full-zone response to a
directional perturbation; here we differentiate `F` as defined, and the reference is a central
difference of the same `F`.)

Refused by name, in `sternheimer_jacobian`: DFT+U (`dns`, `adddvhubscf`), meta-GGA (`dH` from `dtau`
is not a local potential), a fixed filling that cuts a degenerate shell at any `nspin`
(`_require_a_gap_at_the_cut` is today only called for nspin=2 / noncollinear; here no derivative
exists), and every regime `require_a_sternheimer_regime(metals=True, spin_polarized=True,
noncollinear=True, gamma_ok=True)` refuses (sawtooth, fields, tetrahedra, from_input, spirals, two
Fermi levels with smearing).

CG threshold: `SternheimerSolver`'s `THRESHOLD = 1e-11` by default. Alternative: tie to GMRES's
forcing.

### Measurements (D22, `JAX_PLATFORMS=cpu`, reuse `review/degeneracy/cut_jvp.py`)

Cells: `si-generic` (no degeneracy), `si-gamma`, `o-smear` (smeared O, triplet split 1.8e-14),
`o-smear-nosym`, plus `al-metal.in` (smeared metal, the item-2 trigger), an ultrasoft and a PAW cell
(`si-epsilon-us`, a PAW si), `nspin = 2` (a smeared magnetic cell) and a nonmagnetic spinor
(`alas-epsilon-us-soc` or smaller). For each: `|Jv_stern - Jv_fd|/|Jv|` at eps 1e-4..1e-7 (expect the
`eps^2` law down to the FD floor on every cell, si-gamma included), against autodiff's numbers.
Gauge test: rotate `psi` by a random unitary inside each degenerate block (gap < 1e-8) before
building the solver: `J v` unchanged to the CG threshold (for autodiff this is not even defined).
Newton end-to-end: si-gamma and al with `jvp='sternheimer'` vs `'finite-difference'`: outer steps,
F evaluations, wall time warm.

Acceptance: si-gamma within 10x of si-generic at each eps (the fable trigger for Part XXVI); al's
Sternheimer-vs-FD at si-generic's level; gauge test at the CG threshold.

Default: stays `'finite-difference'` unless the end-to-end measurement says otherwise (fable decides).

### Record the rest (not fixed)

- A gauge sweep over the `Calculator` front door (64 `get_*`, 9 test files with a gauge test, mostly on
  model data): rotate the converged eigenvectors inside every multiplet and measure every observable.
  Into `PLAN.md` "What is outstanding" and `notes/DEGENERACY-NEXT.md` as item 7.
- `SternheimerSolver.fermi_level_shift` sums the magnetization into the charge for `nspin_mag = 4`;
  unreachable today; one line in `OPEN.md`.

### Questions for the reviewer

1. Is the central claim true: does steps 1-7 give the exact derivative of `F` as implemented (to
   Davidson's `ethr` and the CG threshold), in all of NC/US/PAW, nspin 1/2, nonmagnetic spinor, metal
   and insulator, gamma storage? In particular: the US/PAW `S` does not depend on `x` (so no `dS`);
   for PAW is `dddd_paw` from the *input* `dbecsum` the only one-centre term; does `perturbed_coefficients`
   (`int3`) cover the US `D_ij(V)` dependence; is anything else in `F` (e.g. `ns`, `tau`, the
   `vltot`, the sawtooth) x-dependent that this misses.
2. The metal: is `wwg` with `alpha_pv` and the `ef_shift` applied on states the exact derivative of
   `F`'s `wg = smeared_occupations(eig)` assembly, including Methfessel-Paxton and cold smearing, and
   is the band cutoff `projector_mask = eig < ef + xmax degauss` consistent with `F` filling every
   `nbnd` band (bands above the cutoff carry `wg ~ 1e-?` in `F`)?
3. Fixed occupations with `nbnd > nocc`: the insulator solver keeps only `nocc` bands. `F`'s
   `fixed_occupations` gives the empty bands zero weight, so their tangent does not enter. Correct?
4. CG threshold: fixed 1e-11, or something tied to `ethr`/forcing?
5. Anything in the plan that would not have read differently if it were wrong (rule: "a confirming
   number counts only if the alternative would have read differently")? Which cell would break it?

## Fable's review (2026-10-14)

**Verdict on the central claim.** True, with one correction and one reframing. Steps 1 to 7 are the exact directional derivative of F's assembly (`residual.py:174-198`) for the eigenpairs of H[x], invariant under any rotation inside a degenerate block, provided the metal branch is built with every band in the projector mask (below). The reframing: what the route differentiates is the `ethr -> 0` limit of F, the exact-eigenpair map, not Davidson as implemented. That is the right object (Part XXVI shows F as implemented is not differentiable on a degenerate block), but the text should say "the derivative of the map F approximates, to Davidson's `ethr` on the eigenpairs and the CG threshold on `dpsi`", since the central difference of F sees Davidson's own error and the Sternheimer does not.

**Q1 (terms of F).** Checked term by term against `step`: `potential(rho)` is one `jvp` (step 4), `S` carries only the dataset's `q_ij` so there is no `dS`; the US `D_ij(V)` dependence is `local_terms`' `deeq = coefficients(total, ddd_paw)` (`driver.py:5660`), and `perturbed_coefficients` (`sternheimer.py:1388-1391`) is the `jvp` of that same function along `(dv, dddd_paw)`, so for PAW `dddd_paw = paw_response(dbecsum_in, becsum_in)` (`:1407-1409`) is the only one-centre term, matching F's `onecenter(becsum_in)` at `residual.py:181`. `vltot`, `rho_core`, the quantization axis are constants. The symmetrisation matches by construction: F calls `calc.becsum(psi, wg)` with the group average (`driver.py:4835-4836`) and `calc.density(psi, wg, becsum_out)` ending in `symmetrize(augmented(lift(...)))` (`:5536`); step 6 differentiates the identical calls, both linear in the tangent, so the jvp is a central difference of F to round-off. Reusing `solver.density_weights` is right in both branches: insulator `W = wg[:, :, :nocc]` (`:416`, zero on a deficient channel's empty bands), metal `W = wk` on every band with `f` inside `dpsi` (`:418-420`, the `wg1` scaling at `:660`); `calc.becsum` is bilinear in `psi` with the same weights, so `dbecsum` and its Fermi-shift term come out of the one jvp once `fermi_level_shift_states` has been applied to `dpsi` (the phonon applies the shift to `drho` only, `phonon.py:806-809`, so do not copy that part). Nothing is double counted and nothing symmetrised twice.

**Q2 (metal), the one real defect.** Inside the solved set (`keep = nbnd`, `:383`) every pair weight is exact whatever the mask: for `j` outside it the bare operator gives `(f_i - f_j) t_ji / (eps_i - eps_j)`, for `j` inside the `alpha_pv` term in `wwg` (`:657`) cancels `Q` exactly, and the two orderings sum to the divided difference through `t_ji + t_ij = 1`. The complement is weighted `f_i / (eps_i - eps_j)` with `f_j = 0`, which is F's own truncation at `nbnd`. The miss is the diagonal: the term `-wk f'_i/degauss <i|dV|i> |psi_i|^2` (F's `dwg_i` from `deps_i`) is delivered through the `alpha_pv` channel and only for `i` in `projector_mask` (`:658`); for `i` above `ef + xmax degauss` (`xmax = 3` for every Gaussian-type smearing, `:300`) the rhs component along `psi_i` is `f_i - wwg_ii = 0` and the term is dropped. Its size is `w0gauss(-3) = 7e-5` per band, so the acceptance "al at si-generic's 3.5e-7" would fail for a reason that is not a bug. Fix for the Jacobian: `projector_mask = True` on every band and `alpha_pv = 2 (eps_max - eps_min)` over the whole set (then `eps_j - eps_i + alpha_pv >= eps_max - eps_min > 0` on every row, and the diagonal term is carried for every band F fills). Both are attributes set after `__init__`; `_alpha_pv`'s metal value `cutoff - emin` (`:1479`) is not positive definite on rows above the cutoff anyway. The `ef_shift` is right: `-int drho_raw / dos_ef` equals `bisect_fermi`'s custom tangent `sum w f' deps / sum w f'` (`occupations.py:567-575`) because the off-diagonal pair densities integrate to `<n|S|m> = 0`; `charge_channel` for the integral is correct (`fermi_level_shift` at `:1157` sums the channel axis, wrong only at `nspin_mag = 4`). MP and cold are covered by the formulas (`wgauss(.., ngauss)` at `:647`, `w0gauss` at `:648`); `al-metal.in` is Marzari-Vanderbilt at `degauss = 0.05`, so the asymmetric case is in the measurement. Gauge invariance on a pair the solver splits by `delta`: the `close` guard at 1e-5 Ry (`:654`) uses the exact `-f'` limit, so `J v` is invariant to `O(delta/degauss)`, not to the CG threshold; state the test's tolerance that way.

**Q3.** Correct: `fixed_occupations` gives `wg = wk (band < nocc)` (`occupations.py:98-99`), the empty bands' tangent is multiplied by zero in F, and `P_c` drops the occupied-occupied rotations, which cancel in the density. The refusal `_require_a_gap_at_the_cut` needs `nbnd > nocc` to see anything (`:1437-1439`); with `nbnd == nocc` the cut is trusted, as today.

**Q4.** Fixed 1e-11 for the accuracy measurement (so the CG is not the floor), then make it an argument and measure the Newton end to end at 1e-8 against 1e-11 (GMRES iterations, wall time). Two notes: the test at `solve_at:749` is absolute, so normalise `v` before the solve and scale `dpsi` back; and below about `sqrt(ethr)` the eigenpairs' own residual bounds `dpsi` whatever the CG does. `pass_threshold`'s schedule is for a self-consistent loop and does not apply, one solve per matvec.

**Q5.** Cells that would not read differently if wrong: no US or PAW *metal* (the only place a double-counted `f` in `becsum` or a missing shift on `dbecsum` shows); Al rrkjus and kjpaw datasets are in `tests/data/pseudo`, so add one. No fixed-count LSDA insulator (the deficient-channel mask branch, `:590-611`); add one or refuse `two_fermi_energies` with fixed occupations as unmeasured. Measure si-generic with `nosym` as planned, since a symmetric cell hides a term the group zeroes.

**Refuse that the plan does not.** A streamed host store: `calc.density` on one goes through `stream_densities` (`driver.py:5411`), which has no jvp. Magnetic noncollinear (`nspin_mag = 4`): `noncollinear=True` lets it through with no cell measuring it. Do **not** inherit the response stack's `nspin = 2` GGA refusal: that one is about matching `dgcxc_spin`, while here the jvp of F's own `potential` is F's derivative by definition.

**Smaller.** `step` returns `psi` but not the eigenvalues (`residual.py:198`); return them and skip step 1's re-diagonalisation, so "the same eigenpairs" is literal. Working set is three wavefunction sets (`self.psi` is a sliced copy at `:388`, plus `dpsi`), state it.

**Default.** Decide after the end-to-end run, on wall time to `conv_thr` on `al-metal` and `si-gamma` warm (FD costs 2 F per matvec, the Sternheimer one CG solve of `nocc` bands); both converge, so accuracy is not the deciding number. Keep `finite-difference` unless the Sternheimer wins on both cells.
