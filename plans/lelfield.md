# Plan: P145, the finite homogeneous field by the Berry phase (`lelfield`)

Follows `plans/efield.md` (P144 done: tefield/dipfield) and fable's review there (Q3, Q7).
Literature: Souza, Iniguez, Vanderbilt, PRL 89, 117602 (2002), arXiv:cond-mat/0205633; Umari and
Pasquarello, PRL 89, 157602 (2002); Souza et al. PRB 69, 085106 (2004) (the Hermitian operator QE uses).
QE: `h_epsi_her_set.f90`, `h_epsi_her_apply.f90`, `c_phase_field.f90`, `c_bands.f90:c_bands_efield`,
`electrons.f90:calc_pol`, `forces_bp_efield.f90:forces_ion_efield`, `input.f90`, `setup.f90`.

## Scope of this phase
Norm-conserving, `nspin = 1`, insulator (fixed occupations), one direction `gdir` with `nppstr`
(`l3dstring`/`efield_cart` refused), an explicit k-list of strings as QE's `electric*.in` lay it out,
`nosym`. Refused by name: ultrasoft and PAW (the cross overlap `q_dk` with the structure phase in the
duals, and `forces_us_efield`), `nspin = 2` and 4, metals, `efield_cart`, the stress (QE's field stress
terms are commented out, `stress.f90:170-171`, so its printed stress has no field term and there is no
reference), and a cell where `a_gdir` is not along Cartesian axis `gdir` (QE's ionic dipole takes
`tau(pdir)` Cartesian while the electronic one is along `a_gdir`; both agree only then).

## The physics, as QE has it
- Strings: for `gdir`, `nkort = nks/nppstr` strings of `nppstr` DISTINCT points (the step `dk = G_gdir/nppstr`;
  the last point's neighbour is the first displaced by `G_gdir`, `u_{k+G}(G') = u_k(G'+G)`).
- `M^+(k)_{nm} = <u_{k,n}|u_{k+dk,m}>` over the occupied bands, aligned by Miller index (the trap).
- Duals (`h_epsi_her_set`): `e+_m = sum_n |u_{k+dk,n}> (M^+)^-1_{nm}`, `e-_m = sum_n |u_{k-dk,n}> (M^-)^-1_{nm}`,
  both mapped onto k's sphere. `<u_{k,j}|e+-_m> = delta_jm`.
- `fact = -i E sqrt(2) / (2 dkfact)`, `dkfact = tpiba / |a_gdir| / nppstr` (alat units for `|a_gdir|`),
  E = `efield` in Ry a.u.
- Operator (`h_epsi_her_apply`, NC): `H_E psi = fact (e- - e+) <u|psi> + conj(fact) u (<e-|psi> - <e+|psi>)`,
  i.e. `|a><u| + |u><a|` with `a = fact (e- - e+)`, so `H_E u_n = a_n` (since `<a|u> = 0`).
- Energy: `en_el = -E (el_pol + ion_pol + el_pol_acc)`, `el_pol = 2 sqrt(2) |a_gdir| alat/(2 pi) * (1/nkort)
  sum_s Im ln prod_k det M^+` per-string PRINCIPAL value, `ion_pol = sqrt(2) alat sum_a Z_a tau_a(gdir)`
  (Cartesian, alat units times alat = bohr). `el_pol_acc` adds +-2 pi fc_pol when the phase jumps by more
  than pi between iterations (`calc_pol`).
- `nberrycyc`: inside each SCF iteration, `nberrycyc` times: rebuild the duals from the current
  wavefunctions, diagonalise (`c_bands_efield`).
- Forces: `forces_ion_efield = sqrt(2) E Z_a` along Cartesian `gdir`; NC has no other explicit field force.

## Implementation
1. `defuqumat/scf/berryfield.py`: `BerryField` settings, string layout from the k-list (validate:
   consecutive, `nppstr` per string, the step `G/nppstr`, equal weights), `string_overlaps` (reuse
   `topology.states.PlaneWaveStates` alignment, or a jit-able gather built once from Miller indices,
   since the k-set is fixed), `duals(u)` -> `(e+, e-)` as `(nk, nocc, npwx)`, `operator(u)` ->
   `BerryTerm(u, a)` with `apply/diagonal/matrix` like `HubbardTerm` (diagonal zero), `phase_energy(u)`.
2. `Hamiltonian`/`SpinorHamiltonian`: a `berry` slot beside `hubbard`, in every apply variant, `matrix`
   and `diagonal` (spinor refused for now).
3. Driver: build the term from the current wavefunctions (a copy, not the donated block) before each of
   `nberrycyc` diagonalisations per iteration; the first iteration's wavefunctions come from the starting
   guess (QE's c_bands_efield does the same at iter 1). Energy: `terms["efield"] = en_el` with the branch
   accumulator on the host; `SCFResult.berry_polarization` (el, ion, acc) and `efield_energy`.
4. Forces: add `-E (el_pol(psi) + ion_pol(tau))` to `frozen_energy` (`el_pol` has no tau for NC, so the
   autodiff force gets `sqrt(2) E Z` exactly); `forces/analytic.py` transcribes `forces_ion_efield`.
5. Builder: read `lelfield`, `efield`, `gdir`, `nppstr`, `nberrycyc` (default 1), drop the refusal; refuse
   `efield_cart`/`l3dstring`, a reduced k-set, non-`nosym` (QE errors on it: check `setup.f90:137`).

## Validation
- `pw_electric`: regenerate with the vendored 7.5 `pw.x` the chain electric.in -> electric-1 -> electric-2
  (startingwfc/pot = 'file'), store `reference.out.pw_electric-electric-2`; compare total energy
  (-63.06610313 in the 6.1 benchmark), electronic dipole 0.93125880679857731, ionic 115.17355251966487,
  forces 0.0126. Ours runs from scratch at the same field.
- Unit identity (no SCF): `H_E u_nk = (1/w_k) dE_P/d<u_nk|` with `dE/d<u| = conj(jax.grad)/2`, on random
  orthonormal states over a small string; `<a|u> = 0`; gauge invariance of `E_P` under a unitary rotation of
  the occupied block at one k.
- SCF identity: central difference of the total in `efield` equals `-(el_pol + ion_pol)` (Hellmann-Feynman
  for the enthalpy), and `efield = 0` reproduces the plain SCF.

## Questions for the reviewer
1. Are the operator, `fact`, the duals' index conventions and the energy normalisation above exactly QE's?
   In particular the sign of `fact` relative to `en_el = -E(el+ion)`: with `a = fact (e- - e+)` is `H_E u = +(1/w_k)
   dE_P/d<u|` (consistent) or minus? Give the algebra.
2. QE builds the duals for the occupied bands only (`l_cal`) while `nbnd` may be larger: what does the operator do
   on the empty bands in the Davidson (`a` zero for them? `mat` set to identity rows), and does it matter for
   electric-2 (nbnd default = 16 = occupied)?
3. Is the first iteration's operator built from the random/atomic starting wavefunctions in QE (c_bands_efield at
   iter 1), and is there anything in `electrons.f90` that delays the field (e.g. only from iteration 2)?
4. Is reusing `PlaneWaveStates` overlaps inside a per-iteration build fine for cost on electric-2 (63 k,
   16 bands, ~ 1000 pw), or should the Miller gather be precomputed as index arrays once?
5. The branch: from scratch, is QE's first `calc_pol` the principal value per string with `el_pol_acc = 0`, so a
   fresh run of electric-2 matches the continuation's energy exactly when no jump happens during the run?
6. Anything else that would give a plausible wrong number.
