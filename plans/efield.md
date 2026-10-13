# Plan: external electric fields (`tefield`, `dipfield`, `lelfield`)

Repository: /u/40/ladovj1/data/Documents/programs/claude/defuqumat (package `defuqumat/`).
QE reference: quantum_espresso/qe-7.5-ReleasePack/qe-7.5/ (PW/src, test-suite).

## What exists today
- The three switches are refused at input in `defuqumat/system/builder.py` `_REFUSED_SWITCHES` (~line 1614).
- `defuqumat/scf/fields.py` is the template for an external term: the **energy is the primitive**,
  the potential is `jax.grad` of it, it is added to `v_scf` inside `Calculation.potential`
  (`defuqumat/scf/driver.py` ~4524, `_field_potential`), and `Potential.e_field` carries its energy.
  (There the field energy is *excluded* from etot, QE's `add_bfield` convention; here it must be *included*.)
- Forces are `jax.grad` of a frozen energy functional, `defuqumat/forces/energy.py` (`frozen_energy`),
  checked by the identity "frozen_energy at the converged geometry = SCF total energy".
- Hubbard is a low-rank per-k operator attached to `Hamiltonian`/`SpinorHamiltonian` with
  `apply(psi, ik)`, `diagonal(ik)`, `matrix(ik)` (`defuqumat/hamiltonian/operator.py` ~199, 472, 521, 631, 715).
- The Berry-phase polarization (P56) is `defuqumat/topology/polarization.py` + `topology/links.py`
  + `workflows/polarization.py`: string phases, QE's branch convention, ionic phase. Refuses nspin=2, metals, spirals.

## Part A: saw-tooth field and dipole correction (`tefield`, `dipfield`)
QE: `PW/src/add_efield.f90`, `compute_dip.f90` (`compute_el_dip`, `compute_ion_dip`), `extfield.f90` (`saw`),
called from `setlocal.f90` (into `vltot`, tefield without dipfield) and from `v_of_rho.f90` (into v_scf, with dipfield).
Energy `etotefield` added to etot in `electrons.f90` ~1176; forces `forcefield` in `forces.f90` ~286.

Central claim: QE's two code paths are the gradient of ONE energy
    E_f(rho, tau) = (Omega/4pi) e2 [ -eamp D + [dipfield] D^2/2 ],
    D = ion_dip(tau) - el_dip(rho),
    el_dip = (4pi/Omega) * (Omega/N) sum_r rho(r) saw(x_r) alat/bmod,  ion_dip = (4pi/Omega) sum_a Z_a saw(b_edir . tau_a) alat/bmod
with rho the total charge (sum over collinear channels; component 0 of an nspin=4 stack), electrons positive.
dE/drho(r) = e2 (eamp - D) saw(x_r) alat/bmod, which is exactly QE's added potential in both branches;
-dE/dtau = e2 (eamp - D) Z bg_edir/bmod in the linear region of saw, which is QE's `forcefield`.
With dipfield=false, QE's printed `etotefield` is only the ionic part (the electronic part sits in eband through vltot);
the total is the same.

Implementation:
1. `defuqumat/scf/efield.py` (new): frozen `eqx.Module` `SawtoothField` (edir, eamp, emaxpos, eopreg, dipfield static/leaves),
   `saw()` as `jnp.where` (trap: no clamp), `energy(rho_total, positions, cell)`, `potential = jax.grad`.
2. Wire into `Calculation.potential` beside the magnetic field: add `v` to every charge channel of v_scf
   (each collinear channel; component 0 only for nspin=4), carry the energy as a new `Potential` field,
   and ADD it to the total energy.
3. Add `E_f` to `frozen_energy` so forces (and the ultrasoft augmentation part through rho_aug(tau)) come from grad.
4. Builder: read edir/eamp/emaxpos/eopreg, drop the two refusals, force `nosym` as QE does (input.f90 ~673,
   with a warning), refuse stress / vc-relax (QE switches stress off), refuse `gate` still.
5. Mixing: QE mixes `el_dipole` as an extra degree of freedom of the density with metric (e2/2) d1 d2 Omega/4pi
   (`scf_mod.f90` ~333, 847). Plan: do NOT add it; the fixed point is the same, only the iteration count may differ.
6. Validation: `pw_dipole/dipole.in` (US, smearing, Ni/C/O, K_POINTS gamma -> explicit k=0, dipfield, eamp=0):
   total energy and the printed dipole; `pw_relax/relax-el.in` (CO, US, tefield eamp=0.001): first-step energy,
   `electric field correction = 0.01440000`, forces 0.11753844, and the relaxation path.
   Independent identity: finite difference of the total energy in eamp equals -(Omega/4pi) e2 D... (Hellmann-Feynman in eamp),
   and a central difference of E in tau equals the force.
7. Spin: nspin=1,2 (QE allows LSDA), and nspin=4 too since the term is a scalar potential (QE refuses nspin>2; we would extend).

## Part B: finite homogeneous field by the Berry phase (`lelfield`)
Literature: Souza, Iniguez, Vanderbilt, PRL 89, 117602 (2002), arXiv:cond-mat/0205633;
Umari and Pasquarello, PRL 89, 157602 (2002) (the QE implementation's paper; the h_epsi operator).
QE: `h_epsi_her_set.f90` (dual vectors), `h_epsi_her_apply.f90` (operator), `c_phase_field.f90` (polarization,
energy `en_el = -efield*(el_pol+ion_pol)`, `electrons.f90` ~1593), `forces_bp_efield.f90`, `c_bands.f90` (nberrycyc).

Central claim: QE's operator is H_E = |a><u| + |u><a| with a_n = fact*(e~_- - e~_+), e~_+ = sum_m |u_{k+1,m}> (M^{-1})_{mn},
M = <u_k|S|u_{k+1}>, and <a_m|u_n> = 0 identically (since <e~_+-,m|u_n> = delta), so H_E u_n = a_n, which is
a_n = dE_P/d<u_n| with E_P = -Omega E . P_el[psi] the Berry-phase enthalpy. So:
1. Write E_P(psi_all) as a differentiable function (the P56 string phase as Im ln det of link matrices, with the
   ultrasoft cross-overlap S_{k,k+b}, and the G-shift at the string end -- "align by Miller index" trap).
2. At each outer iteration (and each of `nberrycyc` inner cycles) take a = jax.grad(E_P) w.r.t. conj(psi) at the
   current frozen wavefunctions, and attach `BerryFieldOperator(u, a)` per k with apply/diagonal/matrix like Hubbard.
   The diagonal for the preconditioner: 2 Re(conj(a) u) per plane wave, or zero.
3. Total energy = E_KS + E_P(el) + E_P(ion) = QE's etot + en_el.
4. Forces: add E_P to `frozen_energy`; the ionic part is -e Z E, the ultrasoft part is the tau dependence of S_{k,k+b}
   (QE `forces_us_efield`).
5. Stress: QE computes it for norm-conserving (sigmael, sigmaion), refuses for US. Plan: jax.grad in strain of the same
   functional, compared with electric-2's P=46.70 kbar; if QE's is missing a term, say which.
6. Scope: nspin=1, NC and US, one direction `gdir` with `nppstr`, explicit k-list in string order (QE's electric.in
   form). Refused by name: nspin=2 and 4, PAW, metals, `efield_cart`/`l3dstring` (three directions), a reduced k-set.
7. Validation: `pw_electric/electric-2.in` (Si8 NC PBE, efield=0.001, nberrycyc=3): total energy -63.06610313,
   electronic dipole 0.93125880679857731, ionic dipole 115.17355251966487, forces ~0.0126, stress P=46.70.
   It is a continuation of electric-1 (efield 0) which continues electric.in; we start from scratch.
   Identities: E(efield) - E(0) = -Omega E.P to first order; dP/dE by finite fields equals P55's
   Sternheimer epsilon_inf (or chi) on the same k-mesh -- a check that shares no machinery with the Berry operator.

## Deliverables (per CLAUDE.md)
PLAN.md P144 section with the numbers, README row, docs/features.tex entry with executed snippet + amber box,
a notebook (slab with dipole correction, or Si dielectric constant from finite fields), a timing against pw.x.
Calculator: `get_scf` etc. already take the System, so the fields arrive by input; a `Calculator` option for
python-built systems (edir/eamp/...) and lelfield (efield, gdir, nppstr, nberrycyc).
Tests run on D22, not this workstation.

## Questions for the reviewer
1. Is the Part A central claim true as written, including the sign of el_dip (QE: tot_dipole = -el + ion)
   and the claim that the ultrasoft force contribution (augmentation charge moving through the saw) comes out
   of frozen_energy with nothing else? Check against add_efield.f90 / forces.f90 / addusforce usage of vltot.
2. Part A mixing: is it safe to omit QE's mixed `el_dipole` (scf_mod.f90 ~333, 847)? Any reason the fixed point differs?
3. Is the Part B claim <a_m|u_n> = 0 and H_E u_n = dE_P/d<u_n| right, including the ultrasoft S and the factor
   convention of fact_hepsi (h_epsi_her_set.f90)? If H_E built from jax.grad differs from QE's operator off the
   occupied manifold (on trial vectors outside span(u)), does the converged state still coincide?
4. Should Part B be in this phase at all, or split (P144 Part A, P145 Part B)? Recommend.
5. Symmetry: follow QE's nosym=.true. for tefield, or reduce the group to operations preserving edir and the saw origin?
6. nspin=4 for Part A: extend beyond QE (which refuses), or refuse to match?
7. Anything in the driver (`defuqumat/scf/driver.py`) that would make an added k-coupled operator (Part B) hard:
   look at how hubbard reaches Hamiltonian and how the Davidson is vmapped over k (map_k), and say where the
   per-iteration frozen `a` vectors should live.

## Fable's review (2026-10-13), and what was done with it

- Q1 TRUE. The relax-el printed correction checked by hand: `-2 * 0.001 * (6*(-0.16) + 4*0.06) * 10 = +0.01440000`.
  The ultrasoft force comes out of `frozen_energy` with nothing else, since `addusforce` integrates `vltot + v_scf`.
  Use `total_charge` and `as_potential_components`; do not route through `magnetic_field` (its energy is excluded).
- Q2: omit QE's mixed `el_dipole`; same fixed point. Measured: not needed on the CO cell (the force spread at
  conv_thr 1e-10 was convergence in a mostly empty cell, with or without dipfield, gone at 1e-13).
- Q3: `<a_m|u_n> = 0` TRUE; `H_E u_n = dE_P/d<u_n|` FALSE as written: (i) `H_E u_nk = (1/w_k) dE_P/d<u_nk|`,
  w_k = 2/nk at nspin = 1; (ii) `jax.grad` of a real E returns `2 conj(dE/d conj u)`; (iii) ultrasoft operator is
  `sum_m |a_m><u_m| S_k + S_k |u_m><a_m|`. `c_phase_field` takes the per-string principal value, not bp_c_phase's
  averaged branch; `calc_pol` tracks 2 pi jumps (`el_pol_acc`). QE's lelfield stress has the field terms commented
  out (`stress.f90:170-171`), so electric-2's P = 46.70 kbar has no field term.
- Q4: split. P144 = tefield/dipfield, P145 = lelfield. Taken.
- Q5: follow QE's nosym. Taken.
- Q6: nspin = 4 could be extended with an identity; done for nspin = 2 (unpolarized LSDA = spinless, measured),
  nspin = 4 left as the scalar rule in `as_potential_components` without its own number.
- Q7: Hubbard is the template for the Berry operator; build `a` outside `map_k` per outer iteration and per
  `nberrycyc`; the solve donates its starting block, so the operator's `u` must be a separate array; zero
  preconditioner diagonal.
