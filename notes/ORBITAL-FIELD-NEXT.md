# The orbital magnetic field: Landau levels from first principles, the plan

Drafted 2026-11-07 at the user's request ("implement calculations with orbital magnetic
field, like the one that gives rise to landau levels in graphene ... maybe ultra long range
formalism could be a way ... come up with a plan, check the plan with fable"). Nothing here
is implemented. The phase numbers P175 onwards are proposed, not taken.

`pw.x` has no uniform orbital field: `grep -il 'landau|peierls|gipaw'` over `PW/src/*.f90`
returns nothing (GIPAW is a separate package, perturbative only). Elk has none either:
`afieldc` is a *constant* vector potential, which is a rigid shift of `k` and carries no
`B`, and `grep -il 'landau|peierls'` over `vendor/elk/src/*.f90` returns nothing. So the
README row will carry no tick in either column; the `reference-source` agent confirms that
before the row is written.

## 1. The obstruction, and why the ultracell as built cannot carry a uniform field

A uniform `B` has a vector potential that grows linearly in space, `A = B x y_hat` in the
Landau gauge, so the Hamiltonian `(p + A)^2/2 + V(r)` is not periodic. What survives is the
**magnetic translation** `T_R = e^{-i A(R).r} t_R`, which commutes with `H` for every
lattice vector `R`. Two magnetic translations commute only when the flux through the
parallelogram they span is an integer number of flux quanta, so the states are Bloch-like
only on a **magnetic cell** carrying `n_phi` flux quanta. At flux `p/q` per chemical cell the
magnetic cell is `q` chemical cells. The states then obey the **magnetic Bloch theorem**
(Lü et al. eq. 2.2 and 4)

    u_k(xi1 + 1, xi2, xi3) = exp(-i 2 pi n_phi xi2) u_k(xi1, xi2, xi3),
    u_k periodic in xi2 and xi3,

with `xi` the fractional coordinates of the magnetic cell. The twist is not a gauge choice
that a better gauge removes: the phase winds `n_phi` times around the torus, which is a
Chern number of the line bundle the states live in, so **no set of functions periodic over
the magnetic cell spans them**. Lü's auxiliary function `h = exp(i 2 pi n_phi xi1 xi2) u`
moves the twist from `xi1` to `xi2`; it does not remove it.

This decides the ultracell question. P88's basis `{psi_{k+Q,n}}` is periodic over the
ultracell, so "P88 plus an `A.p` term" is not a Rayleigh-Ritz truncation of the magnetic
problem: it is a truncation in the wrong space, and no `nbnd` converges it. What the
ultracell can carry as built is a vector potential that is itself periodic over the
ultracell, meaning a modulated field `B(r)` with **zero net flux** through the ultracell
(section 6). Landau levels need net flux.

Every route that works puts the twist inside the basis. They are ranked here by where it
lives:

| route | where the twist lives | field reached | cost per state | exact up to |
|---|---|---|---|---|
| W0, Wannier + Peierls | the hopping phases | any, 10 T is `q ~ 7900` | `q num_wann`, sparse | Peierls, `O(B)` intra-cell terms dropped |
| W1, gauge-including Wannier functions | the basis functions' phases | any | same, plus the field-dressed onsite and hopping blocks | the Wannier span |
| P, magnetic Bloch plane waves (Lü 2025) | one twisted axis of the real-space grid | `q <~ 30` here, `>~ 2600 T` in graphene | `q` times a supercell | the grid, exact otherwise |

The ultracell idea does survive in W1, in a different form: the fast, unit-cell part of the
wavefunction is computed once at zero field, and the field acts only on the envelope. W1 is
that idea with localized functions instead of Bloch states, because a localized function
can carry its own gauge phase and a Bloch state cannot.

## 2. The numbers that size everything

Graphene, `a = 2.46 A`, cell area `5.24 A^2`. One flux quantum `h/e = 4.136e-15 T m^2`
through one chemical cell is `B = 7.9e4 T` (Lü quote 80125 T at their lattice constant).
So `p/q = 1/q` per cell is `B = 7.9e4/q T`, and 10 T is `q ~ 7900`, the magnetic length
`l_B = sqrt(hbar/eB) = 8.1 nm`. The first Landau level of the Dirac cone is
`E_1 = v_F sqrt(2 e hbar B) = 0.115 eV` at 10 T for `v_F = 1e6 m/s`, and about 10 percent
lower for the LDA velocity. Wang et al. (arXiv:2609.01053) reach 6.7 T (`q = 12000`) with
gauge-including Gaussians, non-self-consistent on the zero-field density, with KPM for the
Fermi level and a Krylov solve for 100 levels around it.

A plane-wave magnetic cell at `q = 30` is 60 carbon atoms with a twisted grid (section 4),
which is 2600 T: no plane-wave route reaches a laboratory field, so route P is the
**reference** that measures the error of route W at high field, and route W is the one a
user runs at 10 T.

## 3. Stage 1 (P175): Wannier + Peierls on a magnetic torus, non-self-consistent

The user's own route (Lado and Fernández-Rossier, arXiv:1606.03038, ribbons) and the one
Wang et al. use for the headline, here on a torus with magnetic translations.

**What exists.** P162's result carries the centres (cartesian bohr), `rvectors` (`irvec`),
`degeneracy` and `hoppings` `H_mn(R)`, and the Wigner-Seitz shifts. Peierls needs nothing
else; in particular it does **not** need the Wannier functions on a real-space grid, which
P162 does not produce.

**The Hamiltonian.** Magnetic cell `q a1 x a2` (a strip, so `k` along the magnetic
reciprocal vectors is kept), flux `p` through it, Landau gauge `A(r) = 2 pi n_phi (b1.r) b2`
with `b` the magnetic cell's reciprocal basis scaled to `a_i.b_j = delta_ij`. Orbital `mu`
at centre `c_mu` in chemical cell `R_c` of the magnetic cell, hopping to orbital `nu` in
cell `R_c' + P` (`P` a magnetic lattice vector):

    H_{mu nu}(k) = sum_P t_{mu nu}(R_c' + P - R_c) exp(i phi_{mu nu}(P)) exp(i k.P) exp(i A(c_nu').P)

with `phi` the line integral of `A` along the straight bond (exact for a linear `A`:
`A(midpoint).(r2 - r1)`), and the last factor the magnetic Bloch-sum phase of Wang et al.
eq. (4) and (9). That factor is the one that gets dropped, and dropping it gives a matrix
that is Hermitian and wrong. The sign convention (electron charge, `e^{-i e/hbar int A.dl}`)
is fixed once in a units helper and tested by the plaquette guard below.

**Size, and why it is matrix-free.** Dimension `q num_wann`: 10 T with `num_wann = 2`
(`p_z` on each carbon) is 15800, dense `complex128` 4.0 GB; with `num_wann = 8` (`sp3` on
both) 63200, 64 GB. Dense is refused by number above a set size. The apply is banded in the
chemical-cell index along `a1`, `(q, 2 r_max + 1, num_wann, num_wann)` with `r_max` the
hopping range, applied by `jnp.roll` along the strip: static shapes, `jit`, GPU unchanged,
and the phases traced so a later derivative in `B` survives.

**Solvers.** The DOS by KPM through `kpm/moments.py`'s `recursion(apply, start, n)` (the
apply is a callable, so the plane-wave Hamiltonian is not needed), and the levels near the
Fermi level by a Chebyshev-filtered block (or shift-invert on the banded matrix with SciPy
on the host, refused above a size). Lanczos-only paths that return levels without vectors
are not enough: the zero modes' sublattice polarisation is one of the checks.

**Validation, each one against something that shares no machinery with it.**

- The **plaquette guard**: the product of the hopping phases around every hexagon equals
  `exp(i 2 pi p/q)`, and the guard is tested to fire on a deliberately wrong sign.
- The **Dirac formula**: `E_n = sgn(n) v_F sqrt(2 e hbar B |n|)` for `n <= 10` at 10, 20 and
  40 T, with `v_F` read from the *zero-field* P162 bands at `K` (a finite difference of the
  interpolated bands, no field involved). The deviation at large `n` and `B` is the
  trigonal warping and is expected; the plan asks for the number, not a pass.
- The **two zero modes** per spin, sublattice-polarised (Wang et al. Fig. 1c), measured as
  the A and B weight of each.
- The **Landau degeneracy**: `p` states per Landau level per valley per magnetic `k`.
- **pyqula**, `pyqula/peierls.py`, on a wide ribbon built from the same `H_mn(R)`: the bulk
  Landau levels of the ribbon against the torus levels, to the exponentially small edge
  correction. This is the check that catches index and phase bugs, which is where P88's
  effort went.
- The **Hofstadter butterfly** of the `p_z` model at `q <= 50` against the textbook
  honeycomb butterfly of the same nearest-neighbour hopping (Rammal 1985), as a figure.

**Refusals, measured and named.** Spin is carried as two copies (the Zeeman term is P18's
`B_field` at the same `B`, optional and off by default); SOC Wannier functions (`npol = 2`)
are refused until stage 1 is closed. The intra-cell field terms are the named missing term,
and stage 3 measures them.

## 4. Stage 2 (P176): the exact plane-wave magnetic Bloch route, the reference

Lü, Chen, Wang, Dai, Fang, Gong, Wu and Xiang, *General first-principles approach to
crystals in finite magnetic fields*, Phys. Rev. Lett. 135, 196401 (2025), arXiv:2505.06585.
Cai and Galli, Phys. Rev. Lett. 92, 186402 (2004), and Lee, Cai and Galli, J. Comput. Phys.
226, 1310 (2007), are the forerunners without nonlocal pseudopotentials.

**The Hamiltonian.** `(pi^2)/2 + V_per + V_GIPAW`, `pi = -i grad + A`, Landau gauge. The
kinetic term is written `(1/2) sum g^{ab} pi_a pi_b` with `pi_a = a_a.pi`. `pi_2` and `pi_3`
act on `u` by an FFT along `xi2`, `xi3` and a multiplication by `A_a`; `pi_1` acts by Lü
eq. (5), `pi_1 u = -2 pi n_phi xi2 u + exp(-i 2 pi n_phi xi1 xi2) (-i d/dxi1) h`, the
derivative by an FFT along `xi1` of the auxiliary `h`. The nonlocal term carries the GIPAW
phase on every projector,

    beta^B_{tau}(r) = exp(-i A(tau).r - i f(r - tau)) beta_tau(r),   grad f = A_Lan - A_sym,

so that each projector sees the symmetric gauge about its own atom. Norm-conserving only:
the GIPAW correction to the ultrasoft and PAW augmentation is a separate derivation, and US
and PAW are refused by name.

**What this is in this code: a new driver, not an option on the existing one.** The state is
not a sum of plane waves of the magnetic cell, so it does not live on an `npwx`-padded
G-sphere. It is a complex array on the magnetic cell's FFT grid, with a spectral filter at
`ecutwfc` in the mixed representation to keep a cutoff meaningful. The memory cost against
a G-sphere is the box-to-sphere ratio, about 2 for a box of side `2 G_wfc` and about 16 for
the density grid, and which of the two the kinetic apply needs (aliasing of the products by
`A`) is measured, not assumed. `davidson_eigensolver` needs `apply_projected`, the overlap
pieces and two diagonals; whether a flattened grid vector satisfies that protocol, with a
preconditioner built from the `pi_1` and `pi_{2,3}` diagonals of the two representations,
is question 3 for the review.

**Strong translational symmetry, and the trap in it.** The density and the current are
periodic over the *chemical* cell, not only the magnetic one (Lü, proven in their
supplement), because every magnetic translation by a chemical lattice vector commutes with
`H` and acts on a density as a plain translation. So the self-consistent loop mixes a
chemical-cell density, and only the states live on the magnetic cell. The theorem needs the
magnetic Brillouin zone mesh to be closed under the `k` shift that `T_{a1}` induces
(`k -> k + (p/q) b2`-type): a mesh that is not closed gives a density that is periodic over
the magnetic cell only, and folding it onto the chemical cell would hide the error. The
guard refuses such a mesh, and the test measures the spread of the density over the `q`
copies inside the magnetic cell (round-off when closed, and the guard tested to fire on an
open mesh).

**Validation, numbers to reach.**

- The **Landau problem**: all potentials off, flat bands at `(n + 1/2) hbar omega_c` to
  round-off, `p` states per level per `k`, and the Chern number 1 of each level through the
  existing Fukui code (P16), which Lü also report.
- **Magnetizability by finite difference**, `chi = -(E(dB) - 2 E(0) + E(-dB))/dB^2` at one
  flux quantum per supercell, against Lü's Table 2: diamond `chi_val = -11.2`, silicon
  `-0.2` (LDA, `10^-6 cm^3/mol`, valence only), and `E(dB) = E(-dB)` to round-off in a
  time-reversal invariant crystal.
- **The magnetic band shift**: a rigid translation of the atoms by `0.5 a1` changes `eps(k)`
  and leaves the DOS and the total energy unchanged to round-off. This is also the rule for
  every comparison in this file: **never compare `eps(k)` at a named `k` between two routes,
  only the sorted level set over the whole magnetic zone, the DOS, or the energy**.
- **The orbital magnetization identity**, for a cell with a moment (a Chern insulator, or a
  ferromagnet with SOC once spin is in): `-(E(+B) - E(-B))/2B -> M_orb` of P64 as `q` grows.
  `B` is quantised on a torus, so this is a finite difference over the magnetic cells, not a
  `jax.grad`, and its residue at finite `q` is part of what is measured.

**Refusals.** US and PAW, spin-orbit coupling, a k-mesh not closed under the magnetic
translation, forces and stress (Hellmann-Feynman is in reach through `jax.grad` at fixed
`B` but is a later item), and a current-dependent functional (plain LDA and GGA throughout,
as Lü and Wang; current-DFT is named as the missing term, Vignale and Rasolt).

## 5. Stage 3 (P177): route W against route P, then the gauge-including Wannier basis

**The measurement this whole plan is built for.** Routes W0 and P at the same `p/q`,
`q = 1` to the largest that stage 2 affords, on graphene and on one gapped case (hBN, or
MoS2 where the orbital moment of the `d` orbitals gives the valley splitting that Peierls
is known to miss). The difference of the sorted level sets is the error of Peierls, and it
should scale as `B` times the intra-cell orbital moment. Expected: graphene's `p_z`
Peierls is very good (the moment of a single `p_z` orbital about its own centre vanishes),
and the gapped case is not.

**W1, if the difference is not negligible.** Replace each Wannier function by its
gauge-including form `w^B_{R mu}(r) = exp(-i A(R + c_mu).r - i f(r - R - c_mu)) w_mu(r - R - c_mu)`,
the same dressing GIPAW puts on projectors and Wang puts on Gaussians. The matrix elements
then are, at each `B`, `<w^B|H(B)|w^B>` and the overlap `<w^B|w^B>`, which is no longer the
identity. Expanded in `B` about each pair's midpoint, they are the Peierls phase times
`t + B.m + B^2 d + ...`, with `m` the matrix elements of `(r - c) x p / 2` including the
nonlocal commutator (the orbital moment of the Wannier pair) and `d` the diamagnetic term.
At 10 T the expansion parameter is `(spread/l_B)^2 ~ 1e-4`, so first order is the honest
target and second order is a check on it. This **does** need the Wannier functions on a
real-space grid (or their matrix elements of `r` and `r x p` computed in the Bloch basis,
as `pw2wannier90` does for `r` through `M_mn(k,b)`), which is a new piece of P162 and the
reason W1 is not stage 1. It also closes route W against route P as the Wannier span grows
(more disentangled bands), which is the route identity: same object, two bases.

## 6. Stage 4, optional: the modulated field in the ultracell

What the ultracell can carry as built: a vector potential periodic over the ultracell,
`B(r) = B_0 cos(Q.r) z_hat`, zero net flux. Its coupling is `A_Q.v` with `v` the full
velocity including the nonlocal commutator (P64 and the optics already build it) plus
`(A^2)_Q` into P88's scalar `dV`, to second order in `Q` times the projector radius. The
physics it buys is magnetic superlattices and snake states along the zero lines of `B`. It
does **not** give Landau levels, so it is listed last and not proposed for now.

## 7. Two cheap checks that need no new solver

- **Onsager-Roth quantisation with the Berry phase and the orbital moment** (Xiao, Chang and
  Niu, Rev. Mod. Phys. 82, 1959 (2010), section on semiclassical quantisation, and Gao and
  Niu, PNAS 114, 7295 (2017) for the next order), from the zero-field constant-energy
  orbits of the DFT bands, the Berry phase of P16 and the orbital moment of P64. It gives
  the Landau levels at lab fields for large `n` and the zero mode of graphene from the
  Berry phase `pi`, with machinery that shares nothing with route W. It is the
  independent check of stage 1 at 10 T, where route P cannot go.
- **The orbital susceptibility at zero field**, McClure's `delta(E_F)` diamagnetism of
  graphene, by the KPM or the Kubo formula on route W. Perturbative, so it checks the small
  `B` limit of stages 1 and 3, and it is what a magnetometer measures.

## 8. The interface

- `Calculator.get_landau_levels(B=10.0, route="peierls", ...)`: `B` in tesla, the code
  choosing `p/q` from the cell area (and saying the flux it took, since `B` is rounded to
  a rational flux), the Wannier set from `get_wannier` taken as an argument or computed.
  Returns the levels near `E_F`, the KPM DOS, the sublattice or orbital weight of each
  level, and the flux actually used.
- `Calculator.get_magnetic_bands(flux=(p, q), ...)` for route P, returning the magnetic band
  structure on the magnetic zone, the DOS and the total energy, with a refusal for a `q`
  whose memory estimate exceeds the machine.
- The example in the guide is built with `Calculator.from_keywords` (graphene in angstrom).
- `B` is internal only as the flux number `n_phi`; no tesla to Rydberg conversion appears
  outside `io/` and `units.py`.

## 9. What a finished stage leaves behind

The five deliverables of `.claude/skills/phase-audit/deliverables.md` for each stage: the
numbers above in `PLAN.md` section 3, a README row with no QE and no Elk tick (checked), a
guide page `docs/guide/magnetic-field.md` with its Limits note, a notebook (graphene Landau
levels at 10 T against the Dirac formula, one figure), and a timing. There is no reference
binary for either route here, so the timing is against the zero-field calculation of the
same cell (route P) and against pyqula's ribbon at the same size (route W), and the
`PERFORMANCE.md` entry says so.

## 10. Questions for the review

1. Is the obstruction in section 1 stated correctly, and does it rule out the plain
   ultracell route for a uniform field?
2. Is Peierls on P162's `H_mn(R)` expected to give graphene's `n <= 10` levels at 10 T
   within the Dirac formula's own accuracy, and how large is the intra-cell term for
   graphene's `p_z` against a gapped `d`-orbital case?
3. Does `davidson_eigensolver`'s `Hamiltonian` protocol survive a mixed-representation
   grid vector, or is stage 2 a new solver?
4. Which `q` is affordable for stage 2 on this CPU and on D22 at graphene's `ecutwfc`, given
   that `nbnd` grows as `q`?
5. Is the finite-difference orbital magnetization identity of stage 2 exact in the limit, and
   what residue does a finite `q` leave?
6. Torus with magnetic translations (the default here, needed for the stage 3 comparison)
   against the ribbon of the 2016 paper: is the torus the right default?

## 11. Addendum (2026-11-07, after the user's question): the magnetic ultracell

The user asked whether the ultracell can be generalised to a uniform field, since route P
pays for a supercell. It can, provided the twist of section 1 goes into the basis, and the
construction already exists in the model literature: Wang and Vafek, *Revisiting Bloch
electrons in magnetic field: Hofstadter physics via hybrid Wannier states*, Phys. Rev. B
108, 245109 (2023), arXiv:2303.16347 (and PRB 106, L121111 (2022)).

**The physics.** In the Landau gauge `A = B x y_hat` the momentum along `a2` is conserved
with the *chemical* period, and the field shifts it linearly with position along `a1`: a
state near cell `s` along `a1` looks locally like a zero-field state at `k2 + s p/q`. So the
basis is the zero-field **hybrid Wannier state** (localised along `a1`, Bloch along `a2`)
taken at the shifted `k2`:

    |W_{alpha r}(k)> ~ sum_s exp(i 2 pi k1 s) t(a1)^s |w_alpha(0, k2 + r/q)>,
    t(a1) = exp(-i q_phi.r) T(a1),   r = 0..q-1,   k2 in [0, 1/q),

with `t` the magnetic translation (Wang and Vafek eq. 32 region). The `q` wavevectors
`k2 + r/q` are exactly P88's folded set of a `1 x q` ultracell along `a2`, and the
zero-field states at them are computed once and frozen, as in P88. What changes is that
they are combined by the *magnetic* translation along `a1` instead of an ordinary one, so
the twist lives in the basis and the obstruction of section 1 does not apply.

**What it costs.** Dimension `q nbnd` per magnetic `k`, as P88, but the matrix is
**banded** in `s` (hybrid states couple only over their spread along `a1`), so the apply is
`O(q nbnd^2 w)` with `w` the band width in cells, and KPM or a filtered Lanczos replace
P88's dense `(N nbnd)^3`. Graphene at 10 T: `q ~ 7900`, `nbnd ~ 8`, dimension 6.3e4,
banded. The cost moves into the zero-field states: `q` values of `k2` times `N1 ~ 10`
values of `k1` for the hybrid localisation, about 8e4 NSCF k-points at 10 T, or a coarse
mesh with Wannier interpolation along `k2`, which is where route W re-enters.

**Its two limits are the two other routes.** For `l_B` much longer than the hybrid
spread it reduces to the Harper (Peierls) equation (Wang and Vafek show this), so stage 1
is its weak-field limit; and as `nbnd` grows it is a Rayleigh-Ritz truncation of route P's
exact problem. For a frozen potential the eigenvalues are then upper bounds that fall
monotonically with `nbnd`, which is a convergence test that needs no route P. The free
Landau problem (`V = 0`, zero-field states are plane waves) is the exact check:
`(n + 1/2) hbar omega_c`.

**What has to be measured or added, and the traps already visible.**
- Wang and Vafek's `H_B = x p_y B + x^2 B^2/2` is evaluated with local potentials; a
  norm-conserving dataset needs the GIPAW phase on each projector here too.
- On a triangular Bravais lattice (graphene's) their overlap `Lambda` deviates from the
  identity and is fixed by `U D^{-1/2}`: a canonical orthogonalisation with a measured
  rank, which is also where a Chern number shows up (rank deficiency, Streda).
- Chern bands need a different reference centre `n0`; a set of bands with a nonzero
  Chern number is refused until that is worked out, and graphene's composite of all
  occupied bands plus empty ones up to a gap is trivial.
- Their spectra agree quantitatively up to flux 0.2 per cell and then shift upwards at
  their band count: convergence in `nbnd` is the physics check here, as in P88.
- The hybrid states need a smooth gauge along `k1` at every `k2`: P16's Wilson loops along
  one direction give exactly that (1D maximal localisation is the parallel transport).
- Self-consistency stays simple: the density is periodic over the chemical cell (Lü's
  strong translational symmetry), so an update is a unit-cell `dV`, which is what P88's
  `dV_{Q=0}` already handles.

**Effect on the stage order (proposed, for the review).** Stage 1 Peierls stays (cheap,
and the weak-field limit). The magnetic ultracell becomes stage 2 and the main route, at
high field first with direct states (`q <= 100`, seconds of NSCF on a two-atom cell) and
then at 10 T with interpolation. Route P drops to an optional reference, built only if the
`nbnd` convergence of the magnetic ultracell is not conclusive.

## 12. The review (fable, 2026-11-07), and what it changes. This section wins over 1 to 11.

The scripts are in `plans/orbital-field-review/` (`peierls_torus.py`, `variants.py`,
`labfield.py`, `fd_morb.py`, `fd_run.py`), copied from the review's scratchpad.

1. **Section 3's Hamiltonian is wrong as written, and the plaquette guard cannot see it.** A
   straight-line Peierls phase in the Landau gauge with Wang's `exp(+i A(c_nu').P)` is not a
   magnetic Bloch Hamiltonian: NN honeycomb, `q = 400`, `p = 1`, `max |H - H^dagger| = 4.7 eV`,
   levels move with `k` by 0.63 eV and sit at 1.40 times Dirac. Wang's eq. 4, 8, 9 hold for
   GIPAW-dressed orbitals; a straight-line Peierls orbital is a GIPAW one times
   `exp(i A(c).c)`, a per-orbital constant that changes the Bloch-sum factor. Two consistent
   pairs, both Hermitian to 1e-14, flat in `k` to 1e-15, the same spectrum:
   (i) straight-line phase `exp(-i A(mid).(r_mu - r_nu))` with `exp(-i A(P).c_nu')`
   `= exp(-i 2 pi p m1 (b2.c_nu'))`, nonzero on the `a1` wraps and one on the `a2` wraps;
   (ii) the London/GIPAW midpoint phase `[A(r_mu).m + f(m - r_mu)] - [A(r_nu).m + f(m - r_nu)]`,
   `f(d) = A(d).d/2`, with Wang's `exp(+i A(c_nu').P)`.
   The index is the column `nu`; Wang's eq. 9 as printed (`mu`) is a typo (`|H - H^dagger|`
   = 1.1 eV at `q = 10` in that form). Without the factor the matrix is **not** Hermitian (4.7
   straight line, 1.4e-2 GIPAW). **The guard is Hermiticity at a `k` where both wraps are
   active**, plus the plaquette product with its orientation stated: counterclockwise,
   `exp(-i int A.dl)` per hop, `B` along `+z` gives `exp(-i 2 pi p/q)`.
2. **Peierls at 10 T, with the fixed formula** (`q = 7900`, dimension 15800, 2 s shift-invert):
   `E_1 = 0.1002 eV` against Dirac 0.10021 with `v_F = 3|t| a_cc / 2 hbar = 8.74e5 m/s`
   (`t = 2.7 eV`). `E_n / Dirac = 1 - 1.1e-4 n` at 10 T, `1 - 2.3e-4 n` at 20 T, `1 - 4.6e-4 n`
   at 40 T. Two zero modes per `k` with `tr P_A = 1.0` over the pair (rule D4: report the
   trace over the degenerate pair, never one vector). `p` states per valley per level per `k`
   confirmed at `p = 2`. The intra-cell term for graphene: `mu_B x 10 T = 0.58 meV`, 0.6 percent
   of `E_1`; for a heavy TMD band it is comparable to the spacing (not computed).
3. **The obstruction: conclusion right, mechanism misstated.** Periodic plane waves of the
   magnetic cell are complete in L2; what fails is the operator. A linear `A` continued
   periodically is a sawtooth whose curl is the uniform `B` plus a flux sheet `-p Phi0` at the
   cell boundary, so "P88 plus `A.p`" diagonalises a zero-net-flux field. Section 11's
   magnetic ultracell is the right generalisation: basis function at magnetic `k2`
   `exp(-i A(X_j).r) h_{n,X_j}(k2 + 2 pi p j/q)`, `q nbnd` per magnetic `k`, complete as
   `nbnd -> npw`, Rayleigh-Ritz with `nbnd` the knob; it needs `<u_k|x|u_k'>` (P16) and the
   velocity (P64), no real-space Wannier functions.
4. **Davidson for route P: one hook, not a new solver.** The solver is duck-typed
   (`davidson.py:622-671, 805, 840, 1142, 1214-1215, 1330`; the spinor operator is the precedent,
   `operator.py:264-275`). A flat grid vector with an all-true mask passes. The preconditioner
   does not: `_precondition` (`davidson.py:511-526`, called at 845) divides by a diagonal in the
   state's basis, a constant on a grid, and `starting_vectors` (1061-1090) damps by it too.
   Fix: a `precondition(residual, energies, ik)` method on the operator, default the present
   rule, and for the magnetic operator the zero-field kinetic preconditioner through FFTs of
   `h` (`xi1`) and `u` (`xi2`, `xi3`), ignoring the twist. "A spectral filter at `ecutwfc`" has
   no definition and is dropped: the bare grid is the cutoff, as in Lü.
5. **Route P sizes.** Graphene `a = 2.46 A`, `c = 15 A`, 50 Ry: wavefunction box 12x12x64 per
   cell (box/sphere 2.9). With `nbnd = 1.2 x 4q`, ndim `9216 q`, `(2.18 nvecx + 4.2 nbnd) ndim`
   complex128: `q = 30` 5.5 GB (`david` 2), `q = 40` 9.7, `q = 60` 21.8. D22's 31 GB reaches
   `q ~ 60` (1300 T) at `david = 2`; the A2000 `q ~ 40`. If the kinetic apply needs the density
   box, divide `q` by about 3. Non-self-consistent on the zero-field density with a filtered
   block of about 20 levels makes memory linear in `q`, so `q` of a few hundred.
6. **The orbital magnetization identity.** Exact in the limit for an insulator (a metal has de
   Haas-van Alphen terms). Haldane model: trivial insulator `FD/M -> 1` as `O(1/q^2)`; Chern
   insulator as `O(1/q)`, which is `dM/dmu = C/2 pi` times the `O(B)` motion of the band edges.
   The comparison hands P64 a `mu`, and the test expects `1/q` for a Chern insulator.
7. **Torus and ribbon.** The torus is the default for validation and for the route P
   comparison; for the user's question a ribbon is cheaper (`5 l_B` = 165 cells at 10 T against
   15800 on the torus). Both from one hopping builder.
8. **Strong translational symmetry, used rather than guarded.** `T_{a1}` sends
   `k2 -> k2 - 2 pi p/q` (Lü SM eq. S2; spectra equal to 1e-14 at `k2 +- p/q`); closure is
   `N2 = 0 mod q`. The density at `k2 - 2 pi p r/q` is the density at `k2` translated by `r a1`,
   so solve `N2/q` points and fold each onto the chemical cell: cost divided by `q`, closure by
   construction; the test is closed mesh against the fold, to round-off.
9. **Lü's Table 2 depends on the supercell** (diamond -11.16 at 3x, -10.96 at 4x): the target is
   their 3x1x1 conventional cell, 2x6x6 mesh, 30 and 20 Ha, at 2 percent. Each Wigner-Seitz image
   `R + T` (`interpolation.py:111-127`) is its own bond with its own phase.
10. **Order.** Peierls first. For graphene at lab fields the magnetic ultracell is the main
    route, checked against route P at 1300 to 2600 T and by Onsager-Roth at 10 T; route P is
    built for that comparison, Hofstadter and the gapped `d` case, with the preconditioner hook
    of item 4 and the fold of item 8.

## 13. The basis of the magnetic ultracell (fable, 2026-11-07, after P176's refusal). This section wins over 11 and 12 on route U.

The full decision, with every table and the probes, is `plans/orbital-field/route-u-basis/decision.md`;
the D22 logs are under `/l/ladovj1/review/mag/fable-basis/`.

- **The basis.** The hybrid state of trial $j$ is the projected trial itself,
  $h_j = \sum_m |\psi_m\rangle\langle\psi_m|g_j\rangle$ over the lowest $N$ bands at each k, periodized over
  $N_1$ cells, with no Loewdin step and no transport gauge. The overlap $S$ carries the non-orthogonality.
  Adding a trial adds one function, so the spans are nested in $J$ and the levels fall monotonically.
  SMV disentanglement is refused because its subspace nests in nothing; cutting at a gap exists on one case.
- **Why the old basis cannot work.** $\langle x^2\rangle$ of a single sinc diverges, so the Landau-gauge
  matrix elements of hybrid states of an entangled set do not exist. The windowed free-problem experiment
  reads 0.007 to 0.065 Ry off: the window cures the sawtooth, not the span.
- **Measured with the projected basis** (free problem at 1/10, c = 5 A, $N = 64$, a Hermite-Gaussian trial
  of $\sigma = 0.5$ cells): the $n = 0$ error is $3.2\times10^{-2}$, $4.4\times10^{-3}$, $1.8\times10^{-7}\,B$
  at $J = 1, 2, 3$, every level above exact, Hermiticity $3\times10^{-12}$ or better. The reversed field fires
  (Hermiticity 0.6).
- **The guard.** The projectability $1 - \sum_m |\langle\psi_m|g_j\rangle|^2/|g_j|^2$ at every k: $10^{-6}$
  passes, $2.7\times10^{-3}$ gives Hermiticity $2.5\times10^{-4}$, 0.12 ruins the spectrum. Refuse below
  $1 - 10^{-4}$.
- **Acceptance.** Graphene against route P at q = 6 and 10, trials from the HGH dataset's PP_CHI, nested
  $J$ = 2 (p_z), 8, 18, $N = 32$: Hermiticity below $10^{-10}$, U above P at every level, the gap after the
  zero-field span error is subtracted falling with $J$ and below 5 mRy at $J = 18$.
- **Why continue.** At $q \le 10$ route P is exact and route U is not cheaper. Route U is the only route for
  $60 < q < 8000$ with the intra-cell terms Peierls drops, and those are the likely source of the 0.2 eV
  between route P and Peierls. The projected blocks are smooth in $\kappa'$, so they can be interpolated
  to any q, 7900 included; its identity test is at q = 50.
- **Size.** About 230 lines in `hybrid.py` and `landau_ultracell.py` plus four fast tests, one to two days.
  The interpolation and the q = 50 identity are a second step of the same size.
- **Measured (step 3, `plans/orbital-field/u3.md`, 2026-11-08).** Built as above. The gate reads fable's
  numbers through the workflow ($3.24\times10^{-2}$, $4.35\times10^{-3}$, $1.8\times10^{-7}\,B$). Graphene at
  c = 15 A is refused: no atomic trial is inside any solvable window ($1 - p$ of $p_z$ is $4.1\times10^{-3}$ at
  $N = 32$ and $6.1\times10^{-4}$ at $N = 96$, the vacuum ladder again), and with the guard off the line reads
  36.5, 36.5, 14.3 mRy at $J$ = 2, 8, 18, worse at $N = 48$. A projected atomic orbital keeps its weight on
  every band of the window, so its span is 0.2 to 0.7 eV off near $E_D$ at zero field, where the Wannier
  $p_z$ set is within 11 meV. Trials close to the bands (the Wannier sets themselves) are the next try.

## 14. Left from the audit of 2026-11-15 (fable's decision on the -1 flux run)

The `-1` flux SCF of `run_magnetizability` stays on by default (`time_reversal_check=True`).
It is the only place where the fold energy is checked on the caller's own cell. The default
flips to `False` when two tests exist: one that asserts `|E(+) - E(-)|` on a small cell, and
one that asserts the fold total energy against an independent number. The first exists since
2026-11-15 (`tests/unit/test_magnetic_bloch.py::test_the_fold_scf_keeps_time_reversal_on_an_odd_box`,
slow, 8.9 s); the second does not. When the default flips, the guide's
`print(f"{chi.asymmetry:.3e}")` breaks on `None` and moves with it.

What the control sees: only a defect that breaks `H(-A) = H(A)^*`. A defect even in the flux
(a wrong `1/q`, the sign of `deband`, a global sign of `A`) cannot show there. On a box even
along an axis the Nyquist slot breaks the relation as well, measured on two-atom silicon at
12 Ry and `conv_thr = 1e-10`: 2.1e-5 Ry on the default 8 x 8 x 8 box, 6.1e-6 on 10 x 10 x 10,
1.6e-12 on 9 x 9 x 9 (P177). A cheaper form of the control, not costed: one band pass at flux
`-1` on the converged `+1` potential, comparing `eband`, tests the same relation without the
SCF.
