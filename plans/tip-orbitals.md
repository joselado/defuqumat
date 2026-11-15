# The orbital of the tip in the Landauer transport map (plan, 2026-11-15)

Status: implemented on branch `tip-orbitals` (2026-11-15), as `PLAN.md` P179 records; this file is the plan as it was before the code, and P179 says where the code departs from it (the vacuum continuation, `match_height`, is not in this plan). Reviewed once by a fable subagent against the code
(2026-11-15, one python under a 5 GB cap, no SCF and no pytest); its findings are folded in
below and marked "(review)". The review, its scripts and the later kappa scan are in
`plans/tip-orbitals-review/`. **No number here comes from an SCF**: they come from random
coefficients, a model honeycomb of s wells, and the pseudo-orbitals of two UPF files. The
graphene contrast inversion of section 7 and the Pb statement of section 12 are predictions.
It would be phase P179.

## 1. What we have now, and what the point tip already is

`run_vertical_transport` (`defuqumat/workflows/transport.py`, P66) computes

    T(r; E) = sum_k w_k  a_k(r)^T S_k a_k(r)^*,      a_kn(r) = psi_kn(r) sqrt(delta(E - e_kn)/eta)

with `S_k` the Gram matrix of the bands on the exit plane (`transport/substrate.py:exit_overlap`)
and the contraction in `transport/green.py:transmission` (or `spin_transmission` for a magnetic
tip, a 2x2 acceptance `P_t` in spin space). The tip enters only through `psi_kn(r)`, sampled at
the tip point by `basis/sample.py:sample_wavefunctions`.

The question of the user (2026-11-15): replace the point by an orbital of the tip apex, an s, a
pz, or a pair px, py, with an orbital size as an input and preset values for typical tips.

**The present point tip is exactly the s tip.** In Bardeen's formalism a tip state with the
vacuum tail `C k_0(kappa |r - r0|) Y_00` couples to a sample state that obeys
`(nabla^2 - kappa^2) psi = 0` near the tip as `M ~ psi(r0)`, which is Tersoff and Hamann. So the
existing map is the `s` channel of what follows.

## 2. The physics: Chen's derivative rule, written as Landauer channels

Chen (PRB 42, 8841 (1990)) expands the tip state around the apex `r0` in vacuum solutions,
`chi(r) = sum_b C_b k_l(kappa rho) Y_b(theta, phi)`, with real spherical harmonics. The Bardeen
element is then a derivative of the sample state at the apex,

    M_n ~ sum_b C_b^* D_b psi_n(r0),

with the operators of Mandi and Palotas (PRB 91, 165406 (2015), arXiv:1503.09189, Table I,
opened; journal reference confirmed on arXiv): `D_s = 1`, `D_pa = (1/kappa) d/dx_a`, `D_dxy = (1/kappa^2) d^2/dx dy` and its partners,
`D_dx2-y2 = (1/kappa^2)(d^2/dx^2 - d^2/dy^2)`. For `dz2` the table writes
`(3/kappa^2) d^2/dz^2 - 1`. We use `(3 d^2/dz^2 - nabla^2)/kappa^2` instead, which is the same in
an exact vacuum, where `nabla^2 psi = kappa^2 psi`, and is a pure l = 2 operator with one power
of `kappa` (review). The common prefactor `4 pi^2 hbar^4/(kappa^2 m^2)` is dropped; the map is in
arbitrary units already.

`C_b^*` is right: `a_n = <chi|psi_n>` is what `a^T S a^*` takes (`green.py:338`, review). It
matters only for a complex combination such as `px + i py`.

**When the rule is exact.** The sample state must obey `(nabla^2 - kappa^2) psi = 0` around the
apex. In the vacuum above a slab every in-plane component decays as
`exp(-sqrt(kappa^2 + |k+G|_par^2) z)`, so the whole state obeys it with `kappa^2 = V_vac - E`
(Ry, `hbar^2/2m = 1`). It holds where the potential is flat. How flat it is at the tip is a
diagnostic (section 5).

**In Landauer language the tip is a set of incoherent channels.** `Gamma_t = sum_c |chi_c><chi_c|`.
One channel is one tip state, a coherent combination of orbitals, which interfere inside it
(Mandi and Palotas eq. 7, the s-pz cross term). Two channels are two independent tip states, which
add as `|M_1|^2 + |M_2|^2`; this is how PP-STM treats every tip orbital (Krejci, Hapala, Ondracek and
Jelinek, PRB 95, 045407 (2017), arXiv:1609.09462, opened, reference confirmed: "each of which serve as an independent tunneling
channel", weights fitted, kappa from the work function of the sample) and how Gross et al.
(PRL 107, 086101 (2011)) combine s and p for CO. Then

    T(r; E) = sum_c T_c(r; E),   T_c = the present quadratic form with a_kn -> sum_b C_cb^* D_b a_kn.

Each `T_c` is a Gram form, so it is non-negative. A magnetic tip whose spin and orbit factor is
`Gamma_t = Gamma_orb (x) P_t`, one `spin_transmission` per channel with the same `P_t`. A tip
whose states are spin-orbit coupled (a Pb apex, section 12) does not factor; its channels are
spinor orbitals, `a_c = sum_{b,s} C_cbs^* D_b psi_s`, and they are a later extension, refused by
name in this phase.

**What this does not model.** The energy dependence of the tip's own states beyond a fixed set of
weights (Abilio and Palotas, PRB 111, 245425 (2025), arXiv:2504.11303, abstract read), multiple
scattering between tip and sample, and the relaxation of a flexible apex (PP-STM).

## 3. How it goes into the plane-wave code

**`D_b` is diagonal in G**: `d/dx_a -> i (k+G)_a` in cartesian components of the tip frame, from
the Miller indices and `kcrystal` through the reciprocal basis. A channel is a multiplier on the
coefficients, `F_c(k+G) = sum_b C_cb^* f_b(k+G)`. Checked (review, ran, random coefficients,
`npol` 1 and 2, padded mask, hexagonal cell and a sheared `a3`, `k3 = 0` and 0.25, through
`sample_wavefunctions` alone and through the `_tip_sphere` fold): the multiplier against a
central difference at `h = 1e-3` agrees to 0.9e-6 to 1.2e-6 for the three first derivatives,
slope 2.00 in `h`, and to 5e-7 to 6e-7 for `d^2/dz^2`; folded against unfolded, 6e-15 to 9e-15.

**Where it goes.** The multiplier acts on the window before `_tip_sphere` (`transport.py:834`).
Strictly only a z derivative needs that, together with a tilted `tip_frame` or a fold along
another axis: `b3` is normal to `a1` and `a2`, so `(k+G)_x` and `(k+G)_y` do not depend on the
folded index even in a sheared cell (review, 3e-16). We keep "before" for all of them.

**Sample the shell, contract per point** (review). Rather than one multiplied window per channel,
we sample the basis amplitudes of every shell the tip uses (1 for s, 3 for p, 5 for d, both
spinor components by the same factor) once per k-point and window, and form each channel at each
tip point as `a_c(r) = sum_b C_cb(r, E)^* A_b(r)`. This costs nothing more for a mixed-l channel
(the `kappa(E)^-l` goes into `C_cb(E)`), and it gives each tip point its own coefficients, which
the wedge needs (section 9). The channel loop sits inside the k loop, after `S_k` is built, so
`S_k`, `_diagnose`, `_band_window`, `transmission`, `spin_transmission` and `channel_basis` are
unchanged. The amplitude block holds `n_shell x npol x n_window x pixels`, and `_AMPLITUDE_BLOCK`
bounds it by taking fewer pixels.

**Names.** `incoherent` keeps its meaning, every band on its own, and is computed per tip channel
and summed. The channels themselves are always incoherent by definition; the per-channel maps are
`flow.tip_channels`, `(n_channels, nE, n1, n2)`, only when asked for.

## 4. Frame and sign

- **z** is the unit normal of the exit plane, oriented from the material towards the tip, fixed
  **once** from the caller's geometry (review: if each wedge image derives it again, the mirror
  flips pz twice). A non-s tip with `tip=` points on both sides of the material, or with
  `axis != exit_axis`, is refused by name.
- **x** is the first in-plane lattice vector made orthogonal to z; **y** is `z x x`.
  `tip_frame=` (3x3, rows x, y, z, cartesian) overrides it, which is a tilted tip.
- **Sign** (review: argued `k_1(kappa rho) cos theta = (1/kappa) d/dz0 k_0`, so
  `M_pz = +(1/kappa) d psi/dz0`; ran: `pz/s` in [-1.14, -0.90] at 3 to 5 bohr, and
  `T(s - pz)/T(s + pz) >= 220` at every pixel). With z out of the sample, `s - pz` points its lobe
  at the sample and gives about `2 psi`; `s + pz` points it away and nearly cancels.

## 5. Kappa

Default (review): the **plateau** of the planar average of the local potential in the vacuum on
the tip side, `kappa(E)^2 = V_plateau - E`, one value per entry of `energies` and per point of a
`bias` window. The potential is `SCFResult.potential`, `vltot + v_scf` (`scf/driver.py:10739`);
at `nspin = 2` it is `(up, down)` with `vltot` in both (`scf/potential.py:772`), so we take the
spin average there, and component 0 at `nspin = 1` and 4. The plateau and not the value at the
tip plane, for two reasons: then `helmholtz_residue` measures the lateral corrugation and the
approach of the potential together, and a height sweep of a p channel does not pick up a
`kappa(h)^-2l` of its own. The LDA exchange tail alone is about 1.2 eV at `n = 1e-4` and 0.3 eV
at `1e-6` (review, an estimate), so 15 and 3 per cent in `kappa`.

No plateau (a sawtooth field, a thin vacuum) or `V_plateau - E <= 0`: refused by name, and
`tip_kappa=` (1/bohr, a number or one per energy) is required. `tip_kappa` also lets a user match
the literature's `kappa = 1/angstrom`.

Diagnostics in `notes`: `tip_kappa` per energy; `tip_potential_offset`, the planar average at the
tip plane against the plateau; `helmholtz_residue`, `|nabla^2 a - kappa^2 a|/|kappa^2 a|` at the
tip points weighted by the map. **The residue also measures the cutoff in the tail**, and the
docstring says so: with exact eigenvectors of a model sheet it read 4.0e-2, 5.4e-2, 9.9e-2 at 3,
4, 5 bohr at 30 Ry, where the potential alone gives 2.7e-2, 8.3e-4, 9.2e-6, and 2.75e-2, 6.9e-4,
5.6e-4 at 60 Ry (review, ran).

## 6. Checks, each one able to fail

Fast unit tests unless marked slow. Synthetic coefficients are enough only where the identity
holds for any states; V4 and V5 need orthonormal or symmetric sets (review).

- **V1. The s tip is the old map.** `==` on a branch that skips the new code cannot fail, so two
  parts: s through the `F = 1` path against the old branch (round-off), and an A/B of the whole
  map against the commit before the change, run on the same cores (`CLAUDE.local.md`, the two
  kinds of core).
- **V2. Factor, sign and units of each `D_b`**, the review's finite-difference check, on the
  hexagonal and the sheared cell, through both paths (unfolded and folded). Can fail (ran).
- **V3. The fold, not the factor.** The map of one channel integrated over the tip plane equals
  `sum_k W(k)` of `momentum_weights` with the tip Gram built from the multiplied coefficients. It
  shares the multiplier, so it tests the fold and the pixel blocks; V2 tests the factor.
- **V4. An independent route in the volume limit.** `exit_region="volume"` (where `S_k` is the
  identity only for orthonormal rows) with the three p channels gives `kappa^-2 tau_E(r)`, where
  `tau_E` is `scf/density.py:kinetic_energy_density` called with the weights
  `w_k delta(E - e_kn)/eta` in place of the occupations. Its body is read: `tau = sum_i w_i
  |grad psi_i|^2` in Ry, no factor of one half, the gradient built as `i(k+G) c_G`
  (`band_kinetic_density`), so the factor is fixed before the test: exactly `kappa^-2`, with the
  `delta/eta` normalisation of the s map against `run_stm` (P66). The tip points are FFT grid
  points, so no interpolation enters. Orthonormalised synthetic states.
- **V5. Frame.** On a hexagonal cell with a C3-symmetric set of states: the `px` map turned by
  120 degrees is the `px` map of a frame turned by 120 degrees, and the incoherent `pxy` map has
  the symmetry of the lattice. (The rotation of `tip_frame` about z alone would pass with crystal
  components used as cartesian, so it is not the check; V2 on the hexagonal cell is.)
- **V6. Sign of a hybrid** (slow, SCF of the hydrogen sheet): `T(s - pz) > T(s + pz)` at every
  pixel, the ratio recorded, and the same in the mirrored geometry (the tip below, the exit plane
  above).
- **V7. Conjugation of a complex `C`** (fast, the review's model of s wells on a honeycomb, which
  has graphene's pi pair at K by symmetry). The hollow site does not pin it: there the Dirac pair
  carries both windings and `px + i py`, `px - i py` both give 9.968e-08. On an atom, k-set `{K}`:
  site A, `C = (px: 1, py: i)` gives 1.110e-07 and its conjugate 1.1e-34; the winding of the pair,
  measured by sampling `psi` on a circle (no derivative), is 3.164e-06 for `m = +1` and 3.4e-33
  for `m = -1`; site B is reversed. All invariant under a rotation inside the pair (rule D4). At
  the hollow `s` and `pz` read 1.4e-36 and 1.5e-35.
- **V8. Kappa** (slow): on the hydrogen sheet, the decay constant that
  `transport/momentum.py:decay_constants` fits to the `k_par = 0` state at its own energy (not at
  `E_F`, where graphene has no `k_par = 0` state) against `sqrt(V_plateau - e)`. The bound is set
  **before** the run from the xc-tail estimate of section 5, so the check can fail.
- **V9. Wedge** (slow, step 4): a reduced k-set gives the whole-grid map for `px` alone, for
  `s - pz`, and for a complex `C`, to 1e-6, the bound the existing wedge test delivers at
  `conv_thr = 1e-10` (3.3e-9 measured, `tests/regression/test_transport.py:62-64`), on a cell
  with a 3-fold axis so that `g` and `g^-1` read differently.
- **V10. The size statement** (section 12): the mean ratio and the spread over pixels against the
  point map, not a correlation (review: a correlation reads 0.9992 where the weight is off by 5
  per cent).
- **V11. The tail of each shell against the cutoff.** The derivative amplifies the noise of the
  coefficients at large G, at most `(sqrt(ecutwfc)/kappa)^l` (review: noise of norm 1e-6 at 5 bohr
  gives noise over `|psi|` of 3.3e-5 for s, 6.4e-5 px, 7.9e-5 pz, 8.4e-4 dz2). A note
  `tail_noise` per shell from `conv_thr` and `ecutwfc`.

A dark pixel is a number, not a zero: the hollow reads 1.4e-36, so the "identically zero" warning
(`transport.py:906`) does not fire, and the guide says that a map can be dark by symmetry.

## 7. The physics result for the guide and the notebook

Monolayer graphene at `E_F`, 6x6x1 (it contains K). With an `s` or a `pz` tip the hollow is dark,
because the K states vanish there by C3; with `pxy` it is bright, because their lateral gradient
does not (the review's model: 1.4e-36 and 1.5e-35 against 9.968e-08 per chiral component). So the
contrast of the map inverts between a metal tip and a CO-like tip, to be measured on graphene
itself, with the k-points next to K inside the broadening. On the AB bilayer of P66 we measure
whether the A/B contrast of the transport map (A bright, B at 0.029) survives a `pxy` tip. In
momentum space (step 3) a `pxy` tip weights each state by `|k+G|_par^2/kappa^2`, so it does not
see `k_par = 0`; on the hydrogen sheet we record the share of the current near Gamma and near X
for `s` and `pxy`.

## 8. The keywords

On `run_vertical_transport`, `run_momentum_transport` and their `Calculator.get_*` (by name):

- `tip_orbital="s"` (default), `"pz"`, `"px"`, `"py"`, `"pxy"`, later `"dz2"`, `"dxz"`, `"dyz"`,
  `"dxy"`, `"dx2-y2"`, and the presets of section 12 by name.
- `tip_orbital={"s": 1.0, "pz": -0.5}`: one channel, the complex `C_b` in Chen's normalisation.
- `tip_orbital=["px", "py"]` or `[{"s": 1, "pz": -0.5}, "px", "py"]`: channels added
  incoherently.
- `tip_kappa=None`, `tip_frame=None`.
- The result carries the channels as resolved and the notes of sections 5 and 6.

**Decided by the user (2026-11-15): `"pxy"` is two incoherent channels, `px` and `py`** ("pxy
means two incoherent channel, like px and py"). It does not depend on the frame, needs no rotation
on a wedge, and on a time-reversal-symmetric run equals the chiral pair. The coherent diagonal
lobe stays reachable as `{"px": 1, "py": 1}`.

## 9. What else reads the tip, carried or refused by name

- `run_momentum_transport`: carried in step 3, the tip Gram per channel from the multiplied
  coefficients (`momentum_weights` itself not yet read).
- `run_ultracell_transport` (`workflows/ultracell.py:367` passes an explicit argument list, and
  its bundle has no potential): carried with `tip_kappa` required; the default plateau has no
  source there.
- The QPI `landauer` reader (`workflows/qpi.py:241`), the STS spectrum `sample_spectrum`
  (`workflows/stm.py:444`) and `run_ultracell_sts`: refused by name for a non-s tip in this phase.
- A spin spiral: refused by name for a non-s tip. The two spinor components live at `k +- q/2`,
  so one factor for both is wrong there.
- A spin-orbit tip (spinor channels): refused by name (section 2).
- `get_stm`: not changed. The Chen STM image is the `exit_region="volume"` limit of the map.
- A symmetry-reduced k-set: refused by name for a non-s tip until step 4 (only the SCF's own
  reduced set is affected, `grid=` builds the whole grid). Then carried: per operation the p part
  of `C` turns as a polar vector, the l = 2 part as a symmetric traceless `Q_ij -> R Q R^T`
  (review: Table I is not equally normalised, a 45 degree turn maps `dxy` to half of `dx2-y2` and
  `dz2` carries `2 sqrt(3)`), the mirror through the slab flips pz, and **the time-reversed
  operations, which `transport.py:1511` drops, need `C -> C^*`**: `T_-k(C) = T_k(C^*)` to 5e-14,
  and it differs from `T_k(C)` by 0.22 for `px + i py` and 1.9 for `s + 0.5 i px` (review, ran).
  With per-point coefficients (section 3) all images still go through one `_assemble` per exit
  height.

## 10. Cost and memory

The band solve and `S_k` are unchanged. Sampling costs one pass per basis amplitude of the shells
used: `s` 1, `pz`/`px`/`py` 1 (only the one amplitude), `pxy` 2, a hybrid s-pz 2, a full p shell 3,
d 5, times the number of images on a wedge (as now for the s tip). The peak does not grow by
more than one multiplied copy of the window, `(n_window, npol*npwx)` complex, held at a time
while each basis amplitude of the shell is sampled (up to 5 in turn for d); the sampled block
`(n_shell, npol, n_window, pixels)` is bounded by `_AMPLITUDE_BLOCK`, and the pixel block
shrinks. Timing pair: neither
`pw.x` nor Elk computes a transport map (P66), and `pp.x`'s STM is Tersoff-Hamann (to confirm with
the `reference-source` agent before the README row), so the pair is `s` against `pz` and `pxy` on
the 3x3 graphene supercell of the transport-window measurement (2.76 s, 123 MB for `s`), in
`PERFORMANCE.md`.

## 11. Steps

0. Before the figure height is chosen: on graphene itself, measure `helmholtz_residue` and the
   tail noise of V11 over heights 2 to 6 angstrom at the cutoff of the guide input, and take the
   range where both are small. The guide and the notebook put the tip inside it. (The review's
   model gives 4e-2 at 3 bohr and 30 Ry, so the guide's present 3 bohr may be outside.)
1. `defuqumat/transport/tip.py`: channels, frame, shell amplitudes, `kappa(E)`. Wired into
   `_assemble` for channels of one `l`; mixed `l`, wedge, spiral, spinor channels, QPI and STS
   refused by name. V1, V2, V5, V7, V11.
2. Mixed `l`, the plateau and the kappa diagnostics. V4, and the slow V6 and V8.
3. `run_momentum_transport` with orbital tips. V3, the Gamma/X shares.
4. The wedge with turned and conjugated coefficients. V9.
5. d orbitals (`dz2`, the usual tungsten tip): the traceless operators, `Q_ij`, V2 and V9 for them.
6. Sizes and presets (section 12), V10, the PDOS runs on Triton (the `sbatch` proposed first).
7. Deliverables (`.claude/skills/phase-audit/deliverables.md`): the P179 record with V1 to V11;
   the README row; a section "The orbital of the tip" in `docs/guide/stm.md`, snippet built with
   `Calculator.from_keywords` and executed, the graphene figure of section 7, its refusals in the
   Limits note; `notebooks/41_vertical_transport` extended; the timing in `PERFORMANCE.md`.

Runs on this workstation stay under 5 GB and one python at a time. The slow files
(`tests/regression/test_transport.py` and the new V6, V8, V9, V10) are for the user to run, listed
in the record.

## 12. The size of the orbital, and preset tips

**What the size does in the vacuum.** For a function with `(nabla^2 - kappa^2) psi = 0` around
`r0`, the projection on `Y_b` over a sphere of radius `rho` is `a_b i_l(kappa rho)` with `a_b`
proportional to `D_b psi(r0)`; for `l = 0` this is the mean-value property, the spherical mean is
`psi(r0) sinh(kappa rho)/(kappa rho)`. So a tip orbital of any size, centred at `r0` and inside
the flat vacuum, couples as `K_b D_b psi(r0)`: the size does not change the map of one orbital,
it sets the weight `K_b` of each orbital in a channel and between channels, and the absolute scale
through `kappa(E)`. What blurs a map is the distance from the centre to the sample, `height`.

Measured (review, ran, a Gaussian s of width `w` sampled through its Fourier transform, `G_z`
included, kappa = 0.84; the map over the point map, divided by `K_0^2 = exp(kappa^2 w^2)`):

| width (bohr) | 3 bohr | 4 bohr | 5 bohr |
|---|---|---|---|
| 0.5 | 0.9925 | 0.9995 | 0.9998 |
| 1 | 0.889 | 0.978 | 0.997 |
| 2 | 0.19 | 0.38 | 0.60 |

with a spread over pixels at 3 bohr of 0.9, 8 and 22 per cent. So the shape holds while the
orbital stays in the vacuum, and the weight fails before the shape does. A large orbital close to
the sample is outside the tunnelling picture; a guard warns when the potential varies over a
sphere of the orbital's radius around `r0`.

**Which weight is Bardeen's** (review, ran). Bardeen's element is `<chi|U_t|psi>`, with `U_t` the
potential of the tip atom. For an atomic eigenstate `U_t R = (epsilon_b - T_l) R`, so
`K_b = int i_l(kappa rho) (epsilon_b - T_l) R rho^2 d rho`, candidate (c). The tail amplitude
matched in value and log-derivative, candidate (b), is its surface form by Green's theorem (a square
well: equal to six digits, -3.318145 for l = 0 and -5.134452 for l = 1). The plain overlap
`int R i_l rho^2 d rho`, candidate (a), is not Bardeen's element: it grows with the radial cutoff
(10.5, 25.1, 54.2 at 5, 10, 20 bohr) and as `(zeta - kappa)^-2`. At `kappa = 0.57/bohr` on the
files on disk (no W file, Pt 6s stands in):

| orbital | (a) | (b) | (c) |
|---|---|---|---|
| O 2p | 3.82, half of it beyond 5.3 bohr | 0.80 at 3 bohr, 0.36 at 8 | 1.31 |
| Pt 6s | 61, half of it beyond 10.8 bohr | 4.3 at 6 to 8 bohr | 6.6 |
| ratio 6s over 2p | 16 | 5.5 | 5.0 |

Integration by parts gives `(c) = (kappa_atom^2 - kappa^2) (a)` plus boundary terms, so we
checked which of the two carries the pole (`kappa_scan.py`, ran after the review). (c), computed
directly from `U_t R`, is converged in the radial cutoff (O 2p at kappa = 0.57: 1.2997, 1.3028,
1.3020 at 8, 12, 16 bohr) and smooth through `kappa_atom` (2.22 at 0.815), with no zero. (a)
grows with the cutoff (2.770, 3.439, 3.694) and diverges for `kappa >= kappa_atom`, and
`(kappa_atom^2 - kappa^2)(a)` approaches (c) from below (0.94, 1.17, 1.25 against 1.30). So the
factor `1/(kappa_atom^2 - kappa^2)` is a pole of the overlap and not a property of the weight.
The source `U_t R` holds 99.9 per cent of its weight inside 5.95 bohr for O 2p and 9.54 bohr for
Pt 6s, where 5 per cent of (c) lies beyond 8 bohr: the xc potential of a neutral atom decays as a
power of the density. So the weight of a diffuse s orbital is defined to a few per cent, and a
preset records the fraction of its source inside the tip height.

So `K_b` is (c), on the pseudo-atomic orbital of the UPF file (`PP_PSWFC`) with its own eigenvalue.
**Never on a Slater fit**: a Slater function is not the eigenstate of any atomic potential, so
`U_t R` is not defined for it, and the overlap route on Slater shapes puts the 6s over 2p weight
off by 3.5 to 8. The atomic decay is not the vacuum one (O 2p 0.815/bohr, Pt 6s 0.645/bohr, against
0.57 at `E_F`), and a preset states both.

**Decided by the user (2026-11-15): free weights and UPF presets.** The user's pick, from three
options (free weights with presets from UPF orbitals; that plus a spherical-well radius as a free
size; Chen weights only): "Weights + UPF presets". So there is no free size parameter. You give a
free weight per orbital, in Chen's normalisation, as in section 8. A preset takes the size of each
orbital from the real orbital of the tip atom in its UPF file, through `K_b` above, and takes the
share of each orbital at `E_F` from a PDOS of a small tip model. A tip atom with no preset is
built the same way from any UPF file the user has (`tip_presets.from_upf`).

**Preset tips** (`defuqumat/transport/tip_presets.py`), each a list of channels with the weight of
each orbital and where it came from:

- `"s"`: the present point tip. A noble-metal apex (Au, Ag, Cu, Pt) should be mainly s at `E_F`.
- `"W"`: `dz2` and `s` (Chen's argument for atomic resolution with tungsten).
- `"Pb"`: in the metal `E_F` is in the 6p band and the 6s band is far below (Bi, the neighbour on
  disk: 6s at -1.064 Ry, 6p at -0.336 Ry, 9.9 eV apart; review, read), so a Pb apex is p-like.
  With spin-orbit coupling the tip states are `p1/2` and `p3/2`; traced over spin on a sample
  without spin-orbit coupling each is an isotropic p shell, so on such a sample `"Pb"` is
  `px, py, pz` incoherent with one weight, and on a spinor sample it needs the spinor channels of
  section 2 and is refused until they exist. The weights from a `j`-resolved PDOS (P69).
- `"CO"`: the O 2p pi pair (`px`, `py`, two channels) and the sigma part (`s` and `pz` in one
  channel).

The share of each orbital at `E_F` (which orbital is at the Fermi level, the larger effect) comes
from a `projwfc` PDOS of the apex atom of a small tip model (an adatom on a 3x3 slab of the metal,
a CO on Cu(111)), computed with defuqumat on Triton; the weight is `sqrt(n_b) K_b`, where Mandi and
Palotas (their 2.B(iii)) take `sqrt(n_b)` alone. Each preset records its input, the commit, the
PDOS fractions and both decay constants.

## 13. A better route, later

Two steps beyond this phase, in order of cost. (i) The full orbital matrix of the tip apex at
`E`, `Gamma_bb'(E) = sum_nu delta(E - E_nu) C_nu,b C_nu,b'^*`, from the states of a computed tip,
which keeps the s-pz interference and its change with energy (Abilio and Palotas). Its
eigenvectors are the channels of this plan, so nothing in the contraction changes. (ii) Bardeen
with an explicit tip: the states of a tip calculation and of the sample on one separation plane in
the vacuum, `M = int (chi^* d_z psi - psi d_z chi^*) d^2 r`, with no orbital expansion at all; the
map over all tip positions is a two-dimensional convolution, one FFT per band. This is the
reference Mandi and Palotas test the derivative rule against, and the derivative rule is its limit
for a tip whose states are single-`l` tails.
