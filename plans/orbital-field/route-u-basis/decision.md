# Route U's basis: the decision (fable, 2026-11-07)

Logs: D22 /l/ladovj1/review/mag/fable-basis/ (F/<log>); the 1D model and the probe are in
this scratch directory (model1d.py, model1d.log, model1d_extra.log, probe_projected.py).
No repository file was edited; the probe imports the committed modules of w-mag-record
(11bd76e5) and replaces only the gauge step between diagonalize and hybrid_set.

## Decision

Candidate (a), in its exactly nested form: the hybrid state of trial j in cell X is the
projected trial itself, h_{j,0}(kappa') = (1/N1) sum_i sum_m psi_{m,i} <psi_{m,i}|g_j>
over the lowest N bands at each k of the loop (N the outer window), periodized over N1
cells, with no Loewdin step and no transport gauge; the overlap S carries the
non-orthogonality and `solve` orthogonalizes canonically as it already does. Adding a
trial adds a function and changes none of the others, so the spans are nested in J and the
levels fall monotonically. The matrix elements are u1's window + box + symmetric form, so
every level is an upper bound whatever the tails. N is not a knob: it is set by the
projectability of the trials, 1 - sum_m |<psi_m|g_j>|^2/|g_j|^2 at every k, measured and
refused above a threshold (measured here: 1e-6 passes, 2.7e-3 gives Hermiticity 2.5e-4 and
tails 1e-4, 0.12 ruins the spectrum). SMV (b) is refused because its subspace nests in
nothing; the gap cut (c) exists on one case.

## What the windowed sinc experiment read (the review's first try)

F/win-n20.log: free 1/10, window (4,5), N1 20, nbnd 4: Hermitian 5e-9, levels above exact,
(n,m=0) errors 0.007 to 0.030 Ry (n=0), 0.013 to 0.028 (n=1), 0.025 to 0.035 (n=2);
230 s. F/win-n30.log: window (6, 7.5), N1 30, nbnd 4: Hermitian 1.8e-10, errors 0.056 to
0.065 Ry, worse; 666 s. u1's (2, 2.5) read 0.043 to 0.066 at nbnd 4. So the windowed sinc
basis does not converge with the window: the sawtooth is cured, the span is not. (N1 28
refuses: the window needs N1 >= 29.)

## 1D model (model1d.log)

Oscillator p^2 + B^2 x^2, B = 0.7256 /cell^2 (graphene 1/10), lattice of functions at
every cell, exact line vs the code's box rule.
- sinc (hybrid Wannier of the lowest N 1D free bands): <x^2> diverges on a single sinc, so
  the Landau-gauge matrix elements do not exist for an entangled set; on a 40-cell periodic
  line the levels are 4e-2 B off at N = 1, 2, 4 alike (the sawtooth, not the band limit);
  in the box rule (N1 10, 20) they are 0.15 to 0.65 B off and below the exact ones.
- windowed sinc: (2,2.5) 2.5e-2 to 8e-2 B; (4,5) 1.7e-4 B at N = 4; (6,7.5) 2.5e-5 B;
  box rule identical to the line (compact support), Hermitian to 1e-16.
- projected Hermite-Gaussians, sigma 0.5 cells, N 4 or 8 (identical): J 1, 2, 3 give
  2.2e-2, 3.2e-3, 2e-11 B on n = 0, monotone; d_min of S 0.17, 1.8e-3, 3.2e-8; J = 4 is
  complete at N = 4 (d_min 7e-15). sigma 0.35: J 3, 4 give 7e-5, 1.7e-7 B, d_min 2e-3,
  1e-6. sigma 0.7: J 1, 2 give 1.6e-4, 1.6e-7 B. sigma 1.0 (l_B = 1.17): J 1 1.3e-6 B.
  The box rule reproduces the line to the digits at J <= 2 (Hermitian 2e-12 at N 8) and
  shows the conditioning limit at J = 3 (Hermiticity 2.7e-7 after the canonical step).

## Real code, free problem at 1/10, N1 10 (or 16 with the window), kmesh (2,2)

**Which logs are results.** Only `per-*.log` (the periodic trial) and `win-*.log` (the
review's windowed sinc experiment) are results of the method. Every run under `proj-*`,
`c5-*`, `chirp-*`, `nochirp-*`, `zero-*` and `blocks-*` used a G2 = 0 stripe at every reduced
kappa', a trial that is not periodic in kappa', and measures that defect of the probe (the
half chain, below); their level errors (0.33 to 1.2 B_ph) are not route U's numbers. Two
facts from them stand on their own: the 15 A vacuum's ladder (next paragraph) and the
projectability guard firing.

c = 15 A (the committed cell): the trial is a stripe Gaussian in xi1 times the G2 = G3 = 0
plane wave of the reduced k; at k1 = 1/2 and kappa' >= 0.75 its G1 = -1 half sits at
|(-b1/2 + 0.95 b2)|^2 = 3.9 Ry, above the top of 64 free bands (2.7 to 3.5 Ry, the
(2 pi m/c)^2 ladder of the 15 A vacuum fills the window): projectability 0.12 at N 64, 1.0
at N 16, 24, 40 (F/proj-N64-s05.log, proj-N16-s05.log, proj-N40-s07.log). The hybrids are
then flat along xi1 (outside weights 0.8, 0.6, 0.4, 0.2 at w = 1..4 at N 16) and the
levels are off by 0.4 to 1.2 B_ph. This is the guard: a trial not inside the window
delocalizes the whole basis.

c = 5 A (the ladder 9x sparser), N 64, sigma 0.5: projectability 1e-6 (J 1), 3e-5 (J 2),
2e-3 (J 3); tails at w = 2: 7e-8, 2e-6, 2e-5; Hermiticity bare 4e-6, 6e-5, 4e-4
(F/c5-N64-s05.log) and windowed (3,4), N1 16: 9e-13, 1e-11, 6e-11 (F/c5-N64-s05-win.log);
every level above the exact one in the windowed runs; n = 0 error 0.658, 0.556, 0.327 B_ph
(2.2e-2, 1.9e-2, 1.1e-2 Ry) at kappa2 = 0, monotone in J; d_min 0.19, 3e-3, 2.3e-8 (J = 3 at
the conditioning limit, as in 1D; the bare J = 3 run keeps a d = 2.4e-8 direction whose
noise gives a spurious level at -200 B_ph: the window is not optional). N 32: projectability
2.7e-3, Hermiticity 2.5e-4, tails 1e-4 (F/c5-N32-s05.log). sigma 0.7, J 1: projectability
2e-12, tails 1e-9 at w 3, Hermitian 7e-10 bare, error 0.377 B_ph; J 2 ill-conditioned
(d -9e-7, rank 19/20, levels below exact: noise in a near-null direction). sigma 1.0, J 1:
0.123 B_ph. Reversed field (scale -1, F/c5-N64-s05-flip.log): Hermiticity 0.13, levels
+43 and -6 B_ph: the control fires.

Why the real-code J = 1, 2 errors (0.66, 0.56 B_ph) are 30 to 170 times the 1D model's
(2.2e-2, 3.2e-3): on the hexagonal cell the Landau gauge a = c1 xi1 b2 has a part along
b1, and every eigenstate carries the quadratic phase exp(-i f), f = c1 (b1.b2) x^2/2 (u1's
analytic state, item 0b); f = -pi p xi1^2/(2q) = -0.157 xi1^2 at 1/10. A real Gaussian
basis has to build that chirp from H_2 and neighbours, which is what J = 3 begins to do.
Chirped trial runs: F/chirp-s*.log (below).

## Chirped trials and the longitudinal gauge term (F/chirp-*.log, model1d_gauge.log)

On the hexagonal cell H = |b1 p + b2 (K + c1 xi1)|^2 = |b1|^2 [(p + gamma (K + c1 xi1))^2 +
(|b2_perp|/|b1|)^2 (K + c1 xi1)^2], gamma = (b1.b2)/|b1|^2 = -1/2: a 1D oscillator with
beta = c1 sin 120 = 0.544 /cell^2 (l_B = 1.356 cells of a sin 60 = 4.03 bohr) and a
longitudinal gauge term alpha = gamma c1 = -pi p/q per cell^2 (-0.314 at 1/10), whose
eigenstates carry exp(-i alpha xi1^2/2) (u1's analytic state). A chirped trial (the Gaussian
times that phase about its own cell) changes nothing at sigma 0.5 (0.656 against 0.658
B_ph) and at sigma = l_B = 1.356 gives n = 0 to 2.9e-3 B_ph (1e-4 Ry), n = 1 to 0.156, n = 2
to 1.21 B_ph, Hermitian 1.6e-9, reversed field 0.33 (F/chirp-s1.356.log,
chirp-s1.17-flip.log). The 1D model with alpha, the same window (3,4) and the per-cell chirp
gives 1.5e-3, 1.0e-2, 5.1e-2 B_ph (model1d_win.py): the window cuts the l_B Gaussian at
2.2 sigma, which is the n = 0 residual in both; at q = 10 the n = 2 state is 3 cells wide
and the window must be 8 to 10 cells (N1 about 30).

Resolved: the 20x gap between the real code and the 1D model was the probe's trial, not the
method. A G2 = 0 stripe at every reduced kappa' is not periodic in kappa', and the code reuses
the reduced state for X and X +- q with the Miller relabelling, so the cells X < 0 carried the
y-momentum kappa + b2 (F/blocks-s05.log: S(Delta = -1, -2) = 1e-17 against S(+1) = 0.37): each
oscillator saw a half chain of Gaussians with its centre at the chain's end. With the lowest
band's stripe, G2 = -round(kappa'), periodic as a crystal's Bloch sum of a localized orbital
is, the real code reproduces the 1D model (F/per-*.log, c = 5 A, N = 64, window + box +
symmetric):

| trial | J | n = 0, 1, 2 error / B_ph | in Ry (n < 3) | Hermiticity | d_min |
|---|---|---|---|---|---|
| sigma 0.5, window (3,4), N1 16 | 1 | 3.24e-2, 1.17e-1, 2.9e-1 | 1.1e-3 to 9.8e-3 | 3e-14 | 0.19 |
| | 2 | 4.4e-3, 1.7e-3, 1.9e-3 | 5.6e-5 to 1.5e-4 | 5e-13 | 3.1e-3 |
| | 3 | 1.8e-7, 6.0e-6, 9.1e-5 | 5.7e-10 to 3.1e-6 | 3e-12 | 2.2e-8 |
| the same, window (5,6), N1 24 | 1, 2, 3 | the same to the printed digits | | 7e-15 to 6e-13 | |
| sigma 1.356 = l_B |b1|, chirped, (3,4) | 1 | 1.95e-3, 1.5e-2, 8.8e-2 | 6e-5 to 2.9e-3 | 6e-9 | 7.9e-5 |
| the same, window (5,6), N1 24 | 1 | 1.7e-7, 1.5e-3, 3.6e-4 (kappa2 0); 6.4e-5, 6e-6, 1.5e-2 (kappa2 1/20) | 5.8e-9 to 5.0e-4 | 6e-12 | 5.7e-5 |
| sigma 1.356 chirped, field reversed | 1 | +12, +8.4, -3.5 | | 0.59 to 0.62 | |

1D model (model1d_gauge.log, alpha included): sigma 0.5 J 1, 2, 3: 3.3e-2, 4.4e-3, 3.7e-11
(J = 3 in the real code stops at 1.8e-7: rank 30 of 30, nothing dropped, the 3e-12 residual over d_min 2.2e-8);
every level above the exact one; monotone in J at every k. The window (3,4) against (5,6)
changes nothing at sigma 0.5 (the tails are 1e-8 at w = 2) and takes the l_B Gaussian's
n = 0 from 1.95e-3 to 1.7e-7 B_ph (its weight beyond 3 cells, erfc(2.2), was the cut).
## Acceptance test

Gate, the free Landau problem at 1/10 on the graphene cell with c = 5 A (the 15 A vacuum's
(2 pi m/c)^2 ladder fills any band window below the trial: 0.12 of a sigma 0.5 trial lies
above 64 bands at kappa' = 0.95, F/proj-N64-s05.log), N = 64, N1 16, window (3,4), box,
symmetric, kmesh (2,2), the trial the lowest band's stripe times a Hermite-Gaussian of
sigma = 0.5 cells along xi1 (not l_B): (i) projectability above 1 - 1e-8 at every k (the
guard; 2.7e-3 reads Hermiticity 2.5e-4); (ii) Hermiticity below 1e-10 at every k; (iii)
every level above (2n+1) B_ph; (iv) the n = 0 error falls monotonically J 1, 2, 3 and reads
3.2e-2, 4.4e-3, below 1e-6 B_ph (1.1e-3, 1.5e-4, below 3e-8 Ry), with d_min 0.19, 3e-3, 2e-8
(J = 3 is the conditioning limit of three Hermite-Gaussians of sigma 0.5 at unit spacing:
nothing was dropped, rank 30 of 30, and the 3e-12 non-Hermiticity over d_min 2e-8 is the
1.8e-7 floor, not the span's); (v) the plumbing line: the chirped Gaussian of sigma = l_B |b1| =
1.356 at J 1, window (5,6), N1 24, reads n = 0 to 2e-7 B_ph at kappa2 = 0 (the exact state is
in the span and comes back) and the reversed field reads Hermiticity 0.6 and levels at
+12 B_ph. All five were read here on the probe; the implementation has to read them again
through the workflow.

The acceptance proper, graphene against route P, which the gate cannot replace because the
free problem's trials know their centres and their stripes. Trials: the pseudo-atomic
orbitals of C.pbe-hgh.UPF (it carries PP_CHI 2S and 2P, so pao_projections /
projection_matrices give the 8-function set as they do for projections="auto"), then the
nested supersets J = 2 (p_z), 8 (s, p), 18 (+ d hydrogenic at zona 2 per atom through
parse_projections), at fixed N = 32 (the band count of the committed route P sets), window
(3,4), N1 16 (the box is 16 x 24 x 24 x 128 at c = 15 A; the blocks at J = 18 are 5 J npts
complex, 1.3 GB). Three lines, in this order:
1. p = 0 first: the projected basis at zero flux against the sphere bands at the same J and
   the same k, the span's own zero-field error per level (the restated zero-flux check; the
   J = N case must return the bands to 1e-10). It is subtracted below as cross.md subtracts
   the box-sphere floor, so the field's part is separated from what the trials cannot hold
   at zero field. Seconds.
2. Primary: the full Hamiltonian with the GIPAW term at q = 6 against route-p-q6-full.npz
   and at q = 10 against route-p-q10-reps.npz, the levels inside E_D +- 2.5 eV (the
   crosscheck's item 3 window, where the 8-function set is within 1.8 meV of the sphere at
   zero field), pooled over the four representatives. Pass: every projectability above
   1 - 1e-4 (the hydrogenic d may fail it: then the record says so and zona goes to 3),
   Hermiticity below 1e-10, U above P at every level, (U - P) - (line 1) falling
   monotonically J 2 -> 8 -> 18, and below 5 mRy at J 18 (the design's line).
3. Secondary: local-only at q = 6 against route-p-q6-local.npz, the lowest 4q pooled, with
   cross.md's item 1 floor (today 2.8e-3 to 6.2e-2 Ry at nbnd 32 and not falling). It is
   secondary because the lowest 4q there hold the collapsed deep pair (box against sphere
   0.118 to 0.141 Ry, the hardest states of the crosscheck), whose projectability on a 2s
   trial is unmeasured; a miss there would say nothing about the method.
If 5 mRy is not met at J 18 the record reports the number and whether it falls N 32 -> 48
(not a nested knob, reported as such); at l_B = 1.4 to 1.8 cells the span of 18 atomic
functions per cell is not guaranteed to hold the Landau states to 5 mRy, and that would be a
measured limit of route U at q <= 10, where route P is exact anyway.

## Is route U worth continuing

Yes, in one form only, and not as a cheaper route P. At q <= 10 route P is exact (1e-11 on
the free ladder, 1234 s at q = 10) and route U converges slowly (0.33 B_ph at J 3 with sigma
0.5 Gaussians; the expansion parameter (spread/l_B)^2 is 0.1 to 0.5). Where route U is the
only route is 60 < q < 8000 with the intra-cell terms Peierls drops (mu_B B L on a Wannier
orbital, the diamagnetic term, the hopping change: 0.6 per cent of E_1 for graphene p_z at
10 T, comparable to the spacing for a d band; the crosscheck's unattributed 0.2 eV at q 6 to
10 is this question). Two facts make it reachable:
1. The span identity: chi_{n,X}(kappa) = sum_R2 exp(2 pi i kappa R2) T(X a1 + R2 a2) w_{n,0},
   so route U's basis is the Bloch sum along a2 of the magnetically translated 2D Wannier
   functions of the same subspace: it is section 5's gauge-including Wannier basis (W1)
   evaluated in k-space. The two items are one.
2. The projected trial P_N(k) g_j is gauge-free, so h_{j,0}(kappa') is a smooth function of
   kappa' and the blocks B_X(Delta) = B(kappa_X, Delta) can be computed on a coarse kappa'
   mesh (50 to 100 points, N1 x 100 NSCF k and 100 (2w+1) block calls, minutes) and
   interpolated to kappa + X p/q for any q, 7900 included; the assembly is a q J banded
   matrix like the Peierls torus (63200 at J 8, 40 s). This is a claim about Bloch-sum
   trials (atom-centred, smooth in kappa'); the free stripe trial switches G2 at kappa' = 1/2
   and is not the one to test it on. The identity test is at q = 50 on graphene: a direct
   projected run first (u2's cost at nbnd 8: 285 s NSCF, 620 s blocks; u2's own q = 50 blocks
   are the refused sinc basis and do not serve), then the interpolated levels against it to
   1e-6 Ry.
With the Loewdin or SMV gauge neither holds (the gauge of each k1 loop changes with every
knob), which is the second reason for the raw projected basis.

If route U is dropped, what is lost is exactly that: the intra-cell orbital terms at
laboratory fields with a variational knob, and the attribution of the 0.2 eV. Route P
cannot reach q > 60, Peierls has neither.

## Size

- hybrid.py: `projected_hybrid(states, trials, N)` returning the (N1, N, J) rotation
  (A = <psi|g>, no Loewdin) and the projectability per k; the stripe Hermite-Gaussian trial
  for the free problem (periodic: G2 = -round(kappa')) and the atom-centred trials through
  wannier/trial.py and wannier/pao.py as they are. 150 lines.
- landau_ultracell.py: `trials=` (a list of projections, "pao", or the free stripe), `N`
  separated from `nbnd` (the result's nbnd becomes J), the projectability guard (refuse below
  1 - 1e-4 by default, the threshold a keyword), window + box + symmetric as the defaults
  of a projected run, the zero-flux check restated (at p = 0 the levels are the bands only
  inside the span: a test that the J = N case returns them to 1e-10 and the J < N case
  returns them above by the projectability floor). 80 lines.
- Tests (fast, small cell): nested monotone fall in J on a 4-cell free problem; the guard
  fires on a trial outside the window; the reversed field; zero flux at J = N. 4 tests.
- Probes and records: the gate (five runs, 15 min), graphene q 6 local (3 J x NSCF 120 k x
  32 bands, 10 min), full q 6 and 10 (30 min), cross.md's item 1 analysis reused on
  `levels_J<J>` keys; u3.md and the P176 section. One implementer step of u1's size, one to
  two days.
- The kappa' interpolation and the q = 50 identity: a second step of the same size, after
  the acceptance.

## For docs/d22-notes.local.md (the parent records; no file was edited here)

- `pkill -f <pattern>` run over ssh kills the ssh session itself when the pattern occurs in
  the session's own command line: the call returns exit 255 and the commands after it in the
  same ssh line never run (here: four launches silently did not happen). Kill by pid, or use a
  pattern that is not on the launch line.
- A window (inner, outer) of the ultracell needs N1 >= ceil(2 outer) - 1 + 2 outer: (6, 7.5)
  refuses at N1 28 ("needs N1 >= 29 cells") and ran at 30; (3, 4) needs 15, (5, 6) needs 23.
- The free problem on the committed 15 A graphene cell cannot be projected on any band count
  below about 160: the (2 pi m/c)^2 ladder (0.049 Ry per m^2) fills the window below the
  stripe's energy, so a sigma 0.5 trial has 0.12 of its weight above 64 bands at kappa' 0.95
  (1.0 at N 16 to 40). The symptom reads like a wrong trial: flat hybrid profiles with outside
  weights 0.8, 0.6, 0.4, 0.2 at w = 1..4. c = 5 A (ladder 0.44 Ry per m^2) holds it at N 64.
- A free-electron stripe trial must be periodic in kappa' (G2 = -round(kappa')); a G2 = 0
  stripe at every reduced point gives S(Delta < 0) = 1e-17 against S(+1) = 0.37 for the bra at
  X = 0 (blocks-s05.log), a half chain, and levels 0.3 to 1.2 B_ph off that look like a slow
  basis.
- Probe costs in a slot: N 64 free NSCF at 200 k, c 5 A: 70 to 110 s; blocks at J 1, 2, 3 with
  N1 16 window (3,4): 16, 23, 35 s; N1 24 window (5,6): 24, 52, 78 s. The windowed sinc runs:
  N1 20 nbnd 4 231 s, N1 30 nbnd 4 666 s.
