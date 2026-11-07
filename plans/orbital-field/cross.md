# The crosscheck of route P, route U and Peierls on one graphene state

Branch `mag-cross`: route U step 2 (`mag-u2`, 7ead1841) with route P step 2 (`mag-p2`, 1572b852)
merged in, and the probe `plans/orbital-field/mag_cross.py` on top. Every number below was computed
on D22 in the worktree `/l/ladovj1/w-mag-cross`, through `slot.sh` (one core, 6 GB). The logs are in
`/l/ladovj1/review/mag/cross/`, written `C/<log>` below. Logs of the route steps are written `U1/`,
`U2/`, `P1/`, `P2/` for `/l/ladovj1/review/mag/{u1,u2,p1,p2}/`. The load average was 4.5 to 8.

Everything compared is a sorted level set pooled over the four magnetic representatives
`kappa1 in {0, 1/2}`, `kappa2 in {0, 1/(2q)}`, a DOS, or an energy. It is never `eps(k)` at a named
`k`. The one exception is the local-only against full check on route P, which compares two operators
on the same grid at the same `k`, not two routes. All three routes read the same zero-field
potential, `plans/orbital-field/graphene-scf.npz`.

## What was run here

- `C/exact.log`: route P's full mesh against `q` copies of its representatives, from every committed
  `route-p-q*.npz`.
- `C/floor-full.log`, `C/floor-local.log` (and `C/floor-full-q6.log`, the first run on fewer points):
  route P's box at zero field (`p = 0`, `q = 1`) and the sphere Davidson of the same Hamiltonian, on
  the 108 chemical `k` that fold onto the representatives of `q = 1, 2, 3, 4, 6, 8, 10`, and the three
  zone corners. The box took 133 s for 111 k, the sphere 49 s. Saved as
  `plans/orbital-field/cross-floor-{full,local}.npz`.
- `C/reps-q8.log`, `C/reps-q10.log`: route P full H at `1/8` (9864 T) and `1/10` (7891 T) on the four
  representatives only, nbnd `4q + 8`. 822 s and 1234 s, 95 to 102 Davidson steps at `q = 8`, peak
  RSS 2.04 GB at `q = 8`. Saved as `plans/orbital-field/route-p-q{8,10}-reps.npz`. These go beyond the
  design, which stopped the comparison at `q = 6`; the reason is under "Decisions".
- `C/landau-10-3.log`: route P's free Landau problem at `3/10`, kmesh (2, 2), with 600 Davidson steps
  allowed. The first try at 200 steps was refused by the solver, 13 of 16 bands unsettled
  (`C/landau-10-3-200.log`).
- `C/analyse5.log`: items 1, 3 and 4 and the checks below, about 2 s. `C/diag.log`: the zero pair of
  route P against Peierls and route U, with a scan of `kappa` offsets.
- `C/t-unit.log`: `tests/unit/test_magnetic_cross.py`, 3 passed in 0.95 s.

## Route P alone: the full mesh is `q` copies of the representatives

All 16 committed sets, local, filtered and full, `q = 1` to `6`: the worst is **6.9e-11 Ry**
(`route-p-q4-local-filtered.npz`), and every two-grid set is within 3.5e-12 Ry (`C/exact.log`). The
design asks 1e-9. The same theorem holds for the Peierls model on the `p_z` set at `q = 6` to round-off,
and an open set of representatives (`kappa2 in {0, 2/(2q)}`) misses by more than 1e-4 Ry, which is the
guard firing (unit test).

## Item 1: local-only operator, route U against route P

Route U is the sphere-based ultracell, route P the box. The lowest `4q` levels per k, pooled over the
four representatives and sorted (`16q` levels), `U - P` in Ry (`C/analyse5.log`):

| q | nbnd 8: min / max | nbnd 16: min / max | nbnd 32: min / max | levels with U < P - 1e-8 |
|---|---|---|---|---|
| 1 | 0.147 / 0.422 | 0.137 / 0.437 | 0.137 / 0.389 | 0 of 16 |
| 2 | 0.063 / 0.231 | 0.083 / 0.233 | 0.086 / 0.217 | 0 of 32 |
| 3 | 0.042 / 0.181 | 0.065 / 0.245 | 0.059 / 0.193 | 0 of 48 |
| 4 | 0.032 / 0.159 | 0.059 / 0.283 | 0.060 / 0.429 | 0 of 64 |
| 6 | 0.022 / 0.141 | 0.028 / 0.170 | 0.035 / 0.148 | 0 of 96 |

Route U is above route P at every level, at every `q` and every nbnd. So the variational order holds,
and no level flags a defect in either route. The gap does **not** shrink with nbnd: from 8 to 32 the
smallest gap grows at every `q` but the first. The expected `< 5 mRy` at nbnd 32 is not reached; the
smallest gap at nbnd 32 is 35 mRy (`q = 6`) and the largest 0.43 Ry (`q = 4`).

Part of the gap is the basis, and it does not depend on the field. At zero field route U is the sphere
(u2 acceptance 1, 1.8e-15 Ry) and route P is the box, and the box is below the sphere at every one of
the 51 k of `q <= 6` and all 10 bands, by 1.5e-4 to 0.141 Ry for the local-only operator
(`C/floor-local.log`; the deep two bands 0.118 to 0.141 Ry, the others below 0.035 Ry). For the full H
it is below at all 111 k, by 1.5e-4 to 2.4e-2 Ry (`C/floor-full.log`). Folded onto the same representatives,
this zero-field floor runs from 7.3e-3 to 0.14 Ry. What is left above it, `(U - P) - floor`, is
2.8e-3 to 0.062 Ry at `q = 6`, nbnd 32, and up to 0.37 Ry at `q = 1`. This part is route U's own
truncation error, and it does not fall with nbnd either. Route U's matrix at these runs is Hermitian
only to 5e-3 to 3e-2 (relative, from its npz), against 5e-15 for route P's operator, which matches an
independent Kronecker-product matrix to 3.7e-16 (route-p.md, item 2). So the defect is route U's basis,
which is what u1 and u2 found: the hybrid states of an entangled band set are not localized.

## Item 2: the free Landau problem

`(2n+1) B_ph + (2 pi m/c)^2`, graphene geometry, all potentials off. `B_ph` is 0.0335723 Ry at
`p/q = 1/10` (7891.2 T); the design's 0.03355 is the rounded cell area.

| route | p/q | B (T) | kmesh | nbnd | max error against the formula | flat over k | states per level | log |
|---|---|---|---|---|---|---|---|---|
| P | 1/10 | 7891.2 | 4 x 4 | 12 | 2.9e-12 Ry | 2.7e-12 Ry | 1 | `P1/landau-10-1-final.log` |
| P | 2/5 | 31564.9 | 4 x 4 | 12 | 3.2e-12 Ry | 2.0e-12 Ry | 2 | `P1/landau-5-2-buffer.log` |
| P | 2/11 | 14347.7 | 4 x 4 | 12 | 1.7e-11 Ry | 1.3e-11 Ry | 2 | `P1/landau-11-2c-400.log` |
| P | 3/10 | 23673.7 | 2 x 2 | 12 | **9.8e-12 Ry** | 5.3e-12 Ry | 3 | `C/landau-10-3.log` (new, 310 to 391 steps) |
| U | 1/10 | 7891.2 | 2 x 2 | nbnd 2 | 1.2e-2 to 0.57 Ry over 20 levels; n = 0 0.012 to 0.121 | 0.11 Ry | (not resolved) | `U1/free1.log` |
| U | 1/10 | 7891.2 | 2 x 2 | nbnd 4 | n = 0 0.030 to 0.037, n = 1 0.061 to 0.072, n = 2 0.059 to 0.079 Ry | 0.18 Ry | | `U1/free1.log` |
| U | 3/10 | 23673.7 | 2 x 2 | nbnd 2 / 4 | 0.118 to 0.414 / 0.120 to 0.244 Ry | 0.11 / 0.031 Ry | | `U1/free-p3.log` |

`p = 2, q = 10` is not a reduced flux and `MagneticCell.build` refuses it, so route P ran `2/5` and
`2/11` instead, and route U ran nothing at `p = 2`. Route P meets the design's 1e-8 Ry at every flux,
now including `3/10`; route U misses it by 1e-2 to 1e-1 Ry at every flux it ran (u1's refusal).

## Item 3: full Hamiltonian, route P against Peierls

**The comparison by index of the design does not compare the same states, so it was replaced by
energy windows.** Route P's box holds every state of the slab. At zero field, measured on the box at
`Gamma` (`C/dbg.log`), the nearly-free vacuum states start 3.2 eV above `E_D` and the top of the sigma
bands is 3.1 eV below it, and neither Wannier set holds them. So the levels `3q .. 5q - 1` of route P
are not the levels `0 .. 2q - 1` of the `p_z` set (at `q = 2`, by index, the difference is 2.1 eV
already at zero field). What is compared instead: the **zero pair**, the two levels at the gap per k
(index `4q - 1, 4q` for route P and the 8-function set, `q - 1, q` for `p_z`), and every level within
1 eV or 2.5 eV of each route's own `E_D`, compared only when the count per k agrees. Each set is taken
about its own `E_D`: route P's on its box at `K` (-0.130751159 Ry), Peierls' from its interpolated bands
(-0.119587541 Ry, which is the sphere's `E_D` to 1e-14). The box `E_D` is 11.16 mRy (152 meV) below the
sphere's. `E_1` is the Dirac formula at the field, with `v_F = 8.42e5` m/s of the `p_z` set.

The zero pair, `p_z` set (`C/analyse5.log`; the 8-function set gives maxima within 15 per cent of these at
`q = 2, 3` and within 2 per cent from `q = 4` on):

| q | B (T) | E_1 Dirac (eV) | max abs(P - W) (meV) | mean P - W (meV) | max / E_1 | the same at B = 0, max / mean (meV) |
|---|---|---|---|---|---|---|
| 2 | 39456 | 6.069 | 474 | +44 | 0.078 | 2102 / -195 |
| 3 | 26304 | 4.955 | 294 | -123 | 0.059 | 1746 / -37 |
| 4 | 19728 | 4.291 | 291 | -187 | 0.068 | 64 / +4 |
| 6 | 13152 | 3.504 | 270 | -222 | 0.077 | 17 / +9 |
| 8 | 9864 | 3.034 | 240 | -218 | 0.079 | 17 / +13 |
| 10 | 7891 | 2.714 | 214 | -204 | 0.079 | 17 / +14 |

The B = 0 column is the same statistic on the chemical k that fold onto the representatives (box at
`p = 0` against the interpolated Wannier bands). At `q = 2, 3` the two levels at the gap of the folded
zero-field set are far from `E_D` and the column says nothing; from `q = 4` on it is 17 to 64 meV, so
the zero-field bases agree near `E_D` and the difference in the table is the field's. On the folded k
the `p_z` set is within 11.4 meV of the sphere for every level within 2.5 eV of `E_D`, and the
8-function set within 1.8 meV.

**What the numbers say.** The difference is not linear in B, and it does not fall towards 10 T. From
`q = 6` to `q = 10` route P's zero pair sits a nearly rigid 0.20 to 0.22 eV below Peierls', while the
pair itself narrows from 1.05 to 0.34 eV (at the three representatives where it is split, P
(-0.894, +0.159) against W (-0.704, +0.429) eV at `q = 6`, and P (-0.475, -0.140) against W
(-0.279, +0.074) eV at `q = 10`; the P pair is 5 to 7 per cent narrower). So `max/E_1` stays at 0.08
while B falls by 1.7. The fit `s = a + c B` of the maximum over `q = 2 .. 10` has residuals of 53 meV
on values of 214 to 474 meV, and `s/E_1` is no more linear in B (residual 1.25e-2) than in `sqrt(B)`
(1.18e-2). The design's extrapolation to 10 T, taken anyway, reads 7.3e-2 meV for the slope times
10 T, 7.5e-4 of `E_1(10 T) = 96.6` meV (`p_z`, the maximum), or 8.4e-4 from the mean; these are
numbers from a fit that does not describe its own points, and three decades away from them, so they are
**not** a measurement of the intra-cell term and are not compared with the 0.6 per cent of section 12.
The expected linear regime is at fields where `l_B` is much longer than the cell, and `q <= 10` is
not there: at `q = 10` the magnetic length is 2.9 A, about two bond lengths.

Window comparisons. Within 1 eV of `E_D` the zero pair is the only content from `q = 6` on; at
`q = 2, 3, 4` the counts agree and give the same maxima as the zero pair. Within 2.5 eV the counts agree
only at `q = 3, 4, 10`: at `q = 10`, 21 levels, max 214 meV, mean -1 meV; at `q = 2, 6, 8` Peierls
puts one level per k more or fewer in the window than route P, and those are not compared.

**Is the 0.2 eV route P's defect or Peierls' error?** Four checks, none of which settles it alone:

- Not a labelling offset. A scan of 11 x 11 offsets of the Peierls `kappa` finds its best match at
  zero offset for `q = 3, 4, 6`, at 270 meV for `q = 6` (`C/diag.log`). Over a dense 10 x 120 magnetic
  mesh at `q = 6` the Peierls pair has a mean of -141 meV about `E_D`, against -364 meV for route P's
  four representatives.
- Not the nonlocal term. C.pbe-hgh carries one s projector, which is even under the mirror `z -> -z`,
  and the pi states are odd. Each full-H level within 1 eV of `E_D` is within **3.0e-7 to 9.4e-7 Ry**
  of a local-only level at the same k at `q = 2, 3` (`C/analyse5.log`; at `q = 4, 6` the local-only
  set of `4q + 8` levels stops below `E_D`, since the local-only operator has two more bands per cell
  below the pi pair). At zero field, at `K`, the two differ by 6.2e-7 Ry on the box and agree to the ten
  printed digits on the sphere.
- Not a uniform offset of route P. For a local potential no level of `(p + A)^2 + V` lies below the
  zero-field ground state (the diamagnetic inequality). Route P's local-only lowest level is above the
  zero-field box minimum by +460, +218, +186, +144, +97 meV at `q = 1, 2, 3, 4, 6`. With the full H the
  lowest level rises by +1576, +1070, +806, +540 meV at `q = 2, 3, 4, 6`, against `(m/m*) B_ph` =
  1618, 1078, 809, 539 meV for the box's own deep-band mass at `Gamma` (`m/m* = 0.708`). A rigid
  0.2 eV offset of route P's spectrum would show here as a 40 per cent miss at `q = 6`; it is 0.1 per
  cent.
- Route U is no help: its zero pair at `q = 6`, nbnd 32, is 0.39 eV above route P's and 0.17 eV above
  Peierls' on average, and it is not converged in nbnd (`C/diag.log`).

What would separate the two: route P at `q = 6` and `q = 10` at 70 Ry, where the box's own error at `K`
falls (item 1 of p2: 2.0e-2 Ry at 50 Ry, 7.2e-3 at 70 Ry at `Gamma`), and a Wannier set with the
mirror-odd states above the pi bands in it (the odd vacuum states 3 to 4 eV up), which are the only
partners the field can mix into the pi states. Adding the sigma functions does not move it (the
8-function set gives the same 0.2 eV), as the mirror forbids. Neither was run.

Route P against route U at nbnd 32 (route U above route P everywhere, `C/analyse5.log`), `U - P` in Ry:

| q | lowest 4q: min / max | 2q about E_D (index 3q .. 5q - 1): min / max | route U Hermiticity |
|---|---|---|---|
| 2 | 0.068 / 0.201 | 0.067 / 0.159 | 3.1e-2 |
| 3 | 0.038 / 0.123 | 0.045 / 0.100 | 3.7e-2 |
| 4 | 0.029 / 0.090 | 0.036 / 0.083 | 3.2e-2 |
| 6 | 0.023 / 0.052 | 0.029 / 0.064 | 1.9e-2 |

No level of route U is below route P's at any nbnd (0 of `16q`). The zero-field floor of the full H is
smaller than the local-only one, 7.3e-3 to 2.0e-2 Ry, so route U's own error is 1.1e-2 to 0.19 Ry
above it. Both index windows are fair here, since both routes hold every state.

## Item 4: the DOS at `q = 6`

Gaussian of 20 meV, each set about its own `E_D`, the four representatives (route P's full mesh is six
copies of them), states per eV per magnetic k (`C/analyse5.log`).

| window | Wannier set | states per k, P / W | relative L2, abs(D_P - D_W) / abs(D_P) |
|---|---|---|---|
| E_D +- 1 eV | p_z | 2.00 / 2.00 | 1.445 |
| E_D +- 1 eV | 8 functions | 2.00 / 2.00 | 1.448 |
| E_D +- 2.5 eV | p_z | 2.00 / 2.75 | 1.593 |
| E_D +- 2.5 eV | 8 functions | 2.00 / 2.75 | 1.595 |

Within 1 eV both DOS hold the zero pair only, the n = 0 Landau band (`E_1` is 3.5 eV here). Their
peaks are 0.2 to 0.27 eV apart, ten broadenings, so they do not overlap and the L2 difference is the
`sqrt(2)` of two disjoint sets of peaks: the DOS says the same as item 3 and no more. Within 2.5 eV
Peierls also puts the next level at +2.43 eV at three of the four representatives, which route P does
not. The KPM DOS of the Peierls operator was not used (dimension 12, the level histogram is exact).

## Item 5: reported from the route records, not recomputed

- Diamond, Lu's 3x1x1 conventional cell, 60 Ry, kmesh (2, 6, 6), 10877.4 T: `chi = -11.40e-6`
  cm^3/mol per C2 (-5.70 per C), against Lu's -11.16 (LDA): 0.24 apart, 2 per cent.
  `E(+) - E(-) = 1.63e-7` Ry, above the design's 1e-7 and below the 1e-6 refuse line
  (`P2/chi.log`, `P2/diamond-p0.log`, `P2/diamond-p1.log`, `P2/diamond-m1.log`).
- The band shift, graphene at `n_Phi = 1`, kmesh (6, 6): `eps(k)` at the same named k moves by up to
  0.291 Ry, the sorted level set by 9.7e-13 Ry and the band energy by 2.0e-14 Ry (`P2/shift.log`).

## Decisions taken here, and why

No subagent could be started from this step (the Agent tool is not available to it), so each open
choice is the conservative one, recorded with the number it rests on.

- **Route P's `E_D` is taken on its own box at `K`**, not the sphere's, so that each set of levels is
  measured from a point of its own basis. The two differ by 152 meV; with the sphere's the P - W
  difference of item 3 would be 152 meV larger.
- **Item 3 compares in energy windows, not by index**, for the reason above: by index the windows hold
  different states (2.1 eV apart at zero field at `q = 2`).
- **A zero-field floor was computed** (box and sphere at `p = 0`, 3 minutes), because without it the
  variational gap of item 1 and the P - W difference of item 3 mix the box's basis with the field.
- **`q = 8` and `q = 10` were added** to item 3 (4 representatives each, 14 and 21 minutes in a slot),
  because at `q <= 6` the difference rose and fell without a trend; two smaller fields were the
  cheapest way to see whether it falls with B. It does not, from `q = 6` to `q = 10`.
- **The free problem at `3/10` was run on route P** (6 minutes), to put the two routes side by side
  at the flux route U ran.
- The fits of item 3 are printed but not used as a measurement, since they do not describe their own
  points.

## Tests

- `tests/unit/test_magnetic_cross.py` (new, 3 tests, 0.95 s in total on D22, `C/t-unit.log`): route P's
  full meshes are `q` copies of the representatives to 1e-9 in every committed set; an open set of
  representatives fails by more than 1e-4; the Peierls model on the `p_z` set is Hermitian at `q = 6`
  and obeys the same theorem. It reads the committed npz files and builds a 12-row Peierls matrix.
- No slow test was added. The comparison itself is 2 s on committed files. The two runs that compute
  (`floor`, 3 minutes, and `reps q`, 14 to 21 minutes) are probe subcommands and not tests.
- Slow files that the merged branches touch, for the user to run (not run here):
  `tests/regression/test_magnetic_bloch_slow.py`, `tests/regression/test_magnetic_ultracell_slow.py`.

## What is left

- The 0.2 eV of item 3: route P at 70 Ry at `q = 6` and `q = 10` (box error), and a Wannier set with
  the mirror-odd vacuum states (Peierls' missing partners). Until one of them is run, the size of the
  intra-cell term of graphene is not measured; what is measured is that Peierls and route P differ by
  0.08 `E_1` at 7900 to 13000 T, nearly independent of the field over that range.
- The linear regime of section 12 item 2 needs `q` of 30 and more on route P (estimated 10 GB at
  `q = 30` for the SCF in p2; non-self-consistent at four representatives, less).
- Item 1's 5 mRy and the nbnd convergence: blocked on route U's basis (the smooth band subspace of u2's
  "What is left"), not on anything here.
- The DOS at `q = 6` by KPM on the route P operator was not built; the level histogram is exact here.

## For `docs/d22-notes.local.md` (the Record stage)

- Route P's free problem at `3/10` (three states per level) needs about 400 Davidson steps per k; at
  the default 200 it is refused with "Davidson left 13 of 16 bands unsettled at k-point 0 after 200
  steps", which reads like a broken preconditioner. `MAG_MAXIT=600` on the slot.sh line is enough
  (310 to 391 steps, 347 s for 4 k).
- Route P full H on four representatives: `q = 8` 822 s, peak 2.04 GB; `q = 10` 1234 s, in one slot.
- The worktree `/l/ladovj1/w-mag-cross` is removable once `mag-cross` is merged.
