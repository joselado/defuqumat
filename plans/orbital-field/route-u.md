# Route U step 1: the acceptance table

The full record, with the reasons and the decisions, is `plans/orbital-field/u1.md`. Logs are
on D22 in `/l/ladovj1/review/mag/u1/`. Status: **refused**: the bare hybrid basis of an
entangled band set is not localized.

| item | measured | design | log |
|---|---|---|---|
| 0. zero flux, levels against eps(k) | 5.3e-15 Ry | (exactness check) | `zero1.log` |
| 0b. field kernels on the analytic Landau state, E0/B_ph - 1, E1/3B_ph - 1 (N1 = 18) | 4e-16, 2e-15; reversed field +0.667 | (operator check) | `landau-check-n18.log` |
| 1. free q = 10, p = 1, (n, m=0) error, n <= 2, nbnd 2 / 4 | up to 0.143 / 0.079 Ry | 1e-8 Ry | `free1.log` |
| 1. free, monotone in nbnd | rise +2.5e-2 Ry (2 -> 4) | no rise > 1e-10 | `free1.log` |
| 1. free, p = 3 | errors 0.118 to 0.180 Ry | 3 states per level | `free-p3.log` |
| 1. free, rank | q nbnd at every nbnd | q nbnd | `free1.log` |
| 2. Hermiticity, free / graphene | 5.7e-2 / 6.6e-3 (q = 6) | < 1e-10 | `free1.log`, `cost-q6.log` |
| 2. Hermiticity, isolated composite (nbnd 2), eigen / box H0 | 2.0e-4 / 1.7e-6 | | `iso-n10.log`, `isobox-n12.log` |
| 3. N1 = 8 against 12, q = 6, nbnd 8, lowest 24 | 2.28e-2 Ry | 1e-5 (refuse 1e-3) | `cmp-acc3.log` |
| 3. spread: weight outside w = 1 .. 5 (N1 = 12) | 0.231, 0.111, 0.068, 0.043, 0.021 | < 1e-8 | `acc3-n12.log` |
| 4. nbnd 8, 16, 32 at q = 1, 2, 3, 4, 6, lowest 4q | not monotone, rises 0.02 to 0.21 Ry | monotone | `acc4-4q.log` |
| 6. cost q = 6 / q = 10, nbnd 8, one core | NSCF 28.3 / 37.8 s, blocks 54.8 / 91.4 s | (record) | `cost-q6.log`, `cost-q10.log` |

The level sets `route-u-q{1,2,3,4,6}-local.npz` are measurements of this refused construction,
not converged upper bounds.

# Route U step 2: the acceptance table

The full record, with the reasons and the decisions, is `plans/orbital-field/u2.md`. Logs are
on D22 in `/l/ladovj1/review/mag/u2/`. Status: **refused at item 2** for step 1's reason (the
hybrid basis of an entangled band set is not localized); the nonlocal term itself checks out.

| item | measured | design | log |
|---|---|---|---|
| 1. zero flux, full H, levels against eps(k) | 1.8e-15 Ry | 1e-9 Ry | `zero-full.log` |
| 1b. grid V_NL^0 against vkb, graphene / silicon (s, p) | 4.4e-6 / 4.3e-6 Ry | (projector check) | `zero-full.log`, `t-unit2.log` |
| 1c. dressed term and T(a1) on the grid; bare; flipped sign | 6e-17; 7.3e-2; 0.100 relative | (commutation) | `t-unit2.log` |
| 1d. Hermiticity of the localized composite, none / V_NL^B / V_NL^0 | 1.7e-6 / 1.5e-6 / 3.2e-4 | (operator check) | `iso-{none,gipaw,bare}.log` |
| 2. Hermiticity, full H, q = 10 / 50, nbnd 8 | 2.1e-2 to 2.3e-2 / 1.2e-2 to 1.3e-2 | < 1e-10 (refuse 1e-8) | `q10-full.log`, `q50-full.log` |
| 2. the same, bare control, q = 10 | 2.1e-2 to 3.0e-2 | | `q10-bare.log` |
| 3. 2q levels about E_D, nbnd 16 -> 32, q = 2, 3, 4, 6 | not monotone, rises 2.6e-2 to 4.3e-2 Ry | monotone | `cross-q*.log` |
| 3. q = 25, nbnd 8 -> 16 | rise 8.1e-3, fall 1.3e-2 Ry | monotone | `q25-full.log` |
| 3. zero-mode pair, tr (w_A, w_B), q = 50 | (1.001, 0.999) | (report) | `q50-full.log` |
| 4. max abs(U - P)/E_1 at n = +-1, nbnd 8, q = 10 / 25 / 50 | 0.41 / 0.17 / 0.09 (nbnd 16 at q = 25: 0.05) | falls with q | `peierls.log` |
| 5. crosscheck sets q = 2, 3, 4, 6, nbnd 8, 16, 32 | saved, `route-u-q{2,3,4,6}-full.npz` | saved | `cross-q*.log` |
| 6. cost q = 50, nbnd 8, one core | NSCF 285 s, blocks 620 s, solve 2.8 s, 4.2 GB | (record, at q = 100) | `q50-full.log` |

The level sets `route-u-q{2,3,4,6}-full.npz` are measurements of this refused construction,
not converged upper bounds. `q = 100` was not run (estimated 30 min and 6 GB, `u2.md`).

**What is merged is a measured negative result with its guards.** The default path of
`run_magnetic_ultracell` (`strict=True`) raises on every acceptance case that was run, so what
route U delivers is the refusal, plus the basis-independent pieces (the transport gauge, the
field kernels, the GIPAW grid projectors and their commutation test) that a disentangled basis
will reuse unchanged. The nine level sets `route-u-q*-{local,full}.npz` stay labelled
unconverged; the crosscheck reads them for its item 1 and its route U table only.
