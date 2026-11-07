# Route U step 1: the acceptance table

The full record, with the reasons and the decisions, is `plans/orbital-field/u1.md`. Logs are
on D22 in `/l/ladovj1/review/mag/u1/`. Status: **refused**: the bare hybrid basis of an
entangled band set is not localized.

| item | measured | design | log |
|---|---|---|---|
| 0. zero flux, levels against eps(k) | 5.3e-15 Ry | (exactness check) | `zero1.log` |
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
