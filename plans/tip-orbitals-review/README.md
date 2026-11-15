# Review of `plans/tip-orbitals.md`, 2026-11-15

A fable subagent reviewed the plan against the code. It ran one python process at a time under a
5 GB cap, with no SCF and no pytest. Its scripts are the `.py` files here, copied from the
session scratchpad. They run from the repository root, for example `python3
plans/tip-orbitals-review/q1_fd.py`. The model of `model.py` is a honeycomb sheet of Gaussian s
wells, diagonalised exactly in plane waves, at kappa = 1.15 or 0.84 per bohr. It is not graphene.

`kappa_scan.py` came after the review. It checks the weight of a pseudo-atomic orbital through
the orbital's own decay constant, the question the review left open.

Each item is the conclusion of the review, with its numbers. The plan quotes them where they are
used.

1. **The central claim is true (ran).** A multiplier `i (k+G)_a / kappa` agrees with a central
   difference at `h = 1e-3` to 0.9e-6 to 1.2e-6, with slope 2.00 in `h`, for d/dx, d/dy and
   d/dz. The second derivative d2/dz2 agrees to 5e-7 to 6e-7. The folded path agrees with the
   unfolded one to 6e-15 to 9e-15. The test covered random coefficients, `npol` 1 and 2, a padded
   mask, a hexagonal and a sheared `a3`, and `k3 = 0` and 0.25 (`q1_fd.py`). `(k+G)_x` and
   `(k+G)_y` do not depend on the folded index, because `b3` is normal to `a1` and `a2` (3e-16).
2. **Frame and sign are right.** `k_1(kappa rho) cos theta = (1/kappa) d/dz0 k_0` (argued). The
   ratio `pz/s` lies in [-1.14, -0.90] at 3 to 5 bohr, and `T(s - pz)/T(s + pz) >= 220` at every
   pixel (ran, `model.py`). The direction "towards the tip" is fixed once, so that the wedge
   mirror does not flip pz twice. A non-s tip is refused for `tip=` points on both sides and for
   `axis != exit_axis`.
3. **`C_b^*` is right (read, `green.py:338`), and the hollow site does not pin it (ran).** At the
   hollow both windings are present: `px + i py` and `px - i py` give 9.968e-08 each. With the
   k-set `{K}` and the tip on atom A, `C = (px: 1, py: i)` gives 1.110e-07 and its conjugate
   1.1e-34. The winding pair sums are 3.164e-06 for `m = +1` and 3.4e-33 for `m = -1`. Atom B is
   reversed. All are invariant under a rotation inside the pair. At the hollow, `s` gives 1.4e-36
   and `pz` 1.5e-35.
4. **The wedge (ran and read).** The time-reversed operations, dropped at `transport.py:1511`,
   need `C -> C^*`. `T_-k(C) = T_k(C^*)` holds to 5e-14, and it differs from `T_k(C)` by 0.22 for
   `px + i py` and by 1.9 for `s + 0.5 i px`. The plan samples the shell once and contracts with
   coefficients per point. An l = 2 channel is stored as a symmetric traceless `Q_ij`, because
   Table I is not equally normalised. V9 can deliver 1e-6 and not 1e-12
   (`tests/regression/test_transport.py:62-64`), and it needs a 3-fold axis.
5. **Kappa: the plateau on the tip side (argued).** At `nspin = 2`, `potential[0]` is the up
   channel (`scf/potential.py:772`). The LDA exchange tail is about 1.2 eV at `n = 1e-4` and
   0.3 eV at `1e-6` (estimate). Use `(3 dz^2 - nabla^2)/kappa^2` for dz2. The Helmholtz residue
   with exact eigenvectors (ran, `residue.py`):

   | height (bohr) | residue at 30 Ry | residue at 60 Ry | `\|V\|/kappa^2` |
   |---|---|---|---|
   | 3 | 4.0e-2 | 2.75e-2 | 2.7e-2 |
   | 4 | 5.4e-2 | 6.9e-4 | 8.3e-4 |
   | 5 | 9.9e-2 | 5.6e-4 | 9.2e-6 |

6. **The band window holds (argued and ran).** The derivative amplifies noise at most by
   `(sqrt(ecutwfc)/kappa)^l`. Noise of norm 1e-6 at 5 bohr, measured as noise over `|psi|`:

   | s | px | pz | dz2 |
   |---|---|---|---|
   | 3.3e-5 | 6.4e-5 | 7.9e-5 | 8.4e-4 |

   The zero warning (`transport.py:906`) does not fire at a pixel that is dark by symmetry.
7. **The checks.** V1 needs the `F = 1` path and an A/B on the same cores. V3 tests the fold and
   not the factor. V4 needs orthonormal rows. V5 needs a symmetric set of states. V8 as first
   written could not fail.
8. **`"pxy"` = two incoherent channels.** This is the user's decision, and the review recommended
   the same.
9. **Paths that were missed (read).** These were added to section 9 of the plan:
   - `sample_spectrum` (`workflows/stm.py:444`)
   - `run_ultracell_sts`
   - the QPI reader (`workflows/qpi.py:241`)
   - the argument list of the ultracell, which carries no potential (`workflows/ultracell.py:367`)
   - spin spirals, where one factor on both components is wrong at `k +- q/2`
10. **Size (ran, `size.py`, kappa = 0.84).** A Gaussian s sampled through its Fourier transform,
    divided by `K_0^2`:

    | width (bohr) | 3 bohr | 4 bohr | 5 bohr |
    |---|---|---|---|
    | 0.5 | 0.9925 | 0.9995 | 0.9998 |
    | 1 | 0.889 | 0.978 | 0.997 |
    | 2 | 0.19 | 0.38 | 0.60 |

    The spread over pixels at 3 bohr is 0.9, 8 and 22 per cent. A correlation hides the failure
    of the weight.
11. **Which weight is Bardeen's (ran, `radial.py`, `planc.py`).** In a square well (b) equals (c)
    to six digits. (a) is not Bardeen's element. At kappa = 0.57:

    | orbital | (a) | (b) | (c) |
    |---|---|---|---|
    | O 2p | 3.82 | 0.80 to 0.36 | 1.31 |
    | Pt 6s | 61 | 4.3 | 6.6 |
    | ratio 6s over 2p | 16 | 5.5 | 5.0 |

    The atomic decay constants are 0.815 for O 2p and 0.645 for Pt 6s.
12. **Pb (argued).** 6s and 6p are 9.9 eV apart in Bi (read). With spin-orbit coupling a Pb apex
    needs spinor channels and the `j`-resolved PDOS.

## After the review: the weight through the atomic decay (`kappa_scan.py`, ran 2026-11-15)

Integration by parts gives `(c) = (kappa_atom^2 - kappa^2) (a)` plus boundary terms, which are
zero only in the limit of an infinite radius. So which side carries the pole had to be measured.

(c) is computed directly from `U_t R = (epsilon - T_l) R` on the pseudo-orbital. The table gives
(c) at three radial cutoffs:

| orbital, kappa | cutoff 8 bohr | cutoff 12 bohr | cutoff 16 bohr |
|---|---|---|---|
| O 2p, 0.57 | 1.2997 | 1.3028 | 1.3020 |
| O 2p, 0.40 | 0.8458 | 0.8467 | 0.8466 |
| O 2p, kappa_atom = 0.815 | 2.2188 | 2.2371 | 2.2229 |
| Pt 6s, 0.57 | 6.1914 | 6.4990 | 6.5148 |

So (c) is converged and smooth through `kappa_atom`, with no zero. (a) at kappa = 0.57 grows with
the cutoff: 2.770, 3.439, 3.694 for O 2p, and 20.8, 33.9, 43.7 for Pt 6s. It diverges for kappa
at or above `kappa_atom`.

The identity is approached from below. For O 2p at kappa = 0.57, `(kappa_atom^2 - kappa^2)(a)` is
0.94, 1.17 and 1.25 at the three cutoffs, against (c) = 1.30. At kappa = 0.40 it is 0.74, 0.83 and
0.84, against 0.8466.

So the factor `1/(kappa_atom^2 - kappa^2)` is a pole of the overlap (a), and (c) has no such
pole. The source `U_t R` holds 99.9 per cent of its weight inside 5.95 bohr for O 2p and inside
9.54 bohr for Pt 6s. For Pt 6s, 5 per cent of (c) lies beyond 8 bohr, because the xc potential of
a neutral atom decays as a power of the density.
