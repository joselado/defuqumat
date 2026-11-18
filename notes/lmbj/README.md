# The local mBJ (P182)

The harness behind `PLAN.md` P182, run on this workstation. The scripts carry absolute paths of
the repository and are meant to be copied into a scratch directory and run there.

- `proto.py`: a NumPy prototype of `c(r)` on converged Si and diamond pseudo-densities (the
  fraction of the grid below `n_th`, the local and global `c`).
- `identity.py <variants>`: `si2-tb09.in` under `tb09`, `lmbj` at `sigma = rs = 1000` bohr (the
  bulk-limit identity), `lmbj` and `lmbj` with the `'koller'` set.
- `slab.py <dft> <height>`: graphane at a cell height in bohr; `slab_tight.py <dft> <height>
  <conv_thr> <tag>` the same with the gap also read off the band route.
- `timing.py`: TB09 against the local form, one core, median of three warm runs.
- `tb09_vs_pwx.py`: `si2-tb09` three ways in defuqumat against `pw.x` 7.2's `tb09` (P30's note).
