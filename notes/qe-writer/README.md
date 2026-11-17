# Writing a pw.x save directory (P180)

The harness behind `PLAN.md` P180, run on this workstation with the vendored `pw.x` 7.5.

- `gorder.py <save dirs>`: `ggen` + `hpsort_eps` against the Miller record of `pw.x`'s files.
- `pwx_reads_ours.py <pw.in> <workdir> [--control]`: `pw.x` from scratch, defuqumat's SCF and
  `write_qe_save`, the coefficients against `pw.x`'s file, and `pw.x` restarted with
  `startingpot = 'file'` from ours; `--control` writes the collinear density as `(up, down)`.
- `run_fixtures.py <names>`: `pw.x` on `inputs/<name>.in`, the fixtures of `tests/data/qe/save/`.
- `compare_seed.py <name>`: each array `get_qe_seed` loads against defuqumat's own converged one.
- `energy_only.py <names>`: defuqumat against `pw.x` total and Hubbard energies (`OPEN.md` Part
  XXXI, the `pt-ldau-*` inputs).
- `time_write.py`: the writer's timing in `PERFORMANCE.md`.

The scripts expect to be run from a work directory that holds `inputs/` (copy this folder).
