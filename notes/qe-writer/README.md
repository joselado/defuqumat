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

On Triton (2026-11-18), with the QE 7.2 module (libxc, `.dat` files) or the conda `pw.x` 7.5
(HDF5), `pw.x` serial through `srun`; each script names its partition, cores and memory:

- `triton/ni_soc_roundtrip.py` with `ni_soc_roundtrip.sbatch` (conda 7.5) and
  `ni_soc_roundtrip_qe72.sbatch` (7.2): Ni fcc, spin-orbit PAW with a moment, both directions.
- `triton/regimes_qe72.sbatch`: `pwx_reads_ours.py` on the six inputs `pw.x` 7.5 here could not
  run or had not run, and Fe kind 1 again under `noncollinear_gga = 'kubler'` (task 6, through
  `--ours=old->new`, which edits defuqumat's input only). `PW_X` names the `pw.x` and `TASKSET=0`
  drops the pin.
- `triton/tb09_swap.py` with `tb09_swap.sbatch`: `pw.x` from its own TB09 state with only
  `ekin-density.dat` replaced (its own, ours, ours in a wrong layout).
