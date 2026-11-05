# Wannier90 references

Each directory holds the inputs of one reference run and its outputs. The run is
QE 7.5 `pw.x` (scf, then nscf on the full mesh with `nosym`), `wannier90.x -pp`,
`pw2wannier90.x`, then Wannier90 3.1.0 `wannier90.x`, all on one core. The
`.win` gives the cell in bohr to twelve digits. An angstrom cell to six
decimals moves the spread by 1e-6 A^2. `pseudo_dir` is `../../pseudo/`, so
`pw.x` is run from inside the directory.

- `diamond/`: C.pbe-hgh, ecutwfc 25, scf 4x4x4, four bond-centred s orbitals
  on the 4x4x4 mesh. Holds the `.mmn`, `.amn` and `.eig` of `pw2wannier90.x`.
- `si/`: Si.pz-vbc, ecutwfc 20, 12 bands and 8 sp3 orbitals, frozen window
  below 6.4 eV. Holds the matrices (gzipped), and `si_band.dat` and
  `si_band.kpt` along L-G-X.
- `si-us/`, `si-paw/`: the ultrasoft and PAW `psl` 0.1 datasets, ecutwfc 30,
  ecutrho 240, frozen window below 6.6 eV (the top of the valence band is
  6.404 and 6.405 eV there). Inputs and `.wout` only.
- `scdm/`: `pw2wannier90.x` with `scdm_proj = .true.`, then `wannier90.x` with
  `dis_num_iter = 0`, `num_iter = 3000`, `conv_window = 3`, `conv_tol = 1e-10`, on the
  4x4x4 mesh. `diamond-iso` is the `diamond/` cell, four isolated bands. `si-iso-g16`
  (four valence bands, `exclude_bands = 5-16`) and `si-erfc16-g16` (16 bands to 8, erfc,
  `scdm_mu = -6.302716706994456`, `scdm_sigma = 6.509102310175368`) are the `si-pdwf/`
  cell with the nscf on a 16x16x16 grid (`nr1 = nr2 = nr3 = 16`), the grid of our SCF.
  The `nosym` nscf alone would use 15x15x15. Each holds `pw2wan.in`, `w.win`, the `.wout`
  (gzipped for the long one) and `pw2wan.out`, which prints the selected points. The
  `pseudo_dir` in the nscf inputs is the absolute path of the D22 run.
