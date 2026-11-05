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
