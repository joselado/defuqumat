# Your own crystal

In this page we compute a crystal that is not in the repository, and for
concreteness we take diamond. We have to choose four things before a number from
a plane-wave code has a meaning:

1. The structure, which is the Bravais lattice, the lattice constant and the atomic positions.
2. The pseudopotential of each element, which also fixes the functional.
3. The plane-wave cutoff.
4. The k-mesh, which says how finely the Brillouin zone is sampled.

The first two are physics. The last two are numerical parameters, so we have to
converge them for the quantity we want.

## The input file

We start with the input file of diamond:

```fortran title="docs/figures/own-crystal_convergence/diamond.in"
--8<-- "own-crystal_convergence/diamond.in"
```

It is a normal `pw.x` input file, and it means the same thing as in Quantum
ESPRESSO. `ibrav = 2` is the fcc lattice, `celldm(1)` is the cubic lattice
constant in bohr, and the two atoms are given in crystal coordinates. An input
file written for `pw.x` runs here without changes, and an input file written here
runs in `pw.x`. We compute its total energy with `diamond.get_scf()`:

```python
from defuqumat import Calculator

diamond = Calculator.from_file("docs/figures/own-crystal_convergence/diamond.in",
                               pseudo_dir="tests/data/pseudo")
print(diamond.get_scf().total_energy)   # in Ry
```

The pseudopotentials are UPF files, and the code reads their names from the
`ATOMIC_SPECIES` card and looks for them in `pseudo_dir`. If you do not give
`pseudo_dir`, the code uses the `pseudo_dir` of the `&control` namelist, and
then the folder of the input file.

## Without an input file

We can also give the crystal as keywords, with lengths in angstrom and energies
in eV, which is how a structure is usually given in a paper or a database. This
is the same diamond, with `Calculator.from_keywords`:

```python
from defuqumat import Calculator

diamond = Calculator.from_keywords(
    lattice="fcc", a=3.567,                       # angstrom
    atoms=[("C", (0, 0, 0)), ("C", (0.25, 0.25, 0.25))],
    pseudos={"C": "C.pz-rrkjus.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=476.0, ecutrho=3808.0,                   # eV
    kmesh=(6, 6, 6), nbands=8,
)
print(diamond.get_scf().total_energy)   # in Ry
```

```
-22.855579403081986
```

This is $2\times10^{-5}$ Ry away from the input file, because 3.567 Å and
476 eV are the rounded values of 6.74 bohr and 35 Ry. With the exact values the
two give the same energy to the last digit. The reason is that the code writes
the keywords as a `pw.x` input and then reads it in the same way as a file. So
the defaults and the errors are the same for both, and
`defuqumat.io.keywords.pw_input_text(...)` gives you this input as text, for
example to run the same crystal in `pw.x`.

The keywords are the following.

- **`lattice`**: `"sc"`, `"fcc"`, `"bcc"`, `"hexagonal"`, `"tetragonal"` or
  `"orthorhombic"`, with the lengths `a`, `b` and `c` in angstrom. For any other
  lattice you give the three lattice vectors in angstrom, as rows.
- **`atoms`**: one `(species, (x, y, z))` per atom. The positions are in
  crystal coordinates, or cartesian in angstrom with `coordinates="angstrom"`.
- **`pseudos`**: the UPF file of each species. The mass is the standard atomic
  weight of the element, unless you give `masses={"C": 12.011}`.
- **`ecut`**, **`ecutrho`**: the two cutoffs in eV. If you do not give
  `ecutrho`, it is four times `ecut`, as in `pw.x`.
- **`kmesh`**, **`kshift`**: the Monkhorst-Pack mesh and its half-step shift,
  or `kmesh="gamma"` for the single point. You always give `kmesh`, as you
  always give `ecut`, because no default is right for every crystal.
- **`smearing`**, **`smearing_type`**: the width in eV for a metal, and
  `"gaussian"`, `"methfessel-paxton"`, `"marzari-vanderbilt"` (or `"cold"`) or
  `"fermi-dirac"`. The short names of `pw.x`, `"mp"`, `"mv"` and `"fd"`,
  also work.
- **`spin`**: `"collinear"` or `"noncollinear"`. The starting moment of an atom
  is a third entry in `atoms`, in Bohr magnetons, for example
  `("Fe", (0, 0, 0), 2.5)`. It is a number for a collinear run and a cartesian
  vector for a noncollinear one.
- **`spin_orbit`**: `True` adds spin-orbit coupling, and needs fully
  relativistic pseudopotentials.
- **`nbands`**, **`functional`**: the number of bands and the functional, as
  `nbnd` and `input_dft` of `pw.x`.
- **`qe`**: any other variable of `pw.x`, with its own name and in its own
  units, for example `qe={"electrons": {"conv_thr": 1e-10}}`. A variable that
  a keyword already sets gives an error, so each value is given only once.

Any other keyword is an option of the calculator, as in `Calculator.from_file`.
With keywords a convergence test is a loop over a number:

```python
for ecut in (272, 408, 544, 680, 816):          # eV
    calc = Calculator.from_keywords(
        lattice="fcc", a=3.567,
        atoms=[("C", (0, 0, 0)), ("C", (0.25, 0.25, 0.25))],
        pseudos={"C": "C.pz-rrkjus.UPF"}, pseudo_dir="tests/data/pseudo",
        ecut=ecut, ecutrho=8 * ecut, kmesh=(6, 6, 6), nbands=8,
    )
    print(ecut, calc.get_scf().total_energy)
```

## Where to get a pseudopotential

The repository has only the pseudopotentials that its tests need, in
`tests/data/pseudo/`. For other elements you can take them from three common
sources:

- The table of Quantum ESPRESSO, at `pseudopotentials.quantum-espresso.org`.
- The PseudoDojo set.
- The SSSP library, which also gives a recommended cutoff for each element.

The code reads norm-conserving, ultrasoft and PAW files. The header of the file
says which functional was used to generate it, and we use this functional unless
you set `input_dft`. Notebook
[05](https://github.com/joselado/defuqumat/blob/master/notebooks/05_gradient_corrections.ipynb)
shows what changes between LDA and PBE.

The functionals we have are LDA (PZ), PBE, revPBE and PBEsol. We write down only
the energy of the functional, and the code takes the potential as its gradient.
Two more functionals work the other way around. `input_dft = 'tb09'` (Tran and
Blaha) and `input_dft = 'bj06'` (Becke and Johnson) are meta-GGA potentials
with no energy behind them. They are used for band gaps, and for silicon
`tb09` gives a gap of 1.13 eV, where LDA gives 0.49 eV.

## The two convergence tests

Let us now converge the two numerical parameters. A plane-wave basis has one
parameter, the cutoff `ecutwfc`, and the integral over the Brillouin zone has one
parameter, the k-mesh. Neither is a property of the crystal, so we cannot look
them up, and we measure them instead.

The input is text, so a convergence test is just a loop over a string
replacement:

```python
text = open("docs/figures/own-crystal_convergence/diamond.in").read()

for ecut in (20, 30, 40, 50, 60):
    new = text.replace("ecutwfc = 35.0, ecutrho = 280.0",
                       f"ecutwfc = {ecut}, ecutrho = {8 * ecut}")
    calc = Calculator.from_text(new, pseudo_dir="tests/data/pseudo")
    print(ecut, calc.get_scf().total_energy)
```

![The total energy of diamond against the cutoff and against the k-mesh.](img/own-crystal_convergence.png#only-light)
![The total energy of diamond against the cutoff and against the k-mesh.](img/own-crystal_convergence-dark.png#only-dark)

In the left panel we show the total energy against `ecutwfc`, measured from the
value at 60 Ry, on the $6\times6\times6$ mesh of the file. In the right panel we
show the total energy against the mesh $n\times n\times n$, measured from the
$10\times10\times10$ mesh, at the 35 Ry of the file. The dashed line is
$10^{-4}$ Ry, which is roughly the accuracy that an energy difference between two
structures needs.

We see that the cutoff converges from above, because a larger basis can only
lower a variational energy. For this carbon dataset it converges slowly, and at
35 Ry the total energy is still $10^{-3}$ Ry above the value at 60 Ry. The k-mesh
converges fast, because diamond is an insulator. A metal has a Fermi surface, so
it needs many more k-points for the same accuracy.

What you have to converge is the quantity that you will report, not only the
total energy. A band gap or a lattice constant is often converged at a lower
cutoff than the total energy, because the errors cancel in a difference.

!!! note "A band gap from the SCF mesh"
    The smallest gap among the k-points of the SCF mesh is not the band gap.
    The conduction band minimum of diamond is at about 85% of the way from
    $\Gamma$ to X, and this point is on none of the meshes. So we compute the
    gap from a band structure along a k-path, as in [Band structure](bands.md).

??? example "Code of this figure (docs/figures/own-crystal_convergence/)"

    ```python title="compute.py"
    --8<-- "own-crystal_convergence/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "own-crystal_convergence/plot.py"
    ```

## Changing the crystal from Python

We can also change the crystal without writing a new input file. Three methods
give a new calculator with a changed crystal, and the old calculator does not
change. When this makes sense, each method passes the converged density to the
new calculator as a starting guess.

- **`calc.with_positions(positions)`**: the same cell with the atoms moved. The
  positions are cartesian, in bohr, with shape `(atoms, 3)`. The density of the
  old geometry is the starting guess, so the new SCF needs fewer iterations.
- **`calc.with_cell(at, positions=None)`**: a deformed cell. `at` holds the three
  lattice vectors in bohr, as rows. The density is not used as a starting
  guess, because a new cell has a new grid.
- **`calc.with_kpoints(kpoints)`**: the same crystal on another k-set, for
  example the full mesh next to its irreducible wedge. The two must give the same
  energy, so this is a check of the symmetry reduction.

## Before you trust a new crystal

Before we trust the numbers of a new crystal, we check four things.

- The functional that the code reports. It comes from the header of the
  pseudopotential file, unless `input_dft` sets it.
- The symmetry that the code found. An atom typed one thousandth of a bohr
  off its site removes symmetry operations, and then the run uses more k-points
  and takes longer, but the result is the same.
- The vacuum, for a slab or a molecule. It must be thick enough that the
  periodic images do not see each other, and this is a convergence test of its
  own.
- The number of bands. An insulator needs the occupied bands and the ones
  you want to look at, and a metal needs bands well above the Fermi level.

!!! note "Limits"
    An input variable that changes the physics and is not implemented gives an
    error with its name, so it is never read and ignored. These are a charged
    cell (`tot_charge`), the corrections for an isolated system
    (`assume_isolated`), two chemical potentials (`twochem`, in `&system` or in
    `&control`), the occupations of an isolated atom (`one_atom_occupations`),
    the constant-cutoff kinetic functional (`qcutz`) and a fixed FFT grid (`nr1`
    to `nr3s`). The four symmetry switches `no_t_rev`, `force_symmorphic`,
    `use_all_frac` and `nosym_evc` also give an error, because each one removes
    symmetry operations that the code would still use. A value equal to the
    default of `pw.x` gives no error, so an ordinary input never meets these.
    `calculation = 'md'` and `calculation = 'vc-md'` give an error, because we
    have no molecular dynamics.

    With `Calculator.from_keywords`, a named lattice without its `b` or `c`
    gives an error. So does a species without a pseudopotential, and an atom
    with a moment but no `spin`. A variable in `qe` that a keyword already sets
    also gives an error. A misspelt keyword gives an error that lists all the
    keywords. The `qe` dictionary reaches only the namelists, so a `HUBBARD`
    card or a k-path still needs an input file.

    The old DFT+U variables of `&system` give an error, as in `pw.x` 7.5. These
    are `lda_plus_u`, `lda_plus_u_kind`, `Hubbard_U`, `Hubbard_U_back`,
    `Hubbard_J0`, `Hubbard_J`, `Hubbard_V`, `U_projection_type`,
    `Hubbard_parameters` and `backall`, and you write the `HUBBARD` card instead.
    The test is on the value, so `lda_plus_u = .false.` and `Hubbard_U = 0.0`
    give no error.

    A lattice given twice gives an error, as in `pw.x`. `ibrav` different from 0
    together with a `CELL_PARAMETERS` card gives an error, and to use the card
    you set `ibrav = 0`. `celldm` together with `A` gives an error. A
    `CELL_PARAMETERS bohr` or `angstrom` card together with `celldm(1)` or `A`
    gives an error, and you either write the card in `alat` units or remove
    `celldm(1)` or `A`.

    Only UPF version 2 files are read. A UPF version 1 file gives an error, and
    you can convert it with `upfconv.x` of Quantum ESPRESSO. Pseudopotentials
    generated with different functionals give an error, unless you set
    `input_dft`, and then the code warns that the functional differs from the
    files.

    A functional that is not implemented gives an error, and the code never
    replaces it with LDA. This covers the hybrid functionals (EXX) and the
    meta-GGAs with an energy (TPSS, SCAN, M06L). `tb09` and `bj06` give an error
    with an ultrasoft dataset, because it has no partial waves to build the
    kinetic energy density inside the spheres, and you use a PAW or
    norm-conserving one instead. They also give an error with a spin spiral and
    with DFT+U (a `HUBBARD` card).
    With `tb09` or `bj06` the code warns that the total energy is not the
    energy of any functional, and forces, stress, phonons and the responses give
    an error.

The notebook for this page is
[25, your own crystal](https://github.com/joselado/defuqumat/blob/master/notebooks/25_your_own_crystal.ipynb).
