# Your own crystal

On this page we compute a crystal that is not in the repository. The example is
diamond. Four things must be chosen before a number from a plane-wave code has a
meaning:

1. the structure: the Bravais lattice, the lattice constant and the atomic positions;
2. the pseudopotential of each element, which also fixes the functional;
3. the plane-wave cutoff;
4. the k-mesh, which says how finely the Brillouin zone is sampled.

The first two are physics. The last two are numerical parameters, and you must
converge them for the quantity you want.

## The input file

This is the input file of diamond:

```fortran title="docs/figures/own-crystal_convergence/diamond.in"
--8<-- "own-crystal_convergence/diamond.in"
```

It is a normal `pw.x` input file, and it means the same thing as in Quantum
ESPRESSO. `ibrav = 2` is the fcc lattice, and `celldm(1)` is the cubic lattice
constant in bohr. The two atoms are given in crystal coordinates. An input file
written for `pw.x` runs here without changes, and an input file written here runs
in `pw.x`.

```python
from defuqumat import Calculator

diamond = Calculator.from_file("docs/figures/own-crystal_convergence/diamond.in",
                               pseudo_dir="tests/data/pseudo")
print(diamond.get_scf().total_energy)   # in Ry
```

The pseudopotentials are UPF files. The code reads their names from the
`ATOMIC_SPECIES` card and looks for them in `pseudo_dir`. If you do not give
`pseudo_dir`, the code uses the `pseudo_dir` of the `&control` namelist, and
then the folder of the input file.

## Where to get a pseudopotential

The repository has only the pseudopotentials that its tests need, in
`tests/data/pseudo/`. For other elements, there are three common sources:

- the table of Quantum ESPRESSO, at `pseudopotentials.quantum-espresso.org`;
- the PseudoDojo set;
- the SSSP library, which also gives a recommended cutoff for each element.

The code reads norm-conserving, ultrasoft and PAW files. The header of the file
says which functional was used to generate it, and the code uses this functional
unless you set `input_dft`. Notebook
[05](https://github.com/joselado/defuqumat/blob/master/notebooks/05_gradient_corrections.ipynb)
shows what changes between LDA and PBE.

## The two convergence tests

A plane-wave basis has one parameter, the cutoff `ecutwfc`. The integral over the
Brillouin zone has one parameter, the k-mesh. Neither is a property of the
crystal, so you cannot look them up. You measure them.

The input is text, so a convergence test is a loop over a string replacement:

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

The left panel is the total energy against `ecutwfc`, measured from the value at
60 Ry, on the $6\times6\times6$ mesh of the file. The right panel is the total
energy against the mesh $n\times n\times n$, measured from the $10\times10\times10$
mesh, at the 35 Ry of the file. The dashed line is $10^{-4}$ Ry. This is roughly
the accuracy that an energy difference between two structures needs.

The cutoff converges from above, because a larger basis can only lower a
variational energy. For this carbon dataset it converges slowly: at 35 Ry the
total energy is still $10^{-3}$ Ry above the value at 60 Ry. The k-mesh
converges fast, because diamond is an insulator. A metal has a Fermi surface,
and it needs many more k-points for the same accuracy.

Converge the quantity that you will report, not only the total energy. A band gap
or a lattice constant is often converged at a lower cutoff than the total energy,
because the errors cancel in a difference.

!!! note "A band gap from the SCF mesh"
    The smallest gap among the k-points of the SCF mesh is not the band gap.
    The conduction band minimum of diamond is at about 85% of the way from
    $\Gamma$ to X, and this point is on none of the meshes. Compute the gap from a
    band structure along a k-path, as in [Band structure](bands.md).

??? example "Code of this figure (docs/figures/own-crystal_convergence/)"

    ```python title="compute.py"
    --8<-- "own-crystal_convergence/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "own-crystal_convergence/plot.py"
    ```

## Changing the crystal from Python

Three methods give a new calculator, with a changed crystal. The old calculator
does not change. Each method passes the converged density to the new calculator
as a starting guess, when this makes sense.

- **`calc.with_positions(positions)`**: the same cell with the atoms moved. The
  positions are cartesian, in bohr, with shape `(atoms, 3)`. The density of the
  old geometry is the starting guess, so the new SCF needs fewer iterations.
- **`calc.with_cell(at, positions=None)`**: a deformed cell. `at` holds the three
  lattice vectors in bohr, as rows. The density does not cross, because a new
  cell has a new grid.
- **`calc.with_kpoints(kpoints)`**: the same crystal on another k-set, for
  example the full mesh next to its irreducible wedge. The two must give the same
  energy, and this is a check of the symmetry reduction.

## Before you trust a new crystal

- Check the functional that the code reports. It comes from the header of the
  pseudopotential file, unless `input_dft` sets it.
- Check the symmetry that the code found. An atom typed one thousandth of a
  bohr off its site removes symmetry operations. Then the run uses more k-points
  and takes longer, but the result is the same.
- For a slab or a molecule, check that the vacuum is thick enough. The periodic
  images must not see each other. This is a convergence test of its own.
- Check the number of bands. An insulator needs the occupied bands and the ones
  you want to look at. A metal needs bands well above the Fermi level.

!!! note "Limits"
    An input variable that changes the physics and is not implemented gives an
    error with its name. It is never read and ignored. Examples are
    `tot_charge`, `assume_isolated`, `twochem` and a fixed FFT grid
    (`nr1` to `nr3s`). The old DFT+U variables of `&system` (`lda_plus_u`,
    `Hubbard_U`, and the others) also give an error: use the `HUBBARD` card.
    `ibrav` different from 0 together with a `CELL_PARAMETERS` card gives an
    error, and so does `celldm` together with `A`. Set `ibrav = 0` to use the
    card.

The notebook for this page is
[25, your own crystal](https://github.com/joselado/defuqumat/blob/master/notebooks/25_your_own_crystal.ipynb).
