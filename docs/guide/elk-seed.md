# Reading and writing an all-electron ground state

In this page we compute a ground state that starts from the converged density of
Elk, an all-electron LAPW code, and we compare the two densities point by point.
At the end we go the other way, and we write our own PAW ground state as a state
that Elk continues from.
We take simple cubic hydrogen at $a = 3.0$ bohr, which has no core electrons, so the all-electron density and the valence density count the same
single electron.

## Reading the all-electron density

An all-electron code and a pseudopotential code do not converge to the same
density. The all-electron density has every electron of the atom and the cusp at
the nucleus, while the pseudopotential density is a smooth valence density, which
is built to agree with the true one only outside a cutoff radius around each atom.
Elk stores its density in two parts. Inside a sphere of radius $r_a$ around each
atom, the muffin-tin sphere, it is an expansion in real spherical harmonics on a
radial mesh, and outside all spheres it is a Fourier series:

$$
\rho(\mathbf r) =
\begin{cases}
  \sum_{lm} f_{lm}(r)\,R_{lm}(\hat{\mathbf r}), & |\mathbf r - \mathbf R_a| < r_a,\\[4pt]
  \sum_{\mathbf G} \tilde\rho(\mathbf G)\, e^{i\mathbf G\cdot\mathbf r}, & \text{otherwise.}
\end{cases}
$$

To put this density on our grid, the code asks for each grid point which region it
is in, and then it interpolates the radial functions or sums the Fourier series.
The points inside a sphere are evaluated one by one, so the cusp is not cut by a
Fourier truncation on the grid. The run itself cuts it at its first iteration, as
we will see in the next section.

We compute it with `calc.get_elk_seed()`:

```python
from defuqumat import Calculator

calc = Calculator.from_file("docs/figures/elk-seed_hydrogen/h.in",
                            pseudo_dir="tests/data/pseudo")
report = []
seed = calc.get_elk_seed("docs/figures/elk-seed_hydrogen/elk", report=report)
print(seed.shape)
print(f"charge before scaling   {report[0].integral:.6f}")
print(f"charge in the sphere    {report[0].muffin_tin:.6f}")
```

```
(1, 15, 15, 15)
charge before scaling   1.000298
charge in the sphere    0.612576
```

The argument of `get_elk_seed` is the folder of an Elk run, with `STATE.OUT` and
`GEOMETRY.OUT` side by side. `STATE.OUT` has the density but not the lattice
vectors or the atomic positions, so the code reads them from `GEOMETRY.OUT`, which
is also the only file written in the frame Elk really used. The `elk.in` that made
this run is in the same folder. The seed has the shape (spin, grid), here one
channel on the $15^3$ grid of the run. The Elk density integrates to 1.000298
electrons on our grid, and the code scales it to the valence charge of the run,
here exactly 1, unless you pass `renormalise=False`. Of this one electron, 0.612576
is inside the muffin-tin sphere of the hydrogen atom. The list you pass as `report`
receives these numbers, and you can leave it out.

## Starting the SCF from it

We now give the Elk density to `get_scf` as the starting density, and we compare
with a normal run, which starts from a sum of atomic charges:

```python
seeded = calc.get_scf(starting_density=seed)
plain = Calculator.from_file("docs/figures/elk-seed_hydrogen/h.in",
                             pseudo_dir="tests/data/pseudo").get_scf()
print(f"from the Elk density   {seeded.total_energy:.12f} Ry  {seeded.iterations} iterations")
print(f"from atomic charges    {plain.total_energy:.12f} Ry  {plain.iterations} iterations")
```

```
from the Elk density   -1.080181442650 Ry  4 iterations
from atomic charges    -1.080181442649 Ry  4 iterations
```

Both runs take four iterations and end at the same total energy, to $10^{-12}$ Ry,
so the SCF of the pseudopotential code still decides the answer and the seed only
decides where it starts. Here the seed saves no iterations at all. For a hydrogen
atom the sum of atomic charges is already close to the answer, and the part where
the Elk density is different from it is the cusp, which is in the region where the
pseudopotential density is built not to follow it. So what `get_elk_seed` gives you
is the bridge itself: an all-electron ground state, computed in a different basis,
on the grid of this code, where you can compare it with the pseudopotential
density point by point.

The run also prints a warning that 2 per cent of the starting density is not
symmetric. This is the part of the cusp at wave vectors beyond the cutoff of the
run, which the first iteration drops. The seed itself has the full symmetry of the
cube, so you can ignore the warning here.

## Where the two densities differ

We now compare the two densities at the atom:

```python
print(f"at the atom, Elk        {float(seed[0, 0, 0, 0]):.4f} e/bohr^3")
print(f"at the atom, converged  {float(seeded.density[0, 0, 0, 0]):.4f} e/bohr^3")
```

```
at the atom, Elk        0.2858 e/bohr^3
at the atom, converged  0.1762 e/bohr^3
```

At the nucleus the converged pseudopotential density is 38 per cent lower than the
all-electron one, which is the cusp that the pseudopotential removes on purpose.

![The Elk density and the converged pseudopotential density of hydrogen on the grid points along x and along the diagonal of the cube.](img/elk-seed_hydrogen.png#only-light)
![The Elk density and the converged pseudopotential density of hydrogen on the grid points along x and along the diagonal of the cube.](img/elk-seed_hydrogen-dark.png#only-dark)

The figure shows the two densities on the grid points of the run, from the atom to
the middle of the cell, along $x$ (left) and along the diagonal of the cube
(right). We see that the all-electron density falls steeply from the atom while the
pseudopotential density is flat there. At 0.2 bohr along $x$ they are still 15 per
cent apart, and from 0.4 bohr outwards they agree to within 3 per cent on both
lines, the largest difference being 2.8 per cent at 0.6 bohr along $x$. Both runs use a
$4\times4\times4$ k-mesh, which is not converged for this metal, and the Elk run
uses its default smearing, LDA parametrization and basis, not the Gaussian smearing
of 0.02 Ry and the PZ LDA of `h.in`. So this 3 per cent is an upper estimate, which
includes the error of the pseudopotential but also these differences in the
settings. Note that the grid has a spacing of 0.2 bohr along $x$
and 0.35 bohr along the diagonal, so the cusp is seen at one grid point only, and
the lines between the points are only a guide to the eye.

??? example "Code of this figure (docs/figures/elk-seed_hydrogen/)"

    ```python title="compute.py"
    --8<-- "elk-seed_hydrogen/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "elk-seed_hydrogen/plot.py"
    ```

## Writing our state for Elk

We now go the other way. Elk starts a calculation with `task = 1` from its file
`STATE.OUT`, and it starts from the potential in that file, not from the density.
So a file for Elk needs the all-electron density and the all-electron Kohn-Sham
potential, and a pseudopotential code has neither. With PAW we can still build
both. Inside each PAW sphere the all-electron density is

$$n(\mathbf r) = \tilde n(\mathbf r) + \sum_{ij} \rho_{ij}
\left(\phi_i\phi_j - \tilde\phi_i\tilde\phi_j - \hat Q_{ij}\right) + n_{\rm core}(\mathbf r),$$

where $\tilde n$ is the density on our grid, $\rho_{ij}$ are the PAW occupations,
$\phi_i$ and $\tilde\phi_i$ are the all-electron and pseudo partial waves of the
dataset, $\hat Q_{ij}$ is the compensation charge, and $n_{\rm core}$ is the frozen
all-electron core of the dataset. The code evaluates this density on the meshes of
Elk, which are a logarithmic radial mesh with real spherical harmonics in each
muffin tin and a grid outside. Then it computes the potential of Elk from it with
the same steps as Elk: the Coulomb potential by the method of Weinert, the
exchange-correlation potential on the angular points of Elk, and the average over
the symmetry group of the crystal. The meshes and the parameters come from a
template, which is any `STATE.OUT` of an Elk run of the same crystal, with its
`INFO.OUT` and `GEOMETRY.OUT`. The cheapest template is one loop of Elk, with
`maxscl 1` and `nwrite 1` in `elk.in`. Without `nwrite 1`, a run of one loop does
not write `STATE.OUT`.

We compute it with `calc.write_elk_state()`. Here we take silicon with the PAW
pseudopotential `Si.pz-n-kjpaw_psl.0.1` and Elk's own lattice vectors, and the
small Elk template of the repository:

```python
import numpy as np
from defuqumat import Calculator
from defuqumat.io.elk import ElkState

text = """&control
 calculation='scf'
/
&system
 ibrav=0, nat=2, ntyp=1, ecutwfc=15.0, ecutrho=120.0
/
&electrons
 conv_thr=1d-8
/
CELL_PARAMETERS bohr
 0.0 5.10 5.10
 5.10 0.0 5.10
 5.10 5.10 0.0
ATOMIC_SPECIES
 Si 28.086 Si.pz-n-kjpaw_psl.0.1.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""
calc = Calculator.from_text(text, pseudo_dir="tests/data/pseudo")
calc.write_elk_state("elk_run", template="tests/data/elk/si2_template")

import shutil
shutil.copy("tests/data/elk/si2_template/GEOMETRY.OUT", "elk_run")
state = ElkState.read("elk_run")
r = state.rsp[0][:state.nrmt[0]]
print(f"electrons in each muffin tin: {state.muffin_tin_charges()[0]:.4f}")
print(f"l = 3 component at r = 1.8 bohr: {state.rhomt[10, np.searchsorted(r, 1.8), 0]:.4f}")
```

```
electrons in each muffin tin: 12.3391
l = 3 component at r = 1.8 bohr: -0.0550
```

This example uses an input file and not keywords, because the lattice vectors must
be exactly the ones of Elk, and the keyword `fcc` gives the vectors of `pw.x`. Each
muffin tin holds 12.34 electrons. This is the 10 core electrons of silicon and
the part of the valence charge inside the sphere. The component with $l = 3$ is
the tetrahedral bond. A superposition of free atoms has no such component, and
Elk starts from such a superposition when it starts from scratch. Then Elk runs
with `task = 1` in the folder `elk_run`.

How much does this help Elk? We measured it on the full Si2 run of Elk with the
Perdew-Zunger LDA. From our state Elk starts with a change of the potential of
0.0050 Ha in its first loop, against 0.083 Ha from scratch, and it converges in
16 loops instead of 19, at the same energy. For an O2 molecule as a collinear
magnet, and as a noncollinear magnet with its moment tilted, Elk needs 14 loops
instead of 17. The gain is limited by the mixer of Elk, because even from its own
converged state Elk needs 8 loops at the same threshold. With PBE there is no
gain. The PBE potential very close to the nucleus depends on the exact shape of
the Dirac core of Elk, which a scalar relativistic PAW core does not have. So for
PBE the code takes the density of the template within 0.01 bohr of each nucleus,
and Elk then needs as many loops as from scratch. For the O2 molecule as a magnet
with PBE, Elk needs 31 loops from our state and 32 from scratch, and the code
gives a warning that the state is not a faster start. We made these comparisons
with the official release of Elk 11.0.2. On one core, writing the state
takes about 2.5 s, and computing the potential of Elk takes 0.45 s, where Elk
itself needs 0.016 s.

!!! note "Limits"
    The Elk state must be for the same cell as the run, because a density cannot
    be moved from one lattice to another. The folder must contain `GEOMETRY.OUT`
    from the same Elk run as `STATE.OUT`. A folder without `GEOMETRY.OUT` gives an
    error. A `GEOMETRY.OUT` with a different number of species, or a different
    number of atoms in a species, also gives an error. We do not check the atomic
    positions against `STATE.OUT`, so you must make sure that both files come from
    the same run. A `STATE.OUT` written by Elk older than version 2.0.0 gives an
    error.

    A collinear spin-polarized Elk state can start a run with `nspin=2`. The code
    moves the charge and the magnetization of Elk to the grid. Both are scaled by
    the same factor, so the polarization at each point is the one of Elk. A
    noncollinear Elk state gives an error. This includes a spin spiral, because
    `STATE.OUT` does not say that the run was a spiral. An Elk state with a fixed
    spin moment also gives an error. A spin-polarized Elk state with an `nspin=1`
    run gives an error, because the moment would be lost. An unpolarized Elk
    state with an `nspin=2` run gives an error, because the seed would have no
    moment. A noncollinear run gives an error with any Elk state. A DFT+U state
    works as a seed. `STATE.OUT` does not contain the occupation matrix of Elk, so
    only the density is used, and the code gives a warning. The Hubbard
    occupations then start from `starting_ns` of the run, as in a normal DFT+U
    run. A spin-polarized DFT+U state gives an error, because we have not checked
    the two together. A state with fixed tensor moments gives an error, because
    the code does not have this constraint.

    An element with core electrons gives an error. The muffin-tin density of Elk
    contains the core states, and nothing in `STATE.OUT` separates them from the
    valence charge. So when the Elk density integrates to a charge that is further
    from the valence charge of the run than half of it (or than 0.5 electrons, if
    that is larger), the code gives an error and does not scale it. Hydrogen, with
    no core, is the case this is made for.

    The Elk density can only be a starting density. You cannot use it as a fixed
    density with no SCF on top, because the two converged densities are different
    functions near every nucleus.

    Writing an Elk state needs a PAW pseudopotential for every species, because
    only PAW has the all-electron partial waves to build the density inside the
    spheres. Any other pseudopotential gives an error. The PAW correction must end
    inside the muffin tin of Elk, so a PAW radius larger than the muffin-tin
    radius gives an error. For example, O2 at its bond length needs muffin tins
    smaller than the PAW radius of `O.pz-kjpaw`. The template must have the same
    lattice vectors, not only the same lattice, the same atoms up to one
    translation, and the same spin setting: unpolarized, collinear for `nspin=2`,
    or noncollinear for a magnetic noncollinear run. The code computes the
    potential of Elk for the LDA (`xctype` 2 and 3) and for the PBE family of Elk
    (`xctype` 20, 21 and 22). A noncollinear state with PBE, a functional of
    libxc, a meta-GGA, a finite nucleus and an external magnetic field give an
    error. We have not run a case with spin-orbit coupling through Elk. With
    k-point pools (`DEFUQUMAT_POOLS`) `write_elk_state` gives an error, because
    every pool would write the same `STATE.OUT` at the same time.

The notebook for this page is
[42, starting from an all-electron ground state](https://github.com/joselado/defuqumat/blob/master/notebooks/42_all_electron_start.ipynb).
