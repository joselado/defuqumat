# Forces and relaxation

In this page we compute the forces on the atoms, we relax a structure to its
minimum, and we add the van der Waals attraction that a semilocal functional does
not have.

## The force on an atom

The force on atom $I$ is minus the derivative of the total energy with respect to
its position:

$$
\mathbf F_I = -\left.\frac{\partial E}{\partial \boldsymbol\tau_I}\right|_{\psi\ \text{fixed}} .
$$

The wavefunctions stay fixed in the derivative, and we are allowed to do this
because the converged energy is stationary in the wavefunctions. This is the
Hellmann-Feynman theorem. We take this derivative of the total energy with
automatic differentiation, so the terms from the basis that moves with the atoms,
and from the augmentation charge of an ultrasoft or PAW dataset, come out of the
same derivative. A force says where the atoms want to go, and a relaxation
follows it to the equilibrium structure, which diffraction measures. The
derivatives of the forces are the force constants, which give the phonons.

An atom on a high-symmetry site has zero force, so for an example we move the
second atom of silicon by 0.02 alat along $x$. We compute the forces with
`calc.get_forces()`:

```python
from defuqumat import Calculator

calc = Calculator.from_file("docs/figures/forces_relaxation/si-displaced.in",
                            pseudo_dir="tests/data/pseudo")
forces = calc.get_forces()
print(forces.forces.round(6))      # (atoms, 3), Ry/bohr
```

```
[[ 0.060397 -0.        0.      ]
 [-0.060397  0.       -0.      ]]
```

There was no ground state yet, so the code ran the SCF first and said so on
stderr, because a force is a derivative at the ground state. We see that the two
forces are equal and opposite, and nothing in the code imposes this. The total
force on a crystal is zero, because a rigid translation does not change the
energy. The force on the second atom points along $-x$, back towards its site.
The forces are in Ry/bohr, as in `pw.x`.

A noncollinear or spin-orbit run has forces too, with the same call. The energy is
then written for two-component spinors, so the nonlocal term takes a $2\times2$
matrix in spin space and nothing else in the derivative changes.

## Relaxing the structure

We now let the atoms move. `get_relax()` moves them downhill with the BFGS method
until the largest force is below `forc_conv_thr` and the energy change is below
`etot_conv_thr`. These are the keywords of the `&control` namelist, $10^{-3}$
Ry/bohr and $10^{-4}$ Ry if you do not give them, and you can also pass them as
keywords. The symmetry of the crystal is found at the start and kept during the
relaxation.

```python
relax = calc.get_relax()

for step in relax.steps:
    print(f"step {step.index}   E = {step.total_energy:.8f} Ry   "
          f"largest force {step.max_force:.6f} Ry/bohr")
r = relax.positions
print((r[1] - r[0]) / calc.system.cell.alat)    # the bond, in alat
```

```
step 1   E = -15.78740371 Ry   largest force 0.060397 Ry/bohr
step 2   E = -15.79256152 Ry   largest force 0.024848 Ry/bohr
step 3   E = -15.79359588 Ry   largest force 0.000369 Ry/bohr
step 4   E = -15.79359610 Ry   largest force 0.000002 Ry/bohr
[0.25000073 0.25       0.25      ]
```

The bond is back at $(1/4, 1/4, 1/4)$, the diamond structure. Both atoms moved,
by half of the displacement each, because the forces on them are equal and
opposite. `relax.steps` holds the positions, the energy and the largest force of
each step, and `relax.converged` says whether the thresholds were reached.
`calc.relaxed()` returns a new calculator at the relaxed structure, with its SCF,
so you can go on with any other `get_*` method there. If you want to fix an atom,
set its `if_pos` flags to 0 in `ATOMIC_POSITIONS`, as in `pw.x`.

![The energy and the largest force at each step of the relaxation of silicon.](img/forces_relaxation.png#only-light)
![The energy and the largest force at each step of the relaxation of silicon.](img/forces_relaxation-dark.png#only-dark)

The left panel shows the energy at each step, and the right panel the largest
force. The energy falls by 6.2 mRy, and the force falls from 0.06 to
$2\times10^{-6}$ Ry/bohr in four steps. At step 3 the force is already below
`forc_conv_thr`, but the energy changed by 1 mRy from step 2, which is more than
`etot_conv_thr`. Step 4 changes the energy by $2\times10^{-7}$ Ry, so both
criteria hold and the relaxation stops. Near a minimum the energy is flat while
the force is still linear in the distance, so the force is the more sensitive
measure of how close the structure is to the minimum.

??? example "Code of this figure (docs/figures/forces_relaxation/)"

    ```python title="compute.py"
    --8<-- "forces_relaxation/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "forces_relaxation/plot.py"
    ```

## Van der Waals: the D2 correction of Grimme

Let us now add the van der Waals attraction. LDA and GGA see only the density
where it is, while the van der Waals attraction comes from correlated charge
fluctuations on two atoms whose densities do not overlap, so a semilocal
functional misses it. Layered crystals, molecular crystals and molecules on
surfaces are held together by it. The D2 correction of Grimme adds it as a pair
sum over the atoms:

$$
E_{\rm disp} = -\frac{s_6}{2} \sum_{i \neq j} \sum_{\mathbf L}
   \frac{C_6^{ij}}{|\mathbf r_{ij} + \mathbf L|^6}\,
   f_{\rm damp}\big(|\mathbf r_{ij} + \mathbf L|\big),
\qquad
f_{\rm damp}(R) = \frac{1}{1 + e^{-d\,(R/R_{\rm vdW} - 1)}} .
$$

Here $C_6^{ij} = \sqrt{C_6^i C_6^j}$ comes from a table of atomic values, and
$\mathbf L$ runs over the lattice. The damping function switches the $R^{-6}$
tail off at short distance, where the functional already describes the
interaction. You switch it on with `vdw_corr = 'grimme-d2'` in `&system`, and
`london_s6` and `london_rcut` set $s_6$ and the cutoff of the sum, as in `pw.x`.
Here we take bilayer graphene:

```python
bilayer = Calculator.from_file("docs/figures/forces_vdw/bilayer.in",
                               pseudo_dir="tests/data/pseudo")
scf = bilayer.get_scf()
print(f"total energy {scf.total_energy:.8f} Ry")
print(f"D2 energy    {scf.energy_terms['dispersion']:.8f} Ry")
```

```
total energy -45.10439956 Ry
D2 energy    -0.02305929 Ry
```

The D2 energy depends only on the positions of the atoms and on the cell, so it
does not change the density, the bands or the potential. The run with D2 has the
same density as the run without it, and the two total energies differ by exactly
$E_{\rm disp}$. The correction enters the forces, the stress and so the
relaxations, and the elastic constants.

![The binding curve of bilayer graphene with PBE alone, the D2 pair sum, and PBE plus D2.](img/forces_vdw.png#only-light)
![The binding curve of bilayer graphene with PBE alone, the D2 pair sum, and PBE plus D2.](img/forces_vdw-dark.png#only-dark)

The figure shows the energy of bilayer graphene against the distance between the
two layers, measured from its value at 10 bohr. PBE alone is repulsive below 8
bohr, and beyond 8 bohr it is flat to 0.3 meV per atom, so PBE does not bind the
layers. The D2 pair sum is attractive at every distance and has no minimum by
itself. Their sum has a minimum at 6.09 bohr (3.22 Å), from a parabola through
the three lowest points, with a depth of 17.6 meV per atom, and the measured
distance in graphite is 3.35 Å. The depth is a lower bound, because the cell is
20 bohr tall. At 10 bohr each layer is as close to the periodic image of the
other layer as to the other layer itself, so the energy at 10 bohr still has
some binding in it.

??? example "Code of this figure (docs/figures/forces_vdw/)"

    ```python title="compute.py"
    --8<-- "forces_vdw/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "forces_vdw/plot.py"
    ```

!!! note "Limits"
    `get_forces(method="analytic")` computes the force from the expressions of
    `pw.x`, term by term, as a check. It gives an error for a noncollinear or
    spin-orbit run, because these expressions have no spinor form, and the
    default method works there. The same holds for a relaxation with
    `force_method="analytic"`. The stress has no analytic method at all, and
    `get_stress(method="analytic")` gives an error on every run, collinear or
    not, because the nonlocal part of the stress is not written out term by
    term. The default method gives the whole stress, spinors included.

    `get_forces(method="analytic")` also gives an error for an ultrasoft or PAW
    cell whose table of the augmentation charge would be larger than 2 GiB,
    because the code then keeps only a radial table and builds the charge a piece
    at a time, while the analytic force needs the whole table. The environment
    variable `DEFUQUMAT_AUG_MAX_BYTES` moves the 2 GiB limit. The default method
    works on such a cell and gives the same force.

    `get_forces()` gives an error for a spin spiral, because its two spinor
    components live on different plane-wave spheres and the nonlocal term then
    needs the projectors of both. So `get_relax()` gives an error for a spiral
    too. `get_stress()` also gives an error for a spiral, because `spiral_q` is
    given in units of the reciprocal lattice vectors, so a strain turns the
    spiral as well. What we compute for a spiral instead is the derivative of
    the energy with respect to its wavevector, on the page about spin spirals.

    A noncollinear or spin-orbit run has forces, a stress and a relaxation, but
    the elastic constants and the electrostriction give an error, because they
    take a third derivative and none has been checked with spinors.

    A potential-only meta-GGA (`tb09`, `bj06`) has no energy functional, so it
    has no forces and no relaxation.

    We implement only D2, and D3, Tkatchenko-Scheffler, MBD and XDM give an
    error. `pw.x` runs them with a warning and no correction at all.

The notebooks for this page are
[09, forces and relaxation](https://github.com/joselado/defuqumat/blob/master/notebooks/09_forces_and_relaxation.ipynb)
and
[22, van der Waals](https://github.com/joselado/defuqumat/blob/master/notebooks/22_van_der_waals.ipynb).
