# Collinear and noncollinear magnetism

In this page we compute magnetic ground states. We start with a collinear magnet
with two spin channels, then we compute a noncollinear texture with a moment on
each atom, and finally we see how to go from one regime to the other.

## Two spin channels

We start with the simplest case, a collinear magnet. With `nspin = 2` the density
has two parts, $n_\uparrow$ and $n_\downarrow$, and the moment is the integral of
their difference:

$$
n_\sigma(\mathbf r) = \sum_{n\mathbf k} f_{n\mathbf k\sigma}\,
   |\psi_{n\mathbf k\sigma}(\mathbf r)|^2,
\qquad
M = \int \big(n_\uparrow - n_\downarrow\big)\, d\mathbf r .
$$

The two channels see different exchange-correlation potentials,
$v_{xc}^\sigma = \delta E_{xc}/\delta n_\sigma$, so the bands of the two channels
come out at different energies. This shift is the exchange splitting, and it is
the whole of band magnetism. There is one Fermi level for both channels, unless
`tot_magnetization` fixes the moment. `starting_magnetization` only seeds the
first density, with a value between $-1$ and 1 for each species, and the moment
is a result of the SCF. The moment per cell is what a magnetometer measures, and
neutron diffraction and XMCD see the moment on each atom. Here we take nickel:

```python
import numpy as np
from defuqumat import Calculator

nickel = Calculator.from_file("docs/figures/magnetism_nickel_dos/ni.in",
                              pseudo_dir="tests/data/pseudo")
scf = nickel.get_scf()
print(f"total energy {scf.total_energy:.8f} Ry")
print(f"magnetization {scf.magnetization:.4f} mu_B per cell")
print(scf.eigenvalues_by_spin.shape)      # (spin, k-points, bands)
```

```
total energy -85.72339901 Ry
magnetization 0.7280 mu_B per cell
(2, 10, 9)
```

`scf.magnetization` is the moment of the cell in Bohr magnetons. The eigenvalues
and the occupations have a first axis for the spin channel in
`scf.eigenvalues_by_spin` and `scf.occupations_by_spin`. The density
`scf.density` has the shape (2, grid), with the two channels, and `get_bands` and
`get_dos` also give both channels.

![The bands and the DOS of fcc nickel, for the majority and the minority channel.](img/magnetism_nickel_dos.png#only-light)
![The bands and the DOS of fcc nickel, for the majority and the minority channel.](img/magnetism_nickel_dos-dark.png#only-dark)

The figure shows fcc nickel in LSDA. We see that the minority $d$ bands (orange)
are the majority $d$ bands (blue) pushed up by 0.5 to 0.9 eV, for example by
0.80 eV at $\Gamma$. The majority $d$ band is full and lies below $E_F$. The
minority $d$ band crosses $E_F$, so it has holes, and the number of these holes is
the moment, 0.728 $\mu_B$, against a measured 0.61 $\mu_B$. This is the Stoner
picture of a band magnet. In the DOS on the right, the minority channel is drawn
to the left. The broad band that starts at $-9.6$ eV at $\Gamma$ is the $4s$
band. Its exchange splitting there is 0.06 eV, because the moment sits in the $d$
shell.

??? example "Code of this figure (docs/figures/magnetism_nickel_dos/)"

    ```python title="compute.py"
    --8<-- "magnetism_nickel_dos/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "magnetism_nickel_dos/plot.py"
    ```

## Noncollinear magnetism

We now let the moments point in any direction. With `noncolin = .true.` the
wavefunctions are two-component spinors, and the magnetization is a vector field:

$$
\mathbf m(\mathbf r) = \sum_{n\mathbf k} f_{n\mathbf k}\,
   \psi^{\dagger}_{n\mathbf k}(\mathbf r)\,\boldsymbol\sigma\,\psi_{n\mathbf k}(\mathbf r) .
$$

Without spin-orbit coupling the energy does not depend on the direction of all
moments together, only on their angles to each other. We give the direction of the
starting moment of a species with `angle1` (from $z$) and `angle2` (from $x$), as
in `pw.x`. To give each atom its own direction, you add a `STARTING_MOMENTS` card
with one row $(m_x, m_y, m_z)$ per atom. Here we take four hydrogen atoms in a
chain, with moments 90 degrees apart in the plane across the chain:

```python
helix = Calculator.from_file("docs/figures/magnetism_texture/h4-helix.in",
                             pseudo_dir="tests/data/pseudo")
scf = helix.get_scf()
print(np.round(scf.site_moments, 4))      # mu_B, one row per atom
print(f"cell: {np.round(scf.magnetization_vector, 5)}")
```

```
[[ 0.4681  0.      0.    ]
 [ 0.      0.4681 -0.    ]
 [-0.4681 -0.     -0.    ]
 [ 0.     -0.4681  0.    ]]
cell: [ 0. -0. -0.]
```

`scf.site_moments` is the moment in a sphere around each atom, one row per atom,
in $\mu_B$, and `scf.site_charges` is the charge in the same spheres. We see that
the moment of the cell is zero, because the four moments cancel. An
antiferromagnet and a cell with no moment at all give the same zero, so for a
compensated magnet you have to look at the site moments. The spheres are a
choice, so you should compare site moments with another code only at the same
radius.

![The moments of the helix on the hydrogen chain.](img/magnetism_texture.png#only-light)
![The moments of the helix on the hydrogen chain.](img/magnetism_texture-dark.png#only-dark)

The left panel shows the three components of the moment on each site along the
chain. The point at 20 bohr is the first atom of the next cell, so the turn
closes. The right panel shows the four moments seen along the chain. They are
0.468 $\mu_B$ each and exactly 90 degrees apart, and $m_z$ is zero. The code keeps
only the symmetry operations that map this texture onto itself, four of them
here, so the texture is not averaged away during the SCF.

??? example "Code of this figure (docs/figures/magnetism_texture/)"

    ```python title="compute.py"
    --8<-- "magnetism_texture/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "magnetism_texture/plot.py"
    ```

Let us now start the same chain from another texture. `with_moments` gives a new
calculator with another set of starting moments, with no new input file, and the
new run starts from these moments and not from the old density:

```python
afm = helix.with_moments([[0, 0, 0.6], [0, 0, -0.6]] * 2)
scf = afm.get_scf()
print(np.round(scf.site_moments, 4))
print(f"energy against the helix {scf.total_energy - helix.get_scf().total_energy:+.6f} Ry")
```

```
[[-0.      0.      0.6007]
 [ 0.     -0.     -0.6007]
 [ 0.     -0.      0.6007]
 [-0.     -0.     -0.6007]]
energy against the helix -0.008368 Ry
```

The collinear antiferromagnet is 8.4 mRy below the helix, so the helix is not the
ground state of this chain. Nothing in the SCF turns a moment towards a lower
texture. A run converges to a state close to where it starts, and it reports
convergence just as well. So when you look for the ground state, you should
compare several starting textures.

## Holding a moment: constraints and fields

We now hold a moment in place. To compute a state that is not a minimum, for
example two moments at a fixed angle, we hold the moment with a constraint.
`constrained_magnetization` in `&system` chooses what is held, as in `pw.x`:
`'total'` (the moment of the cell), `'atomic'` (the moment of each atom),
`'atomic direction'` (only its direction) and `'total direction'`. Each adds a
penalty with the weight `lambda`:

$$
E_{\rm pen} = \lambda \sum_I \big(\mathbf m_I - \mathbf m^{\rm fix}_I\big)^2 .
$$

A penalty still pushes at convergence, so the moment ends a bit away from its
target. `'atomic fsm'` instead adjusts a field on each atom until the moment is
exactly at its target. The field that does this is then a result, and it is the
torque that the texture costs. For a fixed external field you use the `B_field`
of `pw.x`, or a field on each atom with a `LOCAL_MAGNETIC_FIELDS` card. The energy
of a penalty or of a field is not part of the total energy, and the code prints it
apart, as in `pw.x` and Elk.

## From one spin regime to another

Here we start a run in one spin regime from a converged run in another. This goes
from unpolarized to collinear, from collinear to noncollinear, or to a fully
relativistic dataset with spin-orbit coupling. `with_spin(nspin)` returns a new
calculator whose SCF starts from the converged density of the old one. Here we
take the nickel run above to spinors:

```python
spinor = nickel.with_spin(4)              # noncollinear, from the collinear state
scf = spinor.get_scf()
print(f"{scf.iterations} iterations, m = {np.round(scf.magnetization_vector, 4)}")
```

```
1 iterations, m = [ 0.    -0.     0.728]
```

The noncollinear run starts at the converged collinear density, so it needs one
iteration. This is how a magnetic anisotropy calculation starts in practice, as
we see in the page about anisotropy. From an unpolarized run to a collinear one,
the code seeds the moment, because nothing in the SCF breaks the spin symmetry by
itself. Without a seed, the run would stay unpolarized and still report
convergence.

!!! note "Limits"
    With `occupations = 'fixed'` and two spin channels, you must give
    `tot_magnetization`, and it must be an integer, as in `pw.x`.
    `tot_magnetization` works only with `nspin = 2`.

    `noncolin` with `angle1` or `angle2` and no `starting_magnetization` gives an
    error, because the angles then have no moment to point. `STARTING_MOMENTS`
    needs `noncolin` for any $x$ or $y$ component, and it needs one row per atom.

    `constrained_magnetization = 'fsm'` together with spin-orbit coupling gives an
    error, so there you use `'total'` or `'atomic'`. `'total direction'` with
    `nspin = 2` gives an error, as in `pw.x`.

    A symmetry operation that reverses every moment is combined with time
    reversal. The code drops these operations, which gives the same result at
    the cost of more k-points.

    A collinear source cannot be promoted to a target whose species point in
    different directions, and a noncollinear magnetization cannot be split back
    into two collinear channels. The noncollinear linear response gives an error.

The notebooks for this page are
[07, spin polarization](https://github.com/joselado/defuqumat/blob/master/notebooks/07_spin_polarization.ipynb),
[11, noncollinear magnetism and fields](https://github.com/joselado/defuqumat/blob/master/notebooks/11_noncollinear_magnetism_and_fields.ipynb),
[43, magnetic textures](https://github.com/joselado/defuqumat/blob/master/notebooks/43_magnetic_textures.ipynb)
and
[18, continuing a calculation](https://github.com/joselado/defuqumat/blob/master/notebooks/18_continuing_a_calculation.ipynb).
