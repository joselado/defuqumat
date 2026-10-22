# Band structure

In this page we compute the band structure, the energies
$\varepsilon_{n\mathbf k}$ along a path through the Brillouin zone, the band
velocities, the slopes of these bands, and the effective masses, their
curvatures.

## Bands along a k-path

We start from the Kohn-Sham states of a crystal, which are Bloch states labelled
by a band index $n$ and a crystal momentum $\mathbf k$:

$$
H_{\rm KS}[n]\,\psi_{n\mathbf k} = \varepsilon_{n\mathbf k}\,\psi_{n\mathbf k},
\qquad
\psi_{n\mathbf k}(\mathbf r + \mathbf R) = e^{i\mathbf k\cdot\mathbf R}\,\psi_{n\mathbf k}(\mathbf r).
$$

The band structure is $\varepsilon_{n\mathbf k}$ along lines between
high-symmetry points. ARPES measures the occupied part of it, and optical
absorption and transport see the band edges and the gap.

A band structure is not self-consistent. We fix the density at the converged
one and diagonalize the Hamiltonian at each k-point of the path, so the path can
have many points, and it does not need the symmetry of a mesh. We compute it with
`calc.get_bands()`:

```python
from defuqumat import Calculator
from defuqumat.system.kpoints import KPoints

calc = Calculator.from_file("docs/figures/bands_silicon/si.in",
                            pseudo_dir="tests/data/pseudo")

path = KPoints.band_path(
    [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],   # L - Gamma - X
    [20, 20, 1], calc.system.cell, crystal=False,
)
bands = calc.get_bands(kpoints=path, nbnd=8)

print(bands.eigenvalues_ev.shape)
print(f"indirect gap {bands.gap(8):.3f} eV")
```

```
(41, 8)
indirect gap 0.504 eV
```

With `crystal=False` the corners of the path are in cartesian coordinates, in
units of $2\pi/a$, and with `crystal=True` they are in units of the reciprocal
lattice vectors. The second list says how many points each segment has. The last
count is 1, so the path ends on its last corner. A count of 0 makes a jump,
meaning that the next corner starts a new segment at the same distance, as U|K in
the figure below.

From the result we use these attributes and methods:

- `bands.eigenvalues_ev` has the shape (k-points, bands), in eV.
- `bands.path_length` is the distance along the path, the x-axis of the plot.
- `bands.gap(nelec)` is the indirect gap for `nelec` electrons, the lowest
  conduction level on the path minus the highest valence level on the path.
- `bands.plot()` draws the bands with matplotlib, with the zero at the Fermi
  level of the SCF.

If you need the eigenvalues and the occupations on a k-mesh, you can use
`calc.get_nscf(kpoints=...)`, which also diagonalizes at a fixed density and in
addition fills the states.

## A path by the names of its points

We can also give the path by the names of the high-symmetry points, as they are
written on a band-structure plot. Here we take the same silicon, given with
keywords as in [Your own crystal](own-crystal.md):

```python
from defuqumat import Calculator

calc = Calculator.from_keywords(
    lattice="fcc", a=5.40,
    atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
    pseudos={"Si": "Si.pz-vbc.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=217.7, kmesh=(6, 6, 6), kshift=(1, 1, 1),
)
bands = calc.get_bands(kpoints="L-G-X-U|K-G", nbnd=8)
print(bands.eigenvalues_ev.shape)
print(f"indirect gap {bands.gap(8):.3f} eV")

dense = calc.get_bands(kpoints=calc.kpath("L-G-X-U|K-G", npoints=200), nbnd=8)
print(dense.eigenvalues_ev.shape)
```

```
(101, 8)
indirect gap 0.506 eV
(201, 8)
```

The gap is 0.506 eV and not 0.504 eV, because here $a$ = 5.40 Å and in the file
above $a$ = 10.20 bohr = 5.3976 Å. A `-` joins two points and a `|` is a jump,
so U|K is a new segment that starts at K. `G` and `Gamma` are both $\Gamma$. A
string gives 100 points, shared between the segments by their length, and
`calc.kpath(path, npoints=...)` sets another number. The same strings work in
`calc.get_nscf`, `calc.get_band_velocities`, `calc.get_shift_current` and
`calc.get_shg`.

The names exist for the named lattices of `Calculator.from_keywords`:

- **sc**: G, X, M, R.
- **fcc**: G, X, L, W, K, U.
- **bcc**: G, H, N, P.
- **hexagonal**: G, M, K, A, L, H.
- **tetragonal**: G, X, M, Z, R, A.
- **orthorhombic**: G, X, Y, Z, S, U, T, R.

A crystal given by its lattice vectors has no names, because its vectors can be
any choice, and then you give the corners with `KPoints.band_path` as above.

![The band structure of silicon along L-Gamma-X-U|K-Gamma.](img/bands_silicon.png#only-light)
![The band structure of silicon along L-Gamma-X-U|K-Gamma.](img/bands_silicon-dark.png#only-dark)

The figure shows silicon in LDA, with the four valence bands in blue and the
four lowest conduction bands in orange. The zero is the top of the valence band,
at $\Gamma$, where three bands are degenerate. The bottom of the conduction band
is on the line from $\Gamma$ to X, at about 85% of the way to X, so the gap is
indirect. Its value is 0.50 eV, while experiment gives 1.17 eV. This error comes
from LDA and not from the code, since semilocal functionals underestimate band
gaps. The degeneracies at $\Gamma$, X and L are exact, because the density has
the full symmetry of the crystal.

??? example "Code of this figure (docs/figures/bands_silicon/)"

    ```python title="compute.py"
    --8<-- "bands_silicon/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "bands_silicon/plot.py"
    ```

## Band velocities

Let us now compute the band velocities. The group velocity of a Bloch state is
the slope of its band:

$$
\mathbf v_{n\mathbf k} = \frac{1}{\hbar}\,\nabla_{\mathbf k}\,\varepsilon_{n\mathbf k}
  = \frac{1}{\hbar}\,\langle\psi_{n\mathbf k}|\,\nabla_{\mathbf k} H
  - \varepsilon_{n\mathbf k}\nabla_{\mathbf k} S\,|\psi_{n\mathbf k}\rangle .
$$

The second form is the Hellmann-Feynman theorem, and $S$ is the overlap
operator, which is the identity for a norm-conserving pseudopotential. The code
computes it with automatic differentiation of the Hamiltonian, and not by a finite
difference of the bands. At the Fermi level this is the Fermi velocity, which
enters the conductivity and the plasma frequency. We compute it with
`calc.get_band_velocities()`:

```python
v = calc.get_band_velocities(kpoints=path, nbnd=8)
print(v.velocities.shape)   # (k-points, bands, 3), in Ry bohr
```

```
(41, 8, 3)
```

The unit is Ry bohr, with $\hbar = 1$. If you do not pass `kpoints`, the
velocities are on the k-points of the SCF.

## Effective masses

Let us now go one derivative further, to the curvature of a band. Near a band
edge a band is a parabola, and its curvature is the inverse effective mass
tensor:

$$
\left(\frac{1}{m^*}\right)_{ab} = \frac{1}{2}\,
  \frac{\partial^2 \varepsilon_{n\mathbf k}}{\partial k_a\,\partial k_b} ,
$$

in units of $1/m_e$, because $\hbar^2/2m_e$ is exactly 1 Ry bohr$^2$. The mass
sets the mobility of the carriers and the density of states at the band edge,
and cyclotron resonance measures it. The code takes the first derivative from
the band velocity above and the second one from a small finite difference of the
velocity around the k-point you give. We compute it with
`calc.get_effective_mass()`, at the minimum of the conduction band of silicon:

```python
import numpy as np

k = np.array([0.843, 0.0, 0.0])           # the minimum, in units of 2 pi / a
mass = calc.get_effective_mass(calc.system.cell.k_to_crystal(k), nbnd=8)
masses, axes = mass.principal(4)          # band 4, the lowest conduction band
print(np.round(masses, 3))                # in units of m_e
print(np.round(abs(axes[:, 0]), 3))       # the axis of the first mass
```

```
[0.952 0.19  0.19 ]
[1. 0. 0.]
```

The k-point is in crystal coordinates, so we convert it with
`cell.k_to_crystal()`, and the tensor that comes back is cartesian.
`mass.inverse_mass` has the shape (bands, 3, 3), `mass.principal(n)` gives the
three masses of band $n$ and their axes, and `mass.density_of_states_mass(n)`
gives their geometric mean. Inside a degenerate multiplet a single band has no
mass tensor, as for the velocity, so these bands are `nan`, and
`mass.multiplets` holds the summed inverse mass of each multiplet, which is well
defined.

![The conduction valley of silicon along and across Gamma-X, with the parabolas of the two masses.](img/bands_masses.png#only-light)
![The conduction valley of silicon along and across Gamma-X, with the parabolas of the two masses.](img/bands_masses-dark.png#only-dark)

The minimum of the conduction band is on the line from $\Gamma$ to X, at 0.843
of the way to X, and by symmetry there are six such valleys, one on each of the
cubic axes. We see that each valley is a long ellipsoid. Along $\Gamma$-X the
band is flat, with the longitudinal mass $m_l = 0.952\,m_e$, and across it the
band is steep, with the transverse mass $m_t = 0.190\,m_e$. Cyclotron resonance
gives 0.92 and 0.19 $m_e$. The orange parabolas come only from the mass tensor
at the minimum, while the blue points are a separate band calculation along the
two lines. So the two agree near the minimum and move apart further out, where
the band stops being quadratic: at 0.08 1/bohr across the valley the band is
0.036 eV below its parabola. The grey curve is the next conduction band. Along
$\Gamma$-X it comes down and touches the lowest one at X, just outside the left
panel. The finite difference is done at two step sizes and combined so that its
leading error cancels, and `mass.truncation` gives the size of the part that was
removed, 0.034 in units of $1/m_e$ for this band.

??? example "Code of this figure (docs/figures/bands_masses/)"

    ```python title="compute.py"
    --8<-- "bands_masses/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "bands_masses/plot.py"
    ```

!!! note "Limits"
    For a spin-polarized run (`nspin = 2`), `bands.gap(nelec)` needs the filling
    of each channel. You give it as `bands.gap(nelec, tot_magnetization=m)`, or as
    `bands.gap(occupied=(n_up, n_down))`. Without one of them it gives an error.
    The gap is taken over both channels, so it can go from the top of one channel
    to the bottom of the other. `bands.band_edges(...)` gives the two levels.

    Inside a degenerate multiplet a single band has no well-defined velocity.
    There, each row of `v.velocities` holds, for each direction, the slopes of
    the bands that the multiplet splits into, in ascending order, and
    `v.degenerate_by_spin` marks these rows.

    `get_effective_mass()` gives an error on a spin spiral, because its stencil
    needs the bands at new k-points, and a spiral cannot yet move its two
    spheres to another k-list. The band velocities on the k-points of the SCF
    work. Two eigenvalues closer than `degeneracy_tol = 1e-5` Ry at the k-point
    form a degenerate multiplet. Inside it a single band has no mass tensor,
    because the eigensolver can rotate the bands inside the multiplet. These
    bands are `nan` in `mass.inverse_mass`, `mass.principal(n)` gives an error
    for them, and `mass.multiplets` holds the summed inverse mass of the
    multiplet, which is invariant.

    The step `delta`, 0.025 1/bohr by default, is the outermost point of the
    stencil with both `method="velocity"` and `method="eigenvalue"`. The code does not tune it,
    and it reports the error that the two step sizes removed in
    `mass.truncation`.

The notebooks for this page are
[02, silicon](https://github.com/joselado/defuqumat/blob/master/notebooks/02_silicon_scf_and_bands.ipynb)
and
[29, effective masses](https://github.com/joselado/defuqumat/blob/master/notebooks/29_effective_mass_and_angular_momenta.ipynb).
