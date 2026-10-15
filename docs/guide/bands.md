# Band structure

In this page we compute the band structure, the energies
$\varepsilon_{n\mathbf k}$ along a path through the Brillouin zone, and the band
velocities, the slopes of these bands.

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

!!! note "Limits"
    `bands.gap(nelec)` gives an error for a spin-polarized run (`nspin = 2`),
    because a gap at fixed filling is not defined for each channel. In this case
    you take it from `bands.eigenvalues_by_spin` with the occupation of each
    channel.

    Inside a degenerate multiplet a single band has no well-defined velocity.
    There, each row of `v.velocities` holds, for each direction, the slopes of
    the bands that the multiplet splits into, in ascending order, and
    `v.degenerate_by_spin` marks these rows.

The notebook for this page is
[02, silicon](https://github.com/joselado/defuqumat/blob/master/notebooks/02_silicon_scf_and_bands.ipynb).
