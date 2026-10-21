# STM images and tunnelling

In this page we compute what a scanning-tunnelling microscope sees. We start with
the image of a surface at one bias, then we compute the spectrum $\mathrm dI/\mathrm dV$
at one place, and finally we follow the electron through a two-dimensional
material into a substrate below it, first in real space and then in momentum
space. At the end we put one impurity on the surface and compute the
quasiparticle interference it creates.

## The STM image

What does a scanning-tunnelling microscope see? In the picture of Tersoff and
Hamann the tip is a point with no structure of its own, an s-wave tip, and the
current it draws at $\mathbf r$ is proportional to the local density of states of
the sample at the tip:

$$
\rho_{\rm STM}(\mathbf r) = \sum_{n\mathbf k} w_{\mathbf k}\,
   \tilde\delta(E - \varepsilon_{n\mathbf k})\,
   |\psi_{n\mathbf k}(\mathbf r)|^2 ,
$$

with $\tilde\delta$ a smeared delta function at the energy $E$ of the tip, the
Fermi level at zero bias. This is the density built from the states at one energy
only, and not from all the occupied states, so an STM image is not a picture of
the atoms and not a picture of the charge. It is a picture of the states near the
Fermi level, seen in the vacuum where the tip is. The integral of
$\rho_{\rm STM}$ over the cell is the density of states $D(E)$. A microscope
records it in constant-height mode as a current and in constant-current mode as a
height.

Here we take AB-stacked bilayer graphene, which is the top of a graphite crystal.
Half of the atoms of the top layer, which we call A, sit on an atom of the layer
below, and the other half, B, sit on the centre of a hexagon of the layer below.
We compute the image
with `calc.get_stm()`:

```python
from defuqumat import Calculator

graphite = Calculator.from_file("docs/figures/stm_graphite/graphite.in",
                                pseudo_dir="tests/data/pseudo")
image = graphite.get_stm(height=0.85, shape=(60, 60))

a, b, hollow = image.values[0, 0], image.values[40, 20], image.values[20, 40]
print(f"A {a:.3e}   B {b:.3e}   hexagon centre {hollow:.3e}   (1/(bohr^3 Ry))")
print(f"B/A = {b / a:.2f}")
print(f"integral over the cell {image.integral:.4f} states/Ry")
```

```
A 5.883e-04   B 9.649e-04   hexagon centre 2.408e-07   (1/(bohr^3 Ry))
B/A = 1.64
integral over the cell 2.4834 states/Ry
```

The two atoms of the top layer are both carbon and carry the same charge, and
yet at the Fermi level B is 1.64 times brighter than A. You can think of it this
way: A is bound to the atom below it, which pushes the states of A away from the
Fermi level, so the states at the Fermi level sit on B. Above the centre of the
hexagon the tip sees $2.4\times10^{-7}$, more than three orders of magnitude
less. So the brightest spot of each cell is on B, and A is 0.61 of it. This
asymmetry between the two atoms is the origin of the triangular images of
graphite, where only half of the atoms appear.

The integral of the image, 2.4834 states/Ry, is $D(E_F)$ of the cell, here on the
$12\times12\times1$ k-mesh of the SCF with a Gaussian of 0.02 Ry. Bilayer
graphene has very few states at the Fermi level, all near K, so this number
depends strongly on the k-mesh, and in the next section the same width on a
$24\times24\times1$ mesh gives 0.6450 states/Ry. So this value is not converged.
The ratio between B and A is the result that does not move: it is 1.62 on the
finer mesh.

The keywords of `get_stm` are:

- **`height`**: the tip plane, as a crystal coordinate along the third lattice
  vector. Here 0.85 of $c = 20$ bohr is 3.5 bohr above the top layer.
- **`plane`**: any other plane, given by three of its corners in crystal
  coordinates, the origin and the ends of the two edges. Use it for a cut that is
  not parallel to the surface, as the right panel of the figure below does.
- **`shape`**: the number of points on the plane. The value at each point is the
  exact plane-wave sum there, and not an interpolation.
- **`bias`**: the sample bias in Ry. If you do not give it, the tip sees a smeared
  delta at the Fermi level. With a bias, it sees all the states between $E_F$ and
  $E_F + V$, so a negative bias images the filled states and a positive bias the
  empty states.
- **`width`**, **`smearing`**: the width of the delta, in Ry, and its shape. If
  you do not give them, the code uses `degauss` of the run and a Gaussian.
- **`grid`**, **`nbnd`**: compute the bands again on a denser k-mesh, at fixed
  density, before building the image. A delta at the Fermi level needs more
  k-points than the SCF.
- **`mode`**: `"constant-height"`, or `"constant-current"` with a set-point
  `current` and a scan range `heights` in bohr. In the second mode
  `image.heights_bohr` is the height of the tip at each point, which is the
  topograph a microscope records.
- **`spin`**, **`polarization`**: a magnetic tip, with its moment along a
  direction `(nx, ny, nz)`, or `"up"` and `"down"` for a collinear run. The tip
  then sees $\tfrac12[\rho + P\,\hat{\mathbf n}\cdot\mathbf m]$, so the image of a
  noncollinear magnet depends on where the tip points.

`image.values` is the image on the plane, `image.coordinates` the cartesian
coordinates of its points in bohr, `image.extent()` the frame for `imshow`, and
`image.density` the same quantity on the whole FFT grid of the cell.

Let us now open a window with the bias. With a bias, the values are a number of
electrons per bohr$^3$, and not a density of states:

```python
for bias in (-0.15, 0.15):
    window = graphite.get_stm(height=0.85, shape=(60, 60), bias=bias)
    a, b = window.values[0, 0], window.values[40, 20]
    print(f"bias {bias:+.2f} Ry   A {a:.3e}   B {b:.3e}   B/A = {b / a:.2f}")
```

```
bias -0.15 Ry   A 1.909e-04   B 2.042e-04   B/A = 1.07
bias +0.15 Ry   A 2.793e-04   B 2.667e-04   B/A = 0.95
```

With a window of 0.15 Ry, about 2 eV, the contrast almost disappears: B/A is
1.07 for the filled states and 0.95 for the empty states. The window now
collects states from a large part of the $\pi$ bands, which sit on both atoms,
and only the few states very near the Fermi level tell A and B apart. So the
triangular image of graphite is a low-bias image.

![The STM image of graphite at constant height, and the decay of the tunnelling density into the vacuum.](img/stm_graphite.png#only-light)
![The STM image of graphite at constant height, and the decay of the tunnelling density into the vacuum.](img/stm_graphite-dark.png#only-dark)

On the left we see the image at 3.5 bohr, repeated over a few cells. The bright
spots are on B, A is dimmer but still bright, and only the centres of the
hexagons are dark. On the right we follow the tunnelling
density from the top layer up to the middle of the vacuum, 6.5 bohr. Above A and
above B it falls by a factor of 10 every 1.16 bohr, and B stays brighter than A
up to 6 bohr, so the contrast does not depend on the height of the tip. Above
the centre of the hexagon the density stays below $10^{-5}$ at every height. You
can think of it this way: the states at the Fermi level are the states near K,
and by the threefold symmetry the states of the top layer are zero on the axis
through the centre of its hexagon. The small value left there comes from the
atom of the lower layer that sits on this axis, 7.0 bohr further from the tip.
At 6.5 bohr the three curves meet, near
$3\times10^{-6}$, because there the tip is as far from this surface as from the
next copy of the slab, so the image has no contrast left.

The image here is at constant height. A microscope usually records the constant
current mode, `mode="constant-current"`, which needs the tip to reach the
set-point everywhere, also above the dark centres of the hexagons. With 13 bohr
of vacuum, as here, we stay with the constant-height image, which carries the
same contrast, and for a topograph you need a cell with more vacuum.

??? example "Code of this figure (docs/figures/stm_graphite/)"

    ```python title="compute.py"
    --8<-- "stm_graphite/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "stm_graphite/plot.py"
    ```

## The tunnelling spectrum

An image is one energy at many places. Let us now take the other cut of the same
function, one place at many energies, which is what a microscope measures when it
holds the tip still and sweeps the bias:

$$
\frac{\mathrm dI}{\mathrm dV}(\mathbf r, V) \;\propto\;
\rho(\mathbf r, E_F + eV) =
\sum_{n\mathbf k} w_{\mathbf k}\,
   \tilde\delta(E_F + eV - \varepsilon_{n\mathbf k})\,
   |\psi_{n\mathbf k}(\mathbf r)|^2 .
$$

This is scanning tunnelling spectroscopy (STS). It shows a gap, a band edge or a
state inside a gap, which an image at one energy cannot show. The wavefunctions do
not depend on the energy, so the code samples them at the tip once, and each
energy after that costs almost nothing. We compute it with `calc.get_sts()`:

```python
import numpy as np

fermi = graphite.get_scf().fermi_energy
tips = [[0, 0, 0.85], [2 / 3, 1 / 3, 0.85]]         # above A and above B
sts = graphite.get_sts(energies=fermi + np.linspace(-0.3, 0.3, 121), tip=tips,
                       width=0.02, grid=(24, 24, 1), nbnd=16)

zero = np.argmin(np.abs(sts.bias_axis))
print(sts.values.shape)                              # (energies, tips)
print(f"dI/dV at V = 0:  A {sts.values[zero, 0]:.3e}   B {sts.values[zero, 1]:.3e}")
print(f"D(E_F) = {sts.integral[zero]:.4f} states/Ry per cell")
```

```
(121, 2)
dI/dV at V = 0:  A 1.538e-04   B 2.497e-04
D(E_F) = 0.6450 states/Ry per cell
```

At zero bias the tip above B sees 1.62 times more than the tip above A, the same
contrast as in the image. Here the code computes the bands again on the full
$24\times24\times1$ mesh, 576 k-points, so $D(E_F)$ is 0.6450 states/Ry per
cell, against 2.4834 on the k-mesh of the SCF.

The keywords of `get_sts` are:

- **`energies`**: the energies of the tip, in Ry. There is no default, since the
  energy axis is the measurement.
- **`tip`**: the tip positions in crystal coordinates, one row each. One point
  gives a spectrum and a row of points a line cut. With **`height`** instead, the
  code gives a map at every energy.
- **`grid`**, **`nbnd`**: the k-mesh and the bands. A spectrum needs the full
  k-mesh, without symmetry, and `grid` builds it.
- **`width`**, **`smearing`**, **`spin`**, **`polarization`**: as for
  `get_stm`.

`sts.values` has the shape (energies, tips), `sts.bias_axis` is the bias in Ry
measured from the Fermi level, `sts.integral` is $D(E)$ per cell at each energy,
and `sts.current` is $I(V)$, the integral of the spectrum from $V = 0$, negative
for the filled states as in an experiment.

![dI/dV of graphite above the two kinds of surface atom.](img/stm_spectrum.png#only-light)
![dI/dV of graphite above the two kinds of surface atom.](img/stm_spectrum-dark.png#only-dark)

On the left we see the whole range, $\pm 4$ V. The two curves are almost the
same away from the Fermi level. The peak at $-2.42$ V is the saddle point of the
filled $\pi$ band at M, a van Hove singularity, and the peak at $+1.86$ V is its
partner in the empty $\pi^*$ band. Above about 2.7 V the signal grows by a factor
of 4.5. These are states with much weight in the vacuum, and we did not converge
them against the size of the vacuum, so this part of the curve depends on the
cell. On the right we see the zoom near the Fermi level. Above A the spectrum
has a minimum at $V = 0$, which is the semimetal. Above B there is a small peak at
$V = 0$ instead, from the flat low-energy states of the bilayer, which sit on B.
Only near zero bias A and B differ, with B up to 1.63 times A between $-0.1$ and
$+0.1$ V. The width of the delta is
0.02 Ry, 0.27 eV, and the curve is smooth only because of this width. A sharper
spectrum needs a denser k-mesh together with a smaller `width`.

??? example "Code of this figure (docs/figures/stm_spectrum/)"

    ```python title="compute.py"
    --8<-- "stm_spectrum/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "stm_spectrum/plot.py"
    ```

## Vertical transport through a two-dimensional material

So far the tip only looks at the surface. We now ask a different question, what
goes through. An electron enters at a point $\mathbf r$ above a two-dimensional
material and leaves into a substrate below it, an infinite flat plane. The current
is then set by the amplitude to go from the tip to the plane, which is the
nonlocal Green's function $G(\mathbf r, \mathbf r')$, and not by the density of
states at the tip:

$$
T(\mathbf r; E) = \int_{\rm plane}
   \bigl|G(\mathbf r, \mathbf r'; E)\bigr|^2\, d^2 r' ,
\qquad
G(\mathbf r, \mathbf r'; E) = \sum_{n\mathbf k}
   \frac{\psi_{n\mathbf k}(\mathbf r)\,\psi^{*}_{n\mathbf k}(\mathbf r')}
        {E - \varepsilon_{n\mathbf k} + i\eta} .
$$

The sum over bands is taken before the square, so two bands are two paths through
the material and they add as amplitudes, which means that they can interfere. The
substrate is the same under every lattice translation, so it conserves the
in-plane momentum, and only bands at the same $\mathbf k$ interfere. This is a
map of the vertical conductance of a sheet on a metal, as a conductive-tip
microscope measures it. We compute it with `calc.get_vertical_transport()`, here
for one sheet of graphene:

```python
graphene = Calculator.from_file("docs/figures/stm_transport/graphene.in",
                                pseudo_dir="tests/data/pseudo")
flow = graphene.get_vertical_transport(exit_height=0.38, height=0.62, shape=(30, 30),
                                       broadening=0.02, grid=(6, 6, 1), nbnd=20)
stm = graphene.get_stm(height=0.62, shape=(30, 30), width=0.02,
                       grid=(6, 6, 1), nbnd=20)
same = np.corrcoef(flow.image.ravel(), stm.values.ravel())[0, 1]
print(f"graphene: correlation with the STM image {same:.6f}")
print(f"          open channels {flow.notes['channels']:.2f}")
```

```
graphene: correlation with the STM image 1.000000
          open channels 2.85
```

For one sheet of graphene the transport map and the STM image are the same, with
a correlation of 1.000000. At the Fermi level graphene has only the two Dirac
states at each K point, and the symmetry of K makes the substrate couple to both
equally and separately, so there is nothing for them to interfere through. Then
the current that reaches the substrate is just the density of states at the tip.
The substrate offers 2.85 open channels.

Let us now do the same for two sheets, the bilayer of the first section:

```python
flow2 = graphite.get_vertical_transport(exit_height=0.14, height=0.86, shape=(30, 30),
                                        broadening=0.02, grid=(6, 6, 1), nbnd=24)
stm2 = graphite.get_stm(height=0.86, shape=(30, 30), width=0.02,
                        grid=(6, 6, 1), nbnd=24)
same2 = np.corrcoef(flow2.image.ravel(), stm2.values.ravel())[0, 1]
print(f"bilayer:  correlation with the STM image {same2:.4f}")
print(f"          incoherent / coherent {flow2.incoherent.mean() / flow2.image.mean():.1f}")
```

```
bilayer:  correlation with the STM image 0.1562
          incoherent / coherent 26.4
```

For the bilayer the two maps are almost unrelated, with a correlation of 0.1562.
The coherent map, where the bands add as amplitudes, is 26.4 times weaker than
the incoherent one, where every band tunnels alone. So most of the paths through
the two sheets cancel each other, and the current is not what the density of
states at the tip would suggest. The k-mesh of these maps is $6\times6\times1$
with a broadening of 0.02 Ry, which is coarse for a delta at the Fermi level, so
the numbers show the mechanism and are not converged values.

The keywords of `get_vertical_transport` are:

- **`exit_height`**: the substrate plane, as a crystal coordinate along the
  stacking axis, in the vacuum below the material.
- **`height`**: the tip plane, in the vacuum above it. The atoms must be between
  the two planes.
- **`shape`**, **`tip`**: the points of the map, or explicit tip positions.
- **`broadening`**: the width $\eta$ of the window of energies, in Ry. If you do
  not give it, it is $10^{-3}$ Ry, which is too narrow for most k-meshes.
- **`energies`**: several tip energies at once, which cost little more than one.
- **`grid`**, **`nbnd`**: the full k-mesh, with one point along the stacking
  axis.
- **`spin`**, **`polarization`**: a magnetic **substrate**. Note that this is the
  opposite of `get_stm`, where `spin` is the tip.
- **`tip_spin`**, **`tip_polarization`**: a magnetic tip. With both electrodes
  magnetic the map depends on the angle between the two moments, which is a
  tunnelling magnetoresistance image.

`flow.image` is the map, `flow.incoherent` the same map with every band
tunnelling alone, `flow.interference` the difference of the two, and
`flow.notes["channels"]` the number of open channels to the substrate. The map is
in arbitrary units, because the couplings of the tip and of the substrate are
unknown constants, so what has a meaning is its shape and its contrast.

![Bilayer graphene: the STM image at the tip and the map of the current that reaches the substrate.](img/stm_transport.png#only-light)
![Bilayer graphene: the STM image at the tip and the map of the current that reaches the substrate.](img/stm_transport-dark.png#only-dark)

The figure shows the two maps of the bilayer on the same plane, each divided by
its maximum. The STM image is brightest on B, with A at 0.61 of it. The transport
map is the opposite: it is brightest on A, and B has only 0.029 of the maximum.
You can think of it as a question of the path. A sits on an atom of the lower
layer, and this is the strong bond between the two layers, so an electron that
enters at A goes down to the second layer and into the substrate. B is bright for
the tip, but its states do not carry the current through the second layer. So a
conductive tip on a bilayer sees the other sublattice than a normal STM does.

??? example "Code of this figure (docs/figures/stm_transport/)"

    ```python title="compute.py"
    --8<-- "stm_transport/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "stm_transport/plot.py"
    ```

## Which k-points the current comes from

Let us now make the tip a plane too, parallel to the surface. The map in real
space is then gone, and what is left is one weight per k-point, which says from
which part of the Fermi surface the electrons tunnel:

$$
W(\mathbf k) = w_{\mathbf k}\,
  \mathrm{Tr}\!\left[ D_{\mathbf k}\, S^{\rm exit}_{\mathbf k}\,
                      D_{\mathbf k}\, S^{\rm tip}_{\mathbf k} \right],
\qquad
\sum_{\mathbf k} W(\mathbf k) = \int_{\rm tip} T(\mathbf r)\, d^2 r ,
$$

with $S_{\mathbf k}[n, m] = \int_{\rm plane}\psi^{*}_{n\mathbf k}\psi_{m\mathbf k}\,d^2 r$
the overlap of two bands on a plane and $D_{\mathbf k}$ the diagonal of the
on-shell amplitudes $\sqrt{\tilde\delta(E - \varepsilon_{n\mathbf k})}$. A state
with in-plane momentum $\mathbf k_\parallel$ decays into the vacuum as
$e^{-\sqrt{\kappa_0^2 + k_\parallel^2}\,z}$, so the states far from $\Gamma$ reach
the tip and the substrate less, and a part of the Fermi surface can be large and
still carry little current. This is what a momentum-resolved tunnelling
experiment sees. We compute it with `calc.get_momentum_transport()`, here for a
square sheet of hydrogen:

```python
sheet = Calculator.from_file("docs/figures/stm_momentum/h-sheet.in",
                             pseudo_dir="tests/data/pseudo")
run = sheet.get_momentum_transport(exit_height=0.3, height=0.7, grid=(32, 32, 1),
                                   broadening=0.03)

k = run.kpoints[:, :2] - np.round(run.kpoints[:, :2])     # into the zone, in 2 pi/a
share = run.shares(np.linalg.norm(k, axis=1) > 0.43)       # near the X points
print(f"near X: {share['bare']:.3f} of the Fermi surface, {share['weight']:.3f} of the current")
print(f"suppression {share['suppression']:.2f}")
```

```
near X: 0.246 of the Fermi surface, 0.189 of the current
suppression 1.30
```

In the outer part of the zone, near the X points, there is 24.6% of the Fermi
surface but only 18.9% of the current, so these states tunnel 1.30 times less
than their share of the Fermi surface. The reason is their larger
$|\mathbf k_\parallel|$, which makes them decay faster into the vacuum on both
sides of the sheet.

`get_momentum_transport` takes the same `exit_height`, `broadening`, `grid`,
`spin` and `polarization` as `get_vertical_transport`, and `height` can be a list
of heights, which costs one band calculation for all of them. Four columns come
back, one value per k-point of the full grid: `run.weight` is $W(\mathbf k)$,
`run.tersoff_hamann` is the same with a structureless substrate, the local
density of states on the tip plane, `run.bare` is the Fermi surface itself,
$\sum_n \tilde\delta(E - \varepsilon_{n\mathbf k})$, and `run.incoherent` is
every band tunnelling alone. `run.as_grid("weight")` puts a column on the
Monkhorst-Pack grid for a picture of the zone, and `run.shares(mask)` gives the
share of each column in a region of the zone.

![The Fermi surface of a square hydrogen sheet, and the part of it the tunnelling current comes from.](img/stm_momentum.png#only-light)
![The Fermi surface of a square hydrogen sheet, and the part of it the tunnelling current comes from.](img/stm_momentum-dark.png#only-dark)

On the left we see the Fermi surface of the sheet, a rounded square around
$\Gamma$, with the most weight near the X points, where the band is flat. In the
middle we see the current that tunnels through, which has more weight on the
diagonals instead. On the right we divide the two, k-point by k-point: the
current per state of the Fermi surface falls by a factor of 2 between
$|\mathbf k_\parallel| = 0.33$ and $0.51$, in units of $2\pi/a$. This is the
factor $e^{-\sqrt{\kappa_0^2 + k_\parallel^2}\,z}$ at work, for the tip and for
the substrate, each 3 bohr from the sheet. The mesh is $32\times32\times1$ with a
broadening of 0.03 Ry.

??? example "Code of this figure (docs/figures/stm_momentum/)"

    ```python title="compute.py"
    --8<-- "stm_momentum/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "stm_momentum/plot.py"
    ```

## Quasiparticle interference around one impurity

We now put one impurity on the surface. It scatters the Bloch states, and the
scattered waves interfere with the incoming ones, so the local density of states
gets standing waves around the impurity. Their Fourier transform shows the
scattering vectors $\mathbf q = \mathbf k - \mathbf k'$ that join two states at
the same energy. This is quasiparticle interference (QPI), and it is how an STM
measures the band structure of a surface. We follow the T-matrix method of the
review by Rhodes, Kohsaka, Hanaguri, de Almeida Marques and Wahl,
[arXiv:2607.22487](https://arxiv.org/abs/2607.22487).

The impurity is a Gaussian $\phi$ of width $\sigma$ at $\mathbf r_{\rm imp}$, with
the potential $V_{\rm imp} = V|\phi\rangle\langle\phi|$. Here $V$ is in Ry, and it is
the energy shift of a state that lies inside $\phi$. For this potential the whole
Dyson series sums to one number at each energy:

$$
T(\omega) = \frac{V}{1 - V g_0(\omega)},
\qquad
g_0(\omega) = \frac1N \sum_{n\mathbf k}
  \frac{|\langle\psi_{n\mathbf k}|\phi\rangle|^2}{\omega - \varepsilon_{n\mathbf k} + i\eta},
$$

and the Green's function with the impurity is
$G = G_0 + G_0\,T\,G_0$. The k-mesh of $N$ points is a supercell of $N$ surface
cells with the impurity in one of them, and the map covers that whole supercell.
So no matrix of the size of the supercell is ever inverted. The bands are
sampled at the tip points of one cell, and the other cells follow from the Bloch
theorem.

There are two ways to read $G$ at the tip, and you choose them with `readers`.
The first one, `"ldos"`, is the Tersoff-Hamann answer, the change of the local
density of states at the tip,
$\delta\rho(\mathbf r) = -\frac1\pi \mathrm{Im}\,[G_0(\mathbf r,\phi)\,T\,G_0(\phi,\mathbf r)]$.
The second one, `"landauer"`, is the tunnelling transmission of the section on
vertical transport. The electron enters at the tip point $\mathbf r$ and leaves
through an exit plane below the material:

$$
T(\mathbf r) = \int_{\rm exit} |G(\mathbf r, \mathbf r')|^2\, d^2 r' .
$$

So each scattered wave is weighted by how well it tunnels from the tip to the
exit plane, which is the matrix element of the STM, and not only by its weight at
the tip. We compute both with `calc.get_qpi()`, here for the square sheet of
hydrogen of the previous section, with the impurity on an atom and the energy at
the Fermi level:

```python
sheet = Calculator.from_file("docs/figures/stm_qpi/h-sheet.in",
                             pseudo_dir="tests/data/pseudo")
fermi = sheet.get_scf().fermi_energy
qpi = sheet.get_qpi(0.5, 0.7, impurity=0, sigma=1.0, energies=[fermi],
                    readers=("ldos", "landauer"), exit_height=0.3,
                    shape=(4, 4), broadening=0.03, grid=(32, 32, 1))
print(f"T-matrix          {qpi.t_matrix[0, 0]:.4f} Ry")
print(f"wraparound        {qpi.wraparound[0, 0]:.4f}")
print(f"contour k-points  {qpi.contour_points[0, 0]:.0f}")
```

```
T-matrix          0.0431-0.0994j Ry
wraparound        0.0024
contour k-points  316
```

The impurity is weak here, $|T| = 0.108$ Ry against $V = 0.5$ Ry, because only a
small part of each state lies inside $\phi$. At the edge of the supercell of
$32\times32$ cells, $|G_0(\mathbf r,\phi)|$ has fallen to 0.0024 of its peak, so
the waves of the next impurity are not in the map, and the exit leg sees the
Fermi surface on 316 k-points.

The band calculation on the full k-mesh is the expensive step, and the
impurity, the energy and the broadening do not change it. So we compute it
once with `calc.get_qpi_bands()` and give it to every map as `states`:

```python
states = sheet.get_qpi_bands(grid=(32, 32, 1))
for eta in (0.03, 0.05):
    qpi = sheet.get_qpi(0.5, 0.7, impurity=0, sigma=1.0, energies=[fermi],
                        shape=(4, 4), broadening=eta, states=states)
    print(f"eta = {eta}  T-matrix {qpi.t_matrix[0, 0]:.4f} Ry  "
          f"wraparound {qpi.wraparound[0, 0]:.4f}")
```

```
eta = 0.03  T-matrix 0.0431-0.0994j Ry  wraparound 0.0024
eta = 0.05  T-matrix 0.0472-0.1145j Ry  wraparound 0.0007
```

At $\eta = 0.03$ Ry this is the T-matrix of the call above. A larger $\eta$
makes $G_0$ decay faster, so the wraparound drops.

The keywords of `get_qpi` are:

- **`potential`**: $V$ in Ry. A negative value attracts the electrons.
- **`height`**: the tip plane, a crystal coordinate along `axis`, in the vacuum.
- **`impurity`**: an atom index, for an impurity on that atom, or a position in
  crystal coordinates. If you do not give it, it is atom 0.
- **`sigma`**: the width of the Gaussian in bohr. It is the size of the impurity,
  and a small impurity scatters to large $q$.
- **`energies`**: the energies $\omega$ in Ry. If you do not give them, it is the
  Fermi level.
- **`readers`**: `"ldos"`, `"landauer"` or both. For `"landauer"` you also give
  **`exit_height`**, the exit plane on the other side of the slab.
- **`shape`**: tip points per surface cell. The Fourier transform then shows
  `shape` Brillouin zones along each direction.
- **`broadening`**: $\eta$ in Ry. It must be larger than the level spacing of the
  k-mesh on the constant-energy contour.
- **`exchange`**: for an `nspin = 2` run, a magnetic impurity, with $V + J$ for
  spin up and $V - J$ for spin down.
- **`born`**: the first Born approximation, $T = V$.
- **`bands`**: the band window the Green's function and the impurity live in.
  If you do not give it, every computed band is used.
- **`exit_method`**: the exit leg of `"landauer"`, `"spectral"` (on shell, the
  default) or `"resolvent"`. Only the first one converges with the number of
  bands, see the Limits note.
- **`grid`**, **`nbnd`**: the full k-mesh and the number of bands of the band
  calculation. The k-mesh is the supercell, so it decides the size of the map.
- **`states`**: the bands of the full k-mesh from `calc.get_qpi_bands(grid=...,
  nbnd=...)`. The band calculation is the expensive step, and with `states` many
  impurities, energies and broadenings use one band calculation.

`qpi.maps[reader]` is the change in real space, with the impurity cell in the
middle, and `qpi.transforms[reader]` is its Fourier transform, with $q = 0$ in
the middle. `qpi.qpi[reader]` is the modulus, and `qpi.backgrounds[reader]` is
the clean surface in one cell. Two numbers tell you whether the k-mesh is enough.
`qpi.wraparound` is how much of $|G_0(\mathbf r,\phi)|$ is left at the edge of the
supercell, and if it is not small, the waves of the next impurity are in the map.
`qpi.contour_points` is the number of k-points the exit leg of `"landauer"` sees
on the constant-energy contour.

![A square sheet of hydrogen with one impurity on an atom: the QPI pattern of the local density of states at the tip, and of the transmission from the tip to the exit plane.](img/stm_qpi.png#only-light)
![A square sheet of hydrogen with one impurity on an atom: the QPI pattern of the local density of states at the tip, and of the transmission from the tip to the exit plane.](img/stm_qpi-dark.png#only-dark)

We see the $2k_F$ contour in both maps, a rounded square at about
0.9 bohr$^{-1}$ from $q = 0$, which joins opposite sides of the Fermi surface.
Its copies, shifted by the reciprocal lattice vectors, cross it, because the tip
sees the sheet at the atomic scale. The two readers have the same contour, so
the scattering vectors come from the band structure. What changes is the weight
along it. The compute script measures the strongest peak along the axes and
along the diagonals:

```
ldos      axial / diagonal 2k_F peak 0.495
landauer  axial / diagonal 2k_F peak 0.424
```

The axial peaks join the flat sides of the Fermi surface, near X. These states
have the larger $|\mathbf k_\parallel|$, so they decay faster into the vacuum and
tunnel less, as in the previous section. So with the matrix element of the STM the
axial peaks are 14% weaker, relative to the diagonal ones, than the local density
of states says. On a $40\times40$ mesh with $\eta = 0.025$ Ry the factor is 11%,
so its size moves by a few per cent with the mesh and its sign does not.

??? example "Code of this figure (docs/figures/stm_qpi/)"

    ```python title="compute.py"
    --8<-- "stm_qpi/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "stm_qpi/plot.py"
    ```

!!! note "Limits"
    The STM image of a spin spiral works for the charge, which is periodic in
    the cell. A spin-resolved image of a spiral gives an error, because the
    moment turns from cell to cell, and so does a denser `grid`. A run with a
    fixed total magnetization (`tot_magnetization`) gives an error, because it
    has one Fermi level per spin and a tip sees one energy. A run in an applied
    magnetic field also gives an error, because the energy of the field is not
    in the total, so its Fermi level is not the one of the run without the
    field. A magnetic tip gives an error on a run with no magnetization, and a
    tip direction across the axis of a collinear run also gives an error,
    because such a run has no transverse magnetization, which is not the same as
    a zero one.

    A constant-current scan longer than the period of the cell along the normal
    gives an error, because there the tip meets the next copy of the surface. A
    point where the tip never reaches the set-point comes back as `nan`, with a
    warning that says how many points. At zero bias the image of an insulator is
    zero, because the delta sits in the gap, and the code warns about it, so for
    an insulator you give a `bias`. All four calculations of this page work with
    `K_POINTS gamma`. They agree with an explicit $k = 0$ to 1e-7, which is what
    two SCFs at `conv_thr = 1e-15` agree to.

    The tunnelling spectrum gives the same errors as the image, and two more,
    because here $\psi(\mathbf r)$ is sampled and squared at the tip. On a
    symmetry-reduced k-mesh the spectrum at each tip point is averaged over the
    images of the point under the symmetry operations, so it is the same as on
    the full mesh. A reduced k-mesh still gives an error for a magnetic run, for
    `nspin = 2` and for a spinor, which have not been checked, and for a shifted
    mesh, which the group does not map onto itself. There you pass `grid`, which
    builds the full mesh, or run with `nosym`. A tip inside the augmentation
    sphere of an ultrasoft or PAW atom gives an error, because there the
    pseudo-wavefunction is not the true one. So the tip has to be farther from
    every atom than the augmentation radius, and in practice this means in the
    vacuum above a surface. On a dense bulk crystal with these datasets every
    tip position gives an error, because the layers are closer together than
    that radius (2.4 bohr for SiC), and this is the expected answer,
    since a tip belongs above a surface. An ultracell already needs `nosym`, so
    the error for a symmetry-reduced k-mesh never applies to it. The
    augmentation-sphere error does apply, to its image, its spectrum and its
    transmission, and we check it on the atoms of the unit cell, which are the
    same in every copy. A `bias` together with `energies` gives an error,
    since `sts.current` is already that window, and the constant-current mode
    gives an error, since a tip that moves while the bias is swept is a
    different experiment. `sts.current` gives an error if the step of the
    energy axis is larger than `width`, because the integral would then be wrong
    by orders of magnitude, not by a few per cent. For a run with no Fermi level,
    the zero of bias is the middle of the gap, with a warning, and if the run
    has no gap either, `sts.current` gives an error.

    The vertical transport gives an error for a k-mesh with more than one point
    along the stacking axis, because the momentum along the normal is not
    conserved and two such points interfere with a phase that depends on where
    the substrate plane is. A symmetry-reduced k-mesh gives an error, because a
    mirror through the slab exchanges the tip and the substrate, and `grid`
    builds the full mesh. The substrate plane is always a lattice plane, set by
    `exit_height`, while the tip plane may be tilted. A tip or a substrate plane
    inside an augmentation sphere of an ultrasoft or PAW atom gives an error,
    and in the vacuum these datasets need nothing extra. A magnetic tip or a
    magnetic substrate gives an error on a run with no magnetization, and a
    direction across the axis of a collinear run gives an error for both. A spin
    spiral gives an error, because the spectrum samples each state on one
    plane-wave list, and so do a fixed `tot_magnetization` and an applied field,
    as for the image. A `bias` window whose energy step is larger than
    `broadening` gives an error, since the current would be wrong by orders of
    magnitude, and the error says which `nenergies` is enough. If the atoms are
    not between the two planes, the code warns. The tip is an s-wave tip, the
    substrate is infinite and flat, and the map has no absolute scale, because
    the couplings of the tip and of the substrate are unknown constants.

    The momentum-resolved transport gives the same errors as the vertical
    transport for the k-mesh, the two planes and the substrate, and the code warns if both planes are on the same side of the
    slab. It has no magnetic tip (`tip_spin`), because a magnetic plane tip
    mixes the two spinor components in a way the integral over the plane does
    not reduce, so for a magnetic tip you use the map of
    `get_vertical_transport`. The tip plane is always parallel to the layers,
    given by `height`. The weights have no absolute scale, so `run.shares()`
    gives shares of a total.

    QPI needs three things at the same time, and they are one condition on the
    k-mesh. The broadening $\eta$ must be much smaller than the distance in
    energy to the edge of the band, or the constant-energy contour is a disk. The
    supercell, $N a$, must be larger than about $2\pi v/\eta$, or the exit leg of
    `"landauer"` sees the contour on a few k-points, and the map takes the shape
    of the k-mesh. It must also be larger than about $9 v/\eta$, or the waves of
    the next impurity are in the map. We measured the second one on graphene,
    $18\times18$: the deviation from the threefold symmetry of $|\delta T(q)|$ was
    0.35 at $\eta = 0.03$ Ry and 5e-3 at $\eta = 0.08$ Ry. Graphene at 1.1 eV
    from the Dirac point with $\eta = 0.06$ Ry gives no QPI pattern at all,
    because the contour is smeared out and 96% of $|\delta\rho|^2$ is within 4 bohr
    of the impurity. `qpi.wraparound` and `qpi.contour_points` are the numbers to
    check. The Green's function and the impurity live in the computed bands, as in
    a Wannier model, so $T$ depends on `nbnd`. For graphene it moves from 1.90
    to 1.70 Ry between 12 and 20 bands. For the hydrogen sheet the QPI pattern
    changes by 1e-3 between 32 and 128 bands with the on-shell exit leg. With the
    literal $1/(\omega - \varepsilon + i\eta)$ on the exit leg
    (`exit_method="resolvent"`) it changes by 4e-2 between 32 and 64 bands and
    does not converge, as in the vertical transport. A spinor run, a spin spiral and an
    ultrasoft or PAW dataset give an error, the last one because the overlap
    $\langle\psi|\phi\rangle$ uses the pseudo-wavefunction, which is not the true
    one inside the augmentation sphere. A symmetry-reduced k-mesh and a k-mesh
    with more than one point along the normal give an error, and `grid` builds
    the full mesh. The impurity is one Gaussian, and a tip and an exit plane in
    the vacuum are needed, as for the vertical transport.

The notebooks for this page are
[40, STM images](https://github.com/joselado/defuqumat/blob/master/notebooks/40_stm_images.ipynb),
[41, vertical transport](https://github.com/joselado/defuqumat/blob/master/notebooks/41_vertical_transport.ipynb)
and
[55, quasiparticle interference](https://github.com/joselado/defuqumat/blob/master/notebooks/55_quasiparticle_interference.ipynb).
