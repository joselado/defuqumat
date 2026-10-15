# STM images and tunnelling

In this page we compute what a scanning-tunnelling microscope sees. We start with
the image of a surface at one bias, then we compute the spectrum $\mathrm dI/\mathrm dV$
at one place, and finally we follow the electron through a two-dimensional
material into a substrate below it, first in real space and then in momentum
space.

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
less. So a microscope sees one bright spot per cell, on B, which is a triangular
lattice and not the honeycomb of the crystal. This is the reason why only half of
the atoms of graphite appear in an STM image.

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
spots are on B, A is dimmer, and the centres of the hexagons are dark, which is
the triangular lattice of the experiment. On the right we follow the tunnelling
density from the top layer up to the middle of the vacuum, 6.5 bohr. Above A and
above B it falls by a factor of 10 every 1.16 bohr, and B stays brighter than A
up to 6 bohr, so the contrast does not depend on the height of the tip. Above
the centre of the hexagon the density stays below $10^{-5}$ at every height. You
can think of it this way: the states at the Fermi level are the states near K,
and by the threefold symmetry their wavefunction is zero on the axis through the
centre of the hexagon. At 6.5 bohr the three curves meet, near
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
cell. On the right we see the zoom near the Fermi level. The spectrum has a
minimum at $V = 0$, which is the semimetal, and only there A and B differ, with B
up to 1.63 times A between $-0.1$ and $+0.1$ V. The width of the delta is
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

!!! note "Limits"
    The STM image gives an error for a spin spiral, for a run with a fixed total
    magnetization (`tot_magnetization`, which has one Fermi level per spin), and
    for a run in an applied magnetic field. A magnetic tip gives an error on a run
    with no magnetization, and a tip direction across the axis of a collinear run
    also gives an error, because such a run has no transverse magnetization. A
    constant-current scan longer than the period of the cell along the normal
    gives an error, and a point where the tip never reaches the set-point comes
    back as `nan`, with a warning. At zero bias the image of an insulator is zero,
    because the delta sits in the gap, so for an insulator you give a `bias`. All
    four calculations of this page give an error with the half-sphere storage of
    `K_POINTS gamma`, and an explicit $k = 0$ gives the same states.

    The tunnelling spectrum gives the same errors, and two more. A
    symmetry-reduced k-mesh gives an error, so you pass `grid`, which builds the
    full mesh, or run with `nosym`. A tip inside the augmentation sphere of an
    ultrasoft or PAW atom gives an error, because there the pseudo-wavefunction is
    not the true one. A tip above a surface, in the vacuum, is always fine. A
    `bias` together with `energies`, and the constant-current mode, also give an
    error. `sts.current` gives an error if the step of the energy axis is larger
    than `width`. For a run with no Fermi level, the zero of bias is the middle of
    the gap, with a warning.

    The vertical transport gives an error for a k-mesh with more than one point
    along the stacking axis, for a symmetry-reduced k-mesh (`grid` builds the full
    one), for a substrate plane that is not parallel to the layers, and for a tip
    or a substrate plane inside an augmentation sphere. A magnetic tip or
    substrate gives an error on a run with no magnetization, and a direction
    across the axis of a collinear run gives an error for both. A spin spiral, a
    fixed `tot_magnetization` and an applied field give an error, as for the image.
    A `bias` window whose energy step is larger than `broadening` gives an error,
    since it would be wrong by orders of magnitude. If the atoms are not between
    the two planes, the code warns. The tip is an s-wave tip, the substrate is
    infinite and flat, and the map has no absolute scale.

    The momentum-resolved transport gives the same errors, and also for a
    magnetic tip plane and for a tip plane that is not parallel to the layers.

The notebooks for this page are
[40, STM images](https://github.com/joselado/defuqumat/blob/master/notebooks/40_stm_images.ipynb)
and
[41, vertical transport](https://github.com/joselado/defuqumat/blob/master/notebooks/41_vertical_transport.ipynb).
