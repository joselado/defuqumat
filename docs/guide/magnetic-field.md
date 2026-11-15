# Landau levels in a magnetic field

In this page we compute what a uniform orbital magnetic field does to the electrons
of a crystal. The field bends their orbits into circles, and the energies become
Landau levels. In a crystal the levels also feel the lattice, which gives the
Hofstadter bands and their Chern numbers. Scanning tunnelling spectroscopy,
cyclotron resonance and the quantum Hall effect measure these levels. We take
graphene as the main example, and diamond for the magnetizability.

The field acts on the orbital motion only. The Zeeman term of the spin is not in
these results, so each level holds one electron per spin.

## The magnetic cell

What happens to Bloch's theorem in a field? The vector potential $\mathbf A(\mathbf r)$
of a uniform field is not periodic, so a lattice translation is no longer a symmetry.
A translation followed by a gauge change still is. These are the magnetic
translations, and two of them commute when the flux through the cell they span is a
whole number of flux quanta $\Phi_0 = h/e$.

So, at a flux $p/q$ flux quanta per cell, with $p$ and $q$ coprime, the magnetic
cell is $q$ chemical cells. The code puts them along $\mathbf a_1$, and the field is
along $\mathbf a_3$. The states are magnetic Bloch states, labelled by a momentum in a
zone $q$ times smaller. We use the Landau gauge

$$
\mathbf a(\mathbf r) = 2\pi \frac{p}{q}\, (\mathbf b_1\cdot\mathbf r)\, \mathbf b_2 ,
$$

which is $(e/\hbar)\mathbf A$ written with the reciprocal vectors $\mathbf b_i$ of the
cell ($\mathbf a_i \cdot \mathbf b_j = \delta_{ij}$).

The flux per cell is large for normal fields, because the cell is small. One flux
quantum through the graphene cell ($a = 2.46$ Å) is 78912 T, so 10 T is
$p/q = 1/7891$. Through the $3\times1\times1$ conventional cell of diamond
($a = 3.56$ Å) it is 10877 T. The code takes a field in tesla and rounds it to the
nearest flux $p/q$. The flux and the field that were used are on the result.

## Three routes, and when to use each

- **Peierls substitution on Wannier functions.** This is the route for laboratory
  fields. We build a tight-binding model of the bands with
  [Wannier functions](wannier.md), and each hopping $t(\mathbf R)$ picks up the phase
  $e^{-i\int \mathbf a\cdot d\mathbf l}$ along its bond. The magnetic cell then has
  $q$ times the number of Wannier functions, a sparse matrix that we solve near the
  energy we want. At 10 T on graphene this is 15782 rows for a $p_z$ model, solved in
  about 5 s. It is exact for orbitals much smaller than the magnetic length
  $\ell_B = \sqrt{\hbar/eB}$, which is 8.1 nm at 10 T. `calc.get_landau_levels()`.
- **Plane waves in the magnetic cell.** This is the exact route, with no model, as Lu
  et al. do it (PRL 135, 196401 (2025)). The kinetic term is $(-i\nabla + \mathbf a)^2$
  on a real-space grid of the whole magnetic cell, and the projectors of the
  pseudopotential are dressed by the field (the GIPAW phase), so that they move with
  the magnetic translations. It also runs self-consistently in the field. Its cost
  grows with $q$, so it is for very high fields: we ran it up to $q = 10$ on graphene,
  7891 T. `calc.get_magnetic_bands()` and `calc.get_magnetizability()`.
- **The magnetic ultracell** of Wang and Vafek (PRB 108, 245109 (2023)). It builds the
  magnetic Bloch states from the zero-field Bloch states, so it would reach
  $q$ of a few hundred. Today it gives an error. Its basis, the hybrid Wannier states of
  the lowest bands, is not localized when the bands are entangled, and then the
  matrix is not Hermitian. We measured $2\times10^{-2}$ of its largest element at
  $q = 10$, where the code asks for less than $10^{-8}$. `calc.get_landau_levels(route="ultracell")`.

## Graphene at 10 T against the Dirac formula

We start with the simplest case, graphene. Near the Dirac energy $E_D$ the electrons
are massless, with velocity $v_F$, and their Landau levels are

$$
E_n - E_D = \mathrm{sgn}(n)\, v_F \sqrt{2 e\hbar B |n|}, \qquad n = 0, \pm1, \pm2, \dots
$$

meaning that the levels grow as $\sqrt{|n| B}$ and that there is a level at $E_D$ for
any field. We build a $p_z$ model of the DFT bands, two Wannier functions from 16
bands with the states within 3 eV of the Fermi level kept exactly, and we compute the
levels at 10 T with `calc.get_landau_levels()`. With `dirac=True`, the default, the
result also has the Dirac formula at the velocity of the same model at $K$:

```python
import numpy as np
from defuqumat import Calculator
from defuqumat.units import RY_TO_EV

graphene = Calculator.from_keywords(
    lattice="hexagonal", a=2.46, c=15.0, atoms=[("C", (0, 0, 0)), ("C", (1 / 3, 2 / 3, 0))],
    pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=50 * RY_TO_EV, kmesh=(12, 12, 1), smearing=0.01 * RY_TO_EV, smearing_type="mv")
ef = graphene.get_scf().fermi_energy * RY_TO_EV
pz = graphene.get_wannier(2, ["C:pz"], num_bands=16, dis_froz=(ef - 3, ef + 3), symmetry=False,
                          engine_options=dict(num_iter=1000, dis_num_iter=1000))
ll = graphene.get_landau_levels(B=10.0, wannier=pz, nlevels=44)
row = ll.dirac["per_k"][0]
print(f"flux {ll.flux}, B = {ll.field_tesla:.3f} T, dimension {ll.dim}")
print(f"v_F = {ll.dirac['v_F']:.4e} m/s, E_1 Dirac = {ll.dirac['E1_dirac_ev'] * 1e3:.2f} meV")
for n in (1, 2, 3, 10, -1, -2, -3, -10):
    print(f"n = {n:3d}: E_n - E_D = {row['energies_ev'][n] * 1e3:8.2f} meV, ratio {row['ratios'][n]:.5f}")
print(f"zero level {row['zero_level_ev'] * 1e6:.1f} ueV, {row['zero_multiplicity']} states, "
      f"site traces {np.round(row['zero_site_traces'], 6)}")
print(f"Hermiticity {ll.hermiticity.max():.1e} Ry, plaquette {ll.plaquette}")
```

```
flux (1, 7891), B = 10.000 T, dimension 15782
v_F = 8.4207e+05 m/s, E_1 Dirac = 96.62 meV
n =   1: E_n - E_D =    96.48 meV, ratio 0.99860
n =   2: E_n - E_D =   136.33 meV, ratio 0.99776
n =   3: E_n - E_D =   166.85 meV, ratio 0.99705
n =  10: E_n - E_D =   303.43 meV, ratio 0.99314
n =  -1: E_n - E_D =   -96.75 meV, ratio 1.00136
n =  -2: E_n - E_D =  -136.87 meV, ratio 1.00168
n =  -3: E_n - E_D =  -167.66 meV, ratio 1.00186
n = -10: E_n - E_D =  -306.18 meV, ratio 1.00213
zero level -66.7 ueV, 2 states, site traces [1. 1.]
Hermiticity 3.9e-16 Ry, plaquette (0.999999682995351-0.0007962469450318828j)
```

The first level is within 0.15 per cent of the Dirac formula. The ratio moves away
from one slowly as $|n|$ grows, and not in the same way for electrons and holes. This
is the band structure of graphene beyond the Dirac cone, which a pure Dirac model does
not have. The two zero modes sit at $E_D$, one state on each sublattice, as in the
Dirac picture. The Hermiticity residual is round-off. The plaquette is the phase of a
loop around one cell, $e^{-i 2\pi p/q}$, which checks the sign and the size of the
flux. In the figure the Dirac formula is a straight line.

![The Landau levels of graphene at 10 T lie on the straight line of the Dirac formula, with two zero modes at the Dirac energy.](img/landau_graphene.png#only-light)
![The Landau levels of graphene at 10 T lie on the straight line of the Dirac formula, with two zero modes at the Dirac energy.](img/landau_graphene-dark.png#only-dark)

The levels at $|n| \le 10$, against $\mathrm{sgn}(n)\sqrt{|n|}$. Each level holds two
states per magnetic k-point, one per valley, and the valleys stay degenerate to
$10^{-12}$ eV with this model.

??? example "Code of this figure (docs/figures/landau_graphene/)"

    ```python title="compute.py"
    --8<-- "landau_graphene/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "landau_graphene/plot.py"
    ```

The magnetic cell is a torus by default: periodic, with the magnetic translations.
A ribbon is the other choice, `geometry="ribbon"`, open along $\mathbf a_1$. Its width is
`width` cells, and ten magnetic lengths if you do not give it. On a ribbon the momentum
$\kappa_2$ places the orbit: its centre is about $q(K_2 - \kappa_2)$ cells from the first
cell, so a ribbon narrower than $q$ cells holds the bulk levels only for $\kappa_2$ near
the $K_2$ of a valley. Here we put the orbit in the middle of a ribbon of 330 cells:

```python
k2 = ll.dirac["K"][1] - 165 / ll.flux[1]
ribbon = graphene.get_landau_levels(B=10.0, wannier=pz, nlevels=20, geometry="ribbon", width=330,
                                    kappa=((0.0, k2),))
print(f"ribbon of {ribbon.dim // 2} cells at kappa2 = {k2:.5f}: levels (meV from E_D)",
      np.round(np.sort(ribbon.levels[0] - ll.dirac["E_D"]) * RY_TO_EV * 1e3, 2)[7:13])
```

```
ribbon of 330 cells at kappa2 = 0.31242: levels (meV from E_D) [-1.3687e+02 -9.6750e+01 -7.9600e+00 -7.0000e-02  9.6480e+01  1.3633e+02]
```

The levels $n = \pm1$ and $\pm2$ are those of the torus to the printed digits, with 660
rows instead of 15782, and one state sits at $E_D$. The torus has no level at $-8$ meV.
We did not check which state this is, the second zero mode moved by the walls or a
state at an edge. The ribbon must be wide enough. With 165 cells, five magnetic lengths, the walls are 2.5 $\ell_B$ from
the orbit, and the level $n = 1$ moves up by 4 meV (measured with the 8-function Wannier set).

## The Hofstadter butterfly

What happens when the flux per cell is not small? The Landau levels broaden into
bands, and the spectrum against the flux becomes the Hofstadter butterfly. Here it is
for the nearest-neighbour honeycomb model ($t = -2.7$ eV), with every flux $p/q$ up to
$q = 50$:

![The Hofstadter butterfly of the honeycomb model: Landau levels at small flux, a fractal set of bands at large flux.](img/landau_butterfly.png#only-light)
![The Hofstadter butterfly of the honeycomb model: Landau levels at small flux, a fractal set of bands at large flux.](img/landau_butterfly-dark.png#only-dark)

At small flux, near the left and right edges, the bands are the Landau levels, and
near $E = 0$ they fan out as $\sqrt{n B}$, as in the graphene figure above. The
spectrum is symmetric under $E \to -E$ (the two sublattices) and under
$p/q \to (q-p)/q$. The second symmetry holds only for the whole spectrum over the
magnetic zone, not at each k-point: we checked it to $10^{-14}$ eV on a $6\times6$
mesh, while at a single k-point the two spectra differ by 1.4 eV.

??? example "Code of this figure (docs/figures/landau_butterfly/)"

    ```python title="compute.py"
    --8<-- "landau_butterfly/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "landau_butterfly/plot.py"
    ```

## The Landau problem on plane waves, and the Chern number of a level

We now use the exact route. As a check, we switch off every potential and solve free
electrons in the graphene box at $p/q = 1/10$, which is 7891 T. The levels must be
$(2n+1)B_{ph} + (2\pi m/c)^2$ in Rydberg units, with $B_{ph} = 2\pi\, (p/q) / A$ the
field in units of the cell area $A$. These are the Landau levels times the standing
waves along the box of height $c$. We compute them with `calc.get_magnetic_bands()`:

```python
free = graphene.get_magnetic_bands(flux=(1, 10), kmesh=(1, 1), nbnd=6, free=True, local_only=True)
print(f"B = {free.field_tesla:.1f} T, B_ph = {free.b_ph:.8f} Ry, ndim {free.ndim}")
print("levels / B_ph:", np.round(free.eigenvalues[0] / free.b_ph, 8))
```

```
B = 7891.2 T, B_ph = 0.03357231 Ry, ndim 92160
levels / B_ph: [1.         2.46351843 2.46351843 3.         4.46351843 4.46351843]
```

The levels are 1, 3 for $m = 0$, and $1 + (2\pi/c)^2/B_{ph}$ twice for $m = \pm1$, to the
printed digits. On a $4\times4$ magnetic mesh with 12 bands every level is within
$3\times10^{-12}$ Ry of the formula and flat in k to $3\times10^{-12}$ Ry. At
$p/q = 2/5$ each level holds two states, $p$ in general.

Each Landau level is a band with a Chern number, and `chern=True` gives the Chern
number of every isolated group of bands, from the overlaps on the closed mesh. On the
free problem at $1/10$ each of the three lowest $m = 0$ levels has $C = +1$. With the
whole Hamiltonian of graphene at one flux quantum per cell ($p/q = 1/1$, 78912 T,
$12\times12$ mesh, 8 bands), the lowest four bands carry $+1$, $0$, $-3$ and $+6$, and
the third and fourth together $+3$. This is the pattern of Lu et al. (1, 0, and 3 for
the pair), as exact integers, and the same on a $24\times24$ mesh. The call is
`graphene.get_magnetic_bands(flux=(1, 1), kmesh=(12, 12), nbnd=8, chern=True)`, about
6 minutes on one core.

With the crystal potential on (the default, `local_only=False`), the operator is the
full Hamiltonian at the zero-field potential. `scf=True` runs the self-consistent field
in the field instead, from the zero-field density: graphene at $1/6$ (13152 T)
converges to `conv_thr = 1e-8` in 7 iterations, and at $1/10$ in 6. Inside, the code
solves only $N_2/q$ of the k-points of the mesh and gets the others by a translation,
which is exact: the density from the reduced set agrees with the density from the full
mesh to $1.4\times10^{-12}$.

## Never read a magnetic band at a named k-point

In a field the energy of a magnetic Bloch state at a given $\mathbf k$ is not a property
of the crystal alone. It depends on where the origin of the gauge is. We moved all the
atoms of graphene by $\mathbf a_1/2$ at one flux quantum per cell: the energies at the
same named k-points moved by up to 0.291 Ry (4 eV), while the set of all levels over the
mesh moved by $10^{-12}$ Ry and the band energy by $2\times10^{-14}$ Ry. So compare
the sorted levels over a closed mesh, a density of states, or a total energy, and never
$\varepsilon_n(\mathbf k)$ at one $\mathbf k$. The $\kappa$ of `get_landau_levels` is
there to sample the magnetic zone, not to name a point of it.

## The magnetizability of diamond

How much does a field lower or raise the energy of an insulator? To second order the
change is $-\frac{1}{2}\chi B^2$, with $\chi$ the orbital magnetizability. We follow Lu
et al.: three self-consistent runs at $-1$, $0$ and $+1$ flux quantum through the
supercell, on the same magnetic box, and

$$
\chi = -\frac{E(+\Delta B) - 2E(0) + E(-\Delta B)}{\Delta B^2} .
$$

We take their cell, the $3\times1\times1$ conventional cell of diamond (24 atoms,
$\Delta B$ = 10877 T), at 60 Ry with a $2\times6\times6$ mesh, and we compute it with
`calc.get_magnetizability()`:

```python
a = 3.56
base = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)]
sites = base + [(x + .25, y + .25, z + .25) for x, y, z in base]
diamond = Calculator.from_keywords(
    lattice=[[3 * a, 0, 0], [0, a, 0], [0, 0, a]],
    atoms=[("C", ((x + j) / 3, y, z)) for j in range(3) for x, y, z in sites],
    pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=60 * RY_TO_EV, kmesh=(2, 6, 6))
diamond.get_scf(conv_thr=1e-11)
chi = diamond.get_magnetizability(kmesh=(2, 6, 6), nbnd=52, conv_thr=1e-8)
print(f"E(+) - E(-) = {chi.asymmetry:.3e} Ry")
print(f"chi = {chi.chi_per_unit:.3f} x 1e-6 cm^3/mol per C2, {chi.chi_per_atom:.3f} per C atom")
```

```
E(+) - E(-) = 1.631e-07 Ry
chi = -11.401 x 1e-6 cm^3/mol per C2, -5.700 per C atom
```

Lu et al. give $-11.16\times10^{-6}$ cm³/mol per C₂ with the LDA, and we use PBE, so
the two agree to about 2 per cent. The asymmetry $E(+) - E(-)$ must be zero by time
reversal. At `conv_thr = 1e-8` it is at the level the SCF resolves the energy, about
$10^{-7}$ Ry, against $E(+) - E(0) = 6.2\times10^{-2}$ Ry. The three runs took 95
minutes on four cores. Note that Lu et al. find that the value depends on the supercell
($-10.96$ for the $4\times1\times1$ cell), because $\Delta B$ is one flux quantum
through it.

The run at $-1$ is a control, because time reversal fixes $E(-) = E(+)$ for every
case this route accepts. If you trust it, `time_reversal_check=False` skips that run,
which saves about 40 per cent of the time, and then `chi.asymmetry` is `None`. If one of
the field runs does not converge, `get_magnetizability` gives an error, because the total
energy of an unconverged run is not variational.

## How the routes compare

At 7900 to 13000 T we compared the plane-wave route with Peierls on the Wannier models,
on the same graphene potential. The two zero modes of the plane-wave route sit 0.20 to
0.22 eV below those of Peierls, which is 8 per cent of the spacing $E_1$ at these
fields, and the ratio stays the same while the field changes by a factor 1.7. The
cutoff of the plane-wave box, the nonlocal term and a rigid offset do not explain it,
so where the difference comes from is still open. At 10 T the expansion parameter of
Peierls, (orbital size$/\ell_B)^2$, is about $10^{-4}$, so this comparison does not
measure the error of Peierls at laboratory fields.

!!! note "Limits"
    - The field is along $\mathbf a_3$ only. Put the direction you want along the third
      lattice vector.
    - The Zeeman term is not in: `zeeman=True` gives an error. Each level holds one
      electron per spin.
    - A spinor run (spin-orbit coupling, noncollinear magnetism), `nspin = 2`,
      ultrasoft and PAW datasets, DFT+U, a spin spiral, a meta-GGA and `K_POINTS gamma`
      give an error, in all three routes.
    - The current-density functional is not in: the exchange-correlation potential
      is that of the density alone. Forces and stress in the field are not computed.
    - Peierls substitution keeps only the bands of the Wannier model, and its phase is
      exact only for orbitals much smaller than $\ell_B$. The intra-cell term it drops
      is not measured at laboratory fields (see "How the routes compare").
    - The magnetic ultracell (`route="ultracell"`) gives an error on every case we ran,
      because its basis is not localized for entangled bands. It takes only `flux`,
      $q \le 100$, and needs `nbnd`.
    - The plane-wave route grows with $q$. The memory of one k-point is estimated
      before the run, and a run above the limit gives an error. Measured on one core
      for graphene at 50 Ry: the SCF at $q = 6$ in 525 to 644 s and 1.9 to 2.2 GB (two
      runs at different loads), at $q = 10$ in 1631 s and 2.9 GB. At $q = 30$
      the estimate is 10 GB.
    - The plane-wave box is a parallelepiped with its own Nyquist frequency, so it is
      a slightly larger basis than the sphere of the same cutoff, and it splits the
      $\Gamma$ doublet of graphene by $2\times10^{-3}$ Ry at 50 Ry. Compare energies on
      the same box, as `get_magnetizability` does.
    - Time reversal, $H(-B, -\mathbf k) = H(B, \mathbf k)^*$, is exact on the grid only
      for a box with an odd number of points on every axis.
    - The occupations are smearing or fixed. Tetrahedra give an error. With fixed
      occupations the gap is checked over the whole mesh.
    - A Peierls model from a Wannier set of one spin channel gives an error.
