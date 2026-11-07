# Density of states

In this page we compute the density of states, the number of states per unit
energy, and then its projection on the atomic orbitals, which tells us which atom
and which orbital each state comes from. At the end we compute the density of
states with no band at all, by the kernel polynomial method.

## The density of states

We start with the density of states itself, which is

$$
D(E) = \sum_n \int_{\rm BZ} \frac{d\mathbf k}{(2\pi)^3}\,
  \delta(E - \varepsilon_{n\mathbf k})
  \;\approx\; \sum_{n\mathbf k} w_{\mathbf k}\,\delta(E - \varepsilon_{n\mathbf k}).
$$

It says how many states the crystal has at energy $E$. This is what tunnelling
spectroscopy (STS) and photoemission measure, and its value at the Fermi level
enters the specific heat and the Pauli susceptibility of a metal. Its integral up
to $E$,

$$
N(E) = \int_{-\infty}^{E} D(E')\,dE',
$$

counts the states below $E$, so at the Fermi level $N$ is the number of valence
electrons.

On a finite k-mesh the delta function is a problem, and we can treat it in two
ways. **Smearing** replaces the delta function by a peak of finite width
`degauss`: a Gaussian, or the Methfessel-Paxton, Marzari-Vanderbilt or
Fermi-Dirac functions. **Tetrahedra** interpolate the bands linearly between the
k-points and integrate exactly inside each tetrahedron, so they give no
artificial width. Smearing is the normal choice for a metal inside the SCF,
because it makes the occupations smooth. Here we take silicon with tetrahedra:

```python
from defuqumat import Calculator

calc = Calculator.from_file("docs/figures/bands_silicon/si.in",
                            pseudo_dir="tests/data/pseudo")
dos = calc.get_dos(grid=(12, 12, 12), nbnd=8, scheme="tetrahedra")

print(f"{dos.states_below(dos.fermi_energy):.4f} states below the top of the valence band")
```

```
8.0000 states below the top of the valence band
```

A density of states is a non-self-consistent run. We keep the density of the SCF
and compute the bands again on the denser `grid`, so the DOS can use many more
k-points than the SCF. If you do not give `grid`, the code uses the k-points of
the SCF, which are rarely enough.

The keywords of `get_dos` are:

- **`grid`**: the Monkhorst-Pack mesh of the DOS, for example `(12, 12, 12)`.
  It is reduced with the symmetry of the crystal.
- **`nbnd`**: the number of bands. Add empty bands to see the conduction band.
- **`scheme`**: `"tetrahedra"`, `"tetrahedra_opt"`, `"tetrahedra_lin"`,
  `"gaussian"`, `"m-p"`, `"m-v"` or `"f-d"`. If you do not give it, the code
  uses the occupation scheme of the SCF.
- **`degauss`**: the width of the smearing, in Ry.
- **`emin`, `emax`, `delta_e`**: the energy window and its step, in Ry.

`dos.energies` is in Ry and `dos.dos` in states per Ry per cell, and
`dos.energies_ev` and `dos.dos_ev` are the same in eV. The spin degeneracy is
included. For a spin-polarized run, `dos.dos` has a first axis for the two spin
channels. `dos.at(E)` gives $D(E)$ and `dos.states_below(E)` gives $N(E)$, both
interpolated, and `dos.plot()` draws the DOS with the zero at the Fermi level.

![The density of states of silicon and of aluminium.](img/dos_silicon_aluminium.png#only-light)
![The density of states of silicon and of aluminium.](img/dos_silicon_aluminium-dark.png#only-dark)

On the left we see silicon on a $12\times12\times12$ mesh. The grey band is the
gap between the highest valence level and the lowest conduction level of this
mesh. With tetrahedra, the DOS in the gap is exactly zero. A Gaussian of 0.02 Ry
puts states into the gap, 0.02 states/eV at mid-gap, and these states do not
exist. The Gaussian also rounds the sharp peaks, which are the van Hove
singularities of the bands. So for an insulator, tetrahedra are the better choice.

On the right we see aluminium on a $16\times16\times16$ mesh, with tetrahedra. Up
to about 6 eV below the Fermi level, the DOS follows $\sqrt{E - E_0}$, the DOS of
free electrons in three dimensions (dashed). This is why aluminium is the
standard example of a nearly-free-electron metal. Near the Fermi level, the
deviations come from the gaps that the lattice potential opens at the zone
boundary. The integral of the DOS up to the Fermi level is 3.000001, the three
valence electrons of aluminium.

??? example "Code of this figure (docs/figures/dos_silicon_aluminium/)"

    ```python title="compute.py"
    --8<-- "dos_silicon_aluminium/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "dos_silicon_aluminium/plot.py"
    ```

## The projected density of states

Let us now ask which atom and which orbital a state belongs to. We answer it by
projecting each Kohn-Sham state on the atomic orbitals $\phi_i$, which are in the
pseudopotential file:

$$
D_i(E) = \sum_{n\mathbf k} w_{\mathbf k}\,
  \left|\langle\tilde\phi_i|\hat S|\psi_{n\mathbf k}\rangle\right|^2\,
  \delta(E - \varepsilon_{n\mathbf k}),
\qquad
|\tilde\phi\rangle = O^{-1/2}|\phi\rangle,\quad
O_{ij} = \langle\phi_i|\hat S|\phi_j\rangle .
$$

We first make the orbitals orthonormal, which is the Löwdin construction, so the
weights of one state add up to at most one. $\hat S$ is the overlap operator of
an ultrasoft or PAW dataset, and the identity for a norm-conserving one. This is
the same projection as `projwfc.x` of Quantum ESPRESSO. The integral of $D_i$ up
to the Fermi level is the Löwdin charge of orbital $i$. The part of the occupied
states that the atomic orbitals cannot describe is the spilling parameter:

$$
\text{spilling} = 1 - \frac{1}{N_{\rm elec}}\sum_i\int^{E_F} D_i(E)\,dE .
$$

A small spilling means that the atomic orbitals are a good language for the
bonds of this crystal. The orbital character of the states is what X-ray
absorption and resonant photoemission see. Here we take silicon with an
ultrasoft dataset:

```python
silicon = Calculator.from_file("docs/figures/dos_pdos/si-us.in",
                               pseudo_dir="tests/data/pseudo")
pdos = silicon.get_pdos()

print(pdos.charges.format(("Si",)))
s = pdos.select(l="s")              # both atoms, states/Ry
p1 = pdos.select(atom=0, l="p")     # the first atom only
```

```
Lowdin Charges:
     Atom #   1 (Si): total charge =   3.9647
       s =   1.1596
       p =   2.8051, pz=  0.9350, px=  0.9350, py=  0.9350
     Atom #   2: total charge =   3.9647
       s =   1.1596
       p =   2.8051, pz=  0.9350, px=  0.9350, py=  0.9350
     Spilling Parameter:   0.0088
```

`get_pdos` takes the same `grid`, `nbnd`, `scheme` and `degauss` as `get_dos`.
`pdos.select(atom=..., l=..., species=..., m=...)` sums the channels that match,
and `l` can be a number or a letter. `pdos.total` is the normal DOS on the same
energies. `pdos.charges` holds the Löwdin charges: `charges.total` per atom,
`charges.charges` per atom and $l$, and `charges.spilling`. `pdos.plot(by="l")`
draws the projections by angular momentum, and you can also choose `by="shell"`
or `by="species"`. For a spin-orbit run, the channels are resolved by $j$ and
$m_j$ instead of $m$.

![The projected density of states of silicon, by angular momentum.](img/dos_pdos.png#only-light)
![The projected density of states of silicon, by angular momentum.](img/dos_pdos-dark.png#only-dark)

The figure shows silicon with an ultrasoft dataset, on a $12\times12\times12$
mesh with tetrahedra, and the grey area is the total DOS. We see that the bottom
of the valence band, from $-12$ to about $-8$ eV, is mostly $s$, and the top,
from about $-4$ eV to the gap, is almost only $p$. In between, the two mix. This
is the $sp^3$ picture of a covalent semiconductor, here read from the
projections. The conduction band is mostly $p$ too. The coloured curves sum to
slightly less than the total, and the difference is the spilling, 0.9% of the
valence charge here.

??? example "Code of this figure (docs/figures/dos_pdos/)"

    ```python title="compute.py"
    --8<-- "dos_pdos/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "dos_pdos/plot.py"
    ```

## The density of states without bands

So far the DOS came from the bands, so we needed every band in the energy window.
The cost of the bands grows with the square of their number, and in a large cell
there are many of them. We now compute the same DOS with no band at all, by the
kernel polynomial method (KPM). The DOS is a trace,
$D(E) = \sum_{\mathbf k} w_{\mathbf k}\,{\rm Tr}\,\delta(E - H_{\mathbf k})$,
taken in the plane-wave basis. We rescale the Hamiltonian to $H' = (H - c)/s$, so
that its whole spectrum is inside $[-1, 1]$, and we expand the delta function in
the Chebyshev polynomials $T_n$:

$$
D(E) = \frac{1}{\pi s\sqrt{1-x^2}}\Big[g_0\mu_0 + 2\sum_{n\ge1} g_n\,\mu_n\,T_n(x)\Big],
\qquad \mu_n = \sum_{\mathbf k} w_{\mathbf k}\,{\rm Tr}\,T_n(H'_{\mathbf k}),
$$

with $x = (E - c)/s$. The polynomials obey $T_{n+1}(H') = 2H'T_n(H') - T_{n-1}(H')$,
so each moment $\mu_n$ costs one application of $H$ to a vector, and nothing is
diagonalized. We take the trace as an average over a few random vectors, so the
DOS has a statistical error, which falls as one over the square root of their
number. The factors $g_n$ are the Jackson kernel, which turns the cut series of a
delta function into a positive peak, close to a Gaussian. Its width is the
broadening, and it sets how many moments we need.

We compute it with `calc.get_kpm_dos()`:

```python
from defuqumat import Calculator

calc = Calculator.from_keywords(
    lattice="fcc", a=5.398, atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
    pseudos={"Si": "Si.pz-vbc.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=217.7, kmesh=(6, 6, 6), kshift=(1, 1, 1))
top = calc.get_scf().homo
kpm = calc.get_kpm_dos(broadening=0.01, n_random=16)

print(f"{kpm.n_moments} moments on [{kpm.centre - kpm.half_width:.2f}, "
      f"{kpm.centre + kpm.half_width:.2f}] Ry")
print(f"{kpm.count_at(top + 0.02):.2f} +- {kpm.count_error_at(top + 0.02):.2f} "
      "states below the gap")
```

```
1610 moments on [-0.74, 16.10] Ry
7.89 +- 0.15 states below the gap
```

The count is 8, the four valence bands with their spin, within its error. With
an `ecut` of 16 Ry the interval reaches 16 Ry, so we need 1610 moments for a
broadening of 0.01 Ry. The
keywords of `get_kpm_dos` are:

- **`broadening`**: the half width at half maximum of the peak, in Ry, at
  `reference_energy`. It sets the number of moments. The Jackson peak narrows
  towards the edges of the spectrum, and `kpm.broadening_at(E)` gives it at any
  energy.
- **`n_moments`**: the number of moments, if you want to give it instead of
  `broadening`.
- **`n_random`**: the number of random vectors per k-point. The error falls as
  one over its square root.
- **`seed`**: the seed of the random vectors, kept on the result.
- **`kernel`**: `"jackson"` (the default), `"lorentz"`, which gives a Lorentzian
  peak, or `"fejer"`.
- **`trace`**: `"stochastic"`, or `"exact"`, which uses every plane wave as a
  start vector. The exact trace has no error, and it is for small cells.
- **`emin`, `emax`, `delta_e`**: the energy window and its step, in Ry. If you do
  not give them, the window goes from below the lowest band to 1 Ry above the
  Fermi level (the top of the valence band for an insulator).

`kpm.dos` is in states per Ry per cell and `kpm.dos_error` is its standard error.
`kpm.count_at(E)` and `kpm.count_error_at(E)` give $N(E)$ and its error. The moments
stay on the result, so `kpm.reconstructed(n_moments=..., kernel=...)` gives the DOS
at a coarser broadening, or with another kernel, with no new calculation.

![The KPM density of states of silicon, near the gap and over the whole basis.](img/dos_kpm.png#only-light)
![The KPM density of states of silicon, near the gap and over the whole basis.](img/dos_kpm-dark.png#only-dark)

On the left we see silicon on a $4\times4\times4$ mesh with a broadening of
0.01 Ry, and the DOS of 12 bands with a Gaussian of the same half width
(dashed). The two agree within the shaded band of two standard errors. On the
right we use the same moments over the whole interval, cut to 200 moments. This
is the DOS of every state of the basis, and a band calculation would need about
180 bands per k-point to draw it. Far above the gap the states are almost free
electrons, so the DOS follows $\Omega\sqrt{E - V_0}/(2\pi^2)$ (dashed, in Ry units
with the spin), with $V_0$ the average potential. It stops at $V_0$ plus the
cutoff, because the basis has no plane wave above it.

When does it pay? In a plane-wave basis the top of the spectrum is set by the
cutoff, so the interval is wide, and a broadening of 0.01 Ry needs about a
thousand moments. For a small cell the bands are cheaper. The cost of KPM does not
depend on the number of bands, and the number of random vectors for a given error
falls as one over the number of atoms, so KPM wins for large cells. We measured
silicon with one k-point, at a broadening of 0.01 Ry and a cutoff of 12 Ry, on one
core:

| cell | from the bands | KPM, 8 random vectors | error with 8 vectors | vectors for a 10% error | KPM at a 10% error |
|---|---|---|---|---|---|
| 16 atoms | 1.6 to 8.1 s | 2.4 s | 27% | 58 | 15.5 s |
| 64 atoms | 32 to 85 s | 14.6 s | 14% | 16 | 16.6 s |
| 128 atoms | 275 to 375 s | 43.6 s | 10% | 8 | 37 to 44 s |

The time of the bands moves with the number of empty bands and with the load of the
machine, so we give the range we measured. The error is the rms difference from the
DOS of the bands, over the valence band, relative to the mean DOS, averaged over
eight seeds of the random vectors. It falls as one over the square root of the
number of atoms, so a bigger cell needs fewer vectors. With 16 atoms the bands are
faster, with 64 and more atoms the KPM is faster, and the gain grows with the cell.

**Less noise from the atomic orbitals.** Near the Fermi level the states live
mostly on the atomic orbitals and on the part of them that leaks between the atoms.
With `control_variate=True`, the code filters the atomic orbitals with a Chebyshev
polynomial of $H$ that keeps their part below $E_F + 1$ Ry (`filter_cut`, of degree
`filter_degree = 8`), and solves $H$ in that small space. This gives approximate
eigenstates $|y_j\rangle$ with energies $\theta_j$. For each random vector it
subtracts what these states predict, and adds back its exact average:

$$
\mu_n \;\to\; \mu_n - N_{\rm pw}\sum_j |\langle y_j|r\rangle|^2\,T_n(\theta_j') + \sum_j T_n(\theta_j').
$$

The average of the subtracted term is exactly the term added back, so the DOS stays
unbiased. The noise falls where the $|y_j\rangle$ are close to eigenstates, which is
the valence band and the lowest conduction states. On the same cells, with one
random vector:

| cell | from the bands | KPM, 8 random vectors | KPM with the orbitals, 1 random vector |
|---|---|---|---|
| 16 atoms | 1.6 to 3.9 s | 2.5 s, error 26% | 0.7 s, error 0.4% |
| 64 atoms | 32 to 85 s | 14 s, error 13% | 10.5 s, error 0.2% |
| 128 atoms | 275 to 1449 s | 44 to 95 s, error about 10% | 123 s, error 0.2% |

The times of the 128-atom cell were taken while the machine was heavily loaded, so
they are a range. The error is the same rms difference from the bands as above,
averaged over four seeds. With one random vector the code cannot estimate its own
error, so `dos_error` is zero there. Ask for at least two vectors to get the error
bar.

So one random vector with the atomic orbitals is about 60 to 100 times more accurate
than eight without them, at about the same cost, and faster than the bands already at
16 atoms. The extra cost is nine applications of $H$ per atomic orbital and a small dense
diagonalization in their space, which grows like the bands do, so the gain is
largest when you want an error of about one per cent.

??? example "Code of this figure (docs/figures/dos_kpm/)"

    ```python title="compute.py"
    --8<-- "dos_kpm/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "dos_kpm/plot.py"
    ```

## Projected, local and unfolded density of states by KPM

We now keep the same expansion, but we choose the start vectors instead of taking
random ones. Then the trace has no statistical error, and only the broadening
remains. Three choices answer three physical questions.

**The projected DOS.** With an atomic orbital $|\phi_a\rangle$ as the start vector,

$$
D_a(E) = \sum_{\mathbf k} w_{\mathbf k}\,\langle\phi_a|\,\delta(E - H_{\mathbf k})\,|\phi_a\rangle
= \sum_{n\mathbf k} w_{\mathbf k}\,|\langle\phi_a|\psi_{n\mathbf k}\rangle|^2\,\delta(E - \varepsilon_{n\mathbf k}),
$$

which is the projected DOS of `get_pdos`, with the same Löwdin orbitals. Here it
has every state of the basis in it, the empty ones up to the cutoff too, and no
band is computed. One orbital is not invariant under the symmetry of the crystal,
so the code runs this on the whole k-mesh and not on the irreducible wedge.

**The local DOS.** With the plane-wave form of $\delta(\mathbf r - \mathbf r_0)$ as
the start vector, the trace is
$\sum_{n\mathbf k} w_{\mathbf k}\,|\psi_{n\mathbf k}(\mathbf r_0)|^2\,\delta(E - \varepsilon_{n\mathbf k})$,
the local DOS at $\mathbf r_0$. This is what STM measures at a tip position. Its
integral up to the Fermi level is the charge density at $\mathbf r_0$.

```python
from defuqumat import Calculator

calc = Calculator.from_keywords(
    lattice="fcc", a=5.398, atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
    pseudos={"Si": "Si.pz-vbc.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=163.3, kmesh=(4, 4, 4))
gap = calc.get_scf().homo + 0.02
pdos = calc.get_kpm_pdos(broadening=0.005)
charges = pdos.charges_at(gap)[0]
print(f"first atom: s {charges[0]:.4f}, p {charges[1:4].sum():.4f} electrons")
ldos = calc.get_kpm_ldos([(0.125, 0.125, 0.125)], broadening=0.005)
print(f"density at the bond centre: {ldos.density_at(gap)[0, 0]:.5f} electrons/bohr^3")
```

```
first atom: s 1.1001, p 2.8672 electrons
density at the bond centre: 0.08891 electrons/bohr^3
```

`pdos.charges_at(E)` gives the charge of each orbital below $E$, which at the
Fermi level is its Löwdin charge, and `pdos.select(atom=..., l=...)` sums the
channels as in `get_pdos`. `ldos.density_at(E)` gives the charge density below
$E$ at each point. The points are in crystal coordinates. We compared both with
the bands on a $2\times2\times2$ mesh, where the gap is only 0.033 Ry. At a
broadening of 0.005 Ry the charges agree with the Löwdin charges of `get_pdos`, and
the density with the SCF density, within $4\times10^{-4}$. The difference is the tail
of the Jackson peak across the gap, and at 0.0025 Ry it falls to $4\times10^{-5}$.

**The unfolded spectral function.** In a supercell, the plane waves
$|\mathbf k + \mathbf G_p\rangle$, with $\mathbf G_p$ a reciprocal vector of the
primitive cell, carry the crystal momentum $\mathbf k$ of the primitive cell. The
trace over them,

$$
A(\mathbf k, E) = \sum_{\mathbf G_p}\langle\mathbf k + \mathbf G_p|\,\delta(E - H_{\mathbf k})\,|\mathbf k + \mathbf G_p\rangle,
$$

is the spectral function of the supercell unfolded to the primitive cell, which
is what ARPES measures on a crystal with defects. We compute it with
`calc.get_kpm_unfolded(primitive_lattice, kpoints)`, where `primitive_lattice` is
the $3\times3$ matrix of primitive lattice vectors in bohr (rows) and `kpoints` are
in crystal coordinates of the primitive cell. `trace="exact"` takes every such
plane wave as a start vector, and `trace="stochastic"` takes `n_random` random
vectors on them, which is cheaper for a long path.

![The spectral function of a perfect Si8 cell and of one with a vacancy, unfolded to the fcc path.](img/dos_kpm_unfold.png#only-light)
![The spectral function of a perfect Si8 cell and of one with a vacancy, unfolded to the fcc path.](img/dos_kpm_unfold-dark.png#only-dark)

On the left is a perfect cubic cell of 8 silicon atoms, unfolded along L-Γ-X of
the fcc cell with 8 random vectors, and the dashed lines are the bands of the
two-atom cell. The weight sits on them, because in a perfect crystal these plane
waves do not mix with the others. Below the Fermi level it integrates to 4.000,
3.995 and 4.006 at L, Γ and X (exact trace), the four valence bands per spin. On
the right one atom is removed. The lowest band survives, the upper valence bands
lose much of their weight (the integral at Γ falls to 1.79), and flat features
appear near $-2$ and $+1$ eV, which are the localized states of the vacancy. Two
different cells have no common zero of energy, so each panel is shifted to put
the bottom of its valence band at Γ on the dashed line, by $-0.86$ eV for the
vacancy.

**The spin-resolved DOS and the Fermi level.** For a noncollinear run,
`get_kpm_dos(spin_resolved=True)` also gives
$\sum_{\mathbf k} w_{\mathbf k}\,{\rm Tr}[\sigma_i\,\delta(E - H_{\mathbf k})]$ as
`kpm.spin_dos`, the DOS of the spin along x, y and z, from the same random vectors.
This is what spin-polarized STM and spin-resolved photoemission measure.
`kpm.fermi_level()` gives the energy where the count reaches the number of
electrons, with no band, so it is the Fermi level of a cell too large to
diagonalize.

??? example "Code of this figure (docs/figures/dos_kpm_unfold/)"

    ```python title="compute.py"
    --8<-- "dos_kpm_unfold/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "dos_kpm_unfold/plot.py"
    ```

!!! note "Limits"
    The projected density of states works with `K_POINTS gamma`. It agrees with
    an explicit $k = 0$ to 4e-12.

    The tetrahedron schemes need an automatic k-mesh. If you do not give `grid`
    and the SCF used an explicit list of k-points, tetrahedra give an error,
    because a list of points carries no tetrahedra.

    A noncollinear run is symmetrized, with the spin rotated together with the
    orbitals. A spin-orbit run (`lspinorb`) is symmetrized too when it is not
    magnetic. A magnetic spin-orbit run with symmetry gives an error. For it you
    run with `nosym = .true.` and the full k-mesh, which is the same physics, or
    pass `symmetrize=False`.

    A noncollinear run, with or without spin-orbit coupling, has no
    $m$-resolved charges, and `charges.charges_lm` is `None`. The charges per
    atom and per $l$, `charges.charges`, are still there.

    A fully relativistic dataset with `lspinorb = .false.` gives an error in the
    projection, because its channels would be resolved by $j$ and still be
    labelled as up and down. So you run it with `lspinorb = .true.`, which is what a
    fully relativistic dataset is for.

    The total KPM density of states works with ultrasoft and PAW datasets, for
    `nspin = 1` and `2`, through $S^{-1/2} H S^{-1/2}$. With a spinor they give an
    error, because the overlap couples the two spin components and we have not
    checked that case. The projected and local DOS, the unfolding and the
    conductivity by KPM need norm-conserving pseudopotentials, because their start
    vectors or their operator need the overlap $S$, and the code gives an error. The
    KPM traces also give an error for `K_POINTS gamma` (the half sphere of plane
    waves) and for a spin spiral. A wedge of k-points is fine, because the DOS is a scalar trace. If the
    interval does not hold the whole spectrum, a moment grows above one and the
    code gives an error that says so, instead of a wrong DOS. The same holds for
    the projected, local and unfolded DOS by KPM. The local DOS is the smooth
    pseudo-wavefunction at the point, with no augmentation, which is exact for a
    norm-conserving pseudopotential outside the core radius. The unfolding needs
    the cell to be an integer supercell of the primitive lattice you give, and it
    gives an error otherwise. `spin_resolved=True` needs a noncollinear run.
    `control_variate=True` needs random vectors (it gives an error with
    `trace="exact"`), and it needs atomic orbitals in the pseudopotential files.
    It does not change `spin_dos`, which keeps the noise of the plain trace.

The notebooks for this page are
[06, the density of states](https://github.com/joselado/defuqumat/blob/master/notebooks/06_density_of_states.ipynb),
[16, the projected density of states](https://github.com/joselado/defuqumat/blob/master/notebooks/16_projected_density_of_states.ipynb),
[60, the density of states without bands](https://github.com/joselado/defuqumat/blob/master/notebooks/60_kpm_density_of_states.ipynb)
and
[61, unfolding a supercell with a vacancy](https://github.com/joselado/defuqumat/blob/master/notebooks/61_kpm_unfolding.ipynb).
