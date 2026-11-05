# Wannier functions

In this page we compute maximally localized Wannier functions. This is the work that
`pw2wannier90.x` and `wannier90.x` do after a `pw.x` run, here in one call. We take a
set of Bloch bands, rotate them at each k-point, and Fourier transform them to
functions centred in the unit cell. The rotation is chosen so that the functions are
as localized as possible. What we get is a tight-binding model of the bands: the
centres and spreads of the functions, the hoppings $H_{mn}(\mathbf R)$, and the bands
interpolated at any k-point.

## What is minimised

What does "maximally localized" mean? For $N$ functions $w_n$ the spread is the sum
of their variances:

$$
\Omega = \sum_n \left[ \langle w_n | r^2 | w_n \rangle - |\langle w_n | \mathbf r | w_n \rangle|^2 \right]
= \Omega_I + \Omega_D + \Omega_{OD} .
$$

The first part, $\Omega_I$, does not depend on the rotation. It depends only on the
subspace of the bands, and it is the same invariant as the one we compute in
[the quantum metric](topology.md#the-quantum-metric). The other two parts, the
diagonal and the off-diagonal one, depend on the rotation, and the code minimises
their sum (Marzari and Vanderbilt, PRB 56, 12847 (1997)). On a k-mesh everything is
written with the overlaps between neighbouring k-points:

$$
M_{mn}(\mathbf k, \mathbf b) = \langle u_{m\mathbf k} | u_{n,\mathbf k+\mathbf b} \rangle ,
$$

and the starting rotation comes from the projections on trial orbitals $g_n$, such
as an $sp^3$ hybrid on an atom:

$$
A_{mn}(\mathbf k) = \langle \psi_{m\mathbf k} | S | g_n \rangle .
$$

These are the `.mmn` and `.amn` files of `pw2wannier90.x`. When there are more bands
than functions, the code first chooses the best subspace inside an energy window, the
disentanglement of Souza, Marzari and Vanderbilt (PRB 65, 035109 (2001)). The bands
inside the frozen window are kept exactly.

## Diamond: four bond orbitals

We start with the simplest case, the four valence bands of diamond. They are isolated
from the conduction bands, so no window is needed. We put one $s$ orbital at the
centre of each of the four bonds of an atom, which is the `example05` of Wannier90.
The trial orbitals are written as in a `.win` file. We compute them with
`calc.get_wannier()`:

```python
import numpy as np
from defuqumat import Calculator

diamond = Calculator.from_keywords(
    lattice="fcc", a=3.228,
    atoms=[("C", (-0.125, -0.125, -0.125)), ("C", (0.125, 0.125, 0.125))],
    pseudos={"C": "C.pbe-hgh.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=340.1, kmesh=(4, 4, 4),
)
diamond.get_scf(conv_thr=1e-10)
bonds = ["f=0,0,0:s", "f=0,0,0.5:s", "f=0,0.5,0:s", "f=0.5,0,0:s"]
w = diamond.get_wannier(4, bonds, engine_options=dict(num_iter=100, conv_window=3))
print(np.round(w.centres_angstrom, 4))
print(np.round(w.spreads_angstrom2, 4))
print({k: round(v, 5) for k, v in w.spread_terms_angstrom2().items()})
```

```
[[-0.     0.     0.   ]
 [-0.807  0.807 -0.   ]
 [-0.     0.807  0.807]
 [-0.807  0.     0.807]]
[0.5873 0.5873 0.5873 0.5873]
{'omega_i': 1.97388, 'omega_d': 0.0, 'omega_od': 0.37528, 'omega_total': 2.34917}
```

The four centres are the four bond centres, and the four spreads are equal, as the
symmetry of the crystal says. $\Omega_D$ is zero because each centre sits on a point
of high symmetry. The Wannier mesh is the k-mesh of the SCF here. You can choose
another one with `divisions=(n1, n2, n3)`. The mesh must be uniform and centred at
$\Gamma$, because the overlaps connect each k-point with its neighbours.

The same cell through `pw.x`, `pw2wannier90.x` and `wannier90.x` gives
$\Omega = 2.349146$ Å$^2$ when the cell is given as $a = 6.1$ bohr. The small
difference to the 2.34917 above comes from the cell: 3.228 Å is $6 \times 10^{-6}$ Å
away from 6.1 bohr. On the same cell the two pipelines agree to $2 \times 10^{-9}$
Å$^2$ (see [How we checked it](#how-we-checked-it) below).

## Silicon: sp3 orbitals from twelve bands

We now take eight $sp^3$ hybrids in silicon, four on each atom. Eight functions
describe the valence bands and the lowest four conduction bands. These bands are
entangled with the bands above them, so we take twelve bands and an outer window up
to 17 eV. The frozen window keeps every band below 6.4 eV, which is the valence band,
exactly. Energies are in eV, on the scale of the eigenvalues:

```python
si = Calculator.from_keywords(
    lattice="fcc", a=5.3976, atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
    pseudos={"Si": "Si.pz-vbc.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=272.1, kmesh=(8, 8, 8),
)
si.get_scf(conv_thr=1e-10)
w = si.get_wannier(8, ["Si:sp3"], num_bands=12, divisions=(4, 4, 4),
                   dis_win=(None, 17.0), dis_froz=(None, 6.4),
                   engine_options=dict(num_iter=200, conv_window=3))
print(np.round(w.spreads_angstrom2, 4))
print({k: round(v, 5) for k, v in w.spread_terms_angstrom2().items()})
print(np.round(w.bands_ev([[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]), 3))   # L and Gamma
```

```
[1.8177 1.8177 1.8177 1.8177 2.1956 2.1956 2.1956 2.1956]
{'omega_i': 11.81348, 'omega_d': 0.13752, 'omega_od': 4.10218, 'omega_total': 16.05319}
[[-3.422 -0.825  5.027  5.027  7.812  9.588  9.588 13.835]
 [-5.816  6.247  6.247  6.247  8.814  8.814  8.814  9.722]]
```

The two atoms give two groups of spreads, 1.8177 and 2.1956 Å$^2$, although they are
equivalent by symmetry. The reason is the trial orbitals. `Si:sp3` puts the lobes along
$(1,1,1)$, $(1,-1,-1)$, $(-1,1,-1)$ and $(-1,-1,1)$ on both atoms. On the second atom
these are the directions of its four bonds, but on the atom at the origin the bonds
point the opposite way. So the two sets start from different orbitals and end in two
different local minima of $\Omega$, and `wannier90.x` finds the same two. If we turn
the axes of the second atom by 90 degrees about $z$, both atoms start in the same way:

```python
w = si.get_wannier(8, ["f=0,0,0:sp3", "f=0.25,0.25,0.25:sp3:z=0,0,1:x=0,1,0"],
                   num_bands=12, divisions=(4, 4, 4),
                   dis_win=(None, 17.0), dis_froz=(None, 6.4),
                   engine_options=dict(num_iter=200, conv_window=3))
print(np.round(w.spreads_angstrom2, 4))
print({k: round(v, 5) for k, v in w.spread_terms_angstrom2().items()})
```

```
[1.8046 1.8046 1.8046 1.8046 1.8046 1.8046 1.8046 1.8046]
{'omega_i': 11.81348, 'omega_d': 0.10428, 'omega_od': 2.51871, 'omega_total': 14.43648}
```

Now the eight spreads are equal and $\Omega$ is lower by 1.6 Å$^2$. $\Omega_I$ does not
change, because the subspace is the same. Each centre sits 0.80 Å from its atom, on the
side away from its bonds. So the minimisation is local, and the trial orbitals decide
which minimum we reach, in this code as in `wannier90.x`.

The model is `w.hoppings`, the matrix $H_{mn}(\mathbf R)$ in Ry on the lattice
vectors `w.rvectors`. `w.bands(k)` diagonalises

$$
H_{mn}(\mathbf k) = \sum_{\mathbf R} \frac{e^{i\mathbf k\cdot\mathbf R}}{N_{\mathbf R}} H_{mn}(\mathbf R)
$$

at any k-point. By default each pair $(m, n)$ is moved to its closest image in the
supercell, which is the `use_ws_distance` of Wannier90 and its default.

![Wannier-interpolated bands of silicon against the bands computed directly.](img/wannier_silicon.png#only-light)
![Wannier-interpolated bands of silicon against the bands computed directly.](img/wannier_silicon-dark.png#only-dark)

The figure shows the twelve bands computed directly (grey) and the eight interpolated
bands from a 4x4x4 and a 6x6x6 Wannier mesh, along L-Γ-X-W-K-Γ. The horizontal line
is the top of the frozen window. Below it the interpolation follows the valence bands,
within 0.317 eV on 4x4x4 and 0.064 eV on 6x6x6. Above it the model gives
four bands that follow the lowest conduction bands where they are not entangled with
the others. The spread grows with the mesh, from 16.05 to 19.49 Å$^2$. This is
because the finite-difference spread approaches its limit from below, as $\Omega_I$
does in [the quantum metric](topology.md#the-quantum-metric).

??? example "Code of this figure (docs/figures/wannier_silicon/)"

    ```python
    --8<-- "wannier_silicon/compute.py"
    ```
    ```python
    --8<-- "wannier_silicon/plot.py"
    ```

## Without choosing the orbitals

So far we wrote the trial orbitals by hand. Can the code choose them? Every
pseudopotential file carries the orbitals of the free atom, in its `PP_PSWFC` block.
We now use these orbitals $g_n$ on every atom as the trial orbitals, and we sort the
Bloch states by how much of them these orbitals describe. This is the
projectability:

$$
p_{m\mathbf k} = \sum_n |\langle \tilde g_n | S | \psi_{m\mathbf k}\rangle|^2 ,
\qquad \tilde g = g\, O^{-1/2}, \quad O_{nn'} = \langle g_n | S | g_{n'} \rangle ,
$$

meaning that the orbitals are first made orthonormal (Löwdin) and then $p$ is the
weight of the state on them, between 0 and 1. A state with $p \geq 0.95$ is kept
exactly, a state with $p < 0.01$ is left out, and the states in between go into the
disentanglement. Every state up to 2 eV above an edge is also kept exactly. The edge
is the Fermi level of a metal and the conduction band minimum of an insulator. This is
the projectability disentanglement of Qiao, Pizzi and Marzari (arXiv:2303.07877), with
their recommended energy part. The decisions are taken on the mean of $p$ over each
degenerate multiplet, so a multiplet is kept, left out or disentangled as a whole.

We compute it with `calc.get_wannier()` and no orbitals. Here we take silicon on a
4x4x4 mesh:

```python
from defuqumat.units import RY_TO_EV

si = Calculator.from_keywords(
    lattice="fcc", a=5.3976, atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
    pseudos={"Si": "Si.pz-vbc.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=163.27, kmesh=(4, 4, 4),
)
si.get_scf(conv_thr=1e-10)
w = si.get_wannier(engine_options=dict(num_iter=200, conv_window=3))
print(w.num_wann, w.num_bands, w.projection_labels[:4])
print(round(w.edge * RY_TO_EV, 4))
print(w.frozen.sum(axis=1).min(), w.frozen.sum(axis=1).max(), (~w.outer).sum())
print(np.round(w.spreads_angstrom2, 4))
print({k: round(v, 5) for k, v in w.spread_terms_angstrom2().items()})
```

```
8 16 ('Si1 3S:s', 'Si1 3P:pz', 'Si1 3P:px', 'Si1 3P:py')
6.9653
4 8 110
[1.5016 2.1295 2.1295 2.1295 1.5016 2.1295 2.1295 2.1295]
{'omega_i': 10.05687, 'omega_d': 0.07511, 'omega_od': 5.64819, 'omega_total': 15.78017}
```

The file gives $3s$ and $3p$ on each atom, so there are eight functions. If you do not
give `num_bands`, the code solves twice as many bands as orbitals, here 16. The edge is
the conduction band minimum, 6.965 eV. Between 4 and 8 states are kept exactly at each
k-point, and 110 states of the 1024 are left out. The functions keep the shape of the
atomic orbitals: one $s$ and three equal $p$ on each atom. `w.projectability`, `w.outer`
and `w.frozen` are arrays of shape `(nk, num_bands)`, so you can see which state went
where. The projectability is averaged over each degenerate multiplet, so the states of one
multiplet always go to the same set. You can change the thresholds with `proj_min` and `proj_max`, the energy part
with `froz_above_edge` (in eV, `None` to switch it off), and `num_bands`.

A semicore shell is a shell with a partner of the same $l$ and a higher principal number
in the same file, for example $3s$ beside $4s$. Its bands are deep and flat, so they are
left out if they lie below the valence bands with a gap. Here is chromium, a metal,
with the $3s$, $4s$, $3p$ and $3d$ orbitals of its file:

```python
cr = Calculator.from_keywords(
    lattice="bcc", a=2.8787, atoms=[("Cr", (0, 0, 0))],
    pseudos={"Cr": "Cr.pbe-spn-rrkjus_psl.1.0.0.UPF"}, pseudo_dir="tests/data/pseudo",
    ecut=476.2, ecutrho=3809.6, kmesh=(6, 6, 6),
    smearing=0.272, smearing_type="marzari-vanderbilt",
)
cr.get_scf(conv_thr=1e-10)
w = cr.get_wannier(divisions=(4, 4, 4), engine_options=dict(num_iter=200, conv_window=3))
print(w.num_wann, w.num_bands, w.excluded_bands, w.semicore)
print(w.projection_labels)
print(round(w.edge * RY_TO_EV, 4), w.frozen.sum(axis=1).min(), w.frozen.sum(axis=1).max())
```

```
9 19 (1,) cut (gap 28.32)
('Cr1 4S:s', 'Cr1 3P:pz', 'Cr1 3P:px', 'Cr1 3P:py', 'Cr1 3D:dz2', 'Cr1 3D:dxz', 'Cr1 3D:dyz', 'Cr1 3D:dx2-y2', 'Cr1 3D:dxy')
18.8628 8 9
```

The $3s$ band lies 28.32 eV below the next band, so it is left out. This is the
`exclude_bands = 1` of Wannier90, and `w.excluded_bands` keeps it. The $3p$ shell has no
$4p$ partner in this file, so it stays. The edge is now the Fermi level. With
`semicore=False` the code keeps every shell. With `semicore=True` it cuts the semicore
shells even when their bands touch the valence bands. For given projections,
`exclude_bands` is a keyword of its own, with the 1-based band numbers of Wannier90.

How good is the model? We use the band distance of Qiao and coworkers. It is the root
mean square difference between the interpolated and the direct bands on 50 random
k-points, weighted by a Fermi function at $E_F + 2$ eV with a width of 0.1 eV, called
$\eta_2$. For an insulator we put $E_F$ at the top of the valence band. Here is
silicon (the cell above) and aluminium ($s$ and $p$, four functions), in meV:

| case | mesh | $\eta_0$ | $\eta_2$ | $\Omega$ (Å$^2$) |
|---|---|---|---|---|
| Si, automatic | 4x4x4 | 94.5 | 111.7 | 15.780 |
| Si, automatic | 6x6x6 | 17.5 | 50.4 | 17.187 |
| Si, automatic | 8x8x8 | 8.8 | 17.5 | 18.759 |
| Si, `Si:sp3`, 12 bands, frozen to the same edge | 4x4x4 | 74.2 | 97.5 | 18.095 |
| Si, `Si:sp3`, 12 bands, frozen to the same edge | 6x6x6 | 20.3 | 30.3 | 21.999 |
| Si, `Si:sp3`, 12 bands, frozen to the same edge | 8x8x8 | 7.7 | 10.4 | 23.400 |
| Al, automatic | 4x4x4 | 419.2 | 463.1 | 7.209 |
| Al, automatic | 6x6x6 | 132.3 | 149.8 | 8.823 |
| Al, automatic | 8x8x8 | 46.6 | 75.4 | 9.326 |

The density is the one of the 4x4x4 SCF in every row, and only the Wannier mesh
changes. For the occupied bands ($\eta_0$) the automatic set and the hand-made hybrids
are close, and the automatic set needs no input. Up to 2 eV above the gap ($\eta_2$)
the hybrids do better here. Qiao and coworkers find that the lowest conduction band of
silicon near X has a $d$ component, which the $s$ and $p$ orbitals of this file cannot
describe. The error of aluminium falls fast with the mesh, from 463 meV on 4x4x4 to
75 meV on 8x8x8.

## Running wannier90.x on our matrices

The result keeps the matrices in the layout of Wannier90. `w.write_w90(prefix)` writes
`prefix.mmn`, `.amn`, `.eig` and a `.win` with the cell, the mesh and the keywords you
pass. So you can run `wannier90.x` itself, or its post-processing (`postw90.x`), on the
states of this code:

```python
crystal = np.asarray(si.system.structure.positions_crystal(si.system.cell))
w.write_w90("si", projections=["Si:sp3"],
            atoms=list(zip(si.system.structure.symbols, crystal)),
            extra=dict(dis_win_max=17.0, dis_froz_max=6.4, num_iter=200, conv_window=3))
```

## Wannier functions with the symmetry of the crystal

We now ask the Wannier functions to keep the symmetry of the crystal. The
minimisation is local, so a minimum can break the symmetry a little, and then the
model breaks it too. With `symmetry=True` the code imposes, for every operation $g$
of the space group of the run:

$$
U(g\mathbf k) = d(g, \mathbf k)\, U(\mathbf k)\, D(g, \mathbf k)^\dagger .
$$

Here $d$ is the matrix that takes the Bloch states at $\mathbf k$ into those at
$g\mathbf k$, and $D$ takes the trial orbitals into each other. This is the
`site_symmetry` of Wannier90 (Sakuma, PRB 87, 235109 (2013)), and the two matrices
are the `.dmn` file that `pw2wannier90.x` writes with `write_dmn`. Only the
irreducible k-points are free, and the rest of the mesh follows from them. A frozen
window is not allowed in this mode, in Wannier90 as here, so we keep only the outer
window:

```python
for symmetry in (False, True):
    w = si.get_wannier(8, ["Si:sp3"], num_bands=12, divisions=(4, 4, 4),
                       dis_win=(None, 17.0), symmetry=symmetry,
                       engine_options=dict(num_iter=200, conv_window=3))
    print(w.nsym, np.round(w.spreads_angstrom2, 4), round(w.spread_terms_angstrom2()["omega_total"], 5))
```

```
1 [1.7158 1.7158 1.7158 1.7158 2.2261 2.2261 2.2261 2.2261] 15.76758
48 [1.7243 1.7243 1.7243 1.7243 2.2191 2.2191 2.2191 2.2191] 15.77369
```

`w.nsym` is the number of operations imposed, 48 for the diamond group, and half of
them carry a fractional translation. The total is lower than the 16.05 Å$^2$ of the
section above only because the frozen window is gone. With symmetry $\Omega$ goes up
by $6 \times 10^{-3}$ Å$^2$, which means that the free minimum was not exactly
symmetric. The two groups of spreads stay, because `Si:sp3` itself is not
symmetric: an operation that exchanges the two atoms turns a lobe along $(1,1,1)$
into one along $(-1,-1,-1)$, which is not in the set of the other atom. With the
turned axes of the section above the eight functions are equivalent, and we can
check that the bands at a k-point and at its 48 images are the same:

```python
k = np.array([[0.1, 0.2, 0.3]])
for symmetry in (False, True):
    w = si.get_wannier(8, ["f=0,0,0:sp3", "f=0.25,0.25,0.25:sp3:z=0,0,1:x=0,1,0"],
                       num_bands=12, divisions=(4, 4, 4), dis_win=(None, 17.0),
                       symmetry=symmetry, engine_options=dict(num_iter=200, conv_window=3))
    print(w.nsym, np.round(w.spreads_angstrom2, 4), round(w.spread_terms_angstrom2()["omega_total"], 5))
    # the same k turned by every operation of the group: the bands should not change
    group = si.system.symmetry_group(nosym=False)
    images = np.array([k[0] @ np.asarray(r).T for r in group.rotations])  # crystal, rows
    e = np.sort(w.bands_ev(np.vstack([k, images])), axis=1)
    print(f"  max change of the bands over the {len(images)} images of k: {np.abs(e - e[0]).max():.1e} eV")
```

```
1 [1.7234 1.7234 1.7234 1.7234 1.7234 1.7234 1.7234 1.7234] 13.78719
  max change of the bands over the 48 images of k: 1.9e-08 eV
48 [1.7234 1.7234 1.7234 1.7234 1.7234 1.7234 1.7234 1.7234] 13.78719
  max change of the bands over the 48 images of k: 6.6e-13 eV
```

Here the free minimum is already symmetric to $2 \times 10^{-8}$ eV, and the
constraint makes it exact. For `Si:sp3` the same test gives $3 \times 10^{-7}$ eV
without symmetry and $5 \times 10^{-13}$ eV with it. With a `nosym` run the group is the identity
alone, and `symmetry=True` then changes nothing. `w.write_w90(prefix)` also writes
`prefix.dmn` and `site_symmetry = .true.`, so `wannier90.x` can run the same
symmetric minimisation on our files. Notebook
[59](https://github.com/joselado/defuqumat/blob/master/notebooks/59_wannier_symmetry.ipynb)
shows the spreads of the four runs.

## How we checked it

We checked the two halves apart, against `pw2wannier90.x` and `wannier90.x` of QE 7.5
and Wannier90 3.1.0, on the same inputs. The engine is the one of `wannier90.x`, a port
of its Fortran. On the `.mmn`, `.amn` and `.eig` files of `pw2wannier90.x` it gives the
spread of `wannier90.x` to $2 \times 10^{-8}$ Å$^2$ for diamond and $1.3 \times 10^{-7}$
Å$^2$ for silicon with the windows. Then `wannier90.x` itself, run on our files, gives
the spread it gives on the files of `pw2wannier90.x`:

| cell | `wannier90.x` on `pw2wannier90.x` (Å$^2$) | `wannier90.x` on ours (Å$^2$) |
|---|---|---|
| diamond, 4 bands, ecut 40 Ry | 2.317428277 | 2.317428275 |
| silicon, 12 to 8, norm-conserving | 16.053195110 | 16.053194713 |
| silicon, 12 to 8, ultrasoft | 18.358070827 | 18.358070728 |
| silicon, 12 to 8, PAW | 18.362772917 | 18.362772978 |

The differences are those of two SCF runs, at most $4 \times 10^{-7}$ Å$^2$. The bands
interpolated by this code from the matrices of `wannier90.x` agree with its
`si_band.dat` to $9 \times 10^{-6}$ eV.

For `projections="auto"` the reference is `pw2wannier90.x` with `atom_proj = .true.`
and `atom_proj_ortho = .true.`, which writes the projections on the orthonormal atomic
orbitals. That code writes the orbitals with the spherical harmonics of QE and these
use the ones of Wannier90, so we compare $p_{m\mathbf k}$, averaged over each
degenerate multiplet, which does not depend on that choice. On the 4x4x4 mesh:

| cell | bands | largest difference of $p$ | mean difference |
|---|---|---|---|
| Si, norm-conserving | 16 | $1.1 \times 10^{-6}$ | $1.7 \times 10^{-7}$ |
| Si, ultrasoft | 16 | $7.2 \times 10^{-8}$ | $1.0 \times 10^{-8}$ |
| Al, norm-conserving | 8 | $1.5 \times 10^{-7}$ | $2.4 \times 10^{-8}$ |
| Cr, ultrasoft, $3s$ excluded | 19 | $1.4 \times 10^{-7}$ | $1.0 \times 10^{-8}$ |

For chromium `pw2wannier90.x` ran with `exclude_bands = 1` and `atom_proj_exclude(1) =
1`, the same cut as ours. When the kept and left-out states are energy windows, we can
also give our files to `wannier90.x` with the same windows. On silicon with 14 bands it
gives $\Omega = 16.253091461$ Å$^2$ against our 16.253091603 Å$^2$ (ultrasoft:
16.380204649 against 16.380204793). The difference is the same, $1.4 \times 10^{-7}$
Å$^2$, at every convergence threshold. It is the conversion from bohr to Å: with the
older value of the Bohr radius, 0.52917720859 Å instead of 0.529177210903 Å, our
spread is that of `wannier90.x` to $10^{-9}$ Å$^2$.

For `symmetry=True` we checked the same two halves. On the two site-symmetry
examples of the Wannier90 test suite (GaAs, isolated, and H$_3$S, twenty bands to
twelve) the engine gives the spreads of `wannier90.x` to every printed digit. On
silicon our matrix $D$ is the one of `pw2wannier90.x` to $10^{-15}$, and the
characters of $d$ agree to $3 \times 10^{-11}$. The whole chain of `pw.x`,
`pw2wannier90.x` and `wannier90.x` with `site_symmetry` gives:

| silicon, 12 to 8, outer window only | QE chain (Å$^2$) | this code (Å$^2$) |
|---|---|---|
| `Si:sp3` | 15.773697207 | 15.773696950 |
| turned axes | 13.787202895 | 13.787202775 |

!!! note "Limits"
    - Spinor calculations (`noncolin`, spin-orbit coupling, a spin spiral) give an
      error. The projections on spinor trial orbitals were not checked against
      `pw2wannier90.x`.
    - At `nspin = 2` one channel is computed at a time: pass `spin=0` or `spin=1`,
      as the `spin_component` of `pw2wannier90.x`.
    - Selective localization from the input, SCDM (`scdm_proj`), random projections
      and `K_POINTS gamma` are not implemented. A `K_POINTS gamma` run gives an error.
    - `projections="auto"` needs the `PP_PSWFC` orbitals on every species. A file
      without them, such as the SG15 ones, gives an error. It cannot be combined with
      `dis_win`, `dis_froz` or `exclude_bands`, and `num_wann` must be the number of
      orbitals that are kept.
    - The automatic set uses only the orbitals of the file. If a band needs an orbital
      the file does not have (the $d$ component near X in silicon), its error stays
      large. Orbitals of higher $l$ cannot be added yet.
    - A semicore shell is found from the labels of the orbitals (`3S` beside `4S`).
      It is cut only if its bands are the lowest ones at every k-point, with most of
      their weight on it. Otherwise it is kept, with a warning.
    - The states kept exactly and the states left out are chosen for each k-point. So
      `wannier90.x` cannot redo an automatic run from the `.win` file, which has only
      energy windows. It can redo it when the sets are energy windows, for example
      with `proj_max=2.0, proj_min=0.0`.
    - With `symmetry=True` a frozen window gives an error, as in Wannier90, and so do
      `precond`, projections that are not a list of projection lines, and a
      magnetic group with operations that need time reversal. A set of trial
      orbitals that the group does not map onto itself gives an error by name. So
      does a band set or an outer window that cuts a degenerate multiplet at some
      k-point, because then no rotation satisfies the constraint.
    - With `symmetry=True` time reversal is not imposed. On the GaAs example of
      Wannier90 the imaginary part of $H_{mn}(\mathbf R)$ is $3 \times 10^{-10}$ eV
      anyway.
    - `symmetry=True` cannot be combined with `projections="auto"` yet, because
      the automatic set freezes states and a frozen window gives an error with
      symmetry.
    - `zona` is in 1/bohr, as `pw2wannier90.x` reads it. The user guide of Wannier90
      writes it in 1/Å. The default, 1, is the same number in both codes.
    - The Wannier functions themselves on a real-space grid (`wannier_plot`) and the
      position matrix $\mathbf r_{mn}(\mathbf R)$ are not written.
    - The minimisation runs in NumPy on the host. On silicon with twelve bands it
      takes 2 s, about four times `wannier90.x`.
