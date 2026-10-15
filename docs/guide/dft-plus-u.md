# DFT+U

In this page we add a Hubbard $U$ to a localized shell, here the $3d$ shell of iron
in FeO, and we see what it does to the occupations and to the density of states.

## The Hubbard correction

We start with what goes wrong without $U$. LDA and GGA spread the electrons of a
localized $d$ or $f$ shell too evenly over its orbitals, so they often make a metal
of an oxide that is an insulator. DFT+U adds an energy that is zero when each
orbital of the shell is full or empty, and positive in between. In the simplest
form (Dudarev) it is:

$$
E_U = \sum_{I,\sigma} \frac{U^I}{2}\,
   \mathrm{Tr}\Big[ n^{I\sigma} \big( 1 - n^{I\sigma} \big) \Big],
\qquad
n^{I\sigma}_{mm'} = \sum_{n\mathbf k} f_{n\mathbf k\sigma}
   \langle \psi_{n\mathbf k\sigma} | \hat S | \varphi^I_{m'}\rangle
   \langle \varphi^I_{m} | \hat S | \psi_{n\mathbf k\sigma}\rangle .
$$

Here $n^{I\sigma}$ is the occupation matrix of the shell on atom $I$, the
projection of the occupied states on its atomic orbitals $\varphi^I_m$. The
potential of $E_U$ pushes the occupied orbitals down and the empty ones up, by
about $U/2$ each. This is the physics of a Mott insulator that the functional
misses, and photoemission and inverse photoemission see the gap between the two.

You give $U$ in the `HUBBARD` card of `pw.x`, in eV:

```
HUBBARD {atomic}
U Fe1-3d 4.3
U Fe2-3d 4.3
```

The projectors are `atomic`, `ortho-atomic` or `norm-atomic`. A value of $U$ is
only meaningful together with the projectors it was chosen for. Here we take
antiferromagnetic FeO, with the moments of the two iron atoms opposite, and we
compute its ground state with `get_scf()`:

```python
from defuqumat import Calculator

folder = "docs/figures/dft-plus-u_feo/"
feo = Calculator.from_file(folder + "feo.in", pseudo_dir="tests/data/pseudo")
scf = feo.get_scf()
print(f"total energy {scf.total_energy:.8f} Ry")
print(f"Hubbard term {scf.energy_terms['hubbard']:.8f} Ry")
for atom, (up, down, total) in scf.hubbard_occupations.items():
    print(f"atom {atom + 1}: Tr n = {up:.4f} up, {down:.4f} down")
```

```
total energy -174.47156068 Ry
Hubbard term 0.31370438 Ry
atom 3: Tr n = 4.9911 up, 1.8454 down
atom 4: Tr n = 1.8454 up, 4.9911 down
```

`scf.energy_terms["hubbard"]` is $E_U$, and it is part of the total energy.
`scf.hubbard_occupations` gives, for each atom with a $U$, the traces of
$n^{I\uparrow}$ and $n^{I\downarrow}$ and their sum. `scf.ns` is the occupation
matrix itself, with the axes (spin, atom, $m$, $m'$). We see that atom 3 has 4.99
electrons in its majority shell and 1.85 in its minority shell, and atom 4 is its
mirror image. The forces and the stress include the Hubbard term.

![The DOS of FeO and the eigenvalues of the occupation matrices, with U close to zero and with U = 4.3 eV.](img/dft-plus-u_feo.png#only-light)
![The DOS of FeO and the eigenvalues of the occupation matrices, with U close to zero and with U = 4.3 eV.](img/dft-plus-u_feo-dark.png#only-dark)

In the right panel we see what $U$ does. The eigenvalues of the occupation matrix
are the occupations of the natural orbitals of the shell. With $U = 4.3$ eV they
move towards 0 and 1, and their mean distance from the nearer of 0 and 1 falls from
0.200 to 0.148. The left panel shows the DOS of the cell. $U$ pushes the empty
minority $d$ states up, to about 2 eV, and the occupied $d$ states down. But a
narrow peak stays at the Fermi level, so this run is still a metal. The reason is
the one minority $d$ electron of Fe$^{2+}$. In this cell it is shared by a pair of
degenerate orbitals, with 0.59 in each. As long as the symmetry keeps the two
orbitals equal, $U$ cannot fill one and empty the other. To get the insulator, we
must break this degeneracy, for example with a lower symmetry or a starting
occupation (`starting_ns_eigenvalue`, as in `pw.x`). The run above uses neither.

??? example "Code of this figure (docs/figures/dft-plus-u_feo/)"

    ```python title="compute.py"
    --8<-- "dft-plus-u_feo/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "dft-plus-u_feo/plot.py"
    ```

## The full interaction, and the double counting

Let us now go beyond a single $U$. $U$ alone is the cost of two electrons in the
same shell, and it does not say which two. Hund's rule says that two electrons in
different orbitals with parallel spins cost less. To describe this, you add a `J`
on the card, which asks for the full interaction of the shell, built from the
Slater integrals $F^0 = U$, $F^2$ and $F^4$:

```
HUBBARD {atomic}
U Fe1-3d 4.3
J Fe1-3d 1.0
```

The functional already counts part of the interaction, and the double counting
removes it. You choose how with `hubbard_double_counting`. The fully localized
limit (`'fll'`, the default) assumes that each orbital is full or empty, which fits
a magnetic insulator. Around mean field (`'amf'`) assumes that the shell is filled
evenly, so an even shell gets no correction:

```python
for name in ("feo-fll.in", "feo-amf.in"):
    scf = Calculator.from_file(folder + name, pseudo_dir="tests/data/pseudo").get_scf()
    print(f"{name}: E_U = {scf.energy_terms['hubbard']:+.4f} Ry")
```

```
feo-fll.in: E_U = +0.2321 Ry
feo-amf.in: E_U = -0.0052 Ry
```

The two runs have the same $U$ and $J$, and only the double counting differs. We
see that $E_U$ changes from 0.23 Ry to almost nothing. Which one to use is a
statement about the material.

With `hubbard_slater = 'yukawa'` the code computes $F^2$, $F^4$ and $J$ from the
radial function of the shell itself, with a screened Coulomb interaction. The
screening length is chosen so that $F^0$ is the $U$ on the card. This needs a PAW
dataset, because only PAW has the all-electron partial wave.

!!! note "Limits"
    The intersite $V$, background channels, the orbital-resolved form and the
    `wf` and `pseudo` projectors give an error. PAW with a GGA and a
    noncollinear magnetization together gives an error.

    A collinear DFT+U run needs `nosym` when its magnetic group has operations
    combined with time reversal. A spinor run does not. A fixed tensor moment
    needs a spinor run.

    `hubbard_slater = 'yukawa'` needs a PAW dataset. A norm-conserving or
    ultrasoft orbital gives a far too small $F^0$.

    A Hubbard $U$ gives an error in the force theorem for the anisotropy, in the
    magnons and in the linear response. The relaxed anisotropy works.

The notebook for this page is
[13, DFT+U](https://github.com/joselado/defuqumat/blob/master/notebooks/13_dft_plus_u.ipynb).
