# Electric fields

In this page we put a molecule or a slab in a uniform electric field, we put a
crystal in a finite field through the Berry phase, and we compute the
polarization of a crystal.

## A field across a molecule or a slab

How do we put a uniform field into a periodic cell? A linear potential is not
periodic, so we use a saw-tooth, as `pw.x` does. The potential rises linearly
over most of the cell along the reciprocal vector $\mathbf b_{\rm edir}$, and
then it falls back in a thin region of width `eopreg`, which must be in the
vacuum. You switch the field on with `tefield = .true.`, and `eamp` is its
amplitude in Hartree atomic units. `edir` and `emaxpos` set its direction and
where it falls back.

A molecule or a slab with a dipole creates a field in the vacuum through its
periodic images. The dipole correction, `dipfield = .true.`, cancels it by
adding a second saw-tooth whose slope follows the dipole of the cell. The energy
of the field, for the electrons and the ions together, is:

$$
E_f = \frac{\Omega}{4\pi}\,e^2\Big[-E_{\rm amp}\,D + \tfrac12 D^2\Big],
\qquad
\frac{\Omega}{4\pi}\,D = \sum_a Z_a\, s(x_a) - \int n(\mathbf r)\, s(x)\, d\mathbf r .
$$

Here $s$ is the saw-tooth, and $\Omega D/4\pi$ is the dipole of the cell in
$e\,$bohr. The $D^2/2$ term is there only with `dipfield`. The potential and the
forces are derivatives of this energy, so a relaxation in a field works. The
input is the input of `pw.x`. Here we take CO in a 10 bohr box with a field of
0.001 Ha a.u. along the bond:

```python
from defuqumat import Calculator

co = Calculator.from_file("docs/figures/electric-field_dipole/co.in",
                          pseudo_dir="tests/data/pseudo")
scf = co.get_scf()
print(f"dipole        {scf.efield_dipole:+.5f} e bohr")
print(f"field energy  {scf.efield_energy:.4e} Ry")
print(co.get_forces().forces[:, 2])     # along the field, Ry/bohr
```

```
dipole        -0.03376 e bohr
field energy  8.1848e-05 Ry
[ 0.11696953 -0.11696953]
```

The code warns that the run uses no symmetry, because a field breaks the
symmetry, and `pw.x` does the same. `scf.efield_dipole` is the dipole of the
cell, and `scf.efield_energy` is $E_f$, which is part of the total energy. We see
that the two forces are equal and opposite. The field pulls the ions and the
electrons in opposite directions, so it stretches or compresses the bond.

![The dipole of CO against the field, and the change of the density along the bond.](img/electric-field_dipole.png#only-light)
![The dipole of CO against the field, and the change of the density along the bond.](img/electric-field_dipole-dark.png#only-dark)

The left panel shows the dipole at five fields. It is linear in the field, and
its slope is the polarizability along the bond, $\alpha_{zz} = 13.5$ bohr$^3$.
The measured static polarizability of CO along the bond is about 15.6 bohr$^3$.
The number here is a rough estimate, because the box is small and the cutoff is
low. The right panel shows the change of the electron density, averaged over
the $xy$ plane, when the field goes from 0 to 0.004 Ha a.u. The electrons move
towards $-z$, to the oxygen side, against the field, and the change is zero far
from the molecule.

??? example "Code of this figure (docs/figures/electric-field_dipole/)"

    ```python title="compute.py"
    --8<-- "electric-field_dipole/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "electric-field_dipole/plot.py"
    ```

## A finite field in a crystal

Let us now put a field in a crystal. In a crystal there is no vacuum, so the
saw-tooth has nowhere to fall back. We also cannot add $-e\mathbf E\cdot\mathbf r$
to the Hamiltonian, because $\mathbf r$ is not periodic. So we minimize the
electric enthalpy instead (Souza, Iniguez and Vanderbilt, arXiv:cond-mat/0205633):

$$
F[\psi] = E_{\rm KS}[\psi] - \Omega\,\mathbf E\cdot\mathbf P[\psi] .
$$

$\mathbf P$ is the Berry-phase polarization of the occupied bands, which we
compute below. Its derivative with respect to the wavefunctions couples each
k-point to its two neighbours on a string of k-points along the field. This is
`lelfield` of `pw.x`, with the same input: `efield` in Ry atomic units, `gdir`
the direction, `nppstr` the points of a string, and an explicit list of
k-points. We switch the field on from a converged state without field, which is
what `pw.x` does too.

```python
import numpy as np

folder = "docs/figures/electric-field_dipole/"
ground = Calculator.from_file(folder + "si8-nofield.in",
                              pseudo_dir="tests/data/pseudo").get_scf()
text = open(folder + "si8-field.in").read()

dipole = {}
for e in (-0.001, 0.001):
    run = Calculator.from_text(text.replace("efield=0.001", f"efield={e}"),
                               pseudo_dir="tests/data/pseudo")
    scf = run.get_scf(starting_from=ground)
    dipole[e] = scf.berry_dipoles[0]       # electronic part, Ry a.u.
    print(f"E = {e:+.3f}   p_el = {dipole[e]:.6f}")

volume = float(run.system.cell.volume)
chi = (dipole[0.001] - dipole[-0.001]) / 0.002 / volume
print(f"eps_inf = {1 + 4 * np.pi * chi:.2f}")
```

```
E = -0.001   p_el = -0.929969
E = +0.001   p_el = 0.929969
eps_inf = 12.08
```

`scf.berry_dipoles` holds the electronic dipole, the ionic dipole and the number
of $2\pi$ jumps of the phase, in Ry atomic units. The slope of the electronic
dipole is the susceptibility, and $\varepsilon_\infty = 1 + 4\pi\chi$. In Ry
atomic units the dipole and the field each carry a factor $\sqrt2$, and the two
cancel. The measured $\varepsilon_\infty$ of silicon is 11.7, and LDA gives a
larger value because its gap is too small. This mesh of $3\times3\times7$ points
is also not converged. The page about phonons computes $\varepsilon_\infty$ from
linear response, with no field at all.

The field must stay below the breakdown field of the k-mesh. Above it, electrons
can tunnel from the valence to the conduction band between neighbouring
k-points, and the enthalpy has no minimum.

## The Berry-phase polarization

We now compute the polarization of a crystal. The polarization of a crystal is
not the dipole of its unit cell, because that dipole depends on where you cut
the cell. What is a property of the crystal is a phase, the Berry phase that the
occupied bands collect when $\mathbf k$ crosses the zone once along
$\mathbf b_i$ (King-Smith and Vanderbilt):

$$
\Omega\,\mathbf P = \sum_i \frac{\varphi_i}{2\pi}\,\mathbf a_i,
\qquad
\varphi_i = \operatorname{Im}\ln \prod_j \det M(\mathbf k_j, \mathbf k_{j+1})
  + 2\pi\sum_a Z_a\, s_{ai} .
$$

$M$ is the matrix of overlaps of the occupied states at two neighbouring
k-points. The first term is the electronic phase, and the second the ionic one,
with $s_{ai}$ the crystal coordinate of atom $a$. The phase is defined modulo
$2\pi$, so $\mathbf P$ is defined only modulo a quantum. Moving one electron by
one lattice vector changes the dipole of the cell by exactly this quantum, and
the crystal is the same. What is physical is a difference of $\mathbf P$ between
two structures, on the same branch. This is what a ferroelectric hysteresis loop
measures, and it gives the Born charges and the piezoelectric tensor. Here we
take AlAs and compute it with `alas.get_polarization()` on more and more
strings:

```python
alas = Calculator.from_file(folder + "alas.in", pseudo_dir="tests/data/pseudo")
for n in (2, 4, 6, 8):
    pol = alas.get_polarization(transverse=(n, n))
    print(f"{n}x{n} strings: electronic {pol.electronic_phase:+.5f}   "
          f"P = {pol.polarization_si:+.5f} C/m^2")
print(f"ionic {pol.ionic_phase:+.5f}, modulo {pol.quantum:.0f}, "
      f"or {pol.quantum_si:.5f} C/m^2")
```

```
2x2 strings: electronic +0.00876   P = -0.34909 C/m^2
4x4 strings: electronic -0.00184   P = -0.36444 C/m^2
6x6 strings: electronic -0.00331   P = -0.36657 C/m^2
8x8 strings: electronic -0.00355   P = -0.36691 C/m^2
ionic -0.25000, modulo 1, or 1.44708 C/m^2
```

`get_polarization()` reads `lberry`, `gdir` and `nppstr` from the input, as in
`pw.x`, and you can also pass them as keywords. Here `gdir` counts from 0, so
`gdir=2` is the third reciprocal vector. `transverse=(n1, n2)` sets the number
of strings in the two other directions, $4\times4$ if you do not give it. The
phases are in units of $2\pi$, and `pol.quantum_si` is the quantum in C/m$^2$.

We see that the ionic phase is exactly $-1/4$, modulo 1. The electronic phase is
small, and it needs enough strings. On $2\times2$ strings, the mesh of notebook
34, it has the wrong sign. From $6\times6$ to $8\times8$ it moves by
$2.4\times10^{-4}$, so $P$ is converged to about 0.3 mC/m$^2$. AlAs has no
inversion center, so its phase is not fixed by symmetry. In a crystal with
inversion, the phase is 0 or half a quantum.

!!! note "Limits"
    A saw-tooth field (`tefield`) gives no stress, so no `vc-relax`, as in
    `pw.x`. It also gives an error with linear response, real-time propagation
    and the ultracell. A charged plate (`gate`) needs a charged cell, and
    `tot_charge` gives an error. `dipfield` without `tefield` gives an error,
    because in `pw.x` it does nothing.

    A finite field by the Berry phase (`lelfield`) gives an error for an
    ultrasoft or PAW dataset, for `nspin = 2` and spinors, and for a metal. It
    also gives an error for `efield_cart` (the field in three directions), for
    an automatic k-mesh, for a cell where $\mathbf a_g$ is not along the
    Cartesian axis $g$, with DFT+U, a spin spiral, `tefield`, and linear
    response. It gives no stress.

    The Berry-phase polarization gives an error for a metal, for `nspin = 2`
    and for a spin spiral. It works for a spinor run, where the quantum is half
    the scalar one, and for ultrasoft and PAW datasets.

The notebooks for this page are
[53, a molecule in an electric field](https://github.com/joselado/defuqumat/blob/master/notebooks/53_electric_field_molecule.ipynb),
[54, silicon in a finite electric field](https://github.com/joselado/defuqumat/blob/master/notebooks/54_finite_electric_field.ipynb)
and
[34, electric polarization](https://github.com/joselado/defuqumat/blob/master/notebooks/34_electric_polarization.ipynb).
