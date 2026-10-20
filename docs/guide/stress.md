# Stress and the cell

In this page we compute the stress tensor and the equation of state, we relax
the cell at an applied pressure, and we compute the elastic constants.

## The stress tensor

We deform the cell with a small strain $\epsilon$, so each lattice vector becomes
$\mathbf a_i \to (1 + \epsilon)\,\mathbf a_i$. The stress is the derivative of
the energy with respect to the strain, per volume:

$$
\sigma_{ab} = -\frac{1}{\Omega}
  \left.\frac{\partial E}{\partial \epsilon_{ab}}\right|_{\psi\ \text{fixed}},
\qquad
P = \tfrac13\,\mathrm{tr}\,\sigma .
$$

As for the forces, the wavefunctions stay fixed, because the energy is
stationary in them, and we take this derivative with automatic differentiation.
A positive pressure means that the crystal wants to expand. The stress tells us
whether the lattice constant is at its equilibrium, and it is what a diamond
anvil cell or a strained substrate puts on the crystal. We compute it with
`calc.get_stress()`:

```python
from defuqumat import Calculator

calc = Calculator.from_file("docs/figures/stress_eos/si-eos.in",
                            pseudo_dir="tests/data/pseudo")
stress = calc.get_stress()
print(stress.kbar.diagonal().round(3))    # xx, yy, zz in kbar
print(f"pressure {stress.pressure_kbar:.2f} kbar")
```

```
[7.369 7.369 7.369]
pressure 7.37 kbar
```

`stress.kbar` is the tensor in kbar, and `stress.tensor` the same in
Ry/bohr$^3$. The cubic symmetry makes the tensor diagonal with three equal
entries, and the three printed entries are indeed equal. `calc.get_stress(terms=True).terms` gives
the kinetic, Hartree, local, nonlocal, exchange-correlation and Ewald parts. If
the input has `tstress = .true.`, the SCF already computes the stress and
`get_stress()` returns it.

## The equation of state

We now get the pressure in a second way. The pressure is also $P = -dE/dV$, a
derivative along a series of separate SCF runs at different volumes. This curve
$E(V)$ is the equation of state. Its minimum is the equilibrium volume, and its
curvature gives the bulk modulus, $B = V\,d^2E/dV^2$.

![The equation of state of silicon, and the pressure from the stress and from -dE/dV.](img/stress_eos.png#only-light)
![The equation of state of silicon, and the pressure from the stress and from -dE/dV.](img/stress_eos-dark.png#only-dark)

The left panel is the total energy of silicon against the volume, at 40 Ry, for
nine lattice constants from 0.96 to 1.04 times 10.20 bohr. The right panel shows
the pressure in two ways. The line with points is the pressure from the stress,
one SCF per volume. The dashed line is $-dE/dV$ along the curve on the left,
which uses central differences, so it has no value at the two end points. On the
seven inner volumes the two agree to 1.9 kbar or better, on a pressure from
$-83$ to 161 kbar. The difference comes from the finite difference along the
curve and from the Pulay stress, below. The pressure is zero at 267.5
bohr$^3$, a lattice constant of 10.23 bohr, which is the equilibrium of silicon
in LDA with this pseudopotential. The measured lattice constant is 10.26 bohr.

The Pulay stress is an error of the basis. At a fixed `ecutwfc`, a larger cell
holds more plane waves, so the energy also falls because the basis gets better.
On this cell it is about 44 kbar at 12 Ry, 1.1 kbar at 40 Ry and 0.11 kbar at
60 Ry. So you should converge the cutoff on the stress, not only on the energy,
before you trust a lattice constant.

??? example "Code of this figure (docs/figures/stress_eos/)"

    ```python title="compute.py"
    --8<-- "stress_eos/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "stress_eos/plot.py"
    ```

## Relaxing the cell at a pressure

Let us now relax the cell. `get_relax(variable_cell=True)` is the `vc-relax` of
`pw.x`, which moves the atoms and the lattice vectors together and minimizes the
enthalpy

$$
H = E + P\,\Omega .
$$

$P$ is the applied pressure, `press` in the `&cell` namelist, in kbar. At the
minimum of $H$ the stress is $\sigma = P\,\mathbf 1$, so a relaxed crystal
carries the applied pressure and not zero stress. Here we relax the eight-atom
cubic cell of silicon at 100 kbar:

```python
cell = Calculator.from_file("docs/figures/stress_eos/si8-pressure.in",
                            pseudo_dir="tests/data/pseudo")
result = cell.get_relax(variable_cell=True)

print(f"{result.nsteps} steps")
print(f"volume {result.steps[0].volume:.2f} -> {result.volume:.2f} bohr^3")
print(f"enthalpy {result.enthalpy:.8f} Ry")
print(f"Pulay error {result.pulay_error:.1e} Ry")
```

```
4 steps
volume 1061.21 -> 972.47 bohr^3
enthalpy -62.71117414 Ry
Pulay error 2.4e-03 Ry
```

We see that the volume falls by 8%. The relaxation runs in the plane-wave basis
of the first cell, and at the end the code runs one more SCF in the basis of the
relaxed cell. The energy difference between the two is `result.pulay_error`, the
error of the fixed basis, and if it is large, the relaxed volume is not
converged in the cutoff. `result.cell` holds the relaxed lattice vectors in
bohr, and `result.stress` the stress at the end. `cell_dofree` in `&cell`
chooses which parts of the cell can move, as in `pw.x`.

## Elastic constants

Finally we compute the elastic constants, the second derivative of the energy
with respect to the strain:

$$
C_{ijkl} = \frac{1}{\Omega}\,\frac{\partial^2 E}{\partial \epsilon_{ij}\,\partial \epsilon_{kl}} .
$$

They give the speed of sound, which ultrasound and Brillouin scattering measure.
We compute them from the linear response of the wavefunctions to a strain, in
one calculation and with no strained cells, with
`silicon.get_elastic_constants()`:

```python
silicon = Calculator.from_file("docs/figures/stress_eos/si-elastic.in",
                               pseudo_dir="tests/data/pseudo")
c = silicon.get_elastic_constants()
C = c.voigt                                # 6 x 6, Voigt notation, GPa
print(f"C11 = {C[0, 0]:.1f}   C12 = {C[0, 1]:.1f}   C44 = {C[3, 3]:.1f} GPa")
print(f"bulk modulus {c.bulk_modulus:.1f} GPa")
```

```
C11 = 209.4   C12 = 68.0   C44 = 134.0 GPa
bulk modulus 115.1 GPa
```

`c.voigt` is the $6\times6$ matrix in Voigt notation, in GPa, and `c.tensor` the
full tensor in Ry/bohr$^3$. The measured values of silicon are $C_{11} = 165.7$,
$C_{12} = 63.9$ and $C_{44} = 79.6$ GPa, and two things make the numbers here
larger. First, this cell has a 12 Ry cutoff and a $2\times2\times2$ mesh, which
is far from converged. Second, the elastic constants here are clamped-ion,
meaning that the atoms keep their crystal coordinates inside the strained cell.
In the diamond structure this is the full answer for $C_{11}$ and $C_{12}$, by
symmetry, but for $C_{44}$ it is not. A shear strain moves the two atoms of the
cell against each other, and this internal relaxation lowers $C_{44}$.

!!! note "Limits"
    The elastic constants work with norm-conserving, ultrasoft and PAW datasets. For an
    ultrasoft or PAW dataset they need every k-point in memory at once, so a run that
    streams the k-points in chunks gives an error. This happens in memory mode on a GPU,
    for any chunk, and on a CPU in `memory_mode='memory'` with a `k_batch` smaller than
    the k-mesh. The elastic constants also need `nspin = 1`, an insulator and the full
    k-mesh (`nosym`), so a spin-polarized, noncollinear or spin-orbit run,
    a metal and a symmetry-reduced k-set each give an error. The full k-mesh
    must also be an unshifted Monkhorst-Pack grid. A shifted grid with `nosym`
    gives an error, because a shifted grid does not have the symmetry of the
    crystal. They are clamped-ion only. The page about piezoelectric and strain responses has the
    other strain responses, which also work for ultrasoft and PAW.

    A strain response that did not converge gives an error, because a diverged
    response gives an elastic tensor that is not even symmetric under
    $C_{ijkl} = C_{klij}$. This happens mainly on a slab with vacuum, where the
    response diverges at `alpha_mix = 0.7`. If you set `mixing_mode='linear'`,
    first try the default Anderson mixer. Otherwise use a larger
    `max_iterations` or a smaller `alpha_mix`, which you pass to
    `get_strain_response()`. `allow_unconverged=True` in
    `get_elastic_constants()` turns the error off for a diagnostic run.

    A potential-only meta-GGA (`tb09`, `bj06`) has no stress. Its total energy
    is not the minimum of any functional, so its derivative has no meaning.

    An electric field (`tefield` or `lelfield`) gives no stress, and so no
    `vc-relax`, as in `pw.x`.

The notebooks for this page are
[15, the stress tensor](https://github.com/joselado/defuqumat/blob/master/notebooks/15_stress.ipynb),
[23, variable-cell relaxation](https://github.com/joselado/defuqumat/blob/master/notebooks/23_variable_cell_relaxation.ipynb)
and
[21, electrostriction and the elastic constants](https://github.com/joselado/defuqumat/blob/master/notebooks/21_electrostriction.ipynb).
