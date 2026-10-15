# Total energy and the density

On this page we compute the ground state of a crystal: its total energy and its
electron density. Everything else in this guide starts from this ground state.

## The Kohn-Sham total energy

The ground state is the minimum of the Kohn-Sham total energy, as a functional of
the density $n(\mathbf r)$:

$$
E[n] = T_s[n] + \int v_{\rm ext}(\mathbf r)\, n(\mathbf r)\, d\mathbf r
       + E_{\rm H}[n] + E_{\rm xc}[n] + E_{\rm Ewald}.
$$

$T_s$ is the kinetic energy of non-interacting electrons with density $n$.
$v_{\rm ext}$ is the potential of the ions, here a pseudopotential. $E_{\rm H}$
is the classical Coulomb energy of the density, and $E_{\rm xc}$ is the
exchange-correlation energy, which is LDA or a GGA. $E_{\rm Ewald}$ is the
Coulomb energy of the ions.

The density comes from the occupied Kohn-Sham states:

$$
\left[-\nabla^2 + v_{\rm eff}[n](\mathbf r)\right]\psi_{n\mathbf k}
  = \varepsilon_{n\mathbf k}\,\psi_{n\mathbf k},
\qquad
n(\mathbf r) = \sum_{n\mathbf k} w_{\mathbf k}\, f_{n\mathbf k}\,
  |\psi_{n\mathbf k}(\mathbf r)|^2 .
$$

The equation is in Rydberg units, so the kinetic operator is $-\nabla^2$. The
potential $v_{\rm eff}$ depends on the density, and the density depends on the
states. So the problem is a fixed point: the density that goes in must be the
density that comes out. This is the self-consistent field, or SCF. The code
starts from a sum of atomic densities, solves for the states, builds a new
density, mixes it with the old one, and repeats until the two agree.

## Running the SCF

```python
from defuqumat import Calculator

calc = Calculator.from_file("docs/figures/bands_silicon/si.in",
                            pseudo_dir="tests/data/pseudo")
scf = calc.get_scf()

print(f"converged in {scf.iterations} iterations")
print(f"total energy {scf.total_energy:.8f} Ry")
for name, value in scf.energy_terms.items():
    print(f"{name:<13} {value:>15.8f} Ry")
```

```
converged in 7 iterations
total energy -15.83929992 Ry
one-electron       4.79887528 Ry
hartree            1.07474038 Ry
xc                -4.81315700 Ry
ewald            -16.89975858 Ry
```

The energy terms are the ones `pw.x` prints. The one-electron term is the band
energy minus the double counting. The four terms add up to the total energy.

The SCF stops when the estimated error of the energy is below `conv_thr`, in Ry.
This has the same meaning as in `pw.x`, so the two codes stop at the same point.
You set it in the `&electrons` namelist or as a keyword:
`calc.get_scf(conv_thr=1e-10)`. Other keywords of `get_scf` are
`mixing_beta`, `mixing_mode` and `max_iterations`.

The result is stored in `calc`. A second call to `get_scf()` with the same
keywords returns it without running again. A call with other keywords runs the
SCF again and replaces it.

## The density

The density is on the real-space grid of the run, in electrons per bohr$^3$. The
first axis is the spin channel, and it is there also for a spin-unpolarized run:

```python
n = scf.density                        # electrons/bohr^3 on the real-space grid
omega = calc.system.cell.volume        # bohr^3
print(n.shape)
print(f"{float(n.mean()) * float(omega):.6f} electrons in the cell")
```

```
(1, 20, 20, 20)
8.000000 electrons in the cell
```

The integral of the density is the number of valence electrons, four per silicon
atom. X-ray diffraction measures the Fourier components of the total density,
and these are on the page about structure factors.

![The valence density of silicon in the (1-10) plane.](img/ground-state_density.png#only-light)
![The valence density of silicon in the (1-10) plane.](img/ground-state_density-dark.png#only-dark)

The figure shows the valence density of silicon in the $(1\bar10)$ plane. This
plane holds the zigzag chains of bonds. The circles are the atoms. The density
has its maxima in the middle of the bonds, between two atoms, and not on the
atoms. This is the covalent bond of silicon. On the atoms the valence density is
small, because the pseudopotential removes the core and the valence states have
nodes there. The large dark regions are the open channels of the diamond
structure.

??? example "Code of this figure (docs/figures/ground-state_density/)"

    ```python title="compute.py"
    --8<-- "ground-state_density/compute.py"
    ```

    ```python title="plot.py"
    --8<-- "ground-state_density/plot.py"
    ```

## When the SCF does not converge

Most crystals converge with the default mixing. Some cells are hard: a metal slab
with vacuum, a magnet with a soft direction, a cell with DFT+U. For these,
`mixing_mode='robust'` chooses the mixing from the cell and checks every step:

```python
scf = calc.get_scf(mixing_mode="robust")
```

If the cell has vacuum and a smeared occupation, the mode screens the charge
with the local density of states at the Fermi level. Otherwise it uses plain
Anderson mixing. Each step is judged by the Kohn-Sham energy of the states that
the step produces. This energy cannot go below the ground-state energy. A step is
kept if this energy or the residual goes down. Otherwise the step is cut in half.
This is the adaptive damping of Herbst and Levitt (arXiv:2109.14018).

If the SCF reaches `max_iterations` without converging, the code warns, and
`scf.converged` is `False`. All quantities computed from this result then come
from an unconverged density.

!!! note "Limits"
    `K_POINTS gamma` asks for the half-sphere storage of the gamma trick. For an
    ultrasoft or PAW dataset, a run with symmetry, a spinor or spiral run and a
    Hubbard $U$, the code uses an explicit $k = 0$ on the full sphere instead.
    The result is the same and the cost is twice as large, and the code warns.

    A functional that the pseudopotentials name and the code does not implement
    gives an error. It is never replaced by LDA.

    The robust mode is not guaranteed to converge. On a cell with several
    self-consistent solutions, it can find a different one than plain Anderson.
    Compare the energy and the moments, not only the number of iterations. With
    Marzari-Vanderbilt or Methfessel-Paxton smearing, its energy is not a lower
    bound, so it keeps a step on the residual only, and warns. It gives an error
    for tetrahedra, a potential-only meta-GGA, `lelfield`, k-point pools and
    checkpoints.

The notebooks for this page are
[02, silicon](https://github.com/joselado/defuqumat/blob/master/notebooks/02_silicon_scf_and_bands.ipynb)
and
[17, reaching self-consistency](https://github.com/joselado/defuqumat/blob/master/notebooks/17_reaching_self_consistency.ipynb).
