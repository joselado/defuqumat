# Continuing a Quantum ESPRESSO ground state

In this page we load the converged ground state of a `pw.x` run and continue it
here. Quantum ESPRESSO and this code solve the same Kohn-Sham equations, so with the
same crystal, pseudopotentials, functional and cutoffs the two codes converge to the
same density. A density converged by `pw.x` is then already self-consistent here,
and our SCF stops after one or two iterations. We take silicon with a PAW
pseudopotential on a $4\times4\times4$ k-mesh.

## Running pw.x on the same crystal

We first write the crystal as keywords and let the code write the `pw.x` input for
it, so the two codes see exactly the same cell:

```python
from defuqumat.io.keywords import pw_input_text

silicon = dict(lattice="fcc", a=5.43,
               atoms=[("Si", (0, 0, 0)), ("Si", (0.25, 0.25, 0.25))],
               pseudos={"Si": "Si.pz-n-kjpaw_psl.0.1.UPF"},
               ecut=272.0, ecutrho=2177.0, kmesh=(4, 4, 4),
               qe={"control": {"outdir": "out", "pseudo_dir": "tests/data/pseudo"},
                   "electrons": {"conv_thr": 1e-10}})
open("si.in", "w").write(pw_input_text(**silicon))
```

The cutoffs are in eV, so `ecut=272.0` is 19.99 Ry and `ecutrho=2177.0` is
160.0 Ry. The `qe` dictionary adds the namelist entries that only `pw.x` needs,
here the folder where it writes its state. Then we run `pw.x` on this file:

```bash
pw.x -in si.in > si.out
```

`pw.x` converges in 8 iterations, at a total energy of -89.27293945 Ry, and leaves
its state in the folder `out/pwscf.save`. The density is in `charge-density.dat`,
or in `charge-density.hdf5` if `pw.x` was compiled with HDF5. The code reads
both, but we have only tested the first, because our `pw.x` has no HDF5. The folder
`tests/data/qe/save/si-paw-444` of the repository is the one this run wrote, so you
can try the next step without `pw.x`.

## Loading the density

The file holds the Fourier coefficients of the density,

$$\rho(\mathbf r) = \sum_{\mathbf G} \rho(\mathbf G)\, e^{i\mathbf G\cdot\mathbf r},$$

with each $\mathbf G$ given by its three integer coordinates on the reciprocal
lattice. So $\rho(\mathbf G = 0)$ times the cell volume is the number of electrons,
and the code places each coefficient at its point of our FFT grid. For a
spin-polarized run the file has the charge $n$ and the magnetization $m_z$, which
the code turns into the two spin channels $(n \pm m_z)/2$. A noncollinear run has
$n, m_x, m_y, m_z$, and these are taken as they are. For PAW the density is not the
whole state, because each atom also has the occupations of its projectors, which
give the density inside the PAW sphere. `pw.x` writes them in `paw.txt`, and the
code loads them too.

We compute it with `calc.get_qe_seed()`:

```python
from defuqumat import Calculator

calc = Calculator.from_keywords(pseudo_dir="tests/data/pseudo", **silicon)
seed = calc.get_qe_seed("tests/data/qe/save/si-paw-444")
print(sorted(seed))
print(seed["starting_density"].shape)
result = calc.get_scf(**seed)
print(f"from the pw.x density  {result.total_energy:.8f} Ry  {result.iterations} iteration")
```

```
['diago_thr_init', 'starting_becsum', 'starting_density']
(1, 32, 32, 32)
from the pw.x density  -89.27293945 Ry  1 iteration
```

The argument of `get_qe_seed` is the save folder itself, `<outdir>/<prefix>.save`.
The result is a dictionary of options for `get_scf`. It has the density, with the
shape (spin, grid), here one channel on the $32^3$ grid of the run, the PAW
occupations, and `diago_thr_init=1e-5`. This is the threshold of the first
diagonalization that `pw.x` itself uses when it starts from a density. With the
looser threshold of a start from atomic charges, the first iteration loses most of
what the density gives, and the run takes about as many iterations as a fresh one.

The SCF stops after one iteration at -89.27293945 Ry, which is the energy of `pw.x`
to all printed digits. From atomic charges the same run here takes 8 iterations and
ends at the same energy. The result is a normal ground state, so `get_bands`,
`get_dos` and every other method that needs one work on it as usual.

We checked this on five cases against `pw.x` 7.5, each with `conv_thr=1e-10`. Silicon
with a norm-conserving pseudopotential under `K_POINTS gamma` takes 1 iteration
against 8 from atomic charges, and silicon with PAW 1 against 7. An O2 molecule in
a box with an ultrasoft pseudopotential and `nspin=2` takes 2 iterations against 11,
the same molecule with PAW 2 against 12, and with an ultrasoft pseudopotential as a
noncollinear magnet 2 against 12. In each case the energy is the one of `pw.x` to
$10^{-9}$ Ry. With spin-orbit coupling, on fully relativistic ultrasoft
pseudopotentials, germanium takes 1 iteration against 8, at the energy of `pw.x`
to $10^{-9}$ Ry. A single arsenic atom in a box, a noncollinear magnet with a moment
of 3 $\mu_B$ off every axis, takes 4 iterations against 11. There both of our runs
end $9\times10^{-7}$ Ry below `pw.x`, and since the run from atomic charges shows the
same difference, it comes from our SCF and not from the loaded density. The PAW occupations loaded from `paw.txt` agree with our own converged
ones to $4\times10^{-6}$. On one core, reading the files takes 2.5 ms, and the SCF
from the density takes 0.16 s, the same time as the restart of `pw.x` from its own
density (`startingpot='file'`).

!!! note "Limits"
    The save folder must belong to the same calculation. A different cell, a
    different number of atoms, atoms in other places or in another order, a
    different number of electrons, a different `ecutrho` and a different spin
    setting each give an error, because a density on one lattice or one G-sphere
    means nothing on another. A different `ecutwfc` with the same `ecutrho` is
    not checked, and we have not measured how many iterations it then takes.
    A different pseudopotential with the same number of electrons is not detected,
    and the run then converges to its own answer from a worse start.

    A DFT+U run gives an error, because its state also has the occupation matrix in
    `occup.txt`, which is not read. A meta-GGA run gives an error, because its state
    also has the kinetic energy density in `ekin-density.dat`, which is not read. A
    noncollinear PAW run gives an error, because its projector occupations are not
    mapped. A PAW run whose folder has no `paw.txt` gives an error. The wavefunctions
    of `pw.x` are not read, so the first iteration diagonalizes from scratch.

    The density can only start an SCF. There is no route that computes bands or a
    density of states from it with no SCF at all. With the same input this costs one
    or two iterations.

The notebook for this page is
[56, continuing a Quantum ESPRESSO ground state](https://github.com/joselado/defuqumat/blob/master/notebooks/56_quantum_espresso_restart.ipynb).
