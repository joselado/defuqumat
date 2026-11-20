# Reading and writing a Quantum ESPRESSO ground state

In this page we load the converged ground state of a `pw.x` run and continue it
here, and then we write our own ground state so that `pw.x` can continue it. Quantum ESPRESSO and this code solve the same Kohn-Sham equations, so with the
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
both. Reading the HDF5 file needs the Python package `h5py`. We tested it on a
nickel run with spin-orbit coupling, from a `pw.x` 7.5 built with HDF5. The folder
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

We checked this on many cases against `pw.x` 7.5, each with `conv_thr=1e-10`. Silicon
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
ones to $4\times10^{-6}$.

The state of `pw.x` can have more than the density and the PAW occupations, and
the code loads all of it. A DFT+U run writes its occupation matrix in
`occup.txt`, which becomes `starting_ns`. A meta-GGA run writes its kinetic energy
density in `ekin-density.dat`, which becomes `starting_tau`. A noncollinear PAW
run writes four components of the PAW occupations, the charge and the three
components of the magnetization, and they are loaded as they are. For nickel with
a Hubbard $U$ and its moment tilted off the $z$ axis, the occupation matrix
loaded from `occup.txt` agrees with our own converged one to $4\times10^{-6}$,
and the SCF stops after one iteration. On an O2 molecule with PAW and its moment
in the plane, the SCF also stops after one iteration. Two platinum atoms with PAW
and spin-orbit coupling take 3 iterations, both at the energy of `pw.x`.

## Writing our state for pw.x

We now go the other way. We converge silicon here and write the state as a save
folder of `pw.x`. We do it with `calc.write_qe_save()`:

```python
calc = Calculator.from_keywords(pseudo_dir="tests/data/pseudo", **silicon)
result = calc.get_scf(conv_thr=1e-10)
print(f"defuqumat  {result.total_energy:.8f} Ry  {result.iterations} iterations")
calc.write_qe_save("out/pwscf.save")

import os
print(sorted(os.listdir("out/pwscf.save")))

silicon["qe"]["electrons"]["startingpot"] = "file"
open("si-restart.in", "w").write(pw_input_text(**silicon))
```

```
defuqumat  -89.27293945 Ry  8 iterations
['charge-density.dat', 'data-file-schema.xml', 'paw.txt']
```

The argument is again the save folder, `<outdir>/<prefix>.save`, so here `pw.x`
looks for it with `outdir='out'` and the default prefix `pwscf`. The folder has
the density and, for PAW, `paw.txt`. A DFT+U run also gets `occup.txt` and a
meta-GGA run gets `ekin-density.dat`. Each file has the layout that `pw.x` writes
itself. The last line writes the same input with `startingpot='file'`, which tells
`pw.x` to start from the density in the folder. Then we run `pw.x` on it:

```bash
pw.x -in si-restart.in > si-restart.out
```

`pw.x` prints "The initial density is read from file", and it stops after one
iteration at -89.27293944 Ry, with an estimated accuracy of $6\times10^{-12}$ Ry
in its first iteration. From atomic charges it needs 8 iterations.

One detail is important. `pw.x` reads the coefficients of the density in the
order they are stored, and it does not look at their $\mathbf G$ vectors. So the
code writes the $\mathbf G$ vectors in exactly the order that `pw.x` makes them.
We checked this order against the files of `pw.x` on ten cells, a triclinic one
among them. With a wrong order `pw.x` reads a wrong density and does not say so.

We checked the writer on the same kind of cases as the reader. `pw.x` continues
our state in one to three iterations for silicon with PAW, an O2 molecule with
PAW and `nspin=2`, silicon in a triclinic cell, two platinum atoms with PAW and
spin-orbit coupling, an O2 molecule with PAW as a noncollinear magnet, nickel with
a Hubbard $U$ with `nspin=2` and as a tilted noncollinear magnet, and boron
nitride with spin-orbit coupling and a Hubbard $U$. In each case this is as fast
as `pw.x` from its own folder. As a control, we wrote the O2 density as two spin
channels instead of charge and magnetization. Then `pw.x` renormalizes it, starts
with an accuracy of 0.43 Ry and needs 12 iterations. Writing takes 9 ms on one
core for this silicon cell. On one core, reading the files takes 2.5 ms, and the SCF
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

    A DFT+U run of `pw.x` gives an error if our run has no Hubbard term, uses
    other projectors or another `lda_plus_u_kind`. A background Hubbard manifold,
    `Hubbard_V` and the orbital-resolved form give an error, because the code has
    none of them. A meta-GGA run needs `ekin-density.dat` in the folder. A PAW run
    whose folder has no `paw.txt` gives an error. The wavefunctions of `pw.x` are
    not read, so the first iteration diagonalizes from scratch.

    The folder we write is for a restart of `pw.x` with `startingpot='file'` only.
    Its `data-file-schema.xml` has only what our reader checks, so `pp.x`,
    `projwfc.x` and an `nscf` run of `pw.x` cannot read it. Run the short restart
    first, and `pw.x` then writes its own full folder for them. A spin spiral
    cannot be written, because `pw.x` has none. The HDF5 form is never written,
    so the restart must use a `pw.x` built without HDF5. A `pw.x` built with
    HDF5 looks only for `charge-density.hdf5`. It prints "Cannot read rho : file
    not found", starts from atomic charges and gives no error.
    Do not restart with `memory='small'` on more than one process per pool.
    There `pw.x` numbers the G vectors by process and not in the global order
    we write. It then reads a wrong density and does not say so. We found this
    in the source of `pw.x` and have not run it.
    With k-point pools (`DEFUQUMAT_POOLS`) `write_qe_save` gives an error,
    because every pool would write the same files at the same time.
    A meta-GGA restart was checked with a `pw.x` 7.2 built with libxc. It reads
    our `ekin-density.dat` correctly. But `input_dft = 'tb09'` in `pw.x` calls
    libxc with no Laplacian and with `c = 1`, and it adds TPSS correlation.
    So it is not our TB09, which uses LDA correlation. The two
    codes then converge to different states, and `pw.x` needs 7 to 13
    iterations from ours.
    The same holds for a noncollinear GGA run. Our default there is
    `noncollinear_gga = 'grad_m'`, and `pw.x` uses `'kubler'`. The two differ
    when the direction of the magnetization changes in space. Spin-orbit
    coupling always does this. So set `noncollinear_gga = 'kubler'` in
    `&system` for the SCF whose state you write. On iron with DFT+U and
    spin-orbit coupling, `pw.x` then restarts in one iteration, and in 15
    from the `'grad_m'` state. On an arsenic atom it is one against three.
    A spin-orbit run with `soc_scale` other than 1 (`System.with_soc_scale`)
    is written with a warning. `pw.x` has no such switch, and with `lspinorb`
    it always takes the whole coupling. So `pw.x` reads the file as a start
    and converges to another state. We have not run this restart.

    The density can only start an SCF. There is no route that computes bands or a
    density of states from it with no SCF at all. With the same input this costs one
    or two iterations.

The notebook for this page is
[56, continuing a Quantum ESPRESSO ground state](https://github.com/joselado/defuqumat/blob/master/notebooks/56_quantum_espresso_restart.ipynb).
