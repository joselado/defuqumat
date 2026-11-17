# Reading and writing a Quantum ESPRESSO ground state

Quantum ESPRESSO and this code solve the same Kohn-Sham equations. If the crystal,
the pseudopotential, the functional and the cutoffs are the same, the two codes
converge to the same density. So a density converged by `pw.x` is already the
self-consistent density here, and we can load it instead of computing it again.

Here we take silicon with a PAW pseudopotential and a $4\times4\times4$ k-mesh.
We ran `pw.x` once on this crystal and kept its save folder. Started from that
density, our SCF stops after **one iteration**, at **-89.27293945 Ry**, which is
the total energy of `pw.x` to all printed digits. From atomic charges it takes
eight iterations. At the end we go the other way: we write our own ground state
as a save folder, and `pw.x` continues it, also in one iteration.


```python
import numpy as np
import matplotlib.pyplot as plt
from defuqumat import Calculator

silicon = dict(lattice='fcc', a=5.43, atoms=[('Si', (0, 0, 0)), ('Si', (0.25, 0.25, 0.25))],
               pseudos={'Si': 'Si.pz-n-kjpaw_psl.0.1.UPF'}, ecut=272.0, ecutrho=2177.0,
               kmesh=(4, 4, 4), qe={'electrons': {'conv_thr': 1e-10}})
calc = Calculator.from_keywords(pseudo_dir='../tests/data/pseudo', **silicon)
seed = calc.get_qe_seed('../tests/data/qe/save/si-paw-444')
result = calc.get_scf(**seed)
print(f'total energy  {result.total_energy:.8f} Ry  in {result.iterations} iteration')
```

    total energy  -89.27293945 Ry  in 1 iteration


## What is read

`pw.x` writes its converged state in the folder `<outdir>/<prefix>.save`. The
density is stored as its Fourier coefficients:

$$\rho(\mathbf r) = \sum_{\mathbf G} \rho(\mathbf G)\, e^{i\mathbf G\cdot\mathbf r},$$

with each $\mathbf G$ given by its three integer coordinates on the reciprocal
lattice. So the coefficient at $\mathbf G = 0$ times the cell volume is the number
of electrons, and we put each coefficient at its place on our FFT grid. For a
PAW pseudopotential the density alone is not the whole state. Each atom also has
the occupations of its projectors, which give the density inside the PAW sphere,
and `pw.x` writes them in a second file. We load both. For a spin-polarized run
the file has the charge and the magnetization, and we split them into the two
spin channels.

The crystal above is the input that `pw.x` ran, written by the same keywords. The
code checks that the save folder belongs to this crystal: the same cell, the same
atoms in the same places, the same number of electrons, the same density cutoff
and the same spin setting. If one of them is different, it gives an error,
because a density on one lattice means nothing on another.


```python
fresh = Calculator.from_keywords(pseudo_dir='../tests/data/pseudo', **silicon).get_scf()
print('start                  iterations   total energy (Ry)')
print(f'pw.x, atomic charges            8   -89.27293945')
print(f'here, pw.x density     {result.iterations:10d}   {result.total_energy:.8f}')
print(f'here, atomic charges   {fresh.iterations:10d}   {fresh.total_energy:.8f}')
```

    start                  iterations   total energy (Ry)
    pw.x, atomic charges            8   -89.27293945
    here, pw.x density              1   -89.27293945
    here, atomic charges            8   -89.27293945


The table compares three runs. The run started from the `pw.x` density and the
run started from atomic charges end at the same energy, to $10^{-9}$ Ry, and both
agree with `pw.x`. The difference is only how many iterations it takes to get
there.

## Using the loaded state

The result is a normal ground state, so everything that needs one works on it. We
plot how fast each run converges, and the band structure computed from the loaded
state.


```python
bands = calc.get_bands('L-G-X', nbnd=8)

fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
for run, label in [(fresh, 'from atomic charges'), (result, 'from the pw.x density')]:
    left.semilogy([h['iteration'] for h in run.history],
                  [h['accuracy'] for h in run.history], 'o-', label=label)
left.axhline(1e-10, color='gray', ls='--', lw=1)
left.set_xlabel('SCF iteration')
left.set_ylabel('estimated SCF accuracy (Ry)')
left.legend()
bands.plot(ax=right, color='C1', lw=1.8)
gamma = np.argmin(np.linalg.norm(np.asarray(bands.kpoints.coords), axis=1))
right.set_xticks(bands.path_length[[0, gamma, -1]], ['L', r'$\Gamma$', 'X'])
right.set_title('bands from the loaded state')
plt.tight_layout()
plt.show()
```


    
![png](56_quantum_espresso_restart_files/56_quantum_espresso_restart_5_0.png)
    


On the left, the run from atomic charges needs eight iterations to reach the
threshold of $10^{-10}$ Ry, the dashed line, and the run from the `pw.x` density
is below it after the first one. On the right are the bands of silicon from the
loaded state, with the top of the valence band at zero.

This is useful when a ground state is expensive, for example a large cell with a
dense k-mesh. You converge it once with `pw.x`, and here you start every
calculation that needs it from that density. The loaded state is checked against
the `pw.x` energy, and the mapping of the PAW occupations against our own
converged ones, in `tests/unit/test_qe_density.py`. The save folder of a DFT+U
run also holds the occupation matrix, the one of a meta-GGA run the kinetic
energy density, and they are loaded too, for every spin setting and with
spin-orbit coupling.

## Writing our state for pw.x

Now we go the other way. We write the ground state converged here, the run from
atomic charges, as a save folder of `pw.x`. It holds the density and, for PAW, the
occupations of the projectors, in the files and the layout that `pw.x` writes
itself. Here we read the folder back to see that it is the same state.


```python
import os, tempfile

folder = os.path.join(tempfile.mkdtemp(), 'pwscf.save')
calc.write_qe_save(folder, fresh)
print(sorted(os.listdir(folder)))
again = calc.get_scf(**calc.get_qe_seed(folder))
print(f'read back  {again.total_energy:.8f} Ry  in {again.iterations} iteration')
```

    ['charge-density.dat', 'data-file-schema.xml', 'paw.txt']


    read back  -89.27293945 Ry  in 1 iteration


The folder is what `pw.x` reads with `startingpot='file'` in the `&electrons`
namelist, with `outdir` its parent and `prefix='pwscf'`. With the same input
`pw.x` stops after **one iteration** at -89.27293944 Ry, where from atomic charges
it needs eight, so it continues our state as fast as its own. The density is
stored as Fourier coefficients in a fixed order of the $\mathbf G$ vectors, and
`pw.x` reads them in that order without looking at the vectors, so the code
writes them in exactly the order `pw.x` makes them. The folder is for this
restart only: after it, `pw.x` writes its own full save folder, which its other
tools then read.
