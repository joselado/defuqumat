# Performance and many cores

In this page we compute the same small ground states in several ways: with the
kernels compiled or not, with one k-point or all k-points at once, in the two
memory modes and on several cores. We will see that the energy does not
change, and that only the time and the memory do. The measured timings, on
many cells and against other codes, are in
[PERFORMANCE.md](https://github.com/joselado/defuqumat/blob/master/PERFORMANCE.md).

## Where the time goes

Let us first ask where an SCF spends its time. At each iteration we solve one
eigenproblem per k-point,

$$
\hat H_{\mathbf k}\,|\psi_{n\mathbf k}\rangle
  = \varepsilon_{n\mathbf k}\,\hat S\,|\psi_{n\mathbf k}\rangle ,
\qquad
n(\mathbf r) = \sum_{\mathbf k} w_{\mathbf k} \sum_n f_{n\mathbf k}\,
  |\psi_{n\mathbf k}(\mathbf r)|^2 ,
$$

and the k-points only meet in the sum that gives the density and, in a
metal, in the Fermi level. So the
k-points are independent problems, which is where the parallel work is. Inside
one k-point the work is a chain of small steps, one Fourier transform per band
and small matrix products, and many threads do not help much there. This is
why the code gives each process only a few threads, and leaves the other
cores for the k-point pools of the section on pools.

The settings that decide this are environment variables, and they have to be
set before Python imports `defuqumat`. A normal launch on a CPU is

```bash
export DEFUQUMAT_THREADS=4          # threads of one process
export JAX_PLATFORMS=cpu            # stay on the CPU even if a GPU is visible
python3 run.py
```

where `run.py` is a normal script with `Calculator.from_file(...)`. In a
script, you can also set them with `os.environ` before the import. The
variables are:

- **`DEFUQUMAT_THREADS`**: how many CPUs one process uses. If you do not set
  it, the code uses 4, because more threads on one k-point were measured to be
  slower. `0` or `off` leaves the CPUs as they are. The code only reduces the
  set of CPUs it gets from `taskset` or Slurm, it never adds to it.
  `OMP_NUM_THREADS` does not set the number of threads here. A long run that
  stops and uses no CPU at all is a known stall of the thread pool at a small
  number of threads, and the cure is a larger `DEFUQUMAT_THREADS`, such as 8.
- **`JAX_PLATFORMS`**: `cpu` or `cuda`. If you do not set it, the code runs on
  a GPU when it finds one.
- **`DEFUQUMAT_CACHE_DIR`**: the folder where the compiled kernels are kept
  between runs, `~/.cache/defuqumat/jax` if you do not set it. `off` switches
  the cache off.
- **`DEFUQUMAT_MEMORY_MODE`**: `speed` or `memory`, the memory mode of the
  whole process, see below.
- **`DEFUQUMAT_K_BATCH`** and **`DEFUQUMAT_BAND_BATCH`**: how many k-points
  and how many bands are computed at the same time, see below. `all` means all
  of them.
- **`DEFUQUMAT_POOLS`**: the number of processes one calculation runs as, see
  the section on pools.
- **`DEFUQUMAT_MEM_FRACTION`**: the part of the GPU memory the code may take,
  0.9 if you do not set it. Use `off` when another program shares the card.
- **`DEFUQUMAT_GATHER_BUDGET`**: the size in MB above which each
  symmetrisation walks the symmetry operations one at a time, instead of
  holding a copy of the field for every operation. It is 64 if you do not set
  it, and `0` means that it always walks. The result changes only in its last
  digits.

Why is the first run of a script slow? The code is written in JAX, which
compiles each kernel the first time it meets it, for the shapes of this
crystal. Here we switch the cache off and compute the same SCF twice in one
process:

```python
import os
os.environ["DEFUQUMAT_CACHE_DIR"] = "off"   # no kernels from earlier runs

import time
from defuqumat import Calculator

for run in (1, 2):
    calc = Calculator.from_file("docs/figures/bands_silicon/si.in",
                                pseudo_dir="tests/data/pseudo")
    start = time.perf_counter()
    scf = calc.get_scf()
    print(f"run {run}: {time.perf_counter() - start:5.1f} s, "
          f"E = {scf.total_energy:.10f} Ry")
```

```
run 1:  10.7 s, E = -15.8392999169 Ry
run 2:   0.5 s, E = -15.8392999169 Ry
```

The first SCF takes 10.7 s and the second takes 0.5 s, with the same energy
to the last digit, so about 95% of the first run is compilation and not
physics. The second run reuses the kernels of the first, since the crystal,
the cutoff and the mesh are the same. With the cache switched on, as it is if
you do not set `DEFUQUMAT_CACHE_DIR`, the kernels are also kept on the disk,
so a second script with the same shapes does not compile them again. This is
why a time you measure has to be the time of a second call, never of the
first one. These times are from one desktop computer. On yours the two numbers
will be different, but the first run will still be much longer than the
second.

## How many k-points at once: `k_batch`

We now choose how many k-points are solved at the same time. This is
`k_batch`. With `k_batch=1` the code solves one k-point after the other,
so the memory holds the work of one k-point. With
`k_batch=None` it solves all k-points in one step, which needs the memory of
all of them at once. Here we take silicon on its $6\times6\times6$ mesh and
compute the SCF both ways:

```python
from defuqumat import Calculator

for k_batch in (1, None):
    calc = Calculator.from_file("docs/figures/bands_silicon/si.in",
                                pseudo_dir="tests/data/pseudo", k_batch=k_batch)
    scf = calc.get_scf()
    print(f"k_batch={k_batch}: {calc.system.kpoints.nk} k-points, "
          f"{scf.iterations} iterations, E = {scf.total_energy:.12f} Ry")
```

```
k_batch=1: 28 k-points, 7 iterations, E = -15.839299916940 Ry
k_batch=None: 28 k-points, 7 iterations, E = -15.839299916940 Ry
```

Both runs take 7 iterations and give the same total energy to all twelve
printed digits, which is what we expect, since `k_batch` changes only the
order of the work and not the equations. The 28 k-points are the irreducible
part of the shifted $6\times6\times6$ mesh.

The values of `k_batch` are:

- **`1`**: one k-point at a time. This is the default on a CPU, where it is
  also the fastest choice.
- **an integer**: this number of k-points at a time.
- **`None`**: all k-points at once. On a CPU this is slower and needs more
  memory, and on a GPU it is the fastest choice when it fits.
- **`"fit"`**: on a GPU, the largest number of k-points that fits on the
  card. This is the default in the `memory` mode on a GPU. On a CPU it gives
  one k-point at a time.

You give `k_batch` to the `Calculator`, or set `DEFUQUMAT_K_BATCH` for the
whole process. The bands have the same kind of setting, `band_batch`, which is
one band at a time on a CPU and all bands on a GPU.

## One switch for the memory: `memory_mode`

On a GPU the first question is whether the calculation fits on the card.
`memory_mode` answers it with one setting, and it chooses the other settings
for you. `"speed"` keeps everything on the card at once. `"memory"` keeps only
a part of the k-points on the card, computes the projectors of the
pseudopotential only when they are used, and keeps the wavefunctions in the
memory of the host when not all k-points fit on the card at once. Before a run, `calc.estimate()` tells us what it will cost,
without allocating anything:

```python
from defuqumat import Calculator

for mode in ("speed", "memory"):
    calc = Calculator.from_file("docs/figures/bands_silicon/si.in",
                                pseudo_dir="tests/data/pseudo", memory_mode=mode)
    size = calc.estimate()
    print(f"{mode:6s}  k-points at once {size.k_batch} of {size.nk}, "
          f"bands at once {size.band_batch}, projectors {size.projectors}, "
          f"peak {size.peak_bytes / 2**20:.1f} MB")
print(f"all k-points at once: peak {size.at_k_batch(None).peak_bytes / 2**20:.1f} MB")
```

```
speed   k-points at once 1 of 28, bands at once 1, projectors store, peak 5.7 MB
memory  k-points at once 1 of 28, bands at once 1, projectors rebuild, peak 4.8 MB
all k-points at once: peak 18.4 MB
```

On a CPU both modes solve one k-point and one band at a time, so the only
difference is the projectors, which the `memory` mode builds again when it
needs them instead of keeping them for all 28 k-points. This takes the peak
from 5.7 MB to 4.8 MB. If we instead solve all 28 k-points at once, the peak
grows to 18.4 MB, roughly three times more, and this is the memory that grows
with the k-mesh. For two atoms all of these numbers are small, and they
become the question for a cell of tens of atoms on a dense mesh.

The physics, and so the numbers, are the same in both modes up to round-off.
On a CPU the default is `"speed"`, and on a GPU it is `"memory"`. You give
`memory_mode` to the `Calculator`, or set `DEFUQUMAT_MEMORY_MODE` for the
whole process, and a setting you give for `k_batch` or `band_batch` wins over
the mode. The command `defuqumat size scf.in` prints the same estimate for an
input file, with every array in it, in GiB, so it is useful for large cells.

## Many cores: k-point pools

We now run one calculation on several cores. With `DEFUQUMAT_POOLS` the
calculation runs as that number of processes, the pools, and each pool solves
its own part of the k-points. The pools add up the density and share it at
every iteration, so the result is the result of one process, up to the order
in which the k-points are added. You do not change the script. Here we take
silicon with one atom moved away from its site, so it has fewer symmetries and
more k-points, and `run.py` is:

```python
from defuqumat import Calculator
from defuqumat.parallel import current_pools

calc = Calculator.from_file("docs/figures/forces_relaxation/si-displaced.in",
                            pseudo_dir="tests/data/pseudo")
scf = calc.get_scf()
forces = calc.get_forces()
pools = current_pools()
if pools.rank == 0:
    print(f"{pools.size} pool(s), {calc.system.kpoints.nk} k-points: "
          f"E = {scf.total_energy:.10f} Ry, max force {forces.max_force:.10f} Ry/bohr")
```

We run it once as one process, and then as two pools on one machine. Each
pool gets its rank, and all pools get the address of the first one:

```bash
python3 run.py

export DEFUQUMAT_POOLS=2 DEFUQUMAT_COORDINATOR=localhost:23456 DEFUQUMAT_THREADS=1 JAX_PLATFORMS=cpu
DEFUQUMAT_POOL_RANK=0 python3 run.py &
DEFUQUMAT_POOL_RANK=1 python3 run.py
wait
```

```
1 pool(s), 18 k-points: E = -15.7874037086 Ry, max force 0.0603966971 Ry/bohr
[Gloo] Rank 0 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
[Gloo] Rank 1 is connected to 1 peer ranks. Expected number of connected peer ranks is : 1
2 pool(s), 18 k-points: E = -15.7874037086 Ry, max force 0.0603966971 Ry/bohr
```

The two `[Gloo]` lines are printed by the library that connects the pools,
and they say that each pool found the other one. One process and two pools
give the same energy and the same largest force to all printed digits, so
the pools change where the 18 k-points are solved and nothing else.

Under Slurm, the rank and the address come from the job, so the launch is

```bash
#SBATCH --signal=TERM@600
export DEFUQUMAT_POOLS=$SLURM_NTASKS DEFUQUMAT_THREADS=$SLURM_CPUS_PER_TASK
srun python3 run.py
```

with one Slurm task per pool and `--cpus-per-task` threads in each. The best
use of the cores is many pools with one or two cores each, since the threads
of one k-point give little. A calculation without symmetry, for example a
magnet with spin-orbit coupling, has the whole mesh as its k-points, and this
is where pools give the most. The ground state, the forces, the stress and the
relaxations run with pools, also for a spin spiral, a magnetic field and a
constrained moment. If you give `checkpoint_dir`, the SCF writes a checkpoint,
and it can continue with any number of pools, also with one process. The line
`#SBATCH --signal=TERM@600` asks Slurm for a SIGTERM ten minutes before the end
of the job, and then all pools stop at the same iteration and write their
checkpoint.

## GPUs

The same script runs on a GPU without any change, once JAX is installed with
GPU support. On a GPU the
defaults change: `memory_mode` is `"memory"`, `k_batch` is `"fit"` and all
bands of a k-point are computed at once. In the `memory` mode, the memory on
the card grows very little with the k-mesh, so a dense mesh fits on a small card.
If you ask for `"speed"` and leave the other settings to it, the code first estimates the memory, and if the
calculation would not fit on the card it uses `"memory"` instead and gives a
warning. With `band_precision="single"` the bands of the SCF are computed in
single precision, while the density, the potential and the energies stay in
double precision. With `band_precision="mixed"` the SCF starts in single
precision and changes the bands to double precision in the same run, at the
first iteration whose threshold single precision cannot reach, so the state it
converges to is a double one. This is a mode for speed only, and no result in
this guide was computed in it. By default the code takes 90% of the memory of the card,
and `DEFUQUMAT_MEM_FRACTION=off` gives the default of JAX, which is what you
want when another program uses the same card. The speedups we measured on
several cards are in
[PERFORMANCE.md](https://github.com/joselado/defuqumat/blob/master/PERFORMANCE.md).

!!! note "Limits"
    With pools, only `get_scf`, `get_forces`, `get_stress` and `get_relax`
    run. Every other method gives an error that names it. More pools than
    k-points give an error. An SCF with a `scf_solver` other than the normal
    mixing gives an error with pools, and so do the forces and the stress
    with a `method` other than the default `"autodiff"`. A starting state
    (`starting_from`) whose wavefunctions were held by a different number of
    pools gives an error, so you start from its density instead, or use the
    same number of pools. A checkpoint does not have this limit.

    The SCF does not write a checkpoint when the magnetic field comes from a
    fixed-spin-moment scheme (`constrained_magnetization = 'fsm'`,
    `'atomic fsm'` or `'atomic fsm direction'`). It says so once and runs on
    without one, because this field changes at every iteration and no input
    file says what it reached, so a restart would go back to the input field.
    An applied field and a penalty constraint are saved normally.
    `checkpoint_every`, the number of iterations between two checkpoints (10
    if you do not give it), must be at least 1. A run that continues from its
    checkpoint gives an error if you also give `magnetization`, because the
    run already has its own. A checkpoint holds the wavefunctions, which can
    be tens of gigabytes on a large cell, so you put `checkpoint_dir` on a
    scratch directory. We have not timed the default of 10 on a large cell, so
    you may need to set `checkpoint_every` yourself.

    `k_batch="fit"` is chosen for the k-points of the SCF, and a band path or
    an NSCF run uses the same number. It grows the number of k-points only when
    all the bands of one k-point fit on the card, otherwise it stays at one
    k-point. A function that gets `k_batch="fit"` as an argument with no
    crystal to size it gives an error. `DEFUQUMAT_K_BATCH=fit` gives no error
    there. A batch of k-points runs each Davidson step until the slowest
    k-point of the batch has converged. This changes only the speed, and only
    if the k-points need different numbers of steps, which they did not on any
    cell we measured.

    The memory mode only reduces the memory that grows with the k-mesh. The
    memory of one k-point, of the dense grid or of the augmentation charges of
    an ultrasoft or PAW dataset is the same in both modes. The memory of the
    host has to hold all the wavefunctions. If you give your own `k_batch` or
    `projectors`, or set `DEFUQUMAT_K_BATCH`, the code does not check that
    `"speed"` fits, and it does not switch to `"memory"`. A solve with more
    bands than the SCF, such as a spectrum, is checked again for its own
    number of bands, and it can switch to `"memory"` or to fewer bands at a
    time for that solve only, with a warning. The piezoelectric tensor with
    `method="zstar_eu"` and the spectra computed as a sum over states still
    move all the wavefunctions to the card. A `memory_mode` you give to a
    `Calculator` sets up that calculator, and its methods after the SCF follow
    the same mode. These are the bands, the NSCF run, the DOS, the PDOS, STM,
    transport, the structure factors, the spiral scan, the topological
    invariants and the optical and magnetic spectra. The magnetic anisotropy
    and the torque do not follow the `memory_mode` of the `Calculator`. They take it
    from `DEFUQUMAT_MEMORY_MODE` and the platform.

    `band_precision="single"` works only for the SCF of a collinear
    norm-conserving calculation, with one or two spin channels. An ultrasoft
    or PAW dataset, spinors, a spiral, DFT+U and a meta-GGA give an error.
    The forces, the stress and every response give an error on a calculation
    whose bands are in single precision, because a gradient taken through
    single-precision bands carries its round-off as a plausible force. This
    also holds for a `Calculator` built with `"mixed"`, which keeps its
    single-precision setup. For these, you converge again in double
    precision, starting from the single-precision result with
    `starting_from`. A `conv_thr` below $10^{-7}$ Ry gives an error in single
    precision, because the energy does not follow it there. In single
    precision the lowest threshold of the eigensolver is
    $4\,\epsilon_{32}\,E_\mathrm{cut}$, with $\epsilon_{32}$ the machine
    precision of float32 and $E_\mathrm{cut}$ the value of `ecutwfc`, because
    $10^{-13}$ Ry would stall the eigensolver at its budget. On a GPU, single
    precision makes the float32 matrix products run at full precision, without
    TF32, for the whole process. A calculation moved to a new geometry or to a
    band path builds its projectors again in double precision, so it runs fully
    in double precision and does not stay in single precision.

    The code gives a `RuntimeWarning`, once per run, when the eigensolver uses
    its whole budget of 100 steps and leaves bands not converged. The steps of
    each iteration are in `scf.history[i]["davidson_iterations"]`.
    `DEFUQUMAT_ETHR_MIN` sets the lowest threshold of the eigensolver. If you
    do not set it, this threshold is $10^{-13}$ Ry in double precision, as in
    pw.x, and the value above in single precision. A higher value costs
    accuracy in the energy and in the forces.

    On an ultrasoft or PAW dataset, the table of the augmentation charges
    reaches $|\mathbf G| = 2\sqrt{E_\mathrm{rho}}$, with $E_\mathrm{rho}$ the
    value of `ecutrho`, as in pw.x. A strain so large that $|\mathbf G|$ goes
    past this end gives NaN, not a value clamped at the end of the table,
    because a clamped value would look smooth and of the right size but give
    a wrong stress.

    Every number we validated was measured with `projectors="store"`, which
    keeps the projectors in memory. The `memory` mode uses
    `projectors="rebuild"`, which builds them again when they are used.

The notebook for this page is
[03, the eigensolver and performance](https://github.com/joselado/defuqumat/blob/master/notebooks/03_eigensolver_and_performance.ipynb).
