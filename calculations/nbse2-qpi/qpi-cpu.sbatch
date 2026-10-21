#!/bin/bash
# NbSe2 QPI by the T-matrix on CPU cores: the SCF of the input, the bands once on the
# whole N x N grid (the impurity's supercell), then qpi.py's maps. The same driver as
# qpi-gpu.sbatch and qpi-bilayer-gpu.sbatch, for when the card queue does not move.
#
# Not submitted by any tool here (CLAUDE.local.md): by hand, after `sbatch --test-only`.
#
#   sbatch --partition=batch-csl --time=2-00:00:00 --mem=64G --job-name=nbse2-qpi \
#          --export=ALL,DEFUQUMAT_REPO=/scratch/work/ladovj1/apps/defumat-qpi2 \
#          calculations/nbse2-qpi/qpi-cpu.sbatch ../nbse2-fermi-surface/nbse2.in 60 16 nbse2-qpi
#   sbatch --partition=batch-milan --time=5-00:00:00 --mem=128G --job-name=nbse2-qpi-bl \
#          --export=ALL,DEFUQUMAT_REPO=/scratch/work/ladovj1/apps/defumat-qpi2 \
#          calculations/nbse2-qpi/qpi-cpu.sbatch nbse2-bilayer.in 60 32 nbse2-bilayer-qpi
#
# cpus   8: a k-point gains at most about 2x from threads (PERFORMANCE.md, "Threads"),
#        and below 8 the XLA thread pool can park (defuqumat/__init__.py). k-point pools
#        do not apply, since they drop the wavefunctions and QPI needs them.
# time   not measured on CPU: the D22 smokes ran at reduced cutoff on the card. Estimated
#        from the FFT grid at 1 to 5 s a k-point for the monolayer (16 bands) and 3 to 15 s
#        for the bilayer (32 bands, a cell 1.35x taller), over 3600 k-points.
# mem    the states are 9 GB (monolayer) and 24 GB (bilayer) complex.
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/work/ladovj1/calculations/nbse2-qpi/out/cpu-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/nbse2-qpi/out/cpu-%j.err

WORK=/scratch/work/ladovj1/calculations/nbse2-qpi
REPO=${DEFUQUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-qpi2}
PY=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin/python3
mkdir -p "$WORK/out" "$WORK/cache-cpu"
unset XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION \
      DEFUQUMAT_K_BATCH DEFUQUMAT_BAND_BATCH DEFUQUMAT_MEMORY_MODE
export JAX_PLATFORMS=cpu PYTHONPATH="$REPO" DEFUQUMAT_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DEFUQUMAT_CACHE_DIR="$WORK/cache-cpu"

echo "=== node ==="; hostname; date; nproc
"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD
cd "$REPO/calculations/nbse2-qpi" || exit 1
"$PY" -u qpi.py "$1" "$2" "$3" "$WORK/out/$4-${SLURM_JOB_ID}.npz"
echo "exit $?"; date
