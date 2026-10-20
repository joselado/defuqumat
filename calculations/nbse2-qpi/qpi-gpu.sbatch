#!/bin/bash
# 1H-NbSe2 QPI by the T-matrix on one H100: the 12x12 SCF of
# calculations/nbse2-fermi-surface/nbse2.in, the bands once on the whole 60x60
# grid (3600 k-points, the impurity's supercell), then qpi.py's maps.
#
# Not submitted by any tool here (CLAUDE.local.md): by hand, after `sbatch --test-only`.
#
#   sbatch --export=ALL,DEFUQUMAT_REPO=/scratch/work/ladovj1/apps/defumat-qpi calculations/nbse2-qpi/qpi-gpu.sbatch
#
# time   06:00:00: the band solve is 3600 k-points of 16 bands on a 9804-plane-wave
#        sphere; the QPI maps after it are minutes.
# mem    64G of host memory: the states are 9 GB complex and are read to the host
#        one k-point at a time, beside the device copy.
#SBATCH --job-name=nbse2-qpi
#SBATCH --partition=gpu-h100-80g
#SBATCH --time=06:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=/scratch/work/ladovj1/calculations/nbse2-qpi/out/gpu-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/nbse2-qpi/out/gpu-%j.err

WORK=/scratch/work/ladovj1/calculations/nbse2-qpi
REPO=${DEFUQUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-qpi}
PY=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin/python3
mkdir -p "$WORK/out" "$WORK/cache-gpu"
unset XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION JAX_PLATFORMS \
      DEFUQUMAT_K_BATCH DEFUQUMAT_BAND_BATCH DEFUQUMAT_MEMORY_MODE
export PYTHONPATH="$REPO" DEFUQUMAT_THREADS=off OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export DEFUQUMAT_CACHE_DIR="$WORK/cache-gpu"

echo "=== node ==="; hostname; date
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
"$PY" -c "import jax; print('jax', jax.__version__, jax.devices())"
git -C "$REPO" rev-parse HEAD
cd "$REPO/calculations/nbse2-qpi" || exit 1
"$PY" -u qpi.py ../nbse2-fermi-surface/nbse2.in 60 16 "$WORK/out/nbse2-qpi-${SLURM_JOB_ID}.npz"
echo "exit $?"; date
