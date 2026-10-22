#!/bin/bash
# The I122 check on Triton's CPUs (D22 refuses the login, 2026-10-22).
# Array task 0: the fast unit file of the unit (-m 'not slow');
# tasks 1, 2: i122.py at conv_thr 1e-9 and 1e-11 (the ultrasoft oxygen chain's
# quarter turn at ecutrho = 240, widened box (60,60,27), against its four-cell
# supercell on (60,60,108)).
#
# Not submitted by any tool here (CLAUDE.local.md): by hand, after `sbatch --test-only`.
#
#   sbatch --partition=batch-csl --array=0-2 notes/incompat/I122-harness/triton.sbatch
#
# cpus   8: the supercell is four ultrasoft oxygens, spinor, on a 60x60x108 box.
# time   not measured; the 200 Ry pair ran within the D22 slow-test budget.
# mem    16G is a margin, not a measurement.
#SBATCH --job-name=inc-I122
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/work/ladovj1/calculations/incompat/I122/out/%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/incompat/I122/out/%A_%a.err

WORK=/scratch/work/ladovj1/calculations/incompat/I122
REPO=${DEFUQUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-I122}
PY=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin/python3
H=$REPO/notes/incompat/I122-harness
mkdir -p "$WORK/out" "$WORK/cache-cpu"
unset XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION \
      DEFUQUMAT_K_BATCH DEFUQUMAT_BAND_BATCH
export JAX_PLATFORMS=cpu PYTHONPATH="$REPO" DEFUQUMAT_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DEFUQUMAT_CACHE_DIR="$WORK/cache-cpu"

echo "=== node ==="; hostname; date
git -C "$REPO" rev-parse HEAD
cd "$REPO" || exit 1
case "$SLURM_ARRAY_TASK_ID" in
  0) "$PY" -m pytest -q -rs --tb=short -m 'not slow' -p no:cacheprovider \
        tests/unit/test_spiral_gvectors.py ;;
  1) "$PY" -u "$H/i122.py" "$WORK/out/i122-1e-9-${SLURM_ARRAY_JOB_ID}.json" 1e-9 ;;
  2) "$PY" -u "$H/i122.py" "$WORK/out/i122-1e-11-${SLURM_ARRAY_JOB_ID}.json" 1e-11 ;;
esac
echo "exit $?"; date
