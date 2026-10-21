#!/bin/bash
# The I116 and I117 checks on Triton's CPUs, for while D22 refuses the login (2026-10-22).
# Array task 0: the fast unit and regression files of the unit (-m 'not slow');
# task 1: i116.py (two-cell hydrogen under B cos(pi x), three starts, two fields, two conv_thr);
# task 2: i117.py (antiferromagnetic H2 ultracell (3,1,1) against the explicit supercell).
#
# Not submitted by any tool here (CLAUDE.local.md): by hand, after `sbatch --test-only`.
#
#   sbatch --partition=batch-csl --array=0-2 notes/incompat/ultraseed-harness/triton.sbatch
#
# cpus   4: hydrogen cells of two to six atoms, where threads buy little.
# time   not measured: the D22 slot budget for these harnesses was a few hours on one core.
# mem    small cells; 8G is a margin, not a measurement.
#SBATCH --job-name=inc-ultraseed
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/work/ladovj1/calculations/incompat/ultraseed/out/%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/incompat/ultraseed/out/%A_%a.err

WORK=/scratch/work/ladovj1/calculations/incompat/ultraseed
REPO=${DEFUQUMAT_REPO:-/scratch/work/ladovj1/apps/defumat-ultraseed}
PY=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin/python3
H=$REPO/notes/incompat/ultraseed-harness
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
        tests/unit/test_ultracell_seed.py tests/regression/test_ultracell.py ;;
  1) "$PY" -u "$H/i116.py" "$WORK/out/i116-${SLURM_ARRAY_JOB_ID}.json" 0.002,0.02 1e-8,1e-10 16 ;;
  2) "$PY" -u "$H/i117.py" "$WORK/out/i117-${SLURM_ARRAY_JOB_ID}.json" 3 8,16 yes ;;
esac
echo "exit $?"; date
