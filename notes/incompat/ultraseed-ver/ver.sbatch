#!/bin/bash
# Verifier of incompat/ultraseed (I116, I117). Submitted by hand.
#SBATCH --job-name=ver-ultraseed
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/work/ladovj1/calculations/incompat/ultraseed-ver/out/%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/incompat/ultraseed-ver/out/%A_%a.err
WORK=/scratch/work/ladovj1/calculations/incompat/ultraseed-ver
# Read-only use of the implementer's clean worktree at 37cff983 (code identical to 8b531bf5).
REPO=/scratch/work/ladovj1/apps/defumat-ultraseed-v2
PY=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin/python3
H=$WORK/h
mkdir -p "$WORK/out" "$WORK/cache-cpu"
unset XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION DEFUQUMAT_K_BATCH DEFUQUMAT_BAND_BATCH
export JAX_PLATFORMS=cpu PYTHONPATH="$REPO" DEFUQUMAT_THREADS="$SLURM_CPUS_PER_TASK"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export DEFUQUMAT_CACHE_DIR="$WORK/cache-cpu"
hostname; date; git -C "$REPO" rev-parse HEAD
cd "$REPO" || exit 1
O=$WORK/out/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
case "$SLURM_ARRAY_TASK_ID" in
  0) "$PY" -u "$H/v116.py" $O.json 2,2,2 16,64 0.002,0.02 16,32 ;;
  1) "$PY" -u "$H/v116.py" $O.json 2,3,3 16,24,40 0.002 16 ;;
  2) "$PY" -u "$H/v117.py" $O.json 3 12,24 yes 12 ;;
  3) "$PY" -u "$H/v117.py" $O.json 3 8,16 yes 12 --swap ;;
  4) "$PY" -u "$H/v117.py" $O.json 4 8,16 yes 16 ;;
  5) "$PY" -u "$H/v117.py" $O.json 3 16 no 24 ;;
esac
echo "exit $?"; date
