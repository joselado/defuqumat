#!/bin/bash
# P180: Ni fcc with spin-orbit coupling and a moment on the fully relativistic
# PAW dataset, the pw.x save directory both ways (ni_soc_roundtrip.py). Task 0
# at the dataset's suggested 75/478 Ry, task 1 at 45/360 Ry as the fallback.
# Serial pw.x (the conda pw.x is MPI-linked and dies under mpirun with one
# Slurm task, docs/triton-notes.local.md), XLA on the task's CPUs.
#SBATCH --job-name=defuqumat-nisoc
#SBATCH --partition=batch-csl
#SBATCH --array=0-1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/scratch/work/ladovj1/calculations/density-io/out/nisoc-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/density-io/out/nisoc-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/density-io
REPO=${DEFUQUMAT_REPO:-/scratch/work/ladovj1/apps/defuqumat-dio}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv
QEBIN=/scratch/work/ladovj1/apps/qe-env/bin
mkdir -p "$WORK/out" || exit 1
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONPATH="$REPO"
source "$VENV/bin/activate" || exit 1
hostname; date; git -C "$REPO" log --oneline -1
case ${SLURM_ARRAY_TASK_ID:-0} in
    0) CUT="75 478" ;;
    1) CUT="45 360" ;;
esac
python3 "$REPO/notes/qe-writer/triton/ni_soc_roundtrip.py" $CUT \
    "$WORK/nisoc-${SLURM_ARRAY_TASK_ID:-0}" "$QEBIN/pw.x"
date
