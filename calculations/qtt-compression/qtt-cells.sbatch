#!/bin/bash
# Quantics compressibility of the fields of larger cells (notes/QTT-COMPRESSION.md):
# one array task per cell, each the SCF once and then qtt_cell.py's analysis.
#
# Not submitted by any tool here (CLAUDE.local.md): by hand, after `sbatch --test-only`.
#
#   sbatch --partition=batch-csl --array=0-3 \
#          --export=ALL,DEFUQUMAT_REPO=/scratch/work/ladovj1/apps/defuqumat-qtt \
#          calculations/qtt-compression/qtt-cells.sbatch
#
# cells  0 si64      2x2x2 conventional cubic, a = 20.40 bohr: a power-of-two repetition
#                    of si8 along every axis, the control for the repetition
#        1 si63-vac  the same with one atom removed: not a repetition
#        2 si64-disp the same with every atom displaced, 0.05 A rms per component, seed 0
#        3 al-slab   benchmarks/al-slab.in, five Al layers and vacuum, 5.3 x 5.3 x 31 bohr
#        All at ecutwfc = 12 Ry; the Si cells at Gamma with a 0.1 eV Gaussian smearing.
# dR     Si cells: 45^3 FFT grid, Rmin = 6 per axis, so dR = 1, 2 is 2^21 and 2^24
#        points; dR = 3 would be 2^27 and a 2^13 x 2^14 SVD per sweep, left out. Slab:
#        Rmin = (4, 4, 7), dR = 1, 2, 3, at most 2^24.
# time   not measured at this size. On Si2 a whole run is 2 minutes; at 2^24 points one
#        TT-SVD sweep is estimated at 1 to 2 minutes and a run makes about 60 of them,
#        plus TCI at ranks of a few hundred. 12 h is a margin of about 4.
# mem    an SCF of 64 Si atoms at 12 Ry is about 2 GB here. A 2^24 complex field is
#        256 MB and the analysis holds a few copies plus an SVD of 4096 x 4096: 16 GB.
# cpus   8. The SCF runs with BLAS on one thread (XLA has its own pool); the analysis
#        is NumPy SVDs and runs with BLAS on all eight.
#SBATCH --job-name=qtt-cells
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/work/ladovj1/calculations/qtt-compression/out/%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/qtt-compression/out/%A_%a.err

WORK=/scratch/work/ladovj1/calculations/qtt-compression
REPO=${DEFUQUMAT_REPO:-/scratch/work/ladovj1/apps/defuqumat-qtt}
PY=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin/python3
CELLS=(si64 si63-vac si64-disp al-slab)
DRS=(1,2 1,2 1,2 1,2,3)
TCIDR=(2 2 2 3)
i=${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$WORK/out" "$WORK/cache-cpu" "$WORK/results"
unset XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION \
      DEFUQUMAT_K_BATCH DEFUQUMAT_BAND_BATCH
export DEFUQUMAT_MEMORY_MODE=memory JAX_PLATFORMS=cpu PYTHONPATH="$REPO" \
       DEFUQUMAT_THREADS="$SLURM_CPUS_PER_TASK" DEFUQUMAT_CACHE_DIR="$WORK/cache-cpu" \
       QUTECIPY_PATH=/scratch/work/ladovj1/apps/qutecipy

echo "=== node ==="; hostname; date
"$PY" -c "import os; print('affinity', len(os.sched_getaffinity(0)))"
git -C "$REPO" rev-parse HEAD
git -C "$QUTECIPY_PATH" rev-parse HEAD
cd "$REPO" || exit 1

echo "=== SCF of ${CELLS[$i]} ==="
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "$PY" -u calculations/qtt-compression/qtt_cell.py "${CELLS[$i]}" "$WORK/results" \
  --repo "$REPO" --skip orders,tci,matched,gspace
echo "exit $?"; date

echo "=== analysis of ${CELLS[$i]} ==="
N=$SLURM_CPUS_PER_TASK
OMP_NUM_THREADS=$N MKL_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N \
  "$PY" -u calculations/qtt-compression/qtt_cell.py "${CELLS[$i]}" "$WORK/results" \
  --repo "$REPO" --dR "${DRS[$i]}" --tci-dR "${TCIDR[$i]}"
echo "exit $?"; date
