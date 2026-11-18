#!/bin/bash
# P180: Ni fcc with spin-orbit coupling and a moment on the fully relativistic
# PAW dataset, the pw.x save directory both ways (ni_soc_roundtrip.py). Task 0
# at the dataset's suggested 75/478 Ry, task 1 at 45/360 Ry as the fallback.
# This variant runs Triton's own QE 7.2 module, which is built without HDF5 and
# so writes and reads charge-density.dat (the conda pw.x 7.5 writes .hdf5, and
# the venv has no h5py); MPI-linked, so it goes through srun with one task.
# NONCOLLINEAR_GGA=kubler (sbatch --export=ALL,NONCOLLINEAR_GGA=kubler) puts
# defuqumat on pw.x's noncollinear GGA; the default 'grad_m' is another (P143).
#SBATCH --job-name=defuqumat-nisoc72
#SBATCH --partition=batch-csl
#SBATCH --array=0-1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/scratch/work/ladovj1/calculations/density-io/out/nisoc72-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/density-io/out/nisoc72-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/density-io
REPO=${DEFUQUMAT_REPO:-/scratch/work/ladovj1/apps/defuqumat-dio}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv
QEBIN=/scratch/work/ladovj1/apps/qe-env/bin
mkdir -p "$WORK/out" || exit 1
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONPATH="$REPO"
source "$VENV/bin/activate" || exit 1
hostname; date; git -C "$REPO" log --oneline -1
module load triton/2024.1-gcc openmpi/4.1.6 quantum-espresso/7.2 >/dev/null 2>&1
command -v pw.x || { echo 'no pw.x after module load'; exit 1; }
cat > "$WORK/pwx72.sh" <<'WRAP'
#!/bin/bash
exec srun --ntasks=1 --cpus-per-task=1 --overlap pw.x "$@"
WRAP
chmod +x "$WORK/pwx72.sh"
case ${SLURM_ARRAY_TASK_ID:-0} in
    0) CUT="75 478" ;;
    1) CUT="45 360" ;;
esac
python3 "$REPO/notes/qe-writer/triton/ni_soc_roundtrip.py" $CUT \
    "$WORK/nisoc72-${SLURM_ARRAY_TASK_ID:-0}${NONCOLLINEAR_GGA:+-$NONCOLLINEAR_GGA}" "$WORK/pwx72.sh"
date
