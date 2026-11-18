#!/bin/bash
# P180: the TB09 ekin-density swap test (tb09_swap.py) on the two work
# directories regimes_qe72.sbatch left, with Triton's QE 7.2 module (libxc).
#SBATCH --job-name=defuqumat-tb09swap
#SBATCH --partition=batch-csl
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --output=/scratch/work/ladovj1/calculations/density-io/out/tb09swap-%j.out
#SBATCH --error=/scratch/work/ladovj1/calculations/density-io/out/tb09swap-%j.err

WORK=/scratch/work/ladovj1/calculations/density-io
REPO=${DEFUQUMAT_REPO:-/scratch/work/ladovj1/apps/defuqumat-dio}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONPATH="$REPO"
source "$VENV/bin/activate" || exit 1
hostname; date; git -C "$REPO" log --oneline -1
module load triton/2024.1-gcc openmpi/4.1.6 quantum-espresso/7.2 >/dev/null 2>&1
command -v pw.x || { echo 'no pw.x after module load'; exit 1; }
WRAP="$WORK/pwx72-swap.sh"
printf '#!/bin/bash\nexec srun --ntasks=1 --cpus-per-task=1 --overlap pw.x "$@"\n' > "$WRAP"
chmod +x "$WRAP"
cd "$WORK" || exit 1
for NAME in si2-tb09 si2-tb09-lsda; do
    python3 "$REPO/notes/qe-writer/triton/tb09_swap.py" "$WORK/rt-$NAME" "$WRAP" || echo "FAILED $NAME"
done
date
