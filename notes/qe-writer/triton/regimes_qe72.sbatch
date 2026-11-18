#!/bin/bash
# P180: the pw.x save directory written by defuqumat and read back by pw.x
# (pwx_reads_ours.py), for the regimes the workstation's pw.x 7.5 did not
# cover. Triton's QE 7.2 module links libxc 6.2.2, so the TB09 meta-GGA restart
# (ekin-density) runs here and not on the workstation; it writes .dat (no HDF5).
#   0 o2-us-lsda       ultrasoft, nspin 2, fixed moment
#   1 o2-us-noncol     ultrasoft, noncollinear, moment tilted 60/30 deg
#   2 as-atom-soc-us   ultrasoft lspinorb with a tilted moment
#   3 fe-kind1-noncol  DFT+U lda_plus_u_kind 1, noncollinear
#   4 si2-tb09         TB09, nspin 1
#   5 si2-tb09-lsda    TB09, nspin 2
#   6 fe-kind1-noncol  again, with defuqumat on pw.x's noncollinear GGA,
#                      noncollinear_gga = 'kubler' (task 3 ran the default
#                      'grad_m', 2.7e-4 Ry from pw.x on this cell, PLAN.md P143)
#SBATCH --job-name=defuqumat-regimes72
#SBATCH --partition=batch-csl
#SBATCH --array=0-6
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=/scratch/work/ladovj1/calculations/density-io/out/regimes72-%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/density-io/out/regimes72-%A_%a.err

WORK=/scratch/work/ladovj1/calculations/density-io
REPO=${DEFUQUMAT_REPO:-/scratch/work/ladovj1/apps/defuqumat-dio}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv
NAMES=(o2-us-lsda o2-us-noncol as-atom-soc-us fe-kind1-noncol si2-tb09 si2-tb09-lsda fe-kind1-noncol)
NAME=${NAMES[${SLURM_ARRAY_TASK_ID:-0}]}
OURS=()
[ "${SLURM_ARRAY_TASK_ID:-0}" = 6 ] && OURS=("--ours=noncolin=.true.->noncolin=.true., noncollinear_gga='kubler'")
TAG=$NAME${OURS:+-kubler}
mkdir -p "$WORK/out" || exit 1
export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu PYTHONPATH="$REPO"
source "$VENV/bin/activate" || exit 1
hostname; date; git -C "$REPO" log --oneline -1; echo "$NAME"
module load triton/2024.1-gcc openmpi/4.1.6 quantum-espresso/7.2 >/dev/null 2>&1
command -v pw.x || { echo 'no pw.x after module load'; exit 1; }
WRAP="$WORK/pwx72-$TAG.sh"
cat > "$WRAP" <<'EOF'
#!/bin/bash
exec srun --ntasks=1 --cpus-per-task=1 --overlap pw.x "$@"
EOF
chmod +x "$WRAP"
cd "$WORK" || exit 1
PW_X="$WRAP" TASKSET=0 python3 "$REPO/notes/qe-writer/pwx_reads_ours.py" \
    "$REPO/notes/qe-writer/inputs/$NAME.in" "$WORK/rt-$TAG" "${OURS[@]}"
date
