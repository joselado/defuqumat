#!/bin/bash
# The slow tests of a named list of files, one array task per file.
#
# This is for the slow files a change touches, which `PLAN.md` lists for the
# user to run, when the user asks for them on the cluster. The slow set runs
# only on the user's explicit request (`CLAUDE.md`), so nothing submits this
# script: it is written to be submitted by hand. One array rather than one
# submission per file, which is the cluster policy's own rule.
#
# The files come in `SLOW_FILES`, separated by colons (a comma would end the
# `--export` list), and the checkout in `DEFUQUMAT_REPO`; the array size and the
# output paths go on the command line, `--array=0-(number of files - 1)`.
#
# ---- the #SBATCH values ----
#
# partition     batch-csl, where the gate of 2026-11-15 started within 20 s.
# time          04:00:00 per file. The files are not timed one by one on the
#               workstation, and with the cache off each pays its own
#               compilation, so this is headroom rather than an estimate.
# cpus-per-task 8, as `gate.sbatch`: `tests/conftest.py` sets
#               `DEFUQUMAT_THREADS=8`, and a mask of four hung 6 times in 14
#               gate runs (the `test-runs` skill).
# mem           32G, as `regression.sbatch`, whose worst file is killed at 16 G
#               on the workstation.
#SBATCH --job-name=defuqumat-slow-files
#SBATCH --partition=batch-csl
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

REPO=${DEFUQUMAT_REPO:?set DEFUQUMAT_REPO to the checkout to run}
VENV=/scratch/work/ladovj1/calculations/pypresso-gpu/venv

cd "$REPO" || exit 1
source "$VENV/bin/activate" || exit 1

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export PYTHONPATH="$REPO"
export DEFUQUMAT_CACHE_DIR=off

IFS=':' read -r -a FILES <<< "${SLOW_FILES:?set SLOW_FILES to the test files, colon separated}"
TASK=${SLURM_ARRAY_TASK_ID:-0}
FILE=${FILES[$TASK]:-}
if [ -z "$FILE" ]; then
    echo "no file for task $TASK of ${#FILES[@]}"
    exit 1
fi

echo "=== node ==="; hostname; date
echo "commit: $(git -C "$REPO" rev-parse HEAD)  $(git -C "$REPO" log --oneline -1)"
echo "=== task $TASK of ${#FILES[@]}: $FILE ==="
python3 -m pytest -m slow -q -p no:cacheprovider "$FILE"
STATUS=$?
echo "pytest exit $STATUS"

echo "=== done ==="; date
exit $STATUS
