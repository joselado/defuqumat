#!/usr/bin/env bash
# Run docs/figures/<id>/compute.py with the environment pinned.
#
#   bash docs/figures/run.sh <id>
#
# - BLAS/OpenMP and XLA threads pinned to 1 (DEFUQUMAT_THREADS=1),
#   PYTHONPATH set to this checkout and to docs/figures, so the run imports
#   this defuqumat and common.py. compute.py runs inside its own folder.
# - At most one job per entry of FIGURE_CPUS: a job takes one of the slot
#   locks (flock on docs/figures/.locks/slot<n>) and waits while all are
#   taken. Each slot is pinned with taskset to its own physical core. On the
#   Core Ultra 5 225U workstation CPUs 0-3 are two P-cores with their SMT
#   siblings, so the default "0 2" is the two P-cores; on D22 (i5-12600K,
#   CPUs 0-11 are six P-cores with two siblings each) use "0 2 4".
# - Inside a systemd user scope with MemoryMax=8G and no swap, when
#   systemd-run works; otherwise without it, with a note in the log.
# - Writes docs/figures/<id>/compute.log, RUNNING while it runs, and DONE
#   (exit 0 and data.npz written) or FAILED at the end.
#
# Environment: PYTHON (default python3), FIGURE_CPUS (default "0 2").
set -u
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
id="${1:?usage: run.sh <id>}"
dir="$here/$id"
[ -f "$dir/compute.py" ] || { echo "no $dir/compute.py" >&2; exit 2; }
PYTHON="${PYTHON:-python3}"
read -r -a CPUS <<< "${FIGURE_CPUS:-0 2}"

export MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 DEFUQUMAT_THREADS=1
export PYTHONPATH="$repo:$here"

log="$dir/compute.log"
rm -f "$dir/DONE" "$dir/FAILED"
echo "waiting for a slot since $(date '+%F %T')" > "$dir/RUNNING"

# take a free slot; the lock is held by fd 9, inherited by the job, so it
# is released exactly when this script ends
mkdir -p "$here/.locks"
slot=""
while [ -z "$slot" ]; do
    for s in "${!CPUS[@]}"; do
        exec 9>>"$here/.locks/slot$s"
        if flock -n 9; then slot=$s; break; fi
        exec 9>&-
    done
    [ -z "$slot" ] && sleep 20
done
cpu="${CPUS[$slot]}"
trap 'rm -f "$dir/RUNNING"' EXIT
echo "slot $slot cpu $cpu host $(hostname) pid $$ since $(date '+%F %T')" > "$dir/RUNNING"

cmd=(taskset -c "$cpu" "$PYTHON" -u compute.py)
scope=(systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0)
{
    echo "# $id on $(hostname), slot $slot, cpu $cpu, $(date '+%F %T')"
    echo "# PYTHON=$PYTHON PYTHONPATH=$PYTHONPATH"
} > "$log"
if command -v systemd-run > /dev/null && "${scope[@]}" true 2> /dev/null; then
    cmd=("${scope[@]}" "${cmd[@]}")
    echo "# memory cap: systemd scope, MemoryMax=8G, MemorySwapMax=0" >> "$log"
else
    echo "# memory cap: none (systemd-run --user not available)" >> "$log"
fi

cd "$dir"
start=$(date +%s)
"${cmd[@]}" >> "$log" 2>&1
status=$?
echo "# exit $status after $(( $(date +%s) - start )) s" >> "$log"
if [ $status -eq 0 ] && [ -f "$dir/data.npz" ] && [ "$dir/data.npz" -nt "$dir/RUNNING" ]; then
    date '+%F %T' > "$dir/DONE"
else
    echo "exit $status $(date '+%F %T')" > "$dir/FAILED"
fi
exit $status
