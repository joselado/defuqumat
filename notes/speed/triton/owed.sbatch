#!/bin/bash
# The checks owed by OPEN.md Part XXIII item 6 and Part III M5 (2026-10-23), moved to
# Triton's CPUs while D22 is down. In every task the old arm (master fbe14c1b) and the
# new arm run back to back on the same node, so a byte comparison never crosses nodes.
#   0-1  item 6, PAW bit-identity, alas-piezo-tiny-paw at k_batch 1 and 3
#   2    item 6, Raman path bit-identity, si-us-nosym at k_batch 1 and 3
#   3    item 6, warm timing at 27 k-points (alas-piezo-tiny-333, k_batch 3, 5 warm calls) + HLO count
#   4    item 6, ultrasoft bit-identity at a chunk size not used before (alas-piezo-tiny, k_batch 2)
#   5    M5 gate: bi10-soc SCF, cache off, peak and per-k Davidson steps, both arms
#   6    M5: temp bytes and largest HLO shapes, and SCF steps, on si8-us, si8-paw, si16-ecut30, pt-so
#   7    M5: tools/benchmark.py on si16-1k-ecut30 and si8-us-1k, three runs per arm
# Every task except 5 is pinned to one core of its allocation; task 5 uses the four.
# Not submitted by any tool (CLAUDE.local.md): by hand, after sbatch --test-only.
#SBATCH --job-name=speed-owed
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/work/ladovj1/calculations/speed/out/%A_%a.out
#SBATCH --error=/scratch/work/ladovj1/calculations/speed/out/%A_%a.err

W=/scratch/work/ladovj1/calculations/speed
A=/scratch/work/ladovj1/apps
OLD=$A/defumat-speed-old
X6=$A/defumat-speed-x6
M5=$A/defumat-speed-m5
PY=/scratch/work/ladovj1/calculations/pypresso-gpu/venv/bin/python3
H=$W/h
T=$SLURM_ARRAY_TASK_ID
O=$W/out/t$T
mkdir -p "$O"
unset XLA_PYTHON_CLIENT_PREALLOCATE XLA_PYTHON_CLIENT_MEM_FRACTION DEFUQUMAT_K_BATCH DEFUQUMAT_BAND_BATCH
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ONE=$(taskset -cp $$ | sed 's/.*: //; s/[,-].*//')
if [ "$T" = 5 ]; then PIN=""; export DEFUQUMAT_THREADS=$SLURM_CPUS_PER_TASK
else PIN="taskset -c $ONE"; export DEFUQUMAT_THREADS=1; fi
echo "=== node $(hostname) $(date) task $T pin '$PIN'"
lscpu | grep 'Model name'
for r in $OLD $X6 $M5; do echo "$r $(git -C $r rev-parse --short HEAD)"; done

x6 () {  # x6 <tag> <input> <kbatch> <repeats> [count]
  for arm in old new; do
    R=$OLD; [ $arm = new ] && R=$X6
    echo "--- x6 $1 $arm"
    ( cd $R && DEFUQUMAT_CACHE_DIR=$O/cache-$arm PYTHONPATH=$R $PIN $PY -u $H/x6.py $2 $3 $O/$1-$arm $4 $5 )
    echo "exit $?"
  done
  for f in columns depsilon; do
    cmp $O/$1-old/$f.npy $O/$1-new/$f.npy && echo "IDENTICAL $1 $f" || echo "DIFFERENT $1 $f"
  done
}
m5 () {  # m5 <script> <input> <tag>
  for arm in old new; do
    R=$OLD; [ $arm = new ] && R=$M5
    echo "--- m5 $1 $3 $arm"
    ( cd $R && DEFUQUMAT_CACHE_DIR=off PYTHONPATH=$R /usr/bin/time -v $PIN $PY -u $H/$1 $2 $O/$3-$arm.json 2> $O/$3-$arm.time; echo "exit $?"; grep -E 'Maximum resident|Elapsed' $O/$3-$arm.time )
  done
}

case "$T" in
  0) x6 paw-kb1 tests/data/qe/alas-piezo-tiny-paw.in 1 1 ;;
  1) x6 paw-kb3 tests/data/qe/alas-piezo-tiny-paw.in 3 1 ;;
  2) export KIND=raman; x6 raman-kb1 tests/data/qe/si-us-nosym.in 1 1; x6 raman-kb3 tests/data/qe/si-us-nosym.in 3 1 ;;
  3) x6 k27-kb3 $H/alas-piezo-tiny-333.in 3 5 count ;;
  4) x6 us-kb2 tests/data/qe/alas-piezo-tiny.in 2 1 ;;
  5) m5 scf_m5.py tests/data/qe/bi10-soc.in bi10 ;;
  6) for c in si8-us-1k si8-paw-1k si16-1k-ecut30 pt-so-1k; do
       m5 hlo_m5.py benchmarks/$c.in hlo-$c; m5 scf_m5.py benchmarks/$c.in scf-$c; done ;;
  7) for c in si16-1k-ecut30 si8-us-1k; do for i in 1 2 3; do for arm in old new; do
       R=$OLD; [ $arm = new ] && R=$M5
       echo "--- bench $c $arm run $i"
       ( cd $R && DEFUQUMAT_CACHE_DIR=$O/cache-$arm PYTHONPATH=$R $PIN $PY -u tools/benchmark.py benchmarks/$c.in > $O/bench-$c-$arm-$i.log 2>&1; echo "exit $?"; tail -25 $O/bench-$c-$arm-$i.log )
     done; done; done ;;
esac
echo "=== done $(date)"
