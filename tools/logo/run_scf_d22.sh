#!/bin/bash
# The SCF behind the logo, on the CPU of D22 (plans/logo.md). Copied to
# /l/ladovj1/review/logo/ and launched there with
#   ssh -f -o BatchMode=yes ladovj1@D22-0161.org.aalto.fi \
#       'nohup bash /l/ladovj1/review/logo/run_scf_d22.sh > /dev/null 2>&1 &' < /dev/null
# and waited on with `grep -q LOGO_SCF_DONE scf.log` (the sentinel the python
# prints last), never on the process. The card is left to the owner's Blender
# renders (JAX_PLATFORMS=cpu, no visible device), the job is capped the way every
# D22 job is, and it is pinned to performance cores 2-9, leaving 0-1 to the
# render that is usually running.
set -u
RUN=/l/ladovj1/review/logo
CODE=/l/ladovj1/w-logo
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
export PYTHONPATH=$CODE DEFUQUMAT_CACHE_DIR=/l/ladovj1/jaxcache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$RUN" || exit 1
rm -f scf.log cri3.npz
systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0 \
    taskset -c 2-9 /l/ladovj1/venv/bin/python -u "$CODE/tools/logo/compute.py" scf \
        --input "$RUN/cri3.in" --pseudo-dir "$CODE/tests/data/pseudo" --out "$RUN/cri3.npz" \
        > scf.log 2>&1
echo "exit=$? LOGO_SCF_EXIT" >> scf.log
