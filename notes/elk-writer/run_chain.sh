#!/bin/bash
# $1 = run dir already holding STATE.OUT and the hand-off elk.in
set -e
ELK=/u/40/ladovj1/data/Documents/programs/elkpy/build/elk/src/elk
cd "$1"
if grep -q mixrho elk.in; then
  OMP_NUM_THREADS=4 taskset -c 4-7 $ELK > handoff.log 2>&1
  cp INFO.OUT INFO.handoff.OUT; cp STATE.OUT STATE.handoff.OUT
fi
sed 's/^  0$/  1/' ../../../si2-pz/conv/elk.in > elk.in
OMP_NUM_THREADS=4 taskset -c 4-7 $ELK > scf.log 2>&1
echo "$1: loops $(grep -c 'Loop number' INFO.OUT)"
grep "RMS change in Kohn-Sham potential" INFO.OUT | head -3
grep "total energy  " INFO.OUT | tail -1
