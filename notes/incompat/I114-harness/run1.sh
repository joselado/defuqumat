#!/bin/bash
# I114 run 1: tetragonal Co (ultrasoft, fully relativistic), z at 0, 0.25, 0.5, 1 and
# x at 0.25, 0.5, 1, all at conv_thr 1e-12; z at 0.5 also at 1e-10.
export DEFUQUMAT_QE_ROOT=/l/ladovj1/qe-7.5/qe-7.5
W=/l/ladovj1/w-inc-I114
R=/l/ladovj1/review/incompat/I114
H=$R/harness.py
for job in "z 0.5 1e-12" "x 0.5 1e-12" "z 0.25 1e-12" "x 0.25 1e-12" "z 1.0 1e-12" "x 1.0 1e-12" "z 0.0 1e-12" "z 0.5 1e-10"; do
    set -- $job
    tag="$1-$2-$3"
    nohup bash /l/ladovj1/review/guide/slot.sh $W . $R/run1-$tag.log $H $1 $2 $3 $R/run1-$tag.json > /dev/null 2>&1 < /dev/null & disown
    sleep 2
done
