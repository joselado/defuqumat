#!/bin/bash
# I114 run 3: the energy's order in the SCF residual at s = 0, 0.5, 1 (variational
# test), the pre-P118 becsum at 0.5, and qq_so left unblended (the entry's term off).
export DEFUQUMAT_QE_ROOT=/l/ladovj1/qe-7.5/qe-7.5
W=/l/ladovj1/w-inc-I114
R=/l/ladovj1/review/incompat/I114
H=$R/harness.py
for job in "z 0.5 1e-8" "z 0.5 1e-9" "z 0.5 1e-11" "z 0.5 1e-14" \
           "z 0.0 1e-8" "z 0.0 1e-9" "z 0.0 1e-11" \
           "z 1.0 1e-8" "z 1.0 1e-9" "z 1.0 1e-11" "z 1.0 1e-14" \
           "z 0.5 1e-9 old" "z 0.5 1e-10 old" "z 0.5 1e-12 old" \
           "z 0.25 1e-12 qqfull" "x 0.25 1e-12 qqfull" "z 0.02 1e-12 qqfull" \
           "z 0.5 1e-10 qqfull" "z 0.5 1e-12 qqfull"; do
    set -- $job
    tag="$1-$2-$3${4:+-$4}"
    nohup bash /l/ladovj1/review/guide/slot.sh $W . $R/run3-$tag.log $H $1 $2 $3 $R/run3-$tag.json $4 > /dev/null 2>&1 < /dev/null & disown
    sleep 2
done
