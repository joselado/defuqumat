#!/bin/bash
# I114 run 2: small and intermediate scales, the conv_thr control at the ends, the
# Hellmann-Feynman slope at 0, and the pre-P118 becsum as the alternative hypothesis.
export DEFUQUMAT_QE_ROOT=/l/ladovj1/qe-7.5/qe-7.5
W=/l/ladovj1/w-inc-I114
R=/l/ladovj1/review/incompat/I114
H=$R/harness.py
nohup bash /l/ladovj1/review/guide/slot.sh $W . $R/run2-hf.log $R/hf.py $R/run2-hf.json > /dev/null 2>&1 < /dev/null & disown
sleep 2
for job in "z 0.01 1e-12" "x 0.01 1e-12" "z 0.02 1e-12" "x 0.02 1e-12" \
           "z 0.0625 1e-12" "x 0.0625 1e-12" "z 0.125 1e-12" "x 0.125 1e-12" \
           "z 0.75 1e-12" "x 0.75 1e-12" "z 1.0 1e-10" "z 0.0 1e-10" \
           "z 0.25 1e-12 old" "x 0.25 1e-12 old" "z 0.02 1e-12 old"; do
    set -- $job
    tag="$1-$2-$3${4:+-$4}"
    nohup bash /l/ladovj1/review/guide/slot.sh $W . $R/run2-$tag.log $H $1 $2 $3 $R/run2-$tag.json $4 > /dev/null 2>&1 < /dev/null & disown
    sleep 2
done
