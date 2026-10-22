export DEFUQUMAT_QE_ROOT=/l/ladovj1/qe-7.5/qe-7.5
WT=/l/ladovj1/w-inc-ultraseed
R=/l/ladovj1/review/incompat/ultraseed
nohup bash /l/ladovj1/review/guide/slot.sh $WT . $R/unit1.log -m pytest -q -rs --tb=short -m 'not slow' tests/unit/test_ultracell_seed.py tests/regression/test_ultracell.py > /dev/null 2>&1 < /dev/null & disown
nohup bash /l/ladovj1/review/guide/slot.sh $WT . $R/i116-h1.log $WT/notes/incompat/ultraseed-harness/i116.py $R/i116-h1.json 0.002,0.02 1e-8,1e-10 16 > /dev/null 2>&1 < /dev/null & disown
nohup bash /l/ladovj1/review/guide/slot.sh $WT . $R/i117-h1.log $WT/notes/incompat/ultraseed-harness/i117.py $R/i117-h1.json 3 8,16 yes > /dev/null 2>&1 < /dev/null & disown
echo launched
