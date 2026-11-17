#!/bin/bash
# $1 = writer script, $2 = template, $3 = run dir, $4 = epspot
set -e
rm -rf $3
python3 $1 $2 $3 2>&1 | grep -v Warn | tail -1
cd $3
sed -i "s/^  1.d-9$/  $4/" elk.in
OMP_NUM_THREADS=4 taskset -c 4-7 ${ELK:?set ELK to the elk binary} > scf.log 2>&1
echo "$3: loops $(grep -c 'Loop number' INFO.OUT); residuals $(grep 'RMS change in Kohn-Sham' INFO.OUT | sed 's/.*: *\([0-9.E+-]*\).*/\1/' | head -4 | tr '\n' ' ')"
