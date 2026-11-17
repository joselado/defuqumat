import numpy as np
from defuqumat import Calculator
from defuqumat.io.elk import ElkState

text = """&control
 calculation='scf'
/
&system
 ibrav=0, nat=2, ntyp=1, ecutwfc=15.0, ecutrho=120.0
/
&electrons
 conv_thr=1d-8
/
CELL_PARAMETERS bohr
 0.0 5.10 5.10
 5.10 0.0 5.10
 5.10 5.10 0.0
ATOMIC_SPECIES
 Si 28.086 Si.pz-n-kjpaw_psl.0.1.UPF
ATOMIC_POSITIONS crystal
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS automatic
 2 2 2 0 0 0
"""
calc = Calculator.from_text(text, pseudo_dir="tests/data/pseudo")
calc.write_elk_state("elk_run", template="tests/data/elk/si2_template")

import shutil
shutil.copy("tests/data/elk/si2_template/GEOMETRY.OUT", "elk_run")
state = ElkState.read("elk_run")
r = state.rsp[0][:state.nrmt[0]]
print(f"electrons in each muffin tin: {state.muffin_tin_charges()[0]:.4f}")
print(f"l = 3 component at r = 1.8 bohr: {state.rhomt[10, np.searchsorted(r, 1.8), 0]:.4f}")
