"""Which k-points the tunnelling current through a hydrogen sheet comes from."""
from defuqumat import Calculator
from common import PSEUDO, save

calc = Calculator.from_file("h-sheet.in", pseudo_dir=PSEUDO)

# the sheet is at 0.5 of c = 15 bohr: the substrate 3 bohr below, the tip 3 bohr above
run = calc.get_momentum_transport(exit_height=0.3, height=0.7, grid=(32, 32, 1),
                                  broadening=0.03)

save(k=run.kpoints, weight=run.as_grid("weight"), bare=run.as_grid("bare"),
     tersoff_hamann=run.as_grid("tersoff_hamann"))
