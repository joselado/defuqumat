import sys, re
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(REPO))
from defuqumat import Calculator
for name in sys.argv[1:]:
    calc = Calculator.from_text(Path(f"inputs/{name}.in").read_text(), pseudo_dir=REPO / "tests/data/pseudo", announce=False)
    r = calc.get_scf()
    e = float(re.search(r"!\s+total energy\s+=\s+([-0-9.]+)", Path(f"runs/fix/{name}/pw.out").read_text()).group(1))
    hub = re.findall(r"Hubbard energy\s+=\s+([-0-9.]+)", Path(f"runs/fix/{name}/pw.out").read_text())
    print(name, "defuqumat", r.iterations, f"{r.total_energy:.8f}", "pw.x", e, "diff", f"{r.total_energy - e:.2e}",
          "hubbard", r.energy_terms.get("hubbard", r.energy_terms.get("e_hub")), "pw.x hub", hub[-1] if hub else None, flush=True)
