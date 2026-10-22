import sys, re, traceback
from pathlib import Path
WT = Path(sys.argv[1])
sys.path.insert(0, str(WT / "notes/incompat/I120-harness"))
sys.path.insert(0, str(WT))
from jvp import ROOT, edited, CASES  # noqa
from defuqumat.io.pwin import parse_pw_input
from defuqumat.pseudo import read_upf
from defuqumat.scf.driver import Calculation
from defuqumat.scf.residual import make_residual
from defuqumat.system import build_system

cases = [
    ("si2-us", None, "input_dft = 'tb09'"),
    ("si2-paw", None, "input_dft = 'tb09'"),
    ("si2-us", None, "input_dft = 'bj06'"),
    ("h-fcc-spiral-scan", "ferro", "input_dft = 'tb09'"),
    ("si-noncolin-berry", None, "input_dft = 'tb09'"),
    ("h-fcc-spiral-scan", "ferro", "input_dft = 'pbesol'"),
    ("h-fcc-spiral-scan", "ferro", "input_dft = 'revpbe'"),
    ("bn-ldau-noncol-noU", None, None),
    ("alas-magnetoelectric", None, None),
    ("fe2-afm-soc", None, None),
    ("bismuthene-soc-small-lda", None, None),
    ("i-atom-soc-lda", None, None),
    ("h-atom-noncolin", None, None),
    ("h10-chain-noncolin", None, None),
    ("h-fcc-spiral-scan", "ferro", "lda_plus_u = .true."),
]
for case, sub, extra in cases:
    try:
        text = edited(case, None, None, extra, sub)
        system = build_system(parse_pw_input(text))
        pseudos = tuple(read_upf(ROOT / "tests/data/pseudo" / s.pseudo_file)
                        for s in system.structure.species)
        calc = Calculation(system, pseudos)
        info = (f"noncolin={calc.noncolin} nspin={calc.nspin} nspin_mag={calc.nspin_mag} "
                f"us={calc.is_ultrasoft} paw={calc.paw is not None} "
                f"func={calc.functional.name} grad={calc.functional.is_gradient} "
                f"meta={calc.functional.is_meta} lspinorb={getattr(system, 'lspinorb', '?')}")
    except Exception as e:
        print(f"{case}/{sub}/{extra}: BUILD FAILS {type(e).__name__}: {str(e)[:150]}")
        continue
    try:
        res = make_residual(calc, 8, 1e-10)
        res._refuse_sternheimer(None)
        print(f"{case}/{sub}/{extra}: ADMITTED  [{info}]")
    except NotImplementedError as e:
        print(f"{case}/{sub}/{extra}: refused [{info}] :: {str(e)[:160]}")
    except Exception as e:
        print(f"{case}/{sub}/{extra}: OTHER {type(e).__name__} [{info}] :: {str(e)[:160]}")
