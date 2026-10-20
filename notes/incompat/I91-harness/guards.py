"""I91 harness 3: which refusal fires on the committed cells the guard used to stop as ultrasoft.

The body of the slow ``test_dft_plus_u_does_not_fall_through_to_a_ragged_error``
and the spinor ultrasoft cell, run without pytest. Prints each message.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path.cwd()

from defuqumat.io.pwin import read_pw_input  # noqa: E402
from defuqumat.pseudo.upf import read_upf  # noqa: E402
from defuqumat.response.photocurrent import require_a_shift_current_regime  # noqa: E402
from defuqumat.scf.driver import Calculation  # noqa: E402
from defuqumat.system.builder import build_system  # noqa: E402

CASES = ROOT / "tests" / "data" / "qe"
PSEUDO = ROOT / "tests" / "data" / "pseudo"

for name in sys.argv[1:]:
    system = build_system(read_pw_input(CASES / name))
    pseudos = tuple(read_upf(PSEUDO / s.pseudo_file) for s in system.structure.species)
    try:
        require_a_shift_current_regime(Calculation(system, pseudos))
        print(f"{name}: passes the guard")
    except NotImplementedError as error:
        print(f"{name}: NotImplementedError: {str(error)[:300]}")
